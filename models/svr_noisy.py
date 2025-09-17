import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

import torch

# Ensure project root on sys.path so absolute imports work when running from models/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse dataset selection and results I/O helpers from gnn_run
from gnn_run import (
    select_random_pkls,
    select_tim_pkls,
    load_all_svr_results,
    save_svr_results_for_experiment,
)
from models.gnn import (
    build_train_val_test_loaders_two_stage,
    build_full_loader,
)


CONFIG: Dict[str, Any] = {
    # Data dirs
    "random_noisy_dir": "/data/P70087789/GNN/data/dataset_random_noisy",
    "tim_noisy_dir": "/data/P70087789/GNN/data/dataset_tim_noisy",
    "random_base_dir": "/data/P70087789/GNN/data/dataset_random",
    "tim_base_dir": "/data/P70087789/GNN/data/dataset_tim",
    "results_dir": "/data/P70087789/GNN/models/results",

    # Features/training
    "global_feature_variant": "binned152",
    "batch_size": 128,
    "seed": 42,
}


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _collect_xy_from_loader(loader) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for batch in loader:
        # Batch.global_features is [num_graphs, gdim]
        g = batch.global_features
        y = batch.y.view(-1)
        # If PyG flattened per-graph global features into 1D, reshape back to [num_graphs, -1]
        if g.dim() == 1:
            num_graphs = getattr(batch, 'num_graphs', int(y.shape[0]))
            g = g.view(int(num_graphs), -1)
        mask = torch.isfinite(y)
        if mask.sum() == 0:
            continue
        xs.append(g[mask].cpu().numpy().astype(np.float32))
        ys.append(y[mask].cpu().numpy().astype(np.float32))
    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def _train_eval_svr_on_paths(
    in_paths: List[str],
    ex_paths: List[str],
    *,
    global_feature_variant: str,
    batch_size: int,
    seed: int,
) -> Tuple[float, float, float]:
    train_loader, val_loader, test_loader = build_train_val_test_loaders_two_stage(
        in_paths,
        train_split=0.8,
        val_within_train=0.1,
        batch_size=batch_size,
        seed=seed,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=None,
    )
    X_train, y_train = _collect_xy_from_loader(train_loader)
    X_val, y_val = _collect_xy_from_loader(val_loader)
    X_test, y_test = _collect_xy_from_loader(test_loader)

    extra_loader = build_full_loader(
        ex_paths,
        batch_size=batch_size,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=None,
    )
    X_extra, y_extra = _collect_xy_from_loader(extra_loader)

    # Guard against empty splits
    if X_train.size == 0 or X_test.size == 0 or X_extra.size == 0 or X_val.size == 0:
        raise RuntimeError("One of the datasets (train/test/extra) is empty for the given selection.")

    # Hyperparameter search to reduce overfitting
    param_grid = ParameterGrid({
        'svr__C': [0.1, 0.3, 1.0, 3.0],
        'svr__epsilon': [0.05, 0.1, 0.2],
        'svr__gamma': ['scale', 0.1, 0.03, 0.01],
        'svr__kernel': ['rbf'],
    })
    best_model = None
    best_val = float('inf')
    base = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVR(),
    )
    for params in param_grid:
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            SVR(kernel=params['svr__kernel'], C=params['svr__C'], epsilon=params['svr__epsilon'], gamma=params['svr__gamma'])
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = float(mean_squared_error(y_val, val_pred))
        if val_mse < best_val:
            best_val = val_mse
            best_model = model

    assert best_model is not None

    # Refit on train + val for final evaluation
    X_trv = np.concatenate([X_train, X_val], axis=0)
    y_trv = np.concatenate([y_train, y_val], axis=0)
    # Extract tuned SVR params from best_model
    tuned_svr: SVR = best_model.named_steps['svr']
    final_model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVR(kernel=tuned_svr.kernel, C=tuned_svr.C, epsilon=tuned_svr.epsilon, gamma=tuned_svr.gamma)
    )
    final_model.fit(X_trv, y_trv)

    train_mse = float(mean_squared_error(y_trv, final_model.predict(X_trv)))
    test_mse = float(mean_squared_error(y_test, final_model.predict(X_test)))
    extra_mse = float(mean_squared_error(y_extra, final_model.predict(X_extra)))
    return train_mse, test_mse, extra_mse


def run_qubits_noisy(cfg: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    in_qubits = [2, 3, 4, 5]
    ex_qubits = [6]
    base_rand = cfg.get("random_noisy_dir") or cfg["random_base_dir"]
    base_tim = cfg.get("tim_noisy_dir") or cfg["tim_base_dir"]

    random_in = select_random_pkls(base_rand, qubits=in_qubits, gates=None)
    random_ex = select_random_pkls(base_rand, qubits=ex_qubits, gates=None)
    # Noisy TIM datasets typically have trotter steps 1-3 available
    tim_in = select_tim_pkls(base_tim, qubits=in_qubits, trotter=[1, 2, 3], max_trotter=None)
    tim_ex = select_tim_pkls(base_tim, qubits=ex_qubits, trotter=[1, 2, 3], max_trotter=None)

    r_train, r_test, r_extra = _train_eval_svr_on_paths(
        random_in,
        random_ex,
        global_feature_variant=cfg["global_feature_variant"],
        batch_size=int(cfg["batch_size"]),
        seed=int(cfg["seed"]),
    )
    t_train, t_test, t_extra = _train_eval_svr_on_paths(
        tim_in,
        tim_ex,
        global_feature_variant=cfg["global_feature_variant"],
        batch_size=int(cfg["batch_size"]),
        seed=int(cfg["seed"]),
    )
    return {
        "Random": (r_train, r_test, r_extra),
        "TIM": (t_train, t_test, t_extra),
    }


def run_depth_noisy(cfg: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    in_qubits = [2, 3, 4, 5, 6]
    base_rand = cfg.get("random_noisy_dir") or cfg["random_base_dir"]
    base_tim = cfg.get("tim_noisy_dir") or cfg["tim_base_dir"]

    # Random noisy: gates 0-19 for train/test; 20-39 for extra
    random_in = select_random_pkls(base_rand, qubits=in_qubits, gates_range=(0, 19))
    random_ex = select_random_pkls(base_rand, qubits=in_qubits, gates_range=(20, 39))
    # TIM noisy: trotter 1-2 train/test; 3 for extra
    tim_in = select_tim_pkls(base_tim, qubits=in_qubits, trotter=[1, 2], max_trotter=None)
    tim_ex = select_tim_pkls(base_tim, qubits=in_qubits, trotter=[3], max_trotter=None)

    r_train, r_test, r_extra = _train_eval_svr_on_paths(
        random_in,
        random_ex,
        global_feature_variant=cfg["global_feature_variant"],
        batch_size=int(cfg["batch_size"]),
        seed=int(cfg["seed"]),
    )
    t_train, t_test, t_extra = _train_eval_svr_on_paths(
        tim_in,
        tim_ex,
        global_feature_variant=cfg["global_feature_variant"],
        batch_size=int(cfg["batch_size"]),
        seed=int(cfg["seed"]),
    )
    return {
        "Random": (r_train, r_test, r_extra),
        "TIM": (t_train, t_test, t_extra),
    }


def main() -> None:
    cfg = dict(CONFIG)
    _ensure_dir(cfg["results_dir"])

    try:
        depth_noisy = run_depth_noisy(cfg)
    except Exception as e:
        print(f"[WARN] Failed depth noisy SVR: {e}")
        depth_noisy = None

    try:
        qubits_noisy = run_qubits_noisy(cfg)
    except Exception as e:
        print(f"[WARN] Failed qubits noisy SVR: {e}")
        qubits_noisy = None

    # Merge-save without overwriting existing noiseless results
    existing = load_all_svr_results(cfg["results_dir"]) or {}
    if depth_noisy is not None:
        save_svr_results_for_experiment(cfg["results_dir"], "depth_noisy", depth_noisy)
    if qubits_noisy is not None:
        save_svr_results_for_experiment(cfg["results_dir"], "qubits_noisy", qubits_noisy)

    out_path = os.path.join(cfg["results_dir"], "svr_resuls.pkl")
    out = {
        "saved_to": out_path,
        "keys_now": list((load_all_svr_results(cfg["results_dir"]) or {}).keys()),
        "depth_noisy": depth_noisy,
        "qubits_noisy": qubits_noisy,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


