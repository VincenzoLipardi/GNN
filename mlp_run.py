import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch

from models.neural_network import (
    train_with_two_stage_split_mlp,
    evaluate_overall_mse_mlp,
)
from models.gnn import (
    build_full_loader,
)
from gnn_run import (
    load_gnn_results_for_experiment,
    select_random_pkls,
    select_tim_pkls,
)


CONFIG: Dict[str, Any] = {
    # One of: "qubits", "depth", "depth_reverse"
    "experiment": "depth",

    # Data dirs
    "random_base_dir": "/data/P70087789/GNN/data/dataset_random",
    "tim_base_dir": "/data/P70087789/GNN/data/dataset_tim",
    "results_dir": "/data/P70087789/GNN/models/results",

    # Features/architecture/training
    "global_feature_variant": "binned152",  # 152-dim MLP input
    "epochs": 200,
    "lr": 1e-3,
    "batch_size": 64,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.0,
    "train_split": 0.8,
    "val_within_train": 0.1,
    "device": None,  # e.g. "cuda", "cpu" or None for auto
    "seed": 42,
    "model_kwargs": {
        "hidden_layers": (64, 64, 64),
        "activation": "relu",
        "dropout": 0.0,
        "use_batchnorm": False,
    },

    # Experiment-specific
    "tim_trotter": [1, 2, 3, 4, 5],
    "random_gates": None,  # e.g. [50, 100]
}

# =========================================
# MLP results persistence (mirrors GNN API)
# =========================================
MLP_RESULTS_FILENAME = "mlp_results.pkl"


def get_mlp_results_path(results_dir: str) -> str:
    return os.path.join(results_dir, MLP_RESULTS_FILENAME)


def load_all_mlp_results(results_dir: str) -> Optional[Dict[str, Any]]:
    path = get_mlp_results_path(results_dir)
    if not os.path.exists(path):
        return None
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_mlp_results_for_experiment(results_dir: str, experiment: str, results: Any) -> str:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    existing = load_all_mlp_results(results_dir) or {}
    existing[experiment] = results
    path = get_mlp_results_path(results_dir)
    import pickle
    with open(path, "wb") as f:
        pickle.dump(existing, f)
    return path


def _convert_gnn_saved_to_tuples(saved: Optional[Dict[str, Any]]) -> Optional[Dict[str, Tuple[float, float, float]]]:
    if saved is None:
        return None
    try:
        def _mk(res: Dict[str, float]) -> Tuple[float, float, float]:
            return (
                float(res.get('train_mse', 0.0)),
                float(res.get('test_mse', 0.0)),
                float(res.get('extra_mse', 0.0)),
            )
        return {
            'Random': _mk(saved.get('random', {})),
            'TIM': _mk(saved.get('tim', {})),
        }
    except Exception:
        return None


def _load_gnn_tuples_with_fallbacks(results_dir: str, keys: List[str]) -> Optional[Dict[str, Tuple[float, float, float]]]:
    for key in keys:
        saved = load_gnn_results_for_experiment(results_dir, key)
        tuples = _convert_gnn_saved_to_tuples(saved)
        if tuples is not None:
            return tuples
    return None
# Removed grid-search helpers; the MLP now uses fixed hyperparameters aligned with
# the GNN's global feature MLP (64-64-64, ReLU, no dropout, no batchnorm).



def _ensure_results_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    import random
    import numpy as np
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def run_extrapolation_qubits_mlp(
    random_base_dir: str,
    tim_base_dir: str,
    tim_trotter: Optional[List[int]] = None,
    tim_max_trotter: Optional[int] = 5,
    random_gates: Optional[List[int]] = None,
    results_dir: str = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    global_feature_variant: str = "binned152",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.0,
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    device: Optional[str] = None,
    seed: int = 42,
):
    in_qubits = [2, 3, 4, 5]
    ex_qubits = [6]
    results_dir = _ensure_results_dir(results_dir or "/data/P70087789/GNN/models/results")

    if tim_trotter is None:
        tim_trotter = [1, 2, 3, 4, 5]
        tim_max_trotter = None

    random_in = select_random_pkls(random_base_dir, qubits=in_qubits, gates=random_gates)
    random_ex = select_random_pkls(random_base_dir, qubits=ex_qubits, gates=random_gates)
    model_r, train_r, val_r, test_r, dev_r = train_with_two_stage_split_mlp(
        pkl_paths=random_in,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        global_feature_variant=global_feature_variant,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        train_split=train_split,
        val_within_train=val_within_train,
        model_kwargs=model_kwargs,
        seed=seed,
    )
    train_mse_r = evaluate_overall_mse_mlp(model_r, train_r, dev_r)
    test_mse_r = evaluate_overall_mse_mlp(model_r, test_r, dev_r)
    extra_loader_r = build_full_loader(random_ex, batch_size=batch_size, global_feature_variant=global_feature_variant)
    extra_mse_r = evaluate_overall_mse_mlp(model_r, extra_loader_r, dev_r)

    tim_in = select_tim_pkls(tim_base_dir, qubits=in_qubits, trotter=tim_trotter, max_trotter=None)
    tim_ex = select_tim_pkls(tim_base_dir, qubits=ex_qubits, trotter=tim_trotter, max_trotter=None)
    model_t, train_t, val_t, test_t, dev_t = train_with_two_stage_split_mlp(
        pkl_paths=tim_in,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        global_feature_variant=global_feature_variant,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        train_split=train_split,
        val_within_train=val_within_train,
        model_kwargs=model_kwargs,
        seed=seed,
    )
    train_mse_t = evaluate_overall_mse_mlp(model_t, train_t, dev_t)
    test_mse_t = evaluate_overall_mse_mlp(model_t, test_t, dev_t)
    extra_loader_t = build_full_loader(tim_ex, batch_size=batch_size, global_feature_variant=global_feature_variant)
    extra_mse_t = evaluate_overall_mse_mlp(model_t, extra_loader_t, dev_t)

    # Plots removed: results persisted only; use scripts/plot_compare_saved.py for figures


    out = {
        'random': {'train_mse': float(train_mse_r), 'test_mse': float(test_mse_r), 'extra_mse': float(extra_mse_r)},
        'tim': {'train_mse': float(train_mse_t), 'test_mse': float(test_mse_t), 'extra_mse': float(extra_mse_t)},
    }
    save_mlp_results_for_experiment(results_dir, 'qubits', out)
    return out


def run_extrapolation_depth_mlp(
    random_base_dir: str,
    tim_base_dir: str,
    results_dir: str = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    global_feature_variant: str = "binned152",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.0,
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    device: Optional[str] = None,
    seed: int = 42,
):
    in_qubits = [2, 3, 4, 5, 6]
    results_dir = _ensure_results_dir(results_dir or "/data/P70087789/GNN/models/results")

    random_in = select_random_pkls(random_base_dir, qubits=in_qubits, gates_range=(1, 79))
    random_ex = select_random_pkls(random_base_dir, qubits=in_qubits, gates_range=(80, None))
    model_r, train_r, val_r, test_r, dev_r = train_with_two_stage_split_mlp(
        pkl_paths=random_in,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        global_feature_variant=global_feature_variant,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        train_split=train_split,
        val_within_train=val_within_train,
        model_kwargs=model_kwargs,
        seed=seed,
    )
    train_mse_r = evaluate_overall_mse_mlp(model_r, train_r, dev_r)
    test_mse_r = evaluate_overall_mse_mlp(model_r, test_r, dev_r)
    extra_loader_r = build_full_loader(random_ex, batch_size=batch_size, global_feature_variant=global_feature_variant)
    extra_mse_r = evaluate_overall_mse_mlp(model_r, extra_loader_r, dev_r)

    tim_in = select_tim_pkls(tim_base_dir, qubits=in_qubits, trotter=[1, 2, 3, 4], max_trotter=None)
    tim_ex = select_tim_pkls(tim_base_dir, qubits=in_qubits, trotter=[5], max_trotter=None)
    model_t, train_t, val_t, test_t, dev_t = train_with_two_stage_split_mlp(
        pkl_paths=tim_in,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        global_feature_variant=global_feature_variant,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        train_split=train_split,
        val_within_train=val_within_train,
        model_kwargs=model_kwargs,
        seed=seed,
    )
    train_mse_t = evaluate_overall_mse_mlp(model_t, train_t, dev_t)
    test_mse_t = evaluate_overall_mse_mlp(model_t, test_t, dev_t)
    extra_loader_t = build_full_loader(tim_ex, batch_size=batch_size, global_feature_variant=global_feature_variant)
    extra_mse_t = evaluate_overall_mse_mlp(model_t, extra_loader_t, dev_t)

    # Plots removed: results persisted only; use scripts/plot_compare_saved.py for figures


    out = {
        'random': {'train_mse': float(train_mse_r), 'test_mse': float(test_mse_r), 'extra_mse': float(extra_mse_r)},
        'tim': {'train_mse': float(train_mse_t), 'test_mse': float(test_mse_t), 'extra_mse': float(extra_mse_t)},
    }
    save_mlp_results_for_experiment(results_dir, 'depth', out)
    return out


def run_reverse_extrapolation_depth_mlp(
    random_base_dir: str,
    tim_base_dir: str,
    results_dir: str = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    global_feature_variant: str = "binned152",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.0,
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    device: Optional[str] = None,
    seed: int = 42,
):
    in_qubits = [2, 3, 4, 5, 6]
    results_dir = _ensure_results_dir(results_dir or "/data/P70087789/GNN/models/results")

    random_in = select_random_pkls(random_base_dir, qubits=in_qubits, gates_range=(20, 99))
    random_ex = select_random_pkls(random_base_dir, qubits=in_qubits, gates_range=(None, 19))
    model_r, train_r, val_r, test_r, dev_r = train_with_two_stage_split_mlp(
        pkl_paths=random_in,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        global_feature_variant=global_feature_variant,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        train_split=train_split,
        val_within_train=val_within_train,
        model_kwargs=model_kwargs,
        seed=seed,
    )
    train_mse_r = evaluate_overall_mse_mlp(model_r, train_r, dev_r)
    test_mse_r = evaluate_overall_mse_mlp(model_r, test_r, dev_r)
    extra_loader_r = build_full_loader(random_ex, batch_size=batch_size, global_feature_variant=global_feature_variant)
    extra_mse_r = evaluate_overall_mse_mlp(model_r, extra_loader_r, dev_r)

    tim_in = select_tim_pkls(tim_base_dir, qubits=in_qubits, trotter=[1, 2, 3, 4], max_trotter=None)
    tim_ex = select_tim_pkls(tim_base_dir, qubits=in_qubits, trotter=[5], max_trotter=None)
    model_t, train_t, val_t, test_t, dev_t = train_with_two_stage_split_mlp(
        pkl_paths=tim_in,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        global_feature_variant=global_feature_variant,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        train_split=train_split,
        val_within_train=val_within_train,
        model_kwargs=model_kwargs,
        seed=seed,
    )
    train_mse_t = evaluate_overall_mse_mlp(model_t, train_t, dev_t)
    test_mse_t = evaluate_overall_mse_mlp(model_t, test_t, dev_t)
    extra_loader_t = build_full_loader(tim_ex, batch_size=batch_size, global_feature_variant=global_feature_variant)
    extra_mse_t = evaluate_overall_mse_mlp(model_t, extra_loader_t, dev_t)

    # Plots removed: results persisted only; use scripts/plot_compare_saved.py for figures


    out = {
        'random': {'train_mse': float(train_mse_r), 'test_mse': float(test_mse_r), 'extra_mse': float(extra_mse_r)},
        'tim': {'train_mse': float(train_mse_t), 'test_mse': float(test_mse_t), 'extra_mse': float(extra_mse_t)},
    }
    save_mlp_results_for_experiment(results_dir, 'depth_reverse', out)
    return out


def main():
    cfg = CONFIG
    set_global_seed(int(cfg.get("seed", 42)))
    exp = cfg["experiment"]
    common = dict(
        random_base_dir=cfg["random_base_dir"],
        tim_base_dir=cfg["tim_base_dir"],
        results_dir=cfg["results_dir"],
        model_kwargs=cfg["model_kwargs"],
        global_feature_variant=cfg["global_feature_variant"],
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        early_stopping_patience=cfg["early_stopping_patience"],
        early_stopping_min_delta=cfg["early_stopping_min_delta"],
        train_split=cfg["train_split"],
        val_within_train=cfg["val_within_train"],
        device=cfg["device"],
        seed=cfg["seed"],
    )

    if exp == "qubits":
        results = run_extrapolation_qubits_mlp(
            tim_trotter=cfg.get("tim_trotter"),
            tim_max_trotter=None,
            random_gates=cfg.get("random_gates"),
            **common,
        )
    elif exp == "depth":
        results = run_extrapolation_depth_mlp(**common)
    elif exp == "depth_reverse":
        results = run_reverse_extrapolation_depth_mlp(**common)
    else:
        raise ValueError(f"Unknown experiment: {exp}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


