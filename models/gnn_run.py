import os
import json
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch

# Ensure project root on sys.path for model imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent if script_dir.name == "models" else script_dir
if str(project_root) not in sys.path:
    # Prepend so it takes precedence over the script directory (avoids models.py shadowing the package)
    sys.path.insert(0, str(project_root))

from models.gnn import (
    build_full_loader,
    train_with_two_stage_split,
    evaluate_overall_mse,
    evaluate_overall_r2,
    grid_search,
)


CONFIG: Dict[str, Any] = {
    # One of: "qubits", "depth", "depth_reverse", "per_qubit"
    "experiment": "depth",

    # Data dirs
    "random_base_dir": "/data/P70087789/GNN/data/dataset_regression/dataset_random",
    "tim_base_dir": "/data/P70087789/GNN/data/dataset_regression/dataset_tim",
    "results_dir": "/data/P70087789/GNN/models/results",

    # Features/architecture/training
    "global_feature_variant": "binned152",  
    # Node feature backend augmentation: None (noiseless) or a backend tag like "fake_oslo"
    "node_feature_backend_variant":None,
    "epochs": 200,
    "lr": 1e-3,
    "batch_size": 64,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.0,
    "train_split": 0.8,
    "val_within_train": 0.1,
    "device": None,  # e.g. "cuda", "cpu" or None for auto
    "seed": 42,
    # Grid search toggles
    "do_grid_search": False,
    "grid_epochs": 25,
    # Default loss (can be overridden via --loss-type)
    "loss_type": "huber",
    "model_kwargs": {
        "gnn_hidden": 32,
        "gnn_heads": 2,
        "global_hidden": 64,
        "reg_hidden": 128,
    },

    # Experiment-specific
    "tim_trotter": [1, 2, 3, 4, 5],
    "random_gates": None,  # e.g. [50, 100]
    # Multiseed default
    "num_seeds": 10,
}


def set_global_seed(seed: int) -> None:
    import random
    import numpy as np
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
# =========================================
# Hyperparameter grid (small) -- removed (unused)
# =========================================


def big_gnn_grid_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    # Vary requested hyperparameters: loss_type, num_layers, gnn_hidden size
    loss_types = ["huber", "mse"]
    num_layers_list = [3, 4, 5]
    gnn_hidden_list = [32, 64, 128]
    # Keep heads and other dims at defaults unless provided externally
    for loss_type in loss_types:
        for num_layers in num_layers_list:
            for gnn_hidden in gnn_hidden_list:
                configs.append({
                    "loss_type": str(loss_type),
                    "num_layers": int(num_layers),
                    "gnn_hidden": int(gnn_hidden),
                })
    return configs


# =========================================
# Results persistence
# =========================================
GNN_RESULTS_FILENAME = "gnn_results.pkl"

def get_gnn_results_path(results_dir: str) -> str:
    return os.path.join(results_dir, GNN_RESULTS_FILENAME)
def load_all_gnn_results(results_dir: str) -> Optional[Dict[str, Any]]:
    path = get_gnn_results_path(results_dir)
    if not os.path.exists(path):
        return None
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_gnn_results_for_experiment(results_dir: str, experiment: str, results: Any) -> str:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    existing = load_all_gnn_results(results_dir) or {}
    existing[experiment] = results
    path = get_gnn_results_path(results_dir)
    import pickle
    with open(path, "wb") as f:
        pickle.dump(existing, f)
    return path


def load_gnn_results_for_experiment(results_dir: str, experiment: str) -> Optional[Dict[str, Any]]:
    data = load_all_gnn_results(results_dir)
    if data is None:
        return None
    return data.get(experiment)


# Also save grid search results as standalone PKL files (timestamped)
def save_grid_results(
    results_dir: str,
    experiment_tag: str,
    grid_results: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"grid_{experiment_tag}_{ts}.pkl"
    out_path = os.path.join(results_dir, fname)
    payload: Dict[str, Any] = {
        'experiment': experiment_tag,
        'results': grid_results,
        'meta': meta or {},
    }
    import pickle
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)
    return out_path


# =========================================
# Dataset selection utilities
# =========================================
def _extract_int_after_token(name: str, token: str) -> Optional[int]:
    try:
        pos = name.index(token) + len(token)
        digits: List[str] = []
        while pos < len(name) and name[pos].isdigit():
            digits.append(name[pos])
            pos += 1
        return int("".join(digits)) if digits else None
    except Exception:
        return None


def _extract_range_after_token(name: str, token: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        pos = name.index(token) + len(token)
        start_digits: List[str] = []
        while pos < len(name) and name[pos].isdigit():
            start_digits.append(name[pos])
            pos += 1
        if not start_digits:
            return None, None
        start_val = int("".join(start_digits))
        if pos < len(name) and name[pos] == '-':
            pos += 1
            end_digits: List[str] = []
            while pos < len(name) and name[pos].isdigit():
                end_digits.append(name[pos])
                pos += 1
            if end_digits:
                end_val = int("".join(end_digits))
            else:
                end_val = start_val
        else:
            end_val = start_val
        return start_val, end_val
    except Exception:
        return None, None


def select_random_pkls(
    base_dir: str,
    qubits: List[int],
    gates: Optional[List[int]] = None,
    gates_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> List[str]:
    selected: List[str] = []
    base_dir_abs = os.path.abspath(base_dir)
    for name in sorted(os.listdir(base_dir_abs)):
        if not name.endswith('.pkl'):
            continue
        if 'qubits_' not in name and 'basis_rotations+cx_qubits_' not in name:
            continue
        q = _extract_int_after_token(name, 'qubits_')
        if q is None or int(q) not in set(int(x) for x in qubits):
            continue
        if gates_range is not None:
            gstart, gend = _extract_range_after_token(name, 'gates_')
            gmin, gmax = gates_range
            if gstart is None or gend is None:
                continue
            qmin = gmin if gmin is not None else -float('inf')
            qmax = gmax if gmax is not None else float('inf')
            if gend < qmin or gstart > qmax:
                continue
        elif gates is not None:
            g = _extract_int_after_token(name, 'gates_')
            if g is None or int(g) not in set(int(x) for x in gates):
                continue
        selected.append(os.path.abspath(os.path.join(base_dir_abs, name)))
    if not selected:
        raise FileNotFoundError(f"No random PKLs matched in {base_dir} for qubits={qubits} gates={gates}")
    return selected


def select_tim_pkls(
    base_dir: str,
    qubits: List[int],
    trotter: Optional[List[int]] = None,
    max_trotter: Optional[int] = None,
) -> List[str]:
    selected: List[str] = []
    base_dir_abs = os.path.abspath(base_dir)
    for name in sorted(os.listdir(base_dir_abs)):
        if not name.endswith('.pkl'):
            continue
        if 'ising_qubits_' not in name and 'qubits_' not in name:
            continue
        q = _extract_int_after_token(name, 'qubits_')
        if q is None:
            q = _extract_int_after_token(name, 'ising_qubits_')
        if q is None or int(q) not in set(int(x) for x in qubits):
            continue
        steps = _extract_int_after_token(name, 'trotter_')
        if trotter is not None:
            if steps is None or int(steps) not in set(int(x) for x in trotter):
                continue
        elif max_trotter is not None:
            if steps is None or int(steps) > int(max_trotter):
                continue
        selected.append(os.path.abspath(os.path.join(base_dir_abs, name)))
    if not selected:
        raise FileNotFoundError(f"No TIM PKLs matched in {base_dir} for qubits={qubits} trotter={trotter} max_trotter={max_trotter}")
    return selected


# =========================================
# Experiments (callable from main())
# =========================================


def run_extrapolation_depth(*args, **kwargs):
    raise NotImplementedError("Use run_extrapolation_depth_multiseed instead.")


def run_extrapolation_depth_multiseed(
    random_base_dir: str,
    tim_base_dir: str,
    random_noisy_dir: Optional[str] = None,
    tim_noisy_dir: Optional[str] = None,
    results_dir: str = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    global_feature_variant: str = "binned152",
    node_feature_backend_variant: Optional[str] = None,
    use_noisy_datasets: bool = False,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.0,
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    device: Optional[str] = None,
    base_seed: int = 42,
    num_seeds: int = 10,
    do_grid_search: bool = True,
    grid_epochs: int = 25,
    loss_type: Optional[str] = None,
):
    in_qubits = [2, 3, 4, 5, 6]
    results_dir = results_dir or "/data/P70087789/GNN/models/results"
    os.makedirs(results_dir, exist_ok=True)

    base_dir_random = random_base_dir
    base_dir_tim = tim_base_dir

    # Select in-domain and extrapolation sets once (constant across seeds)
    random_in = select_random_pkls(base_dir_random, qubits=in_qubits, gates_range=(1, 79))
    random_ex = select_random_pkls(base_dir_random, qubits=in_qubits, gates_range=(80, None))
    tim_in = select_tim_pkls(base_dir_tim, qubits=in_qubits, trotter=[1, 2, 3, 4], max_trotter=None)
    tim_ex = select_tim_pkls(base_dir_tim, qubits=in_qubits, trotter=[5], max_trotter=None)

    # Optional: small grid search once on Random in-domain to pick model hyperparameters
    chosen_model_kwargs: Dict[str, Any] = dict(model_kwargs or {})
    if do_grid_search:
        configs = big_gnn_grid_configs()
        grid_res = grid_search(
            pkl_paths=random_in,
            configs=configs,
            train_split=train_split,
            epochs=int(grid_epochs),
            batch_size=batch_size,
            lr=lr,
            seed=base_seed,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
        )
        # Save grid results
        try:
            save_grid_results(results_dir, 'depth_multiseed', grid_res, meta={
                'dataset': 'random_in',
                'use_noisy_datasets': bool(use_noisy_datasets),
                'global_feature_variant': str(global_feature_variant),
                'node_feature_backend_variant': (str(node_feature_backend_variant) if node_feature_backend_variant else None),
                'grid_epochs': int(grid_epochs),
                'base_seed': int(base_seed),
            })
        except Exception as e:
            print(f"[WARN] Failed to save grid results for depth_multiseed: {e}")
        if len(grid_res) > 0:
            best = grid_res[0]
            chosen_model_kwargs.update({
                "gnn_hidden": int(best.get("gnn_hidden", chosen_model_kwargs.get("gnn_hidden", 32))),
                "gnn_heads": int(best.get("gnn_heads", chosen_model_kwargs.get("gnn_heads", 2))),
                "global_hidden": int(best.get("global_hidden", chosen_model_kwargs.get("global_hidden", 64))),
                "reg_hidden": int(best.get("reg_hidden", chosen_model_kwargs.get("reg_hidden", 128))),
                "num_layers": int(best.get("num_layers", chosen_model_kwargs.get("num_layers", 3))),
            })
            chosen_loss_type = str(best.get("loss_type", "huber"))
        else:
            chosen_loss_type = "huber"
    else:
        chosen_loss_type = "huber"

    # Explicit override from CLI if provided
    if loss_type is not None:
        chosen_loss_type = str(loss_type).lower()

    seeds = [base_seed + i for i in range(num_seeds)]

    def _run_one(seed_val: int) -> Dict[str, Dict[str, float]]:
        # Random
        model_r, train_r, val_r, test_r, dev_r = train_with_two_stage_split(
            pkl_paths=random_in,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            train_split=train_split,
            val_within_train=val_within_train,
            model_kwargs=chosen_model_kwargs or model_kwargs,
            loss_type=chosen_loss_type,
            seed=seed_val,
        )
        train_mse_r = float(evaluate_overall_mse(model_r, train_r, dev_r))
        test_mse_r = float(evaluate_overall_mse(model_r, test_r, dev_r))
        train_r2_r = float(evaluate_overall_r2(model_r, train_r, dev_r))
        test_r2_r = float(evaluate_overall_r2(model_r, test_r, dev_r))
        extra_loader_r = build_full_loader(
            random_ex,
            batch_size=batch_size,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
        )
        extra_mse_r = float(evaluate_overall_mse(model_r, extra_loader_r, dev_r))
        extra_r2_r = float(evaluate_overall_r2(model_r, extra_loader_r, dev_r))

        # TIM
        model_t, train_t, val_t, test_t, dev_t = train_with_two_stage_split(
            pkl_paths=tim_in,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            train_split=train_split,
            val_within_train=val_within_train,
            model_kwargs=chosen_model_kwargs or model_kwargs,
            loss_type=chosen_loss_type,
            seed=seed_val,
        )
        train_mse_t = float(evaluate_overall_mse(model_t, train_t, dev_t))
        test_mse_t = float(evaluate_overall_mse(model_t, test_t, dev_t))
        train_r2_t = float(evaluate_overall_r2(model_t, train_t, dev_t))
        test_r2_t = float(evaluate_overall_r2(model_t, test_t, dev_t))
        extra_loader_t = build_full_loader(
            tim_ex,
            batch_size=batch_size,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
        )
        extra_mse_t = float(evaluate_overall_mse(model_t, extra_loader_t, dev_t))
        extra_r2_t = float(evaluate_overall_r2(model_t, extra_loader_t, dev_t))

        return {
            'random': {
                'train_mse': train_mse_r, 'test_mse': test_mse_r, 'extra_mse': extra_mse_r,
                'train_r2': train_r2_r, 'test_r2': test_r2_r, 'extra_r2': extra_r2_r,
            },
            'tim': {
                'train_mse': train_mse_t, 'test_mse': test_mse_t, 'extra_mse': extra_mse_t,
                'train_r2': train_r2_t, 'test_r2': test_r2_t, 'extra_r2': extra_r2_t,
            },
        }

    # Run all seeds and aggregate
    all_runs: List[Dict[str, Dict[str, float]]] = []
    for s in seeds:
        res = _run_one(s)
        all_runs.append(res)

    def _avg(keys: List[str], items: List[float]) -> float:
        return float(sum(items) / max(1, len(items)))

    def _aggregate(domain: str) -> Dict[str, float]:
        metrics = ['train_mse', 'test_mse', 'extra_mse', 'train_r2', 'test_r2', 'extra_r2']
        out: Dict[str, float] = {}
        for m in metrics:
            out[m] = _avg([m], [run[domain][m] for run in all_runs])
        return out

    aggregated = {
        'random': _aggregate('random'),
        'tim': _aggregate('tim'),
        'seeds': seeds,
        'num_seeds': int(len(seeds)),
        'global_feature_variant': str(global_feature_variant),
        'node_feature_backend_variant': (str(node_feature_backend_variant) if node_feature_backend_variant else None),
    }

    # Save multiseed results to the results directory with the requested filename
    out_path = os.path.join(results_dir, 'gnn_multiseed_results.pkl')
    import pickle
    with open(out_path, 'wb') as f:
        pickle.dump(aggregated, f)

    return aggregated


def run_extrapolation_qubits_multiseed(
    random_base_dir: str,
    tim_base_dir: str,
    random_noisy_dir: Optional[str] = None,
    tim_noisy_dir: Optional[str] = None,
    results_dir: str = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    global_feature_variant: str = "binned152",
    node_feature_backend_variant: Optional[str] = None,
    use_noisy_datasets: bool = False,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.0,
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    device: Optional[str] = None,
    base_seed: int = 42,
    num_seeds: int = 10,
    do_grid_search: bool = True,
    grid_epochs: int = 25,
    loss_type: Optional[str] = None,
    tim_trotter: Optional[List[int]] = None,
    tim_max_trotter: Optional[int] = None,
    random_gates: Optional[List[int]] = None,
):
    in_qubits = [2, 3, 4, 5]
    ex_qubits = [6]
    results_dir = results_dir or "/data/P70087789/GNN/models/results"
    os.makedirs(results_dir, exist_ok=True)

    base_dir_random = random_base_dir
    base_dir_tim = tim_base_dir

    random_in = select_random_pkls(base_dir_random, qubits=in_qubits, gates=random_gates)
    random_ex = select_random_pkls(base_dir_random, qubits=ex_qubits, gates=random_gates)

    # TIM dataset selection
    if tim_trotter is None and tim_max_trotter is None:
        tim_trotter = [1, 2, 3, 4, 5]
        tim_max_trotter = None
    tim_in = select_tim_pkls(base_dir_tim, qubits=in_qubits, trotter=tim_trotter, max_trotter=tim_max_trotter)
    tim_ex = select_tim_pkls(base_dir_tim, qubits=ex_qubits, trotter=tim_trotter, max_trotter=tim_max_trotter)

    # Optional grid search once on Random in-domain to pick model hyperparameters
    chosen_model_kwargs: Dict[str, Any] = dict(model_kwargs or {})
    if do_grid_search:
        configs = big_gnn_grid_configs()
        grid_res = grid_search(
            pkl_paths=random_in,
            configs=configs,
            train_split=train_split,
            epochs=int(grid_epochs),
            batch_size=batch_size,
            lr=lr,
            seed=base_seed,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
        )
        # Save grid results
        try:
            save_grid_results(results_dir, 'qubits_multiseed', grid_res, meta={
                'dataset': 'random_in',
                'use_noisy_datasets': bool(use_noisy_datasets),
                'global_feature_variant': str(global_feature_variant),
                'node_feature_backend_variant': (str(node_feature_backend_variant) if node_feature_backend_variant else None),
                'grid_epochs': int(grid_epochs),
                'base_seed': int(base_seed),
            })
        except Exception as e:
            print(f"[WARN] Failed to save grid results for qubits_multiseed: {e}")
        if len(grid_res) > 0:
            best = grid_res[0]
            chosen_model_kwargs.update({
                "gnn_hidden": int(best.get("gnn_hidden", chosen_model_kwargs.get("gnn_hidden", 32))),
                "gnn_heads": int(best.get("gnn_heads", chosen_model_kwargs.get("gnn_heads", 2))),
                "global_hidden": int(best.get("global_hidden", chosen_model_kwargs.get("global_hidden", 64))),
                "reg_hidden": int(best.get("reg_hidden", chosen_model_kwargs.get("reg_hidden", 128))),
                "num_layers": int(best.get("num_layers", chosen_model_kwargs.get("num_layers", 3))),
            })
            chosen_loss_type = str(best.get("loss_type", "huber"))
        else:
            chosen_loss_type = "huber"
    else:
        chosen_loss_type = "huber"

    # Explicit override from CLI if provided
    if loss_type is not None:
        chosen_loss_type = str(loss_type).lower()

    seeds = [base_seed + i for i in range(num_seeds)]

    def _run_one(seed_val: int) -> Dict[str, Dict[str, float]]:
        # Random
        model_r, train_r, val_r, test_r, dev_r = train_with_two_stage_split(
            pkl_paths=random_in,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            train_split=train_split,
            val_within_train=val_within_train,
            model_kwargs=chosen_model_kwargs or model_kwargs,
            loss_type=chosen_loss_type,
            seed=seed_val,
        )
        train_mse_r = float(evaluate_overall_mse(model_r, train_r, dev_r))
        test_mse_r = float(evaluate_overall_mse(model_r, test_r, dev_r))
        train_r2_r = float(evaluate_overall_r2(model_r, train_r, dev_r))
        test_r2_r = float(evaluate_overall_r2(model_r, test_r, dev_r))
        extra_loader_r = build_full_loader(
            random_ex,
            batch_size=batch_size,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
        )
        extra_mse_r = float(evaluate_overall_mse(model_r, extra_loader_r, dev_r))
        extra_r2_r = float(evaluate_overall_r2(model_r, extra_loader_r, dev_r))

        # TIM
        model_t, train_t, val_t, test_t, dev_t = train_with_two_stage_split(
            pkl_paths=tim_in,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            train_split=train_split,
            val_within_train=val_within_train,
            model_kwargs=chosen_model_kwargs or model_kwargs,
            loss_type=chosen_loss_type,
            seed=seed_val,
        )
        train_mse_t = float(evaluate_overall_mse(model_t, train_t, dev_t))
        test_mse_t = float(evaluate_overall_mse(model_t, test_t, dev_t))
        train_r2_t = float(evaluate_overall_r2(model_t, train_t, dev_t))
        test_r2_t = float(evaluate_overall_r2(model_t, test_t, dev_t))
        extra_loader_t = build_full_loader(
            tim_ex,
            batch_size=batch_size,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
        )
        extra_mse_t = float(evaluate_overall_mse(model_t, extra_loader_t, dev_t))
        extra_r2_t = float(evaluate_overall_r2(model_t, extra_loader_t, dev_t))

        return {
            'random': {
                'train_mse': train_mse_r, 'test_mse': test_mse_r, 'extra_mse': extra_mse_r,
                'train_r2': train_r2_r, 'test_r2': test_r2_r, 'extra_r2': extra_r2_r,
            },
            'tim': {
                'train_mse': train_mse_t, 'test_mse': test_mse_t, 'extra_mse': extra_mse_t,
                'train_r2': train_r2_t, 'test_r2': test_r2_t, 'extra_r2': extra_r2_t,
            },
        }

    # Run all seeds and aggregate
    all_runs: List[Dict[str, Dict[str, float]]] = []
    for s in seeds:
        res = _run_one(s)
        all_runs.append(res)

    def _avg(items: List[float]) -> float:
        return float(sum(items) / max(1, len(items)))

    def _aggregate(domain: str) -> Dict[str, float]:
        metrics = ['train_mse', 'test_mse', 'extra_mse', 'train_r2', 'test_r2', 'extra_r2']
        out: Dict[str, float] = {}
        for m in metrics:
            out[m] = _avg([run[domain][m] for run in all_runs])
        return out

    aggregated = {
        'random': _aggregate('random'),
        'tim': _aggregate('tim'),
        'seeds': seeds,
        'num_seeds': int(len(seeds)),
        'use_noisy_datasets': bool(use_noisy_datasets),
        'global_feature_variant': str(global_feature_variant),
        'node_feature_backend_variant': (str(node_feature_backend_variant) if node_feature_backend_variant else None),
        'tim_trotter': tim_trotter,
        'random_gates': random_gates,
    }

    # Save multiseed results to the results directory
    out_path = os.path.join(results_dir, 'gnn_multiseed_qubits_results.pkl')
    import pickle
    with open(out_path, 'wb') as f:
        pickle.dump(aggregated, f)

    return aggregated


def _parse_list_arg(arg: Optional[str]) -> Optional[List[int]]:
    if arg is None:
        return None
    arg = arg.strip()
    if arg == "" or arg.lower() == "none":
        return None
    parts = [p.strip() for p in arg.split(',') if p.strip() != ""]
    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def parse_cli_args() -> Optional[Dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Run GNN extrapolation experiments.")
    parser.add_argument('--experiment', '-e', type=str, default=None,
                        choices=['qubits', 'depth', 'depth_reverse', 'per_qubit'],
                        help="Experiment to run (qubits/depth use multiseed).")
    parser.add_argument('--random-base-dir', type=str, default=None)
    parser.add_argument('--tim-base-dir', type=str, default=None)
    # Removed noisy-dir arguments; noisy Oslo runs are handled in gnn_run_oslo.py
    parser.add_argument('--results-dir', type=str, default=None)
    # Removed noisy dataset switch; handled by dedicated script
    parser.add_argument('--global-feature-variant', type=str, default=None)
    parser.add_argument('--node-feature-backend-variant', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--early-stopping-patience', type=int, default=None)
    parser.add_argument('--early-stopping-min-delta', type=float, default=None)
    parser.add_argument('--train-split', type=float, default=None)
    parser.add_argument('--val-within-train', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None, help='Base seed for multiseed runs')
    parser.add_argument('--num-seeds', type=int, default=None)
    parser.add_argument('--do-grid-search', dest='do_grid_search', action='store_true')
    parser.add_argument('--no-grid-search', dest='do_grid_search', action='store_false')
    parser.set_defaults(do_grid_search=None)
    parser.add_argument('--grid-epochs', type=int, default=None)
    parser.add_argument('--tim-trotter', type=str, default=None, help='Comma-separated trotter steps for TIM')
    parser.add_argument('--random-gates', type=str, default=None, help='Comma-separated gate counts for Random')
    parser.add_argument('--loss-type', type=str, choices=['huber','mse'], default=None,
                        help='Override loss function for training (default depends on grid search).')

    # If no args besides program name, return None to use CONFIG
    import sys as _sys
    if len(_sys.argv) <= 1:
        return None

    args = parser.parse_args()
    cfg_updates: Dict[str, Any] = {}
    for k in [
        'experiment', 'random_base_dir', 'tim_base_dir', 'random_noisy_dir', 'tim_noisy_dir', 'results_dir',
        'global_feature_variant', 'node_feature_backend_variant', 'epochs', 'lr', 'batch_size',
        'early_stopping_patience', 'early_stopping_min_delta', 'train_split', 'val_within_train', 'device',
        'seed', 'num_seeds', 'grid_epochs', 'loss_type']:
        v = getattr(args, k, None)
        if v is not None:
            cfg_updates[k] = v
    if args.do_grid_search is not None:
        cfg_updates['do_grid_search'] = bool(args.do_grid_search)
    # No noisy switch propagation
    tlist = _parse_list_arg(getattr(args, 'tim_trotter', None))
    if tlist is not None:
        cfg_updates['tim_trotter'] = tlist
    glist = _parse_list_arg(getattr(args, 'random_gates', None))
    if glist is not None:
        cfg_updates['random_gates'] = glist
    return cfg_updates



def run_reverse_extrapolation_depth(
    random_base_dir: str,
    tim_base_dir: str,
    random_noisy_dir: Optional[str] = None,
    tim_noisy_dir: Optional[str] = None,
    results_dir: str = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    global_feature_variant: str = "binned152",
    node_feature_backend_variant: Optional[str] = None,
    use_noisy_datasets: bool = False,
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
    results_dir = results_dir or "/data/P70087789/GNN/models/results"
    os.makedirs(results_dir, exist_ok=True)

    base_dir_random = random_noisy_dir if use_noisy_datasets else random_base_dir
    base_dir_tim = tim_noisy_dir if use_noisy_datasets else tim_base_dir
    random_in = select_random_pkls(base_dir_random, qubits=in_qubits, gates_range=(20, 99))
    random_ex = select_random_pkls(base_dir_random, qubits=in_qubits, gates_range=(None, 19))
    model_r, train_r, val_r, test_r, dev_r = train_with_two_stage_split(
        pkl_paths=random_in,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        train_split=train_split,
        val_within_train=val_within_train,
        model_kwargs=model_kwargs,
        seed=seed,
    )
    train_mse_r = evaluate_overall_mse(model_r, train_r, dev_r)
    test_mse_r = evaluate_overall_mse(model_r, test_r, dev_r)
    extra_loader_r = build_full_loader(
        random_ex,
        batch_size=batch_size,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )
    extra_mse_r = evaluate_overall_mse(model_r, extra_loader_r, dev_r)

    tim_in = select_tim_pkls(base_dir_tim, qubits=in_qubits, trotter=[1, 2, 3, 4], max_trotter=None)
    tim_ex = select_tim_pkls(base_dir_tim, qubits=in_qubits, trotter=[5], max_trotter=None)
    model_t, train_t, val_t, test_t, dev_t = train_with_two_stage_split(
        pkl_paths=tim_in,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        train_split=train_split,
        val_within_train=val_within_train,
        model_kwargs=model_kwargs,
        seed=seed,
    )
    train_mse_t = evaluate_overall_mse(model_t, train_t, dev_t)
    test_mse_t = evaluate_overall_mse(model_t, test_t, dev_t)
    extra_loader_t = build_full_loader(
        tim_ex,
        batch_size=batch_size,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )
    extra_mse_t = evaluate_overall_mse(model_t, extra_loader_t, dev_t)

    out = {
        'random': {'train_mse': float(train_mse_r), 'test_mse': float(test_mse_r), 'extra_mse': float(extra_mse_r)},
        'tim': {'train_mse': float(train_mse_t), 'test_mse': float(test_mse_t), 'extra_mse': float(extra_mse_t)},
    }
    tag = 'depth_reverse_noisy' if use_noisy_datasets else 'depth_reverse'
    save_gnn_results_for_experiment(results_dir, tag, out)
    return out


def run_per_qubit_experiments(
    random_base_dir: str,
    tim_base_dir: str,
    random_noisy_dir: Optional[str] = None,
    tim_noisy_dir: Optional[str] = None,
    results_dir: str = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    global_feature_variant: str = "binned152",
    node_feature_backend_variant: Optional[str] = None,
    use_noisy_datasets: bool = False,
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
    qubits = [2, 3, 4, 5, 6]
    results_dir = results_dir or "/data/P70087789/GNN/models/results"
    os.makedirs(results_dir, exist_ok=True)

    random_train_mses: List[float] = []
    random_test_mses: List[float] = []
    tim_train_mses: List[float] = []
    tim_test_mses: List[float] = []

    results: Dict[str, Dict[int, Dict[str, float]]] = {'random': {}, 'tim': {}}

    for q in qubits:
        base_dir_random = random_noisy_dir if use_noisy_datasets else random_base_dir
        base_dir_tim = tim_noisy_dir if use_noisy_datasets else tim_base_dir
        random_paths = select_random_pkls(base_dir_random, qubits=[q])
        model_r, train_r, val_r, test_r, dev_r = train_with_two_stage_split(
            pkl_paths=random_paths,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            train_split=train_split,
            val_within_train=val_within_train,
            model_kwargs=model_kwargs,
            seed=seed,
        )
        train_mse_r = evaluate_overall_mse(model_r, train_r, dev_r)
        test_mse_r = evaluate_overall_mse(model_r, test_r, dev_r)
        random_train_mses.append(float(train_mse_r))
        random_test_mses.append(float(test_mse_r))
        results['random'][q] = {'train_mse': float(train_mse_r), 'test_mse': float(test_mse_r)}

        tim_paths = select_tim_pkls(base_dir_tim, qubits=[q], trotter=[1, 2, 3, 4, 5], max_trotter=None)
        model_t, train_t, val_t, test_t, dev_t = train_with_two_stage_split(
            pkl_paths=tim_paths,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            train_split=train_split,
            val_within_train=val_within_train,
            model_kwargs=model_kwargs,
            seed=seed,
        )
        train_mse_t = evaluate_overall_mse(model_t, train_t, dev_t)
        test_mse_t = evaluate_overall_mse(model_t, test_t, dev_t)
        tim_train_mses.append(float(train_mse_t))
        tim_test_mses.append(float(test_mse_t))
        results['tim'][q] = {'train_mse': float(train_mse_t), 'test_mse': float(test_mse_t)}

    tag = 'per_qubit_noisy' if use_noisy_datasets else 'per_qubit'
    save_gnn_results_for_experiment(results_dir, tag, results)
    return results


def main():
    cfg = dict(CONFIG)
    updates = parse_cli_args()
    if updates is not None:
        cfg.update(updates)
    exp = cfg["experiment"]
    set_global_seed(int(cfg.get("seed", 42)))
    common = dict(
        random_base_dir=cfg["random_base_dir"],
        tim_base_dir=cfg["tim_base_dir"],
        random_noisy_dir=None,
        tim_noisy_dir=None,
        results_dir=cfg["results_dir"],
        model_kwargs=cfg["model_kwargs"],
        global_feature_variant=cfg["global_feature_variant"],
        node_feature_backend_variant=cfg["node_feature_backend_variant"],
        use_noisy_datasets=cfg.get("use_noisy_datasets", False),
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        early_stopping_patience=cfg["early_stopping_patience"],
        early_stopping_min_delta=cfg["early_stopping_min_delta"],
        train_split=cfg["train_split"],
        val_within_train=cfg["val_within_train"],
        device=cfg["device"],
        seed=cfg["seed"],
        do_grid_search=cfg.get("do_grid_search", True),
        grid_epochs=cfg.get("grid_epochs", 25),
    )

    if exp in ("qubits", "qubits_multiseed"):
        results = run_extrapolation_qubits_multiseed(
            **{k: v for k, v in common.items() if k not in ('seed', 'do_grid_search', 'grid_epochs')},
            base_seed=int(cfg.get("seed", 42)),
            num_seeds=int(cfg.get("num_seeds", 10)),
            do_grid_search=bool(cfg.get("do_grid_search", True)),
            grid_epochs=int(cfg.get("grid_epochs", 25)),
            tim_trotter=cfg.get("tim_trotter"),
            tim_max_trotter=None,
            random_gates=cfg.get("random_gates"),
            loss_type=cfg.get("loss_type"),
        )
    elif exp in ("depth", "depth_multiseed"):
        results = run_extrapolation_depth_multiseed(
            **{k: v for k, v in common.items() if k not in ('seed', 'do_grid_search', 'grid_epochs')},
            base_seed=int(cfg.get("seed", 42)),
            num_seeds=int(cfg.get("num_seeds", 10)),
            do_grid_search=bool(cfg.get("do_grid_search", True)),
            grid_epochs=int(cfg.get("grid_epochs", 25)),
            loss_type=cfg.get("loss_type"),
        )
    elif exp == "depth_reverse":
        # Filter out keys unused by reverse experiment
        reverse_common = {k: v for k, v in common.items() if k not in ("do_grid_search", "grid_epochs")}
        results = run_reverse_extrapolation_depth(**reverse_common)
    elif exp == "per_qubit":
        results = run_per_qubit_experiments(**common)
    else:
        raise ValueError(f"Unknown experiment: {exp}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


