import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch

from models.gnn import (
    build_full_loader,
    train_with_two_stage_split,
    evaluate_overall_mse,
    grid_search,
)


CONFIG: Dict[str, Any] = {
    # One of: "qubits", "depth", "depth_reverse", "per_qubit"
    "experiment": "depth",

    # Data dirs
    "random_base_dir": "/data/P70087789/GNN/data/dataset_random",
    "tim_base_dir": "/data/P70087789/GNN/data/dataset_tim",
    # Noisy data dirs
    "random_noisy_dir": "/data/P70087789/GNN/data/dataset_random_noisy",
    "tim_noisy_dir": "/data/P70087789/GNN/data/dataset_tim_noisy",
    "results_dir": "/data/P70087789/GNN/models/results",
    # If True, use noisy datasets on disk; graph features stay controlled by node_feature_backend_variant
    "use_noisy_datasets": True,

    # Features/architecture/training
    "global_feature_variant": "binned152",  
    # Node feature backend augmentation: None (noiseless) or a backend tag like "fake_oslo"
    "node_feature_backend_variant":"fake_oslo",
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
    "do_grid_search": True,
    "grid_epochs": 25,
    "model_kwargs": {
        "gnn_hidden": 32,
        "gnn_heads": 2,
        "global_hidden": 64,
        "reg_hidden": 128,
    },

    # Experiment-specific
    "tim_trotter": [1, 2, 3, 4, 5],
    "random_gates": None,  # e.g. [50, 100]
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
# Hyperparameter grid (small)
# =========================================
def small_gnn_grid_configs() -> List[Dict[str, int]]:
    configs: List[Dict[str, int]] = []
    for gnn_hidden in [32, 64]:
        for gnn_heads in [2, 4]:
            for global_hidden in [64]:
                for reg_hidden in [128, 256]:
                    configs.append({
                        "gnn_hidden": int(gnn_hidden),
                        "gnn_heads": int(gnn_heads),
                        "global_hidden": int(global_hidden),
                        "reg_hidden": int(reg_hidden),
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
def run_extrapolation_qubits(
    random_base_dir: str,
    tim_base_dir: str,
    random_noisy_dir: Optional[str] = None,
    tim_noisy_dir: Optional[str] = None,
    tim_trotter: Optional[List[int]] = None,
    tim_max_trotter: Optional[int] = 5,
    random_gates: Optional[List[int]] = None,
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
    do_grid_search: bool = True,
    grid_epochs: int = 25,
):
    in_qubits = [2, 3, 4, 5]
    ex_qubits = [6]
    results_dir = results_dir or "/data/P70087789/GNN/models/results"
    os.makedirs(results_dir, exist_ok=True)

    if tim_trotter is None:
        tim_trotter = [1, 2, 3, 4, 5]
        tim_max_trotter = None

    base_dir_random = random_noisy_dir if use_noisy_datasets else random_base_dir
    base_dir_tim = tim_noisy_dir if use_noisy_datasets else tim_base_dir
    random_in = select_random_pkls(base_dir_random, qubits=in_qubits, gates=random_gates)
    random_ex = select_random_pkls(base_dir_random, qubits=ex_qubits, gates=random_gates)

    # Small grid search on in-domain Random dataset to pick model hyperparameters
    chosen_model_kwargs: Dict[str, Any] = dict(model_kwargs or {})
    if do_grid_search:
        configs = small_gnn_grid_configs()
        grid_res = grid_search(
            pkl_paths=random_in,
            configs=configs,
            train_split=train_split,
            epochs=int(grid_epochs),
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
        )
        if len(grid_res) > 0:
            best = grid_res[0]
            chosen_model_kwargs.update({
                "gnn_hidden": int(best.get("gnn_hidden", chosen_model_kwargs.get("gnn_hidden", 32))),
                "gnn_heads": int(best.get("gnn_heads", chosen_model_kwargs.get("gnn_heads", 2))),
                "global_hidden": int(best.get("global_hidden", chosen_model_kwargs.get("global_hidden", 64))),
                "reg_hidden": int(best.get("reg_hidden", chosen_model_kwargs.get("reg_hidden", 128))),
            })
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

    tim_in = select_tim_pkls(base_dir_tim, qubits=in_qubits, trotter=tim_trotter, max_trotter=None)
    tim_ex = select_tim_pkls(base_dir_tim, qubits=ex_qubits, trotter=tim_trotter, max_trotter=None)
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
    tag = 'qubits_noisy' if use_noisy_datasets else 'qubits'
    save_gnn_results_for_experiment(results_dir, tag, out)
    return out


def run_extrapolation_depth(
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
    do_grid_search: bool = True,
    grid_epochs: int = 25,
):
    in_qubits = [2, 3, 4, 5, 6]
    results_dir = results_dir or "/data/P70087789/GNN/models/results"
    os.makedirs(results_dir, exist_ok=True)

    base_dir_random = random_noisy_dir if use_noisy_datasets else random_base_dir
    base_dir_tim = tim_noisy_dir if use_noisy_datasets else tim_base_dir

    is_noisy = bool(use_noisy_datasets)

    # Random dataset selection
    if is_noisy:
        # Noisy dataset only has gates 0-19 and 20-39
        try:
            random_in = select_random_pkls(base_dir_random, qubits=in_qubits, gates_range=(0, 19))
            random_ex = select_random_pkls(base_dir_random, qubits=in_qubits, gates_range=(20, 39))
        except FileNotFoundError:
            print("[WARN] Random noisy dataset missing required gate ranges; falling back to noiseless Random dataset.")
            random_in = select_random_pkls(random_base_dir, qubits=in_qubits, gates_range=(0, 19))
            random_ex = select_random_pkls(random_base_dir, qubits=in_qubits, gates_range=(20, 39))
    else:
        # Noiseless dataset has full ranges up to 99
        random_in = select_random_pkls(base_dir_random, qubits=in_qubits, gates_range=(1, 79))
        random_ex = select_random_pkls(base_dir_random, qubits=in_qubits, gates_range=(80, None))
    # Small grid search on in-domain Random dataset to pick model hyperparameters
    chosen_model_kwargs: Dict[str, Any] = dict(model_kwargs or {})
    if do_grid_search:
        configs = small_gnn_grid_configs()
        grid_res = grid_search(
            pkl_paths=random_in,
            configs=configs,
            train_split=train_split,
            epochs=int(grid_epochs),
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
        )
        if len(grid_res) > 0:
            best = grid_res[0]
            chosen_model_kwargs.update({
                "gnn_hidden": int(best.get("gnn_hidden", chosen_model_kwargs.get("gnn_hidden", 32))),
                "gnn_heads": int(best.get("gnn_heads", chosen_model_kwargs.get("gnn_heads", 2))),
                "global_hidden": int(best.get("global_hidden", chosen_model_kwargs.get("global_hidden", 64))),
                "reg_hidden": int(best.get("reg_hidden", chosen_model_kwargs.get("reg_hidden", 128))),
            })

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

    # TIM dataset selection
    if is_noisy:
        # Noisy TIM dataset only has trotter steps 1-3
        tim_in = select_tim_pkls(base_dir_tim, qubits=in_qubits, trotter=[1, 2], max_trotter=None)
        tim_ex = select_tim_pkls(base_dir_tim, qubits=in_qubits, trotter=[3], max_trotter=None)
    else:
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
        model_kwargs=chosen_model_kwargs or model_kwargs,
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
    tag = 'depth_noisy' if use_noisy_datasets else 'depth'
    save_gnn_results_for_experiment(results_dir, tag, out)
    return out


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
    cfg = CONFIG
    import sys
    if len(sys.argv) > 1:
        cfg["experiment"] = sys.argv[1]
    exp = cfg["experiment"]
    set_global_seed(int(cfg.get("seed", 42)))
    common = dict(
        random_base_dir=cfg["random_base_dir"],
        tim_base_dir=cfg["tim_base_dir"],
        random_noisy_dir=cfg["random_noisy_dir"],
        tim_noisy_dir=cfg["tim_noisy_dir"],
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

    if exp == "qubits":
        results = run_extrapolation_qubits(
            tim_trotter=cfg.get("tim_trotter"),
            tim_max_trotter=None,
            random_gates=cfg.get("random_gates"),
            **common,
        )
    elif exp == "depth":
        results = run_extrapolation_depth(**common)
    elif exp == "depth_reverse":
        results = run_reverse_extrapolation_depth(**common)
    elif exp == "per_qubit":
        results = run_per_qubit_experiments(**common)
    else:
        raise ValueError(f"Unknown experiment: {exp}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


