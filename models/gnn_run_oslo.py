import os
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse

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
)


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


def select_random_pkls(base_dir: str, qubits: List[int], gates: Optional[List[int]] = None,
                       gates_range: Optional[Tuple[Optional[int], Optional[int]]] = None) -> List[str]:
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
        raise FileNotFoundError(f"No Random PKLs matched in {base_dir} for qubits={qubits} gates={gates} range={gates_range}")
    return selected


def select_tim_pkls(base_dir: str, qubits: List[int], trotter: Optional[List[int]] = None,
                    max_trotter: Optional[int] = None) -> List[str]:
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


def main() -> None:
    cfg: Dict[str, Any] = {
        # Oslo noisy datasets
        "random_noisy_dir": "/data/P70087789/GNN/data/dataset_regression/dataset_random_oslo",
        "tim_noisy_dir": "/data/P70087789/GNN/data/dataset_regression/dataset_tim_oslo",
        # Provide base dirs too (not used when use_noisy_datasets=True)
        "random_base_dir": "/data/P70087789/GNN/data/dataset_regression/dataset_random_oslo",
        "tim_base_dir": "/data/P70087789/GNN/data/dataset_regression/dataset_tim_oslo",
        "results_dir": "/data/P70087789/GNN/models/results_oslo",
        # Use new Oslo feature variant + Oslo node backend augmentation
        "global_feature_variant": "oslo",
        "node_feature_backend_variant": "fake_oslo",
        # Training config
        "epochs": 200,
        "lr": 1e-3,
        "batch_size": 64,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.0,
        "train_split": 0.8,
        "val_within_train": 0.1,
        "device": None,
        "seed": 42,
        # Grid search can be enabled; keep modest to start
        "do_grid_search": True,
        "grid_epochs": 15,
        "loss_type": "huber",
        "model_kwargs": {
            "gnn_hidden": 32,
            "gnn_heads": 8,
            "global_hidden": 16,
            "reg_hidden": 16,
        },
        # Multiseed (default single seed; override with --num-seeds)
        "num_seeds": 10,
    }

    os.makedirs(cfg["results_dir"], exist_ok=True)

    # Force node embedding size to 20 = 7 (type) + 6 (mask) + 7 (backend)
    os.environ["GNN_QUBIT_MASK_DIM"] = "6"

    # Lightweight CLI to control seeds
    parser = argparse.ArgumentParser(description="Run Oslo noisy experiments (qubits and depth)")
    parser.add_argument('--num-seeds', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if args.num_seeds is not None:
        cfg["num_seeds"] = int(args.num_seeds)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    # 1) Extrapolation on qubits (train on qubits 2..5, test on 6)
    # Random: include available gates (0..39)
    # TIM: include available trotter steps (1..3)
    in_qubits = [2, 3, 4, 5]
    ex_qubits = [6]
    random_in = select_random_pkls(cfg["random_noisy_dir"], qubits=in_qubits, gates_range=(0, 39))
    random_ex = select_random_pkls(cfg["random_noisy_dir"], qubits=ex_qubits, gates_range=(0, 39))
    tim_in = select_tim_pkls(cfg["tim_noisy_dir"], qubits=in_qubits, trotter=[1, 2, 3])
    tim_ex = select_tim_pkls(cfg["tim_noisy_dir"], qubits=ex_qubits, trotter=[1, 2, 3])

    def _run_qubits_one(seed_val: int) -> Dict[str, Dict[str, float]]:
        # Random in-domain training and extrapolation evaluation on qubit 6 set
        model_r, train_r, val_r, test_r, dev_r = train_with_two_stage_split(
            pkl_paths=random_in,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            global_feature_variant=cfg["global_feature_variant"],
            node_feature_backend_variant=cfg["node_feature_backend_variant"],
            early_stopping_patience=cfg["early_stopping_patience"],
            early_stopping_min_delta=cfg["early_stopping_min_delta"],
            train_split=cfg["train_split"],
            val_within_train=cfg["val_within_train"],
            model_kwargs=cfg["model_kwargs"],
            loss_type=cfg["loss_type"],
            seed=seed_val,
        )
        train_mse_r = float(evaluate_overall_mse(model_r, train_r, dev_r))
        test_mse_r = float(evaluate_overall_mse(model_r, test_r, dev_r))
        train_r2_r = float(evaluate_overall_r2(model_r, train_r, dev_r))
        test_r2_r = float(evaluate_overall_r2(model_r, test_r, dev_r))
        extra_loader_r = build_full_loader(
            random_ex,
            batch_size=cfg["batch_size"],
            global_feature_variant=cfg["global_feature_variant"],
            node_feature_backend_variant=cfg["node_feature_backend_variant"],
        )
        extra_mse_r = float(evaluate_overall_mse(model_r, extra_loader_r, dev_r))
        extra_r2_r = float(evaluate_overall_r2(model_r, extra_loader_r, dev_r))

        model_t, train_t, val_t, test_t, dev_t = train_with_two_stage_split(
            pkl_paths=tim_in,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            global_feature_variant=cfg["global_feature_variant"],
            node_feature_backend_variant=cfg["node_feature_backend_variant"],
            early_stopping_patience=cfg["early_stopping_patience"],
            early_stopping_min_delta=cfg["early_stopping_min_delta"],
            train_split=cfg["train_split"],
            val_within_train=cfg["val_within_train"],
            model_kwargs=cfg["model_kwargs"],
            loss_type=cfg["loss_type"],
            seed=seed_val,
        )
        train_mse_t = float(evaluate_overall_mse(model_t, train_t, dev_t))
        test_mse_t = float(evaluate_overall_mse(model_t, test_t, dev_t))
        train_r2_t = float(evaluate_overall_r2(model_t, train_t, dev_t))
        test_r2_t = float(evaluate_overall_r2(model_t, test_t, dev_t))
        extra_loader_t = build_full_loader(
            tim_ex,
            batch_size=cfg["batch_size"],
            global_feature_variant=cfg["global_feature_variant"],
            node_feature_backend_variant=cfg["node_feature_backend_variant"],
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

    seeds = [int(cfg["seed"]) + i for i in range(int(cfg["num_seeds"]))]
    runs_q = [_run_qubits_one(s) for s in seeds]

    def _avg(items: List[float]) -> float:
        return float(sum(items) / max(1, len(items)))

    def _agg(domain: str) -> Dict[str, float]:
        metrics = ['train_mse', 'test_mse', 'extra_mse', 'train_r2', 'test_r2', 'extra_r2']
        return {m: _avg([r[domain][m] for r in runs_q]) for m in metrics}

    results_qubits = {
        'random': _agg('random'),
        'tim': _agg('tim'),
        'seeds': seeds,
        'num_seeds': int(len(seeds)),
        'global_feature_variant': str(cfg["global_feature_variant"]),
        'node_feature_backend_variant': str(cfg["node_feature_backend_variant"]),
    }

    # 2) Extrapolation on depth for Oslo datasets
    #    - Random: train gates 0..19, extrapolate 20..39
    #    - TIM: train trotter 1..4, extrapolate 5
    # 2) Extrapolation on depth for Oslo datasets
    # Random: train gates 0..19, extrapolate 20..39
    # TIM: train trotter 1..4, extrapolate 5
    random_in_d = select_random_pkls(cfg["random_noisy_dir"], qubits=[2, 3, 4, 5, 6], gates_range=(0, 19))
    random_ex_d = select_random_pkls(cfg["random_noisy_dir"], qubits=[2, 3, 4, 5, 6], gates_range=(20, 39))
    tim_in_d = select_tim_pkls(cfg["tim_noisy_dir"], qubits=[2, 3, 4, 5, 6], trotter=[1, 2, 3, 4])
    tim_ex_d = select_tim_pkls(cfg["tim_noisy_dir"], qubits=[2, 3, 4, 5, 6], trotter=[5])

    def _run_depth_one(seed_val: int) -> Dict[str, Dict[str, float]]:
        model_r, train_r, val_r, test_r, dev_r = train_with_two_stage_split(
            pkl_paths=random_in_d,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            global_feature_variant=cfg["global_feature_variant"],
            node_feature_backend_variant=cfg["node_feature_backend_variant"],
            early_stopping_patience=cfg["early_stopping_patience"],
            early_stopping_min_delta=cfg["early_stopping_min_delta"],
            train_split=cfg["train_split"],
            val_within_train=cfg["val_within_train"],
            model_kwargs=cfg["model_kwargs"],
            loss_type=cfg["loss_type"],
            seed=seed_val,
        )
        train_mse_r = float(evaluate_overall_mse(model_r, train_r, dev_r))
        test_mse_r = float(evaluate_overall_mse(model_r, test_r, dev_r))
        train_r2_r = float(evaluate_overall_r2(model_r, train_r, dev_r))
        test_r2_r = float(evaluate_overall_r2(model_r, test_r, dev_r))
        extra_loader_r = build_full_loader(
            random_ex_d,
            batch_size=cfg["batch_size"],
            global_feature_variant=cfg["global_feature_variant"],
            node_feature_backend_variant=cfg["node_feature_backend_variant"],
        )
        extra_mse_r = float(evaluate_overall_mse(model_r, extra_loader_r, dev_r))
        extra_r2_r = float(evaluate_overall_r2(model_r, extra_loader_r, dev_r))

        model_t, train_t, val_t, test_t, dev_t = train_with_two_stage_split(
            pkl_paths=tim_in_d,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            global_feature_variant=cfg["global_feature_variant"],
            node_feature_backend_variant=cfg["node_feature_backend_variant"],
            early_stopping_patience=cfg["early_stopping_patience"],
            early_stopping_min_delta=cfg["early_stopping_min_delta"],
            train_split=cfg["train_split"],
            val_within_train=cfg["val_within_train"],
            model_kwargs=cfg["model_kwargs"],
            loss_type=cfg["loss_type"],
            seed=seed_val,
        )
        train_mse_t = float(evaluate_overall_mse(model_t, train_t, dev_t))
        test_mse_t = float(evaluate_overall_mse(model_t, test_t, dev_t))
        train_r2_t = float(evaluate_overall_r2(model_t, train_t, dev_t))
        test_r2_t = float(evaluate_overall_r2(model_t, test_t, dev_t))
        extra_loader_t = build_full_loader(
            tim_ex_d,
            batch_size=cfg["batch_size"],
            global_feature_variant=cfg["global_feature_variant"],
            node_feature_backend_variant=cfg["node_feature_backend_variant"],
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

    runs_d = [_run_depth_one(s) for s in seeds]

    def _agg_d(domain: str) -> Dict[str, float]:
        metrics = ['train_mse', 'test_mse', 'extra_mse', 'train_r2', 'test_r2', 'extra_r2']
        return {m: _avg([r[domain][m] for r in runs_d]) for m in metrics}

    results_depth = {
        'random': _agg_d('random'),
        'tim': _agg_d('tim'),
        'seeds': seeds,
        'num_seeds': int(len(seeds)),
        'global_feature_variant': str(cfg["global_feature_variant"]),
        'node_feature_backend_variant': str(cfg["node_feature_backend_variant"]),
    }

    combined = {
        "qubits": results_qubits,
        "depth": results_depth,
        "config": {
            "seed": int(cfg["seed"]),
            "num_seeds": int(cfg["num_seeds"]),
            "global_feature_variant": str(cfg["global_feature_variant"]),
            "node_feature_backend_variant": str(cfg["node_feature_backend_variant"]),
            "model_kwargs": dict(cfg["model_kwargs"]),
        },
    }

    # Save to PKL
    out_dir = cfg["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gnn_oslo_results.pkl")
    import pickle
    with open(out_path, 'wb') as f:
        pickle.dump(combined, f)

    print(json.dumps({"saved_to": out_path, "summary": {"qubits": results_qubits, "depth": results_depth}}, indent=2))


if __name__ == "__main__":
    main()


