import os
from pathlib import Path
from typing import Any, Dict, Optional


SVR_RESULTS_FILENAME = "svr_results.pkl"


def get_svr_results_path(results_dir: str) -> str:
    return os.path.join(results_dir, SVR_RESULTS_FILENAME)

def get_svr_results_path_with_name(results_dir: str, filename: str) -> str:
    return os.path.join(results_dir, filename)

def load_all_svr_results(results_dir: str) -> Optional[Dict[str, Any]]:
    path = get_svr_results_path(results_dir)
    if not os.path.exists(path):
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def load_svr_results(results_dir: str, filename: str) -> Optional[Dict[str, Any]]:
    path = get_svr_results_path_with_name(results_dir, filename)
    if not os.path.exists(path):
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def save_svr_results_for_experiment(results_dir: str, experiment: str, results: Any) -> str:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    existing = load_all_svr_results(results_dir) or {}
    # Overwrite noisy entries if re-run
    existing[str(experiment)] = results
    path = get_svr_results_path(results_dir)
    import pickle
    with open(path, "wb") as f:
        pickle.dump(existing, f)
    return path

def save_svr_results_for_experiment_with_name(results_dir: str, filename: str, experiment: str, results: Any) -> str:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    existing = load_svr_results(results_dir, filename) or {}
    existing[str(experiment)] = results
    path = get_svr_results_path_with_name(results_dir, filename)
    import pickle
    with open(path, "wb") as f:
        pickle.dump(existing, f)
    return path


