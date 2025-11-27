import argparse
import glob
import os
import pickle
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATA_ROOT = "/data/P70087789/GNN/data/dataset_classification/dataset_type"


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _maybe_float(x: Any) -> Optional[float]:
    try:
        if isinstance(x, (int, float)):
            v = float(x)
            if v == v:
                return v
    except Exception:
        pass
    return None


def extract_sre_total(entry: Any) -> Optional[float]:
    if isinstance(entry, dict):
        for key in ("sre_total", "total_sre", "sre", "sre_value", "state_renyi", "staterenyi"):
            if key in entry:
                v = _maybe_float(entry.get(key))
                if v is not None:
                    return v
        per = entry.get("sre_per_qubit")
        if isinstance(per, (list, tuple)):
            try:
                return float(sum(float(x) for x in per))
            except Exception:
                pass
        inner = entry.get("info") if isinstance(entry, dict) else None
        if isinstance(inner, dict):
            for key in ("sre_total", "sre", "sre_value"):
                if key in inner:
                    v = _maybe_float(inner.get(key))
                    if v is not None:
                        return v
        return None
    if isinstance(entry, (list, tuple)):
        if len(entry) == 2 and isinstance(entry[1], dict):
            return extract_sre_total(entry[1])
        for item in entry:
            v = _maybe_float(item)
            if v is not None:
                return v
        return None
    return _maybe_float(entry)


def coerce_sre_values_from_stats(obj: Any) -> List[float]:
    if isinstance(obj, dict):
        out: List[float] = []
        for v in obj.values():
            ev = extract_sre_total(v)
            if ev is not None:
                out.append(float(ev))
        return out
    if isinstance(obj, (list, tuple)):
        vals: List[float] = []
        for v in obj:
            if isinstance(v, (int, float)):
                mv = _maybe_float(v)
                if mv is not None:
                    vals.append(float(mv))
            else:
                ev = extract_sre_total(v)
                if ev is not None:
                    vals.append(float(ev))
        return vals
    ev = extract_sre_total(obj)
    return [float(ev)] if ev is not None else []


def safe_stats(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None, None
    try:
        mean_v = float(statistics.mean(values))
    except Exception:
        mean_v = None
    try:
        median_v = float(statistics.median(values))
    except Exception:
        median_v = None
    try:
        min_v = float(min(values))
    except Exception:
        min_v = None
    try:
        max_v = float(max(values))
    except Exception:
        max_v = None
    return mean_v, median_v, min_v, max_v


def summarize_sre_stats_files(root: str) -> None:
    pattern = os.path.join(root, "**", "*sre_stats.pkl")
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(files)} '*sre_stats.pkl' files under: {root}")
    all_values: List[float] = []
    for path in files:
        try:
            obj = load_pickle(path)
            vals = coerce_sre_values_from_stats(obj)
            mean_v, median_v, min_v, max_v = safe_stats(vals)
            print(f"- {os.path.basename(path)}: count={len(vals)}, mean={mean_v}, median={median_v}, min={min_v}, max={max_v}")
            all_values.extend(vals)
        except Exception as e:
            print(f"- {os.path.basename(path)}: ERROR reading/parsing -> {e}")
    overall_mean, overall_median, overall_min, overall_max = safe_stats(all_values)
    print(f"Overall across all '*sre_stats.pkl': count={len(all_values)}, mean={overall_mean}, median={overall_median}, min={overall_min}, max={overall_max}")


def count_labels_in_dataset(seq: Iterable[Any]) -> Tuple[int, int]:
    zero = 0
    one = 0
    for item in seq:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                label = int(item[1])
                if label == 0:
                    zero += 1
                elif label == 1:
                    one += 1
        except Exception:
            continue
    return zero, one


def summarize_balanced_files(root: str) -> None:
    bal_pattern = os.path.join(root, "**", "*balanced_by_sre.pkl")
    bal_stat_pattern = os.path.join(root, "**", "*balanced_by_sre_stats.pkl")
    bal_files = sorted(glob.glob(bal_pattern, recursive=True))
    bal_stat_files = sorted(glob.glob(bal_stat_pattern, recursive=True))

    print(f"Found {len(bal_files)} '*balanced_by_sre.pkl' files under: {root}")
    for path in bal_files:
        try:
            obj = load_pickle(path)
            size = len(obj) if hasattr(obj, "__len__") else None
            z, o = count_labels_in_dataset(obj if isinstance(obj, Iterable) else [])
            ratio = (z / float(o)) if o > 0 else None
            print(f"- {os.path.basename(path)}: type={type(obj).__name__}, size={size}, labels 0={z}, 1={o}, ratio 0/1={ratio}")
        except Exception as e:
            print(f"- {os.path.basename(path)}: ERROR reading -> {e}")

    print(f"Found {len(bal_stat_files)} '*balanced_by_sre_stats.pkl' files under: {root}")
    for path in bal_stat_files:
        try:
            obj = load_pickle(path)
            vals = coerce_sre_values_from_stats(obj)
            mean_v, median_v, min_v, max_v = safe_stats(vals)
            print(f"- {os.path.basename(path)}: type={type(obj).__name__}, count={len(vals)}, mean={mean_v}, median={median_v}, min={min_v}, max={max_v}")
        except Exception as e:
            print(f"- {os.path.basename(path)}: ERROR reading -> {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SRE stats and balanced files in dataset_type directory.")
    parser.add_argument("--root", type=str, default=DATA_ROOT, help="Root directory to scan")
    args = parser.parse_args()

    root = args.root
    print(f"Scanning directory: {root}")
    summarize_sre_stats_files(root)
    summarize_balanced_files(root)


if __name__ == "__main__":
    main()

