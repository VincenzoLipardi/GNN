import os
import pickle
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple


BASE_DATASET_PATH = \
    "/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_11_25.pkl"
SRE_STATS_PATH = \
    "/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_11_25_sre_stats.pkl"
EVOLVED_DATASET_PATH = \
    "/data/P70087789/GNN/data/dataset_classification/dataset_type/clifford_evolved_11_25.pkl"

# Global threshold source (median of SRE values from 2-10 balanced stats)
GLOBAL_THRESHOLD_STATS_PATH = \
    "/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_10_balanced_by_sre_stats.pkl"

OUTPUT_BASE_BALANCED_PATH = \
    "/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_11_25_balanced_by_sre.pkl"
OUTPUT_EVOLVED_BALANCED_PATH = \
    "/data/P70087789/GNN/data/dataset_classification/dataset_type/clifford_evolved_11_25_balanced_by_sre.pkl"


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _maybe_float(x: Any) -> Optional[float]:
    try:
        if isinstance(x, (int, float)):
            v = float(x)
            if v == v:  # not NaN
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


def build_base_index_to_sre(sre_stats_obj: Any) -> List[Optional[float]]:
    if isinstance(sre_stats_obj, dict):
        max_idx = max((int(k) for k in sre_stats_obj.keys()), default=-1)
        out: List[Optional[float]] = [None] * (max_idx + 1)
        for k, val in sre_stats_obj.items():
            try:
                idx = int(k)
            except Exception:
                continue
            out[idx] = extract_sre_total(val)
        return out
    if isinstance(sre_stats_obj, (list, tuple)):
        return [extract_sre_total(v) for v in sre_stats_obj]
    raise ValueError("Unrecognized SRE stats format; expected dict or list/tuple")


def _count_labels(dataset: Sequence[Tuple[Dict[str, Any], int]]) -> Tuple[int, int]:
    count_zero = 0
    count_one = 0
    for item in dataset:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            label = int(item[1])
        except Exception:
            continue
        if label == 0:
            count_zero += 1
        elif label == 1:
            count_one += 1
    return count_zero, count_one


def relabel_balanced_by_sre_for_product(
    dataset_path: str,
    sre_stats_path: str,
    override_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[Dict[str, Any], int]], float]:
    data: Sequence[Tuple[Dict[str, Any], int]] = load_pickle(dataset_path)
    sre_stats = load_pickle(sre_stats_path)
    base_sre: List[Optional[float]] = build_base_index_to_sre(sre_stats)

    filtered_indices: List[int] = []
    sre_values: List[float] = []
    for idx, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        label = int(item[1])
        if label != 1:
            continue
        if idx < len(base_sre):
            v = base_sre[idx]
        else:
            v = None
        if v is None:
            continue
        filtered_indices.append(idx)
        sre_values.append(float(v))

    if not sre_values:
        raise RuntimeError("No SRE values found for label=1 items in base dataset.")

    threshold = float(override_threshold) if override_threshold is not None else float(statistics.median(sre_values))

    balanced: List[Tuple[Dict[str, Any], int]] = []
    for idx in filtered_indices:
        info, _ = data[idx]
        v = base_sre[idx]
        new_label = 0 if (v is not None and float(v) < threshold) else 1
        balanced.append((info, int(new_label)))

    return balanced, threshold


def deduce_variants_per_base(evolved_path: str, base_len: int) -> int:
    evolved = load_pickle(evolved_path)
    elen = len(evolved) if hasattr(evolved, "__len__") else 0
    if base_len <= 0 or elen <= 0:
        raise RuntimeError("Cannot deduce variants per base; invalid sizes.")
    ratio = elen / float(base_len)
    r_int = int(round(ratio))
    if r_int <= 0:
        raise RuntimeError("Invalid variants per base computed.")
    return r_int


def relabel_balanced_by_sre_for_evolved(
    evolved_path: str,
    base_dataset_path: str,
    base_sre: List[Optional[float]],
    override_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[Dict[str, Any], int]], float]:
    base_data: Sequence[Tuple[Dict[str, Any], int]] = load_pickle(base_dataset_path)
    evolved: Sequence[Tuple[Dict[str, Any], int]] = load_pickle(evolved_path)
    variants = deduce_variants_per_base(evolved_path, len(base_data))

    filtered_indices: List[int] = []
    sre_values: List[float] = []
    for idx, item in enumerate(evolved):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        label = int(item[1])
        if label != 1:
            continue
        base_idx = idx // int(variants)
        v = base_sre[base_idx] if 0 <= base_idx < len(base_sre) else None
        if v is None:
            continue
        filtered_indices.append(idx)
        sre_values.append(float(v))

    if not sre_values:
        raise RuntimeError("No SRE values found for label=1 items in evolved dataset.")

    threshold = float(override_threshold) if override_threshold is not None else float(statistics.median(sre_values))

    balanced: List[Tuple[Dict[str, Any], int]] = []
    for idx, item in enumerate(evolved):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        label = int(item[1])
        if label != 1:
            continue
        base_idx = idx // int(variants)
        v = base_sre[base_idx] if 0 <= base_idx < len(base_sre) else None
        if v is None:
            continue
        info, _ = item
        new_label = 0 if float(v) < threshold else 1
        balanced.append((info, int(new_label)))

    return balanced, threshold


def main() -> None:
    # Print current ratio in the existing balanced file (if present)
    if os.path.exists(OUTPUT_BASE_BALANCED_PATH):
        try:
            current_balanced = load_pickle(OUTPUT_BASE_BALANCED_PATH)
            c0, c1 = _count_labels(current_balanced)
            ratio_str = (f"{c0/float(c1):.6f}" if c1 > 0 else "inf")
            print(
                f"Current 11-25 balanced file label counts -> 0: {c0}, 1: {c1}, ratio 0/1: {ratio_str}"
            )
        except Exception as e:
            print(f"Warning: failed to read existing balanced file: {e}")
    else:
        print("No existing 11-25 balanced file found; skipping current ratio print.")

    # Compute global median threshold from 2-10 balanced SRE stats
    print("Loading global SRE stats from 2-10 balanced file for median threshold...")
    global_stats_obj = load_pickle(GLOBAL_THRESHOLD_STATS_PATH)

    def _coerce_global_values(obj: Any) -> List[float]:
        if isinstance(obj, dict):
            vals: List[float] = []
            for v in obj.values():
                ev = extract_sre_total(v)
                if ev is not None:
                    vals.append(float(ev))
            return vals
        if isinstance(obj, (list, tuple)):
            vals = []
            for v in obj:
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                else:
                    ev = extract_sre_total(v)
                    if ev is not None:
                        vals.append(float(ev))
            return vals
        ev = extract_sre_total(obj)
        return [float(ev)] if ev is not None else []

    global_values = _coerce_global_values(global_stats_obj)
    if not global_values:
        raise RuntimeError("No valid SRE values found in global threshold stats file.")
    global_threshold = float(statistics.median(global_values))
    print(f"Using global median SRE threshold (from 2-10): {global_threshold:.6f}")

    print("Relabeling base 11-25 dataset with global threshold...")
    base_balanced, base_threshold = relabel_balanced_by_sre_for_product(
        BASE_DATASET_PATH, SRE_STATS_PATH, override_threshold=global_threshold
    )
    print(f"Saving balanced base dataset to: {OUTPUT_BASE_BALANCED_PATH}")
    save_pickle(base_balanced, OUTPUT_BASE_BALANCED_PATH)
    print(f"Balanced base dataset size: {len(base_balanced)}")
    # Print new ratio after overwrite
    nb0, nb1 = _count_labels(base_balanced)
    new_ratio_str = (f"{nb0/float(nb1):.6f}" if nb1 > 0 else "inf")
    print(
        f"New 11-25 balanced file label counts -> 0: {nb0}, 1: {nb1}, ratio 0/1: {new_ratio_str}"
    )

    print("Relabeling evolved 11-25 dataset with global threshold...")
    sre_stats = load_pickle(SRE_STATS_PATH)
    base_sre = build_base_index_to_sre(sre_stats)
    evolved_balanced, evolved_threshold = relabel_balanced_by_sre_for_evolved(
        EVOLVED_DATASET_PATH, BASE_DATASET_PATH, base_sre, override_threshold=global_threshold
    )
    print(f"Saving balanced evolved dataset to: {OUTPUT_EVOLVED_BALANCED_PATH}")
    save_pickle(evolved_balanced, OUTPUT_EVOLVED_BALANCED_PATH)
    print(f"Balanced evolved dataset size: {len(evolved_balanced)}")


if __name__ == "__main__":
    main()


