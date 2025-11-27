import os
import pickle
from typing import Any, List, Optional, Sequence, Tuple, Dict

# Input paths (absolute as requested)
BASE_DATASET_PATH = \
    "/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_25.pkl"
SRE_STATS_PATH = \
    "/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_25_sre_stats.pkl"
BALANCED_DATASET_PATH = \
    "/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_25_balanced_by_sre.pkl"
# Output path
OUTPUT_SRE_VALUES_PATH = \
    "/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_25_balanced_by_sre_stats.pkl"


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


def main() -> None:
    print("Loading datasets and SRE stats...")
    base_data: Sequence[Tuple[Dict[str, Any], int]] = load_pickle(BASE_DATASET_PATH)
    balanced_data: Sequence[Tuple[Dict[str, Any], int]] = load_pickle(BALANCED_DATASET_PATH)
    sre_stats = load_pickle(SRE_STATS_PATH)
    base_sre: List[Optional[float]] = build_base_index_to_sre(sre_stats)

    # Identify original indices corresponding to label==1 in the base dataset,
    # keeping only those with available SRE values. The balanced dataset was
    # produced by removing label==0 items and relabeling the remaining ones;
    # thus, order should match this filtered sequence.
    filtered_indices: List[int] = []
    for idx, item in enumerate(base_data):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        label = int(item[1])
        if label != 1:
            continue
        v = base_sre[idx] if idx < len(base_sre) else None
        if v is None:
            continue
        filtered_indices.append(idx)

    if len(filtered_indices) != len(balanced_data):
        raise RuntimeError(
            f"Mismatch between expected balanced items ({len(filtered_indices)}) and actual balanced items ({len(balanced_data)})."
        )

    balanced_sre_values: List[float] = []
    for idx in filtered_indices:
        v = base_sre[idx]
        if v is None:
            raise RuntimeError("Encountered None SRE where it should be present.")
        balanced_sre_values.append(float(v))

    print(f"Extracted {len(balanced_sre_values)} SRE values. Saving to: {OUTPUT_SRE_VALUES_PATH}")
    save_pickle(balanced_sre_values, OUTPUT_SRE_VALUES_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
