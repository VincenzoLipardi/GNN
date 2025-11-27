import argparse
import json
import math
import os
import pickle
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BIN_COUNT = 30


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_num_qubits_from_path(path: Optional[str], default_value: int = 18) -> int:
    if not path:
        return default_value
    m = re.search(r"(?:^|[_-])(\d+)(?:\.pkl)?$", os.path.basename(path))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return default_value


def find_misclassified_indices(payload: Dict[str, Any]) -> List[int]:
    if "misclassified_indices" in payload and isinstance(payload["misclassified_indices"], list):
        return [int(x) for x in payload["misclassified_indices"]]
    if "misclassified" in payload:
        val = payload["misclassified"]
        if isinstance(val, list):
            return [int(x) for x in val]
        if isinstance(val, dict):
            return [int(k) for k in val.keys()]
    if "indices_test_order" in payload and isinstance(payload["indices_test_order"], list):
        return [int(x) for x in payload["indices_test_order"]]
    if "indices" in payload and isinstance(payload["indices"], list):
        return [int(x) for x in payload["indices"]]
    raise ValueError("Could not locate misclassified indices in the JSON payload.")


def try_extract_sre_and_qubits(entry: Any, default_qubits: int) -> Tuple[Optional[float], int]:
    sre_value: Optional[float] = None
    num_qubits: int = default_qubits

    def _maybe_float(x: Any) -> Optional[float]:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return float(x)
        return None

    if isinstance(entry, dict):
        for key in entry.keys():
            kl = str(key).lower()
            if kl in ("sre", "sre_value", "staterenyi", "state_renyi"):
                v = _maybe_float(entry[key])
                if v is not None:
                    sre_value = v
            if kl in ("num_qubits", "n_qubits", "nqubits"):
                try:
                    num_qubits = int(entry[key])
                except Exception:
                    pass
        info = entry.get("info") if isinstance(entry, dict) else None
        if isinstance(info, dict):
            if "num_qubits" in info:
                try:
                    num_qubits = int(info["num_qubits"])
                except Exception:
                    pass
            for key in ("sre", "sre_value"):
                if key in info:
                    v = _maybe_float(info[key])
                    if v is not None:
                        sre_value = v
    elif isinstance(entry, (list, tuple)):
        if len(entry) == 2 and all(isinstance(x, dict) for x in entry):
            info_dict, sre_dict = entry[0], entry[1]
            if isinstance(info_dict, dict):
                try:
                    num_qubits = int(info_dict.get("num_qubits", num_qubits))
                except Exception:
                    pass
            if isinstance(sre_dict, dict):
                v = _maybe_float(sre_dict.get("sre_total"))
                if v is None and isinstance(sre_dict.get("sre_per_qubit"), (list, tuple)):
                    try:
                        v = float(sum(float(x) for x in sre_dict["sre_per_qubit"]))
                    except Exception:
                        v = None
                if v is not None:
                    sre_value = v
        else:
            for item in entry:
                v = _maybe_float(item)
                if v is not None and sre_value is None:
                    sre_value = v
                if isinstance(item, dict):
                    if "num_qubits" in item:
                        try:
                            num_qubits = int(item["num_qubits"])
                        except Exception:
                            pass
    else:
        v = _maybe_float(entry)
        if v is not None:
            sre_value = v

    return sre_value, num_qubits


def build_base_index_to_density(sre_stats_obj: Any, fallback_qubits: int) -> List[Optional[float]]:
    densities: List[Optional[float]] = []
    if isinstance(sre_stats_obj, dict):
        max_idx = max(int(k) for k in sre_stats_obj.keys()) if sre_stats_obj else -1
        densities = [None] * (max_idx + 1)
        for k, entry in sre_stats_obj.items():
            idx = int(k)
            sre, nq = try_extract_sre_and_qubits(entry, fallback_qubits)
            if sre is not None and nq > 0:
                densities[idx] = sre / float(nq)
        return densities
    if isinstance(sre_stats_obj, (list, tuple)):
        for entry in sre_stats_obj:
            sre, nq = try_extract_sre_and_qubits(entry, fallback_qubits)
            if sre is not None and nq > 0:
                densities.append(sre / float(nq))
            else:
                densities.append(None)
        return densities
    raise ValueError("Unrecognized structure for SRE stats PKL; expected list/tuple/dict")


def compute_variants_per_base(evolved_dataset_path: Optional[str], num_base_items: Optional[int]) -> Optional[int]:
    if not evolved_dataset_path or not os.path.exists(evolved_dataset_path) or num_base_items is None or num_base_items <= 0:
        return None
    try:
        evolved = load_pickle(evolved_dataset_path)
        evolved_len = len(evolved) if hasattr(evolved, "__len__") else None
        if evolved_len is None:
            return None
        ratio = evolved_len / float(num_base_items)
        if ratio.is_integer():
            return int(ratio)
        rounded = int(round(ratio))
        if rounded > 0:
            return rounded
    except Exception:
        return None
    return None


def build_base_index_to_total(sre_stats_obj: Any, fallback_qubits: int) -> List[Optional[float]]:
    totals: List[Optional[float]] = []
    if isinstance(sre_stats_obj, dict):
        max_idx = max(int(k) for k in sre_stats_obj.keys()) if sre_stats_obj else -1
        totals = [None] * (max_idx + 1)
        for k, entry in sre_stats_obj.items():
            idx = int(k)
            sre, _ = try_extract_sre_and_qubits(entry, fallback_qubits)
            if sre is not None and math.isfinite(float(sre)):
                totals[idx] = float(sre)
        return totals
    if isinstance(sre_stats_obj, (list, tuple)):
        for entry in sre_stats_obj:
            sre, _ = try_extract_sre_and_qubits(entry, fallback_qubits)
            if sre is not None and math.isfinite(float(sre)):
                totals.append(float(sre))
            else:
                totals.append(None)
        return totals
    raise ValueError("Unrecognized structure for SRE stats PKL; expected list/tuple/dict")


def generate_ratio_plot(misclassified_json_path: str, sre_stats_pkl_path: str, ratio_output_path: str, use_density: bool = True) -> None:
    payload = read_json(misclassified_json_path)
    indices = find_misclassified_indices(payload)
    evolved_dataset_path = payload.get("dataset") if isinstance(payload, dict) else None
    default_nq = infer_num_qubits_from_path(evolved_dataset_path, default_value=18)

    sre_stats = load_pickle(sre_stats_pkl_path)
    base_values = (
        build_base_index_to_density(sre_stats, fallback_qubits=default_nq)
        if use_density else
        build_base_index_to_total(sre_stats, fallback_qubits=default_nq)
    )
    num_base = len(base_values)

    variants_per_base = compute_variants_per_base(evolved_dataset_path, num_base)
    if variants_per_base is None:
        variants_per_base = 10

    misclassified_values: List[float] = []
    for idx in indices:
        try:
            base_idx = int(idx) // int(variants_per_base)
        except Exception:
            continue
        if 0 <= base_idx < num_base:
            v = base_values[base_idx]
            if v is not None and math.isfinite(float(v)):
                misclassified_values.append(float(v))

    if not misclassified_values:
        raise RuntimeError("No SRE densities could be derived for the provided misclassified indices.")

    all_values_all_labels: List[float] = []
    for v in base_values:
        if v is not None and math.isfinite(float(v)):
            all_values_all_labels.append(float(v))

    data_min = min(min(misclassified_values), min(all_values_all_labels))
    data_max = max(max(misclassified_values), max(all_values_all_labels))
    bin_edges = np.linspace(data_min, data_max, BIN_COUNT + 1)

    mis_counts, _ = np.histogram(misclassified_values, bins=bin_edges)
    all_counts_base, _ = np.histogram(all_values_all_labels, bins=bin_edges)
    all_counts = all_counts_base * int(variants_per_base)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(mis_counts, all_counts)
        ratio[~np.isfinite(ratio)] = 0.0
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    plt.figure(figsize=(14, 8))
    plt.bar(bin_centers, ratio, width=bin_widths, color="indigo", edgecolor="black", alpha=0.85, align="center")
    if use_density:
        plt.xlabel(r"$m_2$", fontsize=24)
    else:
        plt.xlabel(r"$M_2$", fontsize=24)
    plt.ylabel("Misclassified / All", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(ratio_output_path), exist_ok=True)
    plt.savefig(ratio_output_path, dpi=300)
    plt.close()
    print(f"Saved misclassification ratio plot to: {ratio_output_path}")

    total_mis = int(mis_counts.sum())
    total_all = int(all_counts.sum())
    print(f"Ratio plot totals — Misclassified: {total_mis} | All: {total_all}")

    be_list = bin_edges.tolist()
    print("Per-bin counts (inclusive last bin):")
    count = 0
    for i in range(len(be_list) - 1):
        left = float(be_list[i])
        right = float(be_list[i + 1])
        mc = int(mis_counts[i])
        ac = int(all_counts[i])
        count += ac
        right_bracket = ']' if i == len(be_list) - 2 else ')'
        print(f"  Bin {i}: [{left:.6f}, {right:.6f}{right_bracket}  mis={mc}  all={ac}  count={count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot misclassification ratio vs SRE density (including both labels).")
    parser.add_argument(
        "--misclassified_json",
        default="/data/P70087789/GNN/data/dataset_classification/results/misclassified_product_states_2_10_balanced_by_sre_clifford_evolved_2_10_balanced_by_sre.json",
        help="Path to JSON with misclassified indices",
    )
    parser.add_argument(
        "--sre_stats_pkl",
        default="/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_10_balanced_by_sre_stats.pkl",
        help="Path to PKL containing SRE stats for base product circuits",
    )
    parser.add_argument(
        "--base_dataset_pkl",
        default="/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_10_balanced_by_sre.pkl",
        help="Path to PKL containing base dataset (for size/reference)",
    )
    parser.add_argument(
        "--ratio_scatter_output",
        default="/data/P70087789/GNN/data/dataset_classification/images/sre_density_2_10_balanced.png",
        help="Output image path for misclassification ratio bar chart",
    )
    parser.add_argument(
        "--ratio_scatter_output_m2",
        default="/data/P70087789/GNN/data/dataset_classification/images/sre_total_2_10_balanced.png",
        help="Output image path for misclassification ratio bar chart using M2 (total SRE)",
    )
    parser.add_argument(
        "--additional_misclassified_json",
        default="/data/P70087789/GNN/data/dataset_classification/results/misclassified_product_states_2_10_balanced_by_sre_product_states_11_25_balanced_by_sre.json",
        help="Optional: Path to a second JSON with misclassified indices to also plot",
    )
    parser.add_argument(
        "--additional_sre_stats_pkl",
        default="/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_11_25_balanced_by_sre_stats.pkl",
        help="Path to PKL containing SRE stats for the second dataset",
    )
    parser.add_argument(
        "--additional_ratio_output",
        default="/data/P70087789/GNN/data/dataset_classification/images/sre_density_11_25_balanced.png",
        help="Output image path for the second misclassification ratio bar chart",
    )
    parser.add_argument(
        "--additional_ratio_output_m2",
        default="/data/P70087789/GNN/data/dataset_classification/images/sre_total_11_25_balanced.png",
        help="Output image path for the second misclassification ratio bar chart using M2 (total SRE)",
    )
    args = parser.parse_args()

    # Primary plot (clifford-evolved by default)
    generate_ratio_plot(
        misclassified_json_path=args.misclassified_json,
        sre_stats_pkl_path=args.sre_stats_pkl,
        ratio_output_path=args.ratio_scatter_output,
        use_density=True,
    )
    # Additional plot for M2 (total SRE)
    generate_ratio_plot(
        misclassified_json_path=args.misclassified_json,
        sre_stats_pkl_path=args.sre_stats_pkl,
        ratio_output_path=args.ratio_scatter_output_m2,
        use_density=False,
    )

    # Optional second plot for the additional misclassified dataset
    if args.additional_misclassified_json:
        generate_ratio_plot(
            misclassified_json_path=args.additional_misclassified_json,
            sre_stats_pkl_path=args.additional_sre_stats_pkl,
            ratio_output_path=args.additional_ratio_output,
            use_density=True,
        )
        # Additional second plot for M2 (total SRE)
        generate_ratio_plot(
            misclassified_json_path=args.additional_misclassified_json,
            sre_stats_pkl_path=args.additional_sre_stats_pkl,
            ratio_output_path=args.additional_ratio_output_m2,
            use_density=False,
        )


if __name__ == "__main__":
    main()


