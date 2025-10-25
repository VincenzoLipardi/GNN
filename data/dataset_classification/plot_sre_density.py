import argparse
import json
import math
import os
import pickle
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statistics
import numpy as np

# Fixed bin count for both plots
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
    # Prefer explicit keys
    if "misclassified_indices" in payload and isinstance(payload["misclassified_indices"], list):
        return [int(x) for x in payload["misclassified_indices"]]
    if "misclassified" in payload:
        val = payload["misclassified"]
        if isinstance(val, list):
            return [int(x) for x in val]
        if isinstance(val, dict):
            return [int(k) for k in val.keys()]
    # Fallback: use indices_test_order if present (as seen in provided files)
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
        # Look for SRE keys
        for key in entry.keys():
            kl = str(key).lower()
            if kl in ("sre", "sre_value", "staterenyi", "state_renyi"):  # common name variants
                v = _maybe_float(entry[key])
                if v is not None:
                    sre_value = v
            if kl in ("num_qubits", "n_qubits", "nqubits"):
                try:
                    num_qubits = int(entry[key])
                except Exception:
                    pass
        # Some formats wrap metadata inside 'info'
        info = entry.get("info") if isinstance(entry, dict) else None
        if isinstance(info, dict):
            if "num_qubits" in info:
                try:
                    num_qubits = int(info["num_qubits"])
                except Exception:
                    pass
            # SRE might also be here (less likely)
            for key in ("sre", "sre_value"):
                if key in info:
                    v = _maybe_float(info[key])
                    if v is not None:
                        sre_value = v

    elif isinstance(entry, (list, tuple)):
        # Common format: (info_dict, sre_dict) where sre_dict contains 'sre_total'
        if len(entry) == 2 and all(isinstance(x, dict) for x in entry):
            info_dict, sre_dict = entry[0], entry[1]
            if isinstance(info_dict, dict):
                try:
                    num_qubits = int(info_dict.get("num_qubits", num_qubits))
                except Exception:
                    pass
            if isinstance(sre_dict, dict):
                # Prefer explicit total; fallback to aggregate of per-qubit if available
                v = _maybe_float(sre_dict.get("sre_total"))
                if v is None and isinstance(sre_dict.get("sre_per_qubit"), (list, tuple)):
                    try:
                        v = float(sum(float(x) for x in sre_dict["sre_per_qubit"]))
                    except Exception:
                        v = None
                if v is not None:
                    sre_value = v
        else:
            # Generic fallback: search for the first float-like item as SRE and any dict specifying num_qubits
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
        # Maybe the entry itself is just the SRE
        v = _maybe_float(entry)
        if v is not None:
            sre_value = v

    return sre_value, num_qubits


def load_labels_from_product_dataset(path: Optional[str]) -> Optional[List[int]]:
    if not path or not os.path.exists(path):
        return None
    try:
        data = load_pickle(path)
        labels: List[int] = []
        if isinstance(data, dict):
            for k in sorted((int(x) for x in data.keys())):
                entry = data[k]
                lbl: Optional[int] = None
                if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], int):
                    lbl = int(entry[1])
                elif isinstance(entry, dict) and "label" in entry and isinstance(entry["label"], (int, bool)):
                    v = entry["label"]
                    lbl = int(v) if not isinstance(v, bool) else (1 if v else 0)
                labels.append(0 if lbl is None else lbl)
            return labels
        if isinstance(data, (list, tuple)):
            for entry in data:
                lbl: Optional[int] = None
                if isinstance(entry, (list, tuple)):
                    if len(entry) >= 2 and isinstance(entry[1], int):
                        lbl = int(entry[1])
                elif isinstance(entry, dict) and "label" in entry and isinstance(entry["label"], (int, bool)):
                    v = entry["label"]
                    lbl = int(v) if not isinstance(v, bool) else (1 if v else 0)
                labels.append(0 if lbl is None else lbl)
            return labels
    except Exception:
        return None
    return None


def build_base_index_to_density(
    sre_stats_obj: Any,
    fallback_qubits: int,
) -> List[Optional[float]]:
    densities: List[Optional[float]] = []
    if isinstance(sre_stats_obj, dict):
        # If the PKL is a dict keyed by base index, densify to list by index order
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

    # Unknown structure: cannot build
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
        # Expect an integer like 10, 25, etc.
        if ratio.is_integer():
            return int(ratio)
        # Close to integer due to rounding
        rounded = int(round(ratio))
        if rounded > 0:
            return rounded
    except Exception:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot combined frequency-normalized SRE density histograms.")
    parser.add_argument(
        "--misclassified_json",
        default="/data/P70087789/GNN/data/dataset_classification/results/misclassified_product_states_18_clifford_evolved_18.json",
        help="Path to JSON with misclassified indices",
    )
    parser.add_argument(
        "--sre_stats_pkl",
        default="/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_18_sre_stats.pkl",
        help="Path to PKL containing SRE stats for base product circuits",
    )
    parser.add_argument(
        "--labels_pkl",
        default="/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_18.pkl",
        help="Path to PKL containing base dataset with labels (to exclude label 0)",
    )
    # Deprecated single-output options removed; bin count is controlled by BIN_COUNT above
    parser.add_argument(
        "--combined_output",
        default="/data/P70087789/GNN/data/dataset_classification/images/sre_density_combined_18.png",
        help="Output image path for combined frequency-normalized histogram overlay",
    )
    parser.add_argument(
        "--ratio_scatter_output",
        default="/data/P70087789/GNN/data/dataset_classification/images/sre_density_misratio_scatter_18.png",
        help="Output image path for per-bin misclassification ratio scatter plot",
    )
    args = parser.parse_args()

    payload = read_json(args.misclassified_json)
    indices = find_misclassified_indices(payload)

    dataset_path = payload.get("dataset") if isinstance(payload, dict) else None
    default_nq = infer_num_qubits_from_path(dataset_path, default_value=18)

    sre_stats = load_pickle(args.sre_stats_pkl)
    base_densities = build_base_index_to_density(sre_stats, fallback_qubits=default_nq)
    num_base = len(base_densities)
    # Load labels to filter to label==1 circuits only
    labels = load_labels_from_product_dataset(args.labels_pkl)
    if labels is None or len(labels) != num_base:
        raise RuntimeError("Labels are required and must match base dataset size to filter label=1.")

    # Compute variants per base by inspecting evolved dataset length if possible
    variants_per_base = compute_variants_per_base(dataset_path, num_base)
    if variants_per_base is None:
        # Fall back to 25 variants per base as per dataset note
        variants_per_base = 25

    misclassified_densities: List[float] = []
    for idx in indices:
        try:
            base_idx = int(idx) // int(variants_per_base)
        except Exception:
            continue
        if 0 <= base_idx < num_base and int(labels[base_idx]) == 1:
            d = base_densities[base_idx]
            if d is not None and math.isfinite(float(d)):
                misclassified_densities.append(float(d))

    if not misclassified_densities:
        raise RuntimeError("No SRE densities could be derived for the provided misclassified indices.")

    # Build densities for ALL base circuits with label==1 (used for overlay and ratio denominator)
    all_densities_label1: List[float] = []
    for i, d in enumerate(base_densities):
        if int(labels[i]) == 1 and d is not None and math.isfinite(float(d)):
            all_densities_label1.append(float(d))
    if all_densities_label1:
        # Build combined frequency-normalized overlay figure (misclassified vs all (label!=0))
        # Use common bin edges for fair comparison
        combined_output_path = args.combined_output

        # Determine common bins from the union of misclassified and ALL label==1 datasets' ranges
        data_min = min(min(misclassified_densities), min(all_densities_label1))
        data_max = max(max(misclassified_densities), max(all_densities_label1))
        # Construct explicit common bin edges with NumPy for consistent binning across plots
        bin_edges = np.linspace(data_min, data_max, BIN_COUNT + 1)

        # Compute medians for vertical guide lines
        median_mis = None
        median_all = None
        try:
            median_mis = statistics.median(misclassified_densities)
            median_all = statistics.median(all_densities_label1)
        except Exception:
            pass

        plt.figure(figsize=(8, 5))
        plt.hist(
            misclassified_densities,
            bins=bin_edges,
            density=True,
            alpha=1,
            color="#d62728",  # red
            edgecolor="black",
            label="Misclassified",
        )
        plt.hist(
            all_densities_label1,
            bins=bin_edges,
            density=True,
            alpha=0.7,
            color="#1f77b4",  # blue
            edgecolor="black",
            label="All (label=1)",
        )
        if median_mis is not None:
            plt.axvline(
                median_mis,
                color="#d62728",
                linestyle="--",
                linewidth=2,
                label="Median (Misclassified)",
            )
        if median_all is not None:
            plt.axvline(
                median_all,
                color="#1f77b4",
                linestyle="--",
                linewidth=2,
                label="Median (All)",
            )
        plt.xlabel(r"$M_2 / n$")
        plt.ylabel("Frequency (%)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        os.makedirs(os.path.dirname(combined_output_path), exist_ok=True)
        plt.savefig(combined_output_path, dpi=150)
        print(f"Saved combined frequency-normalized histogram overlay to: {combined_output_path}")
        # Print medians of both distributions
        try:
            median_mis = statistics.median(misclassified_densities)
            median_all = statistics.median(all_densities_label1)
            print(f"Median(Misclassified): {median_mis:.6f} | Median(All): {median_all:.6f}")
        except Exception:
            pass

        # Build per-bin misclassification ratio scatter plot using the same bin edges
        if bin_edges is not None:
            mis_counts, _ = np.histogram(misclassified_densities, bins=bin_edges)
            all_counts_base, _ = np.histogram(all_densities_label1, bins=bin_edges)
            all_counts = all_counts_base * int(variants_per_base)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.true_divide(mis_counts, all_counts)
                ratio[~np.isfinite(ratio)] = 0.0
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bin_widths = (bin_edges[1:] - bin_edges[:-1])

            plt.figure(figsize=(8, 5))
            plt.bar(bin_centers, ratio, width=bin_widths, color="indigo", edgecolor="black", alpha=0.85, align="center")
            plt.xlabel(r"$M_2 / n$")
            plt.ylabel("Misclassified / All")
            plt.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            ratio_path = args.ratio_scatter_output
            os.makedirs(os.path.dirname(ratio_path), exist_ok=True)
            plt.savefig(ratio_path, dpi=150)
            print(f"Saved misclassification ratio scatter plot to: {ratio_path}")
            # Print totals used (based on histogram counts)
            total_mis = int(mis_counts.sum())
            total_all = int(all_counts.sum())
            print(f"Ratio plot totals — Misclassified: {total_mis} | All: {total_all}")
            # Print per-bin raw counts
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



if __name__ == "__main__":
    main()


