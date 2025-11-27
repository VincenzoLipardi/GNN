import argparse
import math
import os
import pickle
from typing import Any, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _maybe_float(x: Any) -> Optional[float]:
    try:
        if isinstance(x, (int, float)):
            v = float(x)
            if math.isfinite(v):
                return v
    except Exception:
        pass
    return None


def _extract_sre_from_entry(entry: Any) -> Optional[float]:
    # Accept common key variants; fall back to sum(sre_per_qubit)
    if isinstance(entry, dict):
        for key in ("sre_total", "total_sre", "sre", "sre_value", "state_renyi", "staterenyi"):
            if key in entry:
                v = _maybe_float(entry.get(key))
                if v is not None:
                    return v
        per = entry.get("sre_per_qubit")
        if isinstance(per, (list, tuple)):
            try:
                vals = [float(x) for x in per]
                s = float(sum(vals))
                if math.isfinite(s):
                    return s
            except Exception:
                pass
        inner = entry.get("info")
        if isinstance(inner, dict):
            for key in ("sre_total", "sre", "sre_value"):
                if key in inner:
                    v = _maybe_float(inner.get(key))
                    if v is not None:
                        return v
        return None

    if isinstance(entry, (list, tuple)):
        # Common pattern: (info_dict, sre_dict)
        if len(entry) == 2 and isinstance(entry[1], dict):
            return _extract_sre_from_entry(entry[1])
        for item in entry:
            v = _maybe_float(item)
            if v is not None:
                return v
        return None

    return _maybe_float(entry)


def coerce_to_float_list(obj: Any) -> List[float]:
    values: List[float] = []
    if isinstance(obj, dict):
        iterable: Iterable[Any] = obj.values()
    elif isinstance(obj, (list, tuple)):
        iterable = obj
    else:
        v = _extract_sre_from_entry(obj)
        return [v] if v is not None else []

    for entry in iterable:
        v = _extract_sre_from_entry(entry)
        if v is not None and math.isfinite(float(v)):
            values.append(float(v))
    return values


def plot_violin(datasets: List[List[float]], labels: List[str], output_path: str) -> None:
    plt.figure(figsize=(14, 8))
    parts = plt.violinplot(datasets, showmeans=False, showmedians=True, widths=0.8)
    colors = ["#4C78A8", "#F58518", "#54A24B"]  # blue, orange, green
    for idx, pc in enumerate(parts.get("bodies", [])):
        pc.set_facecolor(colors[idx % len(colors)])
        pc.set_edgecolor("black")
        pc.set_alpha(0.85)
    for k in ("cbars", "cmins", "cmaxes", "cmedians"):
        if k in parts and parts[k] is not None:
            parts[k].set_color("black")
            if k in ("cbars", "cmins", "cmaxes"):
                parts[k].set_alpha(0.3)  # more transparent vertical lines
    if "cmedians" in parts and parts["cmedians"] is not None:
        parts["cmedians"].set_linewidth(2.0)
        parts["cmedians"].set_alpha(1.0)

    # Annotate medians with their numeric values (slightly above the line)
    all_vals: List[float] = [v for data in datasets for v in data]
    if all_vals:
        data_min = float(min(all_vals))
        data_max = float(max(all_vals))
        dy = 0.015 * (data_max - data_min if data_max > data_min else max(abs(data_max), 1.0))
    else:
        dy = 0.0
    for i, data in enumerate(datasets, start=1):
        if not data:
            continue
        med = float(np.median(data))
        plt.text(
            i,
            med + dy,
            f"{med:.2f}",
            ha="center",
            va="bottom",
            fontsize=18,
            color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )
    plt.xticks(range(1, len(labels) + 1), labels, fontsize=20)
    plt.ylabel(r"$M_2$", fontsize=24)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot violin distributions of SRE values for three datasets and print sizes.")
    parser.add_argument(
        "--pkl_2_10",
        default="/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_10_balanced_by_sre_stats.pkl",
        help="Path to PKL with SRE values for 2-10 dataset",
    )
    parser.add_argument(
        "--pkl_11_25",
        default="/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_11_25_balanced_by_sre_stats.pkl",
        help="Path to PKL with SRE values for 11-25 dataset",
    )
    parser.add_argument(
        "--pkl_2_25",
        default="/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_25_balanced_by_sre_stats.pkl",
        help="Path to PKL with SRE values for 2-25 dataset",
    )
    parser.add_argument(
        "--output",
        default="/data/P70087789/GNN/data/dataset_classification/images/sre_violin_balanced.png",
        help="Output image path",
    )
    args = parser.parse_args()

    obj_2_10 = load_pickle(args.pkl_2_10)
    obj_11_25 = load_pickle(args.pkl_11_25)
    obj_2_25 = load_pickle(args.pkl_2_25)

    vals_2_10 = coerce_to_float_list(obj_2_10)
    vals_11_25 = coerce_to_float_list(obj_11_25)
    vals_2_25 = coerce_to_float_list(obj_2_25)

    print(f"Dataset sizes (valid SRE values):")
    print(f"  2-10:  {len(vals_2_10)}")
    print(f"  11-25: {len(vals_11_25)}")
    print(f"  2-25:  {len(vals_2_25)}")

    plot_violin(
        datasets=[vals_2_10, vals_11_25, vals_2_25],
        labels=["PS 2-10", "PS 11-25", "PS 2-25"],
        output_path=args.output,
    )
    print(f"Saved violin plot to: {args.output}")


if __name__ == "__main__":
    main()


