import os
import json
import pickle
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


DATASETS: List[Tuple[str, str]] = [
    ("product_states_18.pkl", "Product States 18 "),
    #("product_states_2_10.pkl", "2-10 Qubits"),
    #("product_states_11_25.pkl", "PS_11_25"),
    ("product_states_2_25.pkl", "Product States 2-25"),
]

# New dataset directory containing PKLs and *_sre_stats.pkl files
DATASET_DIR = "/data/P70087789/GNN/data/dataset_classification/dataset_type"


def find_completed_stats(base_dir: str) -> List[Tuple[str, str, str]]:
    completed: List[Tuple[str, str, str]] = []
    for base_name, label in DATASETS:
        src = os.path.join(base_dir, base_name)
        stats_path = os.path.splitext(src)[0] + "_sre_stats.pkl"
        if os.path.exists(stats_path):
            completed.append((label, base_name, stats_path))
    return completed


def load_sre_totals(stats_path: str) -> np.ndarray:
    with open(stats_path, "rb") as fh:
        items = pickle.load(fh)
    totals: List[float] = []
    for _, stats in items:
        if not isinstance(stats, dict):
            continue
        val = stats.get("sre_total", np.nan)
        try:
            fval = float(val)
        except Exception:
            fval = np.nan
        if np.isfinite(fval):
            totals.append(fval)
    return np.asarray(totals, dtype=float)


def load_sre_totals_with_nans(stats_path: str) -> np.ndarray:
    """Load SRE totals preserving order and NaNs for alignment with labels."""
    with open(stats_path, "rb") as fh:
        items = pickle.load(fh)
    totals: List[float] = []
    for _, stats in items:
        if not isinstance(stats, dict):
            totals.append(float("nan"))
            continue
        val = stats.get("sre_total", np.nan)
        try:
            totals.append(float(val))
        except Exception:
            totals.append(float("nan"))
    return np.asarray(totals, dtype=float)


def load_sre_totals_split_by_label(stats_path: str, dataset_pkl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return two arrays (stabilizer_totals, magic_totals) aligned with dataset labels.

    The stats file is expected to contain a list of (item, stats) in the same order as the
    original dataset PKL. Labels are read from the dataset PKL entries.
    """
    # Load totals from stats (preserve order; allow NaNs for alignment)
    with open(stats_path, "rb") as fh:
        items_stats = pickle.load(fh)
    totals_all: List[float] = []
    for _, stats in items_stats:
        if not isinstance(stats, dict):
            totals_all.append(float("nan"))
            continue
        val = stats.get("sre_total", np.nan)
        try:
            totals_all.append(float(val))
        except Exception:
            totals_all.append(float("nan"))

    # Load labels from dataset PKL
    with open(dataset_pkl_path, "rb") as fh:
        items_ds = pickle.load(fh)
    labels: List[int] = []
    for it in items_ds:
        if isinstance(it, tuple) and len(it) >= 2:
            try:
                labels.append(int(it[1]))
            except Exception:
                labels.append(0)
        elif isinstance(it, dict) and "label" in it:
            try:
                labels.append(int(it.get("label", 0)))
            except Exception:
                labels.append(0)
        else:
            labels.append(0)

    n = min(len(totals_all), len(labels))
    totals_all = totals_all[:n]
    labels = labels[:n]

    stab: List[float] = []
    magic: List[float] = []
    for t, y in zip(totals_all, labels):
        if not np.isfinite(t):
            continue
        if int(y) == 0:
            stab.append(float(t))
        else:
            magic.append(float(t))
    return np.asarray(stab, dtype=float), np.asarray(magic, dtype=float)


def plot_boxplot(base_dir: str, out_path: str) -> str:
    completed = find_completed_stats(base_dir)
    if not completed:
        raise FileNotFoundError("No *_sre_stats.pkl files found among the expected datasets.")

    data_arrays: List[np.ndarray] = []
    labels: List[str] = []
    legend_patches: List[Patch] = []

    # Okabe-Ito colorblind-safe palette (choose first N for N datasets),
    # also reasonably distinct under grayscale by varying luminance.
    palette = [
        "#0072B2",  # blue
        "#CC79A7",  # reddish purple
        "#009E73",  # bluish green
        "#E69F00",  # orange

    ]
    # Use different colors for Stabilizer vs Magic across all datasets
    stabilizer_color = "#CC79A7"  # reddish purple
    magic_color = "#0072B2"       # blue

    for idx, (label, base_name, stats_path) in enumerate(completed):
        dataset_pkl = os.path.join(base_dir, base_name)
        stab, magic = load_sre_totals_split_by_label(stats_path, dataset_pkl)
        data_arrays.extend([stab, magic])
        labels.extend([f"{label} - Stabilizer", f"{label} - Magic"])
        
        # Print min and max for each boxplot
        print(f"\n{label}:")
        if stab.size > 0:
            print(f"  Stabilizer: min={np.min(stab):.3f}, max={np.max(stab):.3f}")
        else:
            print(f"  Stabilizer: no data")
        if magic.size > 0:
            print(f"  Magic: min={np.min(magic):.3f}, max={np.max(magic):.3f}")
        else:
            print(f"  Magic: no data")

    # No extra series included in the main image

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(6, 6))
    
    # Create custom positions to make boxes closer together
    positions = []
    for i in range(len(completed)):
        base_pos = i * 2 + 1  # Start at 1, 3, 5, etc.
        positions.extend([base_pos - 0.3, base_pos + 0.3])  # Closer together

    
    bp = plt.boxplot(data_arrays, positions=positions, patch_artist=True, showfliers=False)


    # Assign colors: Stabilizer = reddish purple, Magic = blue
    # Hide median lines for stabilizer (even indices)
    for i, patch in enumerate(bp["boxes"]):
        color = stabilizer_color if i % 2 == 0 else magic_color  # Even indices = Stabilizer, odd = Magic
        patch.set_facecolor(color)
        patch.set_edgecolor("#000000")
        patch.set_alpha(0.8)
        
        # Hide median line for stabilizer boxplots (even indices)
        if i % 2 == 0:  # Stabilizer
            bp["medians"][i].set_visible(False)

    ax = plt.gca()
    plt.ylabel("Stabilizer Rényi Entropy")
    plt.xlabel("Dataset")

    # Create single tick per dataset positioned between the two boxes
    dataset_labels = []
    tick_positions = []
    for i in range(len(completed)):
        dataset_name = labels[i * 2].split(" - ")[0]  # Extract dataset name
        dataset_labels.append(dataset_name)
        tick_positions.append(i * 2 + 1)  # Center between boxes
    
    # Set x-axis ticks to show only dataset names at center positions
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(dataset_labels, rotation=0, ha="center")
    
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    
    # Simple legend with just Stabilizer and Magic
    legend_patches = [
        Patch(facecolor=stabilizer_color, edgecolor="#000000", alpha=0.9, label="Stabilizer"),
        Patch(facecolor=magic_color, edgecolor="#000000", alpha=0.9, label="Magic"),
    ]
    plt.legend(handles=legend_patches, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot boxplot of SRE totals split by class across datasets.")
    parser.add_argument(
        "--base_dir",
        default=DATASET_DIR,
        help="Directory containing the dataset PKLs and *_sre_stats.pkl outputs (default: dataset_type).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "sre_boxplot.png"),
        help="Output path for the boxplot image.",
    )
    args = parser.parse_args()

    out = plot_boxplot(args.base_dir, args.out)
    print(f"Saved boxplot to: {out}")


if __name__ == "__main__":
    main()


