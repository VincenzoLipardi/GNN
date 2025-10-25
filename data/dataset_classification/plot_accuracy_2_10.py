import json
import os
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt


RESULT_FILES: Dict[str, str] = {
    "product_states_2_10": "/data/P70087789/GNN/data/dataset_classification/results/training_product_states_2_10.json",
    "clifford-evolved_2_10": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_clifford_evolved_2_10.json",
    "entangled_2_10": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_entangled_2_10.json",
    "product_states_11_25": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_product_states_11_25.json",
    "clifford-evolved_11_25": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_clifford_evolved_11_25.json",
    "entangled_11_25": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_entangled_11_25.json",
}


def load_confusion_totals(path: str) -> Tuple[int, int, int, int]:
    """
    Load a results JSON file and return the aggregate (tn, fp, fn, tp) totals.

    Supports:
    - Top-level "confusion_matrix"
    - "per_depth" with per-depth confusion matrices (summed over depths)
    - Training results with nested test set stats at "test_stats.confusion_matrix"
    """
    with open(path, "r") as f:
        data = json.load(f)

    if "confusion_matrix" in data:
        cm = data["confusion_matrix"]
        return int(cm["tn"]), int(cm["fp"]), int(cm["fn"]), int(cm["tp"])

    if "per_depth" in data and isinstance(data["per_depth"], dict):
        tn = fp = fn = tp = 0
        for _depth, entry in data["per_depth"].items():
            cm = entry.get("confusion_matrix", {})
            tn += int(cm.get("tn", 0))
            fp += int(cm.get("fp", 0))
            fn += int(cm.get("fn", 0))
            tp += int(cm.get("tp", 0))
        return tn, fp, fn, tp

    if "test_stats" in data and isinstance(data["test_stats"], dict):
        cm = data["test_stats"].get("confusion_matrix")
        if isinstance(cm, dict):
            return int(cm.get("tn", 0)), int(cm.get("fp", 0)), int(cm.get("fn", 0)), int(cm.get("tp", 0))

    raise ValueError(f"Unrecognized schema in results file: {path}")


def compute_per_label_accuracy(tn: int, fp: int, fn: int, tp: int) -> Tuple[float, float]:
    """
    Compute per-label accuracies as percentages for label 0 (stabilizer) and label 1 (magic).

    - Label 0 accuracy = TN / (TN + FP)
    - Label 1 accuracy = TP / (TP + FN)
    """
    label0_den = tn + fp
    label1_den = tp + fn

    label0_acc = (tn / label0_den) * 100.0 if label0_den > 0 else 0.0
    label1_acc = (tp / label1_den) * 100.0 if label1_den > 0 else 0.0

    return label0_acc, label1_acc


def make_barplot(dataset_order: List[str], results: Dict[str, Tuple[float, float]], out_path: str) -> None:
    # Prepare data
    label0_values = [results[name][0] for name in dataset_order]
    label1_values = [results[name][1] for name in dataset_order]

    x = list(range(len(dataset_order)))
    width = 0.45

    fig, ax = plt.subplots(figsize=(14, 8))

    bars0 = ax.bar([i - width / 2 for i in x], label0_values, width, color="#CC79A7", label="Stabilizer")
    bars1 = ax.bar([i + width / 2 for i in x], label1_values, width, color="#1f77b4", label="Magic")

    ax.set_ylabel("Accuracy (%)", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(["PS 2-10", "CS 2-10", "ES 2-10", "PS 11-25", "CS 11-25", "ES 11-25"], rotation=0, fontsize=18)
    ax.set_ylim(0, 125)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=18)

    # Annotate bar heights
    def annotate(bars):
        for b in bars:
            height = b.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=16)

    annotate(bars0)
    annotate(bars1)

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    dataset_order = [
        "product_states_2_10",
        "clifford-evolved_2_10",
        "entangled_2_10",
        "product_states_11_25",
        "clifford-evolved_11_25",
        "entangled_11_25",
    ]

    per_dataset_results: Dict[str, Tuple[float, float]] = {}
    for name in dataset_order:
        path = RESULT_FILES[name]
        tn, fp, fn, tp = load_confusion_totals(path)
        label0_acc, label1_acc = compute_per_label_accuracy(tn, fp, fn, tp)
        per_dataset_results[name] = (label0_acc, label1_acc)

    out_path = "/data/P70087789/GNN/data/dataset_classification/images/classification_accuracy_product_states_2_10_by_label.png"
    make_barplot(dataset_order, per_dataset_results, out_path)

    # Also print values to stdout for quick reference
    for name in dataset_order:
        l0, l1 = per_dataset_results[name]
        print(f"{name}: stabilizer (label 0) = {l0:.4f}%, magic (label 1) = {l1:.4f}%")


if __name__ == "__main__":
    main()


