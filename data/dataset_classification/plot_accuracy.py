import json
import os
import argparse
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt


RESULT_FILES_BY_VARIANT: Dict[str, Dict[str, str]] = {
    # Variant where training/evaluation base is 18
    "18": {
        "product_states_18": "/data/P70087789/GNN/data/dataset_classification/results/training_product_states_18.json",
        "clifford-evolved_18": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_18_clifford_evolved_18.json",
        "entangled_18": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_18_entangled_18.json",
        "product_states_11_25": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_18_product_states_11_25.json",
        # "clifford-evolved_11_25": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_18_clifford_evolved_11_25.json",
        # "entangled_11_25": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_18_entangled_11_25.json",
    },
    # Variant where training/evaluation base is 2_10
    "2_10": {
        "product_states_2_10": "/data/P70087789/GNN/data/dataset_classification/results/training_product_states_2_10.json",
        "clifford-evolved_2_10": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_clifford_evolved_2_10.json",
        "entangled_2_10": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_entangled_2_10.json",
        "product_states_11_25": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_product_states_11_25.json",
        # "clifford-evolved_11_25": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_clifford_evolved_11_25.json",
        # "entangled_11_25": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_entangled_11_25.json",
    },
    "2_10_balanced": {
        "product_states_2_10_balanced": "/data/P70087789/GNN/data/dataset_classification/results/training_product_states_2_10_balanced_by_sre.json",
        "clifford-evolved_2_10_balanced": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_balanced_by_sre_clifford_evolved_2_10_balanced_by_sre.json",
        #"entangled_2_10_balanced": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_balanced_by_sre_entangled_2_10_balanced_by_sre.json",
        "product_states_11_25_balanced": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_balanced_by_sre_product_states_11_25_balanced_by_sre.json",
        # "clifford-evolved_11_25_balanced": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_clifford_evolved_11_25_balanced.json",
        # "entangled_11_25_balanced": "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_entangled_11_25_balanced.json",
    },
}


def label_for_key(name: str) -> str:
    if name.startswith("product_states_"):
        short = "PS"
        suffix = name.split("_", 2)[2]
    elif name.startswith("clifford-evolved_"):
        short = "CS"
        suffix = name.split("_", 1)[1]
    elif name.startswith("entangled_"):
        short = "ES"
        suffix = name.split("_", 1)[1]
    else:
        return name

    # Drop the balanced qualifier from tick labels if present
    if suffix.endswith("_balanced"):
        suffix = suffix[: -len("_balanced")]

    range_label = suffix.replace("_", "-")
    return f"{short} {range_label}"


def load_confusion_totals(path: str) -> Tuple[int, int, int, int]:
    """
    Load a results JSON file and return the aggregate (tn, fp, fn, tp) totals.

    The file may contain either a top-level "confusion_matrix" or a "per_depth" object
    with per-depth confusion matrices. In the latter case, confusion matrices are summed
    across all depths to obtain an overall total.
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

    # Support the training results file which nests test set results
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

    bars0 = ax.bar([i - width / 2 for i in x], label0_values, width, color="forestgreen", label=r"Low-$M_2$")
    bars1 = ax.bar([i + width / 2 for i in x], label1_values, width, color="navy", label=r"High-$M_2$")

    ax.set_ylabel("Accuracy (%)", fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for_key(name) for name in dataset_order], rotation=0, fontsize=18)
    ax.set_ylim(0, 110)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.xlabel(r"Dataset", fontsize=24)
    ax.tick_params(axis='y', labelsize=22)
    ax.legend(fontsize=20)

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
    parser = argparse.ArgumentParser(description="Plot per-label accuracy for classification results")
    parser.add_argument(
        "--variant",
        choices=["18", "2_10", "2_10_balanced"],
        default="18",
        help="Select which result set to plot: 18 or 2_10",
    )
    args = parser.parse_args()

    # Build the dataset order for the chosen variant
    if args.variant == "18":
        dataset_order = [
            "product_states_18",
            "clifford-evolved_18",
            "entangled_18",
            "product_states_11_25",
            "clifford-evolved_11_25",
            "entangled_11_25",
        ]
    elif args.variant == "2_10_balanced":
        dataset_order = [
            "product_states_2_10_balanced",
            "clifford-evolved_2_10_balanced",
            "entangled_2_10_balanced",
            "product_states_11_25_balanced",
            "clifford-evolved_11_25_balanced",
            "entangled_11_25_balanced",
        ]
    elif args.variant == "2_25_balanced":
        dataset_order = [
            "product_states_2_25_balanced",
            "clifford-evolved__balanced",
            "entangled_18_balanced",
            "product_states_11_25_balanced",
            "clifford-evolved_11_25_balanced",
            "entangled_11_25_balanced",
        ]
    else:
        dataset_order = [
            "product_states_2_10",
            "clifford-evolved_2_10",
            "entangled_2_10",
            "product_states_11_25",
            "clifford-evolved_11_25",
            "entangled_11_25",
        ]

    result_files = RESULT_FILES_BY_VARIANT[args.variant]

    # Only include datasets that are present (skip commented/missing ones)
    available_names = [name for name in dataset_order if name in result_files]
    missing_names = [name for name in dataset_order if name not in result_files]
    if missing_names:
        print(f"Skipping missing datasets (not in RESULT_FILES): {', '.join(missing_names)}")

    if not available_names:
        raise SystemExit("No available datasets to plot. Please enable at least one in RESULT_FILES.")

    per_dataset_results: Dict[str, Tuple[float, float]] = {}
    for name in available_names:
        path = result_files[name]
        tn, fp, fn, tp = load_confusion_totals(path)
        label0_acc, label1_acc = compute_per_label_accuracy(tn, fp, fn, tp)
        per_dataset_results[name] = (label0_acc, label1_acc)

    out_path = f"/data/P70087789/GNN/data/dataset_classification/images/classification_accuracy_product_states_{args.variant}.png"
    make_barplot(available_names, per_dataset_results, out_path)

    # Also print values to stdout for quick reference
    for name in available_names:
        l0, l1 = per_dataset_results[name]
        print(f"{name}: stabilizer (label 0) = {l0:.4f}%, magic (label 1) = {l1:.4f}%")


if __name__ == "__main__":
    main()


