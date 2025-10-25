import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_overall_and_class0_accuracy(eval_exploratory_path: str) -> Tuple[float, float]:
    data = read_json(eval_exploratory_path)
    # The main exploratory file stores metrics in a nested dict under "metrics"
    metrics = data.get("metrics", data)

    overall_accuracy = float(metrics["accuracy"])  # required

    # Expect scikit-learn-style confusion matrix with labels=[0,1]
    # [[TN, FP], [FN, TP]]
    cm = metrics.get("confusion_matrix")
    if not cm or not isinstance(cm, list) or len(cm) != 2 or len(cm[0]) != 2 or len(cm[1]) != 2:
        raise ValueError("confusion_matrix must be a 2x2 list [[TN, FP], [FN, TP]] in eval_exploratory.json")

    tn, fp = int(cm[0][0]), int(cm[0][1])
    class0_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return overall_accuracy, class0_accuracy


def get_simple_accuracy(json_path: str) -> float:
    data = read_json(json_path)
    # Some files store metrics flat, others under "metrics"; prefer flat key first
    if "accuracy" in data:
        return float(data["accuracy"]) 
    metrics = data.get("metrics")
    if metrics and "accuracy" in metrics:
        return float(metrics["accuracy"]) 
    raise KeyError(f"No accuracy found in {json_path}")


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    base_dir = "/data/P70087789/GNN/data/dataset_classification"
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    path_all = os.path.join(base_dir, "eval_exploratory.json")
    path_pos_high = os.path.join(base_dir, "eval_exploratory_pos_high.json")
    path_pos_low = os.path.join(base_dir, "eval_exploratory_pos_low.json")
    path_entangled = os.path.join(base_dir, "eval_entangled.json")
    path_entangled_on_class = os.path.join(base_dir, "eval_entangled_on_class.json")

    overall_acc, class0_acc = get_overall_and_class0_accuracy(path_all)
    pos_high_acc = get_simple_accuracy(path_pos_high)
    pos_low_acc = get_simple_accuracy(path_pos_low)

    labels = [
        "All",
        "Stabilizer",
        "Magic High SRE",
        "Magic Low SRE",
    ]
    values = [overall_acc, class0_acc, pos_high_acc, pos_low_acc]

    # Use distinct colorblind-safe colors for bars
    colors = [
        "#0072B2",  # blue
        "#009E73",  # green
        "#E69F00",  # orange
        "#CC79A7",  # purple
    ]
    sample_sizes = [15000, 255, 7373, 7372]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    x_positions = range(len(values))

    for i, (x, v) in enumerate(zip(x_positions, values)):
        ax.bar(x, v, color=colors[i], edgecolor="black", linewidth=0.8)
        ax.text(x, v + 0.01, format_percentage(v), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(0.2, max(values) + 0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, symbol=""))
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    # Legend with sample sizes per category
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="black", label=f"{labels[i]} ({sample_sizes[i]} samples)")
        for i in range(len(labels))
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    output_path = os.path.join(images_dir, "exploratory_accuracy_barplot.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Saved barplot to: {output_path}")

    # ------------------------
    # Second figure: entangled-related accuracies
    # - Split Product-states→Entangled by class (Stabilizer vs Magic) using confusion matrix
    # - Merge Magic18 and Stabilizer18 into one "Product states" bar

    # Read entangled evaluation and compute per-class accuracies
    ent_data = read_json(path_entangled)
    ent_metrics_main = ent_data.get("metrics", ent_data)
    cm_ent = ent_metrics_main.get("confusion_matrix")
    if not cm_ent or not isinstance(cm_ent, list) or len(cm_ent) != 2 or len(cm_ent[0]) != 2 or len(cm_ent[1]) != 2:
        raise ValueError("confusion_matrix must be a 2x2 list [[TN, FP], [FN, TP]] in eval_entangled.json")
    tn_e, fp_e = int(cm_ent[0][0]), int(cm_ent[0][1])
    fn_e, tp_e = int(cm_ent[1][0]), int(cm_ent[1][1])
    ent_stabilizer_acc = tn_e / (tn_e + fp_e) if (tn_e + fp_e) > 0 else 0.0
    ent_magic_acc = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0.0

    # Overall for product states (entangled-trained model on magic_18 + stabilizer_18)
    entangled_on_class_data = read_json(path_entangled_on_class)
    ent_metrics = entangled_on_class_data.get("metrics", entangled_on_class_data)
    ent_on_class_overall = float(ent_metrics["accuracy"]) if "accuracy" in ent_metrics else float(entangled_on_class_data["accuracy"])  # 1.0 in provided file

    labels2 = [
        "Stabilizer",
        "Magic",
        "Product states",
    ]
    values2 = [ent_stabilizer_acc, ent_magic_acc, ent_on_class_overall]

    # Colors: first two bars (trained on product states) blue, last bar (trained on entangled) red
    colors2 = ["#0072B2", "#0072B2", "#D55E00"]

    # Enlarge figure and reserve right margin for legend
    fig2, ax2 = plt.subplots(figsize=(9.5, 5.0), dpi=150)
    x_positions2 = range(len(values2))
    for i, (x, v) in enumerate(zip(x_positions2, values2)):
        ax2.bar(x, v, color=colors2[i], edgecolor="black", linewidth=0.8)
        ax2.text(x, v + 0.005, format_percentage(v), ha="center", va="bottom", fontsize=10)

    ax2.set_xticks(list(x_positions2))
    ax2.set_xticklabels(labels2, rotation=15, ha="right")
    ax2.set_xlabel("Test Dataset")
    ax2.set_ylabel("Accuracy (%)")
    y_max2 = max(1.2, max(values2) + 0.2)
    ax2.set_ylim(0, y_max2)
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, symbol=""))
    # no title per request
    # Legend explaining training source colors
    from matplotlib.patches import Patch as Patch2
    legend2_handles = [
        Patch2(facecolor="#0072B2", edgecolor="black", label="Trained on Product States"),
        Patch2(facecolor="#D55E00", edgecolor="black", label="Trained on Entangled States"),
    ]
    # Place legend inside the plot in the upper-right blank space
    ax2.legend(handles=legend2_handles, loc="upper center", frameon=True)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax2.set_axisbelow(True)

    output_path2 = os.path.join(images_dir, "entangled_accuracy_barplot.png")
    plt.tight_layout()
    plt.savefig(output_path2)
    plt.close(fig2)

    print(f"Saved barplot to: {output_path2}")


if __name__ == "__main__":
    main()


