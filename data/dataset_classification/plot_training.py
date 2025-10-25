import os
import json
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATS_JSON = "/data/P70087789/GNN/data/dataset_classification/results/training_stats.json"
IMAGES_DIR = "/data/P70087789/GNN/data/dataset_classification/images"


def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_images_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_loss(history: Dict[str, List[float]]) -> Tuple[List[int], List[float], List[float]]:
    epochs = [int(e) for e in history.get("epoch", [])]
    train = [float(x) for x in history.get("train", [])]
    val = [float(x) for x in history.get("val", [])]
    # Truncate to shortest common length to be safe
    n = min(len(epochs), len(train), len(val))
    return epochs[:n], train[:n], val[:n]


def plot_loss_curve(epochs: List[int], train_loss: List[float], val_loss: List[float], out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, constrained_layout=True)
    ax.plot(epochs, train_loss, label="Train", color="#1f77b4", linewidth=1.8)
    ax.plot(epochs, val_loss, label="Validation", color="#d62728", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _dict_cm_to_array(cm_dict: Dict[str, int]) -> np.ndarray:
    tn = int(cm_dict.get("tn", 0))
    fp = int(cm_dict.get("fp", 0))
    fn = int(cm_dict.get("fn", 0))
    tp = int(cm_dict.get("tp", 0))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def _plot_single_cm(ax: plt.Axes, cm: np.ndarray, title: str, class_names: Tuple[str, str] = ("0", "1")) -> None:
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    # Annotate values
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            color = "white" if value > thresh else "black"
            ax.text(j, i, f"{value}", ha="center", va="center", color=color, fontsize=11)
    # Minor grid lines to separate cells
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return


def plot_confusion_matrices(train_cm: np.ndarray, test_cm: np.ndarray, out_path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), dpi=150, constrained_layout=True)
    _plot_single_cm(axes[0], train_cm, "Confusion Matrix - Train")
    _plot_single_cm(axes[1], test_cm, "Confusion Matrix - Test")
    # Add a single colorbar aligned to the right
    im = axes[1].images[0]
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=12)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    _ensure_images_dir(IMAGES_DIR)
    data = _read_json(STATS_JSON)

    # 1) Loss curves
    loss_hist = data.get("loss_history", {}) if isinstance(data, dict) else {}
    epochs, train_loss, val_loss = _extract_loss(loss_hist)  # type: ignore[arg-type]
    if epochs and train_loss and val_loss:
        loss_out = os.path.join(IMAGES_DIR, "training_loss.png")
        out1 = plot_loss_curve(epochs, train_loss, val_loss, loss_out)
        print(f"Saved loss curve to: {out1}")
    else:
        print("No loss history found; skipping loss plot.")

    # 2) Confusion matrices (train and test)
    train_stats = data.get("train_stats", {}) if isinstance(data, dict) else {}
    test_stats = data.get("test_stats", {}) if isinstance(data, dict) else {}
    train_cm_dict = train_stats.get("confusion_matrix", {}) if isinstance(train_stats, dict) else {}
    test_cm_dict = test_stats.get("confusion_matrix", {}) if isinstance(test_stats, dict) else {}

    # Fallback: if only one is present, reuse it for both panes
    if not train_cm_dict and test_cm_dict:
        train_cm_dict = test_cm_dict
    if not test_cm_dict and train_cm_dict:
        test_cm_dict = train_cm_dict

    if train_cm_dict and test_cm_dict:
        train_cm = _dict_cm_to_array(train_cm_dict)  # type: ignore[arg-type]
        test_cm = _dict_cm_to_array(test_cm_dict)    # type: ignore[arg-type]
        cm_out = os.path.join(IMAGES_DIR, "confusion_matrix.png")
        out2 = plot_confusion_matrices(train_cm, test_cm, cm_out)
        print(f"Saved confusion matrices to: {out2}")
    else:
        print("No confusion matrices found; skipping confusion matrix plot.")


if __name__ == "__main__":
    main()


