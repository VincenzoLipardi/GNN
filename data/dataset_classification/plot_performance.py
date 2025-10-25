import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator, FixedLocator, FuncFormatter

# === CONFIGURATION ===
plt.style.use("seaborn-v0_8-whitegrid")

JSON_PATH_ENTANGLED = "/data/P70087789/GNN/data/dataset_classification/results/training_product_states_2_10.json"

JSON_PATH_PRODUCT = "/data/P70087789/GNN/data/dataset_classification/results/training_product_states_18.json"
JSON_PATH_EVAL_ENTANGLED = "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_2_10_clifford_evolved_2_10.json"
JSON_PATH_EVAL_PRODUCT = "/data/P70087789/GNN/data/dataset_classification/results/evaluation_product_states_18_clifford_evolved_18.json"


IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


# === UTILITIES ===
def load_depth_accuracies(json_path):
    """
    Extract test accuracy (in [0,1]) across Clifford depths from JSON.

    Supports two schemas:
    1) "training" schema (single run): contains top-level 'test_stats' only
       -> returns depth [0] with accuracy from 'test_stats'.
    2) Per-depth schema: top-level keys are depths, each containing 'test_stats'.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Schema 1: single-run training JSON with top-level 'test_stats' -> depth 0
    if isinstance(data, dict) and "test_stats" in data and isinstance(data["test_stats"], dict):
        stats = data["test_stats"]
        acc = stats.get("accuracy", None)
        if acc is None:
            cm = stats.get("confusion_matrix", {})
            tn, fp = cm.get("tn", 0), cm.get("fp", 0)
            fn, tp = cm.get("fn", 0), cm.get("tp", 0)
            total = tn + fp + fn + tp
            acc = (tp + tn) / total if total > 0 else np.nan
        return np.array([0]), np.array([acc], dtype=float)

    # Schema 2a: evaluation JSON with 'per_depth' mapping
    if isinstance(data, dict) and "per_depth" in data and isinstance(data["per_depth"], dict):
        accuracies = {}
        for depth_key, entry in data["per_depth"].items():
            if not isinstance(entry, dict):
                continue
            metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
            acc = metrics.get("accuracy", None)
            if acc is None:
                cm = entry.get("confusion_matrix", {})
                tn, fp = cm.get("tn", 0), cm.get("fp", 0)
                fn, tp = cm.get("fn", 0), cm.get("tp", 0)
                total = tn + fp + fn + tp
                acc = (tp + tn) / total if total > 0 else np.nan
            try:
                depth = int(depth_key)
            except ValueError:
                try:
                    depth = float(depth_key)
                except ValueError:
                    continue
            accuracies[depth] = float(acc)

        if not accuracies:
            return np.array([0]), np.array([np.nan])

        depths = sorted(accuracies.keys())
        accs = [accuracies[d] for d in depths]

        # No need to force depth 0 here; evaluation typically starts at 1
        return np.array(depths), np.array(accs)

    # Schema 2b: per-depth mapping at top-level
    accuracies = {}
    for depth_key, depth_val in data.items():
        if not isinstance(depth_val, dict):
            continue
        stats = depth_val.get("test_stats", None)
        if not stats:
            continue
        # Prefer explicit accuracy if provided, else compute from confusion matrix
        acc = stats.get("accuracy", None)
        if acc is None:
            cm = stats.get("confusion_matrix", {})
            tn, fp = cm.get("tn", 0), cm.get("fp", 0)
            fn, tp = cm.get("fn", 0), cm.get("tp", 0)
            total = tn + fp + fn + tp
            acc = (tp + tn) / total if total > 0 else np.nan
        try:
            depth = int(depth_key)
        except ValueError:
            try:
                depth = float(depth_key)
            except ValueError:
                continue
        accuracies[depth] = float(acc)

    if not accuracies:
        # Fallback: no usable entries
        return np.array([0]), np.array([np.nan])

    depths = sorted(accuracies.keys())
    accs = [accuracies[d] for d in depths]

    # Ensure depth 0 is present
    if 0 not in depths:
        depths.insert(0, 0)
        accs.insert(0, accs[0] if accs else np.nan)

    return np.array(depths), np.array(accs)


def load_eval_per_class(json_path):
    """
    Load per-depth class-specific accuracies from an evaluation JSON with 'per_depth'.

    Returns:
        depths (np.ndarray): depths sorted ascending
        magic_acc (np.ndarray): tp/(tp+fn) per depth (positive class accuracy)
        stabilizer_acc (np.ndarray): tn/(tn+fp) per depth (negative class accuracy)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "per_depth" not in data or not isinstance(data["per_depth"], dict):
        return np.array([]), np.array([]), np.array([])

    magic_by_depth = {}
    stab_by_depth = {}
    for depth_key, entry in data["per_depth"].items():
        if not isinstance(entry, dict):
            continue
        cm = entry.get("confusion_matrix", {})
        tn, fp = cm.get("tn", 0), cm.get("fp", 0)
        fn, tp = cm.get("fn", 0), cm.get("tp", 0)

        magic_den = tp + fn
        stab_den = tn + fp
        magic_acc = (tp / magic_den) if magic_den > 0 else np.nan
        stab_acc = (tn / stab_den) if stab_den > 0 else np.nan

        try:
            depth = int(depth_key)
        except ValueError:
            try:
                depth = float(depth_key)
            except ValueError:
                continue
        magic_by_depth[depth] = float(magic_acc)
        stab_by_depth[depth] = float(stab_acc)

    if not magic_by_depth:
        return np.array([]), np.array([]), np.array([])

    depths = sorted(magic_by_depth.keys())
    magic = [magic_by_depth[d] for d in depths]
    stab = [stab_by_depth.get(d, np.nan) for d in depths]
    return np.array(depths), np.array(magic), np.array(stab)


def load_training_depth0_per_class(json_path):
    """
    From a training JSON (with top-level 'test_stats'), compute per-class accuracies at depth 0.

    Returns:
        magic_acc0 (float): tp/(tp+fn)
        stabilizer_acc0 (float): tn/(tn+fp)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    train_stats = data.get("train_stats", {}) if isinstance(data, dict) else {}
    test_stats = data.get("test_stats", {}) if isinstance(data, dict) else {}
    train_cm = train_stats.get("confusion_matrix", {}) if isinstance(train_stats, dict) else {}
    test_cm = test_stats.get("confusion_matrix", {}) if isinstance(test_stats, dict) else {}
    tn = (train_cm.get("tn", 0) + test_cm.get("tn", 0))
    fp = (train_cm.get("fp", 0) + test_cm.get("fp", 0))
    fn = (train_cm.get("fn", 0) + test_cm.get("fn", 0))
    tp = (train_cm.get("tp", 0) + test_cm.get("tp", 0))

    magic_den = tp + fn
    stab_den = tn + fp
    magic_acc0 = (tp / magic_den) if magic_den > 0 else np.nan
    stabilizer_acc0 = (tn / stab_den) if stab_den > 0 else np.nan
    return float(magic_acc0), float(stabilizer_acc0)


def set_adaptive_ylim(ax, values):
    """Adaptive y-axis that includes 100% and provides clear separation."""
    valid = np.array(values[np.isfinite(values)]) * 100
    if len(valid) == 0:
        ax.set_ylim(99.5, 100.2)
        return
    ymin, ymax = np.min(valid), np.max(valid)
    # Keep very tight margins around the data with small headroom
    top_headroom = 0.10  # percentage points
    bottom_margin = 0.05  # percentage points
    min_span = 0.3  # ensure at least this total span

    upper = min(100.2, ymax + top_headroom)
    # Always include 100% within the range
    if upper < 100.0:
        upper = 100.0
    lower = max(0.0, ymin - bottom_margin)
    if (upper - lower) < min_span:
        lower = max(0.0, upper - min_span)
    ax.set_ylim(lower, upper)


def set_dense_yticks(ax):
    """Set clean major/minor y-ticks (nice steps, include 100 if in view). Returns chosen major step."""
    lower, upper = ax.get_ylim()
    span = max(upper - lower, 1e-6)
    if span <= 0.4:
        step = 0.05
    elif span <= 0.8:
        step = 0.1
    elif span <= 1.6:
        step = 0.2
    elif span <= 3.0:
        step = 0.5
    else:
        step = 1.0

    start = max(0.0, np.floor(lower / step) * step)
    ticks = np.arange(start, upper + 1e-9, step)

    # Ensure a tick at 100 if it lies within the limits
    if lower <= 100.0 <= upper:
        ticks = np.unique(np.append(ticks, 100.0))

    ax.yaxis.set_major_locator(FixedLocator(ticks))

    # Minor ticks at half-step (or quarter when step is small)
    minor_step = step / 2.0 if step >= 0.1 else 0.025
    ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
    return step


# === MAIN PLOT FUNCTION ===
def plot_performance(json_product_train, json_entangled_train, json_product_eval, json_entangled_eval, output_dir):
    # Load evaluation per-class accuracies
    depths_prod_eval, magic_prod_eval, stab_prod_eval = load_eval_per_class(json_product_eval)
    depths_ent_eval, magic_ent_eval, stab_ent_eval = load_eval_per_class(json_entangled_eval)
    # Load depth-0 per-class from training JSONs
    magic0_prod, stab0_prod = load_training_depth0_per_class(json_product_train)
    magic0_ent, stab0_ent = load_training_depth0_per_class(json_entangled_train)

    # Ensure depth 0 is included for all series
    def ensure_depth0(depths_arr, magic_arr, stab_arr, magic0, stab0):
        depths_list = list(depths_arr.tolist()) if isinstance(depths_arr, np.ndarray) else list(depths_arr)
        magic_list = list(magic_arr.tolist()) if isinstance(magic_arr, np.ndarray) else list(magic_arr)
        stab_list = list(stab_arr.tolist()) if isinstance(stab_arr, np.ndarray) else list(stab_arr)
        if 0 not in depths_list:
            depths_list.append(0)
            magic_list.append(magic0)
            stab_list.append(stab0)
        return np.array(depths_list), np.array(magic_list), np.array(stab_list)

    depths_prod_eval, magic_prod_eval, stab_prod_eval = ensure_depth0(
        depths_prod_eval, magic_prod_eval, stab_prod_eval, magic0_prod, stab0_prod
    )
    depths_ent_eval, magic_ent_eval, stab_ent_eval = ensure_depth0(
        depths_ent_eval, magic_ent_eval, stab_ent_eval, magic0_ent, stab0_ent
    )

    # Depths to plot (evaluation)
    depths_eval = np.array(sorted(set(depths_prod_eval) | set(depths_ent_eval)))

    def align_depths(all_depths, partial_depths, partial_vals):
        out = np.full_like(all_depths, np.nan, dtype=float)
        for d, v in zip(partial_depths, partial_vals):
            out[np.where(all_depths == d)] = v
        return out

    magic_prod_eval_aligned = align_depths(depths_eval, depths_prod_eval, magic_prod_eval)
    magic_ent_eval_aligned = align_depths(depths_eval, depths_ent_eval, magic_ent_eval)
    stab_prod_eval_aligned = align_depths(depths_eval, depths_prod_eval, stab_prod_eval)
    stab_ent_eval_aligned = align_depths(depths_eval, depths_ent_eval, stab_ent_eval)

    # Convert to percentage
    magic_prod_eval_pct = magic_prod_eval_aligned * 100
    magic_ent_eval_pct = magic_ent_eval_aligned * 100
    stab_prod_eval_pct = stab_prod_eval_aligned * 100
    stab_ent_eval_pct = stab_ent_eval_aligned * 100

    # === Create figure ===
    fig, (ax_magic, ax_stab) = plt.subplots(
        2, 1, figsize=(9.0, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12}
    )

    # Colors
    colors = {
        "prod": "#1f77b4",
        "ent": "#ff7f0e",
    }

    # Top subplot: Magic states
    ax_magic.plot(depths_eval, magic_prod_eval_pct, "o-", linewidth=2, label="Trained on PS 18", color=colors["prod"])
    ax_magic.plot(depths_eval, magic_ent_eval_pct, "s--", linewidth=2, label="Trained on PS 2-10", color=colors["ent"])

    # Bottom subplot: Stabilizer states
    ax_stab.plot(depths_eval, stab_prod_eval_pct, "o-", linewidth=2, label="Trained on PS 18", color=colors["prod"])
    ax_stab.plot(depths_eval, stab_ent_eval_pct, "s--", linewidth=2, label="Trained on PS 2-10", color=colors["ent"])

    # Axis formatting
    set_adaptive_ylim(ax_magic, np.concatenate([magic_prod_eval_aligned, magic_ent_eval_aligned]))
    ax_magic.set_ylabel("Accuracy (%)", fontsize=12)
    ax_magic.grid(True, alpha=0.3)
    step_magic = set_dense_yticks(ax_magic)
    if step_magic >= 1.0:
        ax_magic.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}")
    elif step_magic >= 0.1:
        ax_magic.yaxis.set_major_formatter(lambda x, _: f"{x:.1f}")
    else:
        ax_magic.yaxis.set_major_formatter(lambda x, _: f"{x:.2f}")

    set_adaptive_ylim(ax_stab, np.concatenate([stab_prod_eval_aligned, stab_ent_eval_aligned]))
    ax_stab.set_ylabel("Accuracy (%)", fontsize=12)
    ax_stab.grid(True, alpha=0.3)
    step_stab = set_dense_yticks(ax_stab)
    if step_stab >= 1.0:
        ax_stab.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}")
    elif step_stab >= 0.1:
        ax_stab.yaxis.set_major_formatter(lambda x, _: f"{x:.1f}")
    else:
        ax_stab.yaxis.set_major_formatter(lambda x, _: f"{x:.2f}")

    ax_magic.set_title("Magic States", fontsize=14, fontweight="bold")
    ax_stab.set_title("Stabilizer States", fontsize=14, fontweight="bold")
    ax_stab.set_xlabel("Clifford Depth", fontsize=12)

    # Shared legend and x-axis setup
    for ax in [ax_magic, ax_stab]:
        ax.legend(loc="lower right", frameon=True)
    if len(depths_eval) > 0:
        max_d = int(max(depths_eval))
        odd_ticks = list(range(1, max_d + 1, 2))
        major_ticks = [0] + odd_ticks
        ax_magic.set_xticks(major_ticks)
        ax_stab.set_xticks(major_ticks)
        ax_magic.xaxis.set_minor_locator(MultipleLocator(1))
        ax_stab.xaxis.set_minor_locator(MultipleLocator(1))

    # Layout
    fig.align_labels()
    plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.15)

    # Save output
    filename = "classification_depth_same_size.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved figure to: {filepath}")


# === RUN ===
if __name__ == "__main__":
    plot_performance(
        JSON_PATH_PRODUCT,
        JSON_PATH_ENTANGLED,
        JSON_PATH_EVAL_PRODUCT,
        JSON_PATH_EVAL_ENTANGLED,
        IMAGES_DIR,
    )
