import os
import pickle
from typing import Dict, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "/data/P70087789/GNN/models/results"
MLP_PICKLE = os.path.join(RESULTS_DIR, "mlp_results.pkl")
GNN_PICKLE = os.path.join(RESULTS_DIR, "gnn_results.pkl")
SVR_PICKLE = os.path.join(RESULTS_DIR, "svr_resuls.pkl")


def _load_pickle(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_tuples_from_saved(saved: Optional[Dict[str, Any]]) -> Optional[Dict[str, Tuple[float, float, float]]]:
    if saved is None:
        return None
    try:
        def _mk(res: Dict[str, float]) -> Tuple[float, float, float]:
            return (
                float(res.get("train_mse", 0.0)),
                float(res.get("test_mse", 0.0)),
                float(res.get("extra_mse", 0.0)),
            )

        return {
            "Random": _mk(saved.get("random", {})),
            "TIM": _mk(saved.get("tim", {})),
        }
    except Exception:
        return None


def _fetch_experiment(results: Dict[str, Any], key_options: Tuple[str, ...]) -> Optional[Dict[str, Tuple[float, float, float]]]:
    for key in key_options:
        if key in results:
            return _to_tuples_from_saved(results[key])
    return None


def _load_svr_all(path: str) -> Optional[Dict[str, Dict[str, Tuple[float, float, float]]]]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Backward compatibility: single flat dict with 'Random'/'TIM'
    if isinstance(data, dict) and 'Random' in data and 'TIM' in data:
        return {"default": data}
    return data


def _fetch_svr_experiment(svr_all: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]], key_options: Tuple[str, ...]) -> Optional[Dict[str, Tuple[float, float, float]]]:
    if svr_all is None:
        return None
    for key in key_options:
        if key in svr_all:
            return svr_all[key]
    # fallback
    return svr_all.get("default")


def annotate_bars(bars, fmt: str = "{:.3f}") -> None:
    for rect in bars:
        height = rect.get_height()
        plt.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_compare_groups(
    ref: Dict[str, Tuple[float, float, float]],
    cmp_: Dict[str, Tuple[float, float, float]],
    ref_label: str,
    cmp_label: str,
    out_path: str,
    reverse_pct: bool = False,
    cmp_first: bool = False,
) -> None:
    # Side-by-side bars for Train/Test/Extra with value and percent delta labels
    labels = list(ref.keys())
    train_ref = [ref[k][0] for k in labels]
    test_ref = [ref[k][1] for k in labels]
    extra_ref = [ref[k][2] for k in labels]

    train_cmp = [cmp_[k][0] for k in labels]
    test_cmp = [cmp_[k][1] for k in labels]
    extra_cmp = [cmp_[k][2] for k in labels]

    # Place group centers with a moderate stride; keep groups closer together
    group_stride = 1.4 if len(labels) <= 3 else 1.3
    x = np.arange(len(labels), dtype=float) * group_stride
    # Narrower bars and wider spacing to prevent any overlap
    width = 0.16
    offsets = [-0.36, 0.0, 0.36]

    plt.figure(figsize=(10, 7.5))
    bars_all = []
    # Use the same colors as gnn_mse_bars_depth.png for Train/Test/Extra
    colors_cat = ["#1f77b4", "#5dade2", "#e69f00"]
    groups_ref = [train_ref, test_ref, extra_ref]
    groups_cmp = [train_cmp, test_cmp, extra_cmp]
    for i in range(3):
        center = x + offsets[i]
        if not cmp_first:
            # Reference (e.g., GNN): solid bars on the left
            b_ref = plt.bar(
                center - width / 2,
                groups_ref[i],
                width=width,
                color=colors_cat[i],
                label=(ref_label if i == 0 else None),
            )
            # Comparison (e.g., MLP/SVR): hatched bars on the right
            b_cmp = plt.bar(
                center + width / 2,
                groups_cmp[i],
                width=width,
                color=colors_cat[i],
                hatch="//",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
                label=(cmp_label if i == 0 else None),
            )
        else:
            # Comparison first (left), Reference second (right)
            b_cmp = plt.bar(
                center - width / 2,
                groups_cmp[i],
                width=width,
                color=colors_cat[i],
                hatch="//",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
                label=(cmp_label if i == 0 else None),
            )
            b_ref = plt.bar(
                center + width / 2,
                groups_ref[i],
                width=width,
                color=colors_cat[i],
                label=(ref_label if i == 0 else None),
            )
        bars_all.append((b_ref, b_cmp))

    plt.xticks(x, labels)
    plt.ylabel("Mean Squared Error")
    ax = plt.gca()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    # Build a comprehensive legend:
    # - Category colors: Train/Test/Extrapolation
    # - Model style: ref_label (solid), cmp_label (hatched)
    import matplotlib.patches as mpatches
    cat_handles = [
        mpatches.Patch(color=colors_cat[0], label="Train MSE"),
        mpatches.Patch(color=colors_cat[1], label="Test MSE"),
        mpatches.Patch(color=colors_cat[2], label="Extrapolation MSE"),
    ]
    style_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label=f"{ref_label}"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label=f"{cmp_label}"),
    ]
    handles = style_handles + cat_handles
    ax.legend(handles=handles, loc="best")

    # Annotate absolute numbers and percent delta
    max_height = max([0.0] + train_ref + test_ref + extra_ref + train_cmp + test_cmp + extra_cmp)
    for (b_ref, b_cmp), ref_vals, cmp_vals in zip(bars_all, groups_ref, groups_cmp):
        if not reverse_pct:
            # Default: annotate ref with values, cmp with values + pct (cmp vs ref)
            annotate_bars(b_ref)
            for j, rect in enumerate(b_cmp):
                v = rect.get_height()
                r = ref_vals[j]
                if r == 0:
                    pct = None
                    pct_str = "n/a"
                else:
                    pct = (v - r) / r * 100.0
                    pct_str = f"{pct:+.1f}%"
                plt.annotate(
                    f"{v:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2.0, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                    clip_on=False,
                )
                plt.annotate(
                    pct_str,
                    xy=(rect.get_x() + rect.get_width() / 2.0, v),
                    xytext=(0, 16),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=("#7f7f7f" if pct is None else ("#d62728" if pct > 0 else ("#2ca02c" if pct < 0 else "#7f7f7f"))),
                    clip_on=False,
                )
                # max_height computed globally above
        else:
            # Reverse: annotate cmp with values only, ref with values + pct (ref vs cmp)
            annotate_bars(b_cmp)
            for j, rect in enumerate(b_ref):
                r = rect.get_height()
                v = cmp_vals[j]
                if v == 0:
                    pct = None
                    pct_str = "n/a"
                else:
                    pct = (r - v) / v * 100.0
                    pct_str = f"{pct:+.1f}%"
                plt.annotate(
                    f"{r:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2.0, r),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                    clip_on=False,
                )
                plt.annotate(
                    pct_str,
                    xy=(rect.get_x() + rect.get_width() / 2.0, r),
                    xytext=(0, 16),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=("#7f7f7f" if pct is None else ("#d62728" if pct > 0 else ("#2ca02c" if pct < 0 else "#7f7f7f"))),
                    clip_on=False,
                )
                # max_height computed globally above

    # Expand top y-limit to ensure annotations are visible
    if max_height > 0:
        ax.set_ylim(top=max_height * 1.22)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_compare_three_models_noiseless(
    svr: Dict[str, Tuple[float, float, float]],
    gnn: Dict[str, Tuple[float, float, float]],
    mlp: Dict[str, Tuple[float, float, float]],
    out_path: str,
    title: Optional[str] = None,
) -> None:
    # Merge SVR, GNN, NN bars; percentages relative to SVR
    labels_all = list(svr.keys())
    labels = [k for k in labels_all if (k in gnn and k in mlp)]
    if not labels:
        return

    train_svr = [svr[k][0] for k in labels]
    test_svr = [svr[k][1] for k in labels]
    extra_svr = [svr[k][2] for k in labels]

    train_gnn = [gnn[k][0] for k in labels]
    test_gnn = [gnn[k][1] for k in labels]
    extra_gnn = [gnn[k][2] for k in labels]

    train_mlp = [mlp[k][0] for k in labels]
    test_mlp = [mlp[k][1] for k in labels]
    extra_mlp = [mlp[k][2] for k in labels]

    group_stride = 1.5 if len(labels) <= 3 else 1.35
    x = np.arange(len(labels), dtype=float) * group_stride
    # Three category centers per label
    width = 0.14
    spread = 1.10  # small gap between bars, avoids overlap
    # Ensure no overlap between edge bars of adjacent categories within a label
    cat_sep = width * (1.0 + 2.0 * spread) + 0.02
    cat_offsets = [-cat_sep, 0.0, cat_sep]
    model_offsets = [-width * spread, 0.0, width * spread]

    plt.figure(figsize=(11, 7.5))
    colors_cat = ["#1f77b4", "#5dade2", "#e69f00"]  # Train/Test/Extra
    model_hatches = [None, "//", "xx"]  # SVR, GNN, NN

    groups_svr = [train_svr, test_svr, extra_svr]
    groups_gnn = [train_gnn, test_gnn, extra_gnn]
    groups_mlp = [train_mlp, test_mlp, extra_mlp]

    # Draw bars: per category, draw SVR, then GNN, then NN
    for i in range(3):
        center = x + cat_offsets[i]
        # SVR (reference)
        b_svr = plt.bar(
            center + model_offsets[0],
            groups_svr[i],
            width=width,
            color=colors_cat[i],
            label=("SVR" if i == 0 else None),
        )
        # GNN
        b_gnn = plt.bar(
            center + model_offsets[1],
            groups_gnn[i],
            width=width,
            color=colors_cat[i],
            hatch=model_hatches[1],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label=("GNN" if i == 0 else None),
        )
        # NN (MLP)
        b_mlp = plt.bar(
            center + model_offsets[2],
            groups_mlp[i],
            width=width,
            color=colors_cat[i],
            hatch=model_hatches[2],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label=("NN" if i == 0 else None),
        )

        # Annotate absolute values and percentage vs SVR for GNN/NN
        for j in range(len(labels)):
            v_svr = groups_svr[i][j]
            v_gnn = groups_gnn[i][j]
            v_mlp = groups_mlp[i][j]

            # SVR value only
            rect_svr = b_svr[j]
            plt.annotate(
                f"{v_svr:.3f}",
                xy=(rect_svr.get_x() + rect_svr.get_width() / 2.0, v_svr),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

            # GNN value and % vs SVR
            rect_gnn = b_gnn[j]
            pct_gnn = None if v_svr == 0 else (v_gnn - v_svr) / v_svr * 100.0
            pct_gnn_str = "n/a" if pct_gnn is None else f"{pct_gnn:+.1f}%"
            plt.annotate(
                f"{v_gnn:.3f}",
                xy=(rect_gnn.get_x() + rect_gnn.get_width() / 2.0, v_gnn),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            plt.annotate(
                pct_gnn_str,
                xy=(rect_gnn.get_x() + rect_gnn.get_width() / 2.0, v_gnn),
                xytext=(0, 16),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=("#7f7f7f" if pct_gnn is None else ("#d62728" if pct_gnn > 0 else ("#2ca02c" if pct_gnn < 0 else "#7f7f7f"))),
            )

            # NN value and % vs SVR
            rect_mlp = b_mlp[j]
            pct_mlp = None if v_svr == 0 else (v_mlp - v_svr) / v_svr * 100.0
            pct_mlp_str = "n/a" if pct_mlp is None else f"{pct_mlp:+.1f}%"
            plt.annotate(
                f"{v_mlp:.3f}",
                xy=(rect_mlp.get_x() + rect_mlp.get_width() / 2.0, v_mlp),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            plt.annotate(
                pct_mlp_str,
                xy=(rect_mlp.get_x() + rect_mlp.get_width() / 2.0, v_mlp),
                xytext=(0, 16),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=("#7f7f7f" if pct_mlp is None else ("#d62728" if pct_mlp > 0 else ("#2ca02c" if pct_mlp < 0 else "#7f7f7f"))),
            )

    plt.xticks(x, labels)
    plt.ylabel("Mean Squared Error")
    if title:
        plt.title(title)
    ax = plt.gca()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    import matplotlib.patches as mpatches
    cat_handles = [
        mpatches.Patch(color=colors_cat[0], label="Train MSE"),
        mpatches.Patch(color=colors_cat[1], label="Test MSE"),
        mpatches.Patch(color=colors_cat[2], label="Extrapolation MSE"),
    ]
    style_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="SVR"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="GNN"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="xx", label="NN"),
    ]
    handles = style_handles + cat_handles
    ax.legend(handles=handles, loc="best")

    max_height = max(
        [0.0]
        + train_svr
        + test_svr
        + extra_svr
        + train_gnn
        + test_gnn
        + extra_gnn
        + train_mlp
        + test_mlp
        + extra_mlp
    )
    if max_height > 0:
        ax.set_ylim(top=max_height * 1.12)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    mlp_all = _load_pickle(MLP_PICKLE) or {}
    gnn_all = _load_pickle(GNN_PICKLE) or {}
    svr_all = _load_svr_all(SVR_PICKLE)

    # Experiments: qubits, depth (prefer noiseless if both exist)
    qubits_mlp = _fetch_experiment(mlp_all, ("qubits",))
    depth_mlp = _fetch_experiment(mlp_all, ("depth",))
    
    # SVR experiments keys
    qubits_svr = _fetch_svr_experiment(svr_all, ("qubits",))
    depth_svr = _fetch_svr_experiment(svr_all, ("depth",))

    # SVR vs GNN for noisy experiments
    qubits_gnn_noisy = _to_tuples_from_saved(gnn_all.get("qubits_noisy"))
    depth_gnn_noisy = _to_tuples_from_saved(gnn_all.get("depth_noisy"))
    qubits_svr_noisy = _fetch_svr_experiment(svr_all, ("qubits_noisy",))
    depth_svr_noisy = _fetch_svr_experiment(svr_all, ("depth_noisy",))
    if qubits_gnn_noisy and qubits_svr_noisy:
        plot_compare_groups(
            ref=qubits_gnn_noisy,
            cmp_=qubits_svr_noisy,
            ref_label="GNN",
            cmp_label="SVR",
            out_path=os.path.join(RESULTS_DIR, "compare_qubits_svr_vs_gnn_noisy.png"),
            reverse_pct=True,
            cmp_first=True,
        )

    # Merged noiseless plots: SVR, GNN, NN together, percentages vs SVR
    qubits_gnn_nl = _to_tuples_from_saved(gnn_all.get("qubits")) or _to_tuples_from_saved(gnn_all.get("qubits_noisy"))
    depth_gnn_nl = _to_tuples_from_saved(gnn_all.get("depth")) or _to_tuples_from_saved(gnn_all.get("depth_noisy"))
    qubits_svr_nl = qubits_svr
    depth_svr_nl = depth_svr

    if qubits_mlp and qubits_gnn_nl and qubits_svr_nl:
        plot_compare_three_models_noiseless(
            svr=qubits_svr_nl,
            gnn=qubits_gnn_nl,
            mlp=qubits_mlp,
            out_path=os.path.join(RESULTS_DIR, "compare_qubits_svr_gnn_nn.png"),
            title=None,
        )
    if depth_mlp and depth_gnn_nl and depth_svr_nl:
        plot_compare_three_models_noiseless(
            svr=depth_svr_nl,
            gnn=depth_gnn_nl,
            mlp=depth_mlp,
            out_path=os.path.join(RESULTS_DIR, "compare_depth_svr_gnn_nn.png"),
            title=None,
        )
    if depth_gnn_noisy and depth_svr_noisy:
        plot_compare_groups(
            ref=depth_gnn_noisy,
            cmp_=depth_svr_noisy,
            ref_label="GNN",
            cmp_label="SVR",
            out_path=os.path.join(RESULTS_DIR, "compare_depth_svr_vs_gnn_noisy.png"),
            reverse_pct=True,
            cmp_first=True,
        )


if __name__ == "__main__":
    main()


