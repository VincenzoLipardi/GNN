import os
import re
import pickle
import json
import argparse
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

try:
    from qiskit.qasm2 import loads as qasm2_loads
except Exception:
    qasm2_loads = None


# Expected datasets (filename, human label)
DATASETS: List[Tuple[str, str]] = [
    ("product_states_18.pkl", "Product States 18"),
    ("clifford_evolved_18.pkl", "Clifford-Evolved 18"),
    ("entangled_18.pkl", "Entangled 18"),
    #("product_states_2_10.pkl", "Product States 2-10")
    #("clifford_evolved_2_10.pkl", "Clifford Evolved 2-10"),
    #("entangled_2_10.pkl", "Entangled 2-10"),
    ("product_states_2_25.pkl", "Product States 2-25"),
    ("clifford_evolved_2_25.pkl", "Clifford-Evolved 2-25"),
    ("entangled_2_25.pkl", "Entangled 2-25"),
]



# Regex patterns to count only single-qubit rotations and CX gates in QASM
PATTERNS: Dict[str, re.Pattern[str]] = {
    "rx": re.compile(r"\brx\s*\(", re.IGNORECASE),
    "ry": re.compile(r"\bry\s*\(", re.IGNORECASE),
    "rz": re.compile(r"\brz\s*\(", re.IGNORECASE),
    # Strict word boundary to avoid matching e.g. rcx, ccx, etc.
    "cx": re.compile(r"\bcx\b", re.IGNORECASE),
}


def _circuit_from_qasm(qasm_str: str) -> QuantumCircuit:
    """Load a circuit from QASM compatible with Qiskit 1.x and older."""
    if qasm2_loads is not None:
        return qasm2_loads(qasm_str)
    return QuantumCircuit.from_qasm_str(qasm_str)


def _extract_qasm(item: object) -> str:
    """Return QASM string from a dataset item that is either (info, label) or info dict."""
    if isinstance(item, tuple):
        info = item[0]
    else:
        info = item
    if not isinstance(info, dict) or "qasm" not in info:
        raise ValueError("Unexpected dataset entry format; expected dict with 'qasm' or (dict, label)")
    return str(info["qasm"])


def count_gates_in_qasm(qasm: str) -> Tuple[int, int, int, int]:
    """Return counts of (rx, ry, rz, cx) using robust Qiskit parsing.

    Falls back to regex if circuit construction fails.
    """
    try:
        qc = _circuit_from_qasm(qasm)
        rx = ry = rz = cx = 0
        for ci in qc.data:
            instr = getattr(ci, "operation", None)
            name = getattr(instr, "name", "") if instr is not None else ""
            if name == "rx":
                rx += 1
            elif name == "ry":
                ry += 1
            elif name == "rz":
                rz += 1
            elif name == "cx":
                cx += 1
        return rx, ry, rz, cx
    except Exception:
        # Regex fallback in case parser is unavailable for some entries
        rx = len(PATTERNS["rx"].finditer(qasm))
        ry = len(PATTERNS["ry"].finditer(qasm))
        rz = len(PATTERNS["rz"].finditer(qasm))
        cx = len(PATTERNS["cx"].finditer(qasm))
        return rx, ry, rz, cx


def process_dataset(pkl_path: str) -> np.ndarray:
    """Load a PKL dataset and return an array of shape (N, 4) with per-circuit counts.

    Columns are ordered as [rx, ry, rz, cx].
    """
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    counts: List[Tuple[int, int, int, int]] = []
    rel_name = os.path.basename(pkl_path)
    for item in tqdm(data, desc=f"Gates {rel_name}", unit="circ"):
        try:
            qasm = _extract_qasm(item)
            counts.append(count_gates_in_qasm(qasm))
        except Exception:
            counts.append((0, 0, 0, 0))
    if not counts:
        return np.zeros((0, 4), dtype=int)
    return np.asarray(counts, dtype=int)


def _read_json(path: str) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"datasets": {}}


def _write_json(path: str, data: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _file_meta(path: str) -> Dict[str, float]:
    st = os.stat(path)
    return {"source_size": float(st.st_size), "source_mtime": float(st.st_mtime)}


def plot_gate_statistics(base_dir: str, out_path: str, json_path: Optional[str] = None) -> str:
    # Resolve existing datasets
    existing: List[Tuple[str, str]] = []
    for filename, label in DATASETS:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            existing.append((path, label))

    if not existing:
        raise FileNotFoundError("None of the expected dataset PKLs were found.")

    # Aggregate statistics (with optional JSON caching)
    labels: List[str] = []
    means: List[np.ndarray] = []  # each shape (4,)
    stds: List[np.ndarray] = []   # each shape (4,)
    results_json: Dict[str, object] = {"datasets": {}} if json_path is None else _read_json(json_path)
    datasets_json: Dict[str, object] = results_json.setdefault("datasets", {})
    ns: List[int] = []
    paths_list: List[str] = []

    for path, label in existing:
        meta = _file_meta(path)
        entry = datasets_json.get(label) if isinstance(datasets_json, dict) else None
        reuse = False
        if isinstance(entry, dict):
            try:
                if float(entry.get("source_mtime", -1.0)) == meta["source_mtime"] and float(entry.get("source_size", -1.0)) == meta["source_size"]:
                    reuse = True
            except Exception:
                reuse = False

        if reuse:
            mu_list = entry.get("means", {}) if isinstance(entry, dict) else {}
            sd_list = entry.get("stds", {}) if isinstance(entry, dict) else {}
            mu = np.array([
                float(mu_list.get("rx", 0.0)),
                float(mu_list.get("ry", 0.0)),
                float(mu_list.get("rz", 0.0)),
                float(mu_list.get("cx", 0.0)),
            ], dtype=float)
            sd = np.array([
                float(sd_list.get("rx", 0.0)),
                float(sd_list.get("ry", 0.0)),
                float(sd_list.get("rz", 0.0)),
                float(sd_list.get("cx", 0.0)),
            ], dtype=float)
            n = int(entry.get("n_circuits", 0)) if isinstance(entry, dict) else 0
        else:
            arr = process_dataset(path)  # (N, 4)
            if arr.size == 0:
                mu = np.zeros(4, dtype=float)
                sd = np.zeros(4, dtype=float)
                n = 0
            else:
                mu = np.mean(arr, axis=0)
                sd = np.std(arr, axis=0)
                n = int(arr.shape[0])

            datasets_json[label] = {
                "file": path,
                **meta,
                "n_circuits": int(n),
                "means": {"rx": float(mu[0]), "ry": float(mu[1]), "rz": float(mu[2]), "cx": float(mu[3])},
                "stds": {"rx": float(sd[0]), "ry": float(sd[1]), "rz": float(sd[2]), "cx": float(sd[3])},
            }

        labels.append(label)
        means.append(mu)
        stds.append(sd)
        ns.append(int(n))
        paths_list.append(path)

    # Optionally combine Magic and Stabilizer into a single 'Product states' aggregate
    try:
        idx_magic = labels.index("Magic")
        idx_stab = labels.index("Stabilizer")
    except ValueError:
        idx_magic = idx_stab = -1
    if idx_magic >= 0 and idx_stab >= 0:
        i1, i2 = sorted([idx_magic, idx_stab])
        N1, N2 = int(ns[i1]), int(ns[i2])
        mu1, mu2 = means[i1], means[i2]
        sd1, sd2 = stds[i1], stds[i2]
        if (N1 + N2) > 1:
            mu_comb = (mu1 * N1 + mu2 * N2) / float(N1 + N2)
            var_comb = (
                max(N1 - 1, 0) * (sd1 ** 2)
                + max(N2 - 1, 0) * (sd2 ** 2)
                + N1 * (mu1 - mu_comb) ** 2
                + N2 * (mu2 - mu_comb) ** 2
            ) / float(max(N1 + N2 - 1, 1))
            sd_comb = np.sqrt(var_comb)
        else:
            mu_comb = (mu1 + mu2) / 2.0
            sd_comb = np.sqrt((sd1 ** 2 + sd2 ** 2) / 2.0)
        keep = [k for k in range(len(labels)) if k not in (i1, i2)]
        labels = [labels[k] for k in keep]
        means = [means[k] for k in keep]
        stds = [stds[k] for k in keep]
        ns = [ns[k] for k in keep]
        sources = [paths_list[i1], paths_list[i2]]
        paths_list = [paths_list[k] for k in keep]
        insert_pos = i1
        labels.insert(insert_pos, "Product states")
        means.insert(insert_pos, mu_comb)
        stds.insert(insert_pos, sd_comb)
        ns.insert(insert_pos, int(N1 + N2))
        paths_list.insert(insert_pos, "+".join(sources))
        # Record aggregate in JSON
        datasets_json["Product states"] = {
            "file": "aggregate",
            "sources": sources,
            "n_circuits": int(N1 + N2),
            "means": {"rx": float(mu_comb[0]), "ry": float(mu_comb[1]), "rz": float(mu_comb[2]), "cx": float(mu_comb[3])},
            "stds": {"rx": float(sd_comb[0]), "ry": float(sd_comb[1]), "rz": float(sd_comb[2]), "cx": float(sd_comb[3])},
        }

    means_arr = np.vstack(means)  # (D, 4)
    stds_arr = np.vstack(stds)    # (D, 4)

    # Plot grouped bar chart
    gate_names = ["rx", "ry", "rz", "cx"]
    D = len(labels)
    G = len(gate_names)
    x = np.arange(D)
    width = 0.18
    offsets = (np.arange(G) - (G - 1) / 2.0) * (width + 0.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(12, 9))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for g, gate in enumerate(gate_names):
        means_g = means_arr[:, g]
        stds_g = stds_arr[:, g]
        lower = np.minimum(stds_g, means_g)
        upper = stds_g
        yerr = np.vstack([lower, upper])
        plt.bar(
            x + offsets[g],
            means_g,
            width=width,
            yerr=yerr,
            capsize=3,
            label=gate.upper(),
            color=colors[g % len(colors)],
            alpha=0.8,
            error_kw={"elinewidth": 1.0}
        )

    plt.xticks(x, labels, rotation=15, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Gate count per circuit", fontsize=16)
    plt.xlabel("Dataset", fontsize=16)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.legend(title="Gate Type", loc="best", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    if json_path is not None:
        _write_json(json_path, results_json)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot statistics of gate counts across datasets and save/update JSON summary.")
    parser.add_argument(
        "--base_dir",
        default="/data/P70087789/GNN/data/dataset_classification/dataset_type",
        help="Directory containing the dataset PKLs.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "gate_stats.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--json",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "gate_stats.json"),
        help="Path to write/update JSON summary with per-dataset stats.",
    )
    args = parser.parse_args()

    out = plot_gate_statistics(args.base_dir, args.out, args.json)
    print(f"Saved gate statistics plot to: {out}")
    print(f"Saved/updated JSON summary at: {args.json}")


if __name__ == "__main__":
    main()


