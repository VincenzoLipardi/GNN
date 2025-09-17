import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# Optional heavy imports used by SRE computation helpers
try:
    from itertools import product as _it_product  # type: ignore
    from qiskit import QuantumCircuit as _QuantumCircuit  # type: ignore
    from qiskit.quantum_info import Operator as _Operator  # type: ignore
    try:
        # Qiskit Aer may not be installed in some envs
        from qiskit_aer import AerSimulator as _AerSimulator  # type: ignore
        from qiskit_aer.noise import NoiseModel as _NoiseModel  # type: ignore
        _HAS_AER = True
    except Exception:
        _AerSimulator = None  # type: ignore
        _NoiseModel = None  # type: ignore
        _HAS_AER = False
except Exception:
    _it_product = None  # type: ignore
    _QuantumCircuit = None  # type: ignore
    _Operator = None  # type: ignore
    _AerSimulator = None  # type: ignore
    _NoiseModel = None  # type: ignore
    _HAS_AER = False


def stabilizer_renyi_entropy_ideal(qc: "_QuantumCircuit", alpha: int = 2) -> float:
    """Compute SRE for an ideal (noiseless) circuit state.

    Falls back to a local implementation if helper in label.py is unavailable.
    """
    # Try reusing the implementation in label.py if present (absolute import; this
    # module is not a package, so relative import may fail)
    try:
        from label import calculate_stabilizer_renyi_entropy_qiskit as _calc_sre  # type: ignore
        return float(_calc_sre(qc, alpha=alpha))
    except Exception:
        try:
            # Best-effort relative import if run within a package context
            from .label import calculate_stabilizer_renyi_entropy_qiskit as _calc_sre  # type: ignore
            return float(_calc_sre(qc, alpha=alpha))
        except Exception:
            pass

    # Local fallback using statevector expectation of Pauli products
    if _QuantumCircuit is None or _Operator is None:
        raise RuntimeError("Qiskit not available to compute SRE")

    n = len(qc.qubits)
    d = float(2 ** n)
    single = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }
    # Build statevector once
    from qiskit.quantum_info import Statevector as _Statevector  # type: ignore
    sv = _Statevector(qc)
    A = 0.0
    for combo in _it_product(('I', 'X', 'Y', 'Z'), repeat=n):
        op = single[combo[0]]
        for c in combo[1:]:
            op = np.kron(op, single[c])
        exp_val = float((sv.data.conj().T @ (op @ sv.data)).real)
        xi_p = (1.0 / d) * (exp_val ** 2)
        A += xi_p ** alpha
    entropy = (1.0 / (1 - alpha)) * float(np.log(A)) - float(np.log(d))
    return float(entropy)


def estimate_stabilizer_renyi_entropy_on_backend(
    qc: "_QuantumCircuit",
    backend,
    alpha: int = 2,
    shots: int = 10000,
) -> float:
    """Estimate SRE using a noisy backend by simulating its noise model.

    Approach: simulate the circuit on a density-matrix Aer simulator configured
    from the provided backend (e.g., FakeOslo). Compute Tr(ρ P) over all Pauli
    products and plug into the SRE definition. For α=2 this is efficient for
    up to 6 qubits (4^n terms = 4096 at n=6).

    If Aer is unavailable, falls back to the ideal estimator.
    """
    if not _HAS_AER:
        # Fallback: no Aer available; return ideal value to avoid crashing
        return stabilizer_renyi_entropy_ideal(qc, alpha=alpha)

    # Strip final measurements to avoid collapse when computing the density matrix
    def _strip_measurements(c: "_QuantumCircuit") -> "_QuantumCircuit":
        new_circ = _QuantumCircuit(*c.qregs, *c.cregs, name=c.name)
        for instr, qargs, cargs in c.data:
            if instr.name in ("measure",):
                continue
            new_circ.append(instr, qargs, cargs)
        return new_circ

    n = len(qc.qubits)
    d = float(2 ** n)

    circ = _strip_measurements(qc)
    # Save density matrix for the full system (will be recognized by AerSimulator target)
    try:
        circ.save_density_matrix()
    except Exception:
        try:
            from qiskit_aer.library import save_density_matrix as _save_dm  # type: ignore
            circ.append(_save_dm(), [])
        except Exception:
            return stabilizer_renyi_entropy_ideal(qc, alpha=alpha)

    # Build an Aer simulator with the backend's noise model and density-matrix method
    try:
        noise_model = _NoiseModel.from_backend(backend) if _NoiseModel is not None else None
        sim = _AerSimulator(method='density_matrix', noise_model=noise_model)
    except Exception:
        # If we fail to configure noise/model, fallback to ideal
        return stabilizer_renyi_entropy_ideal(qc, alpha=alpha)

    from qiskit import transpile as _transpile  # type: ignore
    tcirc = _transpile(circ, sim)
    result = sim.run(tcirc, shots=1).result()
    try:
        rho = result.data(0)['density_matrix']
    except Exception:
        # As a fallback, try to use statevector and degrade gracefully
        return stabilizer_renyi_entropy_ideal(qc, alpha=alpha)

    # Ensure rho is ndarray
    rho = np.asarray(rho, dtype=complex)

    single = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }

    A = 0.0
    for combo in _it_product(('I', 'X', 'Y', 'Z'), repeat=n):
        op = single[combo[0]]
        for c in combo[1:]:
            op = np.kron(op, single[c])
        # Tr(ρ P)
        exp_val = float(np.trace(rho @ op).real)
        xi_p = (1.0 / d) * (exp_val ** 2)
        A += xi_p ** alpha

    entropy = (1.0 / (1 - alpha)) * float(np.log(A)) - float(np.log(d))
    return float(entropy)


def _list_common_pkls(dir_a: str, dir_b: str) -> List[str]:
    try:
        names_a = {n for n in os.listdir(dir_a) if n.endswith('.pkl')}
    except Exception:
        names_a = set()
    try:
        names_b = {n for n in os.listdir(dir_b) if n.endswith('.pkl')}
    except Exception:
        names_b = set()
    common = sorted(names_a.intersection(names_b))
    return common


def _extract_qubits_from_filename(name: str) -> Optional[int]:
    token = 'qubits_'
    try:
        idx = name.index(token) + len(token)
        digits: List[str] = []
        while idx < len(name) and name[idx].isdigit():
            digits.append(name[idx])
            idx += 1
        return int(''.join(digits)) if digits else None
    except Exception:
        return None


def _extract_labels_from_pkl(path: str) -> List[float]:
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return []
    labels: List[float] = []
    try:
        if isinstance(data, list) and data:
            # Common format: list of tuples (graph_or_feat, label)
            if isinstance(data[0], tuple) and len(data[0]) >= 2:
                labels = [float(x[1]) for x in data if isinstance(x, tuple) and len(x) >= 2]
            # Alternative: list of dicts with key 'label' or 'y'
            elif isinstance(data[0], dict):
                for item in data:
                    if 'label' in item:
                        labels.append(float(item['label']))
                    elif 'y' in item:
                        try:
                            labels.append(float(item['y']))
                        except Exception:
                            pass
    except Exception:
        labels = []
    return labels


def _load_labels_for_common_files(base_dir: str, other_dir: str, subset: Optional[List[str]] = None) -> Tuple[List[float], List[float], List[str]]:
    common = subset if subset is not None else _list_common_pkls(base_dir, other_dir)
    vals_a: List[float] = []
    vals_b: List[float] = []
    for name in common:
        vals_a.extend(_extract_labels_from_pkl(os.path.join(base_dir, name)))
        vals_b.extend(_extract_labels_from_pkl(os.path.join(other_dir, name)))
    return vals_a, vals_b, common


def _plot_overlay_hist(values_a: List[float], values_b: List[float], label_a: str, label_b: str, title: str, save_path: str):
    if not values_a or not values_b:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path + '.warn.txt', 'w') as f:
            f.write('Insufficient data to plot: ' + title + '\n')
        return
    all_vals = np.array(values_a + values_b, dtype=float)
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(all_vals)) - 1e-6
        vmax = float(np.nanmax(all_vals)) + 1e-6
    bins = np.linspace(vmin, vmax, 30)

    plt.figure(figsize=(8, 5))
    plt.hist(values_a, bins=bins, alpha=0.6, density=True, label=label_a, color="#1f77b4")
    plt.hist(values_b, bins=bins, alpha=0.6, density=True, label=label_b, color="#ff7f0e")

    arr_a = np.asarray(values_a, dtype=float)
    arr_b = np.asarray(values_b, dtype=float)
    arr_a = arr_a[np.isfinite(arr_a)]
    arr_b = arr_b[np.isfinite(arr_b)]
    mean_a = float(np.mean(arr_a)) if arr_a.size > 0 else None
    mean_b = float(np.mean(arr_b)) if arr_b.size > 0 else None

    ax = plt.gca()
    if mean_a is not None:
        plt.axvline(mean_a, color="#1f77b4", linestyle='--', linewidth=1.5, zorder=5)
    if mean_b is not None:
        plt.axvline(mean_b, color="#ff7f0e", linestyle='--', linewidth=1.5, zorder=6)
    ymax = ax.get_ylim()[1]
    if mean_a is not None:
        ax.text(mean_a, ymax * 0.98, f"{mean_a:.3f}", color="#1f77b4", ha="center", va="top")
    if mean_b is not None:
        ax.text(mean_b, ymax * 0.92, f"{mean_b:.3f}", color="#ff7f0e", ha="center", va="top")
    plt.xlabel('Stabilizer Rényi Entropy')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_per_qubit_overlays(
    dataset_name: str,
    noiseless_dir: str,
    noisy_dir: str,
    results_dir: str,
    qubits: Optional[List[int]] = None,
):
    os.makedirs(results_dir, exist_ok=True)
    common = _list_common_pkls(noiseless_dir, noisy_dir)
    # Group common files by qubit count
    qubit_to_files: dict[int, List[str]] = {}
    for fname in common:
        q = _extract_qubits_from_filename(fname)
        if q is None:
            continue
        qubit_to_files.setdefault(int(q), []).append(fname)
    if qubits is None:
        qubit_values = sorted(qubit_to_files.keys())
    else:
        qubit_values = [q for q in sorted(set(qubits)) if q in qubit_to_files]
    if not qubit_values:
        return

    # Single distribution (violin) figure across qubit counts
    fig, ax = plt.subplots(figsize=(8, 5))
    offset = 0.18
    width = 0.28
    any_plotted = False
    for q in qubit_values:
        subset = qubit_to_files[q]
        vals_a: List[float] = []
        vals_b: List[float] = []
        for fname in subset:
            vals_a.extend(_extract_labels_from_pkl(os.path.join(noiseless_dir, fname)))
            vals_b.extend(_extract_labels_from_pkl(os.path.join(noisy_dir, fname)))
        vals_a = [v for v in vals_a if np.isfinite(v)]
        vals_b = [v for v in vals_b if np.isfinite(v)]
        if vals_a:
            vp_a = ax.violinplot(vals_a, positions=[q - offset], widths=width, showextrema=False)
            for b in vp_a['bodies']:
                b.set_facecolor('#1f77b4')
                b.set_edgecolor('#1f77b4')
                b.set_alpha(1)
            # per-qubit mean marker and label
            mean_a = float(np.mean(vals_a))
            ax.scatter([q - offset], [mean_a], color='#1f77b4', s=36, zorder=12, edgecolors='black', linewidths=0.5)
            ax.annotate(f"{mean_a:.3f}", xy=(q - offset, mean_a), xytext=(0, 8), textcoords='offset points',
                        ha='center', va='bottom', color='#1f77b4', fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), zorder=13)
            any_plotted = True
        if vals_b:
            vp_b = ax.violinplot(vals_b, positions=[q + offset], widths=width, showextrema=False)
            for b in vp_b['bodies']:
                b.set_facecolor('#ff7f0e')
                b.set_edgecolor('#ff7f0e')
                b.set_alpha(1)
            # per-qubit mean marker and label
            mean_b = float(np.mean(vals_b))
            ax.scatter([q + offset], [mean_b], color='#ff7f0e', s=36, zorder=12, edgecolors='black', linewidths=0.5)
            ax.annotate(f"{mean_b:.3f}", xy=(q + offset, mean_b), xytext=(0, -10), textcoords='offset points',
                        ha='center', va='top', color='#ff7f0e', fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9), zorder=13)
            any_plotted = True

    ax.set_xlabel('Qubits')
    ax.set_ylabel('Stabilizer Rényi Entropy')
    ax.grid(True, axis='y', alpha=0.3)
    if any_plotted:
        handles = [Patch(facecolor='#1f77b4', edgecolor='#1f77b4', alpha=1, label='Noiseless'),
                   Patch(facecolor='#ff7f0e', edgecolor='#ff7f0e', alpha=1, label='Noisy')]
        ax.legend(handles=handles, frameon=False)
    ax.set_xticks(qubit_values)
    ax.set_xlim(min(qubit_values) - 0.75, max(qubit_values) + 0.75)

    plt.tight_layout()
    out_path = os.path.join(results_dir, f'distribution_{dataset_name}_per_qubit.png'.lower())
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f'Saved: {out_path}')


def compare_noiseless_vs_noisy(
    dataset_name: str,
    noiseless_dir: str,
    noisy_dir: str,
    results_dir: Optional[str] = None,
):
    if results_dir is None:
        results_dir = "/data/P70087789/GNN/models/results"
    os.makedirs(results_dir, exist_ok=True)
    common = _list_common_pkls(noiseless_dir, noisy_dir)
    vals_clean, vals_noisy, _ = _load_labels_for_common_files(noiseless_dir, noisy_dir, subset=common)
    fname = f"distribution_{dataset_name}_noiseless_vs_noisy.png"
    save_path = os.path.join(results_dir, fname.lower())
    title = f"{dataset_name}: Noiseless vs Noisy (matched {len(common)} files)"
    _plot_overlay_hist(vals_clean, vals_noisy, 'Noiseless', 'Noisy', title, save_path)
    print(f"Saved: {save_path} (N={len(vals_clean)} clean, {len(vals_noisy)} noisy across {len(common)} files)")


def run_default_noise_comparisons():
    base_data_dir = str(Path(__file__).resolve().parent)
    images_dir = "/data/P70087789/GNN/models/results"
    compare_noiseless_vs_noisy(
        dataset_name='Random',
        noiseless_dir=os.path.join(base_data_dir, 'dataset_random'),
        noisy_dir=os.path.join(base_data_dir, 'dataset_random_noisy'),
        results_dir=images_dir,
    )
    plot_per_qubit_overlays(
        dataset_name='Random',
        noiseless_dir=os.path.join(base_data_dir, 'dataset_random'),
        noisy_dir=os.path.join(base_data_dir, 'dataset_random_noisy'),
        results_dir=images_dir,
    )
    compare_noiseless_vs_noisy(
        dataset_name='TIM',
        noiseless_dir=os.path.join(base_data_dir, 'dataset_tim'),
        noisy_dir=os.path.join(base_data_dir, 'dataset_tim_noisy'),
        results_dir=images_dir,
    )
    plot_per_qubit_overlays(
        dataset_name='TIM',
        noiseless_dir=os.path.join(base_data_dir, 'dataset_tim'),
        noisy_dir=os.path.join(base_data_dir, 'dataset_tim_noisy'),
        results_dir=images_dir,
    )


if __name__ == "__main__":
    run_default_noise_comparisons()

