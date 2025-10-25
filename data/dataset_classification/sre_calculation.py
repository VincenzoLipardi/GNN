import os
import pickle
import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

try:
    from qiskit.qasm2 import loads as qasm2_loads  
except Exception:
    qasm2_loads = None  


def _extract_qasm_and_info(item: Any) -> Tuple[Dict[str, Any], str]:
    """Return (info_dict, qasm_str) for a dataset item.

    Items may be either a dict with key 'qasm' or a tuple of (dict, label).
    """
    if isinstance(item, tuple):
        info = item[0]
    else:
        info = item
    if not isinstance(info, dict) or "qasm" not in info:
        raise ValueError("Unexpected dataset entry format; expected dict with 'qasm' or (dict, label)")
    return info, str(info["qasm"])  # type: ignore[index]


def _circuit_from_qasm(qasm_str: str) -> QuantumCircuit:
    """Load a circuit from QASM compatible with Qiskit 1.x and older.

    Falls back to QuantumCircuit.from_qasm_str if qasm2 is unavailable.
    """
    if qasm2_loads is not None:
        return qasm2_loads(qasm_str)  # type: ignore[misc]
    return QuantumCircuit.from_qasm_str(qasm_str)


def _make_unitary_only_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of the input circuit with non-unitary ops removed.

    Removes: measure, barrier, reset, delay, snapshot, save/load instructions, classical operations.
    """
    non_unitary_names = {
        "measure",
        "barrier",
        "reset",
        "delay",
        "snapshot",
        "save_state",
        "save_statevector",
        "save_density_matrix",
        "save_probabilities",
        "save_amplitudes",
        "load_state",
    }
    cleaned = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    # Best-effort: try to drop final measurements via provided helper if present
    try:
        circuit = circuit.remove_final_measurements(inplace=False)  # type: ignore[attr-defined]
    except Exception:
        pass
    for instr, qargs, cargs in circuit.data:
        name = getattr(instr, "name", None)
        if name in non_unitary_names:
            continue
        # Skip any op that acts on classical bits
        if cargs:
            continue
        cleaned.append(instr, qargs, [])
    return cleaned


def _single_qubit_sre_from_xyz(x: float, y: float, z: float, alpha: float = 2.0) -> float:
    """Compute stabilizer Rényi entropy S_alpha for a single qubit from <X>,<Y>,<Z>.

    Uses the same formula as in data/label.py, specialized to n=1 (d=2):
      A = sum_P ((1/d) * [Tr(ρ P)]^2)^alpha, with P in {I, X, Y, Z}
      S_alpha(ρ) = (1/(1-α)) * log(A) - log(d)
    Here, Tr(ρ I) = 1 and Tr(ρ X) = <X>, etc.
    """
    d = 2.0
    inv_d_pow = (1.0 / d) ** alpha
    # Include identity term explicitly (Tr(ρ I) = 1)
    a_sum = 1.0 + (abs(x) ** (2.0 * alpha)) + (abs(y) ** (2.0 * alpha)) + (abs(z) ** (2.0 * alpha))
    A = inv_d_pow * a_sum
    # Guard against tiny numerical negatives inside log
    if A <= 0.0:
        # In extreme underflow, clamp to smallest positive float to keep log defined
        A = np.finfo(float).tiny
    entropy = (1.0 / (1.0 - alpha)) * float(np.log(A)) - float(np.log(d))
    return float(entropy)


def _per_qubit_expectations(psi: np.ndarray, num_qubits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-qubit <X>, <Y>, <Z> for a pure statevector psi of n qubits.

    psi: complex vector of length 2**n in Qiskit's basis ordering (little-endian).

    Returns three arrays of shape (n,) for expectations of X, Y, Z per qubit index k (0-based).
    """
    if psi.ndim != 1:
        raise ValueError("psi must be a 1-D statevector")
    if psi.size != (1 << num_qubits):
        raise ValueError("psi length does not match 2**num_qubits")

    # Reshape to (2,)*n so that axis (n-1-k) corresponds to qubit k (little-endian)
    psi_tensor = psi.reshape((2,) * num_qubits)

    x_vals = np.empty(num_qubits, dtype=float)
    y_vals = np.empty(num_qubits, dtype=float)
    z_vals = np.empty(num_qubits, dtype=float)

    for k in range(num_qubits):
        ax = num_qubits - 1 - k
        psi0 = np.take(psi_tensor, 0, axis=ax).reshape(-1)
        psi1 = np.take(psi_tensor, 1, axis=ax).reshape(-1)

        # Probabilities for Z expectation
        p0 = float(np.vdot(psi0, psi0).real)
        p1 = float(np.vdot(psi1, psi1).real)
        z = p0 - p1

        # Cross terms for X and Y expectations
        cross = np.vdot(psi0, psi1)  # sum conj(psi0) * psi1 over paired indices
        x = 2.0 * float(cross.real)
        y = 2.0 * float(cross.imag)

        # Clip to [-1, 1] to mitigate numerical drift
        x_vals[k] = float(np.clip(x, -1.0, 1.0))
        y_vals[k] = float(np.clip(y, -1.0, 1.0))
        z_vals[k] = float(np.clip(z, -1.0, 1.0))

    return x_vals, y_vals, z_vals


def calculate_per_qubit_sre(circuit: QuantumCircuit, alpha: float = 2.0) -> np.ndarray:
    """Compute SRE per qubit for a circuit by evaluating the pure statevector.

    This follows the stabilizer Rényi entropy definition used in data/label.py,
    but applies it independently on each qubit and returns the per-qubit values.
    """
    unitary_circuit = _make_unitary_only_circuit(circuit)
    num_qubits = unitary_circuit.num_qubits
    statevector = Statevector(unitary_circuit)
    psi = np.asarray(statevector.data, dtype=np.complex128)

    x_vals, y_vals, z_vals = _per_qubit_expectations(psi, num_qubits)
    per_qubit = np.empty(num_qubits, dtype=float)
    for k in range(num_qubits):
        per_qubit[k] = _single_qubit_sre_from_xyz(x_vals[k], y_vals[k], z_vals[k], alpha=alpha)
    return per_qubit


def process_dataset_file(
    filename: str,
    out_filename: Optional[str] = None,
    alpha: float = 2.0,
) -> str:
    """Compute per-circuit SRE statistics for all circuits in a PKL file.

    Saves a new PKL with list of (info_dict, stats_dict) where stats_dict contains only:
      - 'sre_total': sum of per-qubit SREs
      - 'sre_per_qubit': list of per-qubit SRE values
    Returns the output filename used.
    """
    with open(filename, "rb") as fh:
        data = pickle.load(fh)

    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    rel_name = os.path.relpath(filename)
    for item in tqdm(data, desc=f"SRE {os.path.basename(rel_name)}", unit="circ"):
        try:
            info, qasm = _extract_qasm_and_info(item)
            qc = _circuit_from_qasm(qasm)

            per_qubit = calculate_per_qubit_sre(qc, alpha=alpha)
            sre_total = float(np.nansum(per_qubit))

            stats = {
                "sre_total": sre_total,
                "sre_per_qubit": per_qubit.tolist(),
            }
        except Exception as exc:
            # Record NaNs on failure but preserve info
            try:
                info, _ = _extract_qasm_and_info(item)
            except Exception:
                info = {}
            per_qubit = np.full((int(info.get("num_qubits", 0)) or 0), np.nan, dtype=float)
            stats = {
                "sre_total": float("nan"),
                "sre_per_qubit": per_qubit.tolist(),
                # Keep error context minimal but available
                "error": str(exc),
            }

        results.append((info, stats))

    if out_filename is None:
        root, ext = os.path.splitext(filename)
        out_filename = f"{root}_sre_stats.pkl"

    with open(out_filename, "wb") as fh:
        pickle.dump(results, fh)

    return out_filename


def main(
    files: Optional[Iterable[str]] = None,
    alpha: float = 2.0,
) -> None:
    if files is None:
        files = [
            "/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_10.pkl",
]

    for f in files:
        out = process_dataset_file(f, alpha=alpha)
        print(f"Saved SRE stats to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-circuit Stabilizer Rényi Entropy (SRE) stats.")
    parser.add_argument(
        "--files",
        nargs="*",
        help="One or more PKL dataset files to process. If omitted, defaults to the four datasets.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Rényi order alpha to use (default: 2.0)",
    )
    args = parser.parse_args()

    main(files=args.files, alpha=args.alpha)


