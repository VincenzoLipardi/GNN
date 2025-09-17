import os
import sys
import pickle
import argparse
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

# Local import of estimator from noise_study.py (same directory)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from noise_study import estimate_stabilizer_renyi_entropy_on_backend  # noqa: E402

from qiskit import QuantumCircuit  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeOslo  # noqa: E402


def _load_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _save_pickle(file_path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def _iter_tim_files(input_dir: str, max_qubits: int, max_trotter: int, exact_qubits: Optional[int] = None) -> List[str]:
    files: List[str] = []
    if not os.path.isdir(input_dir):
        return files
    for name in os.listdir(input_dir):
        if not name.endswith(".pkl"):
            continue
        # Expect names like: ising_qubits_{n}_trotter_{t}.pkl
        parts = name.replace(".pkl", "").split("_")
        if len(parts) < 5:
            continue
        try:
            n_idx = parts.index("qubits") + 1
            t_idx = parts.index("trotter") + 1
            n_val = int(parts[n_idx])
            t_val = int(parts[t_idx])
        except Exception:
            continue
        qubits_ok = (n_val == int(exact_qubits)) if exact_qubits is not None else (n_val <= max_qubits)
        if qubits_ok and t_val <= max_trotter:
            files.append(os.path.join(input_dir, name))
    return sorted(files)


def _extract_qasm_and_info(item: Any) -> Tuple[Dict[str, Any], str]:
    # Dataset may contain either dicts with 'qasm' or tuples (dict, label)
    if isinstance(item, tuple):
        info = item[0]
    else:
        info = item
    if not isinstance(info, dict) or "qasm" not in info:
        raise ValueError("Unexpected dataset entry format; expected dict with 'qasm' key or (dict, label)")
    return info, info["qasm"]


def _to_quantum_circuit(qasm_str: str) -> QuantumCircuit:
    return QuantumCircuit.from_qasm_str(qasm_str)


def process_dataset_tim_noisy(
    input_dir: str,
    output_dir: str,
    max_qubits: int = 5,
    max_trotter: int = 3,
    alpha: int = 2,
    shots: int = 10000,
    exact_qubits: Optional[int] = None,
) -> None:
    """Create a noisy-labeled copy of dataset_tim using FakeOslo.

    - Reads files from input_dir that match constraints (qubits<=max_qubits and trotter<=max_trotter)
    - For each circuit entry, converts QASM to QuantumCircuit
    - Estimates SRE on FakeOslo via noise_study.estimate_stabilizer_renyi_entropy_on_backend
    - Writes tuples (circuit_dict, sre_noisy) to mirrored filenames under output_dir
    """
    backend = FakeOslo()

    target_files = _iter_tim_files(input_dir, max_qubits=max_qubits, max_trotter=max_trotter, exact_qubits=exact_qubits)
    if not target_files:
        if exact_qubits is not None:
            print(f"No matching files found in {input_dir} for qubits== {exact_qubits} and trotter<= {max_trotter}")
        else:
            print(f"No matching files found in {input_dir} for qubits<= {max_qubits} and trotter<= {max_trotter}")
        return

    for in_path in target_files:
        rel_name = os.path.basename(in_path)
        out_path = os.path.join(output_dir, rel_name)

        # Skip if already exists to avoid recomputation
        if os.path.exists(out_path):
            print(f"[skip] Output already exists: {out_path}")
            continue

        try:
            data = _load_pickle(in_path)
        except Exception as exc:
            print(f"[warn] Failed to load {in_path}: {exc}")
            continue

        if not data:
            print(f"[warn] Empty dataset in {in_path}; writing empty output")
            _save_pickle(out_path, [])
            continue

        results: List[Tuple[Dict[str, Any], float]] = []
        print(f"Processing {rel_name} with {len(data)} circuits ...")
        for item in tqdm(data, desc=f"{rel_name}", unit="circ"):
            try:
                circuit_info, qasm = _extract_qasm_and_info(item)
                qc = _to_quantum_circuit(qasm)
                sre_noisy = estimate_stabilizer_renyi_entropy_on_backend(
                    qc, backend=backend, alpha=alpha, shots=shots
                )
            except Exception as exc:
                print(f"  [warn] Failed on one circuit in {rel_name}: {exc}; recording NaN")
                sre_noisy = float("nan")
            results.append((circuit_info, float(sre_noisy)))

        _save_pickle(out_path, results)
        print(f"[done] Wrote noisy dataset to: {out_path}")


def _iter_random_files(
    input_dir: str,
    max_qubits: int,
    max_gates_end: int,
    exact_qubits: Optional[int] = None,
    min_qubits: int = 2,
    min_gates_start: int = 0,
) -> List[str]:
    files: List[str] = []
    if not os.path.isdir(input_dir):
        return files
    for name in os.listdir(input_dir):
        if not name.endswith(".pkl"):
            continue
        # Expect names like: basis_rotations+cx_qubits_{n}_gates_{a}-{b}.pkl
        try:
            parts = name.replace(".pkl", "").split("_")
            n_idx = parts.index("qubits") + 1
            g_idx = parts.index("gates") + 1
            n_val = int(parts[n_idx])
            gates_range = parts[g_idx]
            a_str, b_str = gates_range.split("-")
            a_val = int(a_str)
            b_val = int(b_str)
        except Exception:
            continue
        if exact_qubits is not None:
            qubits_ok = (n_val == int(exact_qubits))
        else:
            qubits_ok = (min_qubits <= n_val <= max_qubits)
        gates_ok = (a_val >= min_gates_start and b_val <= max_gates_end)
        if qubits_ok and gates_ok:
            files.append(os.path.join(input_dir, name))
    return sorted(files)


def process_dataset_random_noisy(
    input_dir: str,
    output_dir: str,
    max_qubits: int = 5,
    max_gates_end: int = 19,
    alpha: int = 2,
    shots: int = 10000,
    exact_qubits: Optional[int] = None,
    min_qubits: int = 2,
    min_gates_start: int = 0,
) -> None:
    """Create a noisy-labeled copy of dataset_random using FakeOslo.

    - Reads files from input_dir with qubits<=max_qubits and gates upper bound<=max_gates_end
    - For each circuit entry, converts QASM to QuantumCircuit
    - Estimates SRE on FakeOslo via noise_study.estimate_stabilizer_renyi_entropy_on_backend
    - Writes tuples (circuit_dict, sre_noisy) to mirrored filenames under output_dir
    """
    backend = FakeOslo()

    target_files = _iter_random_files(
        input_dir,
        max_qubits=max_qubits,
        max_gates_end=max_gates_end,
        exact_qubits=exact_qubits,
        min_qubits=min_qubits,
        min_gates_start=min_gates_start,
    )
    if not target_files:
        if exact_qubits is not None:
            print(
                f"No matching files found in {input_dir} for qubits== {exact_qubits} and gates {min_gates_start}-{max_gates_end}"
            )
        else:
            print(
                f"No matching files found in {input_dir} for qubits {min_qubits}-{max_qubits} and gates {min_gates_start}-{max_gates_end}"
            )
        return

    for in_path in target_files:
        rel_name = os.path.basename(in_path)
        out_path = os.path.join(output_dir, rel_name)

        if os.path.exists(out_path):
            print(f"[skip] Output already exists: {out_path}")
            continue

        try:
            data = _load_pickle(in_path)
        except Exception as exc:
            print(f"[warn] Failed to load {in_path}: {exc}")
            continue

        if not data:
            print(f"[warn] Empty dataset in {in_path}; writing empty output")
            _save_pickle(out_path, [])
            continue

        results: List[Tuple[Dict[str, Any], float]] = []
        print(f"Processing {rel_name} with {len(data)} circuits ...")
        total = len(data)
        # Save every ~5% of progress to avoid losing work on interruptions
        checkpoint_interval = max(1, int(total * 0.05))
        for idx, item in enumerate(tqdm(data, desc=f"{rel_name}", unit="circ")):
            try:
                circuit_info, qasm = _extract_qasm_and_info(item)
                qc = _to_quantum_circuit(qasm)
                sre_noisy = estimate_stabilizer_renyi_entropy_on_backend(
                    qc, backend=backend, alpha=alpha, shots=shots
                )
            except Exception as exc:
                print(f"  [warn] Failed on one circuit in {rel_name}: {exc}; recording NaN")
                sre_noisy = float("nan")
            results.append((circuit_info, float(sre_noisy)))
            # periodic checkpoint
            if (idx + 1) % checkpoint_interval == 0:
                tmp_path = out_path + ".partial"
                try:
                    _save_pickle(tmp_path, results)
                    print(f"  [checkpoint] Saved {idx + 1}/{total} to {tmp_path}")
                except Exception as exc:
                    print(f"  [warn] Failed to write checkpoint {tmp_path}: {exc}")

        _save_pickle(out_path, results)
        print(f"[done] Wrote noisy dataset to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create noisy-labeled datasets (TIM and/or Random)")
    parser.add_argument(
        "--which",
        choices=["random", "tim", "both"],
        default="random",
        help="Select which dataset(s) to process. Default: random",
    )
    args = parser.parse_args()

    base_dir = CURRENT_DIR

    if args.which in ("tim", "both"):
        input_dir_tim = os.path.join(base_dir, "dataset_tim")
        output_dir_tim = os.path.join(base_dir, "data", "dataset_tim_noisy")
        os.makedirs(output_dir_tim, exist_ok=True)
        process_dataset_tim_noisy(
            input_dir=input_dir_tim,
            output_dir=output_dir_tim,
            max_qubits=6,
            max_trotter=3,
            alpha=2,
            shots=10000,
            exact_qubits=6,
        )

    if args.which in ("random", "both"):
        random_input_dir = os.path.join(base_dir, "dataset_random")
        random_output_dir = os.path.join(base_dir, "data", "dataset_random_noisy")
        os.makedirs(random_output_dir, exist_ok=True)
        process_dataset_random_noisy(
            input_dir=random_input_dir,
            output_dir=random_output_dir,
            max_qubits=6,
            max_gates_end=39,
            alpha=2,
            shots=10000,
            exact_qubits=None,
            min_qubits=2,
            min_gates_start=20,
        )


if __name__ == "__main__":
    main()

