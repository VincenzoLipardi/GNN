import argparse
import sys
from typing import Optional, List, Dict, Any

try:
    from qiskit_ibm_runtime.fake_provider import FakeOslo
except Exception as exc:  # pragma: no cover - environment dependent
    FakeOslo = None  # type: ignore

try:
    from qiskit import QuantumCircuit, transpile
except Exception:  # pragma: no cover - environment dependent
    QuantumCircuit = None  # type: ignore
    transpile = None  # type: ignore


def demo_transpile() -> None:
    if FakeOslo is None:
        raise RuntimeError(
            "qiskit_ibm_runtime.fake_provider.FakeOslo is not available. Install qiskit-ibm-runtime."
        )
    if QuantumCircuit is None or transpile is None:
        raise RuntimeError("qiskit is required for the transpilation demo.")

    backend = FakeOslo()

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rx(0.3, 0)
    qc.ry(0.5, 1)
    qc.rz(0.7, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    print("\nOriginal circuit:")
    print(qc)

    tqc = transpile(qc, backend=backend, optimization_level=3)
    print("\nTranspiled circuit for FakeOslo:")
    print(tqc)

    used_gates = sorted(tqc.count_ops().keys())
    print(f"\nUsed gates: {used_gates}")


def _summarize_gate_noise(backend: Any) -> Dict[str, float]:
    props = None
    try:
        props = getattr(backend, "properties", None)
        if callable(props):
            props = props()
    except Exception:
        props = None
    if props is None:
        return {}

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    gates_list = getattr(props, "gates", None)
    if not gates_list:
        return {}
    for gate in gates_list:
        try:
            gate_name = gate.get("gate") if isinstance(gate, dict) else getattr(gate, "gate", None)
            parameters = gate.get("parameters") if isinstance(gate, dict) else getattr(gate, "parameters", None)
            if gate_name is None or not isinstance(parameters, list):
                continue
            err_val = None
            for p in parameters:
                name = p.get("name") if isinstance(p, dict) else getattr(p, "name", None)
                value = p.get("value") if isinstance(p, dict) else getattr(p, "value", None)
                if name == "gate_error":
                    try:
                        err_val = float(value)
                    except Exception:
                        err_val = None
                    break
            if err_val is None:
                continue
            totals[gate_name] = totals.get(gate_name, 0.0) + err_val
            counts[gate_name] = counts.get(gate_name, 0) + 1
        except Exception:
            continue
    return {k: (totals[k] / counts[k]) for k in totals.keys() if counts.get(k)}


def print_gate_noise() -> None:
    if FakeOslo is None:
        raise RuntimeError(
            "qiskit_ibm_runtime.fake_provider.FakeOslo is not available. Install qiskit-ibm-runtime."
        )
    backend = FakeOslo()
    basis = getattr(backend, "basis_gates", None)
    if callable(basis):
        try:
            basis = basis()
        except Exception:
            basis = None
    basis_list = list(basis) if isinstance(basis, (list, tuple, set)) else []

    avg_errors = _summarize_gate_noise(backend)
    print("\n=== Average gate_error per native gate (FakeOslo) ===")
    if not basis_list and not avg_errors:
        print("No data available.")
        return
    printed = set()
    for g in basis_list:
        val = avg_errors.get(g)
        if val is None:
            print(f"{g}: N/A")
        else:
            print(f"{g}: {val}")
        printed.add(g)
    for g, val in avg_errors.items():
        if g not in printed:
            print(f"{g}: {val}")

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Utilities for Qiskit's FakeOslo: transpilation demo and gate noise summary."
    )
    parser.add_argument(
        "--demo-transpile",
        action="store_true",
        help="Run the transpilation demo. If omitted, the demo still runs.",
    )
    parser.add_argument(
        "--gate-noise",
        action="store_true",
        help="Print average gate_error per native gate.",
    )
    args = parser.parse_args(argv)

    try:
        demo_transpile()
        if args.gate_noise:
            print_gate_noise()
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


