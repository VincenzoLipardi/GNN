import os
import sys
import pickle
from typing import Dict, List, Tuple, Optional


def _load_pickle(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _iter_pkls_in(dir_path: str) -> List[str]:
    try:
        names = [n for n in os.listdir(dir_path) if n.endswith('.pkl')]
    except Exception:
        return []
    return sorted(os.path.join(dir_path, n) for n in names)


def _extract_qasm(item) -> Optional[str]:
    try:
        if isinstance(item, tuple) and len(item) >= 1 and isinstance(item[0], dict):
            return item[0].get("qasm")
        if isinstance(item, dict):
            return item.get("qasm")
    except Exception:
        return None
    return None


class FakeOsloPropertyAuditor:
    """Audits presence of backend properties used by BackendFeatureProvider.

    Fields audited per node:
      - q0_T1, q0_T2, q0_readout_error
      - q1_T1, q1_T2, q1_readout_error (only for 2-qubit gates)
      - gate_error (1q representative 'sx' for single-qubit nodes; 'cx' for 2q)
    """

    def __init__(self):
        from qiskit_ibm_runtime.fake_provider import FakeOslo  # type: ignore
        self.backend = FakeOslo()
        try:
            self.properties = self.backend.properties()
        except Exception:
            self.properties = None

        # counters
        self.counts: Dict[str, int] = {
            # totals
            "checks_q0_T1": 0,
            "checks_q0_T2": 0,
            "checks_q0_RO": 0,
            "checks_q1_T1": 0,
            "checks_q1_T2": 0,
            "checks_q1_RO": 0,
            "checks_gate_error_1q": 0,
            "checks_gate_error_cx": 0,
            # missing
            "miss_q0_T1": 0,
            "miss_q0_T2": 0,
            "miss_q0_RO": 0,
            "miss_q1_T1": 0,
            "miss_q1_T2": 0,
            "miss_q1_RO": 0,
            "miss_gate_error_1q": 0,
            "miss_gate_error_cx": 0,
        }

    def _has_qubit_prop(self, qubit: int, name: str) -> bool:
        try:
            qubits = getattr(self.properties, "qubits", None)
            if qubits is None:
                return False
            entries = qubits[int(qubit)]
            for nd in entries:
                if getattr(nd, "name", "").lower() == name.lower():
                    return getattr(nd, "value", None) is not None
            return False
        except Exception:
            return False

    def _has_readout_error(self, qubit: int) -> bool:
        if self._has_qubit_prop(qubit, "readout_error"):
            return True
        # fallback combo
        p01 = self._has_qubit_prop(qubit, "prob_meas1_prep0")
        p10 = self._has_qubit_prop(qubit, "prob_meas0_prep1")
        return bool(p01 or p10)

    def _has_gate_error_single_qubit(self, qubit: int) -> bool:
        # Replicate provider: choose 'sx' and do not try alternatives
        target_name = "sx"
        qargs = [int(qubit)]
        try:
            gates = getattr(self.properties, "gates", [])
            for g in gates:
                if getattr(g, "name", "").lower() != target_name:
                    continue
                if list(getattr(g, "qubits", [])) != qargs:
                    continue
                for par in getattr(g, "parameters", []):
                    if getattr(par, "name", "").lower() == "gate_error":
                        return getattr(par, "value", None) is not None
        except Exception:
            return False
        return False

    def _has_gate_error_cx(self, control: int, target: int) -> bool:
        try:
            gates = getattr(self.properties, "gates", [])
            for g in gates:
                if getattr(g, "name", "").lower() != "cx":
                    continue
                if list(getattr(g, "qubits", [])) != [int(control), int(target)]:
                    continue
                for par in getattr(g, "parameters", []):
                    if getattr(par, "name", "").lower() == "gate_error":
                        return getattr(par, "value", None) is not None
        except Exception:
            return False
        return False

    def audit_circuit(self, qc) -> None:
        # Per-qubit input/measurement nodes: check q0 props and 1q gate_error on each qubit
        n = len(qc.qubits)
        for q in range(n):
            self.counts["checks_q0_T1"] += 1
            self.counts["checks_q0_T2"] += 1
            self.counts["checks_q0_RO"] += 1
            self.counts["checks_gate_error_1q"] += 1
            if not self._has_qubit_prop(q, "T1"):
                self.counts["miss_q0_T1"] += 1
            if not self._has_qubit_prop(q, "T2"):
                self.counts["miss_q0_T2"] += 1
            if not self._has_readout_error(q):
                self.counts["miss_q0_RO"] += 1
            if not self._has_gate_error_single_qubit(q):
                self.counts["miss_gate_error_1q"] += 1

        # Gate nodes
        for instr in getattr(qc, "data", []):
            name = getattr(instr, "operation", getattr(instr, "instr", None))
            op_name = getattr(name, "name", str(name)).lower()
            if op_name in ("barrier",):
                continue
            qargs = [int(getattr(q, "index", getattr(q, "_index", None))) if hasattr(q, "index") or hasattr(q, "_index") else int(str(q).split(",")[-1].rstrip(")")) for q in instr.qubits]

            if op_name in ("rx", "ry", "rz", "h", "input", "measurement"):
                q = qargs[0]
                self.counts["checks_q0_T1"] += 1
                self.counts["checks_q0_T2"] += 1
                self.counts["checks_q0_RO"] += 1
                self.counts["checks_gate_error_1q"] += 1
                if not self._has_qubit_prop(q, "T1"):
                    self.counts["miss_q0_T1"] += 1
                if not self._has_qubit_prop(q, "T2"):
                    self.counts["miss_q0_T2"] += 1
                if not self._has_readout_error(q):
                    self.counts["miss_q0_RO"] += 1
                if not self._has_gate_error_single_qubit(q):
                    self.counts["miss_gate_error_1q"] += 1

            elif op_name == "cx":
                c, t = qargs[0], qargs[1]
                # q0
                self.counts["checks_q0_T1"] += 1
                self.counts["checks_q0_T2"] += 1
                self.counts["checks_q0_RO"] += 1
                if not self._has_qubit_prop(c, "T1"):
                    self.counts["miss_q0_T1"] += 1
                if not self._has_qubit_prop(c, "T2"):
                    self.counts["miss_q0_T2"] += 1
                if not self._has_readout_error(c):
                    self.counts["miss_q0_RO"] += 1
                # q1
                self.counts["checks_q1_T1"] += 1
                self.counts["checks_q1_T2"] += 1
                self.counts["checks_q1_RO"] += 1
                if not self._has_qubit_prop(t, "T1"):
                    self.counts["miss_q1_T1"] += 1
                if not self._has_qubit_prop(t, "T2"):
                    self.counts["miss_q1_T2"] += 1
                if not self._has_readout_error(t):
                    self.counts["miss_q1_RO"] += 1
                # gate error for the pair
                self.counts["checks_gate_error_cx"] += 1
                if not self._has_gate_error_cx(c, t):
                    self.counts["miss_gate_error_cx"] += 1


def _summarize(prefix: str, counts: Dict[str, int]) -> List[str]:
    def pct(miss: int, total: int) -> str:
        if total <= 0:
            return "n/a"
        return f"{(miss/total)*100:.1f}%"

    lines: List[str] = []
    def add(field: str, total_key: str, miss_key: str):
        tot = counts.get(total_key, 0)
        mis = counts.get(miss_key, 0)
        lines.append(f"{prefix}{field}: {mis}/{tot} missing ({pct(mis, tot)})")

    add("q0_T1", "checks_q0_T1", "miss_q0_T1")
    add("q0_T2", "checks_q0_T2", "miss_q0_T2")
    add("q0_readout_error", "checks_q0_RO", "miss_q0_RO")
    add("q1_T1", "checks_q1_T1", "miss_q1_T1")
    add("q1_T2", "checks_q1_T2", "miss_q1_T2")
    add("q1_readout_error", "checks_q1_RO", "miss_q1_RO")
    add("gate_error_1q(sx)", "checks_gate_error_1q", "miss_gate_error_1q")
    add("gate_error_cx(pair)", "checks_gate_error_cx", "miss_gate_error_cx")
    return lines


def main():
    base = "/data/P70087789/GNN/data"
    dirs = [
        os.path.join(base, "dataset_random"),
        os.path.join(base, "dataset_tim"),
    ]
    # Allow overriding via args
    if len(sys.argv) > 1:
        dirs = sys.argv[1:]

    try:
        from qiskit import QuantumCircuit  # type: ignore
    except Exception as exc:
        print(f"[ERROR] Qiskit not available: {exc}")
        sys.exit(1)

    auditor = FakeOsloPropertyAuditor()

    num_files = 0
    num_circuits = 0
    for d in dirs:
        pkls = _iter_pkls_in(d)
        for pkl_path in pkls:
            try:
                data = _load_pickle(pkl_path)
            except Exception as exc:
                print(f"[warn] Failed to load {pkl_path}: {exc}")
                continue
            num_files += 1
            for item in data:
                qasm = _extract_qasm(item)
                if not qasm:
                    continue
                try:
                    qc = QuantumCircuit.from_qasm_str(qasm)
                except Exception:
                    continue
                auditor.audit_circuit(qc)
                num_circuits += 1

    print(f"Audited files: {num_files} | circuits: {num_circuits}")
    lines = _summarize("", auditor.counts)
    print("\n".join(lines))


if __name__ == "__main__":
    main()


