import os
import re
import pickle
from typing import Dict, List, Optional, Sequence, Tuple
import math

import torch
from torch import Tensor

try:
    from torch_geometric.data import Data, InMemoryDataset
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "torch_geometric is required for graph representation. Please install 'torch-geometric'."
    ) from exc

from qiskit import QuantumCircuit


# ------------------------------
# Encoding configuration
# ------------------------------

# Default gate vocabulary used in earlier experiments
NODE_TYPES_BASE: List[str] = [
    "input",
    "measurement",
    "rx",
    "ry",
    "rz",
    "cx",
    "h",
    "x",
]

# Oslo-oriented vocabulary following IBM FakeOslo native-style gates
# Keep base gates for backward compatibility; add 'sx', 'x', 'ecr'
NODE_TYPES_OSLO: List[str] = [
    "input",
    "measurement",
    "rz",
    "sx",
    "x",
    "ecr",
    # Legacy/base gates to support mixed datasets
    "cx",
]


def get_node_types(node_feature_backend_variant: Optional[str]) -> List[str]:
    """Select gate vocabulary based on variant.

    Priority:
    - If the backend variant mentions 'oslo', use the Oslo-oriented set
    - Else, if env var GNN_NODE_TYPES_VARIANT == 'classification', use base minus 'x'
    - Else, use the base set
    """
    if node_feature_backend_variant and "oslo" in node_feature_backend_variant.lower():
        return NODE_TYPES_OSLO
    try:
        variant_env = str(os.environ.get("GNN_NODE_TYPES_VARIANT", "")).strip().lower()
        if variant_env == "classification":
            # Exclude 'x' from the base vocabulary for classification experiments
            return [t for t in NODE_TYPES_BASE if t != "x"]
    except Exception:
        pass
    return NODE_TYPES_BASE


# Derived dimensions
# Qubit mask dimension for node features (set to 25 to support up to 25-qubit circuits)
try:
    QUBIT_MASK_DIM: int = int(os.environ.get("GNN_QUBIT_MASK_DIM", "25"))
except Exception:
    QUBIT_MASK_DIM = 25

# When including backend properties, we append 7 values per node
# [q0_T1, q0_T2, q1_T1, q1_T2, gate_error, q0_readout_error, q1_readout_error]
NODE_FEATURE_BACKEND_EXTRA_DIM: int = 7


def get_node_feature_dim(node_feature_backend_variant: Optional[str]) -> int:
    node_types = get_node_types(node_feature_backend_variant)
    return len(node_types) + QUBIT_MASK_DIM + (NODE_FEATURE_BACKEND_EXTRA_DIM if node_feature_backend_variant else 0)


# Global feature variants
# - baseline: 8 dims [depth, num_param(rx+ry+rz), num_qubits, total_gates, rx, ry, rz, cx]
# - binned152: 152 dims [50 rx bins, 50 ry bins, 50 rz bins, h_count, cx_count]
# - oslo: 8 dims [depth, num_param(rz), num_qubits, total_gates, rz, sx, x, ecr]
GLOBAL_FEATURE_VARIANTS = {"baseline", "binned152", "oslo"}


def get_global_feature_dim(variant: str) -> int:
    v = (variant or "baseline").strip().lower()
    if v == "binned152":
        return 152
    if v == "oslo":
        # 50 bins for RZ + 3 counts (sx, x, ecr)
        return 53
    # baseline
    return 8


def _one_hot(index: int, size: int) -> Tensor:
    vector = torch.zeros(size, dtype=torch.float)
    vector[index] = 1.0
    return vector


def _encode_node_feature(
    node_type: str,
    num_qubits: int,
    active_qubits: Sequence[int],
    backend_feat: Optional[Tensor] = None,
    node_feature_backend_variant: Optional[str] = None,
) -> Tensor:
    """
    Build a node-feature vector:
    - First len(NODE_TYPES) dims: one-hot node type among NODE_TYPES
    - Next QUBIT_MASK_DIM dims: which qubits are targeted
    """
    vocab = get_node_types(node_feature_backend_variant)
    if node_type not in vocab:
        raise ValueError(f"Unsupported node type: {node_type}")

    type_one_hot = _one_hot(vocab.index(node_type), len(vocab))

    # Validate mask capacity for this circuit
    if num_qubits > QUBIT_MASK_DIM:
        raise ValueError(f"Qubit count {num_qubits} exceeds mask capacity {QUBIT_MASK_DIM}")
    qubit_mask = torch.zeros(QUBIT_MASK_DIM, dtype=torch.float)
    for q in active_qubits:
        if q < 0 or q >= QUBIT_MASK_DIM:
            raise ValueError(f"Qubit index out of range for {QUBIT_MASK_DIM}-qubit mask")
        qubit_mask[q] = 1.0

    base = torch.cat([type_one_hot, qubit_mask], dim=0)
    if backend_feat is not None:
        return torch.cat([base, backend_feat], dim=0)
    return base


class BackendFeatureProvider:
    """Provides per-node backend features for a given (fake) backend.

    Returns a 7-dim tensor per node: [q0_T1, q0_T2, q1_T1, q1_T2, gate_error, q0_readout_error, q1_readout_error].
    For single-qubit gates, second-qubit fields are zeros and gate_error refers to a representative 1q gate.
    """

    def __init__(self, backend_name: str = "fake_oslo"):
        self.backend_name = backend_name.lower()
        self._backend = None
        self._properties = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            if self.backend_name in ("fake_oslo", "oslo", "fake-oslo"):
                from qiskit_ibm_runtime.fake_provider import FakeOslo  # type: ignore
                self._backend = FakeOslo()
                self._properties = getattr(self._backend, "properties", lambda: None)()
        except Exception:
            self._backend = None
            self._properties = None

    def _get_qubit_val(self, qubit: int, prop_name: str) -> float:
        try:
            qubits = getattr(self._properties, "qubits", None)
            if qubits is None:
                return 0.0
            entries = qubits[int(qubit)]
            for nd in entries:
                if getattr(nd, "name", "").lower() == prop_name.lower():
                    val = getattr(nd, "value", None)
                    return float(val) if val is not None else 0.0
            # Fallbacks for readout error if available as probabilities
            if prop_name.lower() == "readout_error":
                p01 = self._get_qubit_val(qubit, "prob_meas1_prep0")
                p10 = self._get_qubit_val(qubit, "prob_meas0_prep1")
                if p01 > 0.0 or p10 > 0.0:
                    return float((p01 + p10) / 2.0)
        except Exception:
            return 0.0
        return 0.0

    def _get_gate_error(self, gate_name: str, qubits: Sequence[int]) -> float:
        """
        Attempt to retrieve a realistic gate_error for the requested operation/qubits.

        Robustness improvements over a naive lookup:
        - For 2q 'cx', also consider common native 2q gates like 'ecr', 'cz', 'iswap'.
        - Match both [q0, q1] and [q1, q0] ordering, and finally any unordered pair match.
        - If no exact 2q pair found, fall back to the average 2q gate_error over all pairs.
        - For 1q, try several common names ('sx','x','rz','u3','u') and prefer the first match on that qubit;
          if none exist, fall back to the average 1q gate_error over the device.
        """
        try:
            props = self._properties
            if props is None:
                return 0.0

            gates = getattr(props, "gates", [])
            if not gates:
                return 0.0

            def extract_gate_error(g) -> Optional[float]:
                for par in getattr(g, "parameters", []):
                    if getattr(par, "name", "").lower() == "gate_error":
                        val = getattr(par, "value", None)
                        try:
                            return float(val) if val is not None else None
                        except Exception:
                            return None
                return None

            gate_name_l = gate_name.lower() if gate_name else ""

            # Two-qubit case
            if len(qubits) == 2:
                q0, q1 = int(qubits[0]), int(qubits[1])
                # Candidate 2q names to consider
                twoq_names = [
                    gate_name_l if gate_name_l in ("cx", "cz", "ecr", "iswap") else "cx",
                    "ecr",
                    "cx",
                    "cz",
                    "iswap",
                ]
                # Try exact and reversed ordering for each candidate name
                for cand in twoq_names:
                    for qargs in ([q0, q1], [q1, q0]):
                        for g in gates:
                            try:
                                if getattr(g, "name", "").lower() != cand:
                                    continue
                                g_qubits = list(getattr(g, "qubits", []))
                                if g_qubits == list(qargs):
                                    err = extract_gate_error(g)
                                    if err is not None:
                                        return float(err)
                            except Exception:
                                continue

                # Try any unordered pair match regardless of direction
                target_set = {q0, q1}
                for cand in twoq_names:
                    for g in gates:
                        try:
                            if getattr(g, "name", "").lower() != cand:
                                continue
                            g_qubits = list(getattr(g, "qubits", []))
                            if len(g_qubits) == 2 and set(map(int, g_qubits)) == target_set:
                                err = extract_gate_error(g)
                                if err is not None:
                                    return float(err)
                        except Exception:
                            continue

                # Fallback: average over all 2q gate errors present
                vals: List[float] = []
                for g in gates:
                    try:
                        g_qubits = list(getattr(g, "qubits", []))
                        if len(g_qubits) != 2:
                            continue
                        err = extract_gate_error(g)
                        if err is not None:
                            vals.append(float(err))
                    except Exception:
                        continue
                if vals:
                    return float(sum(vals) / len(vals))
                return 0.0

            # One-qubit case
            q0 = int(qubits[0])
            oneq_candidates = ["sx", "x", "rz", "u3", "u"]
            # Try direct match for each name on this qubit
            for cand in oneq_candidates:
                for g in gates:
                    try:
                        if getattr(g, "name", "").lower() != cand:
                            continue
                        g_qubits = list(getattr(g, "qubits", []))
                        if g_qubits == [q0]:
                            err = extract_gate_error(g)
                            if err is not None:
                                return float(err)
                    except Exception:
                        continue
            # Fallback: average 1q gate error across device
            vals: List[float] = []
            for g in gates:
                try:
                    g_qubits = list(getattr(g, "qubits", []))
                    if len(g_qubits) != 1:
                        continue
                    err = extract_gate_error(g)
                    if err is not None:
                        vals.append(float(err))
                except Exception:
                    continue
            if vals:
                return float(sum(vals) / len(vals))
            return 0.0
        except Exception:
            return 0.0

    def features_for(self, gate_name: str, qubits: Sequence[int]) -> Tensor:
        if not qubits:
            return torch.zeros(NODE_FEATURE_BACKEND_EXTRA_DIM, dtype=torch.float)
        q0 = int(qubits[0])
        q1 = int(qubits[1]) if len(qubits) > 1 else None
        q0_t1 = self._get_qubit_val(q0, "T1")
        q0_t2 = self._get_qubit_val(q0, "T2")
        q1_t1 = self._get_qubit_val(q1, "T1") if q1 is not None else 0.0
        q1_t2 = self._get_qubit_val(q1, "T2") if q1 is not None else 0.0
        gate_err = self._get_gate_error(gate_name, [q0] + ([q1] if q1 is not None else []))
        q0_ro = self._get_qubit_val(q0, "readout_error")
        q1_ro = self._get_qubit_val(q1, "readout_error") if q1 is not None else 0.0
        return torch.tensor([q0_t1, q0_t2, q1_t1, q1_t2, gate_err, q0_ro, q1_ro], dtype=torch.float)


def _count_gates_qiskit(circuit: QuantumCircuit) -> Dict[str, int]:
    counts = {"rx": 0, "ry": 0, "rz": 0, "cx": 0, "h": 0}
    for instr in circuit.data:
        name = instr.operation.name.lower()
        if name in counts:
            counts[name] += 1
    return counts


def _count_gates_oslo(circuit: QuantumCircuit) -> Dict[str, int]:
    counts = {"rz": 0, "sx": 0, "x": 0, "ecr": 0}
    for instr in getattr(circuit, 'data', []):
        name = instr.operation.name.lower()
        if name in counts:
            counts[name] += 1
    return counts


def _global_features_baseline(circuit: QuantumCircuit, num_qubits: int) -> Tensor:
    gate_counts = _count_gates_qiskit(circuit)
    num_param = gate_counts["rx"] + gate_counts["ry"] + gate_counts["rz"]
    total_gates = sum(gate_counts.values())
    depth = float(circuit.depth()) if hasattr(circuit, "depth") else float(total_gates)

    return torch.tensor(
        [
            depth,
            float(num_param),
            float(num_qubits),
            float(total_gates),
            float(gate_counts["rx"]),
            float(gate_counts["ry"]),
            float(gate_counts["rz"]),
            float(gate_counts["cx"]),
        ],
        dtype=torch.float,
    )


def _safe_float_param(val) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _global_features_binned152(circuit: QuantumCircuit) -> Tensor:
    import math

    num_bins = 50
    bin_width = 2 * math.pi / num_bins

    def angle_to_bin(angle: float) -> int:
        # Normalize angle to [0, 2pi)
        a = angle % (2 * math.pi)
        idx = int(a // bin_width)
        if idx >= num_bins:
            idx = num_bins - 1
        return idx

    rx_bins = torch.zeros(num_bins, dtype=torch.float)
    ry_bins = torch.zeros(num_bins, dtype=torch.float)
    rz_bins = torch.zeros(num_bins, dtype=torch.float)
    h_count = 0.0
    cx_count = 0.0

    for instr in getattr(circuit, 'data', []):
        name = instr.operation.name.lower()
        if name in {"rx", "ry", "rz"}:
            params = getattr(instr.operation, 'params', [])
            if not params:
                continue
            val = _safe_float_param(params[0])
            if val is None:
                continue
            b = angle_to_bin(val)
            if name == "rx":
                rx_bins[b] += 1.0
            elif name == "ry":
                ry_bins[b] += 1.0
            else:
                rz_bins[b] += 1.0
        elif name == "h":
            h_count += 1.0
        elif name == "cx":
            cx_count += 1.0

    return torch.cat([rx_bins, ry_bins, rz_bins, torch.tensor([h_count, cx_count], dtype=torch.float)], dim=0)


def _global_features_oslo(circuit: QuantumCircuit, num_qubits: int) -> Tensor:
    import math

    num_bins = 50
    bin_width = 2 * math.pi / num_bins

    def angle_to_bin(angle: float) -> int:
        a = angle % (2 * math.pi)
        idx = int(a // bin_width)
        if idx >= num_bins:
            idx = num_bins - 1
        return idx

    rz_bins = torch.zeros(num_bins, dtype=torch.float)
    sx_count = 0.0
    x_count = 0.0
    ecr_count = 0.0

    for instr in getattr(circuit, 'data', []):
        name = instr.operation.name.lower()
        if name == "rz":
            params = getattr(instr.operation, 'params', [])
            if not params:
                continue
            val = _safe_float_param(params[0])
            if val is None:
                continue
            b = angle_to_bin(val)
            rz_bins[b] += 1.0
        elif name == "sx":
            sx_count += 1.0
        elif name == "x":
            x_count += 1.0
        elif name == "ecr":
            ecr_count += 1.0

    tail = torch.tensor([sx_count, x_count, ecr_count], dtype=torch.float)
    return torch.cat([rz_bins, tail], dim=0)


def qasm_to_pyg_graph(
    qasm_str: str,
    num_qubits_hint: Optional[int] = None,
    global_feature_variant: str = "baseline",
    backend_feature_provider: Optional[BackendFeatureProvider] = None,
) -> Tuple[Data, Dict[str, int]]:
    """
    Convert a QASM string to a PyG Data graph.

    Nodes: input per qubit, gate nodes, and output per qubit.
    Directed edges follow the temporal flow from previous node on a wire to
    the next gate/output node on that wire. Multi-qubit gates (cx) connect
    from each involved wire's last node into the shared gate node.

    Returns:
        (data, meta_counts) where data contains:
            - x: [num_nodes, node_feature_dim] where node_feature_dim = len(NODE_TYPES) + 7 (+7 if backend features enabled)
            - edge_index: [2, num_edges]
            - num_qubits: stored in data.num_qubits
            - global_features: Tensor[8] with
                [depth, num_param_gates, num_qubits, total_gates, rx, ry, rz, cx]
        meta_counts: dict of gate counts for convenience
    """
    circuit = QuantumCircuit.from_qasm_str(qasm_str)
    num_qubits = len(circuit.qubits)
    if num_qubits == 0:
        # Try to recover from malformed/trimmed QASM by reading qreg or using hint
        match = re.search(r"qreg\s+\w+\[(\d+)\];", qasm_str)
        if match:
            num_qubits = int(match.group(1))
        elif num_qubits_hint is not None:
            num_qubits = int(num_qubits_hint)

    # Nodes: start with one input node per qubit
    x_features: List[Tensor] = []
    edge_src: List[int] = []
    edge_dst: List[int] = []

    last_node_for_qubit: List[int] = []
    node_backend_variant = getattr(backend_feature_provider, 'backend_name', None)
    vocab = get_node_types(node_backend_variant)

    for q in range(num_qubits):
        idx = len(x_features)
        backend_feat = None
        if backend_feature_provider is not None:
            backend_feat = backend_feature_provider.features_for("input", [q])
        x_features.append(_encode_node_feature("input", num_qubits, [q], backend_feat, node_backend_variant))
        last_node_for_qubit.append(idx)

    # Helper: robustly get qubit index across Qiskit versions
    def _q_index(q) -> int:
        if hasattr(q, "index") and isinstance(getattr(q, "index"), int):
            return int(getattr(q, "index"))
        if hasattr(q, "_index") and isinstance(getattr(q, "_index"), int):
            return int(getattr(q, "_index"))
        # Fallback: parse trailing integer from string repr
        match = re.search(r"(,\s*)(\d+)(\))$", str(q))
        if match:
            return int(match.group(2))
        raise AttributeError("Unable to extract qubit index from Qiskit Qubit object")

    # Gate nodes
    for instr in getattr(circuit, 'data', []):
        op_name = instr.operation.name.lower()
        if op_name == "barrier":
            continue

        if op_name in {"rx", "ry", "rz", "h", "sx", "x"}:
            # Skip ops not present in the selected vocabulary (e.g., drop 'x' for classification)
            if op_name not in vocab:
                continue
            qubit = _q_index(instr.qubits[0])
            node_idx = len(x_features)
            backend_feat = None
            if backend_feature_provider is not None:
                backend_feat = backend_feature_provider.features_for(op_name, [qubit])
            x_features.append(_encode_node_feature(op_name, num_qubits, [qubit], backend_feat, node_backend_variant))
            # connect last node on this wire -> current gate node
            edge_src.append(last_node_for_qubit[qubit])
            edge_dst.append(node_idx)
            last_node_for_qubit[qubit] = node_idx

        elif op_name in {"cx", "ecr"}:
            control = _q_index(instr.qubits[0])
            target = _q_index(instr.qubits[1])
            node_idx = len(x_features)
            backend_feat = None
            if backend_feature_provider is not None:
                backend_feat = backend_feature_provider.features_for(op_name, [control, target])
            x_features.append(_encode_node_feature(op_name, num_qubits, [control, target], backend_feat, node_backend_variant))
            # incoming edges from both involved wires
            edge_src.extend([last_node_for_qubit[control], last_node_for_qubit[target]])
            edge_dst.extend([node_idx, node_idx])
            # this gate becomes the latest on both wires
            last_node_for_qubit[control] = node_idx
            last_node_for_qubit[target] = node_idx

        else:
            # Skip unsupported ops to keep representation consistent
            # (dataset uses rotations+cx)
            continue

    # Output nodes per qubit (only if there was at least an input node)
    if num_qubits > 0:
        for q in range(num_qubits):
            node_idx = len(x_features)
            backend_feat = None
            if backend_feature_provider is not None:
                backend_feat = backend_feature_provider.features_for("measurement", [q])
            x_features.append(_encode_node_feature("measurement", num_qubits, [q], backend_feat, node_backend_variant))
            edge_src.append(last_node_for_qubit[q])
            edge_dst.append(node_idx)
            last_node_for_qubit[q] = node_idx

    if not x_features:
        # Create minimal graph to avoid crashing; this should be rare.
        # Represent an empty circuit with no nodes/edges.
        vocab_len = len(get_node_types(node_backend_variant))
        node_dim = vocab_len + QUBIT_MASK_DIM + (NODE_FEATURE_BACKEND_EXTRA_DIM if backend_feature_provider is not None else 0)
        x = torch.zeros((0, node_dim), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        x = torch.stack(x_features, dim=0)
        # Sanitize in case any upstream property yielded non-finite numbers
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    # Global features
    if global_feature_variant not in GLOBAL_FEATURE_VARIANTS:
        raise ValueError(f"Unknown global_feature_variant: {global_feature_variant}")

    if global_feature_variant == "baseline":
        global_feat = _global_features_baseline(circuit, num_qubits)
    elif global_feature_variant == "binned152":
        global_feat = _global_features_binned152(circuit)
    else:  # "oslo"
        global_feat = _global_features_oslo(circuit, num_qubits)
    global_feat = torch.nan_to_num(global_feat, nan=0.0, posinf=0.0, neginf=0.0)

    # For compatibility with model/global_in_dim, add a small header when using compact variants
    # We follow the naming-based dimension via get_global_feature_dim in dataset file naming and in model code

    data = Data(x=x, edge_index=edge_index)
    data.num_qubits = num_qubits
    data.global_features = global_feat

    return data, _count_gates_qiskit(circuit)


class QuantumCircuitGraphDataset(InMemoryDataset):
    """
    Dataset that loads PKL files containing circuits and labels, converting
    each circuit QASM into a PyG graph with node/global features.

    Expected PKL formats per item:
    - Unlabeled: {"qasm": str, "num_qubits": int}
    - Labeled: ( {"qasm": str, "num_qubits": int}, label: float )
    """

    def __init__(
        self,
        root: str,
        pkl_paths: Optional[List[str]] = None,
        transform=None,
        pre_transform=None,
        global_feature_variant: str = "baseline",
        node_feature_backend_variant: Optional[str] = None,
    ):
        self.pkl_paths = pkl_paths
        self.global_feature_variant = global_feature_variant
        self.node_feature_backend_variant = node_feature_backend_variant
        super().__init__(root, transform, pre_transform)
        processed_path = self.processed_paths[0]
        try:
            # PyTorch >= 2.6 defaults to weights_only=True which breaks generic object loading
            self.data, self.slices = torch.load(processed_path, weights_only=False)
        except Exception:
            # Try adding safe globals for PyG types, then retry
            try:
                from torch.serialization import add_safe_globals, safe_globals  # type: ignore
                try:
                    from torch_geometric.data import Data  # type: ignore
                    from torch_geometric.data.data import DataEdgeAttr  # type: ignore
                    add_safe_globals([Data, DataEdgeAttr])
                except Exception:
                    pass
                with safe_globals([]):
                    self.data, self.slices = torch.load(processed_path)
            except Exception:
                # As a last resort, assume processed file is incompatible/corrupt; rebuild it
                try:
                    if os.path.exists(processed_path):
                        os.remove(processed_path)
                except Exception:
                    pass
                self.process()
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:  # not used by custom loader
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # Version the processed file by node and global feature dimensionality
        gdim = get_global_feature_dim(self.global_feature_variant)
        node_dim = get_node_feature_dim(self.node_feature_backend_variant)
        backend_tag = (self.node_feature_backend_variant or "none")
        return [f"graphs.node{node_dim}.gfeat{gdim}.{self.global_feature_variant}.backend_{backend_tag}.dataset.pt"]

    def download(self):  # pragma: no cover - no network
        return

    def _iter_items_from_pkls(self):
        if not self.pkl_paths:
            raise ValueError("pkl_paths must be provided with absolute paths to .pkl files")
        for pkl_path in self.pkl_paths:
            if not os.path.isabs(pkl_path):
                raise ValueError("Please provide absolute paths for reliability in tooling")
            with open(pkl_path, "rb") as f:
                content = pickle.load(f)
            for item in content:
                if isinstance(item, tuple) and len(item) == 2:
                    circ_info, label = item
                    yield circ_info, float(label)
                elif isinstance(item, dict):
                    yield item, None
                else:
                    # Unknown item format; skip
                    continue

    def process(self):
        data_list: List[Data] = []
        labels: List[float] = []

        # Initialize backend feature provider if requested
        backend_provider = None
        if self.node_feature_backend_variant:
            try:
                backend_provider = BackendFeatureProvider(self.node_feature_backend_variant)
            except Exception:
                backend_provider = None

        for circ_info, label in self._iter_items_from_pkls():
            qasm = circ_info["qasm"]
            graph, _ = qasm_to_pyg_graph(
                qasm,
                global_feature_variant=self.global_feature_variant,
                backend_feature_provider=backend_provider,
            )
            if label is not None:
                val = float(label)
                if not math.isfinite(val):
                    # Skip entries with NaN/Inf labels to avoid NaNs during training
                    continue
                graph.y = torch.tensor([val], dtype=torch.float)
            data_list.append(graph)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


# Convenience utility to quickly convert one QASM to features without creating a dataset file
def encode_single_qasm(
    qasm_str: str,
    label: Optional[float] = None,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: Optional[str] = None,
) -> Data:
    provider = BackendFeatureProvider(node_feature_backend_variant) if node_feature_backend_variant else None
    data, _ = qasm_to_pyg_graph(qasm_str, global_feature_variant=global_feature_variant, backend_feature_provider=provider)
    if label is not None:
        data.y = torch.tensor([float(label)], dtype=torch.float)
    return data


