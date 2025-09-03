import os
import re
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

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

NODE_TYPES: List[str] = [
    "input",          # 0
    "measurement",    # 1
    "rx",             # 2
    "ry",             # 3
    "rz",             # 4
    "cx",             # 5
]


def _one_hot(index: int, size: int) -> Tensor:
    vector = torch.zeros(size, dtype=torch.float)
    vector[index] = 1.0
    return vector


def _encode_node_feature(node_type: str, num_qubits: int, active_qubits: Sequence[int]) -> Tensor:
    """
    Build a 12-dim node-feature vector:
    - First 6 dims: one-hot node type among NODE_TYPES
    - Next 6 dims: which qubits are targeted (dataset has up to 6 qubits)
    """
    if node_type not in NODE_TYPES:
        raise ValueError(f"Unsupported node type: {node_type}")

    type_one_hot = _one_hot(NODE_TYPES.index(node_type), len(NODE_TYPES))

    qubit_mask = torch.zeros(6, dtype=torch.float)
    for q in active_qubits:
        if q < 0 or q >= 6:
            raise ValueError("Qubit index out of range for 6-qubit mask")
        qubit_mask[q] = 1.0

    return torch.cat([type_one_hot, qubit_mask], dim=0)


def _count_gates_qiskit(circuit: QuantumCircuit) -> Dict[str, int]:
    counts = {"rx": 0, "ry": 0, "rz": 0, "cx": 0}
    for instr in circuit.data:
        name = instr.operation.name.lower()
        if name in counts:
            counts[name] += 1
    return counts


def qasm_to_pyg_graph(qasm_str: str, num_qubits_hint: Optional[int] = None) -> Tuple[Data, Dict[str, int]]:
    """
    Convert a QASM string to a PyG Data graph.

    Nodes: input per qubit, gate nodes, and output per qubit.
    Directed edges follow the temporal flow from previous node on a wire to
    the next gate/output node on that wire. Multi-qubit gates (cx) connect
    from each involved wire's last node into the shared gate node.

    Returns:
        (data, meta_counts) where data contains:
            - x: [num_nodes, 12]
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
    for q in range(num_qubits):
        idx = len(x_features)
        x_features.append(_encode_node_feature("input", num_qubits, [q]))
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

        if op_name in {"rx", "ry", "rz"}:
            qubit = _q_index(instr.qubits[0])
            node_idx = len(x_features)
            x_features.append(_encode_node_feature(op_name, num_qubits, [qubit]))
            # connect last node on this wire -> current gate node
            edge_src.append(last_node_for_qubit[qubit])
            edge_dst.append(node_idx)
            last_node_for_qubit[qubit] = node_idx

        elif op_name == "cx":
            control = _q_index(instr.qubits[0])
            target = _q_index(instr.qubits[1])
            node_idx = len(x_features)
            x_features.append(_encode_node_feature("cx", num_qubits, [control, target]))
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
            x_features.append(_encode_node_feature("measurement", num_qubits, [q]))
            edge_src.append(last_node_for_qubit[q])
            edge_dst.append(node_idx)
            last_node_for_qubit[q] = node_idx

    if not x_features:
        # Create minimal graph to avoid crashing; this should be rare.
        # Represent an empty circuit with no nodes/edges.
        x = torch.zeros((0, 12), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        x = torch.stack(x_features, dim=0)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    # Global features
    gate_counts = _count_gates_qiskit(circuit)
    num_param = gate_counts["rx"] + gate_counts["ry"] + gate_counts["rz"]
    total_gates = sum(gate_counts.values())
    depth = float(circuit.depth()) if hasattr(circuit, "depth") else float(total_gates)

    global_feat = torch.tensor(
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

    data = Data(x=x, edge_index=edge_index)
    data.num_qubits = num_qubits
    data.global_features = global_feat

    return data, gate_counts


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
    ):
        self.pkl_paths = pkl_paths
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:  # not used by custom loader
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["graphs.dataset.pt"]

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

        for circ_info, label in self._iter_items_from_pkls():
            qasm = circ_info["qasm"]
            graph, _ = qasm_to_pyg_graph(qasm)
            if label is not None:
                graph.y = torch.tensor([float(label)], dtype=torch.float)
            data_list.append(graph)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


# Convenience utility to quickly convert one QASM to features without creating a dataset file
def encode_single_qasm(qasm_str: str, label: Optional[float] = None) -> Data:
    data, _ = qasm_to_pyg_graph(qasm_str)
    if label is not None:
        data.y = torch.tensor([float(label)], dtype=torch.float)
    return data


