# GNN
Graph Neural Networks for Quantum Circuits

## Usage

### Requirements
- `torch` with CUDA if available
- `torch-geometric`
- `qiskit`

### Data
Label PKL files in `data/dataset_random` contain tuples: `(info_dict, label)` where `info_dict['qasm']` is the circuit QASM and `label` is the Stabilizer Rényi entropy.

### Train
Edit example paths in `models/gnn.py` under `__main__` or import `train`:

```python
from models.gnn import train

paths = [
  "/absolute/path/to/data/dataset_random/basis_rotations+cx_qubits_2_gates_0-19.pkl",
  "/absolute/path/to/data/dataset_random/basis_rotations+cx_qubits_3_gates_0-19.pkl",
]
train(paths, epochs=20, batch_size=64)
```

The graph encoder builds 12-dim node features (6 type one-hot + 6 qubit mask), constructs directed edges following gate order per wire, adds global features `[depth, #param, #qubits, #gates, rx, ry, rz, cx]`, then a GNN with three TransformerConv layers and a regressor predicts the entropy using Huber loss.
