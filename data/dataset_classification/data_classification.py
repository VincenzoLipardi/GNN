import os
import math
import random
import pickle
from typing import List, Tuple, Dict, Any, Optional
from qiskit import QuantumCircuit
try:
    from qiskit.qasm2 import dumps as qasm2_dumps 
    from qiskit.qasm2 import loads as qasm2_loads  
except Exception:  
    qasm2_dumps = None
    qasm2_loads = None
try:
    from qiskit.circuit.library import CXGate  # type: ignore
    from qiskit.circuit import CircuitInstruction  # type: ignore
except Exception:
    CXGate = None  # type: ignore
    CircuitInstruction = None  # type: ignore


# Generation controls
MIN_QUBITS = 2
MAX_QUBITS = 10
NUM_PER_COUNT = 2000
STABILIZER_LABEL = 0
MAGIC_LABEL = 1

GENERATE_CLASS = True
GENERATE_EVOLVED = True
GENERATE_ENTANGLED = True

# Absolute directory for saving all datasets
DATASET_TYPE_DIR = "/data/P70087789/GNN/data/dataset_classification/dataset_type"



# Gate budget controls
MAX_TOTAL_GATES = 50


def count_primitive_gates(qc: QuantumCircuit) -> int:
    """Return the number of primitive operations currently in the circuit."""
    return len(qc.data)


def remaining_gate_budget(qc: QuantumCircuit, max_total_gates: Optional[int] = None) -> int:
    """Return how many more primitive gates can be added without exceeding the budget.

    If max_total_gates is None, treat the budget as unlimited (returns a very large number).
    """
    used = count_primitive_gates(qc)
    if max_total_gates is None:
        # Effectively unlimited budget
        return max(0, 100 - used)
    limit = int(max_total_gates)
    return max(0, limit - used)


def _qubit_has_prior_gate(qc: QuantumCircuit, qubit_index: int) -> bool:
    """Return True if the given qubit has any prior instruction in the circuit."""
    
    target_qubit = qc.qubits[qubit_index]
    for ci in qc.data:
        if target_qubit in ci.qubits:
            return True
    return False


def apply_stabilizer_gates(
    qc: QuantumCircuit,
    num_qubits: int,
    min_gates: int = 10,
    max_gates: int = 50,
    target_qubit = None,
    max_total_gates: Optional[int] = None,
) -> None:
    """Build a stabilizer product-state using stabilizer gates only.

    Allowed gates (randomly sampled per step):
      - RX/RY/RZ with angles in {pi/2, pi, 3pi/2}
      - H via composite: Ry(pi/2) then Rz(pi)
      - S via rotation: Rz(pi/2)

    Procedure:
      - Sample total number of gates G ~ Uniform{min_gates..max_gates}.
      - For each of G steps, choose the target qubit (random unless target_qubit
        is provided), then pick a random gate from the allowed set and apply it.
    """
    angles = (math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    total_gates = random.randint(int(min_gates), int(max_gates))
    for _ in range(total_gates):
        if remaining_gate_budget(qc, max_total_gates) <= 0:
            break
        q = int(target_qubit) if target_qubit is not None else random.randrange(num_qubits)
        is_first_on_q = not _qubit_has_prior_gate(qc, q)
        # Avoid RZ (and S which is Rz(pi/2)) as the first gate on a qubit
        gate_choices = ("rx", "ry", "rz", "h", "s")
        if is_first_on_q:
            gate_choices = ("rx", "ry", "h")

        def gate_cost(kind: str) -> int:
            return 2 if kind == "h" else 1

        remaining = remaining_gate_budget(qc, max_total_gates)
        allowed = [g for g in gate_choices if gate_cost(g) <= remaining and not (is_first_on_q and g == "rz")]
        if not allowed:
            break
        gate = random.choice(allowed)
        if gate == "h":
            qc.ry(math.pi / 2.0, q)
            qc.rz(math.pi, q)
        elif gate == "s":
            qc.rz(math.pi / 2.0, q)
        else:
            theta = random.choice(angles)
            if gate == "rx":
                qc.rx(theta, q)
            elif gate == "ry":
                qc.ry(theta, q)
            else:
                qc.rz(theta, q)


def apply_nonstabilizer_gates(
    qc: QuantumCircuit,
    num_qubits: int,
    min_gates: int = 10,
    max_gates: int = 50,
    target_qubit = None,
    max_total_gates: Optional[int] = None,
) -> None:
    """Apply a random number of non-stabilizer single-qubit rotations.

    - total gates G is sampled uniformly in [min_gates, max_gates]
    - each gate is a single-qubit RX/RY/RZ with a random angle in [0, 2pi)
    - the target qubit is sampled uniformly each step unless target_qubit is provided
    """
    total_gates = random.randint(int(min_gates), int(max_gates))
    for _ in range(total_gates):
        if remaining_gate_budget(qc, max_total_gates) <= 0:
            break
        q = int(target_qubit) if target_qubit is not None else random.randrange(num_qubits)
        is_first_on_q = not _qubit_has_prior_gate(qc, q)
        # Avoid a Z-axis rotation as the first gate on a qubit
        axes = ("x", "y", "z")
        if is_first_on_q:
            axes = ("x", "y")
        axis = random.choice(axes)
        theta = random.uniform(0.0, 2 * math.pi)
        if axis == "x":
            qc.rx(theta, q)
        elif axis == "y":
            qc.ry(theta, q)
        else:
            qc.rz(theta, q)


 

def build_product_state_circuit(
    num_qubits: int,
    kind: str,
    nonstab_percent_min: int = 30,
    nonstab_percent_max: int = 100,
    max_total_gates: Optional[int] = None,
) -> QuantumCircuit:
    """Create a product-state circuit on num_qubits of the specified kind.

    kind in {"stabilizer", "magic"}.
    - stabilizer: build using only Rx(pi), Ry(pi), Rz(pi), Rz(pi/2), and H=Ry(pi/2)+Rz(pi)
    - magic: mix stabilizer and non-stabilizer single-qubit gates; total steps are
      sampled in [1,100], and total non-stabilizer ratio is sampled in
      [nonstab_percent_min, nonstab_percent_max] percent
    """
    qc = QuantumCircuit(num_qubits, name=f"product_{kind}_{num_qubits}")
    # Default: enforce budget for product-state generation only
    if max_total_gates is None:
        max_total_gates = MAX_TOTAL_GATES

    if kind == "stabilizer":
        apply_stabilizer_gates(qc, num_qubits, max_total_gates=max_total_gates)
    elif kind == "magic":
        # Draw total non-stabilizer percentage in [nonstab_percent_min, nonstab_percent_max]
        # and convert it to a total count bounded by total_gates. Interleave placements
        # by sampling a random target qubit at each step.

        total_gates = random.randint(10, 50)
        nonstab_percent = random.randint(int(nonstab_percent_min), int(nonstab_percent_max))
        nonstab_target = int(round(total_gates * (nonstab_percent / 100.0)))
        # Ensure at least one non-stabilizer gate (and not exceeding total)
        nonstab_target = max(1, min(total_gates, nonstab_target))
        stab_target = total_gates - nonstab_target

        plan = ["N"] * nonstab_target + ["S"] * stab_target
        random.shuffle(plan)

        for step_kind in plan:
            if remaining_gate_budget(qc, max_total_gates) <= 0:
                break
            q = random.randrange(num_qubits)
            if step_kind == "N":
                # one non-stabilizer gate at a random angle on chosen qubit
                apply_nonstabilizer_gates(
                    qc,
                    num_qubits,
                    min_gates=1,
                    max_gates=1,
                    target_qubit=q,
                    max_total_gates=max_total_gates,
                )
            else:
                # one stabilizer gate on chosen qubit
                apply_stabilizer_gates(
                    qc,
                    num_qubits,
                    min_gates=1,
                    max_gates=1,
                    target_qubit=q,
                    max_total_gates=max_total_gates,
                )

    return qc


def circuit_to_qasm_str(qc: QuantumCircuit) -> str:
    """Return a QASM string for the circuit, compatible with Qiskit 1.x and 0.x."""
    if qasm2_dumps is not None:
        return qasm2_dumps(qc)
    # Fallback for older Qiskit
    return qc.qasm()  # type: ignore[attr-defined]


def circuit_from_qasm_str(qasm_str: str) -> QuantumCircuit:
    """Construct a QuantumCircuit from a QASM string (QASM2), supporting Qiskit 1.x and 0.x."""
    if qasm2_loads is not None:
        return qasm2_loads(qasm_str)
    # Fallback for older Qiskit
    return QuantumCircuit.from_qasm_str(qasm_str)  # type: ignore[attr-defined]


def generate_product_state_dataset(
    min_qubits: int,
    max_qubits: int,
    num_per_count: int,
    nonstab_percent_min: int = 30,
    nonstab_percent_max: int = 100,
) -> List[Tuple[Dict[str, Any], int]]:
    """Generate product-state circuits for each qubit count in [min_qubits, max_qubits].

    Returns a flat list of items, each with its own `num_qubits` in the info dict.
    For each qubit count, the dataset is balanced between stabilizer and magic.
    """
    items: List[Tuple[Dict[str, Any], int]] = []
    for nq in range(int(min_qubits), int(max_qubits) + 1):
        half = int(num_per_count) // 2
        # Stabilizer half (label 0)
        for _ in range(half):
            qc = build_product_state_circuit(nq, kind="stabilizer")
            info = {"qasm": circuit_to_qasm_str(qc), "num_qubits": int(nq)}
            items.append((info, STABILIZER_LABEL))
        # Magic half (label 1)
        for _ in range(int(num_per_count) - half):
            qc = build_product_state_circuit(
                nq,
                kind="magic",
                nonstab_percent_min=nonstab_percent_min,
                nonstab_percent_max=nonstab_percent_max,
            )
            info = {"qasm": circuit_to_qasm_str(qc), "num_qubits": int(nq)}
            items.append((info, MAGIC_LABEL))
    return items


# Backward-compatible alias used elsewhere in the module
def generate_dataset(min_qubits: int, max_qubits: int, num_per_count: int) -> List[Tuple[Dict[str, Any], int]]:
    return generate_product_state_dataset(min_qubits, max_qubits, num_per_count)


def apply_h_as_rotations(qc: QuantumCircuit, qubit: int) -> None:
    qc.ry(math.pi / 2.0, qubit)
    qc.rz(math.pi, qubit)


def apply_s_as_rotation(qc: QuantumCircuit, qubit: int) -> None:
    qc.rz(math.pi / 2.0, qubit)


def evolve_circuit_with_random_clifford(qc: QuantumCircuit, depth: int, max_total_gates: Optional[int] = None) -> QuantumCircuit:
    """Append a sequence of Clifford ops (H, S, CX) of specific depth to the circuit.

    - H is implemented as Ry(pi/2) then Rz(pi)
    - S is implemented as Rz(pi/2)
    - CX is the standard controlled-X
    """
    num_qubits = qc.num_qubits
    for _ in range(int(depth)):
        remaining = remaining_gate_budget(qc, max_total_gates)
        if remaining <= 0:
            break
        # H has cost 2; S and CX cost 1
        choices = []
        if remaining >= 2:
            choices.append("h")
        if remaining >= 1:
            choices.extend(["s", "cx"])
        if not choices:
            break
        gate = random.choice(tuple(choices))
        if gate == "cx":
            control, target = random.sample(range(num_qubits), 2)
            qc.cx(control, target)
        elif gate == "h":
            q = random.randrange(num_qubits)
            apply_h_as_rotations(qc, q)
        else:
            q = random.randrange(num_qubits)
            apply_s_as_rotation(qc, q)
    return qc


def generate_clifford_evolved_dataset(
    product_path: str,
    min_depth: int = 1,
    max_depth: int = 25,
) -> List[Tuple[Dict[str, Any], int]]:
    with open(product_path, "rb") as f:
        combined = pickle.load(f)
    evolved: List[Tuple[Dict[str, Any], int]] = []
    for info, label in combined:
        qasm = info["qasm"] if isinstance(info, dict) else info
        # Generate variants within the specified depth range [min_depth, max_depth]
        for depth in range(int(min_depth), int(max_depth) + 1):
            base_qc = circuit_from_qasm_str(qasm)
            new_qc = evolve_circuit_with_random_clifford(base_qc, depth=depth)
            new_info = {
                "qasm": circuit_to_qasm_str(new_qc),
                "num_qubits": int(new_qc.num_qubits),
                "clifford_depth": int(depth),
            }
            evolved.append((new_info, int(label)))

    return evolved


def add_random_entanglement(qc: QuantumCircuit, num_pairs: int, max_total_gates: Optional[int] = None) -> QuantumCircuit:
    """Return a new circuit with ``num_pairs`` CX gates interleaved at random positions.

    This avoids direct manipulation of qc.data by rebuilding a new circuit with
    inserted CX operations.
    """
    desired = max(0, int(num_pairs))
    # Respect remaining gate budget for the final circuit
    remain_for_inserts = remaining_gate_budget(qc, max_total_gates)
    if remain_for_inserts < desired:
        desired = remain_for_inserts
    # Extract original instructions as (op, q_idx_list, c_idx_list)
    original: list[tuple] = []
    for item in qc.data:
        if hasattr(item, "operation"):
            op = item.operation
            q_idxs = [qc.qubits.index(q) for q in item.qubits]
            c_idxs = [qc.clbits.index(c) for c in item.clbits]
        else:
            # tuple form (Instruction, qargs, cargs)
            op = item[0]
            q_idxs = [qc.qubits.index(q) for q in item[1]]
            c_idxs = [qc.clbits.index(c) for c in item[2]]
        original.append((op, q_idxs, c_idxs))

    # Plan insert positions 0..len(original)
    num_slots = len(original) + 1
    inserts: dict[int, list[tuple[int, int]]] = {i: [] for i in range(num_slots)}
    for _ in range(desired):
        pos = 0 if num_slots == 0 else random.randint(0, num_slots - 1)
        # Determine used qubits before pos
        used_before_idx: set[int] = set()
        for i in range(0, min(pos, len(original))):
            _, q_idxs, _ = original[i]
            for qi in q_idxs:
                used_before_idx.add(qi)
        if not used_before_idx and qc.num_qubits >= 2:
            c_idx, t_idx = random.sample(range(qc.num_qubits), 2)
        else:
            c_idx = random.choice(list(used_before_idx)) if used_before_idx else 0
            remaining = [i for i in range(qc.num_qubits) if i != c_idx]
            t_idx = random.choice(remaining) if remaining else c_idx
        inserts[pos].append((c_idx, t_idx))

    # Build new circuit with interleaved CNOTs
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits, name=qc.name)
    for slot in range(num_slots):
        # insert planned CXs before original instruction at this slot
        for c_idx, t_idx in inserts.get(slot, []):
            if c_idx != t_idx and qc.num_qubits >= 2:
                new_qc.cx(c_idx, t_idx)
        if slot < len(original):
            op, q_idxs, c_idxs = original[slot]
            new_qc.append(op, [new_qc.qubits[i] for i in q_idxs], [new_qc.clbits[i] for i in c_idxs])

    return new_qc


def generate_entangled_dataset(num_qubits: int, num_samples: int) -> List[Tuple[Dict[str, Any], int]]:
    """Generate balanced entangled circuits from product states plus 1-50 random CX gates."""
    base = generate_product_state_dataset(num_qubits, num_qubits, num_samples)
    entangled: List[Tuple[Dict[str, Any], int]] = []
    for info, label in base:
        qc = circuit_from_qasm_str(info["qasm"]) if isinstance(info, dict) else circuit_from_qasm_str(info)
        pairs = random.randint(1, 20)
        qc_ent = add_random_entanglement(qc, pairs)
        entangled_info = {"qasm": circuit_to_qasm_str(qc_ent), "num_qubits": int(qc_ent.num_qubits)}
        entangled.append((entangled_info, int(label)))
    return entangled


def generate_entangled_datasets_over_range(
    min_qubits: int,
    max_qubits: int,
    num_per_count: int,
) -> List[Tuple[Dict[str, Any], int]]:
    """Generate entangled datasets for each qubit count in [min_qubits, max_qubits]."""
    items: List[Tuple[Dict[str, Any], int]] = []
    for nq in range(int(min_qubits), int(max_qubits) + 1):
        items.extend(generate_entangled_dataset(nq, int(num_per_count)))
    return items


 

def main() -> str:
    out_dir = DATASET_TYPE_DIR

    # Reproducibility seeding 
    random.seed(42)

    dataset: List[Tuple[Dict[str, Any], int]] = []
    summaries: List[str] = []

    if GENERATE_CLASS:
        # Range-based product states (combined stabilizer + magic in one PKL)
        dataset = generate_product_state_dataset(
            min_qubits=MIN_QUBITS,
            max_qubits=MAX_QUBITS,
            num_per_count=NUM_PER_COUNT,
        )
        range_tag = f"{MIN_QUBITS}" if MIN_QUBITS == MAX_QUBITS else f"{MIN_QUBITS}_{MAX_QUBITS}"
        product_path = os.path.join(out_dir, f"product_states_{range_tag}.pkl")
        with open(product_path, "wb") as f:
            pickle.dump(dataset, f)
        summaries.append(f"Product states: {product_path} ({len(dataset)} circuits)")


    if GENERATE_ENTANGLED:
        entangled = generate_entangled_datasets_over_range(
            min_qubits=MIN_QUBITS,
            max_qubits=MAX_QUBITS,
            num_per_count=NUM_PER_COUNT,
        )
        range_tag = f"{MIN_QUBITS}" if MIN_QUBITS == MAX_QUBITS else f"{MIN_QUBITS}_{MAX_QUBITS}"
        entangled_path = os.path.join(out_dir, f"entangled_{range_tag}.pkl")
        with open(entangled_path, "wb") as f:
            pickle.dump(entangled, f)
        summaries.append(f"Entangled: {entangled_path} ({len(entangled)} circuits)")


    if GENERATE_EVOLVED:
        range_tag = f"{MIN_QUBITS}" if MIN_QUBITS == MAX_QUBITS else f"{MIN_QUBITS}_{MAX_QUBITS}"
        product_path = os.path.join(out_dir, f"product_states_{range_tag}.pkl")
        if not os.path.exists(product_path):
            # Ensure class PKLs exist; generate them now
            dataset = generate_product_state_dataset(
                min_qubits=MIN_QUBITS,
                max_qubits=MAX_QUBITS,
                num_per_count=NUM_PER_COUNT,
            )
            with open(product_path, "wb") as f:
                pickle.dump(dataset, f)
            
        evolved_path = os.path.join(out_dir, f"clifford_evolved_{range_tag}.pkl")
        evolved_items = generate_clifford_evolved_dataset(
            product_path=product_path,
            min_depth=1,
            max_depth=25,
        )
        with open(evolved_path, "wb") as f:
            pickle.dump(evolved_items, f)
        summaries.append(f"Clifford-evolved: {evolved_path} ({len(evolved_items)} circuits)")


    for line in summaries:
        print(line)
    return out_dir


if __name__ == "__main__":
    main()
