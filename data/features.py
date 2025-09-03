import pennylane as qml
import pickle
import os
import numpy as np
import itertools
from tqdm import tqdm
from qiskit import QuantumCircuit



def load_circuits(file_path):
    """Load random circuits from a pkl file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_updated_circuits(file_path, updated_circuits):
    """Save the updated circuits to a pkl file."""
    with open(file_path, 'wb') as f:
        pickle.dump(updated_circuits, f)


def compute_classical_shadow_expectation(circuit_func, wire_labels, observable, num_samples=100):
    """
    Compute the expectation value of a local Pauli observable using classical shadows.
    
    Parameters:
    - circuit_func: A function representing the quantum circuit
    - wire_labels: List of wire labels for the device
    - observable: PennyLane observable (e.g., qml.PauliX(0))
    - num_samples: Number of samples for classical shadows
    
    Returns:
    - Estimated expectation value of the observable
    """
    dev = qml.device("default.qubit", wires=wire_labels, shots=num_samples)

    @qml.qnode(dev)
    def shadow_node():
        circuit_func()
        return qml.classical_shadow(wires=wire_labels)
    
    bits, recipes = shadow_node()
    shadow = qml.ClassicalShadow(bits, recipes)
    return shadow.expval(observable)


def generate_pauli_observables(n_qubits, max_qubits):
    """
    Generate Pauli observables acting on at most max_qubits qubits.
    
    Parameters:
    - n_qubits: Total number of qubits in the system.
    - max_qubits: Maximum number of qubits on which each observable should act.
    
    Returns:
    - List of Pauli observables
    """
    pauli_matrices = [qml.PauliX, qml.PauliY, qml.PauliZ]
    observables = []

    for num_acting_qubits in range(1, max_qubits + 1):
        for qubit_indices in itertools.combinations(range(n_qubits), num_acting_qubits):
            for pauli_types in itertools.product(pauli_matrices, repeat=num_acting_qubits):
                pauli_ops = [qml.Identity(i) for i in range(n_qubits)]
                
                for qubit, pauli in zip(qubit_indices, pauli_types):
                    pauli_ops[qubit] = pauli(qubit)

                observable = pauli_ops[0]
                for op in pauli_ops[1:]:
                    observable = observable @ op
                observables.append(observable)

    return observables



def extract_qubit_index(qubit):
    """Extract qubit index from various qubit representations."""
    try:
        return qubit.index
    except AttributeError:
        try:
            return qubit._index
        except AttributeError:
            qubit_str = str(qubit)
            try:
                return int(qubit_str.split(', ')[-1].split(')')[0])
            except (ValueError, IndexError):
                return 0



def qasm_to_gate_bins(qasm_str, num_bins=50):
    """
    Extract gate counts and binned rotation angles from QASM string.
    
    Parameters:
    - qasm_str: QASM string representation
    - num_bins: Number of bins for rotation angles (0, 2π)
    
    Returns:
    - Dictionary with gate counts and binned rotation angles
    """
    circuit = QuantumCircuit.from_qasm_str(qasm_str)
    gate_counts = {"cx": 0, "h": 0}
    
    bin_edges = np.linspace(0, 2*np.pi, num_bins + 1)
    rotation_bins = {
        f"{gate}_bin_{i}": 0 
        for gate in ["rx", "ry", "rz"] 
        for i in range(num_bins)
    }
    
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        
        if gate_name in gate_counts:
            gate_counts[gate_name] += 1
        elif gate_name in ["rx", "ry", "rz"]:
            angle = float(instruction.operation.params[0]) % (2*np.pi)
            if angle < 0:
                angle += 2*np.pi
                
            bin_idx = min(np.digitize(angle, bin_edges) - 1, num_bins - 1)
            rotation_bins[f"{gate_name}_bin_{bin_idx}"] += 1
    
    return {**gate_counts, **rotation_bins}


def process_circuit_data(circuit_data, pennylane_circuit, observables):
    """Process a single circuit and add feature data."""
    circuit_data["gate_bins"] = qasm_to_gate_bins(circuit_data['qasm'])
    
    wire_labels = list(range(circuit_data["num_qubits"]))
    
    for observable in observables:
        observable_name = f"obs_{observable}"
        estimated_expectation = compute_classical_shadow_expectation(
            pennylane_circuit, wire_labels, observable
        )
        circuit_data[observable_name] = estimated_expectation


def process_circuits_file(filename, num_qubit):
    """Process a single circuits file."""
    circuits = load_circuits(filename)
    
    if not isinstance(circuits[0], dict):
        return
        
    has_obs_column = any(
        key.startswith("obs_") 
        for circuit in circuits 
        for key in circuit.keys()
    )
    
    if has_obs_column:
        print(f"Skipping {filename}: Observable columns already present")
        return
        
    print(f"Processing {filename}: Adding observable columns")
    
    pennylane_circuits = [qml.from_qasm(circuit['qasm']) for circuit in circuits]
    observables = generate_pauli_observables(n_qubits=num_qubit, max_qubits=2)
    
    for circuit_data, pennylane_circuit in zip(
        tqdm(circuits, desc="Circuits", leave=False), 
        pennylane_circuits
    ):
        process_circuit_data(circuit_data, pennylane_circuit, observables)
    
    save_updated_circuits(filename, circuits)


def process_circuits(qubit_ranges=range(2, 7), gate_ranges=[(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]):
    """Process random circuits dataset."""
    for num_qubit in qubit_ranges:
        for gate_range in tqdm(gate_ranges, desc="Gate Ranges", leave=False):
            filename = f"dataset_random/basis_rotations+cx_qubits_{num_qubit}_gates_{gate_range[0]}-{gate_range[1]}.pkl"
            process_circuits_file(filename, num_qubit)
    
    print("Random circuits dataset processing completed")


def process_ising_circuits(min_qubit=2, max_qubit=7, min_trotter_step=1, max_trotter_step=11):
    """Process Ising model circuits dataset."""
    for num_qubit in range(min_qubit, max_qubit):
        for trotter_steps in tqdm(range(min_trotter_step, max_trotter_step), desc="Trotter Steps", leave=False):
            filename = f"dataset_tim/ising_qubits_{num_qubit}_trotter_{trotter_steps}.pkl"
            process_circuits_file(filename, num_qubit)
    
    print("Ising circuits dataset processing completed")


if __name__ == "__main__":
    process_circuits()
    # process_ising_circuits()
