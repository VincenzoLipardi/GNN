import pickle
import numpy as np
from itertools import product
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector


def calculate_stabilizer_renyi_entropy_qiskit(circuit, alpha=2):
    n = len(circuit.qubits)     
    d = 2**n
    pauli_gates = ['I', 'X', 'Y', 'Z']
    gate_combinations = product(pauli_gates, repeat=n)    
    A = 0
    
    # Create statevector once outside the loop since circuit doesn't change
    statevector = Statevector(circuit)

    for combination in gate_combinations:
        # Create a new circuit for each Pauli combination
        pauli_circuit = QuantumCircuit(n)
        
        # Apply Pauli gates according to the combination
        for qubit, gate in enumerate(combination):
            if gate == 'X':
                pauli_circuit.x(qubit)
            elif gate == 'Y':
                pauli_circuit.y(qubit)
            elif gate == 'Z':
                pauli_circuit.z(qubit)
            # 'I' gate does nothing, so no else clause needed

        # Calculate expectation value: Tr(ρP)
        pauli_operator = Operator(pauli_circuit)
        exp_val = statevector.expectation_value(pauli_operator).real
        
        # Calculate Ξ_P(ρ) = (1/2^n) * [Tr(ρP)]^2
        xi_p = (1/d) * (exp_val**2)
        
        # Add to the sum: Σ Ξ_P(ρ)^α
        A += xi_p**alpha

    # Calculate S_α(ρ) = (1/(1-α)) * log(Σ Ξ_P(ρ)^α) - log(2^n)
    entropy = (1 / (1 - alpha)) * np.log(A) - np.log(d)

    return entropy


def process_circuit_data(circuit_data, is_tuple_format):
    """Extract circuit info and compute entropy for a single circuit."""
    if is_tuple_format:
        circuit_info, qasm_str = circuit_data[0], circuit_data[0]['qasm']
    else:
        circuit_info, qasm_str = circuit_data, circuit_data['qasm']
    
    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
    entropy = calculate_stabilizer_renyi_entropy_qiskit(qiskit_circuit)
    return (circuit_info, entropy)


def process_dataset_file(filename, description="Processing Circuits"):
    """Generic function to process a dataset file and add entropy labels."""
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    # Check if already processed
    if isinstance(data[0], tuple) and len(data[0]) == 2:
        print(f"File {filename} already processed. Skipping...")
        return
    
    # Determine data format once
    is_tuple_format = isinstance(data[0], tuple)
    
    # Process all circuits
    result = [process_circuit_data(circuit, is_tuple_format) 
              for circuit in tqdm(data, desc=description)]
    
    # Save results
    with open(filename, 'wb') as file:
        pickle.dump(result, file)
    print(f"Updated {len(result)} circuits with entropy values and saved to {filename}")


def label_random_circuits(qubit_ranges=range(2, 3), gate_ranges=[(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]):
    """Label random circuit datasets with stabilizer Renyi entropy."""
    for num_qubit in qubit_ranges:
        for gate_range in tqdm(gate_ranges, desc="Gate Ranges", leave=False):
            print(f"Processing Qubit: {num_qubit}, Gates: {gate_range}")
            filename = f"dataset_random/basis_rotations+cx_qubits_{num_qubit}_gates_{gate_range[0]}-{gate_range[1]}.pkl"
            process_dataset_file(filename, "Processing Random Circuits")


def label_ising_circuits(min_qubit=2, max_qubit=7, min_trotter_step=6, max_trotter_step=11):
    """Label Ising model circuit datasets with stabilizer Renyi entropy."""
    for num_qubit in range(min_qubit, max_qubit):
        for trotter_steps in tqdm(range(min_trotter_step, max_trotter_step), 
                                desc="Trotterization Steps", leave=False):
            filename = f"dataset_tim/ising_qubits_{num_qubit}_trotter_{trotter_steps}.pkl"
            process_dataset_file(filename, "Processing Ising Circuits")


if __name__ == "__main__":
    label_random_circuits()
    #label_ising_circuits()
