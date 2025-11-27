import pennylane as qml
from pennylane import numpy as np
import random
import pickle
import os
from qiskit import QuantumCircuit

def random_gate(num_qubits, basis_gates):
    # Normalize basis_gates input
    basis_gates = basis_gates.lower()
    basis_dict = {
        "clifford+t": [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.S, qml.CNOT, qml.T],
        "clifford":   [qml.Hadamard, qml.S, qml.T, qml.RX, qml.RY, qml.RZ, qml.CNOT],
        "rotations+cx": [qml.RX, qml.RY, qml.RZ, qml.CNOT]
        }

    gate_list = basis_dict[basis_gates]

    gate = random.choice(gate_list)

    if gate == qml.CNOT:
        control, target = random.sample(range(num_qubits), 2)
        return gate(wires=[control, target])
    elif gate in [qml.RX, qml.RY, qml.RZ]:
        angle = np.random.uniform(0, 2 * np.pi)
        qubit = random.randint(0, num_qubits - 1)
        return gate(angle, wires=qubit)
    else:
        qubit = random.randint(0, num_qubits - 1)
        return gate(wires=qubit)
    

def build_qasm_circuit(num_qubits, circuit_body_fn):
    dev = qml.device('lightning.qubit', wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        circuit_body_fn()
        return qml.state()

    circuit.construct([], {})
    qasm_str = circuit.qtape.to_openqasm()
    qasm_lines = qasm_str.split('\n')
    qasm_str = '\n'.join([line for line in qasm_lines if not line.startswith('creg') and not line.startswith('measure')])
    return qasm_str

def random_quantum_circuit(num_qubits, num_gates, basis_gates):
    def body():
        for _ in range(num_gates):
            random_gate(num_qubits, basis_gates)
    return build_qasm_circuit(num_qubits, body)

def trotterized_ising_circuit(num_qubits, trotterization_steps, rx_angle):
    def body():
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        for _ in range(trotterization_steps):
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                rz_angle = np.random.uniform(0, rx_angle)
                qml.RZ(rz_angle, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            for i in range(num_qubits):
                qml.RX(rx_angle, wires=i)
    return build_qasm_circuit(num_qubits, body)


def generate_and_save_circuits(directory, num_qubits, num_circuits, basis_gates, circuit_type='random', gate_range=None, trotterization_steps=1):
    # Create a directory for saving the QASM files
    os.makedirs(directory, exist_ok=True)
    
    circuits = []
    if circuit_type == 'random':
        if gate_range is None:
            raise ValueError("gate_range must be provided for random circuits")
        for _ in range(num_circuits):
            # Randomly select the number of gates from the given range
            num_gates = random.randint(gate_range[0], gate_range[1])
            # Generate the random quantum circuit and get the QASM string
            qasm_str = random_quantum_circuit(num_qubits, num_gates, basis_gates)
            circuits.append({
                'qasm': qasm_str,
                'num_qubits': num_qubits,
                #'num_gates': num_gates
            })
    elif circuit_type == 'ising':
        for _ in range(num_circuits):
            # Generate a random RX angle in the range [0, 2π]
            random_rx_angle = np.random.uniform(0, 2 * np.pi)
            # Generate the Trotterized Ising circuit and get the QASM string
            qasm_str = trotterized_ising_circuit(num_qubits, trotterization_steps, random_rx_angle)
            circuits.append({
                'qasm': qasm_str,
                'num_qubits': num_qubits,
                #'h': 'random',  # Use 'random' to indicate RX angles are random
                #'J': 'random'   # Placeholder for RZ angle, as it is random
            })
    else:
        raise ValueError(f"Invalid circuit type: {circuit_type}")

    # Define the filename based on the circuit type
    if circuit_type == 'random':
        filename = f"basis_{basis_gates}_qubits_{num_qubits}_gates_{gate_range[0]}-{gate_range[1]}.pkl"
    elif circuit_type == 'ising':
        filename = f"ising_qubits_{num_qubits}_trotter_{trotterization_steps}.pkl"

    filepath = os.path.join(directory, filename)
    
    # Check if the file already exists
    if os.path.exists(filepath):
        print(f"File {filepath} already exists")
        return filepath
    
    with open(filepath, 'wb') as f:
        pickle.dump(circuits, f)
    print(f"Saved {num_circuits} {circuit_type} quantum circuits to {filepath}")
    
    return filepath

def pennylane_circuit_from_qasm(qasm_str, num_qubits):
    dev = qml.device('default.qubit', wires=num_qubits) 
    
    @qml.qnode(dev)
    def circuit():
        qml.from_qasm(qasm_str)
        return qml.state()
    return circuit

def qiskit_circuit_from_qasm(qasm_str, num_qubits):
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    return qc

def generate_circuits(
    circuit_type,
    directory,
    num_circuits,
    basis_gates,
    qubit_range,
    gate_ranges=None,
    trotter_steps_list=None
):
    for num_qubits in qubit_range:
        if circuit_type == 'random':
            for gate_range in gate_ranges:
                generate_and_save_circuits(
                    directory=directory,
                    num_qubits=num_qubits,
                    num_circuits=num_circuits,
                    basis_gates=basis_gates,
                    circuit_type='random',
                    gate_range=gate_range
                )
        elif circuit_type == 'ising':
            for trotter_steps in trotter_steps_list:
                generate_and_save_circuits(
                    directory=directory,
                    num_qubits=num_qubits,
                    num_circuits=num_circuits,
                    basis_gates=basis_gates,
                    circuit_type='ising',
                    trotterization_steps=trotter_steps
                )

if __name__ == "__main__":
    # Example usage for random circuits
    generate_circuits(
        circuit_type='random',
        directory="dataset_random",
        num_circuits=10000,
        basis_gates="rotations+cx",
        qubit_range=range(2, 7), 
        gate_ranges=[(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]
    )
    # Example usage for ising circuits
    generate_circuits(
        circuit_type='ising',
        directory="dataset_tim",
        num_circuits=1000,
        basis_gates="rotations+cx",
        qubit_range=range(2, 7),  
        trotter_steps_list=range(1, 10)
    )
    