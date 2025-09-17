import numpy as np
import os
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Operator, Statevector
from typing import List, Optional
import random
from itertools import product
import matplotlib.pyplot as plt


def generate_random_clifford_circuit(num_qubits: int, depth: int, 
                                   gate_set: List[str] = ['h', 's', 'cx', 'cz']) -> QuantumCircuit:
    """
    Generate a random Clifford circuit with specified number of qubits and depth.
    
    Args:
        num_qubits: Number of qubits in the circuit
        depth: Depth of the circuit (number of layers)
        gate_set: List of Clifford gates to choose from. Default includes H, S, CNOT, CZ
        
    Returns:
        QuantumCircuit: Random Clifford circuit
    """
    
    qc = QuantumCircuit(num_qubits)
    
    for layer in range(depth):
        # Randomly choose between single-qubit or two-qubit gate
        gate_type = random.choice(['single', 'two'])
        
        if gate_type == 'single':
            # Apply a single-qubit gate at a random position
            qubit = random.randint(0, num_qubits - 1)
            single_qubit_gates = [g for g in gate_set if g in ['h', 's']]
            if single_qubit_gates:
                gate = random.choice(single_qubit_gates)
                if gate == 'h':
                    qc.h(qubit)
                elif gate == 's':
                    qc.s(qubit)
        else:
            # Apply a two-qubit gate at random positions
            qubit1 = random.randint(0, num_qubits - 1)
            qubit2 = random.randint(0, num_qubits - 1)
            # Ensure different qubits
            while qubit2 == qubit1:
                qubit2 = random.randint(0, num_qubits - 1)
            
            two_qubit_gates = [g for g in gate_set if g in ['cx', 'cz']]
            if two_qubit_gates:
                gate = random.choice(two_qubit_gates)
                if gate == 'cx':
                    qc.cx(qubit1, qubit2)
                elif gate == 'cz':
                    qc.cz(qubit1, qubit2)
    
    return qc


def compute_stabilizer_renyi_entropy_statevector(clifford: Clifford, alpha: float = 2.0) -> float:
    """
    Compute the stabilizer Rényi entropy using statevector simulation.
    
    Args:
        clifford: Clifford operator representing the circuit
        alpha: Rényi entropy parameter (default: 2.0)
        
    Returns:
        float: Stabilizer Rényi entropy for the full system
    """
    num_qubits = clifford.num_qubits
    d = 2**num_qubits
    pauli_gates = ['I', 'X', 'Y', 'Z']
    gate_combinations = product(pauli_gates, repeat=num_qubits)
    A = 0
    
    # Create statevector from the Clifford operator
    circuit = clifford.to_circuit()
    statevector = Statevector(circuit)
    
    for combination in gate_combinations:
        # Create a new circuit for each Pauli combination
        pauli_circuit = QuantumCircuit(num_qubits)
        
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


def compute_stabilizer_renyi_entropy_clifford(clifford: Clifford, alpha: float = 2.0) -> float:
    """
    Compute the stabilizer Rényi entropy using Clifford formalism.
    This is more efficient for Clifford circuits.
    
    Args:
        clifford: Clifford operator representing the circuit
        alpha: Rényi entropy parameter (default: 2.0)
        
    Returns:
        float: Stabilizer Rényi entropy for the full system
    """
    num_qubits = clifford.num_qubits
    d = 2**num_qubits
    
    # Get the stabilizer tableau
    tableau = clifford.tableau
    
    pauli_gates = ['I', 'X', 'Y', 'Z']
    gate_combinations = product(pauli_gates, repeat=num_qubits)
    A = 0
    
    for combination in gate_combinations:
        # Create Pauli operator as a binary vector
        pauli_vector = np.zeros(2 * num_qubits, dtype=int)
        
        # Convert Pauli combination to binary representation
        for qubit, gate in enumerate(combination):
            if gate == 'X':
                pauli_vector[qubit] = 1
            elif gate == 'Y':
                pauli_vector[qubit] = 1
                pauli_vector[qubit + num_qubits] = 1
            elif gate == 'Z':
                pauli_vector[qubit + num_qubits] = 1
        
        # Check if this Pauli operator commutes with all stabilizer generators
        commutes_with_all = True
        for i in range(num_qubits):
            # Get the i-th stabilizer generator from the tableau
            stabilizer_x = tableau[i, :num_qubits]  # X part of stabilizer
            stabilizer_z = tableau[i + num_qubits, :num_qubits]  # Z part of stabilizer
            
            # Compute symplectic inner product
            inner_product = 0
            for j in range(num_qubits):
                # P_x · S_z + P_z · S_x (mod 2)
                inner_product += (pauli_vector[j] * stabilizer_z[j] + 
                                pauli_vector[j + num_qubits] * stabilizer_x[j])
            inner_product %= 2
            
            if inner_product != 0:
                commutes_with_all = False
                break
        
        # The expectation value is 1 if the Pauli operator commutes with all stabilizers
        exp_val = 1.0 if commutes_with_all else 0.0
        
        # Calculate Ξ_P(ρ) = (1/2^n) * [Tr(ρP)]^2
        xi_p = (1/d) * (exp_val**2)
        
        # Add to the sum: Σ Ξ_P(ρ)^α
        A += xi_p**alpha
    
    # Calculate S_α(ρ) = (1/(1-α)) * log(Σ Ξ_P(ρ)^α) - log(2^n)
    entropy = (1 / (1 - alpha)) * np.log(A) - np.log(d)
    
    return entropy


def runtime_comparison_study(num_qubits: int, depth: int, num_circuits: int = 10) -> dict:
    """
    Perform a runtime comparison study between the two methods.
    
    Args:
        num_qubits: Number of qubits in the circuits
        depth: Depth of the circuits
        num_circuits: Number of circuits to test (default: 10)
        
    Returns:
        dict: Dictionary containing runtime analysis results
    """
    import time
    
    print(f"Starting runtime comparison study...")
    print(f"Parameters: {num_qubits} qubits, depth {depth}, {num_circuits} circuits")
    print("-" * 60)
    
    statevector_times = []
    clifford_times = []
    
    for i in range(num_circuits):
        print(f"Testing circuit {i+1}/{num_circuits}...")
        
        # Generate random circuit
        qc = generate_random_clifford_circuit(num_qubits, depth)
        clifford = Clifford(qc)
        
        # Time statevector method
        start_time = time.time()
        entropy_sv = compute_stabilizer_renyi_entropy_statevector(clifford)
        sv_time = time.time() - start_time
        statevector_times.append(sv_time)
        
        # Time Clifford method
        start_time = time.time()
        entropy_cl = compute_stabilizer_renyi_entropy_clifford(clifford)
        cl_time = time.time() - start_time
        clifford_times.append(cl_time)
        
        # Print entropy values for all circuits
        print(f"  Entropy values - Statevector: {entropy_sv:.6f}, Clifford: {entropy_cl:.6f}")
        if abs(entropy_sv - entropy_cl) > 1e-10:
            print(f"  WARNING: Results don't agree!")
            print("  Circuit that caused the discrepancy:")
            print(qc)
        
        print(f"  Statevector: {sv_time:.4f}s, Clifford: {cl_time:.4f}s")
    
    # Calculate statistics
    avg_sv_time = np.mean(statevector_times)
    avg_cl_time = np.mean(clifford_times)
    std_sv_time = np.std(statevector_times)
    std_cl_time = np.std(clifford_times)
    
    speedup = avg_sv_time / avg_cl_time if avg_cl_time > 0 else float('inf')
    
    results = {
        'num_qubits': num_qubits,
        'depth': depth,
        'num_circuits': num_circuits,
        'statevector_times': statevector_times,
        'clifford_times': clifford_times,
        'avg_statevector_time': avg_sv_time,
        'avg_clifford_time': avg_cl_time,
        'std_statevector_time': std_sv_time,
        'std_clifford_time': std_cl_time,
        'speedup': speedup
    }
    
    return results


def generate_scalability_data(qubit_range: range, depth: int, num_circuits: int = 10) -> dict:
    """
    Generate scalability data for both methods across different qubit counts.
    
    Args:
        qubit_range: Range of qubit counts to test
        depth: Fixed depth for all circuits
        num_circuits: Number of circuits to average over
        
    Returns:
        dict: Dictionary containing scalability data
    """
    print(f"Generating scalability data...")
    print(f"Qubit range: {qubit_range}")
    print(f"Depth: {depth}")
    print(f"Circuits per qubit count: {num_circuits}")
    print("-" * 60)
    
    qubit_counts = []
    statevector_times = []
    clifford_times = []
    speedups = []
    
    for num_qubits in qubit_range:
        print(f"Testing {num_qubits} qubits...")
        
        # Run the runtime comparison study
        results = runtime_comparison_study(num_qubits, depth, num_circuits)
        
        # Store the results
        qubit_counts.append(num_qubits)
        statevector_times.append(results['avg_statevector_time'])
        clifford_times.append(results['avg_clifford_time'])
        speedups.append(results['speedup'])
        
        print(f"  Statevector: {results['avg_statevector_time']:.4f}s")
        print(f"  Clifford: {results['avg_clifford_time']:.4f}s")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print()
    
    scalability_data = {
        'qubit_counts': qubit_counts,
        'statevector_times': statevector_times,
        'clifford_times': clifford_times,
        'speedups': speedups,
        'depth': depth,
        'num_circuits': num_circuits
    }
    
    return scalability_data


def plot_scalability_histogram(scalability_data: dict, save_path: Optional[str]):
    """
    Plot a histogram comparing the average computation times for both methods.
    
    Args:
        scalability_data: Results from generate_scalability_data
        save_path: Optional path to save the plot
    """
    qubit_counts = scalability_data['qubit_counts']
    statevector_times = scalability_data['statevector_times']
    clifford_times = scalability_data['clifford_times']
    speedups = scalability_data['speedups']
    depth = scalability_data['depth']
    num_circuits = scalability_data['num_circuits']
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Average computation times
    x = np.arange(len(qubit_counts))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, statevector_times, width, label='Statevector Method', 
                    color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, clifford_times, width, label='Clifford Method', 
                    color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Average Computation Time (seconds)')
    ax1.set_title(f'Average Computation Time vs Qubit Count\n(Depth: {depth}, {num_circuits} circuits per qubit count)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(qubit_counts)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Use log scale to better show the difference
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Speedup factor
    ax2.bar(x, speedups, color='green', alpha=0.7)
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Speedup Factor (Statevector/Clifford)')
    ax2.set_title(f'Speedup Factor vs Qubit Count\n(Depth: {depth}, {num_circuits} circuits per qubit count)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(qubit_counts)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, speedup in enumerate(speedups):
        ax2.text(i, speedup, f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()

    # Ensure images are saved under the global results directory by default
    results_dir = "/data/P70087789/GNN/models/results"
    if not save_path:
        save_path = os.path.join(results_dir, 'scalability_histogram.png')
    elif not os.path.isabs(save_path):
        save_path = os.path.join(results_dir, os.path.basename(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")


def main():
    """
    Main function to demonstrate the usage and run the scalability experiment.
    """
    # Run the scalability experiment
    print("=" * 60)
    print("STABILIZER RÉNYI ENTROPY SCALABILITY EXPERIMENT")
    print("=" * 60)
    
    # Generate scalability data
    scalability_data = generate_scalability_data(qubit_range=range(2, 7), depth=30, num_circuits=10)
    
    # Create and save the histogram
    plot_scalability_histogram(scalability_data, save_path='scalability_histogram.png')
    
    return scalability_data


if __name__ == "__main__":
    scalability_data = main()
