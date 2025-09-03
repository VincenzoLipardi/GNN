import os
import pickle
import numpy as np

def load_circuits(file_path):
    """Load circuits from a pkl file."""
    with open(file_path, 'rb') as f:
        circuits = pickle.load(f)
    return circuits

def save_circuits(file_path, circuits):
    """Save circuits to a pkl file."""

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(circuits, f)

def extract_features(circuit_tuple, feature_set):
    """Extract different feature sets from a circuit dictionary."""

    circuit_dict, sre = circuit_tuple
    if feature_set == 'shadow':
        shadows = [float(v) for k, v in circuit_dict.items() if 'obs_' in k and v is not None]
        return shadows, sre

    if feature_set == 'gate_bins':
        gate_bins = list(circuit_dict.get('gate_bins', {}).values())
        return gate_bins, sre
    if feature_set == 'combined':
        shadows = [float(v) for k, v in circuit_dict.items() if 'obs_' in k and v is not None]
        gate_bins = list(circuit_dict.get('gate_bins', {}).values())
        combined = shadows + gate_bins
        return combined, sre
    

def check_and_create_subfolders(dataset_folder, feature_set):
    """Check if feature subfolders exist and create them if they don't."""
    feature_path = os.path.join(dataset_folder, feature_set)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
        print(f"Created new feature folder: {feature_path}")
    else:
        print(f"Feature folder already exists: {feature_path}")

def process_dataset(dataset_folder, feature_set, num_qubits=None):
    """Process all pkl files in a dataset folder."""

    check_and_create_subfolders(dataset_folder, feature_set)
    
    # Filter pkl files based on number of qubits if specified
    if num_qubits is not None:
        pkl_files = [f for f in os.listdir(dataset_folder) if f.endswith('.pkl') and os.path.isfile(os.path.join(dataset_folder, f)) and f"qubits_{num_qubits}" in f]
    else:
        pkl_files = [f for f in os.listdir(dataset_folder) if f.endswith('.pkl') and os.path.isfile(os.path.join(dataset_folder, f))]
    
    for file in pkl_files:
        file_path = os.path.join(dataset_folder, file)
        base_name = os.path.basename(file)
        feature_file = os.path.join(dataset_folder, feature_set, base_name)
        if os.path.exists(feature_file):
            print(f"Feature file already exists, skipping: {feature_file}")
            continue
        
        # Extract features from each circuit and save to new dataset
        circuits = load_circuits(file_path)
        feature_dataset = []
        for circuit_tuple in circuits:
            if isinstance(circuit_tuple, tuple):
                feature_data, sre = extract_features(circuit_tuple, feature_set)
                feature_dataset.append((feature_data, sre))
        save_circuits(feature_file, feature_dataset)
        print(f"Saved {feature_set} dataset to {feature_file}")

def main():
    # Process dataset folders
    datasets = ['dataset_random']
    feature_sets = ['shadow', 'gate_bins', "combined"]
    num_qubits = 2
    for dataset in datasets:
        if not os.path.exists(dataset):
            os.makedirs(dataset)
            print(f"Created new dataset folder: {dataset}")
        print(f"\nProcessing {dataset}")
        for feature_set in feature_sets:
            process_dataset(dataset, feature_set, num_qubits)
            

if __name__ == "__main__":
    main()