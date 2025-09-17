import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

def load_entropy_data(directory, num_qubits, trotter_steps_list=None):
    entropy_data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pkl') and "qubits_"+str(num_qubits) in filename:
            filepath = os.path.join(directory, filename)
            if 'dataset_tim' in directory:
                try:
                    trotter_steps = int(filename.split('_trotter_')[1].split('.')[0])
                except (IndexError, ValueError):
                    continue
                if trotter_steps_list is not None and trotter_steps not in trotter_steps_list:
                    continue
                key = trotter_steps
            else:
                key = int(filename.split('.pkl')[0][-2:])
                
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
            if isinstance(data[0], tuple):
                file_data = [x[1] for x in data]
                entropy_data[key] = file_data
            else:
                continue
    return entropy_data

def plot_histograms(entropy_data, num_qubits):
    stats = {key: (round(np.mean(cluster), 3), round(np.std(cluster), 3)) 
            for key, cluster in entropy_data.items()}

    plt.figure(figsize=(10, 6))
    sorted_keys = sorted(entropy_data.keys())
    
    all_values = []
    for key in sorted_keys:
        all_values.extend(entropy_data[key])
    
    n_bins = 20
    global_bin_edges = np.histogram(all_values, bins=n_bins)[1]
    global_bin_centers = (global_bin_edges[:-1] + global_bin_edges[1:]) / 2
    
    for key in sorted_keys:
        cluster = entropy_data[key]
        counts, _ = np.histogram(cluster, bins=global_bin_edges)
        normalized_counts = counts / counts.sum()

        label_prefix = 'Trotter Steps' if 'dataset_tim' in directory else 'Gate Count'
        plt.bar(global_bin_centers, normalized_counts, 
               width=(global_bin_edges[1] - global_bin_edges[0]), 
               alpha=0.8, label=f'{label_prefix} {key-19}-{key}. Avg: {stats[key][0]:.3f}, Std: {stats[key][1]:.3f}')
    
    plt.xticks(global_bin_centers, [f'{x:.3f}' for x in global_bin_centers], rotation=45)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.xlabel('Stabilizer Rényi Entropy')
    plt.ylabel('Frequency (%)')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    results_dir = "/data/P70087789/GNN/models/results"
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"magic_distribution_q_{num_qubits}.png")
    plt.savefig(filename)
    print("Image saved as:", filename)
    return stats, 

def plot_combined_histograms(directory, qubit_range, trotter_steps_list=None):
    plt.figure(figsize=(10, 6))
    
    # Store averages for plotting vertical lines and annotations
    averages = []
    for num_qubits in qubit_range:
        if 'dataset_tim' in directory:
            entropy_data = load_entropy_data(directory, num_qubits, trotter_steps_list=trotter_steps_list)
        else:
            entropy_data = load_entropy_data(directory, num_qubits)
        entropy_data = [value for cluster in entropy_data.values() for value in cluster]
        average = np.mean(entropy_data)
        averages.append(average)
        counts, bin_edges = np.histogram(entropy_data, bins=20)
        normalized_counts = counts / counts.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot distribution and add vertical line for average
        line = plt.plot(bin_centers, normalized_counts, label=f'{num_qubits} qubits', alpha=0.7)[0]
        plt.axvline(x=average, color=line.get_color(), linestyle='--', alpha=0.7)
        plt.annotate(f'{average:.2f}', 
                    xy=(average, plt.ylim()[1]),
                    xytext=(average, plt.ylim()[1] * 1.05),
                    ha='center',
                    color=line.get_color())
    
    plt.xlabel('Stabilizer Rényi Entropy')
    plt.ylabel('Frequency (%)')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    regular_ticks = np.arange(0, max(bin_centers) + 0.2, 0.5)
    plt.xticks(regular_ticks, rotation=45)
    
    plt.margins(y=0.15)
    
    results_dir = "/data/P70087789/GNN/models/results"
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, "combined_magic_distribution.png")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    print("Combined image saved as:", filename)



if __name__ == "__main__":
    directories = ['dataset_random']
    qubit_range = range(2, 7)
    trotter_steps_list = [1, 2, 3, 4, 5]  
    for directory in directories:
        for num_qubits in qubit_range:
            if 'dataset_tim' in directory:
                entropy_data = load_entropy_data(directory, num_qubits, trotter_steps_list=trotter_steps_list)
            else:
                entropy_data = load_entropy_data(directory, num_qubits)
            plot_histograms(entropy_data, num_qubits)