import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D

# Optional import for per-qubit overlay plotting (works when run from repo root or data/)
try:
    from .noise_study import plot_per_qubit_overlays as _plot_per_qubit_overlays
except Exception:
    try:
        from data.dataset_regression.noise_study import plot_per_qubit_overlays as _plot_per_qubit_overlays
    except Exception:
        _plot_per_qubit_overlays = None

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



def _load_random_sre_by_qubits_and_bins(dataset_dir):
    data_by_qubits = {}
    if not os.path.isdir(dataset_dir):
        return data_by_qubits
    for filename in os.listdir(dataset_dir):
        if not filename.endswith('.pkl'):
            continue
        if 'qubits_' not in filename or 'gates_' not in filename:
            continue
        parts = filename.replace('.pkl', '').split('_')
        try:
            n_idx = parts.index('qubits') + 1
            g_idx = parts.index('gates') + 1
            num_qubits = int(parts[n_idx])
            gate_range = parts[g_idx]
            a_str, b_str = gate_range.split('-')
            a_val = int(a_str)
            b_val = int(b_str)
        except Exception:
            continue
        path = os.path.join(dataset_dir, filename)
        try:
            with open(path, 'rb') as f:
                items = pickle.load(f)
        except Exception:
            continue
        sre_values = []
        for item in items:
            if isinstance(item, tuple) and len(item) >= 2:
                try:
                    sre_values.append(float(item[1]))
                except Exception:
                    sre_values.append(np.nan)
        bin_key = (a_val, b_val)
        data_by_qubits.setdefault(num_qubits, {}).setdefault(bin_key, []).extend(sre_values)
    return data_by_qubits


def plot_sre_std_scatter_random(dataset_dir):
    by_qubit = _load_random_sre_by_qubits_and_bins(dataset_dir)
    if not by_qubit:
        print(f"No data found in {dataset_dir}")
        return

    # Determine all depth bins and sort by starting gate count
    all_bins = set()
    for bin_map in by_qubit.values():
        all_bins.update(bin_map.keys())
    sorted_bins = sorted(all_bins, key=lambda ab: ab[0])

    # X positions and labels
    x_positions = [b for (_, b) in sorted_bins]
    x_labels = [f"{a}-{b}" for (a, b) in sorted_bins]

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    # Plot per-qubit mean with std shaded area
    for idx, (q, bin_map) in enumerate(sorted(by_qubit.items())):
        mean_vals = []
        std_vals = []
        for bin_key in sorted_bins:
            arr = np.array(bin_map.get(bin_key, []), dtype=float)
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                mean_vals.append(np.nan)
                std_vals.append(np.nan)
            else:
                mean_vals.append(np.mean(valid))
                std_vals.append(np.std(valid))
        color = cmap(idx % 10)
        plt.plot(x_positions, mean_vals, color=color, label=f"{q} qubits")
        lower = (np.array(mean_vals) - np.array(std_vals))
        upper = (np.array(mean_vals) + np.array(std_vals))
        plt.fill_between(x_positions, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Gate Count')
    plt.ylabel('Stabilizer Rényi Entropy')
    plt.xticks(x_positions, x_labels, rotation=45)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    results_dir = "/data/P70087789/GNN/data/dataset_regression/data_distribution"
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "sre_std_scatter_random.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    print("Image saved as:", out_path)


def plot_sre_std_only_random(dataset_dir):
    by_qubit = _load_random_sre_by_qubits_and_bins(dataset_dir)
    if not by_qubit:
        print(f"No data found in {dataset_dir}")
        return

    all_bins = set()
    for bin_map in by_qubit.values():
        all_bins.update(bin_map.keys())
    sorted_bins = sorted(all_bins, key=lambda ab: ab[0])

    x_positions = [b for (_, b) in sorted_bins]
    x_labels = [f"{a}-{b}" for (a, b) in sorted_bins]

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    for idx, (q, bin_map) in enumerate(sorted(by_qubit.items())):
        std_vals = []
        for bin_key in sorted_bins:
            arr = np.array(bin_map.get(bin_key, []), dtype=float)
            valid = arr[~np.isnan(arr)]
            std_vals.append(np.std(valid) if valid.size > 0 else np.nan)
        color = cmap(idx % 10)
        plt.plot(x_positions, std_vals, color=color, marker='o', label=f"{q} qubits")

    plt.xlabel('Gate Count')
    plt.ylabel('σ')
    plt.xticks(x_positions, x_labels, rotation=45)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    results_dir = "/data/P70087789/GNN/data/dataset_regression/data_distribution"
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "sre_std_only_random.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    print("Image saved as:", out_path)


def plot_sre_std_only_tim(dataset_dir):
    by_qubit = _load_tim_sre_by_qubits_and_steps(dataset_dir)
    if not by_qubit:
        print(f"No data found in {dataset_dir}")
        return

    all_steps = set()
    for step_map in by_qubit.values():
        all_steps.update(step_map.keys())
    sorted_steps = [s for s in sorted(all_steps) if 1 <= s <= 5]
    if not sorted_steps:
        print("No Trotter steps in range 1-5 found for TIM dataset")
        return

    x_positions = sorted_steps
    x_labels = [str(s) for s in sorted_steps]

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    for idx, (q, step_map) in enumerate(sorted(by_qubit.items())):
        std_vals = []
        for step in sorted_steps:
            arr = np.array(step_map.get(step, []), dtype=float)
            valid = arr[~np.isnan(arr)]
            std_vals.append(np.std(valid) if valid.size > 0 else np.nan)
        color = cmap(idx % 10)
        plt.plot(x_positions, std_vals, color=color, marker='o', label=f"{q} qubits")

    plt.xlabel('Trotter steps')
    plt.ylabel('σ')
    plt.xticks(x_positions, x_labels, rotation=0)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    results_dir = "/data/P70087789/GNN/data/data_distribution"
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "sre_std_only_tim.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    print("Image saved as:", out_path)

def _load_tim_sre_by_qubits_and_steps(dataset_dir):
    data_by_qubits = {}
    if not os.path.isdir(dataset_dir):
        return data_by_qubits
    for filename in os.listdir(dataset_dir):
        if not filename.endswith('.pkl'):
            continue
        if 'qubits_' not in filename or 'trotter_' not in filename:
            continue
        parts = filename.replace('.pkl', '').split('_')
        try:
            n_idx = parts.index('qubits') + 1
            t_idx = parts.index('trotter') + 1
            num_qubits = int(parts[n_idx])
            trotter = int(parts[t_idx])
        except Exception:
            continue
        path = os.path.join(dataset_dir, filename)
        try:
            with open(path, 'rb') as f:
                items = pickle.load(f)
        except Exception:
            continue
        sre_values = []
        for item in items:
            if isinstance(item, tuple) and len(item) >= 2:
                try:
                    sre_values.append(float(item[1]))
                except Exception:
                    sre_values.append(np.nan)
        data_by_qubits.setdefault(num_qubits, {}).setdefault(trotter, []).extend(sre_values)
    return data_by_qubits


def plot_sre_std_scatter_tim(dataset_dir):
    by_qubit = _load_tim_sre_by_qubits_and_steps(dataset_dir)
    if not by_qubit:
        print(f"No data found in {dataset_dir}")
        return

    all_steps = set()
    for step_map in by_qubit.values():
        all_steps.update(step_map.keys())
    sorted_steps = [s for s in sorted(all_steps) if 1 <= s <= 5]
    if not sorted_steps:
        print("No Trotter steps in range 1-5 found for TIM dataset")
        return

    x_positions = sorted_steps
    x_labels = [str(s) for s in sorted_steps]

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    for idx, (q, step_map) in enumerate(sorted(by_qubit.items())):
        mean_vals = []
        std_vals = []
        for step in sorted_steps:
            arr = np.array(step_map.get(step, []), dtype=float)
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                mean_vals.append(np.nan)
                std_vals.append(np.nan)
            else:
                mean_vals.append(np.mean(valid))
                std_vals.append(np.std(valid))
        color = cmap(idx % 10)
        plt.plot(x_positions, mean_vals, color=color, label=f"{q} qubits")
        lower = (np.array(mean_vals) - np.array(std_vals))
        upper = (np.array(mean_vals) + np.array(std_vals))
        plt.fill_between(x_positions, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Trotter steps')
    plt.ylabel('Stabilizer Rényi Entropy')
    plt.xticks(x_positions, x_labels, rotation=0)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    results_dir = "/data/P70087789/GNN/data/dataset_regression/data_distribution"
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "sre_std_scatter_tim.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    print("Image saved as:", out_path)

if __name__ == "__main__":
    ONLY_GENERATE_NEW = True
    if not ONLY_GENERATE_NEW:
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
    # Create the requested scatter plot for dataset_random
    try:
        plot_sre_std_scatter_random('dataset_random')
        plot_sre_std_only_random('dataset_random')
    except Exception as exc:
        print(f"Failed to create SRE std scatter: {exc}")
    # Create the analogous scatter plot for dataset_tim
    try:
        plot_sre_std_scatter_tim('dataset_tim')
        plot_sre_std_only_tim('dataset_tim')
    except Exception as exc:
        print(f"Failed to create TIM SRE std scatter: {exc}")
    # Create per-qubit overlay: noiseless Random vs Oslo Random
    if _plot_per_qubit_overlays is not None:
        try:
            _plot_per_qubit_overlays(
                dataset_name='Random_OSLO',
                noiseless_dir='/data/P70087789/GNN/data/dataset_regression/dataset_random',
                noisy_dir='/data/P70087789/GNN/data/dataset_regression/dataset_random_oslo',
                results_dir='/data/P70087789/GNN/data/dataset_regression/data_distribution',
            )
        except Exception as exc:
            print(f"Failed to create Random_OSLO per-qubit overlay: {exc}")
    else:
        print("plot_per_qubit_overlays not available; skipping Random_OSLO per-qubit plot.")
    # Create per-qubit overlay: noiseless TIM vs Oslo TIM
    if _plot_per_qubit_overlays is not None:
        try:
            _plot_per_qubit_overlays(
                dataset_name='TIM_OSLO',
                noiseless_dir='/data/P70087789/GNN/data/dataset_regression/dataset_tim',
                noisy_dir='/data/P70087789/GNN/data/dataset_regression/dataset_tim_oslo',
                results_dir='/data/P70087789/GNN/data/dataset_regression/data_distribution',
            )
        except Exception as exc:
            print(f"Failed to create TIM_OSLO per-qubit overlay: {exc}")
    else:
        print("plot_per_qubit_overlays not available; skipping TIM_OSLO per-qubit plot.")