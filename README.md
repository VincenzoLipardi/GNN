## Nonstabilizerness Estimation using Graph Neural Networks

Code and experiments related to the article “Nonstabilizerness Estimation using Graph Neural Networks”.  
This repository studies how Graph Neural Networks (GNNs) can be trained to estimate the stabilizer Rényi entropy of quantum circuits.

- GNN models (PyTorch Geometric) operating on circuit-graph representations
- Reproducible extrapolation experiments (depth, qubits, reverse-depth, per-qubit)
- Data generation for random and Trotterized Ising (TIM) circuits with PennyLane/Qiskit
- Oslo/noisy variants and node feature backends
- Utilities for classification analysis and plots


### What’s inside
- Models and training code in `models/`
- Experiment entry points in `gnn_run.py` and `gnn_run_oslo.py` (plus an optional `mlp_run.py` baseline)
- Data generation in `data/`
- Tutorials on the quantum circuit representation, models and analysis in `tutorials`


## Repository structure
```
GNN/
  gnn_run.py                        # Main CLI for GNN extrapolation experiments
  gnn_run_oslo.py                   # Oslo/noisy dataset experiments
  mlp_run.py                        # Simple MLP baseline (optional)
  models/
    gnn.py                          # Circuit GNN model + loaders + training utils
    gnn_classification.py           # Classification model utilities
    train_classification.py         # Classification training
    eval_gnn_classification.py      # Classification evaluation
    graph_representation.py         # Graph construction and feature definitions
    evaluate_model.py, models.py, svm.py, svr_*.py, ...
    results/, results_oslo/         # Saved results/artifacts (trained models, metrics)
  data/
    circuits_generation.py          # Build datasets with PennyLane/Qiskit
    create_dataset.py, features.py, label.py, ... 
    dataset_random/, dataset_tim/, dataset_random_oslo/, dataset_tim_oslo/
    dataset_classification/
      data_classification.py
      plot_performance.py
      plot_accuracy.py, plot_training.py, plot_gate_stats.py, ...
      images/, models/, results/
    data_distribution.py, noise_simulation.py, noise_study.py, ...
  scripts/
    analyze_backend_node_features.py
    plot_compare_saved.py
  LICENSE
  README.md
```


## Installation (with conda)
Tested with Python 3.9–3.11 on Linux. We recommend a fresh conda environment.

1) Create and activate an environment
```bash
conda create -y -n ml python=3.10
conda activate ml
```

2) Install requirements
```bash
pip install -r requirements.txt
```


## Data: where to find it
The data are generated using the .py files in `data`. The resulting data are saved in 
  - `data/dataset_classification/dataset_type`: Classification Tasks
  - `data/dataset_random/`: Regression task on random quantum circuits
  - `data/dataset_tim/`: Regression task on Trotterized Ising model circuits
  - `data/dataset_*_oslo/`: Predicting the SRE measured on a noisy backend



## Tutorials



### Depth extrapolation (multi-seed)
```bash
python gnn_run.py \
  --experiment depth \
  --random-base-dir /data/P70087789/GNN/data/dataset_random \
  --tim-base-dir /data/P70087789/GNN/data/dataset_tim \
  --results-dir /data/P70087789/GNN/models/results \
  --num-seeds 10 \
  --do-grid-search \
  --grid-epochs 100
```

### Qubit extrapolation (multi-seed)
Optionally restrict TIM trotter steps or Random gate counts:
```bash
python gnn_run.py \
  --experiment qubits \
  --random-base-dir /data/P70087789/GNN/data/dataset_random \
  --tim-base-dir /data/P70087789/GNN/data/dataset_tim \
  --results-dir /data/P70087789/GNN/models/results \
  --num-seeds 10 \
  --do-grid-search \
  --grid-epochs 25 \
  --tim-trotter 1,2,3,4,5 \
  --random-gates 20,40,60,80
```

### Reverse depth extrapolation
Train on higher-depth circuits and test on low depth.
```bash
python gnn_run.py \
  --experiment depth_reverse \
  --random-base-dir /data/P70087789/GNN/data/dataset_random \
  --tim-base-dir /data/P70087789/GNN/data/dataset_tim \
  --results-dir /data/P70087789/GNN/models/results
```

### Per-qubit experiments
```bash
python gnn_run.py \
  --experiment per_qubit \
  --random-base-dir /data/P70087789/GNN/data/dataset_random \
  --tim-base-dir /data/P70087789/GNN/data/dataset_tim \
  --results-dir /data/P70087789/GNN/models/results
```

Common optional flags
- `--global-feature-variant binned152` (default) or alternatives in `models/graph_representation.py`
- `--node-feature-backend-variant fake_oslo` to augment node features with backend info
- `--epochs`, `--lr`, `--batch-size`, `--early-stopping-patience`, `--early-stopping-min-delta`
- `--device cuda` or `cpu` (defaults to auto)
- `--loss-type huber|mse`


## Models: where they are and what gets saved
- Training artifacts, metrics, and model checkpoints are saved as pickles (and related files) under:
  - `models/results/` for standard experiments
  - `models/results_oslo/` for Oslo/noisy experiments
- Classification-specific outputs/plots are saved under:
  - `data/dataset_classification/results/`, `data/dataset_classification/images/`, and `data/dataset_classification/models/`


## Classification utilities and plots
The classification workflow and plots live under `data/dataset_classification/`.

- Performance plot across depths (per-class accuracy):
  - Adjust JSON paths at the top of `data/dataset_classification/plot_performance.py`:
    - `JSON_PATH_*` and `JSON_PATH_EVAL_*`
  - Then run:
    ```bash
    python data/dataset_classification/plot_performance.py
    ```
  - Output figure is saved to `data/dataset_classification/images/`.

Other helpers:
- `plot_accuracy.py`, `plot_training.py`, `plot_sre_density.py`, `plot_sre_boxplot.py`, `plot_gate_stats.py`
- `new_classification_task.py`, `data_classification.py`, `sre_calculation.py`


## Reproducibility
- Deterministic seeding is applied in `gnn_run.py` via `set_global_seed`.
- For exact reproducibility across GPUs/CPUs, consider disabling CuDNN benchmarking and pinning package versions.


## Dependencies (summary)
- Core: torch, torch_geometric, numpy, scipy
- Visualization: matplotlib, seaborn
- Quantum circuit tooling: pennylane (and optionally `pennylane-lightning`), qiskit
- Optional baselines/analysis: scikit-learn


## License
See `LICENSE` for details.


## Citation
If you use this repository in academic work related to “Nonstabilizerness Estimation using Graph Neural Networks”, please cite or reference it appropriately. A BibTeX entry can be added here once available.
