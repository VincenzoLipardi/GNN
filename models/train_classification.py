import os
import sys
from pathlib import Path

# Ensure module imports work when run as a script
this_dir = Path(__file__).resolve().parent
project_root = this_dir.parents[1]
for p in (project_root, this_dir):
	if str(p) not in sys.path:
		sys.path.append(str(p))

# Robust import of train_classifier to support direct script runs from this directory
try:
	from gnn_classification import train_classifier  # type: ignore
except Exception:
	# Ensure local and project root are on sys.path, then retry local import
	this_dir = Path(__file__).resolve().parent
	project_root = this_dir.parents[1]
	for p in (project_root, this_dir):
		if str(p) not in sys.path:
			sys.path.append(str(p))
	from gnn_classification import train_classifier  # type: ignore


def main() -> None:
	datasets = [
		"/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_18_balanced_by_sre.pkl",
		
		#"/data/P70087789/GNN/data/dataset_classification/dataset_type/clifford_evolved_2_10.pkl",
		#"/data/P70087789/GNN/data/dataset_classification/dataset_type/clifford_evolved_18.pkl",
		#"/data/P70087789/GNN/data/dataset_classification/dataset_type/clifford_evolved_2_25.pkl",
		#"/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_10.pkl",
		
		
	]
	for ds in datasets:
		print(f"Training on dataset: {ds}")
		train_classifier(ds)


if __name__ == "__main__":
	main()
