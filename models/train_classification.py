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
	from gnn_classification import train_classifier, grid_search_classification  # type: ignore
except Exception:
	# Ensure local and project root are on sys.path, then retry local import
	this_dir = Path(__file__).resolve().parent
	project_root = this_dir.parents[1]
	for p in (project_root, this_dir):
		if str(p) not in sys.path:
			sys.path.append(str(p))
	from gnn_classification import train_classifier, grid_search_classification  # type: ignore


def main() -> None:
	datasets = [
		"/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_18_balanced_by_sre.pkl",
		
		#"/data/P70087789/GNN/data/dataset_classification/dataset_type/clifford_evolved_2_10.pkl",
		#"/data/P70087789/GNN/data/dataset_classification/dataset_type/clifford_evolved_18.pkl",
		#"/data/P70087789/GNN/data/dataset_classification/dataset_type/clifford_evolved_2_25.pkl",
		#"/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_2_10.pkl",
		
		
	]

	# Optional: run grid search if enabled via env var
	if os.environ.get("GNN_CLF_GRID_SEARCH", "0").strip() in ("1", "true", "True"):
		configs = []
		for dropout_rate in (0.3, 0.4, 0.5):
			for global_hidden in (16, 32, 64):
				for reg_hidden in (16, 32, 64):
					configs.append({
						"dropout_rate": dropout_rate,
						"global_hidden": global_hidden,
						"reg_hidden": reg_hidden,
						"epochs": 50,
						"lr": 1e-3,
					})
		for ds in datasets:
			print(f"Grid searching on dataset: {ds}")
			grid_search_classification(
				pkl_path=ds,
				configs=configs,
				epochs=10,
				batch_size=128,
				lr=1e-3,
				seed=42,
				global_feature_variant="binned152",
				train_ratio=0.7,
			)
		return

	# Default: normal training
	for ds in datasets:
		print(f"Training on dataset: {ds}")
		train_classifier(ds)


if __name__ == "__main__":
	main()
