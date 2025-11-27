import os
import sys
import json
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, List

import torch
from torch import nn, Tensor
from torch_geometric.loader import DataLoader

try:
	from torch.amp import autocast  # type: ignore[attr-defined]
	_AMP_DEVICE_TYPE = 'cuda'
except Exception:
	from torch.cuda.amp import autocast  # type: ignore
	_AMP_DEVICE_TYPE = 'cuda'

# Robust imports for project structure
try:
	from .graph_representation import QuantumCircuitGraphDataset, get_global_feature_dim, get_node_feature_dim
	from .gnn import CircuitGNN, _cache_root_for_paths
except Exception:
	project_root = Path(__file__).resolve().parents[1]
	models_dir = Path(__file__).resolve().parent
	for p in (project_root, models_dir):
		if str(p) not in sys.path:
			sys.path.append(str(p))
	try:
		from models.graph_representation import QuantumCircuitGraphDataset, get_global_feature_dim, get_node_feature_dim
		from models.gnn import CircuitGNN, _cache_root_for_paths
	except Exception:
		from graph_representation import QuantumCircuitGraphDataset, get_global_feature_dim, get_node_feature_dim
		from gnn import CircuitGNN, _cache_root_for_paths


MODELS_DIR = "/data/P70087789/GNN/data/dataset_classification/models"
RESULTS_DIR = "/data/P70087789/GNN/data/dataset_classification/results"
MODEL_PATH_DEFAULT = os.path.join(MODELS_DIR, "model_product_states_2_10_balanced_by_sre.pt")
DATASETS_DEFAULT = [
	#"/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_11_25.pkl",
	#"/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_18.pkl",
	#"/data/P70087789/GNN/data/dataset_classification/dataset_type/entangled_11_25.pkl",
	"/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_11_25_balanced_by_sre.pkl",
	
	
]
GLOBAL_FEATURE_VARIANT_FALLBACK = "binned152"
# Compute the default node feature dim from dataset encoding rules (env-driven)
NODE_IN_DIM_DEFAULT = get_node_feature_dim(None)


# Editable evaluation jobs
#
# Edit the paths below to evaluate specific model checkpoints on one or more datasets.
# You can add multiple job dicts to run several evaluations in one go.
# Each job supports keys:
#   - "model": path to the .pt checkpoint
#   - "datasets": list of dataset .pkl paths
#   - "results_dir": where to save evaluation JSONs (optional; defaults below)
#   - "batch_size": evaluation batch size (optional)
EVAL_JOBS: List[Dict[str, Any]] = [
	{
		"model": MODEL_PATH_DEFAULT,
		"datasets": list(DATASETS_DEFAULT),
		"results_dir": RESULTS_DIR,
		"batch_size": 128,
	},
]


def build_loader(pkl_path: str, global_feature_variant: str, batch_size: int = 128) -> DataLoader:
	suffix = f"{global_feature_variant}"
	root = _cache_root_for_paths([pkl_path], suffix=suffix)
	dataset = QuantumCircuitGraphDataset(
		root=root,
		pkl_paths=[pkl_path],
		global_feature_variant=global_feature_variant,
	)
	num_cpus = os.cpu_count() or 0
	default_workers = 2 if num_cpus > 2 else 0
	pin_mem = torch.cuda.is_available()
	return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem)


def compute_confusion(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[int, int, int, int]:
	model.eval()
	tn = fp = fn = tp = 0
	with torch.no_grad():
		use_amp = (device.type == 'cuda')
		for batch in loader:
			batch = batch.to(device, non_blocking=True)
			with autocast(_AMP_DEVICE_TYPE, enabled=use_amp):
				logits: Tensor = model(batch)
				labels: Tensor = batch.y.view(-1)
				mask = torch.isfinite(labels)
				if mask.sum() == 0:
					continue
				probs = torch.sigmoid(logits[mask])
				preds = (probs >= 0.5).long()
				labs = labels[mask].long()
				tn += int(((preds == 0) & (labs == 0)).sum().item())
				fp += int(((preds == 1) & (labs == 0)).sum().item())
				fn += int(((preds == 0) & (labs == 1)).sum().item())
				tp += int(((preds == 1) & (labs == 1)).sum().item())
	return tn, fp, fn, tp


def compute_confusion_per_depth(model: nn.Module, loader: DataLoader, device: torch.device, depths: List[int]) -> Dict[int, Tuple[int, int, int, int]]:
	"""Accumulate confusion matrix per Clifford depth using sample-aligned depths list."""
	assert len(depths) == len(loader.dataset), "Depth list length must equal dataset length"
	per_depth: Dict[int, Tuple[int, int, int, int]] = {}
	# initialize
	for d in range(1, 26):
		per_depth[d] = (0, 0, 0, 0)
	model.eval()
	idx_offset = 0
	with torch.no_grad():
		use_amp = (device.type == 'cuda')
		for batch in loader:
			batch = batch.to(device, non_blocking=True)
			with autocast(_AMP_DEVICE_TYPE, enabled=use_amp):
				logits: Tensor = model(batch)
				labels: Tensor = batch.y.view(-1)
				mask = torch.isfinite(labels)
			if mask.numel() == 0:
				idx_offset += getattr(batch, 'num_graphs', labels.shape[0])
				continue
			probs = torch.sigmoid(logits)
			preds = (probs >= 0.5).long()
			labs = labels.long()
			batch_size = int(labs.shape[0])
			for j in range(batch_size):
				if not bool(torch.isfinite(labs[j])):
					continue
				depth = int(depths[idx_offset + j])
				# Clamp depths outside 1..25 into range just in case
				if depth < 1:
					depth = 1
				elif depth > 25:
					depth = 25
				tn, fp, fn, tp = per_depth.get(depth, (0, 0, 0, 0))
				if int(preds[j].item()) == 0 and int(labs[j].item()) == 0:
					tn += 1
				elif int(preds[j].item()) == 1 and int(labs[j].item()) == 0:
					fp += 1
				elif int(preds[j].item()) == 0 and int(labs[j].item()) == 1:
					fn += 1
				else:
					tp += 1
				per_depth[depth] = (tn, fp, fn, tp)
			idx_offset += batch_size
	return per_depth


def derive_metrics(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
	total = tn + fp + fn + tp
	accuracy = (tn + tp) / total if total else 0.0
	precision = tp / (tp + fp) if (tp + fp) else 0.0
	recall = tp / (tp + fn) if (tp + fn) else 0.0
	f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
	return {
		"accuracy": float(accuracy),
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
	}


def load_clifford_depths(pkl_path: str) -> List[int]:
	"""Load per-sample clifford_depth from the PKL list of (info, label)."""
	with open(pkl_path, "rb") as f:
		items = pickle.load(f)
	depths: List[int] = []
	for info, _ in items:
		if isinstance(info, dict) and "clifford_depth" in info:
			depths.append(int(info["clifford_depth"]))
		else:
			depths.append(0)
	return depths


def main() -> None:
	jobs: List[Dict[str, Any]] = EVAL_JOBS
	if not isinstance(jobs, list) or not jobs:
		raise RuntimeError("EVAL_JOBS must be a non-empty list of job dicts")

	for job in jobs:
		model_path = str(job.get("model", MODEL_PATH_DEFAULT))
		results_dir = str(job.get("results_dir", RESULTS_DIR))
		datasets: List[str] = list(job.get("datasets", DATASETS_DEFAULT))
		batch_size = int(job.get("batch_size", 128))

		if not os.path.exists(model_path):
			raise FileNotFoundError(f"Model not found at {model_path}")

		ckpt = torch.load(model_path, map_location="cpu")
		cfg = ckpt.get("config", {})
		global_feature_variant = cfg.get("global_feature_variant", GLOBAL_FEATURE_VARIANT_FALLBACK)
		# Derive node feature dim from either the checkpoint or dataset rules
		node_in_dim = int(cfg.get("node_in_dim", NODE_IN_DIM_DEFAULT))
		global_in_dim = int(cfg.get("global_in_dim", get_global_feature_dim(global_feature_variant)))

		# Align dataset node feature encoding with the checkpoint expectations.
		# If the model was trained with the 'classification' node-types variant (32 dims),
		# enforce the same for evaluation via environment variable used by the dataset encoder.
		try:
			if int(node_in_dim) == 32:
				os.environ["GNN_NODE_TYPES_VARIANT"] = "classification"
			else:
				print("happened")
				# Unset to use default (33 dims) unless user already set something explicitly
				if os.environ.get("GNN_NODE_TYPES_VARIANT") == "classification":
					os.environ.pop("GNN_NODE_TYPES_VARIANT", None)
		except Exception:
			pass

		model = CircuitGNN(node_in_dim=node_in_dim, global_in_dim=global_in_dim)
		model.load_state_dict(ckpt["state_dict"])
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)
		if device.type == 'cuda':
			try:
				gpu_name = torch.cuda.get_device_name(0)
			except Exception:
				gpu_name = 'CUDA device'
			print(f"Evaluating on GPU: {gpu_name} (count={torch.cuda.device_count()})")
		else:
			print("Evaluating on CPU")

		os.makedirs(results_dir, exist_ok=True)
		model_stem = Path(model_path).stem.replace("model_", "").replace(".pt", "")
		for pkl_path in datasets:
			loader = build_loader(pkl_path, global_feature_variant, batch_size=batch_size)
			dataset_base = Path(pkl_path).stem
			if "clifford_evolved" in dataset_base:
				depths = load_clifford_depths(pkl_path)
				if len(depths) != len(loader.dataset):
					print(f"Warning: depths length {len(depths)} != dataset length {len(loader.dataset)}; results may misalign.")
				per_depth_conf = compute_confusion_per_depth(model, loader, device, depths)
				per_depth_out: Dict[str, Any] = {}
				for d, conf in per_depth_conf.items():
					tn, fp, fn, tp = conf
					per_depth_out[str(d)] = {
						"confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
						"metrics": derive_metrics(tn, fp, fn, tp),
					}
				result = {
					"dataset": pkl_path,
					"model": model_path,
					"global_feature_variant": global_feature_variant,
					"per_depth": per_depth_out,
				}
			else:
				tn, fp, fn, tp = compute_confusion(model, loader, device)
				result = {
					"dataset": pkl_path,
					"model": model_path,
					"global_feature_variant": global_feature_variant,
					"confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
					"metrics": derive_metrics(tn, fp, fn, tp),
				}

			# Compute and save misclassified sample details (evaluation order == dataset order)
			probs_all: List[float] = []
			labels_all: List[float] = []
			with torch.no_grad():
				use_amp = (device.type == 'cuda')
				for batch in loader:
					batch = batch.to(device, non_blocking=True)
					with autocast(_AMP_DEVICE_TYPE, enabled=use_amp):
						logits: Tensor = model(batch)
						probs = torch.sigmoid(logits.view(-1))
					lbl = batch.y.view(-1).float()
					mask = torch.isfinite(lbl)
					if mask.sum() == 0:
						continue
					probs_all.extend(probs[mask].detach().cpu().tolist())
					labels_all.extend(lbl[mask].detach().cpu().tolist())
			preds_all = [1 if p >= 0.5 else 0 for p in probs_all]
			mis_local_idx = [i for i, (y, yhat) in enumerate(zip(labels_all, preds_all)) if int(round(y)) != int(yhat)]

			# Load original items and optional precomputed SRE stats
			items_list: List[Any] = []
			sre_stats = None
			try:
				with open(pkl_path, "rb") as f:
					items_list = pickle.load(f)
				root_path, _ = os.path.splitext(pkl_path)
				sre_path = root_path + "_sre_stats.pkl"
				if os.path.isfile(sre_path):
					with open(sre_path, "rb") as f:
						sre_stats = pickle.load(f)
			except Exception:
				items_list = []
				sre_stats = None

			# Optional: parse structure using Qiskit (no simulation)
			try:
				from qiskit import QuantumCircuit  # type: ignore
				try:
					from qiskit.qasm2 import loads as qasm2_loads  # type: ignore
				except Exception:
					qasm2_loads = None  # type: ignore
			except Exception:
				QuantumCircuit = None  # type: ignore
				qasm2_loads = None  # type: ignore

			def _load_circuit(qasm_str: str):
				if QuantumCircuit is None:
					return None
				try:
					if qasm2_loads is not None:
						return qasm2_loads(qasm_str)  # type: ignore[misc]
					return QuantumCircuit.from_qasm_str(qasm_str)
				except Exception:
					return None

			def _gate_counts(qc) -> Dict[str, int]:
				counts: Dict[str, int] = {}
				try:
					for instr, _, _ in getattr(qc, "data", []):
						name = getattr(instr, "name", None)
						if not name:
							continue
						ln = str(name).lower()
						counts[ln] = counts.get(ln, 0) + 1
				except Exception:
					pass
				return counts

			mis_out: Dict[str, Any] = {
				"dataset": str(pkl_path),
				"model": str(model_path),
				"num_misclassified": int(len(mis_local_idx)),
				"indices_test_order": mis_local_idx,
				"items": [],
			}
			for i_local in mis_local_idx:
				entry: Dict[str, Any] = {
					"test_order_index": int(i_local),
					"orig_index": int(i_local),
					"label": int(round(labels_all[i_local])) if i_local < len(labels_all) else None,
					"pred": int(preds_all[i_local]) if i_local < len(preds_all) else None,
					"prob": float(probs_all[i_local]) if i_local < len(probs_all) else None,
				}
				try:
					if items_list and i_local < len(items_list):
						it = items_list[i_local]
						qasm = it[0].get("qasm") if isinstance(it, tuple) and isinstance(it[0], dict) else (it.get("qasm") if isinstance(it, dict) else None)
						entry["qasm"] = qasm
						qc = _load_circuit(qasm) if isinstance(qasm, str) else None
						if qc is not None:
							structure = {
								"num_qubits": int(getattr(qc, "num_qubits", 0)),
								"depth": int(qc.depth()) if hasattr(qc, "depth") else None,
								"total_gates": int(len(getattr(qc, "data", []))),
								"gate_counts": _gate_counts(qc),
							}
							entry["structure"] = structure
						if sre_stats and i_local < len(sre_stats):
							_, stats = sre_stats[i_local]
							entry["sre"] = {
								"total": float(stats.get("sre_total")) if isinstance(stats, dict) and (stats.get("sre_total") == stats.get("sre_total")) else None,
								"per_qubit": list(stats.get("sre_per_qubit")) if isinstance(stats, dict) and isinstance(stats.get("sre_per_qubit"), list) else None,
								"alpha": 2.0,
								"source": "precomputed",
							}
				except Exception:
					pass
				mis_out["items"].append(entry)

			mis_path = os.path.join(results_dir, f"misclassified_{model_stem}_{dataset_base}.json")
			with open(mis_path, "w", encoding="utf-8") as f:
				json.dump(mis_out, f, indent=2)
			print(f"Saved misclassified details to: {mis_path}")
			out_path = os.path.join(results_dir, f"evaluation_{model_stem}_{dataset_base}.json")
			with open(out_path, "w", encoding="utf-8") as f:
				json.dump(result, f, indent=2)
			print(f"Saved evaluation for {dataset_base} (model {model_stem}) to {out_path}")


if __name__ == "__main__":
	main()
