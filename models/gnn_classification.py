import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import pickle

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

try:
	from torch.amp import autocast, GradScaler  # type: ignore[attr-defined]
	_AMP_DEVICE_TYPE = 'cuda'
except Exception:
	from torch.cuda.amp import autocast, GradScaler  # type: ignore
	_AMP_DEVICE_TYPE = 'cuda'

# Enforce classification node vocabulary and 25-qubit mask for 32-dim node features
os.environ.setdefault("GNN_NODE_TYPES_VARIANT", "classification")
os.environ.setdefault("GNN_QUBIT_MASK_DIM", "25")

# Robust imports to access dataset and feature dimension helpers
try:
	from .graph_representation import QuantumCircuitGraphDataset, get_global_feature_dim
	from .gnn import CircuitGNN, _cache_root_for_paths
except Exception:
	project_root = Path(__file__).resolve().parents[1]
	models_dir = Path(__file__).resolve().parent
	for p in (project_root, models_dir):
		if str(p) not in sys.path:
			sys.path.append(str(p))
	try:
		from models.graph_representation import QuantumCircuitGraphDataset, get_global_feature_dim
		from models.gnn import CircuitGNN, _cache_root_for_paths
	except Exception:
		from graph_representation import QuantumCircuitGraphDataset, get_global_feature_dim
		from gnn import CircuitGNN, _cache_root_for_paths


def build_balanced_train_test_loaders(
	pkl_path: str,
	global_feature_variant: str,
	train_per_class: int,
	test_per_class: int,
	batch_size: int = 64,
	seed: int = 42,
) -> Tuple[DataLoader, DataLoader, QuantumCircuitGraphDataset]:
	"""Create balanced 10k/5k loaders (5000/2500 per class) from a single PKL path."""
	suffix = f"{global_feature_variant}"
	root = _cache_root_for_paths([pkl_path], suffix=suffix)
	dataset = QuantumCircuitGraphDataset(
		root=root,
		pkl_paths=[pkl_path],
		global_feature_variant=global_feature_variant,
		node_feature_backend_variant=None,
	)
	if len(dataset) < (train_per_class + test_per_class) * 2:
		raise RuntimeError("Dataset too small for requested balanced split.")

	# Collect indices by class using labels at dataset[i].y (expects 0/1)
	class0_indices: List[int] = []
	class1_indices: List[int] = []
	for idx in range(len(dataset)):
		data = dataset[idx]
		label_tensor = getattr(data, 'y', None)
		if label_tensor is None:
			raise RuntimeError("Missing labels 'y' in dataset entries.")
		label_val = int(label_tensor.view(-1)[0].item())
		if label_val == 0:
			class0_indices.append(idx)
		elif label_val == 1:
			class1_indices.append(idx)
		else:
			raise RuntimeError(f"Unexpected label value {label_val}; expected 0 or 1.")

	# Shuffle deterministically
	rng = torch.Generator().manual_seed(seed)
	def shuffled(xs: List[int]) -> List[int]:
		perm = torch.randperm(len(xs), generator=rng).tolist()
		return [xs[i] for i in perm]

	class0_indices = shuffled(class0_indices)
	class1_indices = shuffled(class1_indices)

	if len(class0_indices) < (train_per_class + test_per_class) or len(class1_indices) < (train_per_class + test_per_class):
		raise RuntimeError("Not enough samples per class to build the requested split.")

	train_indices = class0_indices[:train_per_class] + class1_indices[:train_per_class]
	test_indices = class0_indices[train_per_class:train_per_class + test_per_class] + class1_indices[train_per_class:train_per_class + test_per_class]

	# Shuffle within splits to mix classes
	train_indices = [train_indices[i] for i in torch.randperm(len(train_indices), generator=rng).tolist()]
	test_indices = [test_indices[i] for i in torch.randperm(len(test_indices), generator=rng).tolist()]

	train_ds = Subset(dataset, train_indices)
	test_ds = Subset(dataset, test_indices)
	# Attach split indices for downstream introspection
	try:
		setattr(dataset, "_split_indices", {"train": list(train_indices), "test": list(test_indices)})
	except Exception:
		pass

	num_cpus = os.cpu_count() or 0
	default_workers = 2 if num_cpus > 2 else 0
	pin_mem = torch.cuda.is_available()
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=default_workers, pin_memory=pin_mem)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem)
	return train_loader, test_loader, dataset


def build_stratified_train_test_loaders(
	pkl_path: str,
	global_feature_variant: str,
	train_ratio: float = 0.7,
	batch_size: int = 64,
	seed: int = 42,
) -> Tuple[DataLoader, DataLoader, QuantumCircuitGraphDataset]:
	"""Create stratified loaders with a train_ratio split (default 70/30)."""
	suffix = f"{global_feature_variant}"
	root = _cache_root_for_paths([pkl_path], suffix=suffix)
	dataset = QuantumCircuitGraphDataset(
		root=root,
		pkl_paths=[pkl_path],
		global_feature_variant=global_feature_variant,
		node_feature_backend_variant=None,
	)
	if len(dataset) < 2:
		raise RuntimeError("Dataset too small to split.")

	class0_indices: List[int] = []
	class1_indices: List[int] = []
	for idx in range(len(dataset)):
		data = dataset[idx]
		label_tensor = getattr(data, 'y', None)
		if label_tensor is None:
			raise RuntimeError("Missing labels 'y' in dataset entries.")
		label_val = int(label_tensor.view(-1)[0].item())
		if label_val == 0:
			class0_indices.append(idx)
		elif label_val == 1:
			class1_indices.append(idx)
		else:
			raise RuntimeError(f"Unexpected label value {label_val}; expected 0 or 1.")

	rng = torch.Generator().manual_seed(seed)
	def shuffled(xs: List[int]) -> List[int]:
		perm = torch.randperm(len(xs), generator=rng).tolist()
		return [xs[i] for i in perm]

	class0_indices = shuffled(class0_indices)
	class1_indices = shuffled(class1_indices)

	def split_indices(idxs: List[int]) -> Tuple[List[int], List[int]]:
		train_len = int(round(len(idxs) * train_ratio))
		train_len = max(1, min(train_len, len(idxs)-1)) if len(idxs) > 1 else 1
		return idxs[:train_len], idxs[train_len:]

	c0_tr, c0_te = split_indices(class0_indices)
	c1_tr, c1_te = split_indices(class1_indices)

	train_indices = c0_tr + c1_tr
	test_indices = c0_te + c1_te

	# Final shuffle within splits
	train_indices = [train_indices[i] for i in torch.randperm(len(train_indices), generator=rng).tolist()]
	test_indices = [test_indices[i] for i in torch.randperm(len(test_indices), generator=rng).tolist()]

	train_ds = Subset(dataset, train_indices)
	test_ds = Subset(dataset, test_indices)
	# Attach split indices for downstream introspection
	try:
		setattr(dataset, "_split_indices", {"train": list(train_indices), "test": list(test_indices)})
	except Exception:
		pass

	num_cpus = os.cpu_count() or 0
	default_workers = 2 if num_cpus > 2 else 0
	pin_mem = torch.cuda.is_available()
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=default_workers, pin_memory=pin_mem)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem)
	return train_loader, test_loader, dataset


def compute_confusion_matrix(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
) -> Tuple[int, int, int, int]:
	"""Return (tn, fp, fn, tp)."""
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
				# Count confusion entries
				tn += int(((preds == 0) & (labs == 0)).sum().item())
				fp += int(((preds == 1) & (labs == 0)).sum().item())
				fn += int(((preds == 0) & (labs == 1)).sum().item())
				tp += int(((preds == 1) & (labs == 1)).sum().item())
	return tn, fp, fn, tp


def evaluate_average_bce_loss(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	criterion: nn.Module,
) -> float:
	"""Compute average BCEWithLogits loss over a loader (masked for finite labels)."""
	model.eval()
	loss_sum = 0.0
	count = 0
	with torch.no_grad():
		use_amp = (device.type == 'cuda')
		for batch in loader:
			batch = batch.to(device, non_blocking=True)
			labels: Tensor = batch.y.view(-1)
			mask = torch.isfinite(labels)
			if mask.sum() == 0:
				continue
			labels_masked = labels[mask].float()
			with autocast(_AMP_DEVICE_TYPE, enabled=use_amp):
				logits: Tensor = model(batch)[mask]
				loss = criterion(logits, labels_masked)
			loss_sum += loss.detach().item() * int(mask.sum().item())
			count += int(mask.sum().item())
	return loss_sum / max(1, count)


def infer_feature_dims(
	dataset: QuantumCircuitGraphDataset,
	global_feature_variant: str,
) -> Tuple[int, int]:
	"""Infer (node_in_dim, global_in_dim); force node_in_dim to 32 (25 qubits + 7 gate types)."""
	node_in_dim: Optional[int] = None
	global_in_dim: Optional[int] = None
	for i in range(min(256, len(dataset))):
		data = dataset[i]
		x = getattr(data, 'x', None)
		if x is not None and x.numel() > 0:
			node_in_dim = int(x.size(-1))
		gf = getattr(data, 'global_features', None)
		if gf is not None:
			if gf.dim() == 1:
				global_in_dim = int(gf.numel())
			else:
				global_in_dim = int(gf.size(-1))
		if node_in_dim is not None and global_in_dim is not None:
			break
	# Enforce fixed node embedding size: 25 qubits + 7 gate-type components = 32
	assert node_in_dim is None or node_in_dim == 32, f"Expected node feature dim 32 (25 qubits + 7 gate types), got {node_in_dim}"
	node_in_dim = 32
	if global_in_dim is None:
		global_in_dim = int(get_global_feature_dim(global_feature_variant))
	return node_in_dim, global_in_dim


def train_classifier(dataset_pkl: Optional[str] = None) -> None:
	# Configuration
	dataset_pkl = dataset_pkl or "/data/P70087789/GNN/data/dataset_classification/dataset_type/product_states_18.pkl"
	dataset_base = Path(dataset_pkl).stem
	results_dir = "/data/P70087789/GNN/data/dataset_classification/results"
	models_dir = "/data/P70087789/GNN/data/dataset_classification/models"
	global_feature_variant = "binned152"
	train_per_class = 5000
	test_per_class = 2500
	batch_size = 128
	lr = 0.001
	epochs = 100
	seed = 42
	# Early stopping: stop when train loss repeats for N consecutive epochs
	train_loss_repeat_patience = 30
	train_loss_round_ndigits = 6

	# Data
	# If dataset size is 15000 and we can fulfill 5000/2500 per class, use fixed split; otherwise 70/30 stratified
	try:
		# quick check by instantiating dataset once
		suffix = f"{global_feature_variant}"
		root = _cache_root_for_paths([dataset_pkl], suffix=suffix)
		full_dataset_probe = QuantumCircuitGraphDataset(
			root=root,
			pkl_paths=[dataset_pkl],
			global_feature_variant=global_feature_variant,
			node_feature_backend_variant=None,
		)
		ds_len = len(full_dataset_probe)
	except Exception:
		full_dataset_probe = None
		ds_len = 0

	use_fixed = False
	if ds_len == 15000:
		# count per class to ensure we can draw 5000 train + 2500 test per class
		c0 = 0
		c1 = 0
		for i in range(ds_len):
			lbl = int(full_dataset_probe[i].y.view(-1)[0].item())  # type: ignore[index]
			if lbl == 0:
				c0 += 1
			else:
				c1 += 1
		use_fixed = (c0 >= 7500 and c1 >= 7500)

	if use_fixed:
		train_loader, test_loader, full_dataset = build_balanced_train_test_loaders(
			pkl_path=dataset_pkl,
			global_feature_variant=global_feature_variant,
			train_per_class=5000,
			test_per_class=2500,
			batch_size=batch_size,
			seed=seed,
		)
	else:
		print("Using stratified 70/30 split for this dataset size.")
		train_loader, test_loader, full_dataset = build_stratified_train_test_loaders(
			pkl_path=dataset_pkl,
			global_feature_variant=global_feature_variant,
			train_ratio=0.7,
			batch_size=batch_size,
			seed=seed,
		)

	# Model (fixed 32-dim node embedding, infer/check against dataset)
	node_in_dim, global_in_dim = infer_feature_dims(full_dataset, global_feature_variant)
	model = CircuitGNN(node_in_dim=node_in_dim, global_in_dim=global_in_dim, dropout_rate=0.3)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == 'cuda':
		try:
			gpu_name = torch.cuda.get_device_name(0)
		except Exception:
			gpu_name = 'CUDA device'
		print(f"Using GPU: {gpu_name} (count={torch.cuda.device_count()})")
		torch.backends.cudnn.benchmark = True
	else:
		print("Using CPU")
	model.to(device)

	# Optimizer and loss
	criterion = nn.BCEWithLogitsLoss()
	optimizer = Adam(model.parameters(), lr=lr)
	try:
		scaler = GradScaler(device=_AMP_DEVICE_TYPE, enabled=(device.type == 'cuda'))
	except TypeError:
		scaler = GradScaler(enabled=(device.type == 'cuda'))

	best_val = float('inf')
	best_state = None
	epochs_without_improve = 0
	best_epoch = 0

	# For JSON logging of per-epoch losses
	epoch_indices: List[int] = []
	train_losses: List[float] = []
	val_losses: List[float] = []

	# Train-loss repetition tracking
	last_train_loss_key: Optional[float] = None
	repeat_count = 0

	# Train loop
	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0
		total_examples = 0
		use_amp = (device.type == 'cuda')
		for batch in train_loader:
			batch = batch.to(device, non_blocking=True)
			labels: Tensor = batch.y.view(-1)
			mask = torch.isfinite(labels)
			if mask.sum() == 0:
				continue
			labels_masked = labels[mask].float()
			optimizer.zero_grad(set_to_none=True)
			with autocast(_AMP_DEVICE_TYPE, enabled=use_amp):
				logits: Tensor = model(batch)[mask]
				loss = criterion(logits, labels_masked)
			if not torch.isfinite(loss):
				continue
			scaler.scale(loss).backward()
			try:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
			except Exception:
				pass
			scaler.step(optimizer)
			scaler.update()
			total_loss += loss.detach().item() * int(mask.sum().item())
			total_examples += int(mask.sum().item())

		train_loss_epoch = total_loss / max(1, total_examples)
		val_loss_epoch = evaluate_average_bce_loss(model, test_loader, device, criterion)
		print(f"Epoch {epoch:03d} | TrainLoss {train_loss_epoch:.6f} | ValLoss {val_loss_epoch:.6f}")

		# Record for JSON
		epoch_indices.append(epoch)
		train_losses.append(float(train_loss_epoch))
		val_losses.append(float(val_loss_epoch))

		# Early stopping on repeated train loss
		key = round(float(train_loss_epoch), train_loss_round_ndigits)
		if last_train_loss_key is None or key != last_train_loss_key:
			last_train_loss_key = key
			repeat_count = 0
		else:
			repeat_count += 1
			if repeat_count >= train_loss_repeat_patience:
				print(f"Early stopping at epoch {epoch:03d} | Train loss repeated {repeat_count} times (rounded to {train_loss_round_ndigits} digits): {key}")
				best_epoch = epoch
				break

	# No specific "best" by train loss; proceed with final weights

	# Evaluation: confusion matrices on train and test
	tn_tr, fp_tr, fn_tr, tp_tr = compute_confusion_matrix(model, train_loader, device)
	tn_te, fp_te, fn_te, tp_te = compute_confusion_matrix(model, test_loader, device)
	print("Train Confusion Matrix [[TN, FP],[FN, TP]]:")
	print(f"[[{tn_tr}, {fp_tr}], [{fn_tr}, {tp_tr}]]")
	print("Test Confusion Matrix [[TN, FP],[FN, TP]]:")
	print(f"[[{tn_te}, {fp_te}], [{fn_te}, {tp_te}]]")

	# Collect predictions and labels on test set (dataloader order)
	probs_te: List[float] = []
	labels_te: List[float] = []
	with torch.no_grad():
		for batch in test_loader:
			batch = batch.to(device, non_blocking=True)
			logits = model(batch)
			probs = torch.sigmoid(logits.view(-1))
			lbl = batch.y.view(-1).float()
			mask = torch.isfinite(lbl)
			probs_te.extend(probs[mask].detach().cpu().tolist())
			labels_te.extend(lbl[mask].detach().cpu().tolist())

	# Identify misclassified indices relative to test subset order
	preds_te = [1 if p >= 0.5 else 0 for p in probs_te]
	mis_local_idx = [i for i, (y, yhat) in enumerate(zip(labels_te, preds_te)) if int(round(y)) != int(yhat)]

	# Map to original dataset indices if available
	orig_test_indices: List[int] = []
	try:
		orig_test_indices = list(getattr(full_dataset, "_split_indices", {}).get("test", []))  # type: ignore[assignment]
	except Exception:
		orig_test_indices = []

	# Load original PKL items and optional precomputed SRE stats
	items_list: List[Any] = []
	sre_stats = None
	try:
		with open(dataset_pkl, "rb") as f:
			items_list = pickle.load(f)
		root_path, _ = os.path.splitext(dataset_pkl)
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

	# Build misclassified details payload
	mis_out: Dict[str, Any] = {
		"dataset": str(dataset_pkl),
		"num_misclassified": int(len(mis_local_idx)),
		"indices_test_order": mis_local_idx,
		"items": [],
	}
	for i_local in mis_local_idx:
		orig_idx = orig_test_indices[i_local] if i_local < len(orig_test_indices) else None
		entry: Dict[str, Any] = {
			"test_order_index": int(i_local),
			"orig_index": int(orig_idx) if orig_idx is not None else None,
			"label": int(round(labels_te[i_local])) if i_local < len(labels_te) else None,
			"pred": int(preds_te[i_local]) if i_local < len(preds_te) else None,
			"prob": float(probs_te[i_local]) if i_local < len(probs_te) else None,
		}
		try:
			if orig_idx is not None and items_list and orig_idx < len(items_list):
				it = items_list[orig_idx]
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
				if sre_stats and orig_idx < len(sre_stats):
					_, stats = sre_stats[orig_idx]
					entry["sre"] = {
						"total": float(stats.get("sre_total")) if isinstance(stats, dict) and (stats.get("sre_total") == stats.get("sre_total")) else None,
						"per_qubit": list(stats.get("sre_per_qubit")) if isinstance(stats, dict) and isinstance(stats.get("sre_per_qubit"), list) else None,
						"alpha": 2.0,
						"source": "precomputed",
					}
		except Exception:
			pass
		mis_out["items"].append(entry)

	# Save misclassified test details next to results
	mis_path = os.path.join(results_dir, f"misclassified_{dataset_base}.json")
	with open(mis_path, "w", encoding="utf-8") as f:
		json.dump(mis_out, f, indent=2)
	print(f"Saved misclassified test details to: {mis_path}")

	# Derive metrics
	def safe_div(a: float, b: float) -> float:
		return float(a / b) if b != 0 else 0.0

	train_total = tn_tr + fp_tr + fn_tr + tp_tr
	test_total = tn_te + fp_te + fn_te + tp_te

	train_acc = safe_div(tn_tr + tp_tr, train_total)
	train_prec = safe_div(tp_tr, tp_tr + fp_tr)
	train_rec = safe_div(tp_tr, tp_tr + fn_tr)
	train_f1 = safe_div(2 * train_prec * train_rec, train_prec + train_rec) if (train_prec + train_rec) > 0 else 0.0

	test_acc = safe_div(tn_te + tp_te, test_total)
	test_prec = safe_div(tp_te, tp_te + fp_te)
	test_rec = safe_div(tp_te, tp_te + fn_te)
	test_f1 = safe_div(2 * test_prec * test_rec, test_prec + test_rec) if (test_prec + test_rec) > 0 else 0.0

	# Save JSON with history and stats
	os.makedirs(results_dir, exist_ok=True)
	json_path = os.path.join(results_dir, f"training_{dataset_base}.json")
	json_payload = {
		"config": {
			"batch_size": batch_size,
			"lr": lr,
			"epochs_requested": epochs,
			"stop_reason": f"train_loss_repeated_{train_loss_repeat_patience}_times_rounded_{train_loss_round_ndigits}",
			"stop_epoch": best_epoch,
			"node_in_dim": node_in_dim,
			"global_in_dim": global_in_dim,
			"global_feature_variant": global_feature_variant,
			"train_loss_repeat_patience": train_loss_repeat_patience,
			"train_loss_round_ndigits": train_loss_round_ndigits,
		},
		"loss_history": {
			"epoch": epoch_indices,
			"train": train_losses,
			"val": val_losses,
		},
		"train_stats": {
			"confusion_matrix": {"tn": tn_tr, "fp": fp_tr, "fn": fn_tr, "tp": tp_tr},
			"accuracy": train_acc,
			"precision": train_prec,
			"recall": train_rec,
			"f1": train_f1,
		},
		"test_stats": {
			"confusion_matrix": {"tn": tn_te, "fp": fp_te, "fn": fn_te, "tp": tp_te},
			"accuracy": test_acc,
			"precision": test_prec,
			"recall": test_rec,
			"f1": test_f1,
		},
	}
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(json_payload, f, indent=2)
	print(f"Training stats saved to: {json_path}")

	# Save model
	os.makedirs(models_dir, exist_ok=True)
	save_path = os.path.join(models_dir, f"model_{dataset_base}.pt")
	torch.save({
		"state_dict": model.state_dict(),
		"config": {
			"node_in_dim": node_in_dim,
			"global_in_dim": global_in_dim,
			"global_feature_variant": global_feature_variant,
                "dropout_rate": 0.3,
			"train_loss_repeat_patience": train_loss_repeat_patience,
			"train_loss_round_ndigits": train_loss_round_ndigits,
		},
	}, save_path)
	print(f"Model saved to: {save_path}")



def grid_search_classification(
	pkl_path: str,
	configs: List[Dict[str, Any]],
	epochs: int = 10,
	batch_size: int = 128,
	lr: float = 1e-3,
	seed: int = 42,
	global_feature_variant: str = "binned152",
	train_ratio: float = 0.8,
	results_dir: Optional[str] = "/data/P70087789/GNN/data/dataset_classification/results",
) -> List[Dict[str, Any]]:
	"""Run a lightweight grid search over model hyperparameters for classification.

	Returns a list of results sorted by validation BCE loss (ascending).
	Each result contains: config, val_loss, and metrics (accuracy, precision, recall, f1).
	"""
	# Build loaders once per search to keep splits fixed across configs
	train_loader, test_loader, dataset = build_stratified_train_test_loaders(
		pkl_path=pkl_path,
		global_feature_variant=global_feature_variant,
		train_ratio=train_ratio,
		batch_size=batch_size,
		seed=seed,
	)
	# Infer feature dims (enforces node dim 32)
	node_in_dim, global_in_dim = infer_feature_dims(dataset, global_feature_variant)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == 'cuda':
		try:
			gpu_name = torch.cuda.get_device_name(0)
		except Exception:
			gpu_name = 'CUDA device'
		print(f"Grid search using GPU: {gpu_name} (count={torch.cuda.device_count()})")
		torch.backends.cudnn.benchmark = True

	criterion = nn.BCEWithLogitsLoss()
	results: List[Dict[str, Any]] = []

	# Enable AMP scaler
	try:
		scaler = GradScaler(device=_AMP_DEVICE_TYPE, enabled=(device.type == 'cuda'))
	except TypeError:
		scaler = GradScaler(enabled=(device.type == 'cuda'))

	for cfg in configs:
		# Build model with config overrides
		model_kwargs: Dict[str, Any] = {
			"node_in_dim": node_in_dim,
			"global_in_dim": global_in_dim,
            "dropout_rate": 0.3,
		}
		for k in ("gnn_hidden", "gnn_heads", "global_hidden", "reg_hidden", "num_layers", "dropout_rate"):
			if k in cfg and cfg[k] is not None:
				if k == "dropout_rate":
					model_kwargs[k] = float(cfg[k])
				else:
					model_kwargs[k] = int(cfg[k])
		model = CircuitGNN(**model_kwargs).to(device)
		opt = Adam(model.parameters(), lr=float(cfg.get("lr", lr)))

		# Train short run
		for _ in range(int(cfg.get("epochs", epochs))):
			model.train()
			total_loss = 0.0
			total_examples = 0
			use_amp = (device.type == 'cuda')
			for batch in train_loader:
				batch = batch.to(device, non_blocking=True)
				labels: Tensor = batch.y.view(-1)
				mask = torch.isfinite(labels)
				if mask.sum() == 0:
					continue
				labels_masked = labels[mask].float()
				opt.zero_grad(set_to_none=True)
				with autocast(_AMP_DEVICE_TYPE, enabled=use_amp):
					logits: Tensor = model(batch)[mask]
					loss = criterion(logits, labels_masked)
				if not torch.isfinite(loss):
					continue
				scaler.scale(loss).backward()
				try:
					scaler.unscale_(opt)
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
				except Exception:
					pass
				scaler.step(opt)
				scaler.update()
				total_loss += loss.detach().item() * int(mask.sum().item())
				total_examples += int(mask.sum().item())

		# Evaluate
		val_loss = evaluate_average_bce_loss(model, test_loader, device, criterion)
		tn, fp, fn, tp = compute_confusion_matrix(model, test_loader, device)
		def safe_div(a: float, b: float) -> float:
			return float(a / b) if b != 0 else 0.0
		total = tn + fp + fn + tp
		acc = safe_div(tn + tp, total)
		prec = safe_div(tp, tp + fp)
		rec = safe_div(tp, tp + fn)
		f1 = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0

		results.append({
			"config": {**model_kwargs, "lr": float(cfg.get("lr", lr)), "epochs": int(cfg.get("epochs", epochs))},
			"val_loss": float(val_loss),
			"metrics": {
				"accuracy": acc,
				"precision": prec,
				"recall": rec,
				"f1": f1,
				"confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
			},
		})

	# Sort by validation loss ascending
	results.sort(key=lambda r: r["val_loss"])

	# Optionally write results
	try:
		if results_dir:
			os.makedirs(results_dir, exist_ok=True)
			dataset_base = Path(pkl_path).stem
			out_path = os.path.join(results_dir, f"grid_{dataset_base}.json")
			with open(out_path, "w", encoding="utf-8") as f:
				json.dump({"dataset": pkl_path, "results": results}, f, indent=2)
			print(f"Grid search results saved to: {out_path}")
	except Exception:
		pass

	# Print top-3 succinctly
	print("Top-3 configs by ValLoss:")
	for i, r in enumerate(results[:3]):
		cfg = r["config"]
		m = r["metrics"]
		print(f"{i+1:02d}: ValLoss={r['val_loss']:.6f} | Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | cfg={cfg}")

	return results

if __name__ == "__main__":
	train_classifier()
