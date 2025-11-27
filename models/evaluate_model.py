import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from qiskit import QuantumCircuit


# Default to classification node vocabulary unless overridden externally
os.environ.setdefault("GNN_NODE_TYPES_VARIANT", "classification")


# Ensure imports work whether or not 'models' is a package
try:
    from .graph_representation import QuantumCircuitGraphDataset, get_node_feature_dim, get_global_feature_dim
    from .gnn import CircuitGNN
except Exception:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = Path(__file__).resolve().parent
    for p in (project_root, models_dir):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    try:
        from models.graph_representation import QuantumCircuitGraphDataset, get_node_feature_dim, get_global_feature_dim
        from models.gnn import CircuitGNN
    except Exception:
        from graph_representation import QuantumCircuitGraphDataset, get_node_feature_dim, get_global_feature_dim
        from gnn import CircuitGNN


def _save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _build_loader(pkl_paths: List[str], batch_size: int, global_feature_variant: str) -> DataLoader:
    canonical = "|".join(sorted(os.path.abspath(p) for p in pkl_paths))
    root = os.path.join(os.getcwd(), f"pyg_eval_cache_{hash(canonical) & 0xFFFFFFFF:08x}_{global_feature_variant}")
    dataset = QuantumCircuitGraphDataset(
        root=root,
        pkl_paths=pkl_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=None,
    )
    num_cpus = os.cpu_count() or 0
    workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_mem)


def _collect_probs_targets(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[float], List[float]]:
    probs_all: List[float] = []
    targets_all: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            logits = model(batch)
            probs = torch.sigmoid(logits.view(-1))
            target = getattr(batch, 'y', None)
            if target is None:
                # Fill NaNs to maintain length consistency
                probs_all.extend(probs.detach().cpu().tolist())
                targets_all.extend([float('nan')] * int(probs.shape[0]))
            else:
                t = target.view(-1).float()
                mask = torch.isfinite(t)
                p = probs.detach().cpu().tolist()
                tt = t.detach().cpu().tolist()
                # Keep raw values; metrics will handle NaNs
                probs_all.extend(p)
                targets_all.extend(tt)
    return probs_all, targets_all


def _classification_metrics(probs: List[float], targets: List[float], threshold: float = 0.5) -> Dict[str, Any]:
    # Filter out NaN targets for metrics
    valid: List[Tuple[float, float]] = [(p, t) for p, t in zip(probs, targets) if t == t]
    if len(valid) == 0:
        return {
            "num_samples": 0,
            "accuracy": float('nan'),
            "precision": float('nan'),
            "recall": float('nan'),
            "f1": float('nan'),
            "roc_auc": float('nan'),
            "avg_precision": float('nan'),
            "confusion_matrix": [[0, 0], [0, 0]],
            "threshold": float(threshold),
        }
    p_valid = [p for p, _ in valid]
    t_valid = [int(t) for _, t in valid]
    preds = [1 if p >= threshold else 0 for p in p_valid]
    acc = accuracy_score(t_valid, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(t_valid, preds, average='binary', zero_division=0)
    # Avoid undefined warnings by skipping AUC/AP when only one class present
    if len(set(t_valid)) > 1:
        try:
            auc = roc_auc_score(t_valid, p_valid)
        except Exception:
            auc = float('nan')
        try:
            ap = average_precision_score(t_valid, p_valid)
        except Exception:
            ap = float('nan')
    else:
        auc = float('nan')
        ap = float('nan')
    cm = confusion_matrix(t_valid, preds, labels=[0, 1]).tolist()
    return {
        "num_samples": int(len(t_valid)),
        "accuracy": float(acc),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "roc_auc": float(auc),
        "avg_precision": float(ap),
        "confusion_matrix": cm,
        "threshold": float(threshold),
    }


def _infer_doped_depths_by_order(pkl_paths: List[str]) -> List[int]:
    """Infer doped depth as (index % 10) + 1, matching generation order (legacy fallback)."""
    depths: List[int] = []
    for p in pkl_paths:
        with open(p, 'rb') as f:
            items = pickle.load(f)
        for idx, _ in enumerate(items):
            depths.append(int((idx % 10) + 1))
    return depths


def _extract_depths_from_pkls(pkl_paths: List[str]) -> List[int]:
    """Extract per-sample clifford depths from PKLs when available; fallback to legacy inference.

    If an item has a dict with key 'clifford_depth', use that value. Otherwise, fall back to
    the legacy heuristic (index % 10) + 1 used for doped datasets.
    """
    depths: List[int] = []
    has_any_explicit = False
    items_all: List[Any] = []
    for p in pkl_paths:
        with open(p, 'rb') as f:
            items = pickle.load(f)
        items_all.extend(items)
    # First pass: check if any item includes explicit clifford_depth
    for it in items_all:
        info = it[0] if isinstance(it, (list, tuple)) and len(it) >= 1 else None
        if isinstance(info, dict) and ("clifford_depth" in info):
            has_any_explicit = True
            break
    if has_any_explicit:
        for it in items_all:
            info = it[0] if isinstance(it, (list, tuple)) and len(it) >= 1 else None
            if isinstance(info, dict) and ("clifford_depth" in info):
                depths.append(int(info["clifford_depth"]))
            else:
                depths.append(int(1))
        return depths
    # Fallback: legacy order-based inference (1..10 repeating)
    return _infer_doped_depths_by_order(pkl_paths)


def evaluate(
    model_path: str,
    pkl_paths: List[str],
    out_json: str,
    batch_size: int = 128,
    global_feature_variant: str = "baseline",
    threshold: float = 0.5,
    compare_model_path: str = "",
    save_misclassified: str = "",
) -> Dict[str, Any]:
    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    # Determine model dims
    if "model_kwargs" in meta:
        gdim = int(meta["model_kwargs"].get("global_in_dim"))
        node_in_dim = int(meta["model_kwargs"].get("node_in_dim"))
    else:
        gdim = get_global_feature_dim(global_feature_variant)
        node_in_dim = get_node_feature_dim(None)
    model = CircuitGNN(global_in_dim=gdim, node_in_dim=node_in_dim)
    model.load_state_dict(ckpt["state_dict"])  # type: ignore[index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Build loader
    loader = _build_loader(pkl_paths=pkl_paths, batch_size=batch_size, global_feature_variant=global_feature_variant)

    # Collect predictions -> compute metrics only
    probs, targets = _collect_probs_targets(model=model, loader=loader, device=device)
    metrics = _classification_metrics(probs=probs, targets=targets, threshold=threshold)

    # Build JSON output
    to_save: Dict[str, Any] = metrics

    # Only for clifford_evolved.pkl: include per-depth metrics in the same JSON
    try:
        is_evolved = any(os.path.basename(p) == "clifford_evolved.pkl" for p in pkl_paths)
    except Exception:
        is_evolved = False
    if is_evolved:
        depths = _extract_depths_from_pkls(pkl_paths)
        per_depth: Dict[str, Any] = {}
        if len(depths) == len(probs) and len(probs) > 0:
            unique_depths = sorted({int(d) for d in depths})
            for d in unique_depths:
                idx = [i for i, dd in enumerate(depths) if int(dd) == int(d)]
                if not idx:
                    continue
                p_all = [probs[i] for i in idx]
                t_all = [targets[i] for i in idx]
                per_depth[str(d)] = _classification_metrics(p_all, t_all, threshold=threshold)
        to_save = {"metrics": metrics, "per_depth": per_depth}

    # Save JSON
    _save_json(out_json, to_save)

    # Optionally save misclassified circuits
    if save_misclassified:
        try:
            preds = [1 if p >= threshold else 0 for p in probs]
            mis_idx = [i for i, t in enumerate(targets) if t == t and preds[i] != int(t)]
            out: Dict[str, Any] = {
                "model": os.path.abspath(model_path),
                "pkls": [os.path.abspath(p) for p in pkl_paths],
                "threshold": float(threshold),
                "num_misclassified": int(len(mis_idx)),
                "indices": mis_idx,
                "items": [],
            }
            mis_set = set(mis_idx)
            # Prepare SRE stats if precomputed next to PKLs ("<file>_sre_stats.pkl")
            sre_by_path: Dict[str, Optional[List[Any]]] = {}
            for pkl_path in pkl_paths:
                try:
                    root, _ = os.path.splitext(pkl_path)
                    sre_path = root + "_sre_stats.pkl"
                    if os.path.isfile(sre_path):
                        with open(sre_path, "rb") as f:
                            sre_by_path[pkl_path] = pickle.load(f)
                    else:
                        sre_by_path[pkl_path] = None
                except Exception:
                    sre_by_path[pkl_path] = None

            # Lazy imports for circuit parsing and optional SRE computation fallback
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

            # Stream PKLs to attach QASM, labels, structure, and SRE for misclassified items only
            cursor = 0
            for pkl_path in pkl_paths:
                with open(pkl_path, "rb") as f:
                    content = pickle.load(f)
                sre_list = sre_by_path.get(pkl_path)
                for idx_in_file, it in enumerate(content):
                    if cursor in mis_set:
                        if isinstance(it, tuple) and len(it) == 2:
                            circ_info, label = it
                            qasm = circ_info.get("qasm") if isinstance(circ_info, dict) else None
                            item: Dict[str, Any] = {
                                "index": int(cursor),
                                "label": int(label) if label == label else None,
                                "pred": int(preds[cursor]),
                                "prob": float(probs[cursor]),
                                "qasm": qasm,
                            }
                            # Structure via Qiskit parsing (no simulation)
                            qc = _load_circuit(qasm) if isinstance(qasm, str) else None
                            if qc is not None:
                                try:
                                    structure = {
                                        "num_qubits": int(getattr(qc, "num_qubits", 0)),
                                        "depth": int(qc.depth()) if hasattr(qc, "depth") else None,
                                        "total_gates": int(len(getattr(qc, "data", []))),
                                        "gate_counts": _gate_counts(qc),
                                    }
                                except Exception:
                                    structure = {}
                                item["structure"] = structure
                            # SRE from precomputed stats if available
                            sre_entry = None
                            if sre_list and idx_in_file < len(sre_list):
                                try:
                                    _, stats = sre_list[idx_in_file]
                                    sre_entry = {
                                        "total": float(stats.get("sre_total")) if isinstance(stats, dict) and (stats.get("sre_total") == stats.get("sre_total")) else None,
                                        "per_qubit": list(stats.get("sre_per_qubit")) if isinstance(stats, dict) and isinstance(stats.get("sre_per_qubit"), list) else None,
                                        "alpha": 2.0,
                                        "source": "precomputed",
                                    }
                                except Exception:
                                    sre_entry = None
                            item["sre"] = sre_entry
                            out["items"].append(item)
                        elif isinstance(it, dict):
                            qasm = it.get("qasm")
                            item = {
                                "index": int(cursor),
                                "label": None,
                                "pred": int(preds[cursor]),
                                "prob": float(probs[cursor]),
                                "qasm": qasm,
                            }
                            qc = _load_circuit(qasm) if isinstance(qasm, str) else None
                            if qc is not None:
                                try:
                                    structure = {
                                        "num_qubits": int(getattr(qc, "num_qubits", 0)),
                                        "depth": int(qc.depth()) if hasattr(qc, "depth") else None,
                                        "total_gates": int(len(getattr(qc, "data", []))),
                                        "gate_counts": _gate_counts(qc),
                                    }
                                except Exception:
                                    structure = {}
                                item["structure"] = structure
                            sre_entry = None
                            if sre_list and idx_in_file < len(sre_list):
                                try:
                                    _, stats = sre_list[idx_in_file]
                                    sre_entry = {
                                        "total": float(stats.get("sre_total")) if isinstance(stats, dict) and (stats.get("sre_total") == stats.get("sre_total")) else None,
                                        "per_qubit": list(stats.get("sre_per_qubit")) if isinstance(stats, dict) and isinstance(stats.get("sre_per_qubit"), list) else None,
                                        "alpha": 2.0,
                                        "source": "precomputed",
                                    }
                                except Exception:
                                    sre_entry = None
                            item["sre"] = sre_entry
                            out["items"].append(item)
                        else:
                            out["items"].append({
                                "index": int(cursor),
                                "label": int(targets[cursor]) if targets[cursor] == targets[cursor] else None,
                                "pred": int(preds[cursor]),
                                "prob": float(probs[cursor]),
                            })
                    cursor += 1
            _save_json(save_misclassified, out)
        except Exception:
            pass

    print(json.dumps({"metrics": metrics, "saved": out_json}, indent=2))
    return metrics


def _parse_args() -> Dict[str, Any]:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate saved CircuitGNN checkpoint on PKL circuits.")
    parser.add_argument('--model', type=str, default='/data/P70087789/GNN/data/dataset_classification/models/classification_model_new.pt', help='Path to .pt checkpoint saved by train script')
    parser.add_argument('--pkl', type=str, nargs='+', default=['/data/P70087789/GNN/data/dataset_classification/clifford_evolved.pkl'], help='One or more PKL files to evaluate')
    parser.add_argument('--out', type=str, default='/data/P70087789/GNN/data/dataset_classification/results/eval_clifford_evolved.json')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--global-feature-variant', type=str, default='baseline')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--model2', type=str, default='', help='Optional baseline model to compare (dashed lines)')
    parser.add_argument('--save-misclassified', type=str, default='', help='Optional path to save misclassified items JSON')
    return vars(parser.parse_args())


if __name__ == '__main__':
    cfg = _parse_args()
    evaluate(
        model_path=cfg['model'],
        pkl_paths=cfg['pkl'],
        out_json=cfg['out'],
        batch_size=int(cfg['batch_size']),
        global_feature_variant=str(cfg['global_feature_variant']),
        threshold=float(cfg['threshold']),
        compare_model_path=str(cfg.get('model2') or ""),
        save_misclassified=str(cfg.get('save_misclassified') or ""),
    )


