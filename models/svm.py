import os
import sys
from pathlib import Path
from typing import List
import json

import numpy as np
from joblib import dump

# Ensure project root on sys.path so absolute imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gnn import build_full_loader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
import torch


def main() -> None:
    # Balanced two-class dataset paths
    pkls: List[str] = [
        "/data/P70087789/GNN/data/dataset_classification/magic_18.pkl",
        "/data/P70087789/GNN/data/dataset_classification/stabilizer_18.pkl",
    ]
    # Helper to collect features/labels from a loader
    def _collect_xy(loader_):
        xs, ys = [], []
        for batch in loader_:
            g = batch.global_features
            y = batch.y.view(-1)
            if g.dim() == 1:
                num_graphs = getattr(batch, 'num_graphs', int(y.shape[0]))
                g = g.view(int(num_graphs), -1)
            mask = torch.isfinite(y)
            if mask.sum() == 0:
                continue
            xs.append(g[mask].cpu().numpy().astype(np.float32))
            ys.append(y[mask].cpu().numpy().astype(np.float32))
        if not xs:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    # Load each class separately to enforce exact per-class split (5000 train, 2500 test)
    magic_loader = build_full_loader([pkls[0]], batch_size=256, global_feature_variant='binned152', node_feature_backend_variant=None)
    stab_loader = build_full_loader([pkls[1]], batch_size=256, global_feature_variant='binned152', node_feature_backend_variant=None)
    X_magic, y_magic = _collect_xy(magic_loader)
    X_stab, y_stab = _collect_xy(stab_loader)

    # Deterministic balanced split: 5000 per class train, 2500 per class test
    rng = np.random.RandomState(42)
    idx_magic = rng.permutation(X_magic.shape[0])
    idx_stab = rng.permutation(X_stab.shape[0])
    m_tr, m_te = idx_magic[:5000], idx_magic[5000:7500]
    s_tr, s_te = idx_stab[:5000], idx_stab[5000:7500]

    X_train = np.concatenate([X_magic[m_tr], X_stab[s_tr]], axis=0)
    y_train = np.concatenate([y_magic[m_tr], y_stab[s_tr]], axis=0)
    X_test = np.concatenate([X_magic[m_te], X_stab[s_te]], axis=0)
    y_test = np.concatenate([y_magic[m_te], y_stab[s_te]], axis=0)

    # Ensure binary labels {0,1}
    def _to_binary(y):
        y = np.asarray(y)
        uniq = np.unique(y)
        if set(uniq.tolist()) <= {0, 1}:
            return y.astype(int)
        return (y >= 0.5).astype(int)

    y_train_bin = _to_binary(y_train)
    y_test_bin = _to_binary(y_test)

    # Train SVM with probabilities for AUC/AP
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True),
    )
    model.fit(X_train, y_train_bin)

    # Train metrics
    ytr_pred = model.predict(X_train)
    try:
        ytr_scores = model.predict_proba(X_train)[:, 1]
    except Exception:
        ytr_scores = model.decision_function(X_train)
    acc_train = accuracy_score(y_train_bin, ytr_pred)
    pr_train, rc_train, f1_train, _ = precision_recall_fscore_support(y_train_bin, ytr_pred, average='binary', zero_division=0)

    # Test metrics
    yte_pred = model.predict(X_test)
    try:
        yte_scores = model.predict_proba(X_test)[:, 1]
    except Exception:
        yte_scores = model.decision_function(X_test)
    acc_test = accuracy_score(y_test_bin, yte_pred)
    pr_test, rc_test, f1_test, _ = precision_recall_fscore_support(y_test_bin, yte_pred, average='binary', zero_division=0)
    try:
        auc_test = roc_auc_score(y_test_bin, yte_scores)
    except Exception:
        auc_test = float('nan')
    try:
        ap_test = average_precision_score(y_test_bin, yte_scores)
    except Exception:
        ap_test = float('nan')

    print("\n=== Summary (Balanced 10k/5k split) ===")
    print(f"Train - Acc: {acc_train:.4f}, Prec: {pr_train:.4f}, Rec: {rc_train:.4f}, F1: {f1_train:.4f}")
    print(f"Test  - Acc: {acc_test:.4f}, Prec: {pr_test:.4f}, Rec: {rc_test:.4f}, F1: {f1_test:.4f}, AUC: {auc_test:.4f}, AP: {ap_test:.4f}")

    # Save trained model pipeline
    save_path = "/data/P70087789/GNN/data/dataset_classification/models/svm_model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump(model, save_path)
    print(f"Saved model to {save_path}")

    # OOD evaluation on clifford_evolved.pkl
    clifford_paths: List[str] = [
        "/data/P70087789/GNN/data/dataset_classification/clifford_evolved.pkl",
    ]
    loader = build_full_loader(
        clifford_paths,
        batch_size=512,
        global_feature_variant='binned152',
        node_feature_backend_variant=None,
    )

    def _collect_xy(loader_):
        xs, ys = [], []
        for batch in loader_:
            g = batch.global_features
            y = batch.y.view(-1)
            if g.dim() == 1:
                num_graphs = getattr(batch, 'num_graphs', int(y.shape[0]))
                g = g.view(int(num_graphs), -1)
            mask = torch.isfinite(y)
            if mask.sum() == 0:
                continue
            xs.append(g[mask].cpu().numpy().astype(np.float32))
            ys.append(y[mask].cpu().numpy().astype(np.float32))
        if not xs:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    X_ood, y_ood = _collect_xy(loader)
    if X_ood.size == 0:
        print("[WARN] No data loaded from clifford_evolved.pkl for evaluation.")
        return
    # Ensure binary labels
    y_ood_bin = y_ood
    uniq = set(np.unique(y_ood_bin).tolist())
    if not (uniq <= {0, 1}):
        y_ood_bin = (y_ood_bin >= 0.5).astype(int)

    y_pred = model.predict(X_ood)
    # Scores for AUC/AP
    try:
        scores = model.predict_proba(X_ood)[:, 1]
    except Exception:
        scores = model.decision_function(X_ood)

    acc = accuracy_score(y_ood_bin, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_ood_bin, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_ood_bin, scores)
    except Exception:
        auc = float('nan')
    try:
        ap = average_precision_score(y_ood_bin, scores)
    except Exception:
        ap = float('nan')
    cm = confusion_matrix(y_ood_bin, y_pred, labels=[0, 1])

    print("\n=== OOD: clifford_evolved.pkl ===")
    print(f"Acc: {acc:.4f}, Prec: {pr:.4f}, Rec: {rc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Save OOD evaluation metrics JSON
    results_dir = "/data/P70087789/GNN/data/dataset_classification/results"
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "svm_eval_clifford_evolved.json")
    to_save = {
        "dataset": "clifford_evolved.pkl",
        "metrics": {
            "accuracy": float(acc),
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(f1),
            "roc_auc": float(auc),
            "avg_precision": float(ap),
            "confusion_matrix": cm.tolist() if hasattr(cm, 'tolist') else cm,
        },
    }
    with open(out_json, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"Saved OOD evaluation JSON to {out_json}")


if __name__ == "__main__":
    main()
