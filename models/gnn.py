import os
import sys
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
try:
    from torch.amp import autocast, GradScaler  # type: ignore[attr-defined]
    _AMP_DEVICE_TYPE = 'cuda'
except Exception:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _AMP_DEVICE_TYPE = 'cuda'

# Ensure project root on sys.path when executed directly
try:
    from .graph_representation import QuantumCircuitGraphDataset
except Exception:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from models.graph_representation import QuantumCircuitGraphDataset




# =========================================
# Data utils (cache path)
# =========================================
def _cache_root_for_paths(paths: List[str], suffix: str = "") -> str:
    canonical = "|".join(sorted(os.path.abspath(p) for p in paths))
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    tag = f"_{suffix}" if suffix else ""
    return os.path.join(os.getcwd(), f"pyg_cache_{digest}{tag}")



# =========================================
# Architecture (GNN model)
# =========================================
GNN_HIDDEN = 32
GNN_HEADS = 2
GLOBAL_HIDDEN = 64
REG_HIDDEN = 128


class GlobalMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, g: Tensor) -> Tensor:
        return self.net(g)


class RegressorHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class CircuitGNN(nn.Module):
    """GNN with TransformerConv layers, global branch, and regressor."""

    def __init__(
        self,
        node_in_dim: int = 13,
        gnn_hidden: int = GNN_HIDDEN,
        gnn_heads: int = GNN_HEADS,
        global_in_dim: int = 8,
        global_hidden: int = GLOBAL_HIDDEN,
        reg_hidden: int = REG_HIDDEN,
    ):
        super().__init__()

        from torch_geometric.nn import TransformerConv, global_mean_pool  # lazy import to avoid hard dep at import time

        self.global_mean_pool = global_mean_pool

        self.gnn_hidden = gnn_hidden
        self.gnn_heads = gnn_heads
        self.conv1 = TransformerConv(node_in_dim, gnn_hidden, heads=gnn_heads, dropout=0.0, beta=False)
        self.conv2 = TransformerConv(gnn_hidden * gnn_heads, gnn_hidden, heads=gnn_heads, dropout=0.0, beta=False)
        self.conv3 = TransformerConv(gnn_hidden * gnn_heads, gnn_hidden, heads=gnn_heads, dropout=0.0, beta=False)

        self.global_mlp = GlobalMLP(global_in_dim, global_hidden)
        concat_dim = gnn_hidden * gnn_heads + global_hidden
        self.regressor = RegressorHead(concat_dim, reg_hidden)

    def forward(self, data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # Early handle graphs with zero nodes to avoid NaNs in TransformerConv
        if x.size(0) == 0:
            num_graphs = getattr(data, 'num_graphs', 1)
            x_pool = torch.zeros((num_graphs, self.gnn_hidden * self.gnn_heads), device=x.device, dtype=x.dtype)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x_pool = self.global_mean_pool(x, batch)

        # Reshape concatenated global features back to [num_graphs, feat_dim]
        g_raw = data.global_features
        if g_raw.dim() == 1:
            num_graphs = getattr(data, 'num_graphs', int(batch.max().item()) + 1 if batch.numel() > 0 else 1)
            g_raw = g_raw.view(num_graphs, -1)
        g_feat = self.global_mlp(g_raw)

        # Handle graphs with zero nodes which disappear from the pooled tensor
        if x_pool.size(0) < g_feat.size(0):
            pad_rows = g_feat.size(0) - x_pool.size(0)
            pad = torch.zeros((pad_rows, x_pool.size(1)), device=x_pool.device, dtype=x_pool.dtype)
            x_pool = torch.cat([x_pool, pad], dim=0)
        elif x_pool.size(0) > g_feat.size(0):
            pad_rows = x_pool.size(0) - g_feat.size(0)
            pad = torch.zeros((pad_rows, g_feat.size(1)), device=g_feat.device, dtype=g_feat.dtype)
            g_feat = torch.cat([g_feat, pad], dim=0)
        h = torch.cat([x_pool, g_feat], dim=-1)
        out = self.regressor(h)
        return out.view(-1)


# =========================================
# Loaders
# =========================================
def build_dataloader(
    pkl_paths: List[str],
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    root = _cache_root_for_paths(pkl_paths, suffix=suffix)
    dataset = QuantumCircuitGraphDataset(
        root=root,
        pkl_paths=pkl_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check PKL paths and formats.")

    test_len = max(1, int(len(dataset) * test_split))
    val_len = max(1, int(len(dataset) * val_split))
    train_len = max(1, len(dataset) - val_len - test_len)
    while train_len + val_len + test_len > len(dataset):
        train_len -= 1
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    # Efficient loader defaults
    num_cpus = os.cpu_count() or 0
    default_workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=default_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem)
    return train_loader, val_loader, test_loader


def build_train_test_loaders(
    pkl_paths: List[str],
    train_split: float = 0.8,
    batch_size: int = 64,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    root = _cache_root_for_paths(pkl_paths, suffix=suffix)
    dataset = QuantumCircuitGraphDataset(
        root=root,
        pkl_paths=pkl_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )
    if len(dataset) < 2:
        raise RuntimeError("Dataset too small to split. Check PKLs.")
    train_len = max(1, int(len(dataset) * train_split))
    test_len = max(1, len(dataset) - train_len)
    while train_len + test_len > len(dataset):
        train_len -= 1
    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [train_len, test_len], generator=generator)
    num_cpus = os.cpu_count() or 0
    default_workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=default_workers, pin_memory=pin_mem),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem),
    )


def build_full_loader(
    pkl_paths: List[str],
    batch_size: int = 64,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: Optional[str] = None,
):
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    root = _cache_root_for_paths(pkl_paths, suffix=suffix)
    dataset = QuantumCircuitGraphDataset(
        root=root,
        pkl_paths=pkl_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check PKL paths and formats.")
    num_cpus = os.cpu_count() or 0
    default_workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem)


def build_train_val_test_loaders_two_stage(
    pkl_paths: List[str],
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: Optional[str] = None,
):
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}"
    root = _cache_root_for_paths(pkl_paths, suffix=suffix)
    dataset = QuantumCircuitGraphDataset(
        root=root,
        pkl_paths=pkl_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )
    if len(dataset) < 3:
        raise RuntimeError("Dataset too small for train/val/test splitting.")

    generator = torch.Generator().manual_seed(seed)
    primary_train_len = max(1, int(len(dataset) * train_split))
    test_len = max(1, len(dataset) - primary_train_len)
    while primary_train_len + test_len > len(dataset):
        primary_train_len -= 1

    primary_train, test_ds = random_split(dataset, [primary_train_len, test_len], generator=generator)
    val_len = max(1, int(len(primary_train) * val_within_train))
    real_train_len = max(1, len(primary_train) - val_len)
    train_ds, val_ds = random_split(primary_train, [real_train_len, val_len], generator=generator)

    num_cpus = os.cpu_count() or 0
    default_workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=default_workers, pin_memory=pin_mem),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem),
    )


# =========================================
# Training
# =========================================
def train(
    pkl_paths: List[str],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: Optional[str] = None,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.0,
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    train_loader, val_loader, test_loader = build_dataloader(
        pkl_paths,
        batch_size=batch_size,
        val_split=0.2,
        test_split=0.1,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )

    gdim = 152 if global_feature_variant == "binned152" else 8
    model_args = dict(global_in_dim=gdim)
    # Determine node feature dimension
    node_in_dim = 13 + (7 if node_feature_backend_variant else 0)
    model_args.update(dict(node_in_dim=node_in_dim))
    if model_kwargs:
        model_args.update(model_kwargs)
        model_args.setdefault('global_in_dim', gdim)
        model_args.setdefault('node_in_dim', node_in_dim)
    model = CircuitGNN(**model_args)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    model.to(dev)

    criterion = nn.HuberLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    best_state = None
    epochs_without_improve = 0

    # Initialize AMP scaler (fallback to CUDA AMP if torch.amp not available)
    try:
        scaler = GradScaler(device=_AMP_DEVICE_TYPE, enabled=(dev.type == 'cuda'))
    except TypeError:
        scaler = GradScaler(enabled=(dev.type == 'cuda'))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(dev, non_blocking=True)
            if getattr(batch, 'y', None) is None:
                raise RuntimeError("Labels 'y' are missing in dataset. Ensure PKLs are labeled.")
            optimizer.zero_grad(set_to_none=True)
            with autocast(_AMP_DEVICE_TYPE, enabled=(dev.type == 'cuda')):
                pred = model(batch)
                target = batch.y.view(-1)
                mask = torch.isfinite(target)
                if mask.sum() == 0:
                    continue
                loss = criterion(pred[mask], target[mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach().item() * int(mask.sum().item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev, non_blocking=True)
                with autocast(_AMP_DEVICE_TYPE, enabled=(dev.type == 'cuda')):
                    pred = model(batch)
                    target = batch.y.view(-1)
                    mask = torch.isfinite(target)
                    if mask.sum() == 0:
                        continue
                    loss = criterion(pred[mask], target[mask])
                val_loss += loss.detach().item() * int(mask.sum().item())

        train_loss_epoch = total_loss/max(1, len(train_loader.dataset))
        val_loss_epoch = val_loss/max(1, len(val_loader.dataset))
        print(f"Epoch {epoch:03d} | TrainLoss {train_loss_epoch:.4f} | ValLoss {val_loss_epoch:.4f}")

        if val_loss_epoch + early_stopping_min_delta < best_val:
            best_val = val_loss_epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch:03d} (best ValLoss {best_val:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    criterion = nn.HuberLoss()
    test_loss = 0.0
    mae = 0.0
    n = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(dev, non_blocking=True)
            with autocast(_AMP_DEVICE_TYPE, enabled=(dev.type == 'cuda')):
                pred = model(batch)
                target = batch.y.view(-1)
                mask = torch.isfinite(target)
                if mask.sum() == 0:
                    continue
                loss = criterion(pred[mask], target[mask])
            test_loss += loss.detach().item() * int(mask.sum().item())
            mae += torch.mean(torch.abs(pred[mask] - target[mask])).item() * int(mask.sum().item())
            n += int(mask.sum().item())
    print(f"TestLoss {test_loss/max(1,n):.4f} | TestMAE {mae/max(1,n):.4f}")

    return model


@torch.no_grad()
def evaluate_overall_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_se = 0.0
    total_n = 0
    use_amp = (device.type == 'cuda')
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with autocast(_AMP_DEVICE_TYPE, enabled=use_amp):
            pred = model(batch)
            y = batch.y.view(-1)
            mask = torch.isfinite(y)
            if mask.sum() == 0:
                continue
            se = torch.sum((pred[mask] - y[mask]) ** 2).item()
        total_se += se
        total_n += int(mask.sum().item())
    return total_se / max(1, total_n)


def grid_search(
    pkl_paths: List[str],
    configs: List[Dict[str, int]],
    train_split: float = 0.8,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: Optional[str] = None,
) -> List[Dict[str, float]]:
    train_loader, val_loader = build_train_test_loaders(
        pkl_paths,
        train_split=train_split,
        batch_size=batch_size,
        seed=seed,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.HuberLoss()
    results = []

    for cfg in configs:
        model = CircuitGNN(
            node_in_dim=13 + (7 if node_feature_backend_variant else 0),
            gnn_hidden=cfg.get('gnn_hidden', GNN_HIDDEN),
            gnn_heads=cfg.get('gnn_heads', GNN_HEADS),
            global_in_dim=(152 if global_feature_variant == "binned152" else 8),
            global_hidden=cfg.get('global_hidden', GLOBAL_HIDDEN),
            reg_hidden=cfg.get('reg_hidden', REG_HIDDEN),
        ).to(dev)
        optimizer = Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(dev)
                pred = model(batch)
                target = batch.y.view(-1)
                mask = torch.isfinite(target)
                if mask.sum() == 0:
                    continue
                loss = criterion(pred[mask], target[mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        val_mse = evaluate_overall_mse(model, val_loader, dev)
        res = {**cfg, 'val_mse': float(val_mse)}
        results.append(res)

    results.sort(key=lambda r: r['val_mse'])
    return results


def train_with_two_stage_split(
    pkl_paths: List[str],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
    global_feature_variant: str = "binned152",
    node_feature_backend_variant: Optional[str] = None,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    model_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
):
    train_loader, val_loader, test_loader = build_train_val_test_loaders_two_stage(
        pkl_paths,
        train_split=train_split,
        val_within_train=val_within_train,
        batch_size=batch_size,
        seed=seed,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
    )

    gdim = 152 if global_feature_variant == "binned152" else 8
    model_args = dict(global_in_dim=gdim)
    node_in_dim = 13 + (7 if node_feature_backend_variant else 0)
    model_args.update(dict(node_in_dim=node_in_dim))
    if model_kwargs:
        model_args.update(model_kwargs)
        model_args.setdefault('global_in_dim', gdim)
        model_args.setdefault('node_in_dim', node_in_dim)
    model = CircuitGNN(**model_args)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    model.to(dev)

    criterion = nn.HuberLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    best_state = None
    epochs_without_improve = 0

    try:
        scaler = GradScaler(device=_AMP_DEVICE_TYPE, enabled=(dev.type == 'cuda'))
    except TypeError:
        scaler = GradScaler(enabled=(dev.type == 'cuda'))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(dev, non_blocking=True)
            if getattr(batch, 'y', None) is None:
                raise RuntimeError("Labels 'y' are missing in dataset. Ensure PKLs are labeled.")
            optimizer.zero_grad(set_to_none=True)
            with autocast(_AMP_DEVICE_TYPE, enabled=(dev.type == 'cuda')):
                pred = model(batch)
                target = batch.y.view(-1)
                mask = torch.isfinite(target)
                if mask.sum() == 0:
                    continue
                loss = criterion(pred[mask], target[mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach().item() * batch.num_graphs

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev, non_blocking=True)
                with autocast(_AMP_DEVICE_TYPE, enabled=(dev.type == 'cuda')):
                    pred = model(batch)
                    target = batch.y.view(-1)
                    mask = torch.isfinite(target)
                    if mask.sum() == 0:
                        continue
                    loss = criterion(pred[mask], target[mask])
                val_loss += loss.detach().item() * batch.num_graphs

        train_loss_epoch = total_loss/max(1, len(train_loader.dataset))
        val_loss_epoch = val_loss/max(1, len(val_loader.dataset))
        print(f"Epoch {epoch:03d} | TrainLoss {train_loss_epoch:.4f} | ValLoss {val_loss_epoch:.4f}")

        if val_loss_epoch + early_stopping_min_delta < best_val:
            best_val = val_loss_epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch:03d} (best ValLoss {best_val:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loader, val_loader, test_loader, dev
    

