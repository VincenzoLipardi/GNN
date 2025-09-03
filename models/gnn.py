import os
import sys
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import random_split


from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.loader import DataLoader


# Support running as a script or module
try:
    from .graph_representation import QuantumCircuitGraphDataset  # type: ignore
except Exception:
    try:
        from models.graph_representation import QuantumCircuitGraphDataset  # type: ignore
    except Exception:
        # Add project root to sys.path, then retry
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        from models.graph_representation import QuantumCircuitGraphDataset  # type: ignore


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
    """GNN with 3 TransformerConv layers, global branch, and regressor."""

    def __init__(
        self,
        node_in_dim: int = 12,
        gnn_hidden: int = 64,
        gnn_heads: int = 4,
        global_in_dim: int = 8,
        global_hidden: int = 64,
        reg_hidden: int = 128,
    ):
        super().__init__()

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

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x_pool = global_mean_pool(x, batch)

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


def _cache_root_for_paths(paths: List[str]) -> str:
    canonical = "|".join(sorted(os.path.abspath(p) for p in paths))
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    return os.path.join(os.getcwd(), f"pyg_cache_{digest}")


def build_dataloader(
    pkl_paths: List[str],
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root = _cache_root_for_paths(pkl_paths)
    dataset = QuantumCircuitGraphDataset(root=root, pkl_paths=pkl_paths)

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check PKL paths and formats.")

    test_len = max(1, int(len(dataset) * test_split))
    val_len = max(1, int(len(dataset) * val_split))
    train_len = max(1, len(dataset) - val_len - test_len)
    # Adjust to exact total
    while train_len + val_len + test_len > len(dataset):
        train_len -= 1
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train(
    pkl_paths: List[str],
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
):
    train_loader, val_loader, test_loader = build_dataloader(pkl_paths, batch_size=batch_size, val_split=0.2, test_split=0.1)

    model = CircuitGNN()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)

    criterion = nn.HuberLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(dev)
            pred = model(batch)
            if getattr(batch, 'y', None) is None:
                raise RuntimeError("Labels 'y' are missing in dataset. Ensure PKLs are labeled.")
            loss = criterion(pred, batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev)
                pred = model(batch)
                loss = criterion(pred, batch.y.view(-1))
                val_loss += loss.item() * batch.num_graphs

        print(f"Epoch {epoch:03d} | TrainLoss {total_loss/len(train_loader.dataset):.4f} | "
              f"ValLoss {val_loss/len(val_loader.dataset):.4f}")

    # Test evaluation
    model.eval()
    criterion = nn.HuberLoss()
    test_loss = 0.0
    mae = 0.0
    n = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(dev)
            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1))
            test_loss += loss.item() * batch.num_graphs
            mae += torch.mean(torch.abs(pred - batch.y.view(-1))).item() * batch.num_graphs
            n += batch.num_graphs
    print(f"TestLoss {test_loss/max(1,n):.4f} | TestMAE {mae/max(1,n):.4f}")

    return model


def collect_all_random_pkls(base_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(base_dir)):
        if name.endswith('.pkl') and 'basis_rotations+cx_qubits_' in name:
            files.append(os.path.join(base_dir, name))
    if not files:
        raise FileNotFoundError(f"No PKL files found in {base_dir}")
    return files


def collect_pkls_by_qubits(base_dir: str, qubits: List[int]) -> List[str]:
    selected = []
    for q in qubits:
        key = f"qubits_{q}_"
        for name in sorted(os.listdir(base_dir)):
            if name.endswith('.pkl') and 'basis_rotations+cx_qubits_' in name and key in name:
                selected.append(os.path.join(base_dir, name))
    if not selected:
        raise FileNotFoundError(f"No PKLs for qubits {qubits} in {base_dir}")
    return selected


def build_train_test_loaders(
    pkl_paths: List[str],
    train_split: float = 0.8,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    root = _cache_root_for_paths(pkl_paths)
    dataset = QuantumCircuitGraphDataset(root=root, pkl_paths=pkl_paths)
    if len(dataset) < 2:
        raise RuntimeError("Dataset too small to split. Check PKLs.")
    train_len = max(1, int(len(dataset) * train_split))
    test_len = max(1, len(dataset) - train_len)
    # Adjust
    while train_len + test_len > len(dataset):
        train_len -= 1
    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [train_len, test_len], generator=generator)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def subsample_paths(paths: List[str], fraction: float, seed: int = 42) -> List[str]:
    import random
    rnd = random.Random(seed)
    n = max(1, int(len(paths) * fraction))
    return rnd.sample(paths, n)


@torch.no_grad()
def evaluate_by_qubits(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[int, float]:
    model.eval()
    mse_per_q: Dict[int, List[float]] = {}
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(-1)
        se = (pred - y) ** 2
        qubits = batch.num_qubits.view(-1).tolist()
        for i, q in enumerate(qubits):
            mse_per_q.setdefault(int(q), []).append(float(se[i].item()))
    return {q: float(torch.tensor(vals).mean().item()) for q, vals in mse_per_q.items()}


def plot_mse_histogram(results: Dict[str, Dict[int, float]], save_path: str):
    import numpy as np
    import matplotlib.pyplot as plt

    xs = [2, 3, 4, 5, 6]
    train_vals = [results.get('train', {}).get(k, np.nan) for k in xs]
    test_vals = [results.get('test', {}).get(k, np.nan) for k in xs]
    extra_vals = [results.get('extrapolation', {}).get(k, np.nan) for k in xs]

    width = 0.25
    x_idx = np.arange(len(xs))

    plt.figure(figsize=(10, 5))
    plt.bar(x_idx - width, train_vals, width=width, label='Train MSE')
    plt.bar(x_idx, test_vals, width=width, label='Test MSE')
    plt.bar(x_idx + width, extra_vals, width=width, label='Extrapolation MSE')
    plt.xticks(x_idx, xs)
    plt.xlabel('Number of Qubits')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE by Qubit Count (Train/Test/Extrapolation)')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_model_mse_bars(model_name: str, train_mse: float, test_mse: float, extra_mse: float, save_path: str):
    import numpy as np
    import matplotlib.pyplot as plt

    xs = [model_name]
    width = 0.25
    x_idx = np.arange(len(xs))

    plt.figure(figsize=(6, 5))
    plt.bar(x_idx - width, [train_mse], width=width, label='Train MSE')
    plt.bar(x_idx, [test_mse], width=width, label='Test MSE')
    plt.bar(x_idx + width, [extra_mse], width=width, label='Extrapolation MSE')
    plt.xticks(x_idx, xs)
    plt.ylabel('Mean Squared Error')
    plt.title('Model Performance (ID vs OOD)')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def evaluate_overall_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_se = 0.0
    total_n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(-1)
        se = torch.sum((pred - y) ** 2).item()
        total_se += se
        total_n += y.numel()
    return total_se / max(1, total_n)


def save_checkpoint(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint_if_exists(model: nn.Module, path: str) -> bool:
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return True
    return False


def grid_search(
    pkl_paths: List[str],
    configs: List[Dict[str, int]],
    train_split: float = 0.8,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> List[Dict[str, float]]:
    train_loader, val_loader = build_train_test_loaders(pkl_paths, train_split=train_split, batch_size=batch_size, seed=seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.HuberLoss()
    results = []

    for cfg in configs:
        model = CircuitGNN(
            node_in_dim=12,
            gnn_hidden=cfg.get('gnn_hidden', 64),
            gnn_heads=cfg.get('gnn_heads', 4),
            global_in_dim=8,
            global_hidden=cfg.get('global_hidden', 64),
            reg_hidden=cfg.get('reg_hidden', 128),
        ).to(dev)
        optimizer = Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(dev)
                pred = model(batch)
                loss = criterion(pred, batch.y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        val_mse = evaluate_overall_mse(model, val_loader, dev)
        res = {**cfg, 'val_mse': float(val_mse)}
        results.append(res)

    results.sort(key=lambda r: r['val_mse'])
    return results


def save_grid_table(results: List[Dict[str, float]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = ['gnn_hidden', 'gnn_heads', 'global_hidden', 'reg_hidden', 'val_mse']
    lines = ["\t".join(headers)]
    for r in results:
        line = "\t".join(str(r.get(h)) for h in headers)
        lines.append(line)
    with open(path, 'w') as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    base = "/Users/vlipardi/Documents/Github/ML/GNN/data/dataset_random"
    # Collect in-distribution files (2-5 qubits)
    in_dist_pkls_full = collect_pkls_by_qubits(base, [2, 3, 4, 5])

    # 1) GRID SEARCH on 50% random subset
    sample_pkls = subsample_paths(in_dist_pkls_full, fraction=0.5, seed=42)
    configs = [
        {'gnn_hidden': 32, 'gnn_heads': 2, 'global_hidden': 32, 'reg_hidden': 64},
        {'gnn_hidden': 64, 'gnn_heads': 2, 'global_hidden': 64, 'reg_hidden': 128},
        {'gnn_hidden': 64, 'gnn_heads': 4, 'global_hidden': 64, 'reg_hidden': 128},
        {'gnn_hidden': 128, 'gnn_heads': 4, 'global_hidden': 128, 'reg_hidden': 256},
    ]
    grid_results = grid_search(sample_pkls, configs, train_split=0.8, epochs=10, batch_size=64, lr=1e-3, seed=42)
    table_path = os.path.join(os.getcwd(), 'results', 'grid_search.tsv')
    save_grid_table(grid_results, table_path)
    best_cfg = {k: grid_results[0][k] for k in ['gnn_hidden', 'gnn_heads', 'global_hidden', 'reg_hidden']}
    print(f"Best config: {best_cfg} | Val MSE: {grid_results[0]['val_mse']:.6f}. Table saved to {table_path}")

    # 2) TRAIN BEST on full in-distribution data and evaluate
    train_loader, test_loader = build_train_test_loaders(in_dist_pkls_full, train_split=0.8, batch_size=64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CircuitGNN(
        gnn_hidden=int(best_cfg['gnn_hidden']),
        gnn_heads=int(best_cfg['gnn_heads']),
        global_hidden=int(best_cfg['global_hidden']),
        reg_hidden=int(best_cfg['reg_hidden']),
    ).to(dev)
    criterion = nn.HuberLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(dev)
            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        print(f"Epoch {epoch:03d} | TrainLoss {total_loss/len(train_loader.dataset):.4f}")

    train_mse_overall = evaluate_overall_mse(model, train_loader, dev)
    test_mse_overall = evaluate_overall_mse(model, test_loader, dev)

    # 3) OOD on 6-qubit circuits
    ex_pkls = collect_pkls_by_qubits(base, [6])
    _, extra_loader = build_train_test_loaders(ex_pkls, train_split=0.0, batch_size=64)
    extra_mse_overall = evaluate_overall_mse(model, extra_loader, dev)

    # Final bar plot (single model)
    out_bar = os.path.join(os.getcwd(), 'results', 'model_mse_bars.png')
    plot_model_mse_bars('CircuitGNN', train_mse_overall, test_mse_overall, extra_mse_overall, out_bar)
    print(f"Saved model bar histogram to {out_bar} and grid table to {table_path}")

