from __future__ import annotations

from typing import Sequence
import math

import torch
import torch.nn as nn


__all__ = [
    "GlobalMLPRegressor",
    "build_global_mlp",
    "train_with_two_stage_split_mlp",
    "evaluate_overall_mse_mlp",
]


def _get_activation(activation: str) -> nn.Module:
    """Return an activation module by name.

    Supported: "relu", "gelu", "leaky_relu", "elu", "tanh", "sigmoid".
    Defaults to ReLU if an unknown name is provided.
    """
    name = activation.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if name == "elu":
        return nn.ELU(alpha=1.0, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.ReLU(inplace=True)


class GlobalMLPRegressor(nn.Module):
    """Fully-connected regressor for global features.

    This module implements a multi-layer perceptron that consumes a
    152-dimensional global feature vector (no graph encoding) and outputs a
    single scalar regression prediction (e.g., stabilizer Rényi entropy).

    Parameters
    ----------
    input_dim: int
        Dimension of the input feature vector. Defaults to 152.
    hidden_layers: Sequence[int]
        Sizes of hidden layers. If empty, a single Linear(input_dim -> 1) is used.
    activation: str
        Activation function name ("relu", "gelu", "leaky_relu", "elu", "tanh", "sigmoid").
    dropout: float
        Dropout probability applied after each activation. Set 0.0 to disable.
    use_batchnorm: bool
        If True, applies BatchNorm1d after each Linear (before activation).
    """

    def __init__(
        self,
        input_dim: int = 152,
        hidden_layers: Sequence[int] = (64, 64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_layers = tuple(int(h) for h in hidden_layers)
        self.use_batchnorm = bool(use_batchnorm)

        activation_layer = _get_activation(activation)

        feature_layers: list[nn.Module] = []

        previous_dim = self.input_dim
        for hidden_dim in self.hidden_layers:
            feature_layers.append(nn.Linear(previous_dim, hidden_dim))
            if self.use_batchnorm:
                feature_layers.append(nn.BatchNorm1d(hidden_dim))
            feature_layers.append(_get_activation(activation))
            if dropout and dropout > 0.0:
                feature_layers.append(nn.Dropout(p=float(dropout)))
            previous_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*feature_layers) if feature_layers else nn.Identity()

        last_dim = previous_dim
        self.regressor = nn.Linear(last_dim, 1)

        self._initialize_weights(activation_layer)

    def _initialize_weights(self, activation_module: nn.Module) -> None:
        """Initialize Linear layers with Kaiming/He initialization.

        The nonlinearity is inferred from the activation module when applicable.
        """
        nonlinearity = "relu"
        negative_slope = 0.01 if isinstance(activation_module, nn.LeakyReLU) else 0.0

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=negative_slope,
                    mode="fan_in",
                    nonlinearity=nonlinearity,
                )
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute regression prediction.

        Expects x with shape [batch_size, input_dim]. Returns a tensor with
        shape [batch_size], squeezing the final singleton dimension.
        """
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input of shape [batch, {self.input_dim}], got {tuple(x.shape)}"
            )

        features = self.feature_extractor(x)
        output = self.regressor(features)
        return output.squeeze(-1)


def build_global_mlp(
    input_dim: int = 152,
    hidden_layers: Sequence[int] = (64, 64, 64),
    activation: str = "relu",
    dropout: float = 0.0,
    use_batchnorm: bool = False,
) -> GlobalMLPRegressor:
    """Factory for a global-feature MLP regressor.

    Example
    -------
    >>> model = build_global_mlp()
    >>> x = torch.randn(8, 152)
    >>> y = model(x)
    >>> y.shape
    torch.Size([8])
    """
    return GlobalMLPRegressor(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
    )


# =========================================
# Training utilities for global-only MLP
# =========================================
from typing import Optional, Tuple, Dict, Any


def _reshape_global_features(batch: Any, g: torch.Tensor, expected_dim: int) -> torch.Tensor:
    """Ensure global features are shaped [num_graphs, expected_dim].

    PyG concatenates 1D tensors across graphs along dim=0 during batching,
    which can produce a flat vector of length num_graphs * expected_dim.
    This reshapes it back using batch.num_graphs when available.
    """
    if g.dim() == 2:
        return g
    if g.dim() == 1:
        num_graphs = getattr(batch, 'num_graphs', None)
        if num_graphs is None:
            if g.numel() % expected_dim != 0:
                raise ValueError(
                    f"Cannot infer num_graphs: global_features has {g.numel()} elements not divisible by {expected_dim}"
                )
            num_graphs = g.numel() // expected_dim
        return g.view(int(num_graphs), int(expected_dim))
    raise ValueError(f"Unexpected global_features dim {g.dim()} (expected 1 or 2)")


def _device_from(device: Optional[str] = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def train_with_two_stage_split_mlp(
    pkl_paths: Sequence[str],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
    global_feature_variant: str = "binned152",
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    model_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[GlobalMLPRegressor, Any, Any, Any, torch.device]:
    """Train MLP on global features using the same two-stage split as GNN.

    Returns (model, train_loader, val_loader, test_loader, device).
    """
    # Reuse dataset loaders from the GNN module to avoid duplication
    from .gnn import build_train_val_test_loaders_two_stage

    train_loader, val_loader, test_loader = build_train_val_test_loaders_two_stage(
        pkl_paths=pkl_paths,
        train_split=train_split,
        val_within_train=val_within_train,
        batch_size=batch_size,
        seed=seed,
        global_feature_variant=global_feature_variant,
    )

    gdim = 152 if global_feature_variant == "binned152" else 8
    kwargs: Dict[str, Any] = dict(input_dim=gdim)
    if model_kwargs:
        kwargs.update(model_kwargs)
        kwargs.setdefault("input_dim", gdim)
    model = GlobalMLPRegressor(**kwargs)

    dev = _device_from(device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model.to(dev)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    epochs_without_improve = 0

    use_amp = (dev.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(dev, non_blocking=True)
            if getattr(batch, "y", None) is None:
                raise RuntimeError("Labels 'y' are missing in dataset. Ensure PKLs are labeled.")
            optimizer.zero_grad(set_to_none=True)
            g = _reshape_global_features(batch, batch.global_features, model.input_dim)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(g)
                loss = criterion(pred, batch.y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach().item() * batch.num_graphs

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev, non_blocking=True)
                g = _reshape_global_features(batch, batch.global_features, model.input_dim)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(g)
                    loss = criterion(pred, batch.y.view(-1))
                val_loss += loss.detach().item() * batch.num_graphs

        train_loss_epoch = total_loss / max(1, len(train_loader.dataset))
        val_loss_epoch = val_loss / max(1, len(val_loader.dataset))
        print(f"[MLP] Epoch {epoch:03d} | TrainLoss {train_loss_epoch:.4f} | ValLoss {val_loss_epoch:.4f}")

        if val_loss_epoch + early_stopping_min_delta < best_val:
            best_val = val_loss_epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print(f"[MLP] Early stopping at epoch {epoch:03d} (best ValLoss {best_val:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loader, val_loader, test_loader, dev


@torch.no_grad()
def evaluate_overall_mse_mlp(model: nn.Module, loader: Any, device: torch.device) -> float:
    model.eval()
    total_se = 0.0
    total_n = 0
    use_amp = (device.type == "cuda")
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        expected_dim = getattr(model, 'input_dim', None)
        if expected_dim is None:
            gf = batch.global_features
            if gf.dim() == 2:
                expected_dim = int(gf.size(-1))
            else:
                num_graphs = getattr(batch, 'num_graphs', None)
                if num_graphs is None:
                    raise ValueError("Cannot infer expected_dim: model has no input_dim and num_graphs is unknown")
                total_elems = int(gf.numel())
                if total_elems % int(num_graphs) != 0:
                    raise ValueError(
                        f"global_features elements {total_elems} not divisible by num_graphs {int(num_graphs)}"
                    )
                expected_dim = total_elems // int(num_graphs)
        g = _reshape_global_features(batch, batch.global_features, int(expected_dim))
        with torch.amp.autocast('cuda', enabled=use_amp):
            pred = model(g)
            y = batch.y.view(-1)
            se = torch.sum((pred - y) ** 2).item()
        total_se += se
        total_n += y.numel()
    return total_se / max(1, total_n)


# grid_search_mlp and related search utilities have been removed to lock the
# MLP baseline to the same hyperparameters as the GNN's global MLP branch.


