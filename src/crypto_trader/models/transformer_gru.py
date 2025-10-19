"""
Transformer + GRU Hybrid model utilities.

This module defines a lightweight PyTorch Lightning module used by the
TransformerGRUPredictor strategy. Training is optional; the strategy primarily
loads a pretrained checkpoint if available. When PyTorch Lightning is not
installed (e.g., minimal runtime environments), helper functions degrade
gracefully and surface actionable error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:  # Production dependency
    import torch
    from torch import nn
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - guard for environments without GPU stack
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    pl = None  # type: ignore[assignment]


class TransformerGRUModel(pl.LightningModule if pl else object):  # type: ignore[misc]
    """Hybrid encoder that mixes transformer attention with GRU memory."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        gru_hidden: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-4,
    ) -> None:
        if torch is None or nn is None or pl is None:
            raise RuntimeError(
                "PyTorch + PyTorch Lightning must be installed to initialize TransformerGRUModel"
            )

        super().__init__()
        self.save_hyperparameters()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Linear(gru_hidden // 2, 1),
        )
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        projected = self.input_projection(x)
        transformer_out = self.transformer(projected)
        gru_out, _ = self.gru(transformer_out)
        last_hidden = gru_out[:, -1, :]
        return self.regressor(last_hidden).squeeze(-1)

    def training_step(self, batch, batch_idx):  # pragma: no cover - executed during training
        inputs, targets = batch
        preds = self(inputs)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pragma: no cover
        inputs, targets = batch
        preds = self(inputs)
        loss = self.loss_fn(preds, targets)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):  # pragma: no cover
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=self.lr * 0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def load_transformer_gru(
    checkpoint_path: str | Path,
    feature_dim: int,
    map_location: str = "cpu",
) -> Optional[TransformerGRUModel]:
    """
    Load a saved TransformerGRUModel checkpoint if available.

    Args:
        checkpoint_path: Path to `.ckpt` or `.pt` file.
        feature_dim: Number of features expected by the model (used when
            instantiating new models for state dict loading).
        map_location: Torch map location (defaults to CPU).
    """
    if torch is None or pl is None:
        return None

    path = Path(checkpoint_path)
    if not path.exists():
        return None

    state = torch.load(path, map_location=map_location)  # type: ignore[arg-type]
    # PyTorch Lightning checkpoints store metadata under "hyper_parameters"
    hparams = state.get("hyper_parameters", {})
    model = TransformerGRUModel(
        feature_dim=feature_dim,
        d_model=hparams.get("d_model", 256),
        nhead=hparams.get("nhead", 8),
        num_layers=hparams.get("num_layers", 4),
        gru_hidden=hparams.get("gru_hidden", 128),
        dropout=hparams.get("dropout", 0.1),
        lr=hparams.get("lr", 1e-4),
    )
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    if missing or unexpected:
        # Log via print to avoid hard dependency on loguru inside model utils.
        print(
            f"[TransformerGRU] Loaded with missing={missing} unexpected={unexpected}",  # noqa: T201
        )
    model.eval()
    return model


def predict_next_return(
    model: TransformerGRUModel,
    window: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    Convenience helper for single-step inference.

    Args:
        model: Loaded model instance.
        window: np.ndarray of shape (seq_len, feature_dim).
        device: Torch device string.
    """
    if torch is None:
        raise RuntimeError("PyTorch not available for prediction")

    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(window, dtype=torch.float32, device=device)  # type: ignore[arg-type]
        preds = model(tensor.unsqueeze(0)).cpu().numpy()
    return float(preds.squeeze())
