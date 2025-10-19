"""
Transformer-GRU hybrid price prediction strategy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from crypto_trader.models import build_feature_frame, load_transformer_gru
from crypto_trader.models.transformer_gru import predict_next_return
from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy


@register_strategy(
    name="TransformerGRUPredictor",
    description="Hybrid transformer/GRU model forecasting next-period returns",
    tags=["ml", "transformer", "gru", "sota_2025"],
)
class TransformerGRUPredictorStrategy(BaseStrategy):
    def __init__(self, name: str = "TransformerGRUPredictor", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.model_path: Optional[Path] = None
        self.buy_threshold: float = 0.02
        self.sell_threshold: float = 0.02
        self.sequence_length: int = 60
        self.feature_columns: Sequence[str] = ()
        self.model = None

    def initialize(self, config: Dict[str, Any]) -> None:
        self.model_path = Path(config.get("model_path", "models/transformer_gru.ckpt"))
        self.buy_threshold = float(config.get("buy_threshold", 0.02))
        self.sell_threshold = float(config.get("sell_threshold", 0.02))
        self.sequence_length = int(config.get("sequence_length", 60))
        self.feature_columns = tuple(config.get("feature_columns", []))

        feature_dim = len(self.feature_columns) if self.feature_columns else 11
        self.model = load_transformer_gru(self.model_path, feature_dim=feature_dim)
        if self.model is None:
            logger.warning(
                f"{self.name}: No pretrained model found at {self.model_path}. "
                "Falling back to heuristic predictions."
            )
        self._initialized = True

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path) if self.model_path else None,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "sequence_length": self.sequence_length,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self._initialized:
            raise ValueError("TransformerGRUPredictor not initialized")

        frame = build_feature_frame(
            data,
            include_indicators=True,
            extra_feature_cols=self.feature_columns,
        )
        if len(frame) < self.sequence_length + 5:
            logger.debug("Insufficient data for TransformerGRU - emitting HOLD")
            return self._hold_frame(data)

        window = frame.iloc[-self.sequence_length :].values
        predicted_return = self._predict(window)

        signals = pd.DataFrame(
            {
                "timestamp": data.get("timestamp"),
                "signal": SignalType.HOLD.value,
                "confidence": 0.0,
                "metadata": [{} for _ in range(len(data))],
            }
        )

        metadata = {
            "predicted_return": float(predicted_return),
            "model_path": str(self.model_path) if self.model_path else None,
        }

        # Use .at for safer scalar assignment (avoids indexer incompatibility)
        last_idx = signals.index[-1]
        if predicted_return > self.buy_threshold:
            signals.at[last_idx, "signal"] = SignalType.BUY.value
            signals.at[last_idx, "confidence"] = min(0.5 + predicted_return, 1.0)
            signals.at[last_idx, "metadata"] = metadata
        elif predicted_return < -self.sell_threshold:
            signals.at[last_idx, "signal"] = SignalType.SELL.value
            signals.at[last_idx, "confidence"] = min(0.5 + abs(predicted_return), 1.0)
            signals.at[last_idx, "metadata"] = metadata
        else:
            signals.at[last_idx, "metadata"] = metadata
        return signals

    def _predict(self, window: np.ndarray) -> float:
        if self.model is not None:
            try:
                return predict_next_return(self.model, window)
            except Exception as exc:
                logger.error(f"{self.name}: model prediction failed: {exc}")

        # Heuristic fallback: mean of next-period returns from last window
        # Assume last column approximates recent returns; otherwise default to zeros.
        last_column = window[:, -1] if window.shape[1] > 0 else np.zeros(len(window))
        return float(np.tanh(last_column.mean()))

    def _hold_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": data.get("timestamp"),
                "signal": SignalType.HOLD.value,
                "confidence": 0.0,
                "metadata": [{} for _ in range(len(data))],
            }
        )
