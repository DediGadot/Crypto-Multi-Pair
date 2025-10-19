"""
Order flow imbalance strategy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy


@register_strategy(
    name="OrderFlowImbalance",
    description="Detect aggressive buying/selling via order flow delta and imbalance",
    tags=["microstructure", "orderflow", "high_frequency", "sota_2025"],
)
class OrderFlowImbalanceStrategy(BaseStrategy):
    def __init__(self, name: str = "OrderFlowImbalance", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.delta_threshold: float = 100.0
        self.imbalance_threshold: float = 0.25
        self.vpin_threshold: float = 0.6

    def initialize(self, config: Dict[str, Any]) -> None:
        self.delta_threshold = float(config.get("delta_threshold", 100.0))
        self.imbalance_threshold = float(config.get("imbalance_threshold", 0.25))
        self.vpin_threshold = float(config.get("vpin_threshold", 0.6))
        self._initialized = True

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "delta_threshold": self.delta_threshold,
            "imbalance_threshold": self.imbalance_threshold,
            "vpin_threshold": self.vpin_threshold,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self._initialized:
            raise ValueError("OrderFlowImbalance not initialized")

        required_cols = {"micro.delta", "micro.cumulative_delta", "micro.book_imbalance"}
        if not required_cols.issubset(set(data.columns)):
            logger.debug("Order flow features missing - emitting HOLD")
            return self._hold_frame(data)

        df = data.copy()
        df["delta_smooth"] = df["micro.delta"].rolling(5, min_periods=3).mean()
        df["imbalance_smooth"] = df["micro.book_imbalance"].rolling(3, min_periods=1).mean()
        df["vpin"] = df.get("micro.vpin", np.nan).fillna(0.5)

        buy_mask = (
            (df["delta_smooth"] > self.delta_threshold)
            & (df["imbalance_smooth"] > self.imbalance_threshold)
            & (df["vpin"] < self.vpin_threshold)
        )

        sell_mask = (
            (df["delta_smooth"] < -self.delta_threshold)
            & (df["imbalance_smooth"] < -self.imbalance_threshold)
            & (df["vpin"] < self.vpin_threshold + 0.1)
        )

        result = pd.DataFrame(
            {
                "timestamp": data.get("timestamp"),
                "signal": SignalType.HOLD.value,
                "confidence": 0.0,
                "metadata": [{} for _ in range(len(data))],
            }
        )

        result.loc[buy_mask, "signal"] = SignalType.BUY.value
        result.loc[buy_mask, "confidence"] = 0.7
        result.loc[sell_mask, "signal"] = SignalType.SELL.value
        result.loc[sell_mask, "confidence"] = 0.7

        result.loc[buy_mask, "metadata"] = [
            {"reason": "orderflow_buy", "delta": float(d), "imbalance": float(i)}
            for d, i in zip(df.loc[buy_mask, "delta_smooth"], df.loc[buy_mask, "imbalance_smooth"])
        ]
        result.loc[sell_mask, "metadata"] = [
            {"reason": "orderflow_sell", "delta": float(d), "imbalance": float(i)}
            for d, i in zip(df.loc[sell_mask, "delta_smooth"], df.loc[sell_mask, "imbalance_smooth"])
        ]
        return result

    def _hold_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": data.get("timestamp"),
                "signal": SignalType.HOLD.value,
                "confidence": 0.0,
                "metadata": [{} for _ in range(len(data))],
            }
        )
