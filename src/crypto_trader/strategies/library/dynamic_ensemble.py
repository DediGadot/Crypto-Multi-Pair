"""
Dynamic strategy ensemble using recent performance metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from crypto_trader.analysis.performance_store import PerformanceStore
from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import get_strategy, register_strategy


@register_strategy(
    name="DynamicEnsemble",
    description="Meta-strategy that weights underlying strategies based on recent Sharpe",
    tags=["ensemble", "meta", "portfolio", "sota_2025"],
)
class DynamicEnsembleStrategy(BaseStrategy):
    def __init__(self, name: str = "DynamicEnsemble", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.store = PerformanceStore()
        self.child_names: List[str] = []
        self.child_configs: Dict[str, Dict[str, Any]] = {}
        self.child_instances: Dict[str, BaseStrategy] = {}
        self.min_weight: float = 0.0
        self.max_weight: float = 0.4
        self.lookback_days: int = 90
        self.conf_threshold: float = 0.15

    def initialize(self, config: Dict[str, Any]) -> None:
        self.child_names = config.get(
            "strategies",
            [
                "MultiTimeframeConfluence",
                "OnChainAnalytics",
                "Supertrend_ATR",
                "RSIMeanReversion",
            ],
        )
        self.child_configs = config.get("child_configs", {})
        self.min_weight = config.get("min_weight", 0.0)
        self.max_weight = config.get("max_weight", 0.4)
        self.lookback_days = config.get("lookback_days", 90)
        self.conf_threshold = config.get("confidence_threshold", 0.15)

        for name in self.child_names:
            try:
                StrategyClass = get_strategy(name)
                instance = StrategyClass(name=f"{self.name}_{name}")
                instance.initialize(self.child_configs.get(name, {}))
                self.child_instances[name] = instance
            except Exception as exc:
                logger.error(f"Failed to initialize ensemble child {name}: {exc}")

        self._initialized = True

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "strategies": self.child_names,
            "child_configs": self.child_configs,
            "lookback_days": self.lookback_days,
        }

    def _load_weights(self) -> pd.Series:
        if not self.child_names:
            return pd.Series(dtype=float)

        metrics = self.store.recent(self.child_names, days=self.lookback_days)
        if metrics.empty:
            weights = pd.Series(
                np.full(len(self.child_names), 1.0 / len(self.child_names)),
                index=self.child_names,
            )
            return weights

        summary = metrics.groupby("strategy")["sharpe"].mean()
        summary = summary.clip(lower=0)  # pandas uses clip(), not clamp()
        if summary.sum() == 0:
            summary = pd.Series(
                np.full(len(self.child_names), 1.0), index=self.child_names
            )
        weights = summary / summary.sum()
        weights = weights.clip(lower=self.min_weight, upper=self.max_weight)
        weights = weights / weights.sum()
        return weights

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self._initialized:
            raise ValueError("DynamicEnsemble not initialized")

        ensemble_cols = {
            "timestamp": data.get("timestamp"),
            "signal": [],
            "confidence": [],
            "metadata": [],
        }

        weights = self._load_weights()
        if weights.empty:
            return pd.DataFrame(
                {
                    "timestamp": data.get("timestamp"),
                    "signal": SignalType.HOLD.value,
                    "confidence": 0.0,
                    "metadata": [{} for _ in range(len(data))],
                }
            )

        aggregated_conf = np.zeros(len(data))
        aggregated_direction = np.zeros(len(data))
        metadata = [{} for _ in range(len(data))]

        for name, weight in weights.items():
            child = self.child_instances.get(name)
            if child is None:
                logger.warning(f"Child strategy {name} missing - skipping")
                continue
            child_signals = child.generate_signals(data)
            if child_signals.empty:
                continue

            numeric_signal = child_signals["signal"].map(
                {SignalType.BUY.value: 1, SignalType.SELL.value: -1, SignalType.HOLD.value: 0}
            ).astype(float)
            aggregated_direction += numeric_signal.values * weight
            aggregated_conf += child_signals["confidence"].values * weight

            for idx, meta in enumerate(child_signals["metadata"]):
                combined = dict(metadata[idx])
                combined.setdefault("contributors", [])
                combined["contributors"].append(
                    {
                        "strategy": name,
                        "weight": weight,
                        "signal": child_signals.iloc[idx]["signal"],
                        "confidence": child_signals.iloc[idx]["confidence"],
                    }
                )
                metadata[idx] = combined

        final_signal = np.where(
            aggregated_direction > self.conf_threshold,
            SignalType.BUY.value,
            np.where(
                aggregated_direction < -self.conf_threshold,
                SignalType.SELL.value,
                SignalType.HOLD.value,
            ),
        )

        ensemble_df = pd.DataFrame(
            {
                "timestamp": data.get("timestamp"),
                "signal": final_signal,
                "confidence": np.clip(aggregated_conf, 0, 1),
                "metadata": metadata,
            }
        )
        return ensemble_df
