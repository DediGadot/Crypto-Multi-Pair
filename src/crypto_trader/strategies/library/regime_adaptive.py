"""
Volatility Regime Adaptive strategy.

Wraps existing strategies and activates whichever performs best for the
detected regime. When no regime can be determined the strategy remains flat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import get_strategy, register_strategy
from crypto_trader.strategies.library.statistical_arbitrage.regime_detection import (
    RegimeDetector,
)


REGIME_DEFAULTS = {
    "mean_reverting": "RSIMeanReversion",
    "trending": "Supertrend_ATR",
    "volatile": None,
}


@dataclass
class RegimeConfig:
    strategy_name: Optional[str]
    position_size: float


@register_strategy(
    name="VolatilityRegimeAdaptive",
    description="Detect regimes via HMM and route signals to the best strategy",
    tags=["meta", "regime", "adaptive", "sota_2025"],
)
class VolatilityRegimeAdaptiveStrategy(BaseStrategy):
    def __init__(self, name: str = "VolatilityRegimeAdaptive", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.detector = RegimeDetector()
        self.regime_map: Dict[str, RegimeConfig] = {}
        self.child_configs: Dict[str, Dict[str, Any]] = {}
        self.child_instances: Dict[str, BaseStrategy] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        mapping = config.get("regime_map", {})
        position_sizes = config.get(
            "position_size_map",
            {
                "mean_reverting": 0.6,
                "trending": 1.0,
                "volatile": 0.2,
            },
        )
        child_configs = config.get("child_configs", {})

        if not mapping:
            mapping = REGIME_DEFAULTS

        for regime, strategy_name in mapping.items():
            self.regime_map[regime] = RegimeConfig(
                strategy_name=strategy_name,
                position_size=float(position_sizes.get(regime, 0.5)),
            )
            if strategy_name and strategy_name not in child_configs:
                child_configs[strategy_name] = {}

        self.child_configs = child_configs

        for strategy_name, child_config in child_configs.items():
            try:
                StrategyClass = get_strategy(strategy_name)
                instance = StrategyClass(name=f"{self.name}_{strategy_name}")
                instance.initialize(child_config)
                self.child_instances[strategy_name] = instance
            except Exception as exc:
                logger.error(f"Failed to initialize child strategy {strategy_name}: {exc}")

        self._initialized = True
        logger.info(f"{self.name} initialized with regimes: {list(self.regime_map.keys())}")

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "regime_map": {regime: cfg.strategy_name for regime, cfg in self.regime_map.items()},
            "child_configs": self.child_configs,
        }

    def _build_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values("timestamp")
        returns = df["close"].pct_change()
        volatility = returns.rolling(30, min_periods=10).std()
        volume_ratio = (df["volume"] / df["volume"].rolling(20).mean()) - 1
        momentum = df["close"].pct_change(30)

        features = pd.DataFrame(
            {
                "volatility": volatility,
                "correlation": returns.rolling(30, min_periods=10).corr(volume_ratio).fillna(0),
                "spread_vol": momentum.rolling(10, min_periods=5).std(),
            }
        ).dropna()
        return features

    def _get_child(self, strategy_name: str) -> Optional[BaseStrategy]:
        if strategy_name not in self.child_instances:
            try:
                StrategyClass = get_strategy(strategy_name)
                instance = StrategyClass(name=f"{self.name}_{strategy_name}")
                instance.initialize(self.child_configs.get(strategy_name, {}))
                self.child_instances[strategy_name] = instance
            except Exception as exc:
                logger.error(f"Unable to instantiate child strategy {strategy_name}: {exc}")
                return None
        return self.child_instances[strategy_name]

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self._initialized:
            raise ValueError("Strategy not initialized")

        valid = self.validate_data(data)
        if not valid:
            return pd.DataFrame(
                {
                    "timestamp": data.get("timestamp"),
                    "signal": SignalType.HOLD.value,
                    "confidence": 0.0,
                    "metadata": [{} for _ in range(len(data))],
                }
            )

        features = self._build_regime_features(data)
        if len(features) < 30:
            logger.warning("Not enough data for regime detection - emitting HOLD")
            return self._hold_frame(data)

        if not self.detector.is_fitted:
            try:
                self.detector.fit(features)
            except Exception as exc:
                logger.error(f"Regime detector failed to fit: {exc}")
                return self._hold_frame(data)

        try:
            prediction = self.detector.predict(features.iloc[-1:].copy())
        except Exception as exc:
            logger.error(f"Regime prediction failed: {exc}")
            return self._hold_frame(data)

        regime_idx = int(prediction["regime"][-1])
        regime_name = self.detector.regime_params[regime_idx]["name"]
        cfg = self.regime_map.get(regime_name)
        if cfg is None or cfg.strategy_name is None:
            logger.debug(f"No strategy mapped for regime {regime_name}; staying flat")
            return self._hold_frame(data, regime_name)

        child = self._get_child(cfg.strategy_name)
        if child is None:
            return self._hold_frame(data, regime_name)

        signals = child.generate_signals(data)
        signals = signals.copy()
        signals["confidence"] = signals["confidence"] * cfg.position_size
        if "metadata" in signals.columns:
            metadata = []
            for meta in signals["metadata"]:
                meta = dict(meta)
                meta["regime"] = regime_name
                metadata.append(meta)
            signals["metadata"] = metadata
        return signals

    def _hold_frame(self, data: pd.DataFrame, regime: Optional[str] = None) -> pd.DataFrame:
        metadata = [{} for _ in range(len(data))]
        if regime:
            metadata = [{"regime": regime}] * len(data)
        return pd.DataFrame(
            {
                "timestamp": data.get("timestamp"),
                "signal": SignalType.HOLD.value,
                "confidence": 0.0,
                "metadata": metadata,
            }
        )
