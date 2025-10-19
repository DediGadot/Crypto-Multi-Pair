"""
DDQN strategy with XGBoost-based feature selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:  # Optional, heavy dependency
    from stable_baselines3 import DQN
except Exception:  # pragma: no cover
    DQN = None

from crypto_trader.features.engineering import generate_feature_matrix
from crypto_trader.ml import FeatureSelector
from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy


@register_strategy(
    name="DDQNFeatureSelected",
    description="Double DQN policy with dynamically selected features",
    tags=["rl", "ddqn", "ensemble", "sota_2025"],
)
class DDQNFeatureSelectedStrategy(BaseStrategy):
    def __init__(self, name: str = "DDQNFeatureSelected", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.model_path: Optional[Path] = None
        self.feature_path: Optional[Path] = None
        self.selected_features: List[str] = []
        self.policy = None
        self.lookback: int = 120
        self.reward_preference: str = "sharpe"

    def initialize(self, config: Dict[str, Any]) -> None:
        self.model_path = Path(config.get("model_path", "models/ddqn_policy.zip"))
        self.feature_path = Path(config.get("feature_file", "models/ddqn_features.json"))
        self.lookback = int(config.get("lookback", 120))
        self.reward_preference = config.get("reward_metric", "sharpe")

        self.selected_features = self._load_feature_list(self.feature_path)
        if not self.selected_features:
            logger.warning(
                f"{self.name}: no feature selection file found; using default technical set"
            )
            self.selected_features = [
                "return_1",
                "return_5",
                "return_20",
                "volatility_10",
                "volatility_30",
                "volume_z",
            ]

        if DQN is not None and self.model_path.exists():
            try:
                self.policy = DQN.load(self.model_path, print_system_info=False)
                logger.info(f"{self.name}: loaded DDQN policy from {self.model_path}")
            except Exception as exc:
                logger.error(f"Failed to load DDQN policy: {exc}")
                self.policy = None
        else:
            if DQN is None:
                logger.warning("stable-baselines3 not installed; using heuristic DDQN fallback")

        self._initialized = True

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path) if self.model_path else None,
            "feature_count": len(self.selected_features),
            "lookback": self.lookback,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self._initialized:
            raise ValueError("DDQNFeatureSelected not initialized")

        features = generate_feature_matrix(
            data,
            symbol=self.config.get("symbol", "BTC/USDT"),
            timeframe=self.config.get("timeframe", "1h"),
            extra_columns=self.selected_features,
        )
        if len(features) < self.lookback:
            logger.debug("Insufficient data for DDQN policy - emitting HOLD")
            return self._hold_frame(data)

        state = features[self.selected_features].iloc[-1].astype(float).to_numpy()
        action, confidence = self._decide_action(state)

        metadata = {
            "selected_features": self.selected_features,
            "policy_loaded": self.policy is not None,
            "reward_metric": self.reward_preference,
        }

        signal_map = {0: SignalType.HOLD.value, 1: SignalType.BUY.value, 2: SignalType.SELL.value}
        signal_value = signal_map.get(action, SignalType.HOLD.value)

        result = pd.DataFrame(
            {
                "timestamp": data.get("timestamp"),
                "signal": SignalType.HOLD.value,
                "confidence": 0.0,
                "metadata": [{} for _ in range(len(data))],
            }
        )
        result.at[result.index[-1], "signal"] = signal_value
        result.at[result.index[-1], "confidence"] = confidence
        result.at[result.index[-1], "metadata"] = metadata
        return result

    def _decide_action(self, state: np.ndarray) -> tuple[int, float]:
        if self.policy is not None:
            try:
                action, _ = self.policy.predict(state, deterministic=True)
                return int(action), 0.6
            except Exception as exc:
                logger.error(f"DDQN policy prediction failed: {exc}")

        # Fallback heuristic: threshold on weighted feature sum
        score = float(np.tanh(np.dot(state, np.linspace(1.0, 0.5, len(state)))))
        if score > 0.05:
            return 1, min(abs(score), 1.0)
        if score < -0.05:
            return 2, min(abs(score), 1.0)
        return 0, 0.0

    def _load_feature_list(self, path: Path) -> List[str]:
        if not path.exists():
            return []
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text())
                return data.get("selected_features", [])
            df = pd.read_csv(path)
            if "feature" in df.columns:
                return df["feature"].head(50).tolist()
        except Exception as exc:
            logger.error(f"Failed to read feature list {path}: {exc}")
        return []

    def _hold_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": data.get("timestamp"),
                "signal": SignalType.HOLD.value,
                "confidence": 0.0,
                "metadata": [{} for _ in range(len(data))],
            }
        )
