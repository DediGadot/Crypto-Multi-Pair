"""
Gymnasium environment for discrete-action crypto trading.

The DDQN strategy uses this environment during offline training. The runtime
strategy itself does not step through the env; instead it relies on a trained
policy saved via Stable-Baselines3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class TradingEnvConfig:
    feature_columns: list[str]
    initial_capital: float = 10000.0
    trading_fee: float = 0.001
    max_position: float = 1.0  # relative to capital
    reward_metric: str = "sharpe"  # or "return"


class TradingEnv(gym.Env):
    """Minimal discrete trading environment (BUY / HOLD / SELL)."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, config: TradingEnvConfig):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.config = config

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(config.feature_columns),),
            dtype=np.float32,
        )

        self._pos = 0.0
        self._cash = config.initial_capital
        self._equity = config.initial_capital
        self._step = 0
        self._prev_price: Optional[float] = None
        self._returns: list[float] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._pos = 0.0
        self._cash = self.config.initial_capital
        self._equity = self.config.initial_capital
        self._step = 0
        self._prev_price = None
        self._returns = []
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        row = self.data.iloc[self._step]
        obs = row[self.config.feature_columns].astype(float).to_numpy()
        return obs

    def _get_price(self) -> float:
        price = float(self.data.iloc[self._step]["close"])
        return max(price, 1e-8)

    def step(self, action: int):
        price = self._get_price()

        reward = 0.0
        terminated = False
        truncated = False

        if action == 1:  # BUY
            target_pos = self.config.max_position
        elif action == 2:  # SELL / flat
            target_pos = 0.0
        else:  # HOLD
            target_pos = self._pos

        delta = target_pos - self._pos
        if abs(delta) > 1e-6:
            trade_value = delta * self._cash
            fee = abs(trade_value) * self.config.trading_fee
            self._cash -= trade_value + fee
            self._pos = target_pos
            reward -= fee / self.config.initial_capital

        if self._prev_price is not None:
            position_value = self._pos * self.config.initial_capital * (
                price / self._prev_price
            )
            daily_return = (position_value - self._pos * self.config.initial_capital) / (
                self._pos * self.config.initial_capital + 1e-9
            )
            self._returns.append(daily_return)
            reward += daily_return
        self._prev_price = price

        self._step += 1
        if self._step >= len(self.data) - 1:
            terminated = True

        obs = self._get_obs()
        info = {"equity": self._equity}
        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - used manually
        print(f"Step={self._step} Pos={self._pos:.2f} Cash={self._cash:.2f}")  # noqa: T201
