"""
General-purpose feature engineering utilities.

The `FeatureGenerator` consolidates technical, alternative, and statistical
features into a single pandas DataFrame that downstream ML models can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from crypto_trader.features.factory import augment_with_features, DEFAULT_JOIN_CONFIG


@dataclass
class FeatureGenerator:
    symbol: str
    timeframe: str
    extra_columns: Optional[Iterable[str]] = None

    def build(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment OHLCV with alternative features and derived technical factors.
        """
        augmented = augment_with_features(
            market_df,
            symbol=self.symbol,
            timeframe=self.timeframe,
            config=DEFAULT_JOIN_CONFIG,
        ).copy()

        if "timestamp" in augmented.columns:
            augmented.index = pd.to_datetime(augmented["timestamp"], utc=True)
        augmented = augmented.sort_index()

        augmented["return_1"] = augmented["close"].pct_change()
        augmented["return_5"] = augmented["close"].pct_change(5)
        augmented["return_20"] = augmented["close"].pct_change(20)
        augmented["volatility_10"] = augmented["return_1"].rolling(10, min_periods=5).std()
        augmented["volatility_30"] = augmented["return_1"].rolling(30, min_periods=10).std()
        augmented["volume_z"] = (
            (augmented["volume"] - augmented["volume"].rolling(20).mean())
            / (augmented["volume"].rolling(20).std() + 1e-9)
        )

        augmented["trend_strength"] = (
            augmented["close"].rolling(20).mean()
            - augmented["close"].rolling(50).mean()
        )

        augmented.replace([np.inf, -np.inf], np.nan, inplace=True)
        augmented = augmented.dropna()

        if self.extra_columns:
            missing = [c for c in self.extra_columns if c not in augmented.columns]
            if missing:
                for col in missing:
                    augmented[col] = np.nan

        return augmented


def generate_feature_matrix(
    market_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    extra_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    generator = FeatureGenerator(
        symbol=symbol,
        timeframe=timeframe,
        extra_columns=extra_columns,
    )
    return generator.build(market_df)
