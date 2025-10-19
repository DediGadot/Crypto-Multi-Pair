"""
Dataset helpers for deep learning strategies.

The goal is to provide a thin wrapper that converts market data (already
augmented with alternative features) into torch-friendly tensors without
forcing callers to depend on GPU hardware during inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Torch is an optional dependency at runtime
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch is part of production deps
    torch = None
    Dataset = object  # type: ignore[assignment]


DEFAULT_FEATURE_COLUMNS: Sequence[str] = (
    "close",
    "volume",
    "return_1",
    "return_5",
    "return_10",
    "volatility_10",
    "volatility_30",
    "rsi_14",
    "atr_14",
    "macd",
    "macd_signal",
)


def _ta_safe(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return series.astype(float)


def build_feature_frame(
    market_df: pd.DataFrame,
    *,
    include_indicators: bool = True,
    extra_feature_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Build a tabular feature frame for model consumption.

    Args:
        market_df: DataFrame with at least ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        include_indicators: Whether to append classical TA indicators.
        extra_feature_cols: Optional iterable of already-computed feature column names
            that should be preserved even if they contain NaNs.

    Returns:
        Clean DataFrame indexed by timestamp with engineered features.
    """
    df = market_df.copy()

    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_index(inplace=True)

    df["return_1"] = df["close"].pct_change()
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)

    df["volatility_10"] = df["return_1"].rolling(window=10, min_periods=5).std()
    df["volatility_30"] = df["return_1"].rolling(window=30, min_periods=10).std()

    if include_indicators:
        # Avoid bringing pandas_ta into training loops when not installed on CI.
        try:
            import pandas_ta as ta  # pylint: disable=import-error

            df["rsi_14"] = _ta_safe(ta.rsi(df["close"], length=14))
            df["atr_14"] = _ta_safe(ta.atr(df["high"], df["low"], df["close"], length=14))
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd is not None:
                df["macd"] = _ta_safe(macd["MACD_12_26_9"])
                df["macd_signal"] = _ta_safe(macd["MACDs_12_26_9"])
        except Exception:
            df["rsi_14"] = df["return_1"] * 0.0
            df["atr_14"] = (df["high"] - df["low"]).rolling(14, min_periods=5).mean()
            df["macd"] = df["return_1"].rolling(12, min_periods=5).mean()
            df["macd_signal"] = df["macd"].rolling(9, min_periods=5).mean()

    cols_to_keep: List[str] = list(DEFAULT_FEATURE_COLUMNS)
    if extra_feature_cols:
        cols_to_keep.extend(extra_feature_cols)

    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


@dataclass
class SequenceSample:
    features: np.ndarray
    target: float


class SequenceDataset(Dataset):  # type: ignore[misc]
    """
    Basic sliding-window dataset for sequence-to-one forecasting.

    When torch is unavailable, this dataset raises at initialization time.
    """

    def __init__(
        self,
        feature_frame: pd.DataFrame,
        target_col: str = "return_1",
        sequence_length: int = 60,
        feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch not available - install the production dependencies to use SequenceDataset"
            )

        self.sequence_length = int(sequence_length)
        self.target_col = target_col

        cols = [c for c in feature_columns if c in feature_frame.columns]
        if target_col not in feature_frame.columns:
            raise ValueError(f"Target column '{target_col}' not in feature frame")

        matrix = feature_frame[cols].astype(float).values
        target = feature_frame[target_col].astype(float).values

        sequences: List[SequenceSample] = []
        for idx in range(len(feature_frame) - self.sequence_length):
            window = matrix[idx : idx + self.sequence_length]
            label = target[idx + self.sequence_length]
            sequences.append(
                SequenceSample(
                    features=window,
                    target=label,
                )
            )

        if not sequences:
            raise ValueError("Not enough data to build SequenceDataset")

        self._features = torch.tensor(  # type: ignore[assignment]
            np.stack([sample.features for sample in sequences]), dtype=torch.float32
        )
        self._targets = torch.tensor(  # type: ignore[assignment]
            [sample.target for sample in sequences], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[name-defined]
        return self._features[idx], self._targets[idx]
