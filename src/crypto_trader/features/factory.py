"""
Feature Factory: synchronize and join alternative data features onto OHLCV.

This module centralizes logic to:
- Load per-pillar feature frames from FeatureStore
- Align them to the OHLCV index (minute/hour/day)
- Enforce simple freshness/staleness rules
- Join features onto the provided market data DataFrame

Design goals:
- Backward-compatible: if no features exist, return input unchanged
- No look-ahead: resample/forward-fill only up to the current bar close
- Minimal dependencies; callers manage which pillars to request
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .store import FeatureStore, FeatureReadRequest


@dataclass
class FeatureJoinConfig:
    pillars: List[str]  # e.g., ["onchain", "sent", "opt", "micro"]
    max_staleness: Dict[str, pd.Timedelta]  # pillar -> staleness window


DEFAULT_JOIN_CONFIG = FeatureJoinConfig(
    pillars=["onchain", "sent", "opt", "micro"],
    max_staleness={
        "onchain": pd.Timedelta(days=7),
        "sent": pd.Timedelta(hours=6),
        "opt": pd.Timedelta(days=2),
        "micro": pd.Timedelta(minutes=5),
    },
)


def _ffill_to_index(features: pd.DataFrame, market_index: pd.DatetimeIndex) -> pd.DataFrame:
    if features is None or features.empty:
        return pd.DataFrame(index=market_index)

    f = features.copy()
    if 'event_time' not in f.columns:
        raise ValueError("Features must include 'event_time' column")
    idx = pd.to_datetime(f['event_time'], utc=True)
    f = f.drop(columns=['event_time'])
    # Ensure monotonic increasing and unique index
    f.index = idx
    f = f[~f.index.duplicated(keep='last')].sort_index()

    # Reindex to market index (assumed UTC), forward-fill past values
    f = f.reindex(market_index, method=None)
    f = f.ffill()
    return f


def _apply_staleness_mask(
    ffilled: pd.DataFrame,
    raw: pd.DataFrame,
    market_index: pd.DatetimeIndex,
    max_age: pd.Timedelta,
) -> pd.DataFrame:
    """Mask feature values that exceed the staleness threshold based on last update time.

    Adds a column '<pillar>_is_stale' when applicable and nulls stale entries.
    """
    if ffilled.empty:
        return ffilled

    # Build last update time series aligned to market index
    updates = raw[['event_time']].copy()
    updates['event_time'] = pd.to_datetime(updates['event_time'], utc=True)
    updates = updates.dropna(subset=['event_time']).sort_values('event_time')
    updates = updates.set_index('event_time')
    updates['last_update'] = updates.index
    last_update = updates['last_update'].reindex(market_index).ffill()

    # Build series representing each bar's timestamp in UTC
    market_series = pd.Series(market_index, index=market_index)
    if market_series.dt.tz is None:
        market_series = market_series.dt.tz_localize('UTC')
    else:
        market_series = market_series.dt.tz_convert('UTC')

    age = market_series - last_update
    is_stale = age > max_age

    # Mask all feature columns when stale
    masked = ffilled.copy()
    stale_series = pd.Series(is_stale).reindex(market_index).fillna(False)
    for col in masked.columns:
        masked.loc[stale_series, col] = pd.NA
    # Add a generic staleness flag for diagnostics
    masked['is_stale'] = stale_series.astype(bool)
    return masked


def augment_with_features(
    market_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    store: Optional[FeatureStore] = None,
    config: FeatureJoinConfig = DEFAULT_JOIN_CONFIG,
) -> pd.DataFrame:
    """Join available pillars onto OHLCV DataFrame. Safe no-op if none available.

    Args:
        market_df: OHLCV frame with DatetimeIndex or 'timestamp' column
        symbol: trading pair (e.g., 'BTC/USDT')
        timeframe: timeframe string (informational; not used yet)
        store: FeatureStore instance (defaults to CSV store)
        config: pillars to load and staleness thresholds
    """
    if market_df is None or market_df.empty:
        return market_df

    # Resolve index and time bounds
    if 'timestamp' in market_df.columns:
        idx = pd.to_datetime(market_df['timestamp'], utc=True)
    else:
        idx = pd.to_datetime(market_df.index, utc=True)
    start, end = idx.min(), idx.max()

    fs = store or FeatureStore()
    req = FeatureReadRequest(symbol=symbol, pillars=config.pillars, start=start, end=end)
    frames = fs.read(req)

    joined = market_df.copy()
    # Ensure timestamp column exists and preserve datetime index
    if 'timestamp' not in joined.columns:
        joined['timestamp'] = idx
    joined.index = idx

    # For each pillar, forward-fill and apply staleness, then join columns
    for pillar, raw in frames.items():
        if raw is None or raw.empty:
            logger.debug(f"No features found for pillar='{pillar}' symbol='{symbol}'")
            continue

        ffilled = _ffill_to_index(raw, idx)
        max_age = config.max_staleness.get(pillar, pd.Timedelta.max)
        masked = _apply_staleness_mask(ffilled, raw, idx, max_age)

        # Prefix columns with pillar for clarity if not already prefixed
        renamed = {}
        for col in masked.columns:
            if col == 'is_stale':
                renamed[col] = f"{pillar}_is_stale"
            elif not col.startswith(f"{pillar}.") and not col.endswith('_is_stale'):
                renamed[col] = f"{pillar}.{col}"
        masked = masked.rename(columns=renamed)

        # Align by index; ensure we keep the same row count
        masked = masked.reindex(joined.index)
        for col in masked.columns:
            joined[col] = masked[col].values

    return joined
