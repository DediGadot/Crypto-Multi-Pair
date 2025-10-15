"""
On-Chain Ingestor (CSV-based placeholder)

Purpose:
- Provide a pragmatic, offline-friendly path to materialize on-chain features
  for use in backtests. This ingestor expects a local CSV file per symbol or
  can derive a few proxy signals from OHLCV as a fallback so that the feature
  plumbing can be validated without external APIs.

Input options (preferred):
- data/onchain/{symbol_safe}.csv with columns:
    event_time, mvrv_z, sopr, exchange_netflow, whale_ratio, puell_multiple

Fallback (if input is missing):
- Use OHLCV close/volume to synthesize placeholder features with deterministic
  transforms. These are NOT real on-chain metrics; they are for pipeline testing
  only and are clearly named 'proxy_*'.

Output:
- data/features/onchain/{symbol_safe}.csv with columns:
    event_time, <features>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from crypto_trader.features.store import FeatureStore
from crypto_trader.data.storage import OHLCVStorage


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def load_local_onchain_csv(symbol: str, base_dir: str | Path = "data/onchain") -> Optional[pd.DataFrame]:
    path = Path(base_dir) / f"{_safe_symbol(symbol)}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=['event_time'])
        df['event_time'] = pd.to_datetime(df['event_time'], utc=True, errors='coerce')
        df = df.dropna(subset=['event_time']).sort_values('event_time')
        return df
    except Exception as e:
        logger.error(f"Failed to read local on-chain CSV {path}: {e}")
        return None


def _proxy_from_ohlcv(symbol: str, timeframe: str, storage: Optional[OHLCVStorage] = None) -> Optional[pd.DataFrame]:
    """Create deterministic placeholder features from OHLCV for testing.

    This is not intended for production signals; it allows validating the
    end-to-end feature pipeline when external data is unavailable.
    """
    storage = storage or OHLCVStorage()
    df = storage.load_ohlcv(symbol, timeframe)
    if df is None or df.empty:
        return None

    # Use the OHLCV index as event_time
    out = pd.DataFrame({'event_time': pd.to_datetime(df.index, utc=True)})

    # Construct a few proxy series with smoothing
    close = df['close'].astype(float)
    vol = df['volume'].astype(float)

    out['proxy_mvrv_z'] = ((close - close.rolling(200, min_periods=10).mean()) / (close.rolling(200, min_periods=10).std() + 1e-9)).fillna(0.0)
    out['proxy_sopr'] = (close / (close.rolling(30, min_periods=5).mean() + 1e-9)).fillna(1.0)
    out['proxy_exchange_netflow'] = (vol - vol.rolling(20, min_periods=5).mean()).fillna(0.0)
    out['proxy_whale_ratio'] = (vol.rolling(10, min_periods=3).max() / (vol.rolling(10, min_periods=3).sum() + 1e-9)).fillna(0.0)
    out['proxy_puell_multiple'] = (close * vol / ((close * vol).rolling(365, min_periods=30).mean() + 1e-9)).fillna(1.0)

    return out


def ingest_onchain(
    symbol: str,
    timeframe: str = "1h",
    prefer_local_csv: bool = True,
    output_store: Optional[FeatureStore] = None,
) -> bool:
    """Materialize on-chain features for a symbol into the FeatureStore.

    The function tries a local CSV with real on-chain fields first; if missing,
    it writes proxy_* features derived from OHLCV for pipeline validation.
    """
    store = output_store or FeatureStore()

    df = None
    if prefer_local_csv:
        df = load_local_onchain_csv(symbol)

    if df is None:
        logger.warning("On-chain CSV not found; generating proxy features from OHLCV for testing.")
        df = _proxy_from_ohlcv(symbol, timeframe)

    if df is None or df.empty:
        logger.warning(f"No on-chain features available for {symbol}")
        return False

    ok = store.write(df, symbol=symbol, pillar="onchain")
    return ok

