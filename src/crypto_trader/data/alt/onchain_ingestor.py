"""
On-Chain Ingestor with Glassnode API Integration

Purpose:
- Fetch real on-chain metrics from Glassnode API (free tier compatible)
- Materialize on-chain features for use in backtests
- Falls back to proxy signals if API unavailable

Input options (priority order):
1. Glassnode API (requires GLASSNODE_API_KEY env var)
2. Local CSV: data/onchain/{symbol_safe}.csv with columns:
    event_time, mvrv_z, sopr, exchange_netflow, whale_ratio, puell_multiple
3. Fallback: Proxy features from OHLCV

**Third-party packages**:
- requests: https://requests.readthedocs.io/en/latest/
- pandas: https://pandas.pydata.org/docs/
- loguru: https://loguru.readthedocs.io/en/stable/

**Glassnode API**:
- Free tier: https://studio.glassnode.com (limited metrics, 1 hour resolution)
- Docs: https://docs.glassnode.com/api/
- Rate limit: 20 requests/10 minutes (free tier)

Output:
- data/features/onchain/{symbol_safe}.csv with columns:
    event_time, <features>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger
import requests

from crypto_trader.features.store import FeatureStore
from crypto_trader.data.storage import OHLCVStorage


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def _glassnode_symbol(symbol: str) -> str:
    """Convert trading symbol to Glassnode format (e.g., BTC/USDT -> BTC)."""
    return symbol.split('/')[0].upper()


def fetch_glassnode_metrics(
    symbol: str,
    since_timestamp: Optional[int] = None,
    until_timestamp: Optional[int] = None,
    resolution: str = "24h"
) -> Optional[pd.DataFrame]:
    """
    Fetch on-chain metrics from Glassnode API (free tier).

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        since_timestamp: Start timestamp (unix seconds), defaults to 365 days ago
        until_timestamp: End timestamp (unix seconds), defaults to now
        resolution: Data resolution - '24h' (free), '1h' (paid only)

    Returns:
        DataFrame with columns: event_time, mvrv_z, sopr, exchange_netflow

    Note:
        Free tier limitations:
        - 24h resolution only
        - Limited metrics (MVRV, SOPR, exchange flows available)
        - 20 requests per 10 minutes
    """
    api_key = os.getenv('GLASSNODE_API_KEY')

    if not api_key:
        logger.warning("GLASSNODE_API_KEY not found in environment - skipping API fetch")
        return None

    glassnode_sym = _glassnode_symbol(symbol)

    # Glassnode only supports major coins (BTC, ETH, LTC, etc.)
    if glassnode_sym not in ['BTC', 'ETH', 'LTC']:
        logger.info(f"Glassnode doesn't support {glassnode_sym} - skipping API fetch")
        return None

    # Default time range: last 365 days
    if since_timestamp is None:
        since_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
    if until_timestamp is None:
        until_timestamp = int(datetime.now().timestamp())

    base_url = "https://api.glassnode.com/v1/metrics"

    # Free tier metrics
    metrics = {
        'mvrv_z': f"{base_url}/market/mvrv_z_score",
        'sopr': f"{base_url}/indicators/sopr",
        'exchange_netflow': f"{base_url}/distribution/exchange_net_position_change"
    }

    params = {
        'a': glassnode_sym,
        'api_key': api_key,
        's': since_timestamp,
        'u': until_timestamp,
        'i': resolution,
        'f': 'JSON'
    }

    results = {}

    try:
        for metric_name, url in metrics.items():
            logger.info(f"Fetching Glassnode {metric_name} for {glassnode_sym}...")

            response = requests.get(url, params=params, timeout=30)

            if response.status_code != 200:
                logger.error(
                    f"Glassnode API error for {metric_name}: {response.status_code} - {response.text}"
                )
                # Free tier limitations - might not have access to all metrics
                if response.status_code == 402:
                    logger.warning(f"Metric {metric_name} requires paid tier - skipping")
                    continue
                return None

            data = response.json()

            if not data:
                logger.warning(f"No data returned for {metric_name}")
                continue

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['event_time'] = pd.to_datetime(df['t'], unit='s', utc=True)
            df[metric_name] = df['v']
            df = df[['event_time', metric_name]]

            results[metric_name] = df

            logger.success(f"Fetched {len(df)} {metric_name} data points")

            # Rate limiting: wait between requests (free tier: 20 req/10 min)
            import time
            time.sleep(30)  # Conservative: 2 requests/minute

        if not results:
            logger.warning("No metrics successfully fetched from Glassnode")
            return None

        # Merge all metrics on event_time
        merged = None
        for metric_name, df in results.items():
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on='event_time', how='outer')

        # Sort by time
        merged = merged.sort_values('event_time').reset_index(drop=True)

        # Forward fill missing values (on-chain data can be sparse)
        merged = merged.ffill()

        logger.success(f"Successfully fetched {len(merged)} rows of on-chain data from Glassnode")

        return merged

    except Exception as e:
        logger.error(f"Error fetching Glassnode data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


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
    prefer_glassnode: bool = True,
    prefer_local_csv: bool = True,
    output_store: Optional[FeatureStore] = None,
) -> bool:
    """
    Materialize on-chain features for a symbol into the FeatureStore.

    Priority order:
    1. Glassnode API (if prefer_glassnode=True and GLASSNODE_API_KEY is set)
    2. Local CSV (if prefer_local_csv=True)
    3. Proxy features from OHLCV (fallback for testing)

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for proxy fallback
        prefer_glassnode: Try Glassnode API first (default: True)
        prefer_local_csv: Try local CSV second (default: True)
        output_store: FeatureStore instance

    Returns:
        True if features were successfully ingested, False otherwise
    """
    store = output_store or FeatureStore()

    df = None

    # Priority 1: Try Glassnode API
    if prefer_glassnode:
        logger.info(f"Attempting to fetch on-chain data from Glassnode for {symbol}...")
        df = fetch_glassnode_metrics(symbol)

        if df is not None:
            logger.success(f"Successfully fetched {len(df)} rows from Glassnode API")

    # Priority 2: Try local CSV
    if df is None and prefer_local_csv:
        logger.info(f"Attempting to load on-chain data from local CSV for {symbol}...")
        df = load_local_onchain_csv(symbol)

        if df is not None:
            logger.success(f"Successfully loaded {len(df)} rows from local CSV")

    # Priority 3: Fallback to proxy features
    if df is None:
        logger.warning(
            f"No real on-chain data available for {symbol}. "
            f"Generating proxy features from OHLCV for pipeline testing."
        )
        df = _proxy_from_ohlcv(symbol, timeframe)

        if df is not None:
            logger.info(f"Generated {len(df)} rows of proxy on-chain features")

    if df is None or df.empty:
        logger.error(f"Failed to obtain any on-chain features for {symbol}")
        return False

    ok = store.write(df, symbol=symbol, pillar="onchain")

    if ok:
        logger.success(f"Successfully ingested on-chain features for {symbol} into FeatureStore")
    else:
        logger.error(f"Failed to write on-chain features to FeatureStore for {symbol}")

    return ok

