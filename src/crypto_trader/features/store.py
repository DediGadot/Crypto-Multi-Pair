"""
Lightweight Feature Store (CSV-based)

This module provides a minimal, dependency-free feature store for alternative
data pillars (on-chain, sentiment, microstructure, options). It reads/writes
CSV files under a conventional directory structure and exposes a simple API to
load feature frames for a symbol and time range.

Path layout (by default):
    data/features/{pillar}/{symbol_safe}.csv

CSV format:
    Columns: [event_time, <feature columns...>]
    Index: None (event_time is a column parsed as datetime)

Notes:
- CSV is used to avoid adding heavy parquet dependencies. The API can be
  upgraded to Parquet without changing callers.
- This store is intentionally forgiving: reads return empty DataFrames if
  files are missing; callers should handle optional joins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from loguru import logger


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


@dataclass
class FeatureReadRequest:
    symbol: str
    pillars: Optional[List[str]] = None  # e.g., ["onchain", "sent", "micro", "opt"]
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None


class FeatureStore:
    """CSV-backed feature store with a tiny API surface."""

    def __init__(self, base_path: str | Path = "data/features"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"FeatureStore initialized at {self.base_path.resolve()}")

    def _pillar_path(self, pillar: str, symbol: str) -> Path:
        safe = _safe_symbol(symbol)
        p = self.base_path / pillar
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{safe}.csv"

    def write(self, df: pd.DataFrame, symbol: str, pillar: str) -> bool:
        """Persist features for a pillar. Overwrites existing file.

        Args:
            df: Must contain an 'event_time' column or a DatetimeIndex.
        """
        try:
            if df is None or df.empty:
                logger.warning(f"FeatureStore.write: empty frame for {pillar} {symbol}")
                return False

            out = df.copy()
            if 'event_time' not in out.columns:
                # If index is datetime, turn into event_time column
                if isinstance(out.index, pd.DatetimeIndex):
                    out = out.reset_index().rename(columns={'index': 'event_time'})
                else:
                    raise ValueError("Features must include an 'event_time' column or DatetimeIndex")

            # Ensure datetime and sorted
            out['event_time'] = pd.to_datetime(out['event_time'], utc=True, errors='coerce')
            out = out.dropna(subset=['event_time']).sort_values('event_time')

            path = self._pillar_path(pillar, symbol)
            out.to_csv(path, index=False)
            logger.info(f"FeatureStore: wrote {len(out)} rows to {path}")
            return True
        except Exception as e:
            logger.error(f"FeatureStore.write failed for {pillar} {symbol}: {e}")
            return False

    def read_one(self, symbol: str, pillar: str) -> pd.DataFrame:
        """Read a single pillar file, return empty DataFrame if missing."""
        path = self._pillar_path(pillar, symbol)
        try:
            if not path.exists():
                return pd.DataFrame()
            df = pd.read_csv(path, parse_dates=['event_time'])
            # Coerce UTC
            df['event_time'] = pd.to_datetime(df['event_time'], utc=True, errors='coerce')
            df = df.dropna(subset=['event_time']).sort_values('event_time')
            return df
        except Exception as e:
            logger.error(f"FeatureStore.read_one failed for {pillar} {symbol}: {e}")
            return pd.DataFrame()

    def read(self, req: FeatureReadRequest) -> Dict[str, pd.DataFrame]:
        """Read one or more pillars for a symbol and optional time window."""
        pillars = req.pillars or []
        result: Dict[str, pd.DataFrame] = {}
        for pillar in pillars:
            df = self.read_one(req.symbol, pillar)
            if df.empty:
                result[pillar] = df
                continue
            if req.start is not None:
                df = df[df['event_time'] >= pd.to_datetime(req.start, utc=True)]
            if req.end is not None:
                df = df[df['event_time'] <= pd.to_datetime(req.end, utc=True)]
            result[pillar] = df
        return result

