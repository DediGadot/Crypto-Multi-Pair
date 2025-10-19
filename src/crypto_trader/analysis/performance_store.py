"""
Lightweight persistence for recent backtest metrics.

The Dynamic Ensemble strategy relies on historical performance snapshots for
weighting component strategies. This module stores metrics in a CSV file,
keeping the dependency surface minimal while still allowing quick analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import pandas as pd
from loguru import logger

from crypto_trader.core.types import BacktestResult


DEFAULT_STORE_PATH = Path("data/performance/performance_metrics.csv")


@dataclass
class PerformanceRecord:
    timestamp: pd.Timestamp
    strategy: str
    symbol: str
    timeframe: str
    sharpe: float
    total_return: float
    max_drawdown: float
    win_rate: float


class PerformanceStore:
    """
    Simple CSV-backed storage for performance metrics.
    """

    def __init__(self, path: Path = DEFAULT_STORE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "strategy",
                    "symbol",
                    "timeframe",
                    "sharpe",
                    "total_return",
                    "max_drawdown",
                    "win_rate",
                ]
            )
        try:
            # Read CSV with robust error handling for malformed rows
            df = pd.read_csv(
                self.path,
                parse_dates=["timestamp"],
                on_bad_lines='warn'  # Warn about bad lines but continue
            )

            # Ensure timestamp column is datetime type and handle any parsing failures
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

                # Count and warn about invalid timestamps
                invalid_count = df['timestamp'].isna().sum()
                if invalid_count > 0:
                    logger.warning(
                        f"Found {invalid_count} rows with invalid timestamps in {self.path} - "
                        f"these will be dropped"
                    )
                    # Drop rows where timestamp couldn't be parsed
                    df = df.dropna(subset=['timestamp'])

            logger.debug(f"Loaded {len(df)} valid performance records from {self.path}")
            return df
        except Exception as exc:  # pragma: no cover - unexpected IO failure
            logger.warning(f"Failed to load performance store {self.path}: {exc}")
            return pd.DataFrame()

    def record(self, result: Union[BacktestResult, Mapping[str, Any]]) -> None:
        df = self._load()
        if isinstance(result, BacktestResult):
            metrics = result.metrics
            record = {
                "timestamp": pd.Timestamp.utcnow(),
                "strategy": result.strategy_name,
                "symbol": result.symbol,
                "timeframe": result.timeframe.value,
                "sharpe": metrics.sharpe_ratio,
                "total_return": metrics.total_return,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
            }
        else:
            record = {
                "timestamp": pd.Timestamp.utcnow(),
                "strategy": str(result.get("strategy_name", "unknown")),
                "symbol": str(result.get("symbol", "unknown")),
                "timeframe": str(result.get("timeframe", "1h")),
                "sharpe": float(result.get("sharpe_ratio", 0.0)),
                "total_return": float(result.get("total_return", 0.0)),
                "max_drawdown": float(result.get("max_drawdown", 0.0)),
                "win_rate": float(result.get("win_rate", 0.0)),
            }
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.sort_values("timestamp", inplace=True)
        df.to_csv(self.path, index=False)
        strategy_name = record["strategy"]
        logger.debug(f"Performance store updated for {strategy_name}")

    def recent(
        self,
        strategy_names: Optional[Iterable[str]] = None,
        days: int = 90,
    ) -> pd.DataFrame:
        df = self._load()
        if df.empty:
            return df

        if strategy_names:
            df = df[df["strategy"].isin(list(strategy_names))]
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
        return df
