"""
Order flow feature ingestion.

Supports two modes:
1. Preferred: Read precomputed trade/order book snapshots from CSV
2. Optional: Use ccxt.pro websockets for live collection (best-effort)

In both cases the resulting features are written to the FeatureStore under the
`micro` pillar.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import ccxtpro  # type: ignore
except Exception:  # pragma: no cover
    ccxtpro = None

from crypto_trader.features.store import FeatureStore


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def load_local_orderflow(symbol: str, base_dir: str | Path = "data/orderflow") -> Optional[pd.DataFrame]:
    path = Path(base_dir) / f"{_safe_symbol(symbol)}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["event_time"])
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df.sort_values("event_time")


def _calculate_vpin(trade_sizes: pd.Series, bucket_size: int = 50) -> pd.Series:
    buckets = trade_sizes.abs().rolling(bucket_size).sum()
    signed_flow = trade_sizes.rolling(bucket_size).sum()
    vpin = (signed_flow.abs() / (buckets + 1e-9)).rename("vpin")
    return vpin


def _build_features(trades: pd.DataFrame, orderbook: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["signed_volume"] = trades.apply(
        lambda row: row["amount"] if row["side"] == "buy" else -row["amount"], axis=1
    )
    trades["delta"] = trades["signed_volume"]
    trades["cumulative_delta"] = trades["delta"].cumsum()
    trades["vpin"] = _calculate_vpin(trades["delta"], bucket_size=50)

    orderbook = orderbook.copy()
    orderbook["event_time"] = pd.to_datetime(orderbook["event_time"], utc=True)
    trades["event_time"] = pd.to_datetime(trades["event_time"], utc=True)
    merged = pd.merge_asof(
        trades.sort_values("event_time"),
        orderbook.sort_values("event_time"),
        on="event_time",
        direction="backward",
    )

    merged["book_imbalance"] = (
        (merged.get("bid_depth", 0) - merged.get("ask_depth", 0))
        / (merged.get("bid_depth", 0) + merged.get("ask_depth", 0) + 1e-9)
    )
    cols = [
        "event_time",
        "delta",
        "cumulative_delta",
        "book_imbalance",
        "vpin",
    ]
    return merged[cols].dropna()


@dataclass
class OrderFlowConfig:
    symbol: str
    timeframe: str = "1m"
    depth: int = 10
    duration_minutes: int = 30


async def _collect_live_orderflow(config: OrderFlowConfig) -> Optional[pd.DataFrame]:
    if ccxtpro is None:
        logger.warning("ccxt.pro not installed - skipping live order flow collection")
        return None

    exchange = ccxtpro.binance({"enableRateLimit": True})  # type: ignore[attr-defined]
    trades = []
    orderbooks = []

    try:
        end_time = asyncio.get_event_loop().time() + config.duration_minutes * 60
        while asyncio.get_event_loop().time() < end_time:
            trade = await exchange.watch_trades(config.symbol)
            orderbook = await exchange.watch_order_book(config.symbol, limit=config.depth)
            timestamp = pd.Timestamp.utcnow()
            if trade:
                for t in trade:
                    trades.append(
                        {
                            "event_time": pd.to_datetime(t["timestamp"], unit="ms", utc=True),
                            "amount": t["amount"],
                            "side": t.get("side", "buy"),
                        }
                    )
            if orderbook:
                bid_depth = sum(level[1] for level in orderbook["bids"][: config.depth])
                ask_depth = sum(level[1] for level in orderbook["asks"][: config.depth])
                orderbooks.append(
                    {
                        "event_time": timestamp,
                        "bid_depth": bid_depth,
                        "ask_depth": ask_depth,
                    }
                )
    except Exception as exc:
        logger.warning(f"Order flow websocket interrupted: {exc}")
    finally:
        await exchange.close()

    if not trades or not orderbooks:
        return None

    return _build_features(pd.DataFrame(trades), pd.DataFrame(orderbooks))


def ingest_orderflow(
    symbol: str,
    *,
    timeframe: str = "1m",
    prefer_local_csv: bool = True,
    store: Optional[FeatureStore] = None,
    live_config: Optional[OrderFlowConfig] = None,
) -> bool:
    """
    Materialize order flow features and write them into the FeatureStore.
    """
    feature_store = store or FeatureStore()

    if prefer_local_csv:
        df = load_local_orderflow(symbol)
        if df is not None and not df.empty:
            df = df.rename(columns={col: f"micro.{col}" for col in df.columns if col != "event_time"})
            return feature_store.write(df, symbol=symbol, pillar="micro")

    features = None
    if live_config:
        try:
            features = asyncio.run(_collect_live_orderflow(live_config))
        except RuntimeError:
            # Event loop already running (e.g., inside Jupyter) - fall back to new loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                features = loop.run_until_complete(_collect_live_orderflow(live_config))
            finally:
                asyncio.set_event_loop(None)

    if features is None or features.empty:
        logger.warning("Order flow ingestion failed - no data collected")
        return False

    features = features.rename(columns={col: f"micro.{col}" for col in features.columns if col != "event_time"})
    success = feature_store.write(features, symbol=symbol, pillar="micro")
    if success:
        logger.success(f"Order flow features ingested for {symbol}")
    return success
