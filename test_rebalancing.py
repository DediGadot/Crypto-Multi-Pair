#!/usr/bin/env python3
"""Test if rebalancing is actually happening."""

import sys
from pathlib import Path
import numpy as np
from loguru import logger

script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from crypto_trader.data.fetchers import BinanceDataFetcher
from optimize_portfolio_optimized import simulate_portfolio_rebalancing_numba

# Fetch minimal data
fetcher = BinanceDataFetcher()
symbols = ["BTC/USDT", "ETH/USDT"]

logger.info("Fetching 100 days of data...")
data = {}
for symbol in symbols:
    df = fetcher.get_ohlcv(symbol, "1d", limit=100)
    data[symbol] = df['close'].values

# Stack prices
prices = np.vstack([data[s] for s in symbols])
weights = np.array([0.5, 0.5])

logger.info(f"Price array shape: {prices.shape}")
logger.info(f"Simulating with 10% threshold...")

# Test with 10% threshold
equity_curve_10, rebalances_10 = simulate_portfolio_rebalancing_numba(
    prices, weights, 10000.0, 0.10, 1, False, 30, 0, 30
)

logger.info(f"  Rebalances with 10% threshold: {rebalances_10}")
logger.info(f"  Final equity: ${equity_curve_10[-1]:.2f}")

# Test with 15% threshold
logger.info(f"Simulating with 15% threshold...")
equity_curve_15, rebalances_15 = simulate_portfolio_rebalancing_numba(
    prices, weights, 10000.0, 0.15, 1, False, 30, 0, 30
)

logger.info(f"  Rebalances with 15% threshold: {rebalances_15}")
logger.info(f"  Final equity: ${equity_curve_15[-1]:.2f}")

# Test with 5% threshold (should rebalance more)
logger.info(f"Simulating with 5% threshold...")
equity_curve_5, rebalances_5 = simulate_portfolio_rebalancing_numba(
    prices, weights, 10000.0, 0.05, 1, False, 30, 0, 30
)

logger.info(f"  Rebalances with 5% threshold: {rebalances_5}")
logger.info(f"  Final equity: ${equity_curve_5[-1]:.2f}")

logger.info(f"\nSummary:")
logger.info(f"  5% threshold:  {rebalances_5} rebalances")
logger.info(f"  10% threshold: {rebalances_10} rebalances")
logger.info(f"  15% threshold: {rebalances_15} rebalances")

if rebalances_10 == 0 and rebalances_15 == 0 and rebalances_5 == 0:
    logger.error("❌ NO REBALANCING HAPPENING AT ALL!")
elif rebalances_10 == rebalances_15:
    logger.warning("⚠️  10% and 15% thresholds produce same rebalance count")
else:
    logger.success("✓ Different thresholds produce different rebalance counts")
