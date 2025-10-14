#!/usr/bin/env python3
"""
Debug script to understand why backtests are failing even with sufficient data.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from loguru import logger

script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from crypto_trader.data.fetchers import BinanceDataFetcher
from optimize_portfolio_optimized import simulate_portfolio_rebalancing_numba

logger.info("=" * 80)
logger.info("DETAILED BACKTEST FAILURE ANALYSIS")
logger.info("=" * 80)

# Use parameters that we know create splits
timeframe = "1d"
window_days = 235
test_windows = 5

# Fetch data
fetcher = BinanceDataFetcher()
symbols = ["BTC/USDT", "ETH/USDT"]

logger.info(f"\n1. Fetching data...")
historical_data = {}
for symbol in symbols:
    data = fetcher.get_ohlcv(symbol, timeframe, limit=10000)
    if data is not None:
        historical_data[symbol] = data
        logger.success(f"  ✓ {symbol}: {len(data)} periods")

# Convert to NumPy
logger.info(f"\n2. Converting to NumPy...")
price_arrays = {}
timestamp_arrays = {}

for symbol, df in historical_data.items():
    price_arrays[symbol] = df['close'].values
    timestamp_arrays[symbol] = df.index.values
    logger.info(f"  {symbol}: {len(price_arrays[symbol])} prices")

# Get common timestamps
logger.info(f"\n3. Getting common timestamps...")
timestamps = None
for ts_arr in timestamp_arrays.values():
    if timestamps is None:
        timestamps = ts_arr
    else:
        timestamps = np.intersect1d(timestamps, ts_arr)

timestamps = np.sort(timestamps)
logger.info(f"  Common timestamps: {len(timestamps)}")

# Create splits
logger.info(f"\n4. Creating splits...")
periods_per_window = window_days
splits = []

for i in range(test_windows):
    train_start_idx = 0
    train_end_idx = periods_per_window * (i + 1)
    test_end_idx = periods_per_window * (i + 2)

    if test_end_idx > len(timestamps):
        break

    splits.append((
        timestamps[train_start_idx],
        timestamps[train_end_idx - 1],
        timestamps[test_end_idx - 1]
    ))

logger.info(f"  Created {len(splits)} splits")

if len(splits) == 0:
    logger.error("No splits created!")
    sys.exit(1)

# Try backtest on first split
logger.info(f"\n5. Testing backtest on first split...")
train_start, train_end, test_end = splits[0]

# Find indices
train_start_idx = np.searchsorted(timestamps, train_start, side='left')
train_end_idx = np.searchsorted(timestamps, train_end, side='right')
test_end_idx = np.searchsorted(timestamps, test_end, side='right')

logger.info(f"  Train indices: {train_start_idx} to {train_end_idx} ({train_end_idx - train_start_idx} periods)")
logger.info(f"  Test indices: {train_end_idx} to {test_end_idx} ({test_end_idx - train_end_idx} periods)")

# Get price arrays for train period
train_slice = slice(train_start_idx, train_end_idx)
assets = [("BTC/USDT", 0.5), ("ETH/USDT", 0.5)]
weights = np.array([0.5, 0.5])

price_arrays_list = []
for symbol, _ in assets:
    price_arr = price_arrays[symbol][train_slice]
    price_arrays_list.append(price_arr)
    logger.info(f"  {symbol}: {len(price_arr)} prices, range [{price_arr.min():.2f}, {price_arr.max():.2f}]")

price_arrays_2d = np.vstack(price_arrays_list)
logger.info(f"  Stacked array shape: {price_arrays_2d.shape}")

# Check for issues
if np.any(np.isnan(price_arrays_2d)):
    logger.error("  ERROR: NaN values in price array")
    sys.exit(1)

if np.any(price_arrays_2d <= 0):
    logger.error("  ERROR: Non-positive prices")
    sys.exit(1)

# Try simulation
logger.info(f"\n6. Running simulation...")
try:
    initial_capital = 10000.0
    rebalance_threshold = 0.10
    min_rebalance_interval = 1  # 1 day for daily data
    use_momentum_filter = False
    momentum_lookback = 30
    rebalance_method = 0  # threshold
    calendar_period = 30

    logger.info(f"  Initial capital: {initial_capital}")
    logger.info(f"  Weights: {weights}")
    logger.info(f"  Periods: {price_arrays_2d.shape[1]}")

    equity_curve, rebalance_count = simulate_portfolio_rebalancing_numba(
        price_arrays_2d,
        weights,
        initial_capital,
        rebalance_threshold,
        min_rebalance_interval,
        use_momentum_filter,
        momentum_lookback,
        rebalance_method,
        calendar_period
    )

    logger.success(f"  ✓ Simulation succeeded!")
    logger.info(f"  Equity curve length: {len(equity_curve)}")
    logger.info(f"  Initial value: {equity_curve[0]:.2f}")
    logger.info(f"  Final value: {equity_curve[-1]:.2f}")
    logger.info(f"  Return: {(equity_curve[-1] / equity_curve[0] - 1) * 100:.2f}%")
    logger.info(f"  Rebalances: {rebalance_count}")

    # Check for problems
    if np.any(np.isnan(equity_curve)):
        logger.error(f"  ERROR: NaN in equity curve")
        nan_idx = np.where(np.isnan(equity_curve))[0][0]
        logger.error(f"    First NaN at index {nan_idx}")
        logger.error(f"    Prices at that time: {price_arrays_2d[:, nan_idx]}")
        sys.exit(1)

    if np.any(equity_curve <= 0):
        logger.error(f"  ERROR: Non-positive equity")
        bad_idx = np.where(equity_curve <= 0)[0][0]
        logger.error(f"    First bad value at index {bad_idx}: {equity_curve[bad_idx]}")
        sys.exit(1)

    # Calculate metrics
    logger.info(f"\n7. Calculating metrics...")
    returns = np.diff(equity_curve) / equity_curve[:-1]
    logger.info(f"  Returns shape: {returns.shape}")
    logger.info(f"  Returns mean: {returns.mean():.6f}")
    logger.info(f"  Returns std: {returns.std():.6f}")

    if np.any(np.isnan(returns)):
        logger.error(f"  ERROR: NaN in returns")
        sys.exit(1)

    logger.success(f"\n✅ ALL CHECKS PASSED!")
    logger.info(f"\nConclusion: The backtest simulation works correctly.")
    logger.info(f"The issue must be in how process_configuration is being called or how data is shared across processes.")

except Exception as e:
    logger.error(f"\n❌ SIMULATION FAILED: {e}")
    logger.exception("Full traceback:")
    sys.exit(1)
