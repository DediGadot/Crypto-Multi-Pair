#!/usr/bin/env python3
"""
Debug script to understand why optimize_portfolio_optimized.py is failing.

This script will:
1. Load the data the same way the optimizer does
2. Try a single configuration
3. Print detailed debugging information at each step
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from loguru import logger

# Add src directory to Python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from crypto_trader.data.fetchers import BinanceDataFetcher

# Import the Numba-compiled functions
from optimize_portfolio_optimized import (
    simulate_portfolio_rebalancing_numba,
    calculate_sharpe_ratio_numba,
    calculate_max_drawdown_numba,
    calculate_volatility_numba
)

logger.info("=" * 80)
logger.info("DEBUG: Optimization Failure Analysis")
logger.info("=" * 80)

# Step 1: Fetch data
logger.info("\n1. Fetching historical data...")
fetcher = BinanceDataFetcher()
symbols = ["BTC/USDT", "ETH/USDT"]
timeframe = "1h"
window_days = 365
test_windows = 5

total_days = window_days * (test_windows + 1)
limit = total_days * 24  # For 1h timeframe

historical_data = {}
for symbol in symbols:
    try:
        data = fetcher.get_ohlcv(symbol, timeframe, limit=limit)
        if data is not None and len(data) > 0:
            historical_data[symbol] = data
            logger.success(f"  ✓ {symbol}: {len(data)} candles")
        else:
            logger.error(f"  ✗ {symbol}: No data")
    except Exception as e:
        logger.error(f"  ✗ {symbol}: {e}")

if len(historical_data) < 2:
    logger.error("FAILURE: Need at least 2 assets with data")
    sys.exit(1)

# Step 2: Convert to NumPy arrays
logger.info("\n2. Converting to NumPy arrays...")
price_arrays = {}
timestamp_arrays = {}

for symbol, df in historical_data.items():
    price_arrays[symbol] = df['close'].values
    timestamp_arrays[symbol] = df.index.values
    logger.info(f"  {symbol}: {len(price_arrays[symbol])} prices")

# Step 3: Create walk-forward splits
logger.info("\n3. Creating walk-forward splits...")

# Get common timestamps
timestamps = None
for ts_arr in timestamp_arrays.values():
    if timestamps is None:
        timestamps = ts_arr
    else:
        timestamps = np.intersect1d(timestamps, ts_arr)

timestamps = np.sort(timestamps)
logger.info(f"  Common timestamps: {len(timestamps)}")

periods_per_window = window_days * 24
required_periods = periods_per_window * (test_windows + 1)
logger.info(f"  Periods per window: {periods_per_window}")
logger.info(f"  Required periods: {required_periods}")
logger.info(f"  Available periods: {len(timestamps)}")

if len(timestamps) < required_periods:
    logger.error(f"FAILURE: Insufficient data")
    logger.error(f"  Need: {required_periods} periods")
    logger.error(f"  Have: {len(timestamps)} periods")
    logger.error(f"  Missing: {required_periods - len(timestamps)} periods")
    sys.exit(1)

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
    logger.error("FAILURE: No splits created")
    sys.exit(1)

# Step 4: Try backtesting one split
logger.info("\n4. Testing backtest on first split...")

train_start, train_end, test_end = splits[0]
logger.info(f"  Train: {train_start} to {train_end}")
logger.info(f"  Test: {train_end} to {test_end}")

# Find indices
train_start_idx = np.searchsorted(timestamps, train_start, side='left')
train_end_idx = np.searchsorted(timestamps, train_end, side='right')
test_end_idx = np.searchsorted(timestamps, test_end, side='right')

logger.info(f"  Train indices: {train_start_idx} to {train_end_idx}")
logger.info(f"  Test indices: {train_end_idx} to {test_end_idx}")
logger.info(f"  Train periods: {train_end_idx - train_start_idx}")
logger.info(f"  Test periods: {test_end_idx - train_end_idx}")

# Step 5: Try simulating on train period
logger.info("\n5. Testing simulation on train period...")

assets = [("BTC/USDT", 0.5), ("ETH/USDT", 0.5)]
weights = np.array([0.5, 0.5])

# Get price arrays for train period
train_slice = slice(train_start_idx, train_end_idx)
price_arrays_list = []
for symbol, _ in assets:
    if symbol in price_arrays:
        price_arr = price_arrays[symbol][train_slice]
        price_arrays_list.append(price_arr)
        logger.info(f"  {symbol} train prices: shape={price_arr.shape}, min={price_arr.min():.2f}, max={price_arr.max():.2f}")
    else:
        logger.error(f"  {symbol} not in price_arrays!")

if len(price_arrays_list) < 2:
    logger.error("FAILURE: Insufficient price arrays")
    sys.exit(1)

# Stack into 2D array
price_arrays_2d = np.vstack(price_arrays_list)
logger.info(f"  Stacked price array shape: {price_arrays_2d.shape}")

# Check for invalid values
if np.any(np.isnan(price_arrays_2d)):
    logger.error(f"  FAILURE: NaN values in price array")
    sys.exit(1)

if np.any(price_arrays_2d <= 0):
    logger.error(f"  FAILURE: Non-positive prices in array")
    sys.exit(1)

# Step 6: Try running Numba simulation
logger.info("\n6. Testing Numba simulation...")

try:
    rebalance_params = {
        'threshold': 0.10,
        'min_rebalance_interval_hours': 24,
        'use_momentum_filter': False,
        'rebalance_method': 'threshold',
        'calendar_period_days': 30
    }

    rebalance_method = 0  # threshold
    periods_per_year = 24 * 365
    initial_capital = 10000.0

    logger.info(f"  Parameters:")
    logger.info(f"    Initial capital: {initial_capital}")
    logger.info(f"    Weights: {weights}")
    logger.info(f"    Threshold: {rebalance_params['threshold']}")
    logger.info(f"    Method: {rebalance_method}")

    equity_curve, rebalance_count = simulate_portfolio_rebalancing_numba(
        price_arrays_2d,
        weights,
        initial_capital,
        rebalance_params['threshold'],
        24,  # min_rebalance_interval_hours
        False,  # use_momentum_filter
        30 * 24,  # momentum_lookback
        rebalance_method,
        30 * 24  # calendar_period
    )

    logger.success(f"  ✓ Simulation completed!")
    logger.info(f"    Equity curve shape: {equity_curve.shape}")
    logger.info(f"    Initial value: {equity_curve[0]:.2f}")
    logger.info(f"    Final value: {equity_curve[-1]:.2f}")
    logger.info(f"    Return: {(equity_curve[-1] / equity_curve[0] - 1) * 100:.2f}%")
    logger.info(f"    Rebalances: {rebalance_count}")

    # Check for invalid equity values
    if np.any(np.isnan(equity_curve)):
        logger.error(f"  FAILURE: NaN values in equity curve")
        logger.error(f"    First NaN at index: {np.where(np.isnan(equity_curve))[0][0]}")
        sys.exit(1)

    if np.any(equity_curve <= 0):
        logger.error(f"  FAILURE: Non-positive equity values")
        logger.error(f"    First invalid at index: {np.where(equity_curve <= 0)[0][0]}")
        sys.exit(1)

    # Step 7: Calculate metrics
    logger.info("\n7. Testing metric calculations...")

    returns = np.diff(equity_curve) / equity_curve[:-1]
    logger.info(f"  Returns: shape={returns.shape}, mean={returns.mean():.6f}, std={returns.std():.6f}")

    sharpe = calculate_sharpe_ratio_numba(returns, periods_per_year)
    logger.info(f"  Sharpe ratio: {sharpe:.3f}")

    max_dd = calculate_max_drawdown_numba(equity_curve)
    logger.info(f"  Max drawdown: {max_dd:.2%}")

    volatility = calculate_volatility_numba(returns, periods_per_year)
    logger.info(f"  Volatility: {volatility:.2%}")

    logger.success("\n✅ ALL TESTS PASSED - Simulation is working correctly!")
    logger.info("\nThis means the issue is likely in:")
    logger.info("  1. Worker process initialization")
    logger.info("  2. Data sharing between processes")
    logger.info("  3. Exception handling that's swallowing errors")

except Exception as e:
    logger.error(f"\n❌ SIMULATION FAILED: {e}")
    logger.exception("Full traceback:")
    sys.exit(1)

logger.info("\n" + "=" * 80)
logger.info("Debug analysis complete")
logger.info("=" * 80)
