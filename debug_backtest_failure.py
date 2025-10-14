#!/usr/bin/env python3
"""
Debug Why All Backtests Return None

The optimization ran but got 0/12 valid results. This script traces through
a single backtest to identify exactly where it's failing.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
from loguru import logger
from crypto_trader.data.fetchers import BinanceDataFetcher

logger.info("="*80)
logger.info("DEBUGGING BACKTEST FAILURE")
logger.info("="*80)

# Simulate what the optimizer does
logger.info("\n1. Fetch data like optimizer does...")
fetcher = BinanceDataFetcher()

symbols = ["BTC/USDT", "ETH/USDT"]
timeframe = "1h"
limit = 365 * 6 * 24  # 52,560 (365 days × 6 windows × 24 hours)

logger.info(f"   Fetching {limit:,} candles for {symbols}...")

historical_data = {}
for symbol in symbols:
    df = fetcher.get_ohlcv(symbol, timeframe, limit=limit)
    if df is not None and len(df) >= limit * 0.5:
        historical_data[symbol] = df
        logger.success(f"   ✓ {symbol}: {len(df)} candles")
    else:
        logger.error(f"   ✗ {symbol}: Only {len(df) if df is not None else 0} candles")

# Convert to NumPy
logger.info("\n2. Convert to NumPy arrays...")
price_arrays = {}
timestamp_arrays = {}

for symbol, df in historical_data.items():
    price_arrays[symbol] = df['close'].values
    timestamp_arrays[symbol] = df.index.values
    logger.info(f"   {symbol}: {len(price_arrays[symbol])} prices")

# Create splits
logger.info("\n3. Create walk-forward splits...")
window_days = 365
test_windows = 5

# Get common timestamps
timestamps = None
for ts_arr in timestamp_arrays.values():
    if timestamps is None:
        timestamps = ts_arr
    else:
        timestamps = np.intersect1d(timestamps, ts_arr)

timestamps = np.sort(timestamps)
logger.info(f"   Common timestamps: {len(timestamps)}")

periods_per_window = window_days * 24

splits = []
for i in range(test_windows):
    train_start_idx = 0
    train_end_idx = periods_per_window * (i + 1)
    test_end_idx = periods_per_window * (i + 2)

    if test_end_idx > len(timestamps):
        logger.warning(f"   Split {i+1}: Would need {test_end_idx} timestamps, but only have {len(timestamps)}")
        break

    splits.append((
        timestamps[train_start_idx],
        timestamps[train_end_idx - 1],
        timestamps[test_end_idx - 1]
    ))
    logger.info(f"   Split {i+1}: indices [{train_start_idx}, {train_end_idx}, {test_end_idx}]")

logger.info(f"   Created {len(splits)} splits")

# Test a backtest on first split
if splits:
    logger.info("\n4. Test backtest on first split...")
    train_start, train_end, test_end = splits[0]

    # Get indices
    ref_symbol = "BTC/USDT"
    timestamps_ref = timestamp_arrays[ref_symbol]

    train_start_idx = np.searchsorted(timestamps_ref, train_start, side='left')
    train_end_idx = np.searchsorted(timestamps_ref, train_end, side='right')
    test_end_idx = np.searchsorted(timestamps_ref, test_end, side='right')

    logger.info(f"   Train: [{train_start_idx}, {train_end_idx}] = {train_end_idx - train_start_idx} periods")
    logger.info(f"   Test:  [{train_end_idx}, {test_end_idx}] = {test_end_idx - train_end_idx} periods")

    # Check if test period is valid
    test_period_length = test_end_idx - train_end_idx
    if test_period_length < 10:
        logger.error(f"   ❌ Test period too short: {test_period_length} < 10")
        logger.error("   THIS IS WHY BACKTESTS ARE FAILING!")
    else:
        logger.success(f"   ✓ Test period OK: {test_period_length} periods")

    # Check train period
    train_period_length = train_end_idx - train_start_idx
    if train_period_length < 10:
        logger.error(f"   ❌ Train period too short: {train_period_length} < 10")
    else:
        logger.success(f"   ✓ Train period OK: {train_period_length} periods")

else:
    logger.error("\n❌ No splits created!")

logger.info("\n" + "="*80)
logger.info("DIAGNOSIS COMPLETE")
logger.info("="*80)

# Summary
logger.info("\n📋 FINDINGS:")
logger.info(f"   Available data: {len(timestamps):,} candles")
logger.info(f"   Required for {test_windows} windows: {periods_per_window * (test_windows + 1):,} candles")
logger.info(f"   Splits created: {len(splits)}/{test_windows}")

if len(splits) < test_windows:
    logger.error(f"\n💡 PROBLEM: Not enough data for {test_windows} test windows!")
    logger.error(f"   Have: {len(timestamps):,} candles")
    logger.error(f"   Need: {periods_per_window * (test_windows + 1):,} candles")
    logger.error(f"   Missing: {periods_per_window * (test_windows + 1) - len(timestamps):,} candles")
