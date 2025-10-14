#!/usr/bin/env python3
"""
DEBUG VERSION - Find why all results are None

Quick diagnostic script to identify the optimization issue.
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

# Test 1: Can we fetch data?
logger.info("TEST 1: Fetching BTC data...")
fetcher = BinanceDataFetcher()
btc_data = fetcher.get_ohlcv("BTC/USDT", "1h", limit=1000)
logger.info(f"✓ Fetched {len(btc_data)} candles")
logger.info(f"  Date range: {btc_data.index[0]} to {btc_data.index[-1]}")

# Test 2: Can we convert to NumPy?
logger.info("\nTEST 2: Converting to NumPy arrays...")
prices = btc_data['close'].values
timestamps = btc_data.index.values
logger.info(f"✓ Prices shape: {prices.shape}")
logger.info(f"✓ Timestamps shape: {timestamps.shape}")
logger.info(f"  First timestamp: {timestamps[0]} (type: {type(timestamps[0])})")
logger.info(f"  Last timestamp: {timestamps[-1]}")

# Test 3: Can we create splits?
logger.info("\nTEST 3: Creating walk-forward splits...")
periods_per_window = 365 * 24  # 1 year of hourly data
test_windows = 2

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

logger.info(f"✓ Created {len(splits)} splits")
for i, (train_start, train_end, test_end) in enumerate(splits, 1):
    logger.info(f"  Split {i}:")
    logger.info(f"    Train: {train_start} to {train_end}")
    logger.info(f"    Test end: {test_end}")

# Test 4: Can we use searchsorted?
logger.info("\nTEST 4: Testing searchsorted...")
if splits:
    train_start, train_end, test_end = splits[0]

    logger.info(f"  Searching for train_start: {train_start}")
    train_start_idx = np.searchsorted(timestamps, train_start, side='left')
    logger.info(f"    Found at index: {train_start_idx}")

    logger.info(f"  Searching for train_end: {train_end}")
    train_end_idx = np.searchsorted(timestamps, train_end, side='right')
    logger.info(f"    Found at index: {train_end_idx}")

    logger.info(f"  Searching for test_end: {test_end}")
    test_end_idx = np.searchsorted(timestamps, test_end, side='right')
    logger.info(f"    Found at index: {test_end_idx}")

    logger.info(f"\n  Train period length: {train_end_idx - train_start_idx}")
    logger.info(f"  Test period length: {test_end_idx - train_end_idx}")

    if train_end_idx - train_start_idx < 10:
        logger.error("  ❌ Train period too short!")
    else:
        logger.success("  ✓ Train period OK")

    if test_end_idx - train_end_idx < 10:
        logger.error("  ❌ Test period too short!")
    else:
        logger.success("  ✓ Test period OK")

    # Test slicing
    logger.info("\nTEST 5: Testing array slicing...")
    train_slice = slice(train_start_idx, train_end_idx)
    test_slice = slice(train_end_idx, test_end_idx)

    train_prices = prices[train_slice]
    test_prices = prices[test_slice]

    logger.info(f"  Train slice shape: {train_prices.shape}")
    logger.info(f"  Test slice shape: {test_prices.shape}")

    if len(train_prices) < 10:
        logger.error("  ❌ Train slice too short!")
    else:
        logger.success(f"  ✓ Train slice OK: {len(train_prices)} prices")

    if len(test_prices) < 10:
        logger.error("  ❌ Test slice too short!")
    else:
        logger.success(f"  ✓ Test slice OK: {len(test_prices)} prices")

# Test 6: Check if issue is with insufficient data
logger.info("\nTEST 6: Checking total data availability...")
required_periods = periods_per_window * (test_windows + 1)
logger.info(f"  Required periods: {required_periods}")
logger.info(f"  Available periods: {len(timestamps)}")

if len(timestamps) < required_periods:
    logger.error(f"  ❌ INSUFFICIENT DATA! Need {required_periods}, have {len(timestamps)}")
    logger.error(f"  Missing: {required_periods - len(timestamps)} periods")
    logger.error("\n  SOLUTION: Either:")
    logger.error("    1. Reduce window_days (e.g., --window-days 180)")
    logger.error("    2. Reduce test_windows (e.g., --test-windows 2)")
    logger.error("    3. Use daily timeframe (--timeframe 1d)")
else:
    logger.success("  ✓ Sufficient data available")

logger.info("\n" + "="*80)
logger.info("DIAGNOSIS COMPLETE")
logger.info("="*80)
