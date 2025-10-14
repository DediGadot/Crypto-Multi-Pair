#!/usr/bin/env python3
"""
Diagnose Why Only 1,000 Candles Are Loading

The cached CSV files have 52,560+ candles, but the optimizer is only getting 1,000.
This script traces through the data loading process to find where the data is lost.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from loguru import logger
from crypto_trader.data.fetchers import BinanceDataFetcher

logger.info("="*80)
logger.info("DIAGNOSING DATA LOADING ISSUE")
logger.info("="*80)

# Initialize fetcher
logger.info("\n1. Initializing BinanceDataFetcher...")
fetcher = BinanceDataFetcher()

# Check what's in storage
logger.info("\n2. Checking stored data...")
if fetcher.storage:
    has_btc = fetcher.storage.has_data("BTC/USDT", "1h")
    logger.info(f"   Storage has BTC/USDT 1h data: {has_btc}")

    if has_btc:
        file_path = fetcher.storage._get_file_path("BTC/USDT", "1h")
        logger.info(f"   File path: {file_path}")

        # Load from storage directly
        stored_df = fetcher.storage.load_ohlcv("BTC/USDT", "1h")
        if stored_df is not None:
            logger.success(f"   ✓ Storage has {len(stored_df)} candles")
            logger.info(f"   Date range: {stored_df.index.min()} to {stored_df.index.max()}")
        else:
            logger.error("   ✗ Could not load data from storage")

# Test different limit values
test_limits = [100, 1000, 5000, 10000, 52560]

logger.info("\n3. Testing get_ohlcv() with different limits...")
for limit in test_limits:
    logger.info(f"\n   Testing limit={limit:,}...")

    try:
        df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=limit)

        if df is not None and not df.empty:
            logger.success(f"   ✓ Got {len(df)} candles (requested {limit:,})")

            if len(df) < limit:
                logger.warning(f"   ⚠ Expected {limit:,} but got {len(df)} candles")
                logger.warning(f"   This is the problem!")
        else:
            logger.error(f"   ✗ Got None or empty DataFrame")

    except Exception as e:
        logger.error(f"   ✗ Exception: {e}")

# Check cache stats
logger.info("\n4. Cache statistics...")
stats = fetcher.get_cache_stats()
logger.info(f"   {stats}")

logger.info("\n" + "="*80)
logger.info("DIAGNOSIS COMPLETE")
logger.info("="*80)

# Summary
logger.info("\n📋 SUMMARY:")
logger.info("   Storage file: 52,560+ candles ✓")
logger.info("   Problem: Likely in get_ohlcv() or _fetch_with_smart_caching()")
logger.info("   ")
logger.info("   Possible causes:")
logger.info("   1. Smart caching logic returns only tail(limit) instead of full data")
logger.info("   2. Cache has old 1000-candle data that's being reused")
logger.info("   3. Pagination not triggered when it should be")
