#!/usr/bin/env python3
"""Quick test of --max-history feature calculation."""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from crypto_trader.data.fetchers import BinanceDataFetcher
from loguru import logger

def test_max_history_calculation(timeframe="1d", test_windows=3):
    """Test maximum history calculation logic."""
    
    logger.info("=" * 80)
    logger.info(f"TESTING MAX HISTORY CALCULATION")
    logger.info(f"  Timeframe: {timeframe}")
    logger.info(f"  Test windows: {test_windows}")
    logger.info("=" * 80)
    
    fetcher = BinanceDataFetcher()
    
    # Sample assets
    assets = ["BTC/USDT", "ETH/USDT", "DOT/USDT", "SOL/USDT"]
    
    logger.info(f"\nFetching data for {len(assets)} assets...")
    
    min_periods = 10000
    limiting_asset = None
    
    for symbol in assets:
        try:
            data = fetcher.get_ohlcv(symbol, timeframe, limit=10000)
            if data is not None and len(data) > 0:
                if len(data) < min_periods:
                    min_periods = len(data)
                    limiting_asset = symbol
                logger.info(f"  {symbol}: {len(data):,} periods")
        except Exception as e:
            logger.error(f"  {symbol}: {e}")
    
    logger.info(f"\nMinimum periods: {min_periods:,}")
    logger.info(f"Limiting asset: {limiting_asset}")
    
    # Calculate periods per day
    if timeframe == "1h":
        periods_per_day = 24
    elif timeframe == "4h":
        periods_per_day = 6
    elif timeframe == "1d":
        periods_per_day = 1
    else:
        periods_per_day = 24
    
    # Calculate max window days
    usable_periods = int(min_periods * 0.80)
    max_window_days = usable_periods // ((test_windows + 1) * periods_per_day)
    
    logger.success(f"\n✅ CALCULATION RESULTS:")
    logger.info(f"  Available periods: {min_periods:,}")
    logger.info(f"  Usable (80%): {usable_periods:,}")
    logger.info(f"  Periods per day: {periods_per_day}")
    logger.info(f"  Test windows: {test_windows}")
    logger.info(f"  Maximum window_days: {max_window_days:,}")
    
    # Calculate what this means
    total_days_needed = max_window_days * (test_windows + 1)
    logger.info(f"\n📊 WHAT THIS MEANS:")
    logger.info(f"  Each window: {max_window_days} days")
    logger.info(f"  Total span: {total_days_needed} days ({total_days_needed/365:.1f} years)")
    logger.info(f"  Training grows: {max_window_days} → {max_window_days * test_windows} days")
    logger.info(f"  Each test: {max_window_days} days")
    
    logger.info(f"\n💡 TO USE THIS:")
    logger.info(f"  uv run python optimize_portfolio_optimized.py \\")
    logger.info(f"    --max-history \\")
    logger.info(f"    --timeframe {timeframe} \\")
    logger.info(f"    --test-windows {test_windows} \\")
    logger.info(f"    --quick")
    
    return max_window_days

if __name__ == "__main__":
    logger.info("\n🧪 Testing --max-history feature\n")
    
    # Test different timeframes
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Daily timeframe")
    logger.info("=" * 80)
    test_max_history_calculation("1d", 3)
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: 4-hour timeframe")
    logger.info("=" * 80)
    test_max_history_calculation("4h", 3)
    
    logger.success("\n✅ All tests completed!")
