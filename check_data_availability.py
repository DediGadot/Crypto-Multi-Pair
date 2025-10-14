#!/usr/bin/env python3
"""Quick data availability checker."""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from crypto_trader.data.fetchers import BinanceDataFetcher
from loguru import logger
import numpy as np

def check_data(timeframe="1h", window_days=365, test_windows=5):
    """Check if data is sufficient for requested parameters."""

    logger.info(f"\n{'='*80}")
    logger.info(f"DATA AVAILABILITY CHECK")
    logger.info(f"{'='*80}")

    fetcher = BinanceDataFetcher()

    symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "ADA/USDT", "XRP/USDT", "MATIC/USDT", "DOT/USDT",
    ]

    # Calculate requirements
    if timeframe == "1h":
        periods_per_day = 24
    elif timeframe == "4h":
        periods_per_day = 6
    elif timeframe == "1d":
        periods_per_day = 1
    else:
        periods_per_day = 24

    periods_per_window = window_days * periods_per_day
    total_windows = test_windows + 1
    required_periods = periods_per_window * total_windows

    logger.info(f"\nRequested Parameters:")
    logger.info(f"  timeframe: {timeframe}")
    logger.info(f"  window_days: {window_days}")
    logger.info(f"  test_windows: {test_windows}")
    logger.info(f"\nCalculated Requirements:")
    logger.info(f"  periods_per_window: {periods_per_window:,}")
    logger.info(f"  total_windows: {total_windows}")
    logger.info(f"  required_periods: {required_periods:,}")

    # Fetch data
    logger.info(f"\nFetching data for {len(symbols)} assets...")
    all_data = {}

    for symbol in symbols:
        try:
            data = fetcher.get_ohlcv(symbol, timeframe, limit=required_periods + 1000)
            if data is not None and len(data) > 0:
                all_data[symbol] = data
                status = "✓" if len(data) >= required_periods else "⚠"
                logger.info(f"  {status} {symbol:15s}: {len(data):6,} periods")
        except Exception as e:
            logger.error(f"  ✗ {symbol:15s}: {e}")

    # Calculate common timestamps
    logger.info(f"\nCalculating common timestamps...")
    common_timestamps = None
    for symbol, data in all_data.items():
        if common_timestamps is None:
            common_timestamps = set(data.index)
        else:
            common_timestamps = common_timestamps.intersection(set(data.index))

    common_count = len(common_timestamps)

    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"  Assets with data: {len(all_data)}/{len(symbols)}")
    logger.info(f"  Common timestamps: {common_count:,}")
    logger.info(f"  Required periods: {required_periods:,}")
    logger.info(f"  Difference: {common_count - required_periods:,}")

    if common_count >= required_periods:
        margin_pct = (common_count - required_periods) / required_periods * 100
        logger.success(f"\n✅ SUFFICIENT DATA (margin: {margin_pct:.1f}%)")
        logger.info(f"\nYou can use:")
        logger.info(f"  uv run python optimize_portfolio_optimized.py \\")
        logger.info(f"    --timeframe {timeframe} \\")
        logger.info(f"    --window-days {window_days} \\")
        logger.info(f"    --test-windows {test_windows} \\")
        logger.info(f"    --quick")
        return True
    else:
        shortfall = required_periods - common_count
        logger.error(f"\n❌ INSUFFICIENT DATA (shortfall: {shortfall:,} periods)")

        # Calculate what would work
        max_window_days = int((common_count * 0.95) / (total_windows * periods_per_day))

        logger.info(f"\n💡 SOLUTIONS:")
        logger.info(f"\n1. Reduce window size:")
        logger.info(f"  uv run python optimize_portfolio_optimized.py \\")
        logger.info(f"    --timeframe {timeframe} \\")
        logger.info(f"    --window-days {max_window_days} \\")
        logger.info(f"    --test-windows {test_windows} \\")
        logger.info(f"    --quick")

        if timeframe == "1h":
            logger.info(f"\n2. Use 4-hour timeframe (recommended):")
            logger.info(f"  uv run python optimize_portfolio_optimized.py \\")
            logger.info(f"    --timeframe 4h \\")
            logger.info(f"    --window-days {window_days} \\")
            logger.info(f"    --test-windows {test_windows} \\")
            logger.info(f"    --quick")

        return False

if __name__ == "__main__":
    import sys

    # Parse simple args
    timeframe = "1h"
    window_days = 365
    test_windows = 5

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--timeframe" and i + 1 < len(sys.argv) - 1:
            timeframe = sys.argv[i + 2]
        elif arg == "--window-days" and i + 1 < len(sys.argv) - 1:
            window_days = int(sys.argv[i + 2])
        elif arg == "--test-windows" and i + 1 < len(sys.argv) - 1:
            test_windows = int(sys.argv[i + 2])

    check_data(timeframe, window_days, test_windows)
