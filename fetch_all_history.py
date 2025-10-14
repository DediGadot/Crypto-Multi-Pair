#!/usr/bin/env python3
"""
Fetch All Available Historical Data from Binance

This script downloads the complete available history for all assets
and timeframes used in portfolio optimization. Run this ONCE before
your first optimization to ensure sufficient data.

Usage:
    python fetch_all_history.py
    python fetch_all_history.py --symbols BTC/USDT ETH/USDT
    python fetch_all_history.py --timeframes 1h 1d

Why this is needed:
- Binance API returns max 1,000 candles per request by default
- Portfolio optimization with 365-day windows needs 52,560+ hourly candles
- This script uses pagination to fetch all available history
- Data is cached to data/ohlcv/*.csv for fast reuse

Expected runtime: 10-15 minutes for all symbols and timeframes
"""

import sys
from pathlib import Path
from typing import List

script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import typer
from loguru import logger
from tqdm import tqdm
from crypto_trader.data.fetchers import BinanceDataFetcher

app = typer.Typer(help="Download all available historical data from Binance")


def fetch_all_for_symbol(
    fetcher: BinanceDataFetcher,
    symbol: str,
    timeframe: str
) -> bool:
    """
    Fetch all available history for one symbol/timeframe.

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"📥 Fetching {symbol} {timeframe}...")

        # Use fetch_all=True to download complete history
        df = fetcher.get_ohlcv(symbol, timeframe, fetch_all=True)

        if df is None or df.empty:
            logger.error(f"  ✗ No data returned")
            return False

        # Calculate timespan
        date_range_days = (df.index.max() - df.index.min()).days

        logger.success(f"  ✓ {symbol} {timeframe}: {len(df):,} candles")
        logger.info(f"    Range: {df.index.min()} to {df.index.max()} ({date_range_days:.0f} days)")

        # Sanity checks
        expected_candles = {
            '1h': date_range_days * 24,
            '4h': date_range_days * 6,
            '1d': date_range_days
        }

        if timeframe in expected_candles:
            expected = expected_candles[timeframe]
            coverage = (len(df) / expected) * 100 if expected > 0 else 0
            logger.info(f"    Coverage: {coverage:.1f}% of expected {expected:,} candles")

            if coverage < 80:
                logger.warning(f"    ⚠ Low coverage, possible data gaps")

        return True

    except Exception as e:
        logger.error(f"  ✗ {symbol} {timeframe}: {e}")
        return False


@app.command()
def fetch(
    symbols: List[str] = typer.Option(
        None,
        "--symbols", "-s",
        help="Symbols to fetch (defaults to portfolio optimizer universe)"
    ),
    timeframes: List[str] = typer.Option(
        None,
        "--timeframes", "-t",
        help="Timeframes to fetch (defaults to 1h, 4h, 1d)"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force re-fetch even if data exists"
    )
):
    """
    Download all available historical data from Binance.

    This ensures your optimization runs have sufficient data.
    """
    logger.info("="*80)
    logger.info("📥 FETCH ALL HISTORICAL DATA FROM BINANCE")
    logger.info("="*80)

    # Default symbols (same as portfolio optimizer)
    if not symbols:
        symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
            "ADA/USDT", "XRP/USDT", "MATIC/USDT", "DOT/USDT",
        ]
        logger.info(f"Using default symbols: {len(symbols)} assets")
    else:
        logger.info(f"Using custom symbols: {symbols}")

    # Default timeframes
    if not timeframes:
        timeframes = ["1h", "4h", "1d"]
        logger.info(f"Using default timeframes: {timeframes}")
    else:
        logger.info(f"Using custom timeframes: {timeframes}")

    total_combinations = len(symbols) * len(timeframes)
    logger.info(f"\nTotal data fetches: {total_combinations}")
    logger.info(f"Estimated time: {total_combinations * 0.5:.0f}-{total_combinations:.0f} minutes")
    logger.info("="*80)

    # Initialize fetcher
    logger.info("\nInitializing Binance data fetcher...")
    fetcher = BinanceDataFetcher(
        use_storage=True,
        use_cache=True,
        storage_path="data/ohlcv"
    )

    # Check existing data
    if not force:
        logger.info("\n📊 Checking existing data...")
        existing_count = 0
        for symbol in symbols:
            for timeframe in timeframes:
                if fetcher.storage and fetcher.storage.has_data(symbol, timeframe):
                    existing_df = fetcher.storage.load_ohlcv(symbol, timeframe)
                    if existing_df is not None and len(existing_df) > 1000:
                        logger.info(f"  ✓ {symbol} {timeframe}: {len(existing_df):,} candles (cached)")
                        existing_count += 1

        if existing_count > 0:
            logger.info(f"\n  Found {existing_count}/{total_combinations} already cached")
            logger.info("  Use --force to re-fetch all data")

    # Fetch all data
    logger.info("\n🚀 Starting data download...")
    logger.info("="*80)

    success_count = 0
    failed_count = 0
    skipped_count = 0

    total_candles_fetched = 0

    with tqdm(total=total_combinations, desc="Overall Progress", unit="fetch") as pbar:
        for symbol in symbols:
            for timeframe in timeframes:
                # Check if we should skip
                skip = False
                if not force and fetcher.storage and fetcher.storage.has_data(symbol, timeframe):
                    existing_df = fetcher.storage.load_ohlcv(symbol, timeframe)
                    if existing_df is not None and len(existing_df) > 10000:
                        # Already have substantial data
                        skip = True
                        skipped_count += 1
                        total_candles_fetched += len(existing_df)

                if skip:
                    pbar.update(1)
                    continue

                # Fetch
                success = fetch_all_for_symbol(fetcher, symbol, timeframe)

                if success:
                    success_count += 1
                    # Get candle count
                    if fetcher.storage:
                        df = fetcher.storage.load_ohlcv(symbol, timeframe)
                        if df is not None:
                            total_candles_fetched += len(df)
                else:
                    failed_count += 1

                pbar.update(1)

    # Summary
    logger.info("\n" + "="*80)
    logger.success("📊 FETCH COMPLETE")
    logger.info("="*80)

    logger.info(f"\n  Success: {success_count}/{total_combinations}")
    logger.info(f"  Failed:  {failed_count}/{total_combinations}")
    if skipped_count > 0:
        logger.info(f"  Skipped: {skipped_count}/{total_combinations} (already cached)")
    logger.info(f"  Total candles: {total_candles_fetched:,}")

    if failed_count > 0:
        logger.warning(f"\n⚠ {failed_count} fetches failed. Check logs above for details.")
        logger.warning("  Possible causes: network issues, rate limiting, invalid symbols")

    logger.success("\n✅ Historical data download complete!")
    logger.info("  Data saved to: data/ohlcv/")
    logger.info("  You can now run portfolio optimization with full historical data")

    logger.info("\n🚀 Next step:")
    logger.info("  python optimize_portfolio_optimized.py --timeframe 1h --window-days 365 --test-windows 5 --quick")


if __name__ == "__main__":
    app()
