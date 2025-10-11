"""
Data Layer Usage Example

This example demonstrates how to use the crypto_trader data layer
to fetch, store, and cache OHLCV data from Binance.

Usage:
    uv run python examples/data_layer_demo.py
"""

from datetime import datetime, timedelta

from loguru import logger

from crypto_trader.data import BinanceDataFetcher


def main():
    """Demonstrate data layer functionality."""
    logger.info("=" * 70)
    logger.info("Crypto Trading System - Data Layer Demo")
    logger.info("=" * 70)

    # Initialize the data fetcher
    logger.info("\n1. Initializing Binance Data Fetcher...")
    fetcher = BinanceDataFetcher(
        use_storage=True,
        use_cache=True,
        storage_path="data/ohlcv",
        rate_limit=1200
    )
    logger.success(f"Initialized with {len(fetcher.get_available_symbols())} available symbols")

    # Fetch recent BTC/USDT data
    logger.info("\n2. Fetching 100 candles of BTC/USDT 1h data...")
    btc_df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)
    logger.success(f"Fetched {len(btc_df)} candles")
    logger.info(f"Date range: {btc_df.index.min()} to {btc_df.index.max()}")
    logger.info(f"Latest close: ${btc_df['close'].iloc[-1]:,.2f}")
    logger.info(f"Price range: ${btc_df['low'].min():,.2f} - ${btc_df['high'].max():,.2f}")
    logger.info(f"Total volume: {btc_df['volume'].sum():,.2f}")

    # Demonstrate caching
    logger.info("\n3. Testing cache (fetching same data again)...")
    import time
    start = time.time()
    btc_df2 = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)
    elapsed = time.time() - start
    logger.success(f"Cached fetch took {elapsed:.4f}s (should be < 0.001s)")

    # Show cache statistics
    cache_stats = fetcher.get_cache_stats()
    logger.info(f"Cache stats: {cache_stats}")

    # Update with latest data
    logger.info("\n4. Updating with latest data...")
    success = fetcher.update_data("BTC/USDT", "1h")
    if success:
        logger.success("Update completed successfully")
        updated_df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)
        logger.info(f"Now have {len(updated_df)} candles")
        logger.info(f"Latest timestamp: {updated_df.index.max()}")
    else:
        logger.warning("Update failed or no new data available")

    # Batch fetch multiple symbols
    logger.info("\n5. Batch fetching multiple symbols...")
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    results = fetcher.fetch_batch(symbols, "1h", limit=50)
    logger.success(f"Fetched {len(results)} symbols")
    for symbol, df in results.items():
        if not df.empty:
            logger.info(f"  {symbol}: {len(df)} candles, latest close: ${df['close'].iloc[-1]:,.2f}")

    # Fetch with date range
    logger.info("\n6. Fetching specific date range...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    eth_df = fetcher.get_ohlcv("ETH/USDT", "1h", start_date=start_date, end_date=end_date)
    logger.success(f"Fetched {len(eth_df)} candles from {start_date.date()} to {end_date.date()}")
    logger.info(f"ETH price change: ${eth_df['close'].iloc[0]:,.2f} → ${eth_df['close'].iloc[-1]:,.2f}")

    # Display summary statistics
    logger.info("\n7. Summary Statistics:")
    logger.info("-" * 70)

    for symbol, df in results.items():
        if not df.empty:
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            volatility = ((df['high'] - df['low']) / df['close']).mean() * 100

            logger.info(f"\n{symbol}:")
            logger.info(f"  Current Price: ${df['close'].iloc[-1]:,.2f}")
            logger.info(f"  Price Change: {price_change:+.2f}%")
            logger.info(f"  Avg Volatility: {volatility:.2f}%")
            logger.info(f"  Volume: {df['volume'].sum():,.2f}")

    logger.info("\n" + "=" * 70)
    logger.success("Data Layer Demo Complete!")
    logger.info("Data is stored in: data/ohlcv/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
