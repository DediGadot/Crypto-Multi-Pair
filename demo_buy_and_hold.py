#!/usr/bin/env python3
"""
Demo script for Buy and Hold Strategy

This script demonstrates how to use the BuyAndHold strategy as a baseline
benchmark for comparing against other active trading strategies.

Usage:
    uv run python demo_buy_and_hold.py
"""

from datetime import datetime, timedelta

from loguru import logger

from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.strategies.library import BuyAndHoldStrategy


def main():
    """Demonstrate BuyAndHold strategy with real BTC/USDT data."""
    logger.info("="*70)
    logger.info("Buy and Hold Strategy Demonstration")
    logger.info("="*70)

    # 1. Create and initialize the strategy
    logger.info("Step 1: Initializing Buy and Hold strategy...")
    strategy = BuyAndHoldStrategy()
    strategy.initialize({})  # No parameters needed
    logger.success(f"Strategy created: {strategy.name}")
    logger.info(f"Parameters: {strategy.get_parameters()}")
    logger.info(f"Required indicators: {strategy.get_required_indicators()}")

    # 2. Fetch real market data
    logger.info("\nStep 2: Fetching BTC/USDT data from Binance...")
    fetcher = BinanceDataFetcher(use_storage=False, use_cache=False)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)

    data = fetcher.get_ohlcv(
        symbol="BTC/USDT",
        timeframe="1d",
        start_date=start_date,
        end_date=end_date,
        limit=100
    )

    if data is None or data.empty:
        logger.error("Failed to fetch data")
        return

    logger.success(f"Fetched {len(data)} days of data")
    logger.info(f"Period: {data.index.min()} to {data.index.max()}")
    logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # 3. Generate signals
    logger.info("\nStep 3: Generating trading signals...")
    data_reset = data.reset_index()
    signals = strategy.generate_signals(data_reset)

    logger.success(f"Generated {len(signals)} signals")
    logger.info(f"Signal types: {signals['signal'].unique()}")
    logger.info(f"Confidence range: {signals['confidence'].min()} - {signals['confidence'].max()}")

    # 4. Show signal details
    logger.info("\nStep 4: Signal Analysis")
    logger.info(f"First signal: {signals.iloc[0]['signal']} (confidence: {signals.iloc[0]['confidence']})")
    logger.info(f"Metadata: {signals.iloc[0]['metadata']}")

    # 5. Calculate hypothetical returns
    logger.info("\nStep 5: Calculating buy-and-hold returns...")
    initial_price = data['close'].iloc[0]
    final_price = data['close'].iloc[-1]
    total_return = ((final_price - initial_price) / initial_price) * 100

    logger.info(f"Entry price (first day): ${initial_price:,.2f}")
    logger.info(f"Exit price (last day): ${final_price:,.2f}")
    logger.success(f"Total return: {total_return:+.2f}%")

    # 6. Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY: Buy and Hold Strategy")
    logger.info("="*70)
    logger.info(f"Strategy Type: Passive Benchmark")
    logger.info(f"Symbol: BTC/USDT")
    logger.info(f"Timeframe: 1 day")
    logger.info(f"Period: {len(data)} days ({data.index.min().date()} to {data.index.max().date()})")
    logger.info(f"Total Return: {total_return:+.2f}%")
    logger.info(f"Signal Count: {len(signals)} (all HOLD)")
    logger.info("")
    logger.info("✓ This strategy serves as a baseline benchmark")
    logger.info("✓ Active strategies should ideally outperform this after costs")
    logger.info("✓ HOLD signals mean: maintain position throughout period")
    logger.info("="*70)


if __name__ == "__main__":
    main()
