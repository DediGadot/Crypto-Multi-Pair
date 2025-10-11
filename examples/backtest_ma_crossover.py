"""
Example: Backtest MA Crossover Strategy

This example demonstrates how to use the backtesting engine to test a
Moving Average Crossover strategy on historical data.

It shows:
1. How to prepare data with indicators
2. How to configure and initialize a strategy
3. How to run a backtest
4. How to analyze results

Run with: uv run python examples/backtest_ma_crossover.py
"""

import numpy as np
import pandas as pd
from loguru import logger

from crypto_trader.backtesting import BacktestEngine
from crypto_trader.core.config import BacktestConfig
from crypto_trader.core.types import Timeframe
from crypto_trader.strategies.library.moving_average_crossover import MovingAverageCrossover


def create_sample_data_with_crossovers():
    """
    Create synthetic OHLCV data with guaranteed crossovers.

    Returns realistic price data with clear MA crossover patterns.
    """
    # Create 500 hourly candles
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')

    # Create price pattern with multiple trends
    prices = []
    base_price = 45000

    # Phase 1: Downtrend (0-100)
    phase1 = np.linspace(0, -3000, 100) + np.random.randn(100) * 100
    prices.extend(base_price + phase1)

    # Phase 2: Sharp uptrend (100-250) - This will create bullish crossover
    phase2 = np.linspace(0, 8000, 150) + np.random.randn(150) * 150
    prices.extend(base_price - 3000 + phase2)

    # Phase 3: Sideways (250-350)
    phase3 = np.random.randn(100) * 200
    prices.extend(base_price + 5000 + phase3)

    # Phase 4: Downtrend (350-500) - This will create bearish crossover
    phase4 = np.linspace(0, -5000, 150) + np.random.randn(150) * 150
    prices.extend(base_price + 5000 + phase4)

    prices = np.array(prices)

    # Calculate moving averages
    prices_series = pd.Series(prices)
    sma_10 = prices_series.rolling(window=10).mean()
    sma_20 = prices_series.rolling(window=20).mean()

    # Create OHLCV dataframe
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(100, 1000, len(dates)),
        'SMA_10': sma_10,
        'SMA_20': sma_20
    })

    # Remove NaN rows from MA calculation
    data = data.dropna()

    return data


def main():
    """Run the backtest example."""
    print("="*70)
    print(" MA Crossover Strategy Backtest Example")
    print("="*70)
    print()

    # Step 1: Create sample data
    logger.info("Step 1: Creating sample market data")
    data = create_sample_data_with_crossovers()
    logger.info(f"Generated {len(data)} candles")
    logger.info(f"Date range: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
    logger.info(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    print()

    # Step 2: Initialize strategy
    logger.info("Step 2: Initializing MA Crossover strategy")
    strategy = MovingAverageCrossover(
        name="MA_10_20_Cross",
        config={
            "fast_period": 10,
            "slow_period": 20,
            "signal_threshold": 0.5,
            "ma_type": "SMA"
        }
    )
    strategy.initialize(strategy.config)
    logger.info(f"Strategy: {strategy.name}")
    logger.info(f"Parameters: {strategy.get_parameters()}")
    print()

    # Step 3: Preview signals
    logger.info("Step 3: Generating trading signals")
    signals = strategy.generate_signals(data)
    buy_count = len(signals[signals['signal'] == 'BUY'])
    sell_count = len(signals[signals['signal'] == 'SELL'])
    hold_count = len(signals[signals['signal'] == 'HOLD'])
    logger.info(f"Signals: {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD")
    print()

    # Step 4: Configure backtest
    logger.info("Step 4: Configuring backtest parameters")
    config = BacktestConfig(
        initial_capital=10000.0,
        trading_fee_percent=0.001,  # 0.1% fee
        slippage_percent=0.0005,     # 0.05% slippage
        max_position_size=0.95,       # Use 95% of capital
        compound_returns=True
    )
    logger.info(f"Initial capital: ${config.initial_capital:,.2f}")
    logger.info(f"Trading fee: {config.trading_fee_percent:.3%}")
    logger.info(f"Slippage: {config.slippage_percent:.4%}")
    print()

    # Step 5: Run backtest
    logger.info("Step 5: Running backtest")
    engine = BacktestEngine()
    result = engine.run_backtest(
        strategy=strategy,
        data=data,
        config=config,
        symbol="BTCUSDT",
        timeframe=Timeframe.HOUR_1
    )
    logger.info("Backtest completed!")
    print()

    # Step 6: Display results
    print("="*70)
    print(" BACKTEST RESULTS")
    print("="*70)
    print()
    print(f"Strategy: {result.strategy_name}")
    print(f"Symbol: {result.symbol}")
    print(f"Timeframe: {result.timeframe.value}")
    print(f"Period: {result.start_date.date()} to {result.end_date.date()} ({result.duration_days} days)")
    print()

    print("PERFORMANCE METRICS:")
    print("-" * 70)
    m = result.metrics
    print(f"  Total Return:        {m.total_return:>10.2%}")
    print(f"  Sharpe Ratio:        {m.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:       {m.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:        {m.max_drawdown:>10.2%}")
    print(f"  Calmar Ratio:        {m.calmar_ratio:>10.2f}")
    print(f"  Recovery Factor:     {m.recovery_factor:>10.2f}")
    print()

    print("TRADING STATISTICS:")
    print("-" * 70)
    print(f"  Total Trades:        {m.total_trades:>10}")
    print(f"  Winning Trades:      {m.winning_trades:>10}")
    print(f"  Losing Trades:       {m.losing_trades:>10}")
    print(f"  Win Rate:            {m.win_rate:>10.2%}")
    print(f"  Profit Factor:       {m.profit_factor:>10.2f}")
    print(f"  Expectancy:          ${m.expectancy:>9.2f}")
    print()

    if m.total_trades > 0:
        print(f"  Average Win:         ${m.avg_win:>9.2f}")
        print(f"  Average Loss:        ${m.avg_loss:>9.2f}")
        print(f"  Max Consecutive Wins:  {m.max_consecutive_wins:>8}")
        print(f"  Max Consecutive Losses:{m.max_consecutive_losses:>8}")
        print(f"  Avg Trade Duration:  {m.avg_trade_duration:>9.1f} min")
        print()

    print("CAPITAL:")
    print("-" * 70)
    print(f"  Initial Capital:     ${result.initial_capital:>12,.2f}")
    print(f"  Final Capital:       ${m.final_capital:>12,.2f}")
    print(f"  Total P&L:           ${m.final_capital - result.initial_capital:>12,.2f}")
    print(f"  Total Fees Paid:     ${m.total_fees:>12,.2f}")
    print()

    print(f"Quality Rating: {m.risk_adjusted_quality()}")
    print(f"Is Profitable: {'Yes' if m.is_profitable() else 'No'}")
    print()

    # Display trade history
    if len(result.trades) > 0:
        print("TRADE HISTORY (First 5):")
        print("-" * 70)
        for i, trade in enumerate(result.trades[:5]):
            print(f"  Trade #{i+1}:")
            print(f"    Entry: {trade.entry_time} @ ${trade.entry_price:,.2f}")
            print(f"    Exit:  {trade.exit_time} @ ${trade.exit_price:,.2f}")
            print(f"    Side:  {trade.side.value.upper()}")
            print(f"    P&L:   ${trade.pnl:>8.2f} ({trade.pnl_percent:>6.2f}%)")
            print(f"    Duration: {trade.duration_minutes:.0f} minutes")
            print()
    else:
        print("No trades executed during the backtest period.")
        print()

    print("="*70)
    print()

    # Interpretation
    print("INTERPRETATION:")
    print("-" * 70)
    if m.total_trades == 0:
        print("No trades were executed. This could mean:")
        print("  • Signal threshold is too high")
        print("  • No clear crossovers in the data")
        print("  • Try adjusting strategy parameters")
    elif m.is_profitable():
        print("✓ Strategy is profitable!")
        print(f"  Risk-adjusted quality: {m.risk_adjusted_quality()}")
        if m.sharpe_ratio >= 2.0:
            print("  Excellent risk-adjusted returns (Sharpe >= 2.0)")
        elif m.sharpe_ratio >= 1.5:
            print("  Good risk-adjusted returns (Sharpe >= 1.5)")
        elif m.sharpe_ratio >= 1.0:
            print("  Fair risk-adjusted returns (Sharpe >= 1.0)")
    else:
        print("✗ Strategy is not profitable in this period.")
        print("  Consider:")
        print("  • Adjusting MA periods")
        print("  • Testing different timeframes")
        print("  • Adding filters or confirmation signals")

    print("="*70)


if __name__ == "__main__":
    main()
