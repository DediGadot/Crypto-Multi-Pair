"""
Integration test for backtesting engine with real MA Crossover strategy.

Tests the complete backtesting workflow with realistic data and verifies
that all components work together correctly.
"""

import sys

import numpy as np
import pandas as pd
from loguru import logger

from crypto_trader.backtesting import BacktestEngine
from crypto_trader.core.config import BacktestConfig
from crypto_trader.core.types import Timeframe
from crypto_trader.strategies.library.moving_average_crossover import MovingAverageCrossover


def create_trending_data(trend_up=True, volatility=100):
    """Create synthetic data with clear trend for testing."""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    np.random.seed(42)

    # Create trending price series
    base = 50000
    if trend_up:
        trend = np.linspace(0, 5000, 200)
    else:
        trend = np.linspace(0, -5000, 200)

    noise = np.random.randn(200) * volatility
    prices = base + trend + noise

    # Calculate moving averages
    prices_series = pd.Series(prices)
    sma_10 = prices_series.rolling(window=10).mean()
    sma_20 = prices_series.rolling(window=20).mean()

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(100, 1000, 200),
        'SMA_10': sma_10,
        'SMA_20': sma_20
    })

    return data.dropna()


if __name__ == "__main__":
    print("🔍 Integration Test: Backtesting Engine\n")

    # Create uptrend data (should generate profitable trades)
    print("Creating uptrend market data...")
    data = create_trending_data(trend_up=True, volatility=50)
    print(f"  ✓ Generated {len(data)} candles")
    print(f"  ✓ Price: ${data['close'].iloc[0]:,.0f} -> ${data['close'].iloc[-1]:,.0f}")

    # Initialize strategy with low threshold to ensure signals
    print("\nInitializing MA Crossover strategy...")
    strategy = MovingAverageCrossover(
        name="MA_Cross_Test",
        config={
            "fast_period": 10,
            "slow_period": 20,
            "signal_threshold": 0.0,  # Very low to ensure signals are generated
            "ma_type": "SMA"
        }
    )
    strategy.initialize(strategy.config)
    print(f"  ✓ Strategy: {strategy.name}")

    # Generate signals to check
    print("\nGenerating signals...")
    signals = strategy.generate_signals(data)
    buy_signals = len(signals[signals['signal'] == 'BUY'])
    sell_signals = len(signals[signals['signal'] == 'SELL'])
    print(f"  ✓ BUY signals: {buy_signals}")
    print(f"  ✓ SELL signals: {sell_signals}")

    # Create backtest config
    print("\nConfiguring backtest...")
    config = BacktestConfig(
        initial_capital=10000.0,
        trading_fee_percent=0.001,
        slippage_percent=0.0005,
        max_position_size=0.95
    )
    print(f"  ✓ Initial capital: ${config.initial_capital:,.2f}")

    # Run backtest
    print("\nRunning backtest...")
    engine = BacktestEngine()
    result = engine.run_backtest(
        strategy=strategy,
        data=data,
        config=config,
        symbol="BTCUSDT",
        timeframe=Timeframe.HOUR_1
    )
    print(f"  ✓ Backtest completed")

    # Display results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Strategy: {result.strategy_name}")
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Duration: {result.duration_days} days")
    print(f"\nPerformance:")
    print(f"  Total Return: {result.metrics.total_return:>10.2%}")
    print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio: {result.metrics.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown: {result.metrics.max_drawdown:>10.2%}")
    print(f"  Calmar Ratio: {result.metrics.calmar_ratio:>10.2f}")
    print(f"\nTrading:")
    print(f"  Total Trades: {result.metrics.total_trades:>10}")
    print(f"  Win Rate: {result.metrics.win_rate:>10.2%}")
    print(f"  Profit Factor: {result.metrics.profit_factor:>10.2f}")
    print(f"  Winning Trades: {result.metrics.winning_trades:>10}")
    print(f"  Losing Trades: {result.metrics.losing_trades:>10}")
    print(f"\nCapital:")
    print(f"  Initial: ${result.initial_capital:>12,.2f}")
    print(f"  Final: ${result.metrics.final_capital:>12,.2f}")
    print(f"  P&L: ${result.metrics.final_capital - result.initial_capital:>12,.2f}")
    print(f"\nQuality: {result.metrics.risk_adjusted_quality()}")
    print("="*60)

    # Verify some basic assertions
    print("\nValidation checks:")
    checks_passed = 0
    total_checks = 0

    # Check 1: Result is properly structured
    total_checks += 1
    if result.strategy_name == strategy.name:
        print("  ✓ Strategy name matches")
        checks_passed += 1
    else:
        print(f"  ✗ Strategy name mismatch: {result.strategy_name} != {strategy.name}")

    # Check 2: Final capital is reasonable
    total_checks += 1
    if result.metrics.final_capital > 0:
        print("  ✓ Final capital is positive")
        checks_passed += 1
    else:
        print(f"  ✗ Final capital is not positive: ${result.metrics.final_capital}")

    # Check 3: Equity curve exists
    total_checks += 1
    if len(result.equity_curve) > 0:
        print(f"  ✓ Equity curve has {len(result.equity_curve)} points")
        checks_passed += 1
    else:
        print("  ✗ Equity curve is empty")

    # Check 4: Metrics are reasonable (allow inf/nan if no trades)
    total_checks += 1
    if result.metrics.total_trades == 0:
        # If no trades, inf/nan is acceptable
        print(f"  ✓ Sharpe ratio: {result.metrics.sharpe_ratio:.2f} (no trades executed)")
        checks_passed += 1
    elif not np.isnan(result.metrics.sharpe_ratio) and not np.isinf(result.metrics.sharpe_ratio):
        print(f"  ✓ Sharpe ratio is valid: {result.metrics.sharpe_ratio:.2f}")
        checks_passed += 1
    else:
        print(f"  ✗ Sharpe ratio is invalid: {result.metrics.sharpe_ratio}")

    # Check 5: Summary generation works
    total_checks += 1
    try:
        summary = result.summary()
        if 'strategy' in summary and 'total_return' in summary:
            print("  ✓ Summary generation works")
            checks_passed += 1
        else:
            print("  ✗ Summary missing required keys")
    except Exception as e:
        print(f"  ✗ Summary generation failed: {e}")

    print(f"\n{checks_passed}/{total_checks} validation checks passed")

    if checks_passed == total_checks:
        print("\n✅ INTEGRATION TEST PASSED")
        sys.exit(0)
    else:
        print(f"\n❌ INTEGRATION TEST FAILED ({total_checks - checks_passed} checks failed)")
        sys.exit(1)
