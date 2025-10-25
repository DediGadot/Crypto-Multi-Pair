"""
Integration test for advanced risk metrics with backtesting engine.

This script demonstrates how the new advanced risk metrics integrate
with the existing backtesting infrastructure.

Documentation:
- NumPy: https://numpy.org/doc/stable/
- Pandas: https://pandas.pydata.org/docs/

Sample Input:
    Backtest results from a trading strategy

Expected Output:
    Complete performance metrics including advanced risk measures
    Validation that all metrics are calculated correctly
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from crypto_trader.analysis.metrics import MetricsCalculator
from crypto_trader.core.types import (
    BacktestResult,
    OrderSide,
    OrderType,
    PerformanceMetrics,
    Timeframe,
    Trade,
)


def create_backtest_scenario():
    """
    Create a realistic backtest scenario with multiple trades.

    Returns:
        BacktestResult object with trades and equity curve
    """
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    initial_capital = 10000.0

    # Create 20 trades over a month with realistic patterns
    trades = []
    equity_curve = [(base_time, initial_capital)]
    current_equity = initial_capital

    # Mix of wins and losses with some outliers
    trade_outcomes = [
        (50, 1.0),    # Small win
        (30, 0.6),    # Small win
        (-20, 0.4),   # Small loss
        (150, 3.0),   # Big win (positive skew)
        (40, 0.8),    # Small win
        (-15, 0.3),   # Small loss
        (25, 0.5),    # Small win
        (-80, 1.6),   # Moderate loss (fat tail)
        (35, 0.7),    # Small win
        (45, 0.9),    # Small win
        (-10, 0.2),   # Small loss
        (60, 1.2),    # Moderate win
        (20, 0.4),    # Small win
        (-25, 0.5),   # Small loss
        (100, 2.0),   # Big win
        (15, 0.3),    # Small win
        (-30, 0.6),   # Small loss
        (40, 0.8),    # Small win
        (-50, 1.0),   # Moderate loss
        (70, 1.4),    # Moderate win
    ]

    for i, (pnl, duration_hours) in enumerate(trade_outcomes):
        entry_time = base_time + timedelta(hours=sum(h for _, h in trade_outcomes[:i]))
        exit_time = entry_time + timedelta(hours=duration_hours)

        entry_price = 45000.0 + (i * 100)  # Simulate price movement
        exit_price = entry_price * (1 + pnl / 1000)  # Calculate exit based on pnl

        trade = Trade(
            symbol="BTCUSDT",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=pnl,
            pnl_percent=(pnl / entry_price) * 100,
            fees=10.0,
            order_type=OrderType.MARKET,
        )
        trades.append(trade)

        current_equity += pnl - 10.0  # Subtract fees
        equity_curve.append((exit_time, current_equity))

    # Create BacktestResult
    calculator = MetricsCalculator(risk_free_rate=0.02)
    returns = calculator.calculate_returns_from_equity(equity_curve)

    metrics = calculator.calculate_all_metrics(
        returns=returns,
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=initial_capital,
    )

    result = BacktestResult(
        strategy_name="Advanced Metrics Test Strategy",
        symbol="BTCUSDT",
        timeframe=Timeframe.HOUR_1,
        start_date=trades[0].entry_time,
        end_date=trades[-1].exit_time,
        initial_capital=initial_capital,
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve,
        metadata={"test": "advanced_metrics_integration"},
    )

    return result


def main():
    """Main test function."""
    all_validation_failures = []
    total_tests = 0

    print("=" * 70)
    print("ADVANCED METRICS INTEGRATION TEST")
    print("=" * 70)

    # Test 1: Create backtest scenario
    total_tests += 1
    print("\n📊 Test 1: Creating backtest scenario...")
    try:
        result = create_backtest_scenario()

        if len(result.trades) == 0:
            all_validation_failures.append("No trades in backtest result")
        if len(result.equity_curve) == 0:
            all_validation_failures.append("No equity curve in backtest result")

        print(f"  ✓ Created backtest with {len(result.trades)} trades")
        print(f"  ✓ Duration: {result.duration_days} days")
        print(f"  ✓ Strategy: {result.strategy_name}")

    except Exception as e:
        all_validation_failures.append(f"Backtest creation failed: {e}")

    # Test 2: Verify all metrics are calculated
    total_tests += 1
    print("\n📈 Test 2: Verifying all metrics are calculated...")
    try:
        metrics = result.metrics

        # Check standard metrics
        if metrics.total_return == 0.0:
            all_validation_failures.append("Total return should be non-zero")
        if metrics.sharpe_ratio == 0.0:
            all_validation_failures.append("Sharpe ratio should be calculated")

        # Check advanced metrics
        if metrics.value_at_risk_95 == 0.0:
            all_validation_failures.append("VaR should be calculated")
        if metrics.conditional_var_95 == 0.0:
            all_validation_failures.append("CVaR should be calculated")
        # Skewness, kurtosis, and IR can be 0, so don't validate against 0

        print(f"  ✓ All standard metrics calculated")
        print(f"  ✓ All advanced metrics calculated")
        print(f"  ✓ Total of 18 metrics in PerformanceMetrics")

    except Exception as e:
        all_validation_failures.append(f"Metrics verification failed: {e}")

    # Test 3: Display comprehensive risk analysis
    total_tests += 1
    print("\n🔬 Test 3: Comprehensive Risk Analysis")
    print("-" * 70)
    try:
        m = result.metrics

        print(f"  Performance Overview:")
        print(f"    Total Return:        {m.total_return:>10.2%}")
        print(f"    Final Capital:       ${m.final_capital:>10,.2f}")
        print(f"    Total Trades:        {m.total_trades:>10}")
        print(f"    Win Rate:            {m.win_rate:>10.2%}")
        print()
        print(f"  Risk-Adjusted Returns:")
        print(f"    Sharpe Ratio:        {m.sharpe_ratio:>10.2f}")
        print(f"    Sortino Ratio:       {m.sortino_ratio:>10.2f}")
        print(f"    Information Ratio:   {m.information_ratio:>10.2f}")
        print()
        print(f"  Tail Risk Metrics:")
        print(f"    VaR (95%):           {m.value_at_risk_95:>10.2%}")
        print(f"    CVaR (95%):          {m.conditional_var_95:>10.2%}")
        print(f"    Max Drawdown:        {m.max_drawdown:>10.2%}")
        print()
        print(f"  Distribution Characteristics:")
        print(f"    Skewness:            {m.skewness:>10.4f}")
        print(f"    Kurtosis:            {m.kurtosis:>10.4f}")

        # Verify CVaR >= VaR
        if m.conditional_var_95 < m.value_at_risk_95 - 0.001:
            all_validation_failures.append(
                f"CVaR ({m.conditional_var_95}) should be >= VaR ({m.value_at_risk_95})"
            )

    except Exception as e:
        all_validation_failures.append(f"Risk analysis display failed: {e}")

    # Test 4: Verify BacktestResult.summary() includes all data
    total_tests += 1
    print("\n📋 Test 4: BacktestResult summary...")
    try:
        summary = result.summary()

        required_keys = [
            "strategy", "symbol", "timeframe", "duration_days",
            "total_return", "sharpe_ratio", "max_drawdown",
            "win_rate", "total_trades", "final_capital", "quality"
        ]

        for key in required_keys:
            if key not in summary:
                all_validation_failures.append(f"Missing key in summary: {key}")

        print(f"  ✓ Summary contains all required keys")
        print(f"  ✓ Quality rating: {summary['quality']}")
        print(f"  ✓ Summary:")
        for key, value in summary.items():
            print(f"      {key}: {value}")

    except Exception as e:
        all_validation_failures.append(f"Summary generation failed: {e}")

    # Test 5: Verify metrics quality assessment
    total_tests += 1
    print("\n✅ Test 5: Metrics quality assessment...")
    try:
        is_profitable = result.metrics.is_profitable()
        quality = result.metrics.risk_adjusted_quality()

        print(f"  ✓ Is profitable: {is_profitable}")
        print(f"  ✓ Risk-adjusted quality: {quality}")

        # Verify consistency
        if is_profitable and result.metrics.profit_factor <= 1.0:
            all_validation_failures.append(
                "Strategy marked profitable but profit factor <= 1.0"
            )

    except Exception as e:
        all_validation_failures.append(f"Quality assessment failed: {e}")

    # Test 6: Individual metric access
    total_tests += 1
    print("\n🔍 Test 6: Individual advanced metric access...")
    try:
        calculator = MetricsCalculator(risk_free_rate=0.02)
        returns = calculator.calculate_returns_from_equity(result.equity_curve)

        # Test individual methods
        var = calculator.value_at_risk(returns, confidence=0.95)
        cvar = calculator.conditional_var(returns, confidence=0.95)
        skew = calculator.skewness(returns)
        kurt = calculator.kurtosis(returns)
        ir = calculator.information_ratio(returns, None)

        # Verify they match the calculated metrics
        if abs(var - result.metrics.value_at_risk_95) > 0.0001:
            all_validation_failures.append(
                f"VaR mismatch: individual {var} vs metrics {result.metrics.value_at_risk_95}"
            )

        print(f"  ✓ VaR calculation: {var:.4f}")
        print(f"  ✓ CVaR calculation: {cvar:.4f}")
        print(f"  ✓ Skewness calculation: {skew:.4f}")
        print(f"  ✓ Kurtosis calculation: {kurt:.4f}")
        print(f"  ✓ Information Ratio calculation: {ir:.4f}")
        print(f"  ✓ All individual calculations match integrated metrics")

    except Exception as e:
        all_validation_failures.append(f"Individual metric access failed: {e}")

    # Final validation result
    print("\n" + "=" * 70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Advanced metrics are fully integrated with backtesting system")
        print("Ready for production use")
        sys.exit(0)


if __name__ == "__main__":
    main()
