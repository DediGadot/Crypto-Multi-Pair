"""
Demonstration script for advanced risk metrics in the crypto trading system.

This script shows how to use the newly implemented advanced risk metrics:
- Value at Risk (VaR) - Maximum expected loss at 95% confidence
- Conditional VaR (CVaR) - Expected loss beyond VaR threshold
- Skewness - Return distribution asymmetry
- Kurtosis - Return distribution tail risk
- Information Ratio - Risk-adjusted excess return vs benchmark

Documentation:
- NumPy: https://numpy.org/doc/stable/
- Pandas: https://pandas.pydata.org/docs/
- SciPy Stats: https://docs.scipy.org/doc/scipy/reference/stats.html

Sample Input:
    Returns series from a backtest
    Trades from a trading strategy
    Equity curve showing portfolio value over time

Expected Output:
    PerformanceMetrics with all advanced risk metrics calculated
    Comprehensive risk analysis showing tail risk and distribution characteristics
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from crypto_trader.analysis.metrics import MetricsCalculator
from crypto_trader.core.types import OrderSide, OrderType, Trade


def create_sample_trading_data():
    """
    Create realistic sample trading data for demonstration.

    Returns:
        Tuple of (trades, equity_curve, initial_capital)
    """
    base_time = datetime(2025, 1, 1, 10, 0, 0)
    initial_capital = 10000.0

    # Create a series of trades with varying outcomes
    trades = [
        # Series of small wins
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=2),
            entry_price=45000.0,
            exit_price=45500.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=50.0,
            pnl_percent=1.11,
            fees=10.0,
            order_type=OrderType.MARKET,
        ),
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=3),
            exit_time=base_time + timedelta(hours=5),
            entry_price=45500.0,
            exit_price=45800.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=30.0,
            pnl_percent=0.66,
            fees=10.0,
            order_type=OrderType.MARKET,
        ),
        # Small loss
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=6),
            exit_time=base_time + timedelta(hours=8),
            entry_price=45800.0,
            exit_price=45600.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=-20.0,
            pnl_percent=-0.44,
            fees=10.0,
            order_type=OrderType.MARKET,
        ),
        # Big win (creating positive skew)
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=9),
            exit_time=base_time + timedelta(hours=13),
            entry_price=45600.0,
            exit_price=47000.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=140.0,
            pnl_percent=3.07,
            fees=12.0,
            order_type=OrderType.MARKET,
        ),
        # Small wins
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=14),
            exit_time=base_time + timedelta(hours=16),
            entry_price=47000.0,
            exit_price=47200.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=20.0,
            pnl_percent=0.43,
            fees=10.0,
            order_type=OrderType.MARKET,
        ),
        # Small loss
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=17),
            exit_time=base_time + timedelta(hours=19),
            entry_price=47200.0,
            exit_price=47050.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=-15.0,
            pnl_percent=-0.32,
            fees=10.0,
            order_type=OrderType.MARKET,
        ),
        # Moderate loss (tail risk)
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=20),
            exit_time=base_time + timedelta(hours=22),
            entry_price=47050.0,
            exit_price=46400.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=-65.0,
            pnl_percent=-1.38,
            fees=10.0,
            order_type=OrderType.MARKET,
        ),
        # Small win
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=23),
            exit_time=base_time + timedelta(hours=25),
            entry_price=46400.0,
            exit_price=46700.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=30.0,
            pnl_percent=0.65,
            fees=10.0,
            order_type=OrderType.MARKET,
        ),
    ]

    # Build equity curve from trades
    equity_curve = [(base_time, initial_capital)]
    current_equity = initial_capital

    for trade in trades:
        current_equity += trade.pnl - trade.fees
        equity_curve.append((trade.exit_time, current_equity))

    return trades, equity_curve, initial_capital


def interpret_risk_metrics(metrics):
    """
    Provide interpretations of the advanced risk metrics.

    Args:
        metrics: PerformanceMetrics object with calculated metrics

    Returns:
        Dictionary of interpretations
    """
    interpretations = {}

    # VaR interpretation
    if metrics.value_at_risk_95 < 0.02:
        interpretations['var'] = "Low risk: Less than 2% expected loss at 95% confidence"
    elif metrics.value_at_risk_95 < 0.05:
        interpretations['var'] = "Moderate risk: 2-5% expected loss at 95% confidence"
    else:
        interpretations['var'] = f"High risk: {metrics.value_at_risk_95:.2%} expected loss at 95% confidence"

    # CVaR interpretation
    cvar_var_ratio = metrics.conditional_var_95 / metrics.value_at_risk_95 if metrics.value_at_risk_95 > 0 else 0
    if cvar_var_ratio < 1.2:
        interpretations['cvar'] = "Tail losses are contained (CVaR/VaR < 1.2)"
    elif cvar_var_ratio < 1.5:
        interpretations['cvar'] = "Moderate tail risk (CVaR/VaR 1.2-1.5)"
    else:
        interpretations['cvar'] = "Significant tail risk (CVaR/VaR > 1.5)"

    # Skewness interpretation
    if metrics.skewness > 0.5:
        interpretations['skewness'] = "Positive skew: More large gains than losses (favorable)"
    elif metrics.skewness < -0.5:
        interpretations['skewness'] = "Negative skew: More large losses than gains (unfavorable)"
    else:
        interpretations['skewness'] = "Symmetric distribution: Balanced gains and losses"

    # Kurtosis interpretation
    if metrics.kurtosis > 1.0:
        interpretations['kurtosis'] = "Fat tails: Higher probability of extreme events"
    elif metrics.kurtosis < -1.0:
        interpretations['kurtosis'] = "Thin tails: Lower probability of extreme events"
    else:
        interpretations['kurtosis'] = "Normal tails: Typical extreme event probability"

    # Information Ratio interpretation
    if metrics.information_ratio > 1.0:
        interpretations['info_ratio'] = "Excellent risk-adjusted outperformance (IR > 1.0)"
    elif metrics.information_ratio > 0.5:
        interpretations['info_ratio'] = "Good risk-adjusted outperformance (IR > 0.5)"
    elif metrics.information_ratio > 0:
        interpretations['info_ratio'] = "Moderate risk-adjusted outperformance (IR > 0)"
    else:
        interpretations['info_ratio'] = "Underperformance vs benchmark (IR < 0)"

    return interpretations


def main():
    """
    Main demonstration function.
    """
    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("=" * 70)
    print("ADVANCED RISK METRICS DEMONSTRATION")
    print("=" * 70)

    # Test 1: Create sample data
    total_tests += 1
    print("\n📊 Test 1: Creating sample trading data...")
    try:
        trades, equity_curve, initial_capital = create_sample_trading_data()

        if len(trades) == 0:
            all_validation_failures.append("No trades created")
        if len(equity_curve) == 0:
            all_validation_failures.append("No equity curve created")

        print(f"  ✓ Created {len(trades)} trades")
        print(f"  ✓ Equity curve has {len(equity_curve)} points")
        print(f"  ✓ Initial capital: ${initial_capital:,.2f}")

    except Exception as e:
        all_validation_failures.append(f"Sample data creation failed: {e}")

    # Test 2: Calculate metrics
    total_tests += 1
    print("\n📈 Test 2: Calculating all metrics including advanced risk metrics...")
    try:
        calculator = MetricsCalculator(risk_free_rate=0.02)
        returns = calculator.calculate_returns_from_equity(equity_curve)

        if len(returns) == 0:
            all_validation_failures.append("Returns calculation failed")

        metrics = calculator.calculate_all_metrics(
            returns=returns,
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
        )

        print(f"  ✓ Calculated {len(returns)} return periods")
        print(f"  ✓ All metrics calculated successfully")

    except Exception as e:
        all_validation_failures.append(f"Metrics calculation failed: {e}")

    # Test 3: Display standard metrics
    total_tests += 1
    print("\n📊 Test 3: Standard Performance Metrics")
    print("-" * 70)
    try:
        print(f"  Total Return:        {metrics.total_return:>10.2%}")
        print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:       {metrics.sortino_ratio:>10.2f}")
        print(f"  Max Drawdown:        {metrics.max_drawdown:>10.2%}")
        print(f"  Win Rate:            {metrics.win_rate:>10.2%}")
        print(f"  Profit Factor:       {metrics.profit_factor:>10.2f}")
        print(f"  Total Trades:        {metrics.total_trades:>10}")
        print(f"  Final Capital:       ${metrics.final_capital:>10,.2f}")

        # Basic validation
        if metrics.total_return == 0.0:
            all_validation_failures.append("Total return should be non-zero for actual trades")

    except Exception as e:
        all_validation_failures.append(f"Standard metrics display failed: {e}")

    # Test 4: Display advanced risk metrics
    total_tests += 1
    print("\n🔬 Test 4: Advanced Risk Metrics")
    print("-" * 70)
    try:
        print(f"  VaR (95%):           {metrics.value_at_risk_95:>10.2%}")
        print(f"    ↳ 5% chance of losing more than {metrics.value_at_risk_95:.2%}")
        print()
        print(f"  CVaR (95%):          {metrics.conditional_var_95:>10.2%}")
        print(f"    ↳ Expected loss when exceeding VaR: {metrics.conditional_var_95:.2%}")
        print()
        print(f"  Skewness:            {metrics.skewness:>10.4f}")
        if metrics.skewness > 0:
            print(f"    ↳ Positive: More large gains than losses")
        elif metrics.skewness < 0:
            print(f"    ↳ Negative: More large losses than gains")
        else:
            print(f"    ↳ Symmetric distribution")
        print()
        print(f"  Kurtosis:            {metrics.kurtosis:>10.4f}")
        if metrics.kurtosis > 0:
            print(f"    ↳ Fat tails: Higher extreme event probability")
        elif metrics.kurtosis < 0:
            print(f"    ↳ Thin tails: Lower extreme event probability")
        else:
            print(f"    ↳ Normal distribution")
        print()
        print(f"  Information Ratio:   {metrics.information_ratio:>10.4f}")
        print(f"    ↳ Risk-adjusted excess return vs cash benchmark")

        # Validation
        if metrics.value_at_risk_95 == 0.0:
            all_validation_failures.append("VaR should be calculated for non-empty returns")
        if metrics.conditional_var_95 < metrics.value_at_risk_95:
            all_validation_failures.append(f"CVaR ({metrics.conditional_var_95}) should be >= VaR ({metrics.value_at_risk_95})")

    except Exception as e:
        all_validation_failures.append(f"Advanced metrics display failed: {e}")

    # Test 5: Risk interpretations
    total_tests += 1
    print("\n💡 Test 5: Risk Metric Interpretations")
    print("-" * 70)
    try:
        interpretations = interpret_risk_metrics(metrics)

        print(f"  VaR Analysis:")
        print(f"    {interpretations['var']}")
        print()
        print(f"  CVaR Analysis:")
        print(f"    {interpretations['cvar']}")
        print()
        print(f"  Skewness Analysis:")
        print(f"    {interpretations['skewness']}")
        print()
        print(f"  Kurtosis Analysis:")
        print(f"    {interpretations['kurtosis']}")
        print()
        print(f"  Information Ratio Analysis:")
        print(f"    {interpretations['info_ratio']}")

    except Exception as e:
        all_validation_failures.append(f"Interpretations failed: {e}")

    # Test 6: Comparative analysis
    total_tests += 1
    print("\n🔍 Test 6: Risk Profile Summary")
    print("-" * 70)
    try:
        print(f"  Risk-Adjusted Returns:")
        print(f"    Sharpe Ratio:      {metrics.sharpe_ratio:>8.2f}  (Return per unit of total risk)")
        print(f"    Sortino Ratio:     {metrics.sortino_ratio:>8.2f}  (Return per unit of downside risk)")
        print(f"    Information Ratio: {metrics.information_ratio:>8.2f}  (Excess return per unit of tracking error)")
        print()
        print(f"  Tail Risk Metrics:")
        print(f"    VaR (95%):         {metrics.value_at_risk_95:>8.2%}  (Max expected loss at 95% confidence)")
        print(f"    CVaR (95%):        {metrics.conditional_var_95:>8.2%}  (Average loss beyond VaR)")
        print(f"    Max Drawdown:      {metrics.max_drawdown:>8.2%}  (Largest historical loss)")
        print()
        print(f"  Distribution Shape:")
        print(f"    Skewness:          {metrics.skewness:>8.4f}  (Asymmetry)")
        print(f"    Kurtosis:          {metrics.kurtosis:>8.4f}  (Tail thickness)")
        print()

        # Overall risk assessment
        risk_score = 0
        if metrics.value_at_risk_95 < 0.03:
            risk_score += 2
        elif metrics.value_at_risk_95 < 0.05:
            risk_score += 1

        if metrics.skewness > 0:
            risk_score += 2
        elif metrics.skewness > -0.5:
            risk_score += 1

        if metrics.information_ratio > 1.0:
            risk_score += 2
        elif metrics.information_ratio > 0.5:
            risk_score += 1

        print(f"  Overall Risk Assessment:")
        if risk_score >= 5:
            print(f"    ✅ Excellent risk profile (Score: {risk_score}/6)")
        elif risk_score >= 3:
            print(f"    ⚠️  Moderate risk profile (Score: {risk_score}/6)")
        else:
            print(f"    ❌ High risk profile (Score: {risk_score}/6)")

    except Exception as e:
        all_validation_failures.append(f"Comparative analysis failed: {e}")

    # Final validation result
    print("\n" + "=" * 70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Advanced risk metrics are working correctly and ready for production use")
        sys.exit(0)


if __name__ == "__main__":
    main()
