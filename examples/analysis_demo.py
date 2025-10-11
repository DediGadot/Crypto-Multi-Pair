"""
Demonstration of the analysis module functionality.

This script demonstrates how to use the analysis module to calculate metrics,
compare strategies, and generate reports for backtesting results.

Documentation:
- crypto_trader.analysis: Complete analysis module
- Plotly: https://plotly.com/python/

Sample Input:
    python examples/analysis_demo.py

Expected Output:
    - Comprehensive metrics calculations
    - Strategy comparison results
    - Generated HTML report
    - Exported JSON and CSV files
"""

from datetime import datetime, timedelta
from pathlib import Path

from crypto_trader.analysis import MetricsCalculator, ReportGenerator, StrategyComparison
from crypto_trader.core.types import (
    BacktestResult,
    OrderSide,
    OrderType,
    PerformanceMetrics,
    Timeframe,
    Trade,
)


def create_sample_backtest_results() -> list[BacktestResult]:
    """
    Create sample backtest results for demonstration.

    Returns:
        List of BacktestResult objects
    """
    base_time = datetime(2024, 1, 1)
    end_time = datetime(2024, 12, 31)

    # Strategy 1: MA Crossover - High performance
    result1 = BacktestResult(
        strategy_name="MA Crossover",
        symbol="BTCUSDT",
        timeframe=Timeframe.HOUR_4,
        start_date=base_time,
        end_date=end_time,
        initial_capital=10000.0,
        metrics=PerformanceMetrics(
            total_return=0.35,
            sharpe_ratio=2.5,
            sortino_ratio=3.0,
            max_drawdown=0.12,
            calmar_ratio=2.92,
            win_rate=0.68,
            profit_factor=2.1,
            total_trades=45,
            winning_trades=31,
            losing_trades=14,
            avg_win=85.5,
            avg_loss=-42.3,
            expectancy=78.5,
            total_fees=450.0,
            final_capital=13500.0,
        ),
        trades=[
            Trade(
                symbol="BTCUSDT",
                entry_time=base_time + timedelta(days=i * 8),
                exit_time=base_time + timedelta(days=i * 8 + 2),
                entry_price=45000.0 + i * 500,
                exit_price=45000.0 + i * 500 + (300 if i % 3 != 2 else -200),
                side=OrderSide.BUY,
                quantity=0.1,
                pnl=30.0 if i % 3 != 2 else -20.0,
                pnl_percent=0.67 if i % 3 != 2 else -0.44,
                fees=10.0,
                order_type=OrderType.MARKET,
            )
            for i in range(45)
        ],
        equity_curve=[
            (base_time + timedelta(days=i * 30), 10000.0 + i * 350)
            for i in range(13)
        ],
    )

    # Strategy 2: RSI Mean Reversion - Moderate performance
    result2 = BacktestResult(
        strategy_name="RSI Mean Reversion",
        symbol="ETHUSDT",
        timeframe=Timeframe.HOUR_1,
        start_date=base_time,
        end_date=end_time,
        initial_capital=10000.0,
        metrics=PerformanceMetrics(
            total_return=0.22,
            sharpe_ratio=1.8,
            sortino_ratio=2.2,
            max_drawdown=0.18,
            calmar_ratio=1.22,
            win_rate=0.55,
            profit_factor=1.6,
            total_trades=78,
            winning_trades=43,
            losing_trades=35,
            avg_win=62.5,
            avg_loss=-48.3,
            expectancy=28.2,
            total_fees=780.0,
            final_capital=12200.0,
        ),
        trades=[
            Trade(
                symbol="ETHUSDT",
                entry_time=base_time + timedelta(days=i * 4),
                exit_time=base_time + timedelta(days=i * 4 + 1),
                entry_price=3000.0 + i * 20,
                exit_price=3000.0 + i * 20 + (50 if i % 2 == 0 else -30),
                side=OrderSide.BUY,
                quantity=1.0,
                pnl=50.0 if i % 2 == 0 else -30.0,
                pnl_percent=1.67 if i % 2 == 0 else -1.0,
                fees=10.0,
                order_type=OrderType.MARKET,
            )
            for i in range(78)
        ],
        equity_curve=[
            (base_time + timedelta(days=i * 30), 10000.0 + i * 220)
            for i in range(13)
        ],
    )

    # Strategy 3: Bollinger Bands - Poor performance
    result3 = BacktestResult(
        strategy_name="Bollinger Bands",
        symbol="BTCUSDT",
        timeframe=Timeframe.HOUR_4,
        start_date=base_time,
        end_date=end_time,
        initial_capital=10000.0,
        metrics=PerformanceMetrics(
            total_return=-0.05,
            sharpe_ratio=0.5,
            sortino_ratio=0.8,
            max_drawdown=0.35,
            calmar_ratio=-0.14,
            win_rate=0.42,
            profit_factor=0.9,
            total_trades=32,
            winning_trades=13,
            losing_trades=19,
            avg_win=45.2,
            avg_loss=-52.8,
            expectancy=-15.6,
            total_fees=320.0,
            final_capital=9500.0,
        ),
        trades=[
            Trade(
                symbol="BTCUSDT",
                entry_time=base_time + timedelta(days=i * 11),
                exit_time=base_time + timedelta(days=i * 11 + 3),
                entry_price=45000.0 + i * 300,
                exit_price=45000.0 + i * 300 + (100 if i % 5 < 2 else -150),
                side=OrderSide.BUY,
                quantity=0.1,
                pnl=10.0 if i % 5 < 2 else -15.0,
                pnl_percent=0.22 if i % 5 < 2 else -0.33,
                fees=10.0,
                order_type=OrderType.MARKET,
            )
            for i in range(32)
        ],
        equity_curve=[
            (base_time + timedelta(days=i * 30), 10000.0 - i * 50)
            for i in range(13)
        ],
    )

    return [result1, result2, result3]


def demonstrate_metrics_calculator():
    """Demonstrate MetricsCalculator functionality."""
    print("=" * 70)
    print("METRICS CALCULATOR DEMONSTRATION")
    print("=" * 70)

    calculator = MetricsCalculator(risk_free_rate=0.02)

    # Create sample data
    base_time = datetime(2024, 1, 1)
    sample_trades = [
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=i * 4),
            exit_time=base_time + timedelta(hours=i * 4 + 2),
            entry_price=45000.0,
            exit_price=45000.0 + (150 if i % 2 == 0 else -100),
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=15.0 if i % 2 == 0 else -10.0,
            pnl_percent=0.33 if i % 2 == 0 else -0.22,
            fees=5.0,
        )
        for i in range(10)
    ]

    equity_curve = [
        (base_time + timedelta(hours=i * 4), 10000.0 + i * 10) for i in range(11)
    ]

    # Calculate metrics
    returns = calculator.calculate_returns_from_equity(equity_curve)
    metrics = calculator.calculate_all_metrics(
        returns=returns,
        trades=sample_trades,
        equity_curve=equity_curve,
        initial_capital=10000.0,
    )

    print(f"\n📊 Calculated Metrics:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Expectancy: ${metrics.expectancy:.2f} per trade")
    print(f"  Quality Rating: {metrics.risk_adjusted_quality()}")


def demonstrate_strategy_comparison():
    """Demonstrate StrategyComparison functionality."""
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON DEMONSTRATION")
    print("=" * 70)

    results = create_sample_backtest_results()
    comparison = StrategyComparison()

    # Compare all strategies
    print("\n📈 Comparing 3 Strategies:")
    df = comparison.compare_strategies(results)
    print(f"\n{df[['strategy', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'quality']].to_string(index=False)}")

    # Rank by Sharpe ratio
    print("\n🏆 Rankings by Sharpe Ratio:")
    ranked = comparison.rank_strategies(results, metric="sharpe_ratio")
    for idx, row in ranked.iterrows():
        print(f"  #{row['rank']}: {row['strategy']} - Sharpe: {row['sharpe_ratio']:.2f}")

    # Best performers
    print("\n⭐ Best Performers:")
    best_sharpe = comparison.best_performer(results, "sharpe_ratio")
    best_return = comparison.best_performer(results, "total_return")
    best_dd = comparison.best_performer(results, "max_drawdown")

    print(f"  Best Sharpe: {best_sharpe.strategy_name} ({best_sharpe.metrics.sharpe_ratio:.2f})")
    print(f"  Best Return: {best_return.strategy_name} ({best_return.metrics.total_return:.2%})")
    print(f"  Best Drawdown: {best_dd.strategy_name} ({best_dd.metrics.max_drawdown:.2%})")

    # Statistical significance
    print("\n📊 Statistical Significance Test:")
    sig_test = comparison.statistical_significance(results[0], results[1])
    print(f"  Comparing: {sig_test['strategy1']} vs {sig_test['strategy2']}")
    print(f"  P-value: {sig_test['p_value']:.4f}")
    print(f"  Result: {sig_test['message']}")

    # Multi-strategy summary
    print("\n📋 Multi-Strategy Summary:")
    summary = comparison.multi_strategy_summary(results)
    print(f"  Total Strategies: {summary['total_strategies']}")
    print(f"  Profitable Strategies: {summary['profitable_strategies']} ({summary['profitable_rate']:.1%})")
    print(f"  Average Sharpe Ratio: {summary['metrics_summary']['sharpe_ratio']['mean']:.2f}")
    print(f"  Average Max Drawdown: {summary['metrics_summary']['max_drawdown']['mean']:.2%}")

    # Filter strategies
    print("\n🔍 Filtered Strategies (Sharpe > 1.5, Drawdown < 20%):")
    filtered = comparison.filter_strategies(results, min_sharpe=1.5, max_drawdown=0.20)
    for result in filtered:
        print(f"  ✓ {result.strategy_name}: Sharpe {result.metrics.sharpe_ratio:.2f}, DD {result.metrics.max_drawdown:.2%}")


def demonstrate_report_generation():
    """Demonstrate ReportGenerator functionality."""
    print("\n" + "=" * 70)
    print("REPORT GENERATION DEMONSTRATION")
    print("=" * 70)

    results = create_sample_backtest_results()
    reporter = ReportGenerator()

    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate reports for first strategy
    result = results[0]

    print(f"\n📄 Generating reports for: {result.strategy_name}")

    # HTML Report
    html_path = output_dir / "backtest_report.html"
    reporter.generate_html_report(result, str(html_path))
    print(f"  ✓ HTML Report: {html_path}")

    # JSON Export
    json_path = output_dir / "backtest_result.json"
    reporter.export_to_json(result, str(json_path))
    print(f"  ✓ JSON Export: {json_path}")

    # CSV Export
    csv_path = output_dir / "trades.csv"
    reporter.export_to_csv(result, str(csv_path))
    print(f"  ✓ CSV Export: {csv_path}")

    # Comparison Chart
    print("\n📊 Creating comparison chart...")
    comparison_chart = reporter.create_comparison_chart(results, metric="sharpe_ratio")
    comparison_path = output_dir / "comparison_chart.html"
    comparison_chart.write_html(str(comparison_path))
    print(f"  ✓ Comparison Chart: {comparison_path}")

    print(f"\n✅ All reports generated successfully in {output_dir}/")


if __name__ == "__main__":
    """
    Validation function to demonstrate all analysis module functionality.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Analysis Module Integration Demo\n")

    # Test 1: Metrics Calculator
    total_tests += 1
    print("Test 1: MetricsCalculator demonstration")
    try:
        demonstrate_metrics_calculator()
        print("\n  ✓ MetricsCalculator working correctly")
    except Exception as e:
        all_validation_failures.append(f"MetricsCalculator demo exception: {e}")
        print(f"\n  ✗ MetricsCalculator failed: {e}")

    # Test 2: Strategy Comparison
    total_tests += 1
    print("\nTest 2: StrategyComparison demonstration")
    try:
        demonstrate_strategy_comparison()
        print("\n  ✓ StrategyComparison working correctly")
    except Exception as e:
        all_validation_failures.append(f"StrategyComparison demo exception: {e}")
        print(f"\n  ✗ StrategyComparison failed: {e}")

    # Test 3: Report Generation
    total_tests += 1
    print("\nTest 3: ReportGenerator demonstration")
    try:
        demonstrate_report_generation()
        print("\n  ✓ ReportGenerator working correctly")
    except Exception as e:
        all_validation_failures.append(f"ReportGenerator demo exception: {e}")
        print(f"\n  ✗ ReportGenerator failed: {e}")

    # Final validation result
    print("\n" + "=" * 70)
    if all_validation_failures:
        print(
            f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(
            f"✅ VALIDATION PASSED - All {total_tests} demonstrations completed successfully"
        )
        print("All analysis module components are working correctly")
        sys.exit(0)
