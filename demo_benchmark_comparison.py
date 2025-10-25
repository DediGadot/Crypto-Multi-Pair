"""
Demo: Benchmark Comparison Module

Demonstrates how to use the BenchmarkComparator to compare trading strategies
against buy-and-hold benchmarks.

Usage:
    uv run python demo_benchmark_comparison.py
"""

from src.crypto_trader.analysis.benchmark_comparator import (
    BenchmarkComparator,
    BenchmarkComparison
)
from src.crypto_trader.analysis.multipair_aggregator import (
    MultiPairWindowedMetrics,
    CrossPairCorrelation
)
from src.crypto_trader.analysis.aggregator import WindowedMetrics


def create_example_metrics(
    strategy_name: str,
    mean_return: float,
    mean_sharpe: float
) -> MultiPairWindowedMetrics:
    """Create example MultiPairWindowedMetrics for demonstration."""

    # Create mock pair metrics
    btc_metrics = WindowedMetrics(
        strategy_name=strategy_name,
        horizon_name="30d",
        dataset_type="test",
        num_windows=10,
        mean_return=mean_return,
        median_return=mean_return,
        std_return=2.5,
        p25_return=mean_return - 1.5,
        p75_return=mean_return + 1.5,
        weighted_return=mean_return,
        mean_sharpe=mean_sharpe,
        median_sharpe=mean_sharpe,
        std_sharpe=0.4,
        p25_sharpe=mean_sharpe - 0.3,
        p75_sharpe=mean_sharpe + 0.3,
        weighted_sharpe=mean_sharpe,
        mean_drawdown=8.0,
        median_drawdown=7.5,
        std_drawdown=2.0,
        p25_drawdown=6.0,
        p75_drawdown=10.0,
        weighted_drawdown=8.0,
        mean_win_rate=0.58,
        median_win_rate=0.60,
        std_win_rate=0.08,
        p25_win_rate=0.52,
        p75_win_rate=0.64,
        weighted_win_rate=0.58,
        mean_trades=25.0,
        total_trades=250,
        consistency_score=0.75
    )

    eth_metrics = WindowedMetrics(
        strategy_name=strategy_name,
        horizon_name="30d",
        dataset_type="test",
        num_windows=10,
        mean_return=mean_return * 0.92,
        median_return=mean_return * 0.92,
        std_return=2.8,
        p25_return=mean_return * 0.92 - 1.5,
        p75_return=mean_return * 0.92 + 1.5,
        weighted_return=mean_return * 0.92,
        mean_sharpe=mean_sharpe * 0.95,
        median_sharpe=mean_sharpe * 0.95,
        std_sharpe=0.45,
        p25_sharpe=mean_sharpe * 0.95 - 0.3,
        p75_sharpe=mean_sharpe * 0.95 + 0.3,
        weighted_sharpe=mean_sharpe * 0.95,
        mean_drawdown=9.0,
        median_drawdown=8.5,
        std_drawdown=2.2,
        p25_drawdown=7.0,
        p75_drawdown=11.0,
        weighted_drawdown=9.0,
        mean_win_rate=0.56,
        median_win_rate=0.57,
        std_win_rate=0.09,
        p25_win_rate=0.50,
        p75_win_rate=0.62,
        weighted_win_rate=0.56,
        mean_trades=22.0,
        total_trades=220,
        consistency_score=0.72
    )

    # Create mock correlation
    correlation = CrossPairCorrelation(
        pairs=['BTC/USDT', 'ETH/USDT'],
        correlation_matrix={('BTC/USDT', 'ETH/USDT'): 0.68},
        mean_correlation=0.68,
        max_correlation=0.68,
        min_correlation=0.68
    )

    return MultiPairWindowedMetrics(
        strategy_name=strategy_name,
        horizon_name="30d",
        dataset_type="test",
        pairs=['BTC/USDT', 'ETH/USDT'],
        num_windows=10,
        pair_metrics={'BTC/USDT': btc_metrics, 'ETH/USDT': eth_metrics},
        portfolio_mean_return=mean_return,
        portfolio_median_return=mean_return,
        portfolio_std_return=2.6,
        portfolio_sharpe=mean_sharpe,
        portfolio_drawdown=8.5,
        correlation=correlation,
        diversification_ratio=1.12,
        risk_contribution={'BTC/USDT': 53.0, 'ETH/USDT': 47.0},
        effective_num_assets=1.85,
        correlation_matrix_df=None
    )


def main():
    """Demonstrate benchmark comparison functionality."""

    print("=" * 70)
    print("BENCHMARK COMPARISON DEMO")
    print("=" * 70)

    # Initialize comparator
    comparator = BenchmarkComparator()

    # Scenario 1: Outperforming Strategy
    print("\n" + "=" * 70)
    print("Scenario 1: Outperforming Strategy")
    print("=" * 70)

    strategy_metrics = create_example_metrics(
        "Copula Pairs Trading",
        mean_return=18.5,
        mean_sharpe=2.2
    )

    benchmark_metrics = create_example_metrics(
        "Buy and Hold",
        mean_return=12.0,
        mean_sharpe=1.5
    )

    # Per-window returns (strategy beats benchmark in 7/10 windows)
    strategy_window_returns = [15.0, 22.0, 18.0, 25.0, 14.0, 20.0, 17.0, 19.0, 11.0, 24.0]
    benchmark_window_returns = [10.0, 14.0, 11.0, 18.0, 12.0, 13.0, 12.0, 11.0, 13.0, 16.0]

    comparison = comparator.compare_to_benchmark(
        strategy_metrics,
        benchmark_metrics,
        strategy_window_returns,
        benchmark_window_returns
    )

    print("\n" + comparison.summary_string())

    print("\n📈 Interpretation:")
    if comparison.alpha > 0:
        print(f"   ✓ Strategy generates positive alpha of {comparison.alpha:.2f}%")
        print(f"   ✓ This represents {comparison.relative_alpha:.1f}% improvement over benchmark")
    if comparison.sharpe_alpha > 0:
        print(f"   ✓ Risk-adjusted returns (Sharpe) are {comparison.sharpe_alpha:.2f} higher")
    if comparison.win_rate_vs_benchmark > 50:
        print(f"   ✓ Strategy beats benchmark in {comparison.win_rate_vs_benchmark:.1f}% of windows")

    # Scenario 2: Underperforming Strategy
    print("\n" + "=" * 70)
    print("Scenario 2: Underperforming Strategy")
    print("=" * 70)

    weak_strategy_metrics = create_example_metrics(
        "Simple Moving Average",
        mean_return=9.5,
        mean_sharpe=1.1
    )

    # Per-window returns (strategy beats benchmark in only 3/10 windows)
    weak_strategy_returns = [8.0, 11.0, 7.0, 13.0, 9.0, 8.5, 10.5, 9.5, 8.0, 10.0]
    benchmark_returns_2 = [10.0, 14.0, 11.0, 12.0, 12.0, 13.0, 12.0, 11.0, 13.0, 16.0]

    comparison_weak = comparator.compare_to_benchmark(
        weak_strategy_metrics,
        benchmark_metrics,
        weak_strategy_returns,
        benchmark_returns_2
    )

    print("\n" + comparison_weak.summary_string())

    print("\n📉 Interpretation:")
    if comparison_weak.alpha < 0:
        print(f"   ✗ Strategy generates negative alpha of {comparison_weak.alpha:.2f}%")
        print(f"   ✗ This represents {abs(comparison_weak.relative_alpha):.1f}% underperformance")
    if comparison_weak.sharpe_alpha < 0:
        print(f"   ✗ Risk-adjusted returns (Sharpe) are {abs(comparison_weak.sharpe_alpha):.2f} lower")
    if comparison_weak.win_rate_vs_benchmark < 50:
        print(f"   ✗ Strategy only beats benchmark in {comparison_weak.win_rate_vs_benchmark:.1f}% of windows")
    print(f"   → Recommendation: Consider using simple buy-and-hold instead")

    # Scenario 3: Export to Dictionary for Storage/Analysis
    print("\n" + "=" * 70)
    print("Scenario 3: Export Comparison Results")
    print("=" * 70)

    comparison_dict = comparison.to_dict()
    print(f"\n✓ Comparison exported to dictionary")
    print(f"✓ Contains {len(comparison_dict)} fields")
    print(f"✓ Key metrics:")
    print(f"   - Alpha: {comparison_dict['alpha']:.2f}%")
    print(f"   - Sharpe Alpha: {comparison_dict['sharpe_alpha']:.2f}")
    print(f"   - Win Rate: {comparison_dict['win_rate_vs_benchmark']:.1f}%")
    print(f"   - Window Alphas: {len(comparison_dict['window_alphas'])} values")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\n📚 Key Takeaways:")
    print("   1. Alpha measures absolute return difference vs benchmark")
    print("   2. Relative alpha shows percentage improvement/underperformance")
    print("   3. Sharpe alpha adjusts for risk differences")
    print("   4. Win rate shows consistency across time windows")
    print("   5. Per-window data enables distribution analysis and visualization")


if __name__ == "__main__":
    main()
