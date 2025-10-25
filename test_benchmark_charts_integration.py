"""
Integration Test: Benchmark Charts Module

Verifies that the plotly_benchmark_charts module integrates correctly
with the existing benchmark_comparator module and can be used in real workflows.

This test:
1. Imports all functions from the module
2. Creates BenchmarkComparison objects using the real comparator
3. Generates all 4 chart types
4. Validates output quality
5. Tests with actual BenchmarkComparison structure

Usage:
    uv run python test_benchmark_charts_integration.py
"""

import sys
from typing import Dict, List
import numpy as np
from pathlib import Path

from loguru import logger

# Import the benchmark comparator (real module)
from crypto_trader.analysis.benchmark_comparator import BenchmarkComparator, BenchmarkComparison
from crypto_trader.analysis.aggregator import WindowedMetrics
from crypto_trader.analysis.multipair_aggregator import MultiPairWindowedMetrics, CrossPairCorrelation

# Import the new chart functions
from crypto_trader.reports.formatters.plotly_benchmark_charts import (
    create_alpha_comparison_chart,
    create_win_rate_heatmap,
    create_cumulative_returns_chart,
    create_return_distribution_violin
)


def create_mock_windowed_metrics(
    mean_return: float,
    mean_sharpe: float,
    num_windows: int = 10
) -> WindowedMetrics:
    """Create realistic WindowedMetrics for testing."""
    return WindowedMetrics(
        strategy_name="MockStrategy",
        horizon_name="30d",
        dataset_type="test",
        num_windows=num_windows,
        mean_return=mean_return,
        median_return=mean_return,
        std_return=2.0,
        p25_return=mean_return - 1.0,
        p75_return=mean_return + 1.0,
        weighted_return=mean_return,
        mean_sharpe=mean_sharpe,
        median_sharpe=mean_sharpe,
        std_sharpe=0.3,
        p25_sharpe=mean_sharpe - 0.2,
        p75_sharpe=mean_sharpe + 0.2,
        weighted_sharpe=mean_sharpe,
        mean_drawdown=5.0,
        median_drawdown=5.0,
        std_drawdown=1.0,
        p25_drawdown=4.0,
        p75_drawdown=6.0,
        weighted_drawdown=5.0,
        mean_win_rate=0.6,
        median_win_rate=0.6,
        std_win_rate=0.05,
        p25_win_rate=0.55,
        p75_win_rate=0.65,
        weighted_win_rate=0.6,
        mean_trades=20.0,
        total_trades=200,
        consistency_score=0.8
    )


def create_mock_multipair_metrics(
    strategy_name: str,
    horizon: str,
    mean_return: float,
    mean_sharpe: float,
    num_windows: int = 10
) -> MultiPairWindowedMetrics:
    """Create realistic MultiPairWindowedMetrics for testing."""
    # Create pair metrics
    btc_metrics = create_mock_windowed_metrics(mean_return, mean_sharpe, num_windows)
    eth_metrics = create_mock_windowed_metrics(mean_return * 0.9, mean_sharpe * 0.95, num_windows)

    pair_metrics = {
        'BTC/USDT': btc_metrics,
        'ETH/USDT': eth_metrics
    }

    # Create correlation
    correlation = CrossPairCorrelation(
        pairs=['BTC/USDT', 'ETH/USDT'],
        correlation_matrix={('BTC/USDT', 'ETH/USDT'): 0.7},
        mean_correlation=0.7,
        max_correlation=0.7,
        min_correlation=0.7
    )

    return MultiPairWindowedMetrics(
        strategy_name=strategy_name,
        horizon_name=horizon,
        dataset_type="test",
        pairs=['BTC/USDT', 'ETH/USDT'],
        num_windows=num_windows,
        pair_metrics=pair_metrics,
        portfolio_mean_return=mean_return,
        portfolio_median_return=mean_return,
        portfolio_std_return=2.0,
        portfolio_sharpe=mean_sharpe,
        portfolio_drawdown=5.0,
        correlation=correlation,
        diversification_ratio=1.1,
        risk_contribution={'BTC/USDT': 55.0, 'ETH/USDT': 45.0},
        effective_num_assets=1.8,
        correlation_matrix_df=None
    )


def main():
    """Run integration test."""
    all_validation_failures = []
    total_tests = 0

    logger.info("🔗 Testing Benchmark Charts Integration\n")

    # Test 1: Integration with BenchmarkComparator
    total_tests += 1
    print("Test 1: Integration with BenchmarkComparator")
    try:
        comparator = BenchmarkComparator()

        # Create metrics using real structure
        strategy_metrics = create_mock_multipair_metrics(
            "MACD Strategy", "30d", 15.0, 2.0, num_windows=20
        )
        benchmark_metrics = create_mock_multipair_metrics(
            "BuyAndHold", "30d", 10.0, 1.5, num_windows=20
        )

        # Generate window returns
        np.random.seed(42)
        strategy_returns = np.random.normal(15.0, 3.0, 20).tolist()
        benchmark_returns = np.random.normal(10.0, 2.5, 20).tolist()

        # Create comparison using real comparator
        comparison = comparator.compare_to_benchmark(
            strategy_metrics,
            benchmark_metrics,
            strategy_returns,
            benchmark_returns
        )

        # Verify comparison structure
        required_attrs = [
            'strategy_name', 'horizon_name', 'alpha', 'win_rate_vs_benchmark',
            'window_alphas', 'window_returns_strategy', 'window_returns_benchmark'
        ]
        missing_attrs = [attr for attr in required_attrs if not hasattr(comparison, attr)]

        if missing_attrs:
            all_validation_failures.append(f"Comparison missing attributes: {missing_attrs}")
        else:
            print(f"  ✓ BenchmarkComparison created successfully")
            print(f"  ✓ Alpha: {comparison.alpha:+.2f}%")
            print(f"  ✓ Win Rate: {comparison.win_rate_vs_benchmark:.1f}%")

    except Exception as e:
        all_validation_failures.append(f"Test 1 (Integration) failed: {e}")
        import traceback
        print(f"  ✗ Error: {e}")
        print(traceback.format_exc())

    # Test 2: Generate all charts with real comparisons
    total_tests += 1
    print("\nTest 2: Generate Charts from Real Comparisons")
    try:
        # Create multiple comparisons
        comparisons = {}
        heatmap_data = {}

        for strategy_name in ['MACD', 'RSI', 'BB']:
            heatmap_data[strategy_name] = {}

            for horizon in ['30d', '90d']:
                # Different alphas for variety
                alpha_map = {
                    'MACD': {('30d', 5.0), ('90d', 7.0)},
                    'RSI': {('30d', 3.0), ('90d', 4.0)},
                    'BB': {('30d', -2.0), ('90d', -1.0)}
                }

                strat_mean = 10.0 + list(alpha_map[strategy_name])[0 if horizon == '30d' else 1][1]
                bench_mean = 10.0

                strat_metrics = create_mock_multipair_metrics(
                    strategy_name, horizon, strat_mean, 1.8, num_windows=15
                )
                bench_metrics = create_mock_multipair_metrics(
                    "Benchmark", horizon, bench_mean, 1.5, num_windows=15
                )

                np.random.seed(hash(strategy_name + horizon) % 2**32)
                strat_returns = np.random.normal(strat_mean, 3.0, 15).tolist()
                bench_returns = np.random.normal(bench_mean, 2.5, 15).tolist()

                comp = comparator.compare_to_benchmark(
                    strat_metrics, bench_metrics, strat_returns, bench_returns
                )

                key = f"{strategy_name}_{horizon}"
                comparisons[key] = comp
                heatmap_data[strategy_name][horizon] = comp

        # Generate all charts
        output_dir = Path("integration_test_output")
        output_dir.mkdir(exist_ok=True)

        # Alpha chart
        fig_alpha = create_alpha_comparison_chart(comparisons)
        fig_alpha.write_html(output_dir / "alpha.html")

        # Heatmap
        fig_heatmap = create_win_rate_heatmap(heatmap_data)
        fig_heatmap.write_html(output_dir / "heatmap.html")

        # Cumulative (30d only)
        cumulative_30d = {k: v for k, v in comparisons.items() if '30d' in k}
        fig_cumulative = create_cumulative_returns_chart(cumulative_30d)
        fig_cumulative.write_html(output_dir / "cumulative.html")

        # Violin (30d only)
        fig_violin = create_return_distribution_violin(cumulative_30d)
        fig_violin.write_html(output_dir / "violin.html")

        print(f"  ✓ All 4 charts generated successfully")
        print(f"  ✓ Saved to {output_dir}/")

    except Exception as e:
        all_validation_failures.append(f"Test 2 (Chart generation) failed: {e}")
        import traceback
        print(f"  ✗ Error: {e}")
        print(traceback.format_exc())

    # Test 3: Verify chart file existence and size
    total_tests += 1
    print("\nTest 3: Verify Output Files")
    try:
        expected_files = ['alpha.html', 'heatmap.html', 'cumulative.html', 'violin.html']
        missing_files = []

        for filename in expected_files:
            filepath = output_dir / filename
            if not filepath.exists():
                missing_files.append(filename)
            else:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  ✓ {filename}: {size_mb:.2f} MB")

        if missing_files:
            all_validation_failures.append(f"Missing output files: {missing_files}")

    except Exception as e:
        all_validation_failures.append(f"Test 3 (File verification) failed: {e}")

    # Test 4: Type compatibility
    total_tests += 1
    print("\nTest 4: Type Compatibility")
    try:
        # Verify return types
        import plotly.graph_objects as go

        if not isinstance(fig_alpha, go.Figure):
            all_validation_failures.append("Alpha chart not a go.Figure")
        if not isinstance(fig_heatmap, go.Figure):
            all_validation_failures.append("Heatmap not a go.Figure")
        if not isinstance(fig_cumulative, go.Figure):
            all_validation_failures.append("Cumulative chart not a go.Figure")
        if not isinstance(fig_violin, go.Figure):
            all_validation_failures.append("Violin plot not a go.Figure")

        if not all_validation_failures:
            print("  ✓ All charts return correct type (go.Figure)")

    except Exception as e:
        all_validation_failures.append(f"Test 4 (Type compatibility) failed: {e}")

    # Final result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ INTEGRATION TEST FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ INTEGRATION TEST PASSED - All {total_tests} tests successful")
        print("\n✨ Module Integration Verified:")
        print("  - BenchmarkComparator integration: ✓")
        print("  - Chart generation from real comparisons: ✓")
        print("  - Output file creation: ✓")
        print("  - Type compatibility: ✓")
        print(f"\n📁 Test outputs saved to: {output_dir.absolute()}")
        sys.exit(0)


if __name__ == "__main__":
    main()
