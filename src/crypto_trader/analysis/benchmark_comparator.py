"""
Benchmark Comparison Module

This module calculates comparative performance metrics between trading strategies
and buy-and-hold benchmarks, including alpha, relative performance, and win rates.

**Purpose**: Calculate alpha and win rate metrics comparing strategies to buy-and-hold

**Key Classes**:
- BenchmarkComparison: Dataclass storing comparative metrics
- BenchmarkComparator: Computes comparative statistics between strategy and benchmark

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- dataclasses: https://docs.python.org/3/library/dataclasses.html

**Sample Input**:
```python
comparator = BenchmarkComparator()
comparison = comparator.compare_to_benchmark(
    strategy_metrics=strategy_results,  # MultiPairWindowedMetrics
    benchmark_metrics=benchmark_results  # MultiPairWindowedMetrics
)
```

**Expected Output**:
BenchmarkComparison object with:
- alpha (absolute and relative)
- sharpe_alpha (Sharpe ratio difference)
- win_rate_vs_benchmark (% of windows where strategy beat benchmark)
- per-window alpha values for visualization

**Methodology**:
- Alpha = Strategy Return - Benchmark Return (both absolute and relative %)
- Sharpe Alpha = Strategy Sharpe - Benchmark Sharpe
- Win Rate = (# windows where strategy > benchmark) / total windows
- Per-window analysis for detailed comparison and visualization
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

# Handle imports for both module and standalone execution
try:
    from .multipair_aggregator import MultiPairWindowedMetrics
except ImportError:
    from crypto_trader.analysis.multipair_aggregator import MultiPairWindowedMetrics


@dataclass
class BenchmarkComparison:
    """
    Comparative metrics between a strategy and buy-and-hold benchmark.

    Attributes:
        strategy_name: Name of the strategy being compared
        horizon_name: Time horizon (e.g., '30d', '90d')
        dataset_type: 'train' or 'test'

        strategy_return: Mean return of the strategy (%)
        benchmark_return: Mean return of buy-and-hold (%)
        alpha: Absolute alpha (strategy - benchmark) in percentage points
        relative_alpha: Alpha as percentage of benchmark return

        strategy_sharpe: Mean Sharpe ratio of strategy
        benchmark_sharpe: Mean Sharpe ratio of benchmark
        sharpe_alpha: Sharpe difference (strategy - benchmark)

        windows_beat_benchmark: Number of windows where strategy > benchmark
        total_windows: Total number of windows analyzed
        win_rate_vs_benchmark: Percentage of windows where strategy beat benchmark

        window_alphas: Per-window alpha values (for distribution analysis)
        window_returns_strategy: Per-window strategy returns (for visualization)
        window_returns_benchmark: Per-window benchmark returns (for visualization)
    """
    strategy_name: str
    horizon_name: str
    dataset_type: str

    # Return comparison
    strategy_return: float
    benchmark_return: float
    alpha: float
    relative_alpha: float

    # Sharpe comparison
    strategy_sharpe: float
    benchmark_sharpe: float
    sharpe_alpha: float

    # Win rate statistics
    windows_beat_benchmark: int
    total_windows: int
    win_rate_vs_benchmark: float

    # Per-window data for visualization
    window_alphas: List[float]
    window_returns_strategy: List[float]
    window_returns_benchmark: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def summary_string(self) -> str:
        """Generate human-readable summary of benchmark comparison."""
        alpha_sign = "+" if self.alpha > 0 else ""
        sharpe_sign = "+" if self.sharpe_alpha > 0 else ""

        return (
            f"Benchmark Comparison: {self.strategy_name} ({self.horizon_name}, {self.dataset_type})\n"
            f"  Strategy Return: {self.strategy_return:.2f}%\n"
            f"  Benchmark Return: {self.benchmark_return:.2f}%\n"
            f"  Alpha: {alpha_sign}{self.alpha:.2f}% ({self.relative_alpha:+.1f}% relative)\n"
            f"  Strategy Sharpe: {self.strategy_sharpe:.2f}\n"
            f"  Benchmark Sharpe: {self.benchmark_sharpe:.2f}\n"
            f"  Sharpe Alpha: {sharpe_sign}{self.sharpe_alpha:.2f}\n"
            f"  Win Rate: {self.win_rate_vs_benchmark:.1f}% "
            f"({self.windows_beat_benchmark}/{self.total_windows} windows)\n"
            f"  Alpha Distribution: μ={np.mean(self.window_alphas):.2f}%, "
            f"σ={np.std(self.window_alphas):.2f}%"
        )


class BenchmarkComparator:
    """
    Computes comparative statistics between trading strategies and benchmarks.

    Analyzes both portfolio-level and per-window performance to calculate:
    - Alpha (absolute and relative)
    - Sharpe alpha
    - Win rate vs benchmark
    - Distribution of per-window alphas
    """

    def __init__(self):
        """Initialize benchmark comparator."""
        logger.debug("BenchmarkComparator initialized")

    def _extract_window_returns(
        self,
        metrics: MultiPairWindowedMetrics
    ) -> List[float]:
        """
        Extract per-window returns from MultiPairWindowedMetrics.

        For multi-pair portfolios, we need to reconstruct per-window portfolio returns
        from the individual pair metrics.

        Args:
            metrics: MultiPairWindowedMetrics object

        Returns:
            List of per-window portfolio returns
        """
        # Get per-pair window returns
        pair_window_returns = {}

        for pair, pair_metrics in metrics.pair_metrics.items():
            # Each pair has aggregated metrics, but we need individual window data
            # The individual window returns aren't stored in WindowedMetrics
            # We'll need to compute portfolio returns from the original results
            # For now, use a placeholder - this will be filled from actual window data
            pair_window_returns[pair] = []

        # If we don't have window-level data, we can't compute per-window portfolio returns
        # Return empty list - the calling code should handle this
        logger.warning(
            "Cannot extract per-window returns from MultiPairWindowedMetrics alone. "
            "Pass raw results to compare_to_benchmark_with_windows instead."
        )
        return []

    def compare_to_benchmark(
        self,
        strategy_metrics: MultiPairWindowedMetrics,
        benchmark_metrics: MultiPairWindowedMetrics,
        strategy_window_returns: Optional[List[float]] = None,
        benchmark_window_returns: Optional[List[float]] = None
    ) -> BenchmarkComparison:
        """
        Compare strategy performance to benchmark.

        Args:
            strategy_metrics: MultiPairWindowedMetrics for the trading strategy
            benchmark_metrics: MultiPairWindowedMetrics for buy-and-hold benchmark
            strategy_window_returns: Optional list of per-window portfolio returns for strategy
            benchmark_window_returns: Optional list of per-window portfolio returns for benchmark

        Returns:
            BenchmarkComparison object with all comparative metrics

        Raises:
            ValueError: If strategy and benchmark have different configurations
        """
        # Validate inputs
        if strategy_metrics.horizon_name != benchmark_metrics.horizon_name:
            raise ValueError(
                f"Horizon mismatch: strategy={strategy_metrics.horizon_name}, "
                f"benchmark={benchmark_metrics.horizon_name}"
            )

        if strategy_metrics.dataset_type != benchmark_metrics.dataset_type:
            raise ValueError(
                f"Dataset type mismatch: strategy={strategy_metrics.dataset_type}, "
                f"benchmark={benchmark_metrics.dataset_type}"
            )

        if strategy_metrics.num_windows != benchmark_metrics.num_windows:
            logger.warning(
                f"Window count mismatch: strategy={strategy_metrics.num_windows}, "
                f"benchmark={benchmark_metrics.num_windows}. Using minimum."
            )

        # Extract portfolio-level metrics
        strategy_return = strategy_metrics.portfolio_mean_return
        benchmark_return = benchmark_metrics.portfolio_mean_return

        strategy_sharpe = strategy_metrics.portfolio_sharpe
        benchmark_sharpe = benchmark_metrics.portfolio_sharpe

        # Calculate alpha metrics
        alpha = strategy_return - benchmark_return

        # Relative alpha: alpha as percentage of benchmark return
        # Handle division by zero
        if benchmark_return != 0:
            relative_alpha = (alpha / abs(benchmark_return)) * 100
        else:
            relative_alpha = 0.0

        # Calculate Sharpe alpha
        sharpe_alpha = strategy_sharpe - benchmark_sharpe

        # Calculate per-window metrics if available
        if strategy_window_returns and benchmark_window_returns:
            # Ensure same length
            min_windows = min(len(strategy_window_returns), len(benchmark_window_returns))
            strategy_returns = strategy_window_returns[:min_windows]
            benchmark_returns = benchmark_window_returns[:min_windows]

            # Calculate per-window alphas
            window_alphas = [
                strat - bench
                for strat, bench in zip(strategy_returns, benchmark_returns)
            ]

            # Calculate win rate
            wins = sum(1 for strat, bench in zip(strategy_returns, benchmark_returns) if strat > bench)
            total_windows = min_windows
            win_rate = (wins / total_windows * 100) if total_windows > 0 else 0.0

        else:
            # Use num_windows from metrics but can't compute per-window stats
            total_windows = min(strategy_metrics.num_windows, benchmark_metrics.num_windows)
            window_alphas = []
            strategy_returns = []
            benchmark_returns = []
            wins = 0
            win_rate = 0.0

            logger.warning(
                "Per-window returns not provided. Win rate and window alphas unavailable. "
                "Pass strategy_window_returns and benchmark_window_returns for full analysis."
            )

        comparison = BenchmarkComparison(
            strategy_name=strategy_metrics.strategy_name,
            horizon_name=strategy_metrics.horizon_name,
            dataset_type=strategy_metrics.dataset_type,
            strategy_return=strategy_return,
            benchmark_return=benchmark_return,
            alpha=alpha,
            relative_alpha=relative_alpha,
            strategy_sharpe=strategy_sharpe,
            benchmark_sharpe=benchmark_sharpe,
            sharpe_alpha=sharpe_alpha,
            windows_beat_benchmark=wins,
            total_windows=total_windows,
            win_rate_vs_benchmark=win_rate,
            window_alphas=window_alphas,
            window_returns_strategy=strategy_returns,
            window_returns_benchmark=benchmark_returns
        )

        logger.info(f"📊 Benchmark Comparison: {strategy_metrics.strategy_name}")
        logger.info(f"   Alpha: {alpha:+.2f}% ({relative_alpha:+.1f}% relative)")
        logger.info(f"   Sharpe Alpha: {sharpe_alpha:+.2f}")
        if win_rate > 0:
            logger.info(f"   Win Rate: {win_rate:.1f}% ({wins}/{total_windows} windows)")

        return comparison


if __name__ == "__main__":
    """
    Validation block for benchmark comparator.

    Tests comparison calculations with mock MultiPairWindowedMetrics objects.
    """
    import sys

    # Handle imports for standalone execution
    try:
        from .aggregator import WindowedMetrics
        from .multipair_aggregator import CrossPairCorrelation
    except ImportError:
        # Running as standalone script
        from crypto_trader.analysis.aggregator import WindowedMetrics
        from crypto_trader.analysis.multipair_aggregator import CrossPairCorrelation

    all_validation_failures = []
    total_tests = 0

    # Helper function to create mock WindowedMetrics
    def create_mock_windowed_metrics(
        mean_return: float,
        mean_sharpe: float,
        num_windows: int = 10
    ) -> WindowedMetrics:
        """Create mock WindowedMetrics for testing."""
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

    # Helper function to create mock MultiPairWindowedMetrics
    def create_mock_multipair_metrics(
        strategy_name: str,
        mean_return: float,
        mean_sharpe: float,
        num_windows: int = 10
    ) -> MultiPairWindowedMetrics:
        """Create mock MultiPairWindowedMetrics for testing."""
        # CrossPairCorrelation already imported at module level

        # Create mock pair metrics
        btc_metrics = create_mock_windowed_metrics(mean_return, mean_sharpe, num_windows)
        eth_metrics = create_mock_windowed_metrics(mean_return * 0.9, mean_sharpe * 0.95, num_windows)

        pair_metrics = {
            'BTC/USDT': btc_metrics,
            'ETH/USDT': eth_metrics
        }

        # Create mock correlation
        correlation = CrossPairCorrelation(
            pairs=['BTC/USDT', 'ETH/USDT'],
            correlation_matrix={('BTC/USDT', 'ETH/USDT'): 0.7},
            mean_correlation=0.7,
            max_correlation=0.7,
            min_correlation=0.7
        )

        return MultiPairWindowedMetrics(
            strategy_name=strategy_name,
            horizon_name="30d",
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

    # Test 1: Positive Alpha Scenario (Strategy Beats Benchmark)
    total_tests += 1
    print("Test 1: Positive Alpha Scenario")
    try:
        comparator = BenchmarkComparator()

        # Strategy with 15% return, 2.0 Sharpe
        strategy_metrics = create_mock_multipair_metrics("TestStrategy", 15.0, 2.0)

        # Benchmark with 10% return, 1.5 Sharpe
        benchmark_metrics = create_mock_multipair_metrics("BuyAndHold", 10.0, 1.5)

        # Create mock window returns (strategy beats benchmark in 7/10 windows)
        strategy_window_returns = [12.0, 18.0, 14.0, 20.0, 11.0, 16.0, 15.0, 13.0, 9.0, 17.0]
        benchmark_window_returns = [8.0, 12.0, 9.0, 15.0, 10.0, 11.0, 10.0, 9.0, 11.0, 13.0]

        comparison = comparator.compare_to_benchmark(
            strategy_metrics,
            benchmark_metrics,
            strategy_window_returns,
            benchmark_window_returns
        )

        # Verify alpha
        expected_alpha = 15.0 - 10.0  # 5.0
        if abs(comparison.alpha - expected_alpha) > 0.01:
            all_validation_failures.append(
                f"Alpha incorrect: expected {expected_alpha:.2f}, got {comparison.alpha:.2f}"
            )

        # Verify relative alpha
        expected_relative_alpha = (5.0 / 10.0) * 100  # 50%
        if abs(comparison.relative_alpha - expected_relative_alpha) > 0.1:
            all_validation_failures.append(
                f"Relative alpha incorrect: expected {expected_relative_alpha:.1f}%, "
                f"got {comparison.relative_alpha:.1f}%"
            )

        # Verify Sharpe alpha
        expected_sharpe_alpha = 2.0 - 1.5  # 0.5
        if abs(comparison.sharpe_alpha - expected_sharpe_alpha) > 0.01:
            all_validation_failures.append(
                f"Sharpe alpha incorrect: expected {expected_sharpe_alpha:.2f}, "
                f"got {comparison.sharpe_alpha:.2f}"
            )

        # Verify win rate
        # Manual count: 12>8, 18>12, 14>9, 20>15, 11>10, 16>11, 15>10, 13>9, 9<11, 17>13
        expected_wins = 9  # 9 out of 10 windows
        expected_win_rate = 90.0
        if comparison.windows_beat_benchmark != expected_wins:
            all_validation_failures.append(
                f"Win count incorrect: expected {expected_wins}, "
                f"got {comparison.windows_beat_benchmark}"
            )

        if abs(comparison.win_rate_vs_benchmark - expected_win_rate) > 0.1:
            all_validation_failures.append(
                f"Win rate incorrect: expected {expected_win_rate:.1f}%, "
                f"got {comparison.win_rate_vs_benchmark:.1f}%"
            )

        # Verify window alphas
        if len(comparison.window_alphas) != 10:
            all_validation_failures.append(
                f"Window alphas count incorrect: expected 10, got {len(comparison.window_alphas)}"
            )

        print(f"  ✓ Alpha: {comparison.alpha:+.2f}% (relative: {comparison.relative_alpha:+.1f}%)")
        print(f"  ✓ Sharpe Alpha: {comparison.sharpe_alpha:+.2f}")
        print(f"  ✓ Win Rate: {comparison.win_rate_vs_benchmark:.1f}% ({comparison.windows_beat_benchmark}/10)")

    except Exception as e:
        all_validation_failures.append(f"Positive alpha scenario failed: {e}")

    # Test 2: Negative Alpha Scenario (Strategy Underperforms)
    total_tests += 1
    print("\nTest 2: Negative Alpha Scenario")
    try:
        # Strategy with 8% return, 1.2 Sharpe
        strategy_metrics = create_mock_multipair_metrics("WeakStrategy", 8.0, 1.2)

        # Benchmark with 12% return, 1.8 Sharpe
        benchmark_metrics = create_mock_multipair_metrics("BuyAndHold", 12.0, 1.8)

        # Create mock window returns (strategy beats benchmark in only 3/10 windows)
        strategy_window_returns = [7.0, 9.0, 6.0, 11.0, 8.0, 7.5, 9.5, 8.5, 7.0, 6.5]
        benchmark_window_returns = [10.0, 13.0, 11.0, 9.0, 12.0, 14.0, 11.5, 13.5, 12.5, 13.0]

        comparison = comparator.compare_to_benchmark(
            strategy_metrics,
            benchmark_metrics,
            strategy_window_returns,
            benchmark_window_returns
        )

        # Verify negative alpha
        expected_alpha = 8.0 - 12.0  # -4.0
        if abs(comparison.alpha - expected_alpha) > 0.01:
            all_validation_failures.append(
                f"Negative alpha incorrect: expected {expected_alpha:.2f}, got {comparison.alpha:.2f}"
            )

        # Verify alpha is negative
        if comparison.alpha >= 0:
            all_validation_failures.append(
                f"Alpha should be negative, got {comparison.alpha:.2f}"
            )

        # Verify low win rate
        # Manual count: 7<10, 9<13, 6<11, 11>9, 8<12, 7.5<14, 9.5<11.5, 8.5<13.5, 7<12.5, 6.5<13
        expected_wins = 1  # Only 1 out of 10 windows
        if comparison.windows_beat_benchmark != expected_wins:
            all_validation_failures.append(
                f"Win count incorrect for underperforming strategy: expected {expected_wins}, "
                f"got {comparison.windows_beat_benchmark}"
            )

        print(f"  ✓ Alpha: {comparison.alpha:+.2f}% (negative as expected)")
        print(f"  ✓ Sharpe Alpha: {comparison.sharpe_alpha:+.2f} (negative as expected)")
        print(f"  ✓ Win Rate: {comparison.win_rate_vs_benchmark:.1f}% (low as expected)")

    except Exception as e:
        all_validation_failures.append(f"Negative alpha scenario failed: {e}")

    # Test 3: Win Rate Calculation Edge Cases
    total_tests += 1
    print("\nTest 3: Win Rate Calculation Edge Cases")
    try:
        strategy_metrics = create_mock_multipair_metrics("ConsistentStrategy", 10.0, 1.5)
        benchmark_metrics = create_mock_multipair_metrics("BuyAndHold", 10.0, 1.5)

        # Test 3a: All wins
        all_wins_strategy = [12.0] * 5
        all_wins_benchmark = [8.0] * 5

        comparison_all_wins = comparator.compare_to_benchmark(
            strategy_metrics,
            benchmark_metrics,
            all_wins_strategy,
            all_wins_benchmark
        )

        if comparison_all_wins.win_rate_vs_benchmark != 100.0:
            all_validation_failures.append(
                f"All-wins scenario: expected 100% win rate, got {comparison_all_wins.win_rate_vs_benchmark:.1f}%"
            )

        # Test 3b: No wins
        no_wins_strategy = [8.0] * 5
        no_wins_benchmark = [12.0] * 5

        comparison_no_wins = comparator.compare_to_benchmark(
            strategy_metrics,
            benchmark_metrics,
            no_wins_strategy,
            no_wins_benchmark
        )

        if comparison_no_wins.win_rate_vs_benchmark != 0.0:
            all_validation_failures.append(
                f"No-wins scenario: expected 0% win rate, got {comparison_no_wins.win_rate_vs_benchmark:.1f}%"
            )

        # Test 3c: Exact tie
        tie_returns = [10.0, 10.0, 10.0]

        comparison_tie = comparator.compare_to_benchmark(
            strategy_metrics,
            benchmark_metrics,
            tie_returns,
            tie_returns
        )

        if comparison_tie.win_rate_vs_benchmark != 0.0:
            all_validation_failures.append(
                f"Tie scenario: expected 0% win rate (ties don't count as wins), "
                f"got {comparison_tie.win_rate_vs_benchmark:.1f}%"
            )

        print(f"  ✓ All wins: {comparison_all_wins.win_rate_vs_benchmark:.1f}%")
        print(f"  ✓ No wins: {comparison_no_wins.win_rate_vs_benchmark:.1f}%")
        print(f"  ✓ Tie: {comparison_tie.win_rate_vs_benchmark:.1f}%")

    except Exception as e:
        all_validation_failures.append(f"Win rate edge cases failed: {e}")

    # Test 4: Division by Zero Handling
    total_tests += 1
    print("\nTest 4: Division by Zero Handling")
    try:
        # Strategy with 5% return
        strategy_metrics = create_mock_multipair_metrics("TestStrategy", 5.0, 1.5)

        # Benchmark with 0% return (test division by zero)
        benchmark_metrics = create_mock_multipair_metrics("ZeroBenchmark", 0.0, 0.5)

        comparison = comparator.compare_to_benchmark(
            strategy_metrics,
            benchmark_metrics
        )

        # Should handle division by zero gracefully
        if not isinstance(comparison.relative_alpha, (int, float)):
            all_validation_failures.append(
                f"Relative alpha should be numeric even with zero benchmark, "
                f"got {type(comparison.relative_alpha)}"
            )

        # Alpha should still be calculated correctly
        expected_alpha = 5.0 - 0.0
        if abs(comparison.alpha - expected_alpha) > 0.01:
            all_validation_failures.append(
                f"Alpha with zero benchmark incorrect: expected {expected_alpha:.2f}, "
                f"got {comparison.alpha:.2f}"
            )

        print(f"  ✓ Zero benchmark handled: alpha={comparison.alpha:.2f}%, "
              f"relative_alpha={comparison.relative_alpha:.1f}%")

    except Exception as e:
        all_validation_failures.append(f"Division by zero handling failed: {e}")

    # Test 5: Serialization
    total_tests += 1
    print("\nTest 5: Serialization")
    try:
        # Use comparison from Test 1
        strategy_metrics = create_mock_multipair_metrics("TestStrategy", 15.0, 2.0)
        benchmark_metrics = create_mock_multipair_metrics("BuyAndHold", 10.0, 1.5)

        comparison = comparator.compare_to_benchmark(strategy_metrics, benchmark_metrics)
        comparison_dict = comparison.to_dict()

        required_keys = {
            'strategy_name', 'horizon_name', 'dataset_type',
            'strategy_return', 'benchmark_return', 'alpha', 'relative_alpha',
            'strategy_sharpe', 'benchmark_sharpe', 'sharpe_alpha',
            'windows_beat_benchmark', 'total_windows', 'win_rate_vs_benchmark',
            'window_alphas', 'window_returns_strategy', 'window_returns_benchmark'
        }

        missing_keys = required_keys - set(comparison_dict.keys())
        if missing_keys:
            all_validation_failures.append(f"Missing keys in serialization: {missing_keys}")

        print(f"  ✓ All required keys present in serialization")
        print(f"  ✓ Keys: {len(comparison_dict)} total")

    except Exception as e:
        all_validation_failures.append(f"Serialization failed: {e}")

    # Test 6: Summary String Generation
    total_tests += 1
    print("\nTest 6: Summary String Generation")
    try:
        strategy_metrics = create_mock_multipair_metrics("TestStrategy", 15.0, 2.0)
        benchmark_metrics = create_mock_multipair_metrics("BuyAndHold", 10.0, 1.5)

        strategy_window_returns = [12.0, 18.0, 14.0, 20.0, 11.0]
        benchmark_window_returns = [8.0, 12.0, 9.0, 15.0, 10.0]

        comparison = comparator.compare_to_benchmark(
            strategy_metrics,
            benchmark_metrics,
            strategy_window_returns,
            benchmark_window_returns
        )

        summary = comparison.summary_string()

        required_terms = [
            'TestStrategy', 'Alpha', 'Sharpe', 'Win Rate', 'Benchmark'
        ]
        missing_terms = [term for term in required_terms if term not in summary]

        if missing_terms:
            all_validation_failures.append(f"Summary missing required terms: {missing_terms}")

        print(f"  ✓ Summary generated with all required terms")
        print(f"\nSample Summary:\n{summary}")

    except Exception as e:
        all_validation_failures.append(f"Summary string generation failed: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Benchmark comparator validated and ready for production use")
        print("\n📊 Comparison Metrics:")
        print("  - Alpha: Absolute return difference (strategy - benchmark)")
        print("  - Relative Alpha: Alpha as % of benchmark return")
        print("  - Sharpe Alpha: Sharpe ratio difference")
        print("  - Win Rate: % of windows where strategy beat benchmark")
        print("  - Per-Window Analysis: Alpha distribution for visualization")
        sys.exit(0)
