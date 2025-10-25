"""
Window Results Aggregator

This module aggregates backtest results across multiple windows to compute
comprehensive statistics (mean, median, std dev, percentiles, etc.).

**Purpose**: Aggregate and analyze performance across train/test windows

**Key Classes**:
- WindowedMetrics: Aggregated statistics for a strategy across windows
- ResultsAggregator: Computes aggregated metrics from window results

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/

**Sample Input**:
```python
aggregator = ResultsAggregator()
metrics = aggregator.aggregate_windows(
    results=window_results,
    strategy_name='SMA_Crossover',
    horizon_name='30d',
    dataset_type='train'
)
```

**Expected Output**:
WindowedMetrics object with mean, median, std, percentiles for all metrics.

**Statistical Methods**:
- Mean: Simple average across windows
- Median: Robust central tendency
- Std Dev: Measure of consistency
- Percentiles (25th, 50th, 75th): Distribution shape
- Weighted Average: Optional time-weighting for recent windows
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class WindowedMetrics:
    """
    Aggregated metrics for a strategy across multiple windows.

    Provides comprehensive statistics: mean, median, std dev, percentiles.
    """
    strategy_name: str
    horizon_name: str
    dataset_type: str  # 'train' or 'test'
    num_windows: int

    # Return metrics
    mean_return: float
    median_return: float
    std_return: float
    p25_return: float
    p75_return: float
    weighted_return: float

    # Sharpe ratio metrics
    mean_sharpe: float
    median_sharpe: float
    std_sharpe: float
    p25_sharpe: float
    p75_sharpe: float
    weighted_sharpe: float

    # Drawdown metrics
    mean_drawdown: float
    median_drawdown: float
    std_drawdown: float
    p25_drawdown: float
    p75_drawdown: float
    weighted_drawdown: float

    # Win rate metrics
    mean_win_rate: float
    median_win_rate: float
    std_win_rate: float
    p25_win_rate: float
    p75_win_rate: float
    weighted_win_rate: float

    # Trade count
    mean_trades: float
    total_trades: int

    # Consistency score (inverse of std / mean for Sharpe)
    consistency_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def summary_string(self) -> str:
        """Generate human-readable summary."""
        return (
            f"{self.strategy_name} ({self.horizon_name}, {self.dataset_type}):\n"
            f"  Windows: {self.num_windows}\n"
            f"  Return: {self.mean_return:.2%} ± {self.std_return:.2%} "
            f"(median: {self.median_return:.2%})\n"
            f"  Sharpe: {self.mean_sharpe:.2f} ± {self.std_sharpe:.2f} "
            f"(median: {self.median_sharpe:.2f})\n"
            f"  Drawdown: {self.mean_drawdown:.2%} ± {self.std_drawdown:.2%} "
            f"(median: {self.median_drawdown:.2%})\n"
            f"  Consistency: {self.consistency_score:.3f}"
        )


class ResultsAggregator:
    """
    Aggregates backtest results across windows.

    Computes mean, median, std dev, percentiles, and weighted averages
    for all performance metrics.
    """

    def __init__(self, recent_weight: float = 0.6):
        """
        Initialize aggregator.

        Args:
            recent_weight: Weight for most recent window in weighted average (0-1)
                         Higher values give more weight to recent performance
        """
        self.recent_weight = recent_weight
        logger.debug(f"ResultsAggregator initialized with recent_weight={recent_weight}")

    def _calculate_statistics(
        self,
        values: List[float],
        window_ages: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a metric.

        Args:
            values: List of metric values across windows
            window_ages: Optional list of window ages (0 = most recent)

        Returns:
            Dict with mean, median, std, percentiles, weighted_mean
        """
        if not values:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'p25': 0.0,
                'p75': 0.0,
                'weighted': 0.0
            }

        # Filter out inf and nan values before computing statistics
        arr = np.array(values)
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            # All values are inf/nan - return zeros
            logger.warning(f"All values are non-finite (inf/nan), returning zero statistics")
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'p25': 0.0,
                'p75': 0.0,
                'weighted': 0.0
            }

        # Use only finite values
        arr = arr[finite_mask]
        if len(arr) != len(values):
            logger.warning(f"Filtered {len(values) - len(arr)} non-finite values from statistics")

        # Basic statistics
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        std_val = float(np.std(arr))
        p25_val = float(np.percentile(arr, 25))
        p75_val = float(np.percentile(arr, 75))

        # Weighted average (exponentially decay older windows)
        # Note: window_ages correspond to original values, need to filter them too
        if window_ages is not None and len(window_ages) == len(values):
            # Filter window_ages to match filtered arr
            filtered_ages = [age for age, is_finite in zip(window_ages, finite_mask) if is_finite]
            if len(filtered_ages) == len(arr):
                # Exponential weights: newer windows get more weight
                max_age = max(filtered_ages) if filtered_ages else 0
                weights = np.array([
                    self.recent_weight ** (age / (max_age + 1))
                    for age in filtered_ages
                ])
                weights = weights / weights.sum()  # Normalize
                weighted_val = float(np.average(arr, weights=weights))
            else:
                # Mismatch in sizes, use mean
                weighted_val = mean_val
        else:
            # No weighting, use mean
            weighted_val = mean_val

        return {
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'p25': p25_val,
            'p75': p75_val,
            'weighted': weighted_val
        }

    def aggregate_windows(
        self,
        results: List[Dict[str, Any]],
        strategy_name: str,
        horizon_name: str,
        dataset_type: str
    ) -> WindowedMetrics:
        """
        Aggregate results from multiple windows.

        Args:
            results: List of backtest result dictionaries
            strategy_name: Name of strategy
            horizon_name: Horizon name (e.g., '30d')
            dataset_type: 'train' or 'test'

        Returns:
            WindowedMetrics with aggregated statistics
        """
        if not results:
            logger.warning(
                f"No results to aggregate for {strategy_name} "
                f"({horizon_name}, {dataset_type})"
            )
            # Return zero metrics
            return WindowedMetrics(
                strategy_name=strategy_name,
                horizon_name=horizon_name,
                dataset_type=dataset_type,
                num_windows=0,
                mean_return=0.0, median_return=0.0, std_return=0.0,
                p25_return=0.0, p75_return=0.0, weighted_return=0.0,
                mean_sharpe=0.0, median_sharpe=0.0, std_sharpe=0.0,
                p25_sharpe=0.0, p75_sharpe=0.0, weighted_sharpe=0.0,
                mean_drawdown=0.0, median_drawdown=0.0, std_drawdown=0.0,
                p25_drawdown=0.0, p75_drawdown=0.0, weighted_drawdown=0.0,
                mean_win_rate=0.0, median_win_rate=0.0, std_win_rate=0.0,
                p25_win_rate=0.0, p75_win_rate=0.0, weighted_win_rate=0.0,
                mean_trades=0.0, total_trades=0,
                consistency_score=0.0
            )

        # Extract metric lists
        returns = [r.get('total_return', 0.0) for r in results if 'error' not in r]
        sharpes = [r.get('sharpe_ratio', 0.0) for r in results if 'error' not in r]
        drawdowns = [r.get('max_drawdown', 0.0) for r in results if 'error' not in r]
        win_rates = [r.get('win_rate', 0.0) for r in results if 'error' not in r]
        trades = [r.get('total_trades', 0) for r in results if 'error' not in r]

        # Window ages (for weighted average)
        # Assume results are ordered, first is oldest
        window_ages = list(range(len(returns)-1, -1, -1))  # Reverse: [N-1, N-2, ..., 0]

        # Calculate statistics for each metric
        return_stats = self._calculate_statistics(returns, window_ages)
        sharpe_stats = self._calculate_statistics(sharpes, window_ages)
        drawdown_stats = self._calculate_statistics(drawdowns, window_ages)
        win_rate_stats = self._calculate_statistics(win_rates, window_ages)

        # Calculate consistency score (lower variance = higher consistency)
        # Use Sharpe ratio as base metric
        if sharpe_stats['mean'] != 0 and sharpe_stats['std'] != 0:
            # Coefficient of variation: std / mean
            cv = abs(sharpe_stats['std'] / sharpe_stats['mean'])
            # Consistency = 1 / (1 + cv)  -- maps to [0, 1]
            consistency = 1.0 / (1.0 + cv)
        else:
            consistency = 0.0

        metrics = WindowedMetrics(
            strategy_name=strategy_name,
            horizon_name=horizon_name,
            dataset_type=dataset_type,
            num_windows=len(returns),

            mean_return=return_stats['mean'],
            median_return=return_stats['median'],
            std_return=return_stats['std'],
            p25_return=return_stats['p25'],
            p75_return=return_stats['p75'],
            weighted_return=return_stats['weighted'],

            mean_sharpe=sharpe_stats['mean'],
            median_sharpe=sharpe_stats['median'],
            std_sharpe=sharpe_stats['std'],
            p25_sharpe=sharpe_stats['p25'],
            p75_sharpe=sharpe_stats['p75'],
            weighted_sharpe=sharpe_stats['weighted'],

            mean_drawdown=drawdown_stats['mean'],
            median_drawdown=drawdown_stats['median'],
            std_drawdown=drawdown_stats['std'],
            p25_drawdown=drawdown_stats['p25'],
            p75_drawdown=drawdown_stats['p75'],
            weighted_drawdown=drawdown_stats['weighted'],

            mean_win_rate=win_rate_stats['mean'],
            median_win_rate=win_rate_stats['median'],
            std_win_rate=win_rate_stats['std'],
            p25_win_rate=win_rate_stats['p25'],
            p75_win_rate=win_rate_stats['p75'],
            weighted_win_rate=win_rate_stats['weighted'],

            mean_trades=float(np.mean(trades)) if trades else 0.0,
            total_trades=sum(trades),

            consistency_score=consistency
        )

        logger.debug(
            f"Aggregated {len(returns)} windows for {strategy_name} "
            f"({horizon_name}, {dataset_type})"
        )

        return metrics

    def compute_composite_score(
        self,
        metrics: WindowedMetrics,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute composite score from aggregated metrics.

        Args:
            metrics: WindowedMetrics to score
            weights: Optional custom weights for metrics
                    Default: {'return': 0.3, 'sharpe': 0.3, 'drawdown': 0.2, 'consistency': 0.2}

        Returns:
            Composite score (0-100)
        """
        if weights is None:
            weights = {
                'return': 0.3,
                'sharpe': 0.3,
                'drawdown': 0.2,
                'consistency': 0.2
            }

        # Normalize metrics to 0-100 scale
        # Return: map [-100%, +100%] to [0, 100]
        return_score = max(0, min(100, (metrics.mean_return + 1.0) * 50))

        # Sharpe: map [-2, +5] to [0, 100]
        sharpe_score = max(0, min(100, (metrics.mean_sharpe + 2.0) / 7.0 * 100))

        # Drawdown: invert (lower is better), map [0%, 100%] to [100, 0]
        drawdown_score = max(0, 100 * (1.0 - metrics.mean_drawdown))

        # Consistency: already in [0, 1], scale to [0, 100]
        consistency_score = metrics.consistency_score * 100

        # Weighted sum
        composite = (
            weights['return'] * return_score +
            weights['sharpe'] * sharpe_score +
            weights['drawdown'] * drawdown_score +
            weights['consistency'] * consistency_score
        )

        return float(composite)


if __name__ == "__main__":
    """
    Validation block for results aggregator.

    Tests statistical aggregation with synthetic data.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: Basic aggregation
    total_tests += 1
    print("Test 1: Basic Aggregation")
    try:
        aggregator = ResultsAggregator(recent_weight=0.6)

        # Synthetic results for 5 windows
        results = [
            {'total_return': 0.10, 'sharpe_ratio': 1.5, 'max_drawdown': 0.15,
             'win_rate': 0.55, 'total_trades': 20},
            {'total_return': 0.12, 'sharpe_ratio': 1.8, 'max_drawdown': 0.12,
             'win_rate': 0.58, 'total_trades': 22},
            {'total_return': 0.08, 'sharpe_ratio': 1.2, 'max_drawdown': 0.18,
             'win_rate': 0.52, 'total_trades': 18},
            {'total_return': 0.15, 'sharpe_ratio': 2.0, 'max_drawdown': 0.10,
             'win_rate': 0.60, 'total_trades': 25},
            {'total_return': 0.11, 'sharpe_ratio': 1.6, 'max_drawdown': 0.14,
             'win_rate': 0.56, 'total_trades': 21},
        ]

        metrics = aggregator.aggregate_windows(
            results, 'TestStrategy', '30d', 'train'
        )

        # Verify basic stats
        expected_mean_return = np.mean([0.10, 0.12, 0.08, 0.15, 0.11])
        if abs(metrics.mean_return - expected_mean_return) > 0.001:
            all_validation_failures.append(
                f"Mean return incorrect: expected {expected_mean_return:.4f}, "
                f"got {metrics.mean_return:.4f}"
            )

        # Verify median
        expected_median_return = np.median([0.10, 0.12, 0.08, 0.15, 0.11])
        if abs(metrics.median_return - expected_median_return) > 0.001:
            all_validation_failures.append(
                f"Median return incorrect: expected {expected_median_return:.4f}, "
                f"got {metrics.median_return:.4f}"
            )

        # Verify window count
        if metrics.num_windows != 5:
            all_validation_failures.append(
                f"Window count incorrect: expected 5, got {metrics.num_windows}"
            )

        print(f"  ✓ Mean return: {metrics.mean_return:.4f}")
        print(f"  ✓ Median return: {metrics.median_return:.4f}")
        print(f"  ✓ Std return: {metrics.std_return:.4f}")
        print(f"  ✓ Windows: {metrics.num_windows}")

    except Exception as e:
        all_validation_failures.append(f"Basic aggregation failed: {e}")

    # Test 2: Percentile calculation
    total_tests += 1
    print("\nTest 2: Percentile Calculation")
    try:
        # Check 25th and 75th percentiles
        expected_p25 = np.percentile([0.10, 0.12, 0.08, 0.15, 0.11], 25)
        expected_p75 = np.percentile([0.10, 0.12, 0.08, 0.15, 0.11], 75)

        if abs(metrics.p25_return - expected_p25) > 0.001:
            all_validation_failures.append(
                f"25th percentile incorrect: expected {expected_p25:.4f}, "
                f"got {metrics.p25_return:.4f}"
            )

        if abs(metrics.p75_return - expected_p75) > 0.001:
            all_validation_failures.append(
                f"75th percentile incorrect: expected {expected_p75:.4f}, "
                f"got {metrics.p75_return:.4f}"
            )

        print(f"  ✓ 25th percentile: {metrics.p25_return:.4f}")
        print(f"  ✓ 75th percentile: {metrics.p75_return:.4f}")

    except Exception as e:
        all_validation_failures.append(f"Percentile calculation failed: {e}")

    # Test 3: Consistency score
    total_tests += 1
    print("\nTest 3: Consistency Score")
    try:
        # Consistency should be in [0, 1]
        if not (0 <= metrics.consistency_score <= 1.0):
            all_validation_failures.append(
                f"Consistency score out of range: {metrics.consistency_score}"
            )

        # Lower std/mean ratio should give higher consistency
        print(f"  ✓ Consistency score: {metrics.consistency_score:.3f}")
        print(f"    (Sharpe mean: {metrics.mean_sharpe:.2f}, std: {metrics.std_sharpe:.2f})")

    except Exception as e:
        all_validation_failures.append(f"Consistency score calculation failed: {e}")

    # Test 4: Composite score
    total_tests += 1
    print("\nTest 4: Composite Score")
    try:
        composite = aggregator.compute_composite_score(metrics)

        # Composite should be in [0, 100]
        if not (0 <= composite <= 100):
            all_validation_failures.append(
                f"Composite score out of range: {composite}"
            )

        print(f"  ✓ Composite score: {composite:.2f}")

    except Exception as e:
        all_validation_failures.append(f"Composite score calculation failed: {e}")

    # Test 5: Handle empty results
    total_tests += 1
    print("\nTest 5: Handle Empty Results")
    try:
        empty_metrics = aggregator.aggregate_windows(
            [], 'EmptyStrategy', '30d', 'test'
        )

        if empty_metrics.num_windows != 0:
            all_validation_failures.append(
                f"Empty results should have 0 windows, got {empty_metrics.num_windows}"
            )

        if empty_metrics.mean_return != 0.0:
            all_validation_failures.append(
                f"Empty results should have 0 mean return, got {empty_metrics.mean_return}"
            )

        print(f"  ✓ Empty results handled correctly")
        print(f"    Windows: {empty_metrics.num_windows}, Return: {empty_metrics.mean_return}")

    except Exception as e:
        all_validation_failures.append(f"Empty results handling failed: {e}")

    # Test 6: Summary string generation
    total_tests += 1
    print("\nTest 6: Summary String Generation")
    try:
        summary = metrics.summary_string()

        required_terms = ['TestStrategy', '30d', 'train', 'Windows', 'Return', 'Sharpe']
        missing_terms = [term for term in required_terms if term not in summary]

        if missing_terms:
            all_validation_failures.append(
                f"Summary missing required terms: {missing_terms}"
            )

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
        print("Results aggregator validated: comprehensive statistics across windows")
        print("\n📊 Aggregation Methods:")
        print("  - Mean: Simple average")
        print("  - Median: Robust central tendency")
        print("  - Std Dev: Consistency measure")
        print("  - Percentiles (25/75): Distribution shape")
        print("  - Weighted Average: Time-weighted for recent windows")
        print("  - Consistency Score: Inverse coefficient of variation")
        sys.exit(0)
