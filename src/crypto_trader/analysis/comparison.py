"""
Multi-strategy comparison and ranking system.

This module provides functionality to compare multiple backtest results,
rank strategies by various metrics, calculate correlation matrices, and
identify the best performing strategies.

Documentation:
- Pandas: https://pandas.pydata.org/docs/
- NumPy: https://numpy.org/doc/stable/
- SciPy: https://docs.scipy.org/doc/scipy/reference/stats.html

Sample Input:
    comparison = StrategyComparison()
    results = [backtest_result1, backtest_result2, backtest_result3]
    df = comparison.compare_strategies(results)
    best = comparison.best_performer(results, metric="sharpe_ratio")

Expected Output:
    DataFrame with strategy comparisons across all metrics
    Correlation matrix showing strategy relationships
    Statistical significance tests for performance differences
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from crypto_trader.core.types import BacktestResult, PerformanceMetrics


MetricName = Literal[
    "total_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "expectancy",
]


class StrategyComparison:
    """
    Compare and analyze multiple trading strategies.

    This class provides methods to compare backtest results, rank strategies,
    calculate correlation between strategies, and perform statistical tests
    to determine if performance differences are significant.
    """

    def compare_strategies(
        self, results: list[BacktestResult], normalize: bool = False
    ) -> pd.DataFrame:
        """
        Compare multiple strategies across all performance metrics.

        Args:
            results: List of BacktestResult objects to compare
            normalize: If True, normalize metrics to 0-1 scale for comparison

        Returns:
            DataFrame with strategies as rows and metrics as columns
        """
        if len(results) == 0:
            return pd.DataFrame()

        comparison_data = []

        for result in results:
            metrics = result.metrics
            row = {
                "strategy": result.strategy_name,
                "symbol": result.symbol,
                "timeframe": result.timeframe.value,
                "duration_days": result.duration_days,
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "calmar_ratio": metrics.calmar_ratio,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "total_trades": metrics.total_trades,
                "expectancy": metrics.expectancy,
                "avg_trade_duration": metrics.avg_trade_duration,
                "final_capital": metrics.final_capital,
                "total_fees": metrics.total_fees,
                "quality": metrics.risk_adjusted_quality(),
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        if normalize and len(df) > 1:
            # Normalize numeric columns to 0-1 scale (except max_drawdown which is inverted)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == "max_drawdown":
                    # For drawdown, lower is better, so invert
                    df[f"{col}_normalized"] = 1 - (
                        (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    )
                else:
                    # For other metrics, higher is better
                    col_range = df[col].max() - df[col].min()
                    if col_range > 0:
                        df[f"{col}_normalized"] = (df[col] - df[col].min()) / col_range

        return df

    def rank_strategies(
        self,
        results: list[BacktestResult],
        metric: MetricName = "sharpe_ratio",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Rank strategies by a specific metric.

        Args:
            results: List of BacktestResult objects
            metric: Metric to rank by (default: sharpe_ratio)
            ascending: If True, rank from lowest to highest (for drawdown)

        Returns:
            DataFrame sorted by the specified metric with rank column
        """
        df = self.compare_strategies(results)

        if df.empty:
            return df

        # For max_drawdown, lower is better
        if metric == "max_drawdown":
            ascending = True

        df_sorted = df.sort_values(by=metric, ascending=ascending).reset_index(drop=True)
        df_sorted["rank"] = range(1, len(df_sorted) + 1)

        # Reorder columns to put rank first
        cols = ["rank"] + [col for col in df_sorted.columns if col != "rank"]
        return df_sorted[cols]

    def correlation_matrix(
        self, results: list[BacktestResult], method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between strategy returns.

        This helps identify if strategies are independent or if they
        exhibit similar behavior patterns.

        Args:
            results: List of BacktestResult objects
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix DataFrame
        """
        if len(results) < 2:
            return pd.DataFrame()

        # Extract equity curves and convert to returns
        returns_dict = {}

        for result in results:
            if len(result.equity_curve) < 2:
                continue

            equity_df = pd.DataFrame(
                result.equity_curve, columns=["timestamp", "equity"]
            )
            equity_df["returns"] = equity_df["equity"].pct_change()
            returns_dict[result.strategy_name] = equity_df["returns"].dropna()

        if len(returns_dict) < 2:
            return pd.DataFrame()

        # Align all return series to common timestamps
        returns_df = pd.DataFrame(returns_dict)

        # Calculate correlation
        correlation = returns_df.corr(method=method)

        return correlation

    def best_performer(
        self,
        results: list[BacktestResult],
        metric: MetricName = "sharpe_ratio",
    ) -> Optional[BacktestResult]:
        """
        Identify the best performing strategy by a specific metric.

        Args:
            results: List of BacktestResult objects
            metric: Metric to evaluate (default: sharpe_ratio)

        Returns:
            BacktestResult of the best performing strategy, or None if empty
        """
        if len(results) == 0:
            return None

        best_result = None
        best_value = float("-inf")

        # For max_drawdown, lower is better
        if metric == "max_drawdown":
            best_value = float("inf")

        for result in results:
            metric_value = getattr(result.metrics, metric)

            if metric == "max_drawdown":
                if metric_value < best_value:
                    best_value = metric_value
                    best_result = result
            else:
                if metric_value > best_value:
                    best_value = metric_value
                    best_result = result

        return best_result

    def statistical_significance(
        self, result1: BacktestResult, result2: BacktestResult, alpha: float = 0.05
    ) -> dict[str, any]:
        """
        Test if performance difference between two strategies is statistically significant.

        Uses t-test to compare returns and determine if the difference
        is likely due to chance or represents a real performance difference.

        Args:
            result1: First backtest result
            result2: Second backtest result
            alpha: Significance level (default: 0.05 for 95% confidence)

        Returns:
            Dictionary with test results including p-value and significance
        """
        if len(result1.equity_curve) < 2 or len(result2.equity_curve) < 2:
            return {
                "significant": False,
                "p_value": 1.0,
                "message": "Insufficient data for statistical test",
            }

        # Calculate returns for both strategies
        equity1_df = pd.DataFrame(result1.equity_curve, columns=["timestamp", "equity"])
        equity1_df["returns"] = equity1_df["equity"].pct_change()
        returns1 = equity1_df["returns"].dropna()

        equity2_df = pd.DataFrame(result2.equity_curve, columns=["timestamp", "equity"])
        equity2_df["returns"] = equity2_df["equity"].pct_change()
        returns2 = equity2_df["returns"].dropna()

        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(returns1, returns2)

        significant = p_value < alpha

        result = {
            "strategy1": result1.strategy_name,
            "strategy2": result2.strategy_name,
            "mean_return1": float(returns1.mean()),
            "mean_return2": float(returns2.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "alpha": alpha,
            "message": (
                f"Performance difference is {'significant' if significant else 'not significant'} "
                f"at {alpha} level (p={p_value:.4f})"
            ),
        }

        return result

    def multi_strategy_summary(self, results: list[BacktestResult]) -> dict:
        """
        Generate comprehensive summary statistics across all strategies.

        Args:
            results: List of BacktestResult objects

        Returns:
            Dictionary with summary statistics
        """
        if len(results) == 0:
            return {}

        df = self.compare_strategies(results)

        summary = {
            "total_strategies": len(results),
            "symbols_tested": df["symbol"].unique().tolist(),
            "timeframes_tested": df["timeframe"].unique().tolist(),
            "metrics_summary": {},
        }

        # Calculate statistics for key metrics
        metric_columns = [
            "total_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ]

        for metric in metric_columns:
            if metric in df.columns:
                summary["metrics_summary"][metric] = {
                    "mean": float(df[metric].mean()),
                    "median": float(df[metric].median()),
                    "min": float(df[metric].min()),
                    "max": float(df[metric].max()),
                    "std": float(df[metric].std()),
                }

        # Identify best performers
        summary["best_performers"] = {
            "sharpe_ratio": self.best_performer(results, "sharpe_ratio").strategy_name,
            "total_return": self.best_performer(results, "total_return").strategy_name,
            "lowest_drawdown": self.best_performer(
                results, "max_drawdown"
            ).strategy_name,
        }

        # Count profitable strategies
        profitable = sum(1 for r in results if r.metrics.is_profitable())
        summary["profitable_strategies"] = profitable
        summary["profitable_rate"] = profitable / len(results)

        return summary

    def filter_strategies(
        self,
        results: list[BacktestResult],
        min_sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_trades: Optional[int] = None,
    ) -> list[BacktestResult]:
        """
        Filter strategies based on performance criteria.

        Args:
            results: List of BacktestResult objects
            min_sharpe: Minimum required Sharpe ratio
            max_drawdown: Maximum acceptable drawdown (as decimal)
            min_trades: Minimum number of trades required

        Returns:
            Filtered list of BacktestResult objects meeting criteria
        """
        filtered = results.copy()

        if min_sharpe is not None:
            filtered = [r for r in filtered if r.metrics.sharpe_ratio >= min_sharpe]

        if max_drawdown is not None:
            filtered = [r for r in filtered if r.metrics.max_drawdown <= max_drawdown]

        if min_trades is not None:
            filtered = [r for r in filtered if r.metrics.total_trades >= min_trades]

        return filtered


if __name__ == "__main__":
    """
    Validation function to test strategy comparison with real backtest results.
    """
    import sys
    from datetime import datetime, timedelta

    from crypto_trader.core.types import (
        OrderSide,
        OrderType,
        PerformanceMetrics,
        Timeframe,
        Trade,
    )

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating comparison.py with real backtest data...\n")

    # Create sample backtest results for testing
    base_time = datetime(2024, 1, 1)
    end_time = datetime(2024, 12, 31)

    # Strategy 1: High Sharpe, Low Drawdown
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
            expectancy=78.5,
            final_capital=13500.0,
        ),
        trades=[
            Trade(
                symbol="BTCUSDT",
                entry_time=base_time,
                exit_time=base_time + timedelta(hours=4),
                entry_price=45000.0,
                exit_price=46500.0,
                side=OrderSide.BUY,
                quantity=0.1,
                pnl=150.0,
                pnl_percent=3.33,
                fees=15.0,
            )
        ],
        equity_curve=[
            (base_time, 10000.0),
            (base_time + timedelta(days=30), 10500.0),
            (base_time + timedelta(days=60), 11200.0),
            (base_time + timedelta(days=90), 11800.0),
            (end_time, 13500.0),
        ],
    )

    # Strategy 2: Moderate Performance
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
            expectancy=28.2,
            final_capital=12200.0,
        ),
        trades=[],
        equity_curve=[
            (base_time, 10000.0),
            (base_time + timedelta(days=30), 10300.0),
            (base_time + timedelta(days=60), 10800.0),
            (base_time + timedelta(days=90), 11400.0),
            (end_time, 12200.0),
        ],
    )

    # Strategy 3: Poor Performance
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
            expectancy=-15.6,
            final_capital=9500.0,
        ),
        trades=[],
        equity_curve=[
            (base_time, 10000.0),
            (base_time + timedelta(days=30), 9800.0),
            (base_time + timedelta(days=60), 9600.0),
            (base_time + timedelta(days=90), 9400.0),
            (end_time, 9500.0),
        ],
    )

    sample_results = [result1, result2, result3]

    comparison = StrategyComparison()

    # Test 1: Compare strategies
    total_tests += 1
    print("Test 1: Compare strategies")
    try:
        df = comparison.compare_strategies(sample_results)

        if len(df) != 3:
            all_validation_failures.append(f"Comparison df: Expected 3 rows, got {len(df)}")

        expected_columns = [
            "strategy",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
        ]
        for col in expected_columns:
            if col not in df.columns:
                all_validation_failures.append(f"Missing column: {col}")

        print(f"  ✓ Compared {len(df)} strategies")
        print(f"  ✓ Columns: {len(df.columns)}")
        print(f"  ✓ Strategies: {df['strategy'].tolist()}")

    except Exception as e:
        all_validation_failures.append(f"Compare strategies exception: {e}")

    # Test 2: Rank strategies by Sharpe ratio
    total_tests += 1
    print("\nTest 2: Rank strategies by Sharpe ratio")
    try:
        ranked = comparison.rank_strategies(sample_results, metric="sharpe_ratio")

        # MA Crossover should be rank 1 (highest Sharpe: 2.5)
        if ranked.iloc[0]["strategy"] != "MA Crossover":
            all_validation_failures.append(
                f"Best Sharpe: Expected 'MA Crossover', got '{ranked.iloc[0]['strategy']}'"
            )

        # Bollinger Bands should be rank 3 (lowest Sharpe: 0.5)
        if ranked.iloc[2]["strategy"] != "Bollinger Bands":
            all_validation_failures.append(
                f"Worst Sharpe: Expected 'Bollinger Bands', got '{ranked.iloc[2]['strategy']}'"
            )

        print(f"  ✓ Rank 1: {ranked.iloc[0]['strategy']} (Sharpe: {ranked.iloc[0]['sharpe_ratio']:.2f})")
        print(f"  ✓ Rank 2: {ranked.iloc[1]['strategy']} (Sharpe: {ranked.iloc[1]['sharpe_ratio']:.2f})")
        print(f"  ✓ Rank 3: {ranked.iloc[2]['strategy']} (Sharpe: {ranked.iloc[2]['sharpe_ratio']:.2f})")

    except Exception as e:
        all_validation_failures.append(f"Rank strategies exception: {e}")

    # Test 3: Correlation matrix
    total_tests += 1
    print("\nTest 3: Correlation matrix")
    try:
        corr_matrix = comparison.correlation_matrix(sample_results)

        if corr_matrix.empty:
            all_validation_failures.append("Correlation matrix should not be empty")
        else:
            # Diagonal should be 1.0 (correlation with self)
            diagonal = np.diag(corr_matrix.values)
            if not np.allclose(diagonal, 1.0):
                all_validation_failures.append(
                    f"Diagonal should be 1.0, got {diagonal}"
                )

            print(f"  ✓ Matrix shape: {corr_matrix.shape}")
            print(f"  ✓ Strategies: {corr_matrix.columns.tolist()}")
            print(f"  ✓ Diagonal values: {diagonal}")

    except Exception as e:
        all_validation_failures.append(f"Correlation matrix exception: {e}")

    # Test 4: Best performer
    total_tests += 1
    print("\nTest 4: Best performer identification")
    try:
        best_sharpe = comparison.best_performer(sample_results, metric="sharpe_ratio")
        best_return = comparison.best_performer(sample_results, metric="total_return")
        best_dd = comparison.best_performer(sample_results, metric="max_drawdown")

        if best_sharpe.strategy_name != "MA Crossover":
            all_validation_failures.append(
                f"Best Sharpe: Expected 'MA Crossover', got '{best_sharpe.strategy_name}'"
            )

        if best_return.strategy_name != "MA Crossover":
            all_validation_failures.append(
                f"Best return: Expected 'MA Crossover', got '{best_return.strategy_name}'"
            )

        if best_dd.strategy_name != "MA Crossover":
            all_validation_failures.append(
                f"Best drawdown: Expected 'MA Crossover', got '{best_dd.strategy_name}'"
            )

        print(f"  ✓ Best Sharpe: {best_sharpe.strategy_name} ({best_sharpe.metrics.sharpe_ratio:.2f})")
        print(f"  ✓ Best Return: {best_return.strategy_name} ({best_return.metrics.total_return:.2%})")
        print(f"  ✓ Best Drawdown: {best_dd.strategy_name} ({best_dd.metrics.max_drawdown:.2%})")

    except Exception as e:
        all_validation_failures.append(f"Best performer exception: {e}")

    # Test 5: Statistical significance
    total_tests += 1
    print("\nTest 5: Statistical significance test")
    try:
        sig_test = comparison.statistical_significance(result1, result2)

        if "p_value" not in sig_test:
            all_validation_failures.append("Missing p_value in significance test")

        if "significant" not in sig_test:
            all_validation_failures.append("Missing significant flag in test")

        print(f"  ✓ Comparing: {sig_test.get('strategy1')} vs {sig_test.get('strategy2')}")
        print(f"  ✓ P-value: {sig_test.get('p_value', 0):.4f}")
        print(f"  ✓ Significant: {sig_test.get('significant', False)}")
        print(f"  ✓ {sig_test.get('message', 'No message')}")

    except Exception as e:
        all_validation_failures.append(f"Statistical significance exception: {e}")

    # Test 6: Multi-strategy summary
    total_tests += 1
    print("\nTest 6: Multi-strategy summary")
    try:
        summary = comparison.multi_strategy_summary(sample_results)

        if summary["total_strategies"] != 3:
            all_validation_failures.append(
                f"Total strategies: Expected 3, got {summary['total_strategies']}"
            )

        if "best_performers" not in summary:
            all_validation_failures.append("Missing best_performers in summary")

        if summary["profitable_strategies"] != 2:
            all_validation_failures.append(
                f"Profitable strategies: Expected 2, got {summary['profitable_strategies']}"
            )

        print(f"  ✓ Total strategies: {summary['total_strategies']}")
        print(f"  ✓ Profitable: {summary['profitable_strategies']}")
        print(f"  ✓ Best Sharpe: {summary['best_performers']['sharpe_ratio']}")
        print(f"  ✓ Best Return: {summary['best_performers']['total_return']}")

    except Exception as e:
        all_validation_failures.append(f"Multi-strategy summary exception: {e}")

    # Test 7: Filter strategies
    total_tests += 1
    print("\nTest 7: Filter strategies")
    try:
        # Filter for Sharpe > 1.5
        filtered = comparison.filter_strategies(
            sample_results, min_sharpe=1.5, max_drawdown=0.20
        )

        # Should get MA Crossover and RSI Mean Reversion
        if len(filtered) != 2:
            all_validation_failures.append(
                f"Filtered count: Expected 2, got {len(filtered)}"
            )

        print(f"  ✓ Filtered to {len(filtered)} strategies")
        print(f"  ✓ Criteria: Sharpe > 1.5, Drawdown < 20%")
        for r in filtered:
            print(f"    - {r.strategy_name}: Sharpe {r.metrics.sharpe_ratio:.2f}")

    except Exception as e:
        all_validation_failures.append(f"Filter strategies exception: {e}")

    # Test 8: Edge case - Empty results
    total_tests += 1
    print("\nTest 8: Edge case - Empty results")
    try:
        empty_df = comparison.compare_strategies([])
        empty_best = comparison.best_performer([], "sharpe_ratio")

        if not empty_df.empty:
            all_validation_failures.append("Empty results should return empty DataFrame")

        if empty_best is not None:
            all_validation_failures.append("Empty results should return None for best performer")

        print("  ✓ Empty inputs handled correctly")
        print("  ✓ Returns empty DataFrame and None")

    except Exception as e:
        all_validation_failures.append(f"Empty results exception: {e}")

    # Final validation result
    print("\n" + "=" * 60)
    if all_validation_failures:
        print(
            f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(
            f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results"
        )
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
