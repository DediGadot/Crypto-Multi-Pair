"""
Comparison Engine Module

Orchestrates the comparison of multiple trading strategies:
- Aggregates strategy data
- Calculates comparative metrics
- Generates correlation matrices
- Produces comparison results

Documentation:
- NumPy: https://numpy.org/doc/
- Pandas: https://pandas.pydata.org/docs/

Sample Input:
- strategies_data: Dict[str, Dict] with loaded strategy data
- time_horizon: str = "6M"

Expected Output:
- Comprehensive comparison results dictionary with:
  * equity_curves
  * metrics
  * drawdowns
  * correlations
  * returns distributions
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from loguru import logger

from crypto_strategy_comparison.metrics_calculator import MetricsCalculator
from crypto_strategy_comparison.utils import get_time_horizon_dates


class ComparisonEngine:
    """Engine for comparing multiple trading strategies."""

    def __init__(self):
        """Initialize the comparison engine."""
        self.metrics_calculator = MetricsCalculator()
        logger.info("ComparisonEngine initialized")

    def compare(
        self,
        strategies_data: Dict[str, Dict],
        time_horizon: str = "6M"
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies over a time horizon.

        Args:
            strategies_data: Dictionary of strategy data
            time_horizon: Time horizon code ("1W", "1M", "3M", "6M", "1Y", "ALL")

        Returns:
            Comprehensive comparison results dictionary
        """
        logger.info(f"Comparing {len(strategies_data)} strategies over {time_horizon}")

        if not strategies_data:
            logger.warning("No strategies provided for comparison")
            return {}

        # Filter data by time horizon
        filtered_data = self._filter_by_time_horizon(strategies_data, time_horizon)

        # Build comparison results
        results = {
            "time_horizon": time_horizon,
            "strategy_count": len(filtered_data),
            "equity_curves": self._extract_equity_curves(filtered_data),
            "metrics": self._calculate_metrics(filtered_data),
            "drawdowns": self._calculate_drawdowns(filtered_data),
            "returns": self._extract_returns(filtered_data),
            "trades": self._extract_trades(filtered_data),
            "correlation_matrix": self._calculate_correlations(filtered_data),
            "rolling_metrics": self._calculate_rolling_metrics(filtered_data),
        }

        logger.info("Comparison completed successfully")
        return results

    def _filter_by_time_horizon(
        self,
        strategies_data: Dict[str, Dict],
        time_horizon: str
    ) -> Dict[str, Dict]:
        """
        Filter strategy data by time horizon.

        Args:
            strategies_data: Full strategy data
            time_horizon: Time horizon code

        Returns:
            Filtered strategy data
        """
        start_date, end_date = get_time_horizon_dates(time_horizon)

        filtered = {}

        for strategy_name, data in strategies_data.items():
            dates = pd.to_datetime(data["dates"])

            # Find indices within time horizon
            mask = (dates >= start_date) & (dates <= end_date)
            indices = np.where(mask)[0]

            if len(indices) == 0:
                logger.warning(f"No data for {strategy_name} in time horizon {time_horizon}")
                continue

            # Filter all time-series data
            filtered[strategy_name] = {
                "name": data["name"],
                "dates": dates[mask].tolist(),
                "equity": np.array(data["equity"])[mask].tolist(),
                "returns": np.array(data["returns"])[mask].tolist(),
                "trades": self._filter_trades(data["trades"], start_date, end_date),
                "config": data["config"],
            }

        return filtered

    def _filter_trades(
        self,
        trades: List[Dict],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Filter trades by date range."""
        filtered = []

        for trade in trades:
            trade_date = pd.to_datetime(trade["entry_date"])
            if start_date <= trade_date <= end_date:
                filtered.append(trade)

        return filtered

    def _extract_equity_curves(
        self,
        strategies_data: Dict[str, Dict]
    ) -> Dict[str, Dict[str, List]]:
        """Extract equity curves for all strategies."""
        equity_curves = {}

        for strategy_name, data in strategies_data.items():
            equity_curves[strategy_name] = {
                "dates": data["dates"],
                "values": data["equity"],
            }

        return equity_curves

    def _calculate_metrics(
        self,
        strategies_data: Dict[str, Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for all strategies."""
        metrics = {}

        for strategy_name, data in strategies_data.items():
            strategy_metrics = self.metrics_calculator.calculate_all_metrics(
                data["equity"],
                data["returns"],
                data["trades"]
            )
            metrics[strategy_name] = strategy_metrics

        return metrics

    def _calculate_drawdowns(
        self,
        strategies_data: Dict[str, Dict]
    ) -> Dict[str, Dict[str, List]]:
        """Calculate drawdown series for all strategies."""
        drawdowns = {}

        for strategy_name, data in strategies_data.items():
            equity = np.array(data["equity"])
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max * 100

            drawdowns[strategy_name] = {
                "dates": data["dates"],
                "values": drawdown.tolist(),
            }

        return drawdowns

    def _extract_returns(
        self,
        strategies_data: Dict[str, Dict]
    ) -> Dict[str, List[float]]:
        """Extract return distributions for all strategies."""
        returns = {}

        for strategy_name, data in strategies_data.items():
            # Convert to percentage returns
            returns[strategy_name] = [r * 100 for r in data["returns"]]

        return returns

    def _extract_trades(
        self,
        strategies_data: Dict[str, Dict]
    ) -> Dict[str, List[Dict]]:
        """Extract trade data for all strategies."""
        trades = {}

        for strategy_name, data in strategies_data.items():
            trades[strategy_name] = data["trades"]

        return trades

    def _calculate_correlations(
        self,
        strategies_data: Dict[str, Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate return correlations between strategies."""
        if len(strategies_data) < 2:
            return {}

        # Build returns matrix
        strategy_names = list(strategies_data.keys())
        returns_dict = {}

        min_length = min(len(data["returns"]) for data in strategies_data.values())

        for strategy_name in strategy_names:
            returns = strategies_data[strategy_name]["returns"][:min_length]
            returns_dict[strategy_name] = returns

        # Calculate correlation matrix
        df = pd.DataFrame(returns_dict)
        corr_matrix = df.corr()

        # Convert to nested dictionary
        result = {}
        for strat1 in strategy_names:
            result[strat1] = {}
            for strat2 in strategy_names:
                result[strat1][strat2] = round(corr_matrix.loc[strat1, strat2], 2)

        return result

    def _calculate_rolling_metrics(
        self,
        strategies_data: Dict[str, Dict],
        window: int = 90
    ) -> Dict[str, Dict[str, Dict[str, List]]]:
        """
        Calculate rolling metrics (Sharpe, volatility, win rate).

        Args:
            strategies_data: Strategy data
            window: Rolling window size (days)

        Returns:
            Rolling metrics dictionary
        """
        rolling_metrics = {
            "sharpe": {},
            "volatility": {},
            "win_rate": {},
        }

        for strategy_name, data in strategies_data.items():
            returns = pd.Series(data["returns"])
            dates = data["dates"]

            if len(returns) < window:
                logger.warning(
                    f"Insufficient data for rolling metrics: {strategy_name} "
                    f"({len(returns)} < {window})"
                )
                continue

            # Rolling Sharpe ratio
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            rolling_sharpe = (rolling_mean * 252) / (rolling_std * np.sqrt(252))

            # Rolling volatility
            rolling_vol = rolling_std * np.sqrt(252) * 100

            # Rolling win rate (simplified)
            rolling_win_rate = (
                returns.rolling(window=window)
                .apply(lambda x: (x > 0).sum() / len(x) * 100, raw=False)
            )

            # Store results (skip NaN values)
            valid_idx = window - 1

            rolling_metrics["sharpe"][strategy_name] = {
                "dates": dates[valid_idx:],
                "values": rolling_sharpe.iloc[valid_idx:].fillna(0).tolist(),
            }

            rolling_metrics["volatility"][strategy_name] = {
                "dates": dates[valid_idx:],
                "values": rolling_vol.iloc[valid_idx:].fillna(0).tolist(),
            }

            rolling_metrics["win_rate"][strategy_name] = {
                "dates": dates[valid_idx:],
                "values": rolling_win_rate.iloc[valid_idx:].fillna(0).tolist(),
            }

        return rolling_metrics


if __name__ == "__main__":
    # Validation function
    import sys

    print("🔍 Validating comparison_engine.py...")

    all_validation_failures = []
    total_tests = 0

    # Test 1: Basic comparison
    total_tests += 1
    try:
        engine = ComparisonEngine()

        # Create mock strategy data
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        strategy_data = {
            "Strategy A": {
                "name": "Strategy A",
                "dates": dates.tolist(),
                "equity": (10000 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod()).tolist(),
                "returns": np.random.normal(0.001, 0.02, 100).tolist(),
                "trades": [],
                "config": {"asset": "BTC", "type": "Momentum"}
            }
        }

        results = engine.compare(strategy_data, time_horizon="3M")

        if not isinstance(results, dict):
            all_validation_failures.append(
                f"Compare results: Expected dict, got {type(results)}"
            )

        required_keys = [
            "time_horizon", "strategy_count", "equity_curves",
            "metrics", "drawdowns", "returns", "trades",
            "correlation_matrix", "rolling_metrics"
        ]

        for key in required_keys:
            if key not in results:
                all_validation_failures.append(
                    f"Compare results: Missing required key '{key}'"
                )

    except Exception as e:
        all_validation_failures.append(f"Basic comparison test failed: {e}")

    # Test 2: Multi-strategy comparison
    total_tests += 1
    try:
        engine = ComparisonEngine()

        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")

        strategy_data = {
            "Strategy A": {
                "name": "Strategy A",
                "dates": dates.tolist(),
                "equity": (10000 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod()).tolist(),
                "returns": np.random.normal(0.001, 0.02, 100).tolist(),
                "trades": [],
                "config": {"asset": "BTC", "type": "Momentum"}
            },
            "Strategy B": {
                "name": "Strategy B",
                "dates": dates.tolist(),
                "equity": (10000 * (1 + np.random.normal(0.0015, 0.015, 100)).cumprod()).tolist(),
                "returns": np.random.normal(0.0015, 0.015, 100).tolist(),
                "trades": [],
                "config": {"asset": "ETH", "type": "Mean Reversion"}
            }
        }

        results = engine.compare(strategy_data, time_horizon="3M")

        if results["strategy_count"] != 2:
            all_validation_failures.append(
                f"Strategy count: Expected 2, got {results['strategy_count']}"
            )

        if len(results["equity_curves"]) != 2:
            all_validation_failures.append(
                f"Equity curves: Expected 2, got {len(results['equity_curves'])}"
            )

        if len(results["metrics"]) != 2:
            all_validation_failures.append(
                f"Metrics: Expected 2, got {len(results['metrics'])}"
            )

    except Exception as e:
        all_validation_failures.append(f"Multi-strategy comparison test failed: {e}")

    # Test 3: Correlation calculation
    total_tests += 1
    try:
        engine = ComparisonEngine()

        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        returns = np.random.normal(0.001, 0.02, 100)

        strategy_data = {
            "Strategy A": {
                "name": "Strategy A",
                "dates": dates.tolist(),
                "equity": (10000 * (1 + returns).cumprod()).tolist(),
                "returns": returns.tolist(),
                "trades": [],
                "config": {}
            },
            "Strategy B": {
                "name": "Strategy B",
                "dates": dates.tolist(),
                "equity": (10000 * (1 + returns).cumprod()).tolist(),  # Same returns = correlation 1.0
                "returns": returns.tolist(),
                "trades": [],
                "config": {}
            }
        }

        results = engine.compare(strategy_data, time_horizon="3M")

        corr_matrix = results["correlation_matrix"]

        # Check diagonal (self-correlation should be 1.0)
        if abs(corr_matrix["Strategy A"]["Strategy A"] - 1.0) > 0.01:
            all_validation_failures.append(
                f"Correlation: Self-correlation should be 1.0, "
                f"got {corr_matrix['Strategy A']['Strategy A']}"
            )

        # Check correlation between identical strategies
        if abs(corr_matrix["Strategy A"]["Strategy B"] - 1.0) > 0.01:
            all_validation_failures.append(
                f"Correlation: Identical strategies should have correlation 1.0, "
                f"got {corr_matrix['Strategy A']['Strategy B']}"
            )

    except Exception as e:
        all_validation_failures.append(f"Correlation calculation test failed: {e}")

    # Test 4: Time horizon filtering
    total_tests += 1
    try:
        engine = ComparisonEngine()

        # Create 1 year of data
        dates = pd.date_range(end=datetime.now(), periods=365, freq="D")

        strategy_data = {
            "Strategy A": {
                "name": "Strategy A",
                "dates": dates.tolist(),
                "equity": (10000 * (1 + np.random.normal(0.001, 0.02, 365)).cumprod()).tolist(),
                "returns": np.random.normal(0.001, 0.02, 365).tolist(),
                "trades": [],
                "config": {}
            }
        }

        # Compare over 1 month
        results = engine.compare(strategy_data, time_horizon="1M")

        # Check that data is filtered
        equity_curve_length = len(results["equity_curves"]["Strategy A"]["values"])

        if equity_curve_length > 40:  # Should be ~30 days
            all_validation_failures.append(
                f"Time horizon filtering: Expected ~30 days for 1M, got {equity_curve_length}"
            )

    except Exception as e:
        all_validation_failures.append(f"Time horizon filtering test failed: {e}")

    # Final validation result
    if all_validation_failures:
        print(
            f"❌ VALIDATION FAILED - {len(all_validation_failures)} "
            f"of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests successful")
        print("Comparison engine is validated and ready for use")
        sys.exit(0)
