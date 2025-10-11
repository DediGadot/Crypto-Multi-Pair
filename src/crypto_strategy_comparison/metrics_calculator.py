"""
Metrics Calculator Module

Calculates performance metrics for trading strategies:
- Return metrics (total return, CAGR, volatility)
- Risk metrics (max drawdown, VaR, CVaR)
- Risk-adjusted returns (Sharpe, Sortino, Calmar, Omega)
- Trade statistics (win rate, profit factor, average trade)

Documentation:
- NumPy: https://numpy.org/doc/
- Pandas: https://pandas.pydata.org/docs/

Sample Input:
- equity: List[float] = [10000, 10100, 10250, ...]
- returns: List[float] = [0.01, 0.015, -0.005, ...]
- trades: List[Dict] = [{...}, {...}, ...]

Expected Output:
- Dictionary of calculated metrics
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from loguru import logger


class MetricsCalculator:
    """Calculate performance metrics for trading strategies."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"MetricsCalculator initialized with risk_free_rate={risk_free_rate}")

    def calculate_all_metrics(
        self,
        equity: List[float],
        returns: List[float],
        trades: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Args:
            equity: List of portfolio values over time
            returns: List of period returns
            trades: List of trade dictionaries

        Returns:
            Dictionary of calculated metrics
        """
        logger.info("Calculating all metrics...")

        metrics = {}

        # Return metrics
        metrics.update(self._calculate_return_metrics(equity, returns))

        # Risk metrics
        metrics.update(self._calculate_risk_metrics(equity, returns))

        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))

        # Trade statistics
        if trades:
            metrics.update(self._calculate_trade_statistics(trades))

        logger.info(f"Calculated {len(metrics)} metrics")
        return metrics

    def _calculate_return_metrics(
        self,
        equity: List[float],
        returns: List[float]
    ) -> Dict[str, float]:
        """Calculate return-based metrics."""
        if not equity or not returns:
            return {}

        total_return = ((equity[-1] / equity[0]) - 1) * 100

        # CAGR (Compound Annual Growth Rate)
        days = len(equity)
        years = days / 365.0
        if years > 0 and equity[0] > 0:
            cagr = (pow(equity[-1] / equity[0], 1 / years) - 1) * 100
        else:
            cagr = 0.0

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) * 100  # Assuming daily returns

        # Downside deviation
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_deviation = np.std(negative_returns) * np.sqrt(252) * 100
        else:
            downside_deviation = 0.0

        return {
            "total_return": round(total_return, 2),
            "cagr": round(cagr, 2),
            "volatility": round(volatility, 2),
            "downside_deviation": round(downside_deviation, 2),
        }

    def _calculate_risk_metrics(
        self,
        equity: List[float],
        returns: List[float]
    ) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        if not equity or not returns:
            return {}

        # Maximum drawdown
        max_dd, max_dd_duration = self._calculate_max_drawdown(equity)

        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns, 5) * 100

        # Conditional VaR (CVaR) - expected shortfall
        var_threshold = np.percentile(returns, 5)
        tail_losses = [r for r in returns if r <= var_threshold]
        cvar_95 = np.mean(tail_losses) * 100 if tail_losses else 0.0

        return {
            "max_drawdown": round(max_dd, 2),
            "max_dd_duration": int(max_dd_duration),
            "var_95": round(var_95, 2),
            "cvar_95": round(cvar_95, 2),
        }

    def _calculate_max_drawdown(self, equity: List[float]) -> tuple[float, float]:
        """
        Calculate maximum drawdown and duration.

        Args:
            equity: List of portfolio values

        Returns:
            Tuple of (max_drawdown_pct, duration_days)
        """
        if not equity:
            return 0.0, 0.0

        equity_arr = np.array(equity)
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - running_max) / running_max * 100

        max_dd = np.min(drawdown)

        # Calculate duration
        duration = 0
        current_duration = 0
        for i in range(len(equity_arr)):
            if equity_arr[i] < running_max[i]:
                current_duration += 1
                duration = max(duration, current_duration)
            else:
                current_duration = 0

        return max_dd, duration

    def _calculate_risk_adjusted_metrics(
        self,
        returns: List[float]
    ) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        if not returns:
            return {}

        returns_arr = np.array(returns)

        # Sharpe Ratio (annualized)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr)
        if std_return > 0:
            sharpe = (mean_return - self.risk_free_rate / 252) / std_return * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino Ratio (uses downside deviation)
        negative_returns = returns_arr[returns_arr < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                sortino = (mean_return - self.risk_free_rate / 252) / downside_std * np.sqrt(252)
            else:
                sortino = 0.0
        else:
            sortino = sharpe  # If no negative returns, use Sharpe

        # Calmar Ratio
        cagr = (pow(1 + mean_return, 252) - 1) * 100  # Annualized
        max_dd, _ = self._calculate_max_drawdown(
            (1 + returns_arr).cumprod().tolist()
        )
        if max_dd < 0:
            calmar = abs(cagr / max_dd)
        else:
            calmar = 0.0

        # Omega Ratio (gain-loss ratio above threshold)
        threshold = 0.0
        gains = returns_arr[returns_arr > threshold].sum()
        losses = abs(returns_arr[returns_arr < threshold].sum())
        if losses > 0:
            omega = gains / losses
        else:
            omega = 0.0 if gains == 0 else 999.0  # Infinite if no losses

        return {
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "omega_ratio": round(omega, 2),
        }

    def _calculate_trade_statistics(
        self,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate trade-level statistics."""
        if not trades:
            return {}

        pnl_values = [t.get("pnl_pct", 0) for t in trades]

        # Win rate
        winning_trades = sum(1 for pnl in pnl_values if pnl > 0)
        win_rate = (winning_trades / len(trades)) * 100

        # Profit factor
        gross_profit = sum(pnl for pnl in pnl_values if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnl_values if pnl < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 999.0 if gross_profit > 0 else 0.0

        # Average trade
        avg_trade = np.mean(pnl_values)

        # Average win / average loss
        winning_pnl = [pnl for pnl in pnl_values if pnl > 0]
        losing_pnl = [pnl for pnl in pnl_values if pnl < 0]

        avg_win = np.mean(winning_pnl) if winning_pnl else 0.0
        avg_loss = np.mean(losing_pnl) if losing_pnl else 0.0

        # Max consecutive wins/losses
        max_consec_wins = self._max_consecutive(pnl_values, positive=True)
        max_consec_losses = self._max_consecutive(pnl_values, positive=False)

        return {
            "trade_count": len(trades),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_trade": round(avg_trade, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
        }

    def _max_consecutive(self, values: List[float], positive: bool = True) -> int:
        """
        Calculate maximum consecutive wins or losses.

        Args:
            values: List of PnL values
            positive: If True, count wins; if False, count losses

        Returns:
            Maximum consecutive count
        """
        max_count = 0
        current_count = 0

        for val in values:
            if (positive and val > 0) or (not positive and val < 0):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count


if __name__ == "__main__":
    # Validation function
    import sys

    print("🔍 Validating metrics_calculator.py...")

    all_validation_failures = []
    total_tests = 0

    # Test 1: Return metrics calculation
    total_tests += 1
    try:
        calculator = MetricsCalculator()

        # Simple test case: 10% gain
        equity = [10000, 11000]
        returns = [0.1]

        metrics = calculator._calculate_return_metrics(equity, returns)

        if "total_return" not in metrics:
            all_validation_failures.append("Return metrics: Missing 'total_return'")

        if abs(metrics["total_return"] - 10.0) > 0.1:
            all_validation_failures.append(
                f"Return metrics: Expected ~10% return, got {metrics['total_return']}%"
            )

    except Exception as e:
        all_validation_failures.append(f"Return metrics test failed: {e}")

    # Test 2: Max drawdown calculation
    total_tests += 1
    try:
        calculator = MetricsCalculator()

        # Equity curve with 20% drawdown
        equity = [10000, 11000, 9000, 8000, 9500, 10000]  # Peak 11k, trough 8k = -27.3%

        max_dd, duration = calculator._calculate_max_drawdown(equity)

        if max_dd >= 0:
            all_validation_failures.append(
                f"Max drawdown: Expected negative value, got {max_dd}"
            )

        if not (-30 <= max_dd <= -25):  # Allow some tolerance
            all_validation_failures.append(
                f"Max drawdown: Expected ~-27%, got {max_dd}%"
            )

    except Exception as e:
        all_validation_failures.append(f"Max drawdown test failed: {e}")

    # Test 3: Sharpe ratio calculation
    total_tests += 1
    try:
        calculator = MetricsCalculator(risk_free_rate=0.0)  # Simplified

        # Positive returns with low volatility = high Sharpe
        returns = [0.01] * 100  # Consistent 1% returns

        metrics = calculator._calculate_risk_adjusted_metrics(returns)

        if "sharpe_ratio" not in metrics:
            all_validation_failures.append("Risk-adjusted: Missing 'sharpe_ratio'")

        # With consistent returns, Sharpe should be very high
        if metrics["sharpe_ratio"] <= 0:
            all_validation_failures.append(
                f"Sharpe ratio: Expected positive value, got {metrics['sharpe_ratio']}"
            )

    except Exception as e:
        all_validation_failures.append(f"Sharpe ratio test failed: {e}")

    # Test 4: Trade statistics
    total_tests += 1
    try:
        calculator = MetricsCalculator()

        # 7 winning trades, 3 losing trades = 70% win rate
        trades = [
            {"pnl_pct": 5.0},
            {"pnl_pct": 3.0},
            {"pnl_pct": -2.0},
            {"pnl_pct": 4.0},
            {"pnl_pct": 6.0},
            {"pnl_pct": -1.0},
            {"pnl_pct": 2.0},
            {"pnl_pct": 5.0},
            {"pnl_pct": -3.0},
            {"pnl_pct": 4.0},
        ]

        metrics = calculator._calculate_trade_statistics(trades)

        if "win_rate" not in metrics:
            all_validation_failures.append("Trade stats: Missing 'win_rate'")

        expected_win_rate = 70.0  # 7 out of 10
        if abs(metrics["win_rate"] - expected_win_rate) > 0.1:
            all_validation_failures.append(
                f"Win rate: Expected {expected_win_rate}%, got {metrics['win_rate']}%"
            )

        if "profit_factor" not in metrics:
            all_validation_failures.append("Trade stats: Missing 'profit_factor'")

    except Exception as e:
        all_validation_failures.append(f"Trade statistics test failed: {e}")

    # Test 5: Complete metrics calculation
    total_tests += 1
    try:
        calculator = MetricsCalculator()

        # Generate simple test data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # One year of daily returns
        equity = 10000 * (1 + returns).cumprod()

        trades = [{"pnl_pct": r * 100} for r in returns[:100]]

        all_metrics = calculator.calculate_all_metrics(
            equity.tolist(),
            returns.tolist(),
            trades
        )

        required_metrics = [
            "total_return", "sharpe_ratio", "max_drawdown",
            "win_rate", "trade_count"
        ]

        for metric in required_metrics:
            if metric not in all_metrics:
                all_validation_failures.append(
                    f"Complete metrics: Missing required metric '{metric}'"
                )

        if all_metrics["trade_count"] != 100:
            all_validation_failures.append(
                f"Trade count: Expected 100, got {all_metrics['trade_count']}"
            )

    except Exception as e:
        all_validation_failures.append(f"Complete metrics test failed: {e}")

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
        print("Metrics calculator is validated and ready for use")
        sys.exit(0)
