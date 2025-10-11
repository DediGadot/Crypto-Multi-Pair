"""
Performance metrics calculator for trading strategies.

This module calculates comprehensive performance metrics including risk-adjusted
returns, drawdowns, win rates, and statistical measures for backtesting results.

Documentation:
- NumPy: https://numpy.org/doc/stable/
- Pandas: https://pandas.pydata.org/docs/
- SciPy Stats: https://docs.scipy.org/doc/scipy/reference/stats.html

Sample Input:
    calculator = MetricsCalculator(risk_free_rate=0.02)
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    trades = [trade1, trade2, trade3]  # List of Trade objects
    metrics = calculator.calculate_all_metrics(returns, trades, equity_curve)

Expected Output:
    PerformanceMetrics object with all calculated metrics including:
    - sharpe_ratio: 1.45
    - sortino_ratio: 1.82
    - max_drawdown: 0.15
    - win_rate: 0.60
    - profit_factor: 1.75
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from crypto_trader.core.types import PerformanceMetrics, Trade


class MetricsCalculator:
    """
    Calculates comprehensive performance metrics for trading strategies.

    This class provides methods to compute risk-adjusted returns, drawdown metrics,
    trade statistics, and other performance indicators used to evaluate strategy quality.

    Attributes:
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations (default: 0.02)
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(
        self,
        returns: pd.Series,
        trades: list[Trade],
        equity_curve: list[tuple],
        initial_capital: float,
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics from returns and trades.

        Args:
            returns: Series of period returns (e.g., daily returns)
            trades: List of completed Trade objects
            equity_curve: List of (timestamp, equity_value) tuples
            initial_capital: Starting capital amount

        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if len(returns) == 0:
            return PerformanceMetrics()

        # Convert equity curve to pandas for easier calculations
        if len(equity_curve) > 0:
            equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
            final_capital = equity_df["equity"].iloc[-1]
        else:
            final_capital = initial_capital

        # Basic return metrics
        total_return = (final_capital - initial_capital) / initial_capital

        # Risk-adjusted metrics
        sharpe = self.sharpe_ratio(returns, self.risk_free_rate)
        sortino = self.sortino_ratio(returns, self.risk_free_rate)

        # Drawdown metrics
        max_dd = self.max_drawdown(equity_curve)
        calmar = self.calmar_ratio(total_return, max_dd)
        recovery = self.recovery_factor(total_return, max_dd, initial_capital, final_capital)

        # Trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.is_winning)
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        profit_factor = self.profit_factor(trades)

        # Win/loss analysis
        avg_win, avg_loss = self.average_win_loss(trades)
        max_cons_wins, max_cons_losses = self.consecutive_wins_losses(trades)

        # Trade duration
        avg_duration = self.average_trade_duration(trades)

        # Expectancy
        expectancy = self.expectancy(trades)

        # Total fees
        total_fees = sum(t.fees for t in trades)

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_cons_wins,
            max_consecutive_losses=max_cons_losses,
            avg_trade_duration=avg_duration,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            recovery_factor=recovery,
            expectancy=expectancy,
            total_fees=total_fees,
            final_capital=final_capital,
        )

    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """
        Calculate Sharpe ratio - risk-adjusted return metric.

        Formula: (mean_return - risk_free_rate) / std_dev_return

        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio (higher is better, >1 is good, >2 is excellent)
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Convert annual risk-free rate to period rate
        periods_per_year = 252  # Trading days
        period_rf_rate = risk_free_rate / periods_per_year

        excess_returns = returns - period_rf_rate
        sharpe = excess_returns.mean() / returns.std()

        # Annualize the Sharpe ratio
        return sharpe * np.sqrt(periods_per_year)

    def sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """
        Calculate Sortino ratio - downside risk-adjusted return.

        Only considers downside volatility (negative returns) in the denominator,
        making it more appropriate than Sharpe for asymmetric return distributions.

        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio (higher is better)
        """
        if len(returns) == 0:
            return 0.0

        periods_per_year = 252
        period_rf_rate = risk_free_rate / periods_per_year

        excess_returns = returns - period_rf_rate
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_returns.std()
        return sortino * np.sqrt(periods_per_year)

    def max_drawdown(self, equity_curve: list[tuple]) -> float:
        """
        Calculate maximum drawdown - largest peak-to-trough decline.

        Drawdown represents the maximum loss from a peak to a subsequent trough
        before a new peak is achieved.

        Args:
            equity_curve: List of (timestamp, equity_value) tuples

        Returns:
            Maximum drawdown as a positive decimal (e.g., 0.20 for 20% drawdown)
        """
        if len(equity_curve) == 0:
            return 0.0

        equity_values = np.array([equity for _, equity in equity_curve])

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_values)

        # Calculate drawdown at each point
        drawdowns = (running_max - equity_values) / running_max

        return float(np.max(drawdowns))

    def profit_factor(self, trades: list[Trade]) -> float:
        """
        Calculate profit factor - ratio of gross profit to gross loss.

        Formula: sum(winning_trades) / abs(sum(losing_trades))
        A profit factor > 1.0 indicates profitability.

        Args:
            trades: List of completed trades

        Returns:
            Profit factor (>1 is profitable, >2 is excellent)
        """
        if len(trades) == 0:
            return 0.0

        gross_profit = sum(t.pnl for t in trades if t.is_winning)
        gross_loss = abs(sum(t.pnl for t in trades if not t.is_winning))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def average_win_loss(self, trades: list[Trade]) -> tuple[float, float]:
        """
        Calculate average win and average loss amounts.

        Args:
            trades: List of completed trades

        Returns:
            Tuple of (average_win, average_loss)
            average_loss is returned as a negative value
        """
        if len(trades) == 0:
            return (0.0, 0.0)

        winning_trades = [t.pnl for t in trades if t.is_winning]
        losing_trades = [t.pnl for t in trades if not t.is_winning]

        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0

        return (float(avg_win), float(avg_loss))

    def consecutive_wins_losses(self, trades: list[Trade]) -> tuple[int, int]:
        """
        Calculate maximum consecutive wins and losses.

        Args:
            trades: List of completed trades

        Returns:
            Tuple of (max_consecutive_wins, max_consecutive_losses)
        """
        if len(trades) == 0:
            return (0, 0)

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.is_winning:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return (max_wins, max_losses)

    def average_trade_duration(self, trades: list[Trade]) -> float:
        """
        Calculate average trade duration in minutes.

        Args:
            trades: List of completed trades

        Returns:
            Average duration in minutes
        """
        if len(trades) == 0:
            return 0.0

        durations = [t.duration_minutes for t in trades]
        return float(np.mean(durations))

    def calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio - return divided by max drawdown.

        This ratio shows how much return is generated per unit of drawdown risk.

        Args:
            total_return: Total return as decimal (e.g., 0.25 for 25%)
            max_drawdown: Maximum drawdown as positive decimal

        Returns:
            Calmar ratio (higher is better, >3 is excellent)
        """
        if max_drawdown == 0:
            return 0.0
        return total_return / max_drawdown

    def recovery_factor(
        self,
        total_return: float,
        max_drawdown: float,
        initial_capital: float,
        final_capital: float,
    ) -> float:
        """
        Calculate recovery factor - net profit divided by max drawdown.

        Args:
            total_return: Total return as decimal
            max_drawdown: Maximum drawdown as positive decimal
            initial_capital: Starting capital
            final_capital: Ending capital

        Returns:
            Recovery factor (higher is better)
        """
        if max_drawdown == 0:
            return 0.0

        net_profit = final_capital - initial_capital
        max_dd_dollars = initial_capital * max_drawdown

        if max_dd_dollars == 0:
            return 0.0

        return net_profit / max_dd_dollars

    def expectancy(self, trades: list[Trade]) -> float:
        """
        Calculate expectancy - average expected profit per trade.

        Formula: (win_rate * avg_win) - (loss_rate * abs(avg_loss))

        Args:
            trades: List of completed trades

        Returns:
            Expected profit per trade in dollars
        """
        if len(trades) == 0:
            return 0.0

        total_trades = len(trades)
        winning_trades = [t for t in trades if t.is_winning]
        losing_trades = [t for t in trades if not t.is_winning]

        win_rate = len(winning_trades) / total_trades
        loss_rate = len(losing_trades) / total_trades

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0

        expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
        return float(expectancy)

    def calculate_returns_from_equity(self, equity_curve: list[tuple]) -> pd.Series:
        """
        Calculate period returns from equity curve.

        Args:
            equity_curve: List of (timestamp, equity_value) tuples

        Returns:
            Pandas Series of returns
        """
        if len(equity_curve) < 2:
            return pd.Series()

        equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
        equity_df["returns"] = equity_df["equity"].pct_change()

        return equity_df["returns"].dropna()


if __name__ == "__main__":
    """
    Validation function to test metrics calculator with real trading data.
    """
    import sys
    from datetime import datetime, timedelta

    from crypto_trader.core.types import OrderSide, OrderType, Trade

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating metrics.py with real trading data...\n")

    # Create sample trades for testing
    base_time = datetime(2025, 1, 1, 10, 0, 0)

    sample_trades = [
        # Winning trades
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
            order_type=OrderType.MARKET,
        ),
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=5),
            exit_time=base_time + timedelta(hours=9),
            entry_price=46500.0,
            exit_price=47200.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=70.0,
            pnl_percent=1.51,
            fees=10.0,
            order_type=OrderType.MARKET,
        ),
        # Losing trades
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=10),
            exit_time=base_time + timedelta(hours=12),
            entry_price=47200.0,
            exit_price=46800.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=-40.0,
            pnl_percent=-0.85,
            fees=10.0,
            order_type=OrderType.MARKET,
        ),
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=13),
            exit_time=base_time + timedelta(hours=17),
            entry_price=46800.0,
            exit_price=47400.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=60.0,
            pnl_percent=1.28,
            fees=12.0,
            order_type=OrderType.MARKET,
        ),
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(hours=18),
            exit_time=base_time + timedelta(hours=20),
            entry_price=47400.0,
            exit_price=46900.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=-50.0,
            pnl_percent=-1.05,
            fees=10.0,
            order_type=OrderType.MARKET,
        ),
    ]

    # Create sample equity curve
    initial_capital = 10000.0
    equity_curve = [
        (base_time, 10000.0),
        (base_time + timedelta(hours=4), 10135.0),  # +150 - 15 fees
        (base_time + timedelta(hours=9), 10195.0),  # +70 - 10 fees
        (base_time + timedelta(hours=12), 10145.0),  # -40 - 10 fees
        (base_time + timedelta(hours=17), 10193.0),  # +60 - 12 fees
        (base_time + timedelta(hours=20), 10133.0),  # -50 - 10 fees
    ]

    # Calculate returns from equity curve
    calculator = MetricsCalculator(risk_free_rate=0.02)
    returns = calculator.calculate_returns_from_equity(equity_curve)

    # Test 1: Calculate all metrics
    total_tests += 1
    print("Test 1: Calculate all metrics")
    try:
        metrics = calculator.calculate_all_metrics(
            returns=returns,
            trades=sample_trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
        )

        # Verify basic metrics
        if metrics.total_trades != 5:
            all_validation_failures.append(
                f"Total trades: Expected 5, got {metrics.total_trades}"
            )
        if metrics.winning_trades != 3:
            all_validation_failures.append(
                f"Winning trades: Expected 3, got {metrics.winning_trades}"
            )
        if metrics.losing_trades != 2:
            all_validation_failures.append(
                f"Losing trades: Expected 2, got {metrics.losing_trades}"
            )

        expected_win_rate = 0.6  # 3 out of 5
        if abs(metrics.win_rate - expected_win_rate) > 0.01:
            all_validation_failures.append(
                f"Win rate: Expected {expected_win_rate}, got {metrics.win_rate}"
            )

        print(f"  ✓ Total trades: {metrics.total_trades}")
        print(f"  ✓ Win rate: {metrics.win_rate:.2%}")
        print(f"  ✓ Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  ✓ Max drawdown: {metrics.max_drawdown:.2%}")
        print(f"  ✓ Final capital: ${metrics.final_capital:,.2f}")

    except Exception as e:
        all_validation_failures.append(f"Calculate all metrics exception: {e}")

    # Test 2: Sharpe ratio calculation
    total_tests += 1
    print("\nTest 2: Sharpe ratio calculation")
    try:
        test_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        sharpe = calculator.sharpe_ratio(test_returns, 0.02)

        # Sharpe should be positive for positive average returns
        if sharpe <= 0:
            all_validation_failures.append(
                f"Sharpe ratio should be positive for positive returns, got {sharpe}"
            )

        print(f"  ✓ Sharpe ratio: {sharpe:.4f}")
        print(f"  ✓ Returns mean: {test_returns.mean():.4f}")
        print(f"  ✓ Returns std: {test_returns.std():.4f}")

    except Exception as e:
        all_validation_failures.append(f"Sharpe ratio exception: {e}")

    # Test 3: Max drawdown calculation
    total_tests += 1
    print("\nTest 3: Max drawdown calculation")
    try:
        # Equity curve with known drawdown
        test_equity = [
            (base_time, 10000.0),
            (base_time + timedelta(hours=1), 10500.0),  # Peak
            (base_time + timedelta(hours=2), 9500.0),  # Trough (9.52% drawdown)
            (base_time + timedelta(hours=3), 10000.0),  # Recovery
        ]

        max_dd = calculator.max_drawdown(test_equity)
        expected_dd = 0.0952  # (10500 - 9500) / 10500 = 0.0952

        if abs(max_dd - expected_dd) > 0.01:
            all_validation_failures.append(
                f"Max drawdown: Expected {expected_dd:.4f}, got {max_dd:.4f}"
            )

        print(f"  ✓ Max drawdown: {max_dd:.2%}")
        print(f"  ✓ Peak equity: $10,500")
        print(f"  ✓ Trough equity: $9,500")

    except Exception as e:
        all_validation_failures.append(f"Max drawdown exception: {e}")

    # Test 4: Profit factor calculation
    total_tests += 1
    print("\nTest 4: Profit factor calculation")
    try:
        profit_factor = calculator.profit_factor(sample_trades)

        # Calculate expected profit factor
        gross_profit = 150.0 + 70.0 + 60.0  # 280
        gross_loss = 40.0 + 50.0  # 90
        expected_pf = gross_profit / gross_loss  # 3.111

        if abs(profit_factor - expected_pf) > 0.1:
            all_validation_failures.append(
                f"Profit factor: Expected {expected_pf:.2f}, got {profit_factor:.2f}"
            )

        print(f"  ✓ Profit factor: {profit_factor:.2f}")
        print(f"  ✓ Gross profit: ${gross_profit:.2f}")
        print(f"  ✓ Gross loss: ${gross_loss:.2f}")

    except Exception as e:
        all_validation_failures.append(f"Profit factor exception: {e}")

    # Test 5: Consecutive wins/losses
    total_tests += 1
    print("\nTest 5: Consecutive wins and losses")
    try:
        max_wins, max_losses = calculator.consecutive_wins_losses(sample_trades)

        # From sample_trades: W, W, L, W, L
        expected_max_wins = 2
        expected_max_losses = 1

        if max_wins != expected_max_wins:
            all_validation_failures.append(
                f"Max consecutive wins: Expected {expected_max_wins}, got {max_wins}"
            )
        if max_losses != expected_max_losses:
            all_validation_failures.append(
                f"Max consecutive losses: Expected {expected_max_losses}, got {max_losses}"
            )

        print(f"  ✓ Max consecutive wins: {max_wins}")
        print(f"  ✓ Max consecutive losses: {max_losses}")

    except Exception as e:
        all_validation_failures.append(f"Consecutive wins/losses exception: {e}")

    # Test 6: Average trade duration
    total_tests += 1
    print("\nTest 6: Average trade duration")
    try:
        avg_duration = calculator.average_trade_duration(sample_trades)

        # Expected: 4, 4, 2, 4, 2 hours = 240, 240, 120, 240, 120 minutes
        expected_avg = (240 + 240 + 120 + 240 + 120) / 5  # 192 minutes

        if abs(avg_duration - expected_avg) > 1.0:
            all_validation_failures.append(
                f"Average duration: Expected {expected_avg:.1f}, got {avg_duration:.1f}"
            )

        print(f"  ✓ Average duration: {avg_duration:.1f} minutes ({avg_duration/60:.1f} hours)")

    except Exception as e:
        all_validation_failures.append(f"Average duration exception: {e}")

    # Test 7: Expectancy calculation
    total_tests += 1
    print("\nTest 7: Expectancy calculation")
    try:
        expectancy = calculator.expectancy(sample_trades)

        # Expected: (0.6 * 93.33) - (0.4 * 45) = 56 - 18 = 38
        # Avg win: (150 + 70 + 60) / 3 = 93.33
        # Avg loss: (40 + 50) / 2 = 45

        if expectancy <= 0:
            all_validation_failures.append(
                f"Expectancy should be positive for profitable strategy, got {expectancy}"
            )

        print(f"  ✓ Expectancy: ${expectancy:.2f} per trade")
        print(f"  ✓ This means on average, expect ${expectancy:.2f} profit per trade")

    except Exception as e:
        all_validation_failures.append(f"Expectancy exception: {e}")

    # Test 8: Edge case - Empty inputs
    total_tests += 1
    print("\nTest 8: Edge case - Empty inputs")
    try:
        empty_metrics = calculator.calculate_all_metrics(
            returns=pd.Series(),
            trades=[],
            equity_curve=[],
            initial_capital=10000.0,
        )

        if empty_metrics.total_trades != 0:
            all_validation_failures.append(
                f"Empty trades should result in 0 total_trades, got {empty_metrics.total_trades}"
            )
        if empty_metrics.sharpe_ratio != 0.0:
            all_validation_failures.append(
                f"Empty returns should result in 0 sharpe_ratio, got {empty_metrics.sharpe_ratio}"
            )

        print("  ✓ Empty inputs handled correctly")
        print(f"  ✓ Returns PerformanceMetrics with zeros")

    except Exception as e:
        all_validation_failures.append(f"Empty inputs exception: {e}")

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
