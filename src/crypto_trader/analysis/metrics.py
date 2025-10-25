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


# BUGFIX (Phase 1 - Task 1.4.1): Timeframe to annualization factors mapping
TIMEFRAME_TO_PERIODS = {
    '1m': 525600,    # 365 * 24 * 60
    '5m': 105120,    # 365 * 24 * 12
    '15m': 35040,    # 365 * 24 * 4
    '30m': 17520,    # 365 * 24 * 2
    '1h': 8760,      # 365 * 24
    '2h': 4380,      # 365 * 12
    '4h': 2190,      # 365 * 6
    '1d': 365,       # 365 * 1
    '1w': 52,        # ~52 weeks per year
    '1M': 12,        # 12 months per year
}


def detect_timeframe_periods(data: Optional[pd.DataFrame] = None, timeframe: Optional[str] = None) -> int:
    """
    Detect the number of periods per year based on timeframe or data.

    BUGFIX (Phase 1 - Task 1.4.1): Added timeframe detection to fix hardcoded 252 periods/year.

    Args:
        data: DataFrame with timestamp column or DatetimeIndex
        timeframe: Timeframe string (e.g., '1h', '4h', '1d')

    Returns:
        Number of periods per year for annualization

    Raises:
        ValueError: If neither data nor timeframe is provided
    """
    # First, try explicit timeframe parameter
    if timeframe is not None:
        if timeframe in TIMEFRAME_TO_PERIODS:
            return TIMEFRAME_TO_PERIODS[timeframe]
        # Try case-insensitive match
        timeframe_lower = timeframe.lower()
        for tf_key, periods in TIMEFRAME_TO_PERIODS.items():
            if tf_key.lower() == timeframe_lower:
                return periods

    # Second, try to infer from data timestamps
    if data is not None and len(data) > 1:
        if isinstance(data.index, pd.DatetimeIndex):
            timestamps = data.index
        elif 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
        else:
            # No timestamp info available, use daily default
            return 365

        # Calculate median time delta
        deltas = timestamps.diff().dropna()
        if len(deltas) > 0:
            median_delta = deltas.median()
            # Convert to minutes
            minutes = median_delta.total_seconds() / 60

            # Match to closest standard timeframe
            if minutes <= 1.5:
                return TIMEFRAME_TO_PERIODS['1m']
            elif minutes <= 7.5:
                return TIMEFRAME_TO_PERIODS['5m']
            elif minutes <= 22.5:
                return TIMEFRAME_TO_PERIODS['15m']
            elif minutes <= 45:
                return TIMEFRAME_TO_PERIODS['30m']
            elif minutes <= 90:
                return TIMEFRAME_TO_PERIODS['1h']
            elif minutes <= 180:
                return TIMEFRAME_TO_PERIODS['2h']
            elif minutes <= 360:
                return TIMEFRAME_TO_PERIODS['4h']
            elif minutes <= 1440:  # 24 hours
                return TIMEFRAME_TO_PERIODS['1d']
            elif minutes <= 10080:  # 7 days
                return TIMEFRAME_TO_PERIODS['1w']
            else:
                return TIMEFRAME_TO_PERIODS['1M']

    # Default to daily if no information available
    return 365


class MetricsCalculator:
    """
    Calculates comprehensive performance metrics for trading strategies.

    This class provides methods to compute risk-adjusted returns, drawdown metrics,
    trade statistics, and other performance indicators used to evaluate strategy quality.

    BUGFIX (Phase 1 - Task 1.4): Added timeframe support to fix annualization bugs.

    Attributes:
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations (default: 0.02)
        timeframe: Timeframe string (e.g., '1h', '4h', '1d') for correct annualization
        periods_per_year: Number of periods per year (auto-detected from timeframe)
    """

    def __init__(self, risk_free_rate: float = 0.02, timeframe: Optional[str] = None):
        """
        Initialize the metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
            timeframe: Timeframe string for annualization (e.g., '1h', '4h', '1d')
        """
        self.risk_free_rate = risk_free_rate
        self.timeframe = timeframe
        self.periods_per_year = detect_timeframe_periods(timeframe=timeframe) if timeframe else None

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

        # Advanced risk metrics
        var_95 = self.value_at_risk(returns, confidence=0.95)
        cvar_95 = self.conditional_var(returns, confidence=0.95)
        skew = self.skewness(returns)
        kurt = self.kurtosis(returns)

        # Information ratio (vs cash benchmark by default)
        info_ratio = self.information_ratio(returns, benchmark_returns=None)

        # PHASE 2: Advanced risk metrics
        omega = self.omega_ratio(returns, threshold=0.0)
        tail_r = self.tail_ratio(returns)
        max_consec_dd = self.max_consecutive_drawdown_days(equity_curve)
        ulcer = self.ulcer_index(equity_curve)

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
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            skewness=skew,
            kurtosis=kurt,
            information_ratio=info_ratio,
            # PHASE 2: Advanced risk metrics
            omega_ratio=omega,
            tail_ratio=tail_r,
            max_consecutive_drawdown_days=max_consec_dd,
            ulcer_index=ulcer,
        )

    def sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float,
        data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate Sharpe ratio - risk-adjusted return metric.

        BUGFIX (Phase 1 - Task 1.4.2): Fixed hardcoded 252 periods/year.
        Now detects timeframe from self.periods_per_year or data timestamps.

        Formula: (mean_return - risk_free_rate) / std_dev_return

        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free rate
            data: Optional DataFrame to detect timeframe if not set in constructor

        Returns:
            Sharpe ratio (higher is better, >1 is good, >2 is excellent)
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # BUGFIX (Phase 1 - Task 1.4.2): Use detected periods instead of hardcoded 252
        if self.periods_per_year is not None:
            periods_per_year = self.periods_per_year
        else:
            periods_per_year = detect_timeframe_periods(data=data)

        # Convert annual risk-free rate to period rate
        period_rf_rate = risk_free_rate / periods_per_year

        excess_returns = returns - period_rf_rate
        sharpe = excess_returns.mean() / returns.std()

        # Annualize the Sharpe ratio
        return sharpe * np.sqrt(periods_per_year)

    def sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float,
        data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate Sortino ratio - downside risk-adjusted return.

        BUGFIX (Phase 1 - Task 1.4.3): Fixed hardcoded 252 periods/year.
        Now detects timeframe from self.periods_per_year or data timestamps.

        Only considers downside volatility (negative returns) in the denominator,
        making it more appropriate than Sharpe for asymmetric return distributions.

        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free_rate
            data: Optional DataFrame to detect timeframe if not set in constructor

        Returns:
            Sortino ratio (higher is better)
        """
        if len(returns) == 0:
            return 0.0

        # BUGFIX (Phase 1 - Task 1.4.3): Use detected periods instead of hardcoded 252
        if self.periods_per_year is not None:
            periods_per_year = self.periods_per_year
        else:
            periods_per_year = detect_timeframe_periods(data=data)

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

    def value_at_risk(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) at specified confidence level.

        VaR represents the maximum expected loss over a given time period
        at a specified confidence level. For example, 95% VaR of 0.05 means
        there's a 5% chance of losing more than 5% in a period.

        Args:
            returns: Series of period returns
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR as a positive decimal (e.g., 0.05 for 5% VaR)
        """
        if len(returns) == 0:
            return 0.0

        # Calculate the percentile for losses (lower tail)
        # For 95% confidence, we look at the 5th percentile
        var_percentile = 1 - confidence
        var_value = np.percentile(returns, var_percentile * 100)

        # Return as positive value (loss magnitude)
        return float(abs(var_value))

    def conditional_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.

        CVaR is the expected loss given that the loss exceeds VaR. It provides
        a better measure of tail risk than VaR alone by considering the magnitude
        of extreme losses, not just their threshold.

        Args:
            returns: Series of period returns
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            CVaR as a positive decimal (average loss beyond VaR)
        """
        if len(returns) == 0:
            return 0.0

        # Calculate VaR threshold
        var_percentile = 1 - confidence
        var_threshold = np.percentile(returns, var_percentile * 100)

        # Get all returns worse than VaR (in the tail)
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return 0.0

        # CVaR is the average of the tail returns
        cvar_value = tail_returns.mean()

        # Return as positive value (loss magnitude)
        return float(abs(cvar_value))

    def skewness(self, returns: pd.Series) -> float:
        """
        Calculate return distribution skewness using scipy.stats.

        Skewness measures the asymmetry of the return distribution:
        - Negative skew: More frequent large losses (left tail)
        - Zero skew: Symmetric distribution
        - Positive skew: More frequent large gains (right tail)

        For trading, negative skewness is generally undesirable as it
        indicates higher tail risk of large losses.

        Args:
            returns: Series of period returns

        Returns:
            Skewness value (negative, zero, or positive)
        """
        if len(returns) < 3:
            return 0.0

        # Use scipy.stats.skew for unbiased estimate
        # bias=False provides the adjusted Fisher-Pearson standardized moment coefficient
        skew_value = stats.skew(returns, bias=False, nan_policy='omit')

        # Handle case where all values are the same
        if np.isnan(skew_value) or np.isinf(skew_value):
            return 0.0

        return float(skew_value)

    def kurtosis(self, returns: pd.Series) -> float:
        """
        Calculate return distribution kurtosis using scipy.stats.

        Kurtosis measures the "tailedness" of the return distribution:
        - Excess kurtosis > 0: Fat tails (more extreme events than normal distribution)
        - Excess kurtosis = 0: Normal distribution tails
        - Excess kurtosis < 0: Thin tails (fewer extreme events)

        High kurtosis indicates higher probability of extreme returns (both gains and losses).

        Args:
            returns: Series of period returns

        Returns:
            Excess kurtosis value (Fisher's definition, normal distribution = 0)
        """
        if len(returns) < 4:
            return 0.0

        # Use scipy.stats.kurtosis with Fisher=True for excess kurtosis
        # (normal distribution = 0 rather than 3)
        kurt_value = stats.kurtosis(returns, fisher=True, bias=False, nan_policy='omit')

        # Handle case where all values are the same
        if np.isnan(kurt_value) or np.isinf(kurt_value):
            return 0.0

        return float(kurt_value)

    def omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio - probability-weighted gains over losses.

        PHASE 2 (Task 3.2.1): Added Omega Ratio for better risk assessment.

        The Omega Ratio is the probability-weighted ratio of gains versus losses
        relative to a threshold return. It provides a more comprehensive view of
        performance than Sharpe ratio by considering the entire return distribution.

        Formula: Omega = ∫(1 - F(r))dr / ∫F(r)dr for r above/below threshold
        Where F(r) is the cumulative distribution function

        Args:
            returns: Series of period returns
            threshold: Minimum acceptable return (default: 0.0)

        Returns:
            Omega ratio (>1 is good, >2 is excellent)
        """
        if len(returns) == 0:
            return 0.0

        # Calculate returns above and below threshold
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0]
        losses = excess_returns[excess_returns < 0]

        if len(losses) == 0:
            # No losses - infinite Omega (cap at 100 for practical purposes)
            return 100.0

        if len(gains) == 0:
            # No gains - Omega = 0
            return 0.0

        # Omega = sum of gains / abs(sum of losses)
        omega = gains.sum() / abs(losses.sum())

        return float(omega)

    def tail_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Tail Ratio - ratio of right tail to left tail.

        PHASE 2 (Task 3.2.3): Added Tail Ratio for asymmetric risk assessment.

        The Tail Ratio measures the asymmetry between positive and negative extremes.
        A ratio > 1 indicates larger positive outliers than negative outliers.

        Formula: Tail Ratio = abs(95th percentile) / abs(5th percentile)

        Args:
            returns: Series of period returns

        Returns:
            Tail ratio (>1 indicates positive skew in extremes, <1 negative skew)
        """
        if len(returns) < 20:  # Need sufficient data for percentiles
            return 1.0

        # Calculate 95th and 5th percentiles
        right_tail = np.percentile(returns, 95)
        left_tail = np.percentile(returns, 5)

        if left_tail >= 0:
            # No negative tail - return large value
            return 10.0

        if right_tail <= 0:
            # No positive tail - return small value
            return 0.1

        # Tail ratio = |right tail| / |left tail|
        tail_ratio = abs(right_tail) / abs(left_tail)

        return float(tail_ratio)

    def trade_timing_quality(
        self,
        trades: list[Trade],
        price_data: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        """
        Analyze entry/exit timing quality.

        PHASE 3 (Task 3.4.3): Measures how close entries/exits were to optimal prices.

        Args:
            trades: List of completed trades
            price_data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Dict with entry_quality, exit_quality, overall_quality (0-1 scale)
        """
        if len(trades) == 0 or price_data is None:
            return {'entry_quality': 0.0, 'exit_quality': 0.0, 'overall_quality': 0.0}

        entry_qualities = []
        exit_qualities = []

        for trade in trades:
            # Find price range during trade period
            trade_mask = (price_data.index >= trade.entry_time) & (price_data.index <= trade.exit_time)
            if trade_mask.sum() == 0:
                continue

            period_high = price_data.loc[trade_mask, 'high'].max()
            period_low = price_data.loc[trade_mask, 'low'].min()
            price_range = period_high - period_low

            if price_range == 0:
                continue

            # Entry quality: For longs, lower is better; for shorts, higher is better
            if trade.side == 'long':
                entry_quality = 1.0 - ((trade.entry_price - period_low) / price_range)
            else:  # short
                entry_quality = (trade.entry_price - period_low) / price_range

            # Exit quality: For longs, higher is better; for shorts, lower is better
            if trade.side == 'long':
                exit_quality = (trade.exit_price - period_low) / price_range
            else:  # short
                exit_quality = 1.0 - ((trade.exit_price - period_low) / price_range)

            entry_qualities.append(max(0.0, min(1.0, entry_quality)))
            exit_qualities.append(max(0.0, min(1.0, exit_quality)))

        if not entry_qualities:
            return {'entry_quality': 0.0, 'exit_quality': 0.0, 'overall_quality': 0.0}

        avg_entry = float(np.mean(entry_qualities))
        avg_exit = float(np.mean(exit_qualities))
        overall = (avg_entry + avg_exit) / 2.0

        return {
            'entry_quality': avg_entry,
            'exit_quality': avg_exit,
            'overall_quality': overall
        }

    def win_loss_distribution(
        self,
        trades: list[Trade]
    ) -> dict[str, float]:
        """
        Calculate win/loss distribution statistics.

        PHASE 3 (Task 3.4.3): Provides detailed statistics on return distributions.

        Args:
            trades: List of completed trades

        Returns:
            Dict with percentiles, skew, kurtosis of win/loss distributions
        """
        if len(trades) == 0:
            return {
                'win_p25': 0.0, 'win_p50': 0.0, 'win_p75': 0.0,
                'loss_p25': 0.0, 'loss_p50': 0.0, 'loss_p75': 0.0,
                'win_skew': 0.0, 'loss_skew': 0.0,
                'pnl_distribution_quality': 0.0
            }

        winning_trades = [t.pnl for t in trades if t.is_winning]
        losing_trades = [t.pnl for t in trades if not t.is_winning]

        result = {}

        # Winning trade distribution
        if winning_trades:
            result['win_p25'] = float(np.percentile(winning_trades, 25))
            result['win_p50'] = float(np.percentile(winning_trades, 50))
            result['win_p75'] = float(np.percentile(winning_trades, 75))
            result['win_skew'] = float(np.mean(winning_trades) - result['win_p50']) / (np.std(winning_trades) + 1e-9)
        else:
            result['win_p25'] = result['win_p50'] = result['win_p75'] = result['win_skew'] = 0.0

        # Losing trade distribution
        if losing_trades:
            result['loss_p25'] = float(np.percentile(losing_trades, 25))
            result['loss_p50'] = float(np.percentile(losing_trades, 50))
            result['loss_p75'] = float(np.percentile(losing_trades, 75))
            result['loss_skew'] = float(np.mean(losing_trades) - result['loss_p50']) / (np.std(losing_trades) + 1e-9)
        else:
            result['loss_p25'] = result['loss_p50'] = result['loss_p75'] = result['loss_skew'] = 0.0

        # Distribution quality: Positive skew in wins and negative skew in losses is good
        # (means big wins and small losses)
        win_skew_score = max(0.0, min(1.0, (result['win_skew'] + 1.0) / 2.0))
        loss_skew_score = max(0.0, min(1.0, (-result['loss_skew'] + 1.0) / 2.0))
        result['pnl_distribution_quality'] = (win_skew_score + loss_skew_score) / 2.0

        return result

    def trade_clustering_analysis(
        self,
        trades: list[Trade]
    ) -> dict[str, any]:
        """
        Analyze temporal clustering of trades.

        PHASE 3 (Task 3.4.3): Identifies whether trades cluster in time (overtrading risk).

        Args:
            trades: List of completed trades

        Returns:
            Dict with clustering metrics
        """
        if len(trades) < 2:
            return {
                'avg_time_between_trades_hours': 0.0,
                'min_time_between_trades_hours': 0.0,
                'clustering_score': 0.0,
                'rapid_trade_sequences': 0
            }

        # Calculate time gaps between consecutive trades
        time_gaps = []
        for i in range(1, len(trades)):
            gap = (trades[i].entry_time - trades[i-1].exit_time).total_seconds() / 3600.0  # hours
            if gap >= 0:  # Only positive gaps
                time_gaps.append(gap)

        if not time_gaps:
            return {
                'avg_time_between_trades_hours': 0.0,
                'min_time_between_trades_hours': 0.0,
                'clustering_score': 0.0,
                'rapid_trade_sequences': 0
            }

        avg_gap = float(np.mean(time_gaps))
        min_gap = float(np.min(time_gaps))
        std_gap = float(np.std(time_gaps))

        # Clustering score: Low std relative to mean indicates consistent spacing (good)
        # High clustering (low score) means trades bunch together (potentially overtrading)
        clustering_score = 1.0 - min(1.0, std_gap / (avg_gap + 1e-9))

        # Count rapid trade sequences (trades within 1 hour of each other)
        rapid_sequences = sum(1 for gap in time_gaps if gap < 1.0)

        return {
            'avg_time_between_trades_hours': avg_gap,
            'min_time_between_trades_hours': min_gap,
            'clustering_score': clustering_score,
            'rapid_trade_sequences': rapid_sequences
        }

    def statistical_tests(
        self,
        returns: pd.Series
    ) -> dict[str, any]:
        """
        Perform statistical tests on return distribution.

        PHASE 3 (Task 3.4.4): Tests for normality, autocorrelation, stationarity.

        Args:
            returns: Series of period returns

        Returns:
            Dict with test results and interpretations
        """
        from scipy import stats

        if len(returns) < 20:
            return {
                'normality_test': 'insufficient_data',
                'autocorrelation': 0.0,
                'is_stationary': 'unknown',
                'interpretation': 'Insufficient data for statistical testing (need >= 20 observations)'
            }

        results = {}

        # 1. Jarque-Bera test for normality
        # H0: Data is normally distributed
        # p < 0.05 means reject H0 (not normal)
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
            results['jb_statistic'] = float(jb_stat)
            results['jb_pvalue'] = float(jb_pvalue)
            results['is_normal'] = jb_pvalue > 0.05
            results['normality_test'] = 'normal' if jb_pvalue > 0.05 else 'non_normal'
        except Exception as e:
            results['normality_test'] = f'error: {str(e)}'
            results['is_normal'] = False

        # 2. Autocorrelation (lag-1)
        # Measures if returns are correlated with previous returns
        # High autocorrelation indicates predictability or momentum
        if len(returns) > 1:
            lag1_autocorr = returns.autocorr(lag=1)
            results['autocorrelation_lag1'] = float(lag1_autocorr) if not np.isnan(lag1_autocorr) else 0.0
            results['has_momentum'] = abs(results['autocorrelation_lag1']) > 0.1
        else:
            results['autocorrelation_lag1'] = 0.0
            results['has_momentum'] = False

        # 3. Augmented Dickey-Fuller test for stationarity
        # H0: Series has unit root (non-stationary)
        # p < 0.05 means reject H0 (stationary)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(returns.dropna(), autolag='AIC')
            results['adf_statistic'] = float(adf_result[0])
            results['adf_pvalue'] = float(adf_result[1])
            results['is_stationary'] = adf_result[1] < 0.05
        except Exception:
            # statsmodels might not be installed, use simple heuristic
            # Check if mean/std are relatively stable over time
            mid = len(returns) // 2
            first_half_std = returns[:mid].std()
            second_half_std = returns[mid:].std()
            std_ratio = min(first_half_std, second_half_std) / (max(first_half_std, second_half_std) + 1e-9)
            results['is_stationary'] = std_ratio > 0.5  # Heuristic: stable if std doesn't change >2x
            results['adf_pvalue'] = None

        # 4. Interpretation
        interpretation_parts = []

        if results.get('is_normal', False):
            interpretation_parts.append("Returns are approximately normally distributed")
        else:
            interpretation_parts.append("Returns show non-normal distribution (fat tails or skew)")

        if results.get('has_momentum', False):
            interpretation_parts.append(f"Significant autocorrelation ({results['autocorrelation_lag1']:.3f}) detected")
        else:
            interpretation_parts.append("No significant autocorrelation (returns are independent)")

        if results.get('is_stationary', False):
            interpretation_parts.append("Returns series is stationary (mean-reverting)")
        else:
            interpretation_parts.append("Returns may be non-stationary (trending or regime-shifting)")

        results['interpretation'] = ' | '.join(interpretation_parts)

        return results

    def max_consecutive_drawdown_days(self, equity_curve: list[tuple]) -> int:
        """
        Calculate maximum consecutive days in drawdown (underwater period).

        PHASE 2 (Task 3.2.4): Added Max Consecutive Drawdown Days for recovery analysis.

        This metric shows the longest period the strategy spent underwater
        (below previous peak), which is crucial for understanding recovery time.

        Args:
            equity_curve: List of (timestamp, equity_value) tuples

        Returns:
            Maximum number of consecutive periods in drawdown
        """
        if len(equity_curve) == 0:
            return 0

        equity_values = np.array([equity for _, equity in equity_curve])

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_values)

        # Identify underwater periods (current equity < previous peak)
        is_underwater = equity_values < running_max

        # Count consecutive underwater periods
        max_consecutive = 0
        current_consecutive = 0

        for underwater in is_underwater:
            if underwater:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return int(max_consecutive)

    def ulcer_index(self, equity_curve: list[tuple]) -> float:
        """
        Calculate Ulcer Index - alternative downside volatility measure.

        PHASE 2 (Task 3.2.2): Added Ulcer Index for downside risk measurement.

        The Ulcer Index measures the depth and duration of drawdowns,
        providing a more intuitive measure of downside risk than standard deviation.

        Formula: UI = sqrt(mean(drawdown_percentages²))

        Args:
            equity_curve: List of (timestamp, equity_value) tuples

        Returns:
            Ulcer Index (lower is better, measures drawdown severity)
        """
        if len(equity_curve) == 0:
            return 0.0

        equity_values = np.array([equity for _, equity in equity_curve])

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_values)

        # Calculate drawdown percentages
        drawdowns = (equity_values - running_max) / running_max * 100

        # Ulcer Index = RMS of drawdowns
        ulcer = np.sqrt(np.mean(drawdowns ** 2))

        return float(ulcer)

    def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate Information Ratio - excess return vs benchmark per unit of tracking error.

        BUGFIX (Phase 1 - Task 1.4.3): Fixed hardcoded 252 periods/year.
        Now detects timeframe from self.periods_per_year or data timestamps.

        The Information Ratio measures the risk-adjusted returns of a strategy
        relative to a benchmark. It's calculated as:
        IR = (Portfolio Return - Benchmark Return) / Tracking Error

        Where tracking error is the standard deviation of the excess returns.
        Higher IR indicates better risk-adjusted outperformance.

        Args:
            returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns (e.g., buy-and-hold)
                             If None, assumes zero benchmark (cash)
            data: Optional DataFrame to detect timeframe if not set in constructor

        Returns:
            Information Ratio (higher is better, >0.5 is good, >1.0 is excellent)
        """
        if len(returns) == 0:
            return 0.0

        # If no benchmark provided, use zero returns (cash benchmark)
        if benchmark_returns is None or len(benchmark_returns) == 0:
            benchmark_returns = pd.Series(np.zeros(len(returns)), index=returns.index)

        # Align the series in case of different lengths
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')

        if len(aligned_returns) == 0:
            return 0.0

        # Calculate excess returns
        excess_returns = aligned_returns - aligned_benchmark

        # Calculate tracking error (standard deviation of excess returns)
        tracking_error = excess_returns.std()

        if tracking_error == 0 or np.isnan(tracking_error):
            return 0.0

        # Calculate information ratio
        ir = excess_returns.mean() / tracking_error

        # BUGFIX (Phase 1 - Task 1.4.3): Use detected periods instead of hardcoded 252
        if self.periods_per_year is not None:
            periods_per_year = self.periods_per_year
        else:
            periods_per_year = detect_timeframe_periods(data=data)

        ir_annualized = ir * np.sqrt(periods_per_year)

        return float(ir_annualized)


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

    # Test 8: Value at Risk (VaR) calculation
    total_tests += 1
    print("\nTest 8: Value at Risk (VaR) calculation")
    try:
        # Create returns with known distribution
        test_returns = pd.Series([0.02, 0.01, -0.01, 0.015, -0.02, 0.01, -0.03, 0.02, -0.01, 0.005])
        var_95 = calculator.value_at_risk(test_returns, confidence=0.95)

        # VaR should be positive and reasonable
        if var_95 <= 0:
            all_validation_failures.append(
                f"VaR should be positive, got {var_95}"
            )
        if var_95 > 1.0:  # Should not exceed 100%
            all_validation_failures.append(
                f"VaR seems unreasonably high: {var_95}"
            )

        print(f"  ✓ 95% VaR: {var_95:.4f} ({var_95:.2%})")
        print(f"  ✓ This means 5% chance of losing more than {var_95:.2%}")

    except Exception as e:
        all_validation_failures.append(f"VaR calculation exception: {e}")

    # Test 9: Conditional VaR (CVaR) calculation
    total_tests += 1
    print("\nTest 9: Conditional VaR (CVaR) calculation")
    try:
        test_returns = pd.Series([0.02, 0.01, -0.01, 0.015, -0.02, 0.01, -0.03, 0.02, -0.01, 0.005])
        cvar_95 = calculator.conditional_var(test_returns, confidence=0.95)
        var_95 = calculator.value_at_risk(test_returns, confidence=0.95)

        # CVaR should be >= VaR (average of tail is worse than threshold)
        if cvar_95 < var_95 - 0.001:  # Small tolerance for floating point
            all_validation_failures.append(
                f"CVaR ({cvar_95:.4f}) should be >= VaR ({var_95:.4f})"
            )

        print(f"  ✓ 95% CVaR: {cvar_95:.4f} ({cvar_95:.2%})")
        print(f"  ✓ CVaR >= VaR: {cvar_95:.4f} >= {var_95:.4f}")

    except Exception as e:
        all_validation_failures.append(f"CVaR calculation exception: {e}")

    # Test 10: Skewness calculation
    total_tests += 1
    print("\nTest 10: Skewness calculation")
    try:
        # Positive skew: more large gains
        positive_skew_returns = pd.Series([0.01, 0.01, 0.02, 0.05, 0.10, 0.01, 0.01])
        pos_skew = calculator.skewness(positive_skew_returns)

        # Negative skew: more large losses
        negative_skew_returns = pd.Series([-0.01, -0.01, -0.02, -0.05, -0.10, -0.01, -0.01])
        neg_skew = calculator.skewness(negative_skew_returns)

        # Symmetric distribution
        symmetric_returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01, -0.01])
        sym_skew = calculator.skewness(symmetric_returns)

        # Verify signs
        if pos_skew <= 0:
            all_validation_failures.append(
                f"Positive skew distribution should have positive skewness, got {pos_skew}"
            )
        if neg_skew >= 0:
            all_validation_failures.append(
                f"Negative skew distribution should have negative skewness, got {neg_skew}"
            )

        print(f"  ✓ Positive skew returns: {pos_skew:.4f}")
        print(f"  ✓ Negative skew returns: {neg_skew:.4f}")
        print(f"  ✓ Symmetric returns: {sym_skew:.4f}")

    except Exception as e:
        all_validation_failures.append(f"Skewness calculation exception: {e}")

    # Test 11: Kurtosis calculation
    total_tests += 1
    print("\nTest 11: Kurtosis calculation")
    try:
        # Fat tails: extreme values
        fat_tail_returns = pd.Series([0.01, 0.01, 0.01, 0.20, -0.20, 0.01, 0.01, 0.01])
        fat_kurt = calculator.kurtosis(fat_tail_returns)

        # Normal-ish distribution
        normal_returns = pd.Series([0.01, 0.02, 0.015, -0.01, -0.015, 0.01, 0.005, -0.005])
        normal_kurt = calculator.kurtosis(normal_returns)

        # Fat tails should have positive excess kurtosis
        if fat_kurt <= normal_kurt:
            all_validation_failures.append(
                f"Fat tail kurtosis ({fat_kurt:.4f}) should be > normal kurtosis ({normal_kurt:.4f})"
            )

        print(f"  ✓ Fat tail kurtosis: {fat_kurt:.4f}")
        print(f"  ✓ Normal kurtosis: {normal_kurt:.4f}")
        print(f"  ✓ Excess kurtosis > 0 indicates fat tails")

    except Exception as e:
        all_validation_failures.append(f"Kurtosis calculation exception: {e}")

    # Test 12: Information Ratio calculation
    total_tests += 1
    print("\nTest 12: Information Ratio calculation")
    try:
        # Strategy returns outperform benchmark
        strategy_returns = pd.Series([0.02, 0.03, -0.01, 0.015, 0.02])
        benchmark_returns = pd.Series([0.01, 0.01, -0.01, 0.01, 0.01])

        ir_with_benchmark = calculator.information_ratio(strategy_returns, benchmark_returns)
        ir_vs_cash = calculator.information_ratio(strategy_returns, None)

        # IR with positive excess returns should be positive
        if ir_with_benchmark <= 0:
            all_validation_failures.append(
                f"IR should be positive for outperforming strategy, got {ir_with_benchmark}"
            )

        print(f"  ✓ IR vs benchmark: {ir_with_benchmark:.4f}")
        print(f"  ✓ IR vs cash: {ir_vs_cash:.4f}")
        print(f"  ✓ Higher IR indicates better risk-adjusted outperformance")

    except Exception as e:
        all_validation_failures.append(f"Information Ratio exception: {e}")

    # Test 13: Advanced metrics in calculate_all_metrics
    total_tests += 1
    print("\nTest 13: All advanced metrics integrated")
    try:
        metrics = calculator.calculate_all_metrics(
            returns=returns,
            trades=sample_trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
        )

        # Verify all advanced metrics are calculated
        if metrics.value_at_risk_95 == 0.0 and len(returns) > 0:
            all_validation_failures.append("VaR should be calculated for non-empty returns")
        if metrics.conditional_var_95 == 0.0 and len(returns) > 0:
            all_validation_failures.append("CVaR should be calculated for non-empty returns")

        print(f"  ✓ VaR 95%: {metrics.value_at_risk_95:.4f}")
        print(f"  ✓ CVaR 95%: {metrics.conditional_var_95:.4f}")
        print(f"  ✓ Skewness: {metrics.skewness:.4f}")
        print(f"  ✓ Kurtosis: {metrics.kurtosis:.4f}")
        print(f"  ✓ Information Ratio: {metrics.information_ratio:.4f}")

    except Exception as e:
        all_validation_failures.append(f"All advanced metrics exception: {e}")

    # Test 14: Edge case - Empty inputs
    total_tests += 1
    print("\nTest 14: Edge case - Empty inputs")
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
        if empty_metrics.value_at_risk_95 != 0.0:
            all_validation_failures.append(
                f"Empty returns should result in 0 VaR, got {empty_metrics.value_at_risk_95}"
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
