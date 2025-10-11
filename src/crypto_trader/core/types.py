"""
Core type definitions for the crypto trading system.

This module defines all the fundamental types, enums, and dataclasses used
throughout the trading system including timeframes, order types, signals,
and performance metrics.

Documentation:
- Python Enum: https://docs.python.org/3/library/enum.html
- Dataclasses: https://docs.python.org/3/library/dataclasses.html
- Typing: https://docs.python.org/3/library/typing.html

Sample Input:
    timeframe = Timeframe.HOUR_1
    signal = Signal.BUY
    metrics = PerformanceMetrics(
        total_return=0.15,
        sharpe_ratio=1.5,
        max_drawdown=0.10
    )

Expected Output:
    Validated type-safe objects with proper enum values and dataclass attributes
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, Optional


class Timeframe(str, Enum):
    """
    Supported timeframes for candlestick data.

    Inherits from str to ensure JSON serialization compatibility
    and proper string comparison behavior.
    """
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

    def to_minutes(self) -> int:
        """Convert timeframe to minutes for calculations."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
        }
        return mapping[self.value]


class OrderType(str, Enum):
    """
    Order execution types.

    MARKET: Execute immediately at current market price
    LIMIT: Execute only at specified price or better
    """
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(str, Enum):
    """
    Direction of the order.

    BUY: Long position (bullish)
    SELL: Short position or close long (bearish)
    """
    BUY = "buy"
    SELL = "sell"


# Type alias for trading signals
# 1 = Buy/Long, 0 = Hold/Neutral, -1 = Sell/Short
Signal = Literal[1, 0, -1]


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for backtesting and live trading.

    Attributes:
        total_return: Total percentage return (e.g., 0.15 = 15%)
        sharpe_ratio: Risk-adjusted return metric (higher is better)
        max_drawdown: Maximum peak-to-trough decline (e.g., 0.10 = 10%)
        win_rate: Percentage of winning trades (0.0 to 1.0)
        profit_factor: Gross profit / Gross loss (>1 is profitable)
        total_trades: Number of completed trades
        winning_trades: Number of profitable trades
        losing_trades: Number of unprofitable trades
        avg_win: Average profit per winning trade
        avg_loss: Average loss per losing trade (negative value)
        max_consecutive_wins: Longest winning streak
        max_consecutive_losses: Longest losing streak
        avg_trade_duration: Average time in trade (minutes)
        sortino_ratio: Downside risk-adjusted return
        calmar_ratio: Return / Max drawdown
        recovery_factor: Net profit / Max drawdown
        expectancy: Average expected profit per trade
        total_fees: Total transaction costs
        final_capital: Ending portfolio value
    """
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0
    expectancy: float = 0.0
    total_fees: float = 0.0
    final_capital: float = 0.0

    def is_profitable(self) -> bool:
        """Check if the strategy is profitable."""
        return self.total_return > 0 and self.profit_factor > 1.0

    def risk_adjusted_quality(self) -> str:
        """
        Assess risk-adjusted performance quality.

        Returns:
            Quality rating: Excellent, Good, Fair, or Poor
        """
        if self.sharpe_ratio >= 2.0 and self.max_drawdown < 0.15:
            return "Excellent"
        elif self.sharpe_ratio >= 1.5 and self.max_drawdown < 0.25:
            return "Good"
        elif self.sharpe_ratio >= 1.0 and self.max_drawdown < 0.35:
            return "Fair"
        else:
            return "Poor"


@dataclass
class Trade:
    """
    Represents a single completed trade.

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        entry_time: When position was opened
        exit_time: When position was closed
        entry_price: Price at entry
        exit_price: Price at exit
        side: BUY or SELL
        quantity: Amount traded
        pnl: Profit and loss (absolute)
        pnl_percent: Profit and loss (percentage)
        fees: Transaction costs
        order_type: MARKET or LIMIT
    """
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    side: OrderSide
    quantity: float
    pnl: float
    pnl_percent: float
    fees: float
    order_type: OrderType = OrderType.MARKET

    @property
    def duration_minutes(self) -> float:
        """Calculate trade duration in minutes."""
        delta = self.exit_time - self.entry_time
        return delta.total_seconds() / 60

    @property
    def is_winning(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0


@dataclass
class BacktestResult:
    """
    Complete results from a backtest run.

    Attributes:
        strategy_name: Name of the tested strategy
        symbol: Trading pair tested
        timeframe: Candle timeframe used
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting portfolio value
        metrics: Performance metrics
        trades: List of all completed trades
        equity_curve: Portfolio value over time
        metadata: Additional strategy-specific data
    """
    strategy_name: str
    symbol: str
    timeframe: Timeframe
    start_date: datetime
    end_date: datetime
    initial_capital: float
    metrics: PerformanceMetrics
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def duration_days(self) -> int:
        """Calculate backtest duration in days."""
        delta = self.end_date - self.start_date
        return delta.days

    def summary(self) -> dict:
        """Generate a summary dictionary of key results."""
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "duration_days": self.duration_days,
            "total_return": f"{self.metrics.total_return:.2%}",
            "sharpe_ratio": f"{self.metrics.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.metrics.max_drawdown:.2%}",
            "win_rate": f"{self.metrics.win_rate:.2%}",
            "total_trades": self.metrics.total_trades,
            "final_capital": f"${self.metrics.final_capital:,.2f}",
            "quality": self.metrics.risk_adjusted_quality(),
        }


if __name__ == "__main__":
    """
    Validation function to test all type definitions with real data.
    Tests enums, dataclasses, and their methods.
    """
    import sys
    from datetime import timedelta

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating types.py with real data...\n")

    # Test 1: Timeframe enum and conversions
    total_tests += 1
    print("Test 1: Timeframe enum operations")
    try:
        tf_1h = Timeframe.HOUR_1
        if tf_1h.value != "1h":
            all_validation_failures.append(f"Timeframe value: Expected '1h', got '{tf_1h.value}'")
        if tf_1h.to_minutes() != 60:
            all_validation_failures.append(f"Timeframe conversion: Expected 60, got {tf_1h.to_minutes()}")

        tf_1w = Timeframe.WEEK_1
        if tf_1w.to_minutes() != 10080:
            all_validation_failures.append(f"Week conversion: Expected 10080, got {tf_1w.to_minutes()}")

        print(f"  ✓ {tf_1h.value} = {tf_1h.to_minutes()} minutes")
        print(f"  ✓ {tf_1w.value} = {tf_1w.to_minutes()} minutes")
    except Exception as e:
        all_validation_failures.append(f"Timeframe test exception: {e}")

    # Test 2: OrderType and OrderSide enums
    total_tests += 1
    print("\nTest 2: Order enums")
    try:
        order_type = OrderType.MARKET
        order_side = OrderSide.BUY
        if order_type.value != "market":
            all_validation_failures.append(f"OrderType: Expected 'market', got '{order_type.value}'")
        if order_side.value != "buy":
            all_validation_failures.append(f"OrderSide: Expected 'buy', got '{order_side.value}'")
        print(f"  ✓ OrderType: {order_type.value}")
        print(f"  ✓ OrderSide: {order_side.value}")
    except Exception as e:
        all_validation_failures.append(f"Order enums test exception: {e}")

    # Test 3: Signal type
    total_tests += 1
    print("\nTest 3: Signal type literals")
    try:
        buy_signal: Signal = 1
        hold_signal: Signal = 0
        sell_signal: Signal = -1

        signals_valid = (buy_signal == 1 and hold_signal == 0 and sell_signal == -1)
        if not signals_valid:
            all_validation_failures.append(f"Signals: Values don't match expected 1, 0, -1")
        print(f"  ✓ Buy: {buy_signal}, Hold: {hold_signal}, Sell: {sell_signal}")
    except Exception as e:
        all_validation_failures.append(f"Signal test exception: {e}")

    # Test 4: Trade dataclass
    total_tests += 1
    print("\nTest 4: Trade dataclass")
    try:
        entry_time = datetime(2025, 1, 1, 10, 0, 0)
        exit_time = datetime(2025, 1, 1, 14, 30, 0)

        trade = Trade(
            symbol="BTCUSDT",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=45000.0,
            exit_price=46500.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=150.0,
            pnl_percent=3.33,
            fees=15.0,
            order_type=OrderType.MARKET
        )

        expected_duration = 270.0  # 4.5 hours in minutes
        if trade.duration_minutes != expected_duration:
            all_validation_failures.append(
                f"Trade duration: Expected {expected_duration}, got {trade.duration_minutes}"
            )
        if not trade.is_winning:
            all_validation_failures.append("Trade should be winning (pnl > 0)")

        print(f"  ✓ Symbol: {trade.symbol}")
        print(f"  ✓ Duration: {trade.duration_minutes} minutes")
        print(f"  ✓ PnL: ${trade.pnl} ({trade.pnl_percent}%)")
        print(f"  ✓ Is winning: {trade.is_winning}")
    except Exception as e:
        all_validation_failures.append(f"Trade test exception: {e}")

    # Test 5: PerformanceMetrics dataclass
    total_tests += 1
    print("\nTest 5: PerformanceMetrics dataclass")
    try:
        metrics = PerformanceMetrics(
            total_return=0.25,
            sharpe_ratio=2.1,
            max_drawdown=0.12,
            win_rate=0.65,
            profit_factor=1.8,
            total_trades=50,
            winning_trades=32,
            losing_trades=18,
            avg_win=250.0,
            avg_loss=-120.0,
            final_capital=12500.0
        )

        if not metrics.is_profitable():
            all_validation_failures.append("Metrics should be profitable")

        expected_quality = "Excellent"
        if metrics.risk_adjusted_quality() != expected_quality:
            all_validation_failures.append(
                f"Quality rating: Expected '{expected_quality}', got '{metrics.risk_adjusted_quality()}'"
            )

        print(f"  ✓ Total return: {metrics.total_return:.2%}")
        print(f"  ✓ Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  ✓ Max drawdown: {metrics.max_drawdown:.2%}")
        print(f"  ✓ Win rate: {metrics.win_rate:.2%}")
        print(f"  ✓ Is profitable: {metrics.is_profitable()}")
        print(f"  ✓ Quality: {metrics.risk_adjusted_quality()}")
    except Exception as e:
        all_validation_failures.append(f"PerformanceMetrics test exception: {e}")

    # Test 6: BacktestResult dataclass
    total_tests += 1
    print("\nTest 6: BacktestResult dataclass")
    try:
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        result = BacktestResult(
            strategy_name="MA Crossover",
            symbol="ETHUSDT",
            timeframe=Timeframe.HOUR_4,
            start_date=start,
            end_date=end,
            initial_capital=10000.0,
            metrics=metrics,
            trades=[trade],
            equity_curve=[(start, 10000.0), (end, 12500.0)]
        )

        expected_days = 365  # 2024 is a leap year, but datetime delta calculates it as 365
        actual_days = result.duration_days
        # Account for leap year
        if actual_days not in [365, 366]:
            all_validation_failures.append(
                f"Duration days: Expected 365 or 366, got {actual_days}"
            )

        summary = result.summary()
        if summary["strategy"] != "MA Crossover":
            all_validation_failures.append(f"Summary strategy: Expected 'MA Crossover', got '{summary['strategy']}'")
        if summary["symbol"] != "ETHUSDT":
            all_validation_failures.append(f"Summary symbol: Expected 'ETHUSDT', got '{summary['symbol']}'")

        print(f"  ✓ Strategy: {result.strategy_name}")
        print(f"  ✓ Duration: {result.duration_days} days")
        print(f"  ✓ Initial capital: ${result.initial_capital:,.2f}")
        print(f"  ✓ Final capital: ${result.metrics.final_capital:,.2f}")
        print(f"  ✓ Summary keys: {list(summary.keys())}")
    except Exception as e:
        all_validation_failures.append(f"BacktestResult test exception: {e}")

    # Test 7: Edge case - Poor performing metrics
    total_tests += 1
    print("\nTest 7: Poor performing metrics edge case")
    try:
        poor_metrics = PerformanceMetrics(
            total_return=-0.10,
            sharpe_ratio=0.5,
            max_drawdown=0.40,
            profit_factor=0.8
        )

        if poor_metrics.is_profitable():
            all_validation_failures.append("Poor metrics should not be profitable")

        expected_poor_quality = "Poor"
        if poor_metrics.risk_adjusted_quality() != expected_poor_quality:
            all_validation_failures.append(
                f"Poor quality: Expected '{expected_poor_quality}', got '{poor_metrics.risk_adjusted_quality()}'"
            )

        print(f"  ✓ Is profitable: {poor_metrics.is_profitable()}")
        print(f"  ✓ Quality: {poor_metrics.risk_adjusted_quality()}")
    except Exception as e:
        all_validation_failures.append(f"Poor metrics test exception: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
