"""
Risk limit enforcement for crypto trading.

This module implements various risk limit checks to ensure trades comply with
portfolio risk management rules. It provides pre-trade validation to prevent
excessive risk-taking and protect capital.

**Purpose**: Enforce risk limits including position size, portfolio exposure,
drawdown limits, daily loss limits, and per-trade risk constraints.

**Key Components**:
- RiskLimitChecker: Main class for limit validation
- Position size limit enforcement
- Portfolio exposure limits
- Drawdown monitoring and limits
- Daily loss limits
- Per-trade risk validation

**Third-party packages**:
- dataclasses: https://docs.python.org/3/library/dataclasses.html
- datetime: https://docs.python.org/3/library/datetime.html
- loguru: https://loguru.readthedocs.io/

**Sample Input**:
```python
checker = RiskLimitChecker(config)
is_allowed, reason = checker.check_trade(
    position_size=0.1,
    entry_price=50000.0,
    portfolio_state=portfolio,
    current_drawdown=0.08
)
```

**Expected Output**:
```python
(True, "All risk checks passed")
# or
(False, "Daily loss limit exceeded: -3.5% > -3.0%")
```
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from loguru import logger

from crypto_trader.backtesting.portfolio import PortfolioState
from crypto_trader.core.config import RiskConfig


@dataclass
class DailyRiskTracker:
    """
    Track daily risk metrics for enforcement of daily limits.

    Attributes:
        date: Trading date
        starting_equity: Equity at start of day
        current_equity: Current equity
        realized_pnl: Realized profit/loss for the day
        trades_count: Number of trades executed today
        max_loss_hit: Whether max daily loss was hit
    """
    date: datetime
    starting_equity: float
    current_equity: float
    realized_pnl: float = 0.0
    trades_count: int = 0
    max_loss_hit: bool = False

    @property
    def daily_return(self) -> float:
        """Calculate daily return percentage."""
        if self.starting_equity == 0:
            return 0.0
        return (self.current_equity - self.starting_equity) / self.starting_equity

    @property
    def daily_loss(self) -> float:
        """Calculate daily loss (negative return)."""
        return min(0.0, self.daily_return)


class RiskLimitChecker:
    """
    Risk limit checker for validating trades against risk rules.

    Enforces multiple layers of risk limits including position size,
    portfolio exposure, drawdown, and daily loss limits.
    """

    def __init__(self, config: RiskConfig):
        """
        Initialize risk limit checker.

        Args:
            config: Risk configuration with limit parameters
        """
        self.config = config
        self._daily_tracker: Optional[DailyRiskTracker] = None

        logger.info(
            f"RiskLimitChecker initialized: "
            f"max_position_risk={config.max_position_risk:.2%}, "
            f"max_portfolio_risk={config.max_portfolio_risk:.2%}, "
            f"max_daily_loss={config.max_daily_loss_percent:.2%}"
        )

    def check_position_size_limit(
        self,
        position_value: float,
        portfolio_equity: float,
        max_position_percent: float = 0.95
    ) -> Tuple[bool, str]:
        """
        Check if position size is within limits.

        Args:
            position_value: Value of the proposed position
            portfolio_equity: Current portfolio equity
            max_position_percent: Maximum position as % of equity (default 95%)

        Returns:
            Tuple of (is_allowed, reason)
        """
        if portfolio_equity <= 0:
            return False, f"Invalid portfolio equity: ${portfolio_equity:,.2f}"

        position_percent = position_value / portfolio_equity

        # Check against maximum position size
        # This is a separate check from risk - prevents over-concentration
        if position_percent > max_position_percent:
            return False, (
                f"Position size too large: {position_percent:.2%} of portfolio "
                f"(max allowed: {max_position_percent:.2%})"
            )

        logger.debug(
            f"Position size check passed: ${position_value:,.2f} "
            f"({position_percent:.2%} of ${portfolio_equity:,.2f})"
        )

        return True, "Position size within limits"

    def check_position_risk_limit(
        self,
        risk_amount: float,
        portfolio_equity: float
    ) -> Tuple[bool, str]:
        """
        Check if position risk is within limits.

        Args:
            risk_amount: Dollar amount at risk (based on stop loss)
            portfolio_equity: Current portfolio equity

        Returns:
            Tuple of (is_allowed, reason)
        """
        if portfolio_equity <= 0:
            return False, f"Invalid portfolio equity: ${portfolio_equity:,.2f}"

        risk_percent = risk_amount / portfolio_equity

        if risk_percent > self.config.max_position_risk:
            return False, (
                f"Position risk too high: {risk_percent:.2%} "
                f"(max: {self.config.max_position_risk:.2%})"
            )

        logger.debug(
            f"Position risk check passed: ${risk_amount:,.2f} "
            f"({risk_percent:.2%} of ${portfolio_equity:,.2f})"
        )

        return True, "Position risk within limits"

    def check_portfolio_risk_limit(
        self,
        new_risk_amount: float,
        portfolio_state: PortfolioState
    ) -> Tuple[bool, str]:
        """
        Check if total portfolio risk would exceed limits.

        Args:
            new_risk_amount: Risk from proposed new position
            portfolio_state: Current portfolio state

        Returns:
            Tuple of (is_allowed, reason)
        """
        # Calculate current portfolio risk from open positions
        # Estimate risk as number of positions * max position risk
        # This is conservative - assumes each position is at max risk
        num_positions = len(portfolio_state.positions)
        estimated_current_risk_percent = num_positions * self.config.max_position_risk

        new_risk_percent = new_risk_amount / portfolio_state.equity if portfolio_state.equity > 0 else 0
        total_risk_percent = estimated_current_risk_percent + new_risk_percent

        if total_risk_percent > self.config.max_portfolio_risk:
            return False, (
                f"Total portfolio risk too high: {total_risk_percent:.2%} "
                f"(max: {self.config.max_portfolio_risk:.2%}). "
                f"Estimated current: {estimated_current_risk_percent:.2%}, "
                f"New: {new_risk_percent:.2%}"
            )

        logger.debug(
            f"Portfolio risk check passed: total={total_risk_percent:.2%} "
            f"(current={estimated_current_risk_percent:.2%}, new={new_risk_percent:.2%})"
        )

        return True, "Portfolio risk within limits"

    def check_max_positions_limit(
        self,
        current_positions: int
    ) -> Tuple[bool, str]:
        """
        Check if maximum number of open positions would be exceeded.

        Args:
            current_positions: Number of currently open positions

        Returns:
            Tuple of (is_allowed, reason)
        """
        if current_positions >= self.config.max_open_positions:
            return False, (
                f"Maximum positions reached: {current_positions} "
                f"(max: {self.config.max_open_positions})"
            )

        logger.debug(
            f"Max positions check passed: {current_positions}/{self.config.max_open_positions}"
        )

        return True, "Position count within limits"

    def check_daily_loss_limit(
        self,
        current_equity: float,
        timestamp: datetime
    ) -> Tuple[bool, str]:
        """
        Check if daily loss limit has been hit.

        Args:
            current_equity: Current portfolio equity
            timestamp: Current timestamp

        Returns:
            Tuple of (is_allowed, reason)
        """
        # Initialize or reset daily tracker if new day
        if self._daily_tracker is None or self._daily_tracker.date.date() != timestamp.date():
            self._daily_tracker = DailyRiskTracker(
                date=timestamp,
                starting_equity=current_equity,
                current_equity=current_equity
            )
            logger.info(
                f"Daily tracker reset: date={timestamp.date()}, "
                f"starting_equity=${current_equity:,.2f}"
            )
            return True, "Daily tracker reset"

        # Update current equity
        self._daily_tracker.current_equity = current_equity

        # Check if max loss already hit
        if self._daily_tracker.max_loss_hit:
            return False, "Daily loss limit already exceeded - trading halted for today"

        # Check daily loss
        daily_loss = self._daily_tracker.daily_loss

        if abs(daily_loss) > self.config.max_daily_loss_percent:
            self._daily_tracker.max_loss_hit = True
            return False, (
                f"Daily loss limit exceeded: {daily_loss:.2%} "
                f"(max: {-self.config.max_daily_loss_percent:.2%})"
            )

        logger.debug(
            f"Daily loss check passed: {daily_loss:.2%} "
            f"(limit: {-self.config.max_daily_loss_percent:.2%})"
        )

        return True, "Daily loss within limits"

    def check_drawdown_limit(
        self,
        current_equity: float,
        peak_equity: float,
        max_drawdown_limit: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if drawdown has exceeded maximum allowed.

        Args:
            current_equity: Current portfolio equity
            peak_equity: Peak equity achieved
            max_drawdown_limit: Optional override for max drawdown (uses config if None)

        Returns:
            Tuple of (is_allowed, reason)
        """
        if peak_equity <= 0:
            return True, "No drawdown (no peak equity)"

        drawdown = (peak_equity - current_equity) / peak_equity

        # Use provided limit or fall back to config
        # Config uses max_daily_loss as a proxy for max drawdown if no specific field
        limit = max_drawdown_limit if max_drawdown_limit is not None else self.config.max_daily_loss_percent * 3

        if drawdown > limit:
            return False, (
                f"Maximum drawdown exceeded: {drawdown:.2%} "
                f"(max: {limit:.2%}). Peak: ${peak_equity:,.2f}, Current: ${current_equity:,.2f}"
            )

        logger.debug(
            f"Drawdown check passed: {drawdown:.2%} (limit: {limit:.2%})"
        )

        return True, "Drawdown within limits"

    def check_all_limits(
        self,
        position_value: float,
        risk_amount: float,
        portfolio_state: PortfolioState,
        timestamp: datetime,
        peak_equity: float
    ) -> Tuple[bool, str]:
        """
        Run all risk limit checks.

        Args:
            position_value: Value of proposed position
            risk_amount: Dollar risk amount
            portfolio_state: Current portfolio state
            timestamp: Current timestamp
            peak_equity: Peak portfolio equity

        Returns:
            Tuple of (is_allowed, reason) - fails on first check that doesn't pass
        """
        # Check position size limit
        allowed, reason = self.check_position_size_limit(position_value, portfolio_state.equity)
        if not allowed:
            logger.warning(f"Position size limit check failed: {reason}")
            return False, reason

        # Check position risk limit
        allowed, reason = self.check_position_risk_limit(risk_amount, portfolio_state.equity)
        if not allowed:
            logger.warning(f"Position risk limit check failed: {reason}")
            return False, reason

        # Check portfolio risk limit
        allowed, reason = self.check_portfolio_risk_limit(risk_amount, portfolio_state)
        if not allowed:
            logger.warning(f"Portfolio risk limit check failed: {reason}")
            return False, reason

        # Check max positions limit
        allowed, reason = self.check_max_positions_limit(len(portfolio_state.positions))
        if not allowed:
            logger.warning(f"Max positions limit check failed: {reason}")
            return False, reason

        # Check daily loss limit
        allowed, reason = self.check_daily_loss_limit(portfolio_state.equity, timestamp)
        if not allowed:
            logger.warning(f"Daily loss limit check failed: {reason}")
            return False, reason

        # Check drawdown limit
        allowed, reason = self.check_drawdown_limit(portfolio_state.equity, peak_equity)
        if not allowed:
            logger.warning(f"Drawdown limit check failed: {reason}")
            return False, reason

        logger.info("All risk limit checks passed")
        return True, "All risk checks passed"

    def record_trade(self, realized_pnl: float) -> None:
        """
        Record a completed trade for daily tracking.

        Args:
            realized_pnl: Realized profit/loss from the trade
        """
        if self._daily_tracker is not None:
            self._daily_tracker.trades_count += 1
            self._daily_tracker.realized_pnl += realized_pnl
            logger.debug(
                f"Trade recorded: pnl=${realized_pnl:,.2f}, "
                f"daily_trades={self._daily_tracker.trades_count}, "
                f"daily_pnl=${self._daily_tracker.realized_pnl:,.2f}"
            )

    def get_daily_stats(self) -> Optional[Dict]:
        """
        Get current daily statistics.

        Returns:
            Dictionary of daily stats or None if no tracker
        """
        if self._daily_tracker is None:
            return None

        return {
            'date': self._daily_tracker.date,
            'starting_equity': self._daily_tracker.starting_equity,
            'current_equity': self._daily_tracker.current_equity,
            'daily_return': self._daily_tracker.daily_return,
            'daily_pnl': self._daily_tracker.realized_pnl,
            'trades_count': self._daily_tracker.trades_count,
            'max_loss_hit': self._daily_tracker.max_loss_hit
        }

    def reset_daily_tracker(self) -> None:
        """Reset the daily risk tracker."""
        self._daily_tracker = None
        logger.info("Daily risk tracker reset")


if __name__ == "__main__":
    """
    Validation block for risk limit enforcement.
    Tests all limit checks with realistic trading scenarios.
    """
    import sys
    from datetime import timedelta

    from crypto_trader.backtesting.portfolio import Position
    from crypto_trader.core.types import OrderSide

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating limits.py with real risk scenarios...\n")

    # Test 1: RiskLimitChecker initialization
    total_tests += 1
    print("Test 1: RiskLimitChecker initialization")
    try:
        config = RiskConfig(
            max_position_risk=0.02,
            max_portfolio_risk=0.10,
            max_daily_loss_percent=0.05,
            max_open_positions=3
        )

        checker = RiskLimitChecker(config)

        if checker.config.max_position_risk != 0.02:
            all_validation_failures.append(f"Max position risk not set correctly")

        print(f"  ✓ Max position risk: {checker.config.max_position_risk:.2%}")
        print(f"  ✓ Max portfolio risk: {checker.config.max_portfolio_risk:.2%}")
        print(f"  ✓ Max daily loss: {checker.config.max_daily_loss_percent:.2%}")
    except Exception as e:
        all_validation_failures.append(f"Initialization test exception: {e}")

    # Test 2: Position size limit check - pass
    total_tests += 1
    print("\nTest 2: Position size limit check - within limits")
    try:
        allowed, reason = checker.check_position_size_limit(
            position_value=5000.0,
            portfolio_equity=10000.0
        )

        if not allowed:
            all_validation_failures.append(f"Position size should be allowed: {reason}")

        print(f"  ✓ Allowed: {allowed}")
        print(f"  ✓ Reason: {reason}")
    except Exception as e:
        all_validation_failures.append(f"Position size check test exception: {e}")

    # Test 3: Position risk limit check - fail
    total_tests += 1
    print("\nTest 3: Position risk limit check - exceeds limits")
    try:
        allowed, reason = checker.check_position_risk_limit(
            risk_amount=300.0,  # 3% risk, exceeds 2% limit
            portfolio_equity=10000.0
        )

        if allowed:
            all_validation_failures.append("Excessive risk should not be allowed")

        if "too high" not in reason.lower():
            all_validation_failures.append(f"Reason should mention risk too high: {reason}")

        print(f"  ✓ Correctly rejected: {reason}")
    except Exception as e:
        all_validation_failures.append(f"Position risk check test exception: {e}")

    # Test 4: Portfolio risk limit check
    total_tests += 1
    print("\nTest 4: Portfolio risk limit check")
    try:
        # Create portfolio state with some positions
        timestamp = datetime(2025, 1, 1, 10, 0, 0)

        # Portfolio with 4 positions already - each assumed at 2% risk = 8% total
        portfolio_state = PortfolioState(
            timestamp=timestamp,
            cash=2000.0,
            positions={
                f'SYMBOL{i}': Position(
                    symbol=f'SYMBOL{i}',
                    entry_price=50000.0,
                    quantity=0.1,
                    entry_time=timestamp,
                    side=OrderSide.BUY
                ) for i in range(4)
            },
            equity=10000.0
        )

        # Try to add more risk - would bring to 8% + 3% = 11% > 10% limit
        allowed, reason = checker.check_portfolio_risk_limit(
            new_risk_amount=300.0,  # 3% additional risk
            portfolio_state=portfolio_state
        )

        if allowed:
            all_validation_failures.append("Should reject trade that brings portfolio risk above limit")

        print(f"  ✓ Portfolio risk check correctly rejected")
        print(f"  ✓ Reason: {reason}")
    except Exception as e:
        all_validation_failures.append(f"Portfolio risk check test exception: {e}")

    # Test 5: Max positions limit check
    total_tests += 1
    print("\nTest 5: Max positions limit check")
    try:
        # At limit
        allowed, reason = checker.check_max_positions_limit(current_positions=3)

        if allowed:
            all_validation_failures.append("Should reject when at max positions")

        # Below limit
        allowed2, reason2 = checker.check_max_positions_limit(current_positions=2)

        if not allowed2:
            all_validation_failures.append("Should allow when below max positions")

        print(f"  ✓ At limit (3/3): {allowed} - {reason}")
        print(f"  ✓ Below limit (2/3): {allowed2} - {reason2}")
    except Exception as e:
        all_validation_failures.append(f"Max positions check test exception: {e}")

    # Test 6: Daily loss limit check - normal
    total_tests += 1
    print("\nTest 6: Daily loss limit check - within limits")
    try:
        timestamp = datetime(2025, 1, 1, 10, 0, 0)

        # First check initializes tracker
        allowed, reason = checker.check_daily_loss_limit(
            current_equity=10000.0,
            timestamp=timestamp
        )

        if not allowed:
            all_validation_failures.append(f"First check should pass: {reason}")

        # Small loss - should pass
        allowed2, reason2 = checker.check_daily_loss_limit(
            current_equity=9800.0,  # 2% loss, within 5% limit
            timestamp=timestamp + timedelta(hours=1)
        )

        if not allowed2:
            all_validation_failures.append(f"Small loss should pass: {reason2}")

        print(f"  ✓ Initial: {allowed} - {reason}")
        print(f"  ✓ After 2% loss: {allowed2} - {reason2}")
    except Exception as e:
        all_validation_failures.append(f"Daily loss check test exception: {e}")

    # Test 7: Daily loss limit check - exceeded
    total_tests += 1
    print("\nTest 7: Daily loss limit check - limit exceeded")
    try:
        # Reset for new day
        checker.reset_daily_tracker()
        timestamp = datetime(2025, 1, 2, 10, 0, 0)

        # Initialize with starting equity
        checker.check_daily_loss_limit(10000.0, timestamp)

        # Large loss - should fail
        allowed, reason = checker.check_daily_loss_limit(
            current_equity=9400.0,  # 6% loss, exceeds 5% limit
            timestamp=timestamp + timedelta(hours=2)
        )

        if allowed:
            all_validation_failures.append("Large loss should be rejected")

        if "exceeded" not in reason.lower():
            all_validation_failures.append(f"Reason should mention limit exceeded: {reason}")

        print(f"  ✓ Correctly rejected 6% loss: {reason}")
    except Exception as e:
        all_validation_failures.append(f"Daily loss exceeded test exception: {e}")

    # Test 8: Drawdown limit check
    total_tests += 1
    print("\nTest 8: Drawdown limit check")
    try:
        # Small drawdown - should pass
        allowed, reason = checker.check_drawdown_limit(
            current_equity=9500.0,
            peak_equity=10000.0,
            max_drawdown_limit=0.10  # 10% max drawdown
        )

        if not allowed:
            all_validation_failures.append(f"Small drawdown should pass: {reason}")

        # Large drawdown - should fail
        allowed2, reason2 = checker.check_drawdown_limit(
            current_equity=8500.0,
            peak_equity=10000.0,
            max_drawdown_limit=0.10  # 15% drawdown exceeds 10% limit
        )

        if allowed2:
            all_validation_failures.append("Large drawdown should be rejected")

        print(f"  ✓ 5% drawdown: {allowed} - {reason}")
        print(f"  ✓ 15% drawdown: {allowed2} - {reason2}")
    except Exception as e:
        all_validation_failures.append(f"Drawdown check test exception: {e}")

    # Test 9: Check all limits - comprehensive
    total_tests += 1
    print("\nTest 9: Check all limits - comprehensive")
    try:
        checker.reset_daily_tracker()
        timestamp = datetime(2025, 1, 3, 10, 0, 0)

        portfolio_state = PortfolioState(
            timestamp=timestamp,
            cash=9000.0,
            positions={},
            equity=10000.0
        )

        # Should pass all checks
        allowed, reason = checker.check_all_limits(
            position_value=1000.0,
            risk_amount=100.0,  # 1% risk
            portfolio_state=portfolio_state,
            timestamp=timestamp,
            peak_equity=10000.0
        )

        if not allowed:
            all_validation_failures.append(f"Valid trade should pass all checks: {reason}")

        print(f"  ✓ All checks passed: {allowed}")
        print(f"  ✓ Reason: {reason}")
    except Exception as e:
        all_validation_failures.append(f"Check all limits test exception: {e}")

    # Test 10: Trade recording and daily stats
    total_tests += 1
    print("\nTest 10: Trade recording and daily stats")
    try:
        checker.reset_daily_tracker()
        timestamp = datetime(2025, 1, 4, 10, 0, 0)

        # Initialize tracker
        checker.check_daily_loss_limit(10000.0, timestamp)

        # Record some trades
        checker.record_trade(100.0)
        checker.record_trade(-50.0)
        checker.record_trade(75.0)

        stats = checker.get_daily_stats()

        if stats is None:
            all_validation_failures.append("Daily stats should not be None")
        elif stats['trades_count'] != 3:
            all_validation_failures.append(f"Trade count: Expected 3, got {stats['trades_count']}")
        elif abs(stats['daily_pnl'] - 125.0) > 0.01:
            all_validation_failures.append(f"Daily PnL: Expected 125.0, got {stats['daily_pnl']}")

        print(f"  ✓ Trades recorded: {stats['trades_count']}")
        print(f"  ✓ Daily PnL: ${stats['daily_pnl']:,.2f}")
        print(f"  ✓ Max loss hit: {stats['max_loss_hit']}")
    except Exception as e:
        all_validation_failures.append(f"Trade recording test exception: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Risk limit enforcement validated and ready for use")
        sys.exit(0)
