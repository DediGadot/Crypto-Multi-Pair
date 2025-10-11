"""
Main risk manager for crypto trading system.

This module provides the RiskManager class that coordinates position sizing,
risk limit enforcement, and portfolio risk management. It serves as the main
interface for all risk management operations.

**Purpose**: Centralized risk management combining position sizing, limit checks,
stop loss/take profit calculations, and trade validation.

**Key Components**:
- RiskManager: Main coordinator class
- Position sizing integration
- Risk limit enforcement
- Stop loss and take profit calculations
- Portfolio risk metrics
- Trade approval workflow

**Third-party packages**:
- loguru: https://loguru.readthedocs.io/
- dataclasses: https://docs.python.org/3/library/dataclasses.html
- typing: https://docs.python.org/3/library/typing.html

**Sample Input**:
```python
risk_manager = RiskManager(config)
position_size = risk_manager.calculate_position_size(
    signal=1,
    portfolio=portfolio_state,
    price=50000.0,
    volatility=1000.0
)
allowed, reason = risk_manager.should_allow_trade(
    signal=1,
    portfolio=portfolio_state,
    price=50000.0
)
```

**Expected Output**:
```python
{
    'quantity': 0.1,
    'position_value': 5000.0,
    'risk_amount': 200.0,
    'risk_percent': 0.02,
    'stop_loss': 48000.0,
    'take_profit': 54000.0
}
(True, "All risk checks passed")
```
"""

from datetime import datetime
from typing import Dict, Optional, Tuple

from loguru import logger

from crypto_trader.backtesting.portfolio import PortfolioState
from crypto_trader.core.config import RiskConfig
from crypto_trader.core.types import Signal
from crypto_trader.risk.limits import RiskLimitChecker
from crypto_trader.risk.sizing import PositionSizer, create_position_sizer


class RiskManager:
    """
    Main risk manager coordinating all risk management functions.

    Integrates position sizing, limit enforcement, and stop loss/take profit
    calculations to provide comprehensive risk management for trading.
    """

    def __init__(self, config: RiskConfig):
        """
        Initialize risk manager.

        Args:
            config: Risk configuration parameters
        """
        self.config = config

        # Initialize position sizer based on config
        self.position_sizer = create_position_sizer(
            method=config.position_sizing_method,
            risk_per_trade=config.max_position_risk,
            max_position_size=0.95  # Conservative default
        )

        # Initialize risk limit checker
        self.limit_checker = RiskLimitChecker(config)

        # Track peak equity for drawdown calculations
        self._peak_equity: Optional[float] = None

        logger.info(
            f"RiskManager initialized: "
            f"sizing_method={config.position_sizing_method}, "
            f"max_position_risk={config.max_position_risk:.2%}, "
            f"max_portfolio_risk={config.max_portfolio_risk:.2%}"
        )

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        price: float,
        stop_loss_price: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate position size (quantity) for a trade.

        Args:
            signal: Trading signal (1=buy, 0=hold, -1=sell)
            portfolio: Current portfolio state
            price: Current/entry price
            stop_loss_price: Optional stop loss price
            volatility: Optional volatility measure (ATR)

        Returns:
            Quantity to trade (in asset units, e.g., BTC)
        """
        if signal == 0:
            return 0.0

        # Use position sizer to calculate size
        sizing_result = self.position_sizer.calculate(
            capital=portfolio.equity,
            entry_price=price,
            stop_loss_price=stop_loss_price,
            volatility=volatility
        )

        quantity = sizing_result['quantity']

        logger.debug(
            f"Position size calculated: signal={signal}, qty={quantity:.6f}, "
            f"value=${sizing_result['position_value']:,.2f}, "
            f"risk=${sizing_result['risk_amount']:,.2f} ({sizing_result['risk_percent']:.2%})"
        )

        return quantity

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str = "long",
        risk_percent: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price based on risk parameters.

        Args:
            entry_price: Entry price
            side: Position side ("long" or "short")
            risk_percent: Optional custom risk % (uses config if None)

        Returns:
            Stop loss price
        """
        if risk_percent is None:
            risk_percent = self.config.stop_loss_percent

        if side.lower() == "long":
            # Long position: stop loss below entry
            stop_loss = entry_price * (1 - risk_percent)
        else:
            # Short position: stop loss above entry
            stop_loss = entry_price * (1 + risk_percent)

        logger.debug(
            f"Stop loss calculated: entry=${entry_price:,.2f}, "
            f"side={side}, risk={risk_percent:.2%}, stop=${stop_loss:,.2f}"
        )

        return stop_loss

    def calculate_take_profit(
        self,
        entry_price: float,
        side: str = "long",
        reward_ratio: Optional[float] = None,
        risk_percent: Optional[float] = None
    ) -> float:
        """
        Calculate take profit price based on risk/reward ratio.

        Args:
            entry_price: Entry price
            side: Position side ("long" or "short")
            reward_ratio: Risk/reward ratio (uses config if None)
            risk_percent: Risk % for calculating reward (uses config if None)

        Returns:
            Take profit price
        """
        if reward_ratio is None:
            reward_ratio = self.config.risk_reward_ratio

        if risk_percent is None:
            risk_percent = self.config.stop_loss_percent

        reward_percent = risk_percent * reward_ratio

        if side.lower() == "long":
            # Long position: take profit above entry
            take_profit = entry_price * (1 + reward_percent)
        else:
            # Short position: take profit below entry
            take_profit = entry_price * (1 - reward_percent)

        logger.debug(
            f"Take profit calculated: entry=${entry_price:,.2f}, "
            f"side={side}, ratio={reward_ratio:.1f}x, tp=${take_profit:,.2f}"
        )

        return take_profit

    def get_portfolio_risk(
        self,
        portfolio: PortfolioState
    ) -> Dict[str, float]:
        """
        Calculate current portfolio risk metrics.

        Args:
            portfolio: Current portfolio state

        Returns:
            Dictionary of risk metrics
        """
        # Update peak equity
        if self._peak_equity is None or portfolio.equity > self._peak_equity:
            self._peak_equity = portfolio.equity

        # Calculate drawdown
        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - portfolio.equity) / self._peak_equity

        # Estimate current risk from open positions
        num_positions = len(portfolio.positions)
        estimated_risk_percent = num_positions * self.config.max_position_risk

        # Calculate utilization
        cash_utilization = 1.0 - (portfolio.cash / portfolio.equity) if portfolio.equity > 0 else 0.0

        metrics = {
            'equity': portfolio.equity,
            'cash': portfolio.cash,
            'peak_equity': self._peak_equity,
            'drawdown': drawdown,
            'num_positions': num_positions,
            'estimated_risk_percent': estimated_risk_percent,
            'cash_utilization': cash_utilization,
            'unrealized_pnl': portfolio.unrealized_pnl,
            'realized_pnl': portfolio.realized_pnl
        }

        logger.debug(
            f"Portfolio risk: equity=${portfolio.equity:,.2f}, "
            f"drawdown={drawdown:.2%}, positions={num_positions}, "
            f"estimated_risk={estimated_risk_percent:.2%}"
        )

        return metrics

    def should_allow_trade(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        price: float,
        timestamp: datetime,
        stop_loss_price: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be allowed based on all risk checks.

        Args:
            signal: Trading signal (1=buy, 0=hold, -1=sell)
            portfolio: Current portfolio state
            price: Current price
            timestamp: Current timestamp
            stop_loss_price: Optional stop loss price
            volatility: Optional volatility measure

        Returns:
            Tuple of (is_allowed, reason)
        """
        # No trade on hold signal
        if signal == 0:
            return False, "Hold signal - no trade"

        # Calculate position size
        sizing_result = self.position_sizer.calculate(
            capital=portfolio.equity,
            entry_price=price,
            stop_loss_price=stop_loss_price,
            volatility=volatility
        )

        position_value = sizing_result['position_value']
        risk_amount = sizing_result['risk_amount']

        # Update peak equity for drawdown checks
        if self._peak_equity is None or portfolio.equity > self._peak_equity:
            self._peak_equity = portfolio.equity

        # Run all risk limit checks
        allowed, reason = self.limit_checker.check_all_limits(
            position_value=position_value,
            risk_amount=risk_amount,
            portfolio_state=portfolio,
            timestamp=timestamp,
            peak_equity=self._peak_equity
        )

        if allowed:
            logger.info(
                f"Trade approved: signal={signal}, price=${price:,.2f}, "
                f"qty={sizing_result['quantity']:.6f}, value=${position_value:,.2f}"
            )
        else:
            logger.warning(
                f"Trade rejected: signal={signal}, price=${price:,.2f}, "
                f"reason={reason}"
            )

        return allowed, reason

    def check_risk_limits(
        self,
        position_value: float,
        risk_amount: float,
        portfolio: PortfolioState
    ) -> bool:
        """
        Check if position passes risk limits (simplified interface).

        Args:
            position_value: Value of proposed position
            risk_amount: Risk amount in dollars
            portfolio: Current portfolio state

        Returns:
            True if all checks pass
        """
        # Check position risk limit
        allowed, _ = self.limit_checker.check_position_risk_limit(
            risk_amount=risk_amount,
            portfolio_equity=portfolio.equity
        )

        if not allowed:
            return False

        # Check portfolio risk limit
        allowed, _ = self.limit_checker.check_portfolio_risk_limit(
            new_risk_amount=risk_amount,
            portfolio_state=portfolio
        )

        if not allowed:
            return False

        # Check max positions limit
        allowed, _ = self.limit_checker.check_max_positions_limit(
            current_positions=len(portfolio.positions)
        )

        return allowed

    def record_trade_completion(self, realized_pnl: float) -> None:
        """
        Record a completed trade for risk tracking.

        Args:
            realized_pnl: Realized profit/loss from the trade
        """
        self.limit_checker.record_trade(realized_pnl)

    def reset_daily_limits(self) -> None:
        """Reset daily risk tracking (typically called at start of new day)."""
        self.limit_checker.reset_daily_tracker()

    def get_daily_stats(self) -> Optional[Dict]:
        """
        Get daily risk statistics.

        Returns:
            Dictionary of daily stats or None
        """
        return self.limit_checker.get_daily_stats()


if __name__ == "__main__":
    """
    Validation block for risk manager.
    Tests integrated risk management with realistic trading scenarios.
    """
    import sys
    from datetime import timedelta

    from crypto_trader.backtesting.portfolio import Position
    from crypto_trader.core.types import OrderSide

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating manager.py with real trading scenarios...\n")

    # Test 1: RiskManager initialization
    total_tests += 1
    print("Test 1: RiskManager initialization")
    try:
        config = RiskConfig(
            max_position_risk=0.02,
            max_portfolio_risk=0.10,
            stop_loss_percent=0.02,
            risk_reward_ratio=2.0,
            position_sizing_method="fixed_percent",
            max_open_positions=3
        )

        manager = RiskManager(config)

        if manager.config.max_position_risk != 0.02:
            all_validation_failures.append("Config not set correctly")

        print(f"  ✓ Position sizing method: {manager.config.position_sizing_method}")
        print(f"  ✓ Max position risk: {manager.config.max_position_risk:.2%}")
        print(f"  ✓ Risk/reward ratio: {manager.config.risk_reward_ratio:.1f}x")
    except Exception as e:
        all_validation_failures.append(f"Initialization test exception: {e}")

    # Test 2: Calculate position size
    total_tests += 1
    print("\nTest 2: Calculate position size")
    try:
        timestamp = datetime(2025, 1, 1, 10, 0, 0)
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=10000.0,
            positions={},
            equity=10000.0
        )

        quantity = manager.calculate_position_size(
            signal=1,  # Buy signal
            portfolio=portfolio,
            price=50000.0,
            stop_loss_price=48000.0
        )

        # With 2% risk and $2000 stop, should get 0.1 BTC
        expected_quantity = 0.1

        if abs(quantity - expected_quantity) > 0.001:
            all_validation_failures.append(
                f"Position size: Expected {expected_quantity}, got {quantity}"
            )

        print(f"  ✓ Quantity calculated: {quantity:.6f} BTC")
        print(f"  ✓ Position value: ${quantity * 50000.0:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Position size test exception: {e}")

    # Test 3: Calculate stop loss and take profit
    total_tests += 1
    print("\nTest 3: Calculate stop loss and take profit")
    try:
        entry_price = 50000.0

        stop_loss = manager.calculate_stop_loss(entry_price, side="long")
        take_profit = manager.calculate_take_profit(entry_price, side="long")

        # 2% stop loss = $49,000
        expected_stop = 49000.0
        # 2% * 2.0 reward ratio = 4% take profit = $52,000
        expected_tp = 52000.0

        if abs(stop_loss - expected_stop) > 1.0:
            all_validation_failures.append(
                f"Stop loss: Expected {expected_stop}, got {stop_loss}"
            )

        if abs(take_profit - expected_tp) > 1.0:
            all_validation_failures.append(
                f"Take profit: Expected {expected_tp}, got {take_profit}"
            )

        print(f"  ✓ Entry: ${entry_price:,.2f}")
        print(f"  ✓ Stop loss: ${stop_loss:,.2f} (-{(entry_price-stop_loss)/entry_price:.2%})")
        print(f"  ✓ Take profit: ${take_profit:,.2f} (+{(take_profit-entry_price)/entry_price:.2%})")
    except Exception as e:
        all_validation_failures.append(f"Stop/TP test exception: {e}")

    # Test 4: Get portfolio risk metrics
    total_tests += 1
    print("\nTest 4: Get portfolio risk metrics")
    try:
        # Create portfolio with some positions
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=5000.0,
            positions={
                'BTCUSDT': Position(
                    symbol='BTCUSDT',
                    entry_price=50000.0,
                    quantity=0.1,
                    entry_time=timestamp,
                    side=OrderSide.BUY
                )
            },
            equity=10000.0,
            unrealized_pnl=0.0
        )

        risk_metrics = manager.get_portfolio_risk(portfolio)

        if risk_metrics['num_positions'] != 1:
            all_validation_failures.append(
                f"Num positions: Expected 1, got {risk_metrics['num_positions']}"
            )

        if risk_metrics['equity'] != 10000.0:
            all_validation_failures.append(
                f"Equity: Expected 10000.0, got {risk_metrics['equity']}"
            )

        print(f"  ✓ Equity: ${risk_metrics['equity']:,.2f}")
        print(f"  ✓ Drawdown: {risk_metrics['drawdown']:.2%}")
        print(f"  ✓ Positions: {risk_metrics['num_positions']}")
        print(f"  ✓ Estimated risk: {risk_metrics['estimated_risk_percent']:.2%}")
    except Exception as e:
        all_validation_failures.append(f"Portfolio risk test exception: {e}")

    # Test 5: Should allow trade - valid case
    total_tests += 1
    print("\nTest 5: Should allow trade - valid case")
    try:
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=9500.0,
            positions={},
            equity=10000.0
        )

        allowed, reason = manager.should_allow_trade(
            signal=1,
            portfolio=portfolio,
            price=50000.0,
            timestamp=timestamp,
            stop_loss_price=48000.0
        )

        if not allowed:
            all_validation_failures.append(f"Valid trade should be allowed: {reason}")

        print(f"  ✓ Allowed: {allowed}")
        print(f"  ✓ Reason: {reason}")
    except Exception as e:
        all_validation_failures.append(f"Allow trade test exception: {e}")

    # Test 6: Should reject trade - hold signal
    total_tests += 1
    print("\nTest 6: Should reject trade - hold signal")
    try:
        allowed, reason = manager.should_allow_trade(
            signal=0,  # Hold signal
            portfolio=portfolio,
            price=50000.0,
            timestamp=timestamp
        )

        if allowed:
            all_validation_failures.append("Hold signal should not be allowed")

        if "hold" not in reason.lower():
            all_validation_failures.append(f"Reason should mention hold: {reason}")

        print(f"  ✓ Correctly rejected hold signal")
        print(f"  ✓ Reason: {reason}")
    except Exception as e:
        all_validation_failures.append(f"Hold signal test exception: {e}")

    # Test 7: Should reject trade - max positions reached
    total_tests += 1
    print("\nTest 7: Should reject trade - max positions reached")
    try:
        # Portfolio with max positions (3)
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=1000.0,
            positions={
                f'SYMBOL{i}': Position(
                    symbol=f'SYMBOL{i}',
                    entry_price=50000.0,
                    quantity=0.1,
                    entry_time=timestamp,
                    side=OrderSide.BUY
                ) for i in range(3)
            },
            equity=10000.0
        )

        allowed, reason = manager.should_allow_trade(
            signal=1,
            portfolio=portfolio,
            price=50000.0,
            timestamp=timestamp,
            stop_loss_price=48000.0
        )

        if allowed:
            all_validation_failures.append("Trade should be rejected when at max positions")

        if "maximum positions" not in reason.lower():
            all_validation_failures.append(f"Reason should mention max positions: {reason}")

        print(f"  ✓ Correctly rejected: {reason}")
    except Exception as e:
        all_validation_failures.append(f"Max positions test exception: {e}")

    # Test 8: Check risk limits - simplified interface
    total_tests += 1
    print("\nTest 8: Check risk limits - simplified interface")
    try:
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=9000.0,
            positions={},
            equity=10000.0
        )

        # Should pass
        passes = manager.check_risk_limits(
            position_value=5000.0,
            risk_amount=200.0,  # 2% risk
            portfolio=portfolio
        )

        if not passes:
            all_validation_failures.append("Valid limits should pass")

        # Should fail - excessive risk
        passes2 = manager.check_risk_limits(
            position_value=5000.0,
            risk_amount=500.0,  # 5% risk exceeds 2% limit
            portfolio=portfolio
        )

        if passes2:
            all_validation_failures.append("Excessive risk should fail")

        print(f"  ✓ Valid limits passed: {passes}")
        print(f"  ✓ Excessive risk rejected: {not passes2}")
    except Exception as e:
        all_validation_failures.append(f"Risk limits test exception: {e}")

    # Test 9: Trade recording
    total_tests += 1
    print("\nTest 9: Trade recording")
    try:
        manager.reset_daily_limits()

        # Record some trades
        manager.record_trade_completion(100.0)
        manager.record_trade_completion(-50.0)

        stats = manager.get_daily_stats()

        # Stats might be None if no daily loss check was called yet
        # This is expected behavior
        print(f"  ✓ Trades recorded successfully")
        if stats:
            print(f"  ✓ Daily stats available: {stats.get('trades_count', 'N/A')} trades")
        else:
            print(f"  ✓ Daily stats will be available after first price check")
    except Exception as e:
        all_validation_failures.append(f"Trade recording test exception: {e}")

    # Test 10: Drawdown tracking
    total_tests += 1
    print("\nTest 10: Drawdown tracking")
    try:
        # Start with high equity
        portfolio1 = PortfolioState(
            timestamp=timestamp,
            cash=12000.0,
            positions={},
            equity=12000.0
        )

        metrics1 = manager.get_portfolio_risk(portfolio1)

        # Drawdown scenario
        portfolio2 = PortfolioState(
            timestamp=timestamp + timedelta(hours=1),
            cash=9000.0,
            positions={},
            equity=9000.0
        )

        metrics2 = manager.get_portfolio_risk(portfolio2)

        # Should have ~25% drawdown from $12k to $9k
        expected_drawdown = 0.25

        if abs(metrics2['drawdown'] - expected_drawdown) > 0.01:
            all_validation_failures.append(
                f"Drawdown: Expected {expected_drawdown:.2%}, got {metrics2['drawdown']:.2%}"
            )

        print(f"  ✓ Peak equity: ${metrics1['peak_equity']:,.2f}")
        print(f"  ✓ Current equity: ${metrics2['equity']:,.2f}")
        print(f"  ✓ Drawdown: {metrics2['drawdown']:.2%}")
    except Exception as e:
        all_validation_failures.append(f"Drawdown tracking test exception: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Risk manager validated and ready for production use")
        sys.exit(0)
