"""
Portfolio Management for Backtesting

This module manages portfolio state during backtesting including position tracking,
equity curve calculation, and cash management. It provides real-time portfolio
metrics and position sizing logic.

**Purpose**: Track portfolio value, positions, and cash through the backtesting period
with support for long-only and long-short strategies.

**Key Components**:
- PortfolioState: Dataclass for current portfolio state
- PortfolioManager: Manages portfolio through backtest lifecycle
- Position sizing calculations
- Equity curve tracking

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- dataclasses: https://docs.python.org/3/library/dataclasses.html

**Sample Input**:
```python
manager = PortfolioManager(initial_capital=10000.0, max_position_size=0.95)
manager.update_position(entry_price=50000.0, quantity=0.1, side="BUY")
manager.calculate_equity(current_price=51000.0)
```

**Expected Output**:
```python
{
    'equity': 10100.0,
    'cash': 5000.0,
    'position_value': 5100.0,
    'unrealized_pnl': 100.0,
    'total_return': 0.01
}
```
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from crypto_trader.core.types import OrderSide


@dataclass
class Position:
    """
    Represents a single open position.

    Attributes:
        symbol: Trading pair
        entry_price: Average entry price
        quantity: Position size (positive for long, negative for short)
        entry_time: When position was opened
        side: BUY (long) or SELL (short)
    """
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    side: OrderSide
    entry_fee: float = 0.0

    @property
    def market_value(self) -> float:
        """Calculate current market value at entry price."""
        return abs(self.quantity) * self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss at current price."""
        abs_quantity = abs(self.quantity)
        if self.side == OrderSide.BUY:
            # Long position
            return abs_quantity * (current_price - self.entry_price)
        else:
            # Short position
            return abs_quantity * (self.entry_price - current_price)


@dataclass
class PortfolioState:
    """
    Current state of the portfolio.

    Attributes:
        timestamp: Current time
        cash: Available cash
        positions: Dictionary of open positions by symbol
        equity: Total portfolio value (cash + positions)
        unrealized_pnl: Total unrealized profit/loss
        realized_pnl: Total realized profit/loss
    """
    timestamp: datetime
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    equity: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_equity(self, current_prices: Dict[str, float]) -> None:
        """
        Update equity and unrealized PnL based on current prices.

        Args:
            current_prices: Dictionary mapping symbols to current prices
        """
        total_unrealized = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_unrealized += position.unrealized_pnl(current_prices[symbol])

        self.unrealized_pnl = total_unrealized
        self.equity = self.cash + sum(
            abs(pos.quantity) * current_prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )


class PortfolioManager:
    """
    Manages portfolio state during backtesting.

    Tracks cash, positions, equity curve, and provides position sizing
    functionality. Supports both long-only and long-short strategies.
    """

    def __init__(
        self,
        initial_capital: float,
        max_position_size: float = 0.95,
        enable_short_selling: bool = False
    ):
        """
        Initialize portfolio manager.

        Args:
            initial_capital: Starting portfolio value
            max_position_size: Maximum position size as % of equity (0-1)
            enable_short_selling: Whether to allow short positions
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.enable_short_selling = enable_short_selling

        # Initialize state
        self._cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._realized_pnl = 0.0

        logger.info(
            f"PortfolioManager initialized: capital=${initial_capital:,.2f}, "
            f"max_position={max_position_size:.1%}, shorts={enable_short_selling}"
        )

    @property
    def cash(self) -> float:
        """Get available cash."""
        return self._cash

    @property
    def positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return self._positions.copy()

    @property
    def equity_curve(self) -> List[Tuple[datetime, float]]:
        """Get equity curve history."""
        return self._equity_curve.copy()

    def get_state(self, timestamp: datetime, current_prices: Dict[str, float]) -> PortfolioState:
        """
        Get current portfolio state.

        Args:
            timestamp: Current timestamp
            current_prices: Dictionary of current prices by symbol

        Returns:
            PortfolioState object with current portfolio metrics
        """
        state = PortfolioState(
            timestamp=timestamp,
            cash=self._cash,
            positions=self._positions.copy(),
            realized_pnl=self._realized_pnl
        )
        state.update_equity(current_prices)
        return state

    def calculate_position_size(
        self,
        price: float,
        equity: float,
        position_percent: Optional[float] = None
    ) -> float:
        """
        Calculate position size (quantity) based on available equity.

        Args:
            price: Current price per unit
            equity: Current portfolio equity
            position_percent: Optional specific position size % (uses max if None)

        Returns:
            Quantity to trade
        """
        if position_percent is None:
            position_percent = self.max_position_size
        else:
            position_percent = min(position_percent, self.max_position_size)

        position_value = equity * position_percent
        quantity = position_value / price

        return quantity

    def can_open_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float
    ) -> bool:
        """
        Check if we can open a new position.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Desired quantity
            price: Current price

        Returns:
            True if position can be opened
        """
        # Check if short selling is allowed
        if side == OrderSide.SELL and not self.enable_short_selling:
            logger.warning(f"Short selling not enabled for {symbol}")
            return False

        # Check if we already have a position
        if symbol in self._positions:
            logger.warning(f"Already have open position in {symbol}")
            return False

        # Check if we have enough cash
        required_cash = quantity * price
        if required_cash > self._cash:
            logger.warning(
                f"Insufficient cash for {symbol}: need ${required_cash:,.2f}, "
                f"have ${self._cash:,.2f}"
            )
            return False

        return True

    def open_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        timestamp: datetime,
        fees: float = 0.0
    ) -> bool:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Quantity to trade
            price: Execution price
            timestamp: Time of trade
            fees: Transaction fees

        Returns:
            True if position opened successfully
        """
        if not self.can_open_position(symbol, side, quantity, price):
            return False

        trade_value = quantity * price
        stored_quantity = quantity if side == OrderSide.BUY else -quantity

        if side == OrderSide.BUY:
            total_cost = trade_value + fees
            self._cash -= total_cost
        else:
            total_proceeds = trade_value - fees
            self._cash += total_proceeds

        # Create position
        position = Position(
            symbol=symbol,
            entry_price=price,
            quantity=stored_quantity,
            entry_time=timestamp,
            side=side,
            entry_fee=fees
        )

        self._positions[symbol] = position

        logger.debug(
            f"Opened {side.value} position: {symbol} qty={quantity:.6f} "
            f"@ ${price:.2f}, fees=${fees:.2f}"
        )

        return True

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        fees: float = 0.0
    ) -> Optional[float]:
        """
        Close an existing position.

        Args:
            symbol: Trading symbol
            price: Exit price
            timestamp: Time of trade
            fees: Transaction fees

        Returns:
            Realized PnL if position closed, None if no position exists
        """
        if symbol not in self._positions:
            logger.warning(f"No open position for {symbol}")
            return None

        position = self._positions[symbol]

        abs_quantity = abs(position.quantity)
        if position.side == OrderSide.BUY:
            # Long position
            exit_proceeds = abs_quantity * price - fees
            entry_cost = abs_quantity * position.entry_price + position.entry_fee
            pnl = exit_proceeds - entry_cost
            self._cash += exit_proceeds
        else:
            # Short position
            exit_cost = abs_quantity * price + fees
            entry_proceeds = abs_quantity * position.entry_price - position.entry_fee
            pnl = entry_proceeds - exit_cost
            self._cash -= exit_cost

        # Update realized PnL
        self._realized_pnl += pnl

        # Remove position
        del self._positions[symbol]

        logger.debug(
            f"Closed position: {symbol} @ ${price:.2f}, "
            f"PnL=${pnl:.2f}, fees=${fees:.2f}"
        )

        return pnl

    def update_equity_curve(self, timestamp: datetime, current_prices: Dict[str, float]) -> None:
        """
        Record current equity in the equity curve.

        Args:
            timestamp: Current timestamp
            current_prices: Dictionary of current prices
        """
        state = self.get_state(timestamp, current_prices)
        self._equity_curve.append((timestamp, state.equity))

    def get_equity_dataframe(self) -> pd.DataFrame:
        """
        Get equity curve as a DataFrame.

        Returns:
            DataFrame with timestamp index and equity values
        """
        if not self._equity_curve:
            return pd.DataFrame(columns=['equity'])

        df = pd.DataFrame(self._equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate current portfolio metrics.

        Returns:
            Dictionary of portfolio metrics
        """
        if not self._equity_curve:
            current_equity = self.initial_capital
        else:
            current_equity = self._equity_curve[-1][1]

        total_return = (current_equity - self.initial_capital) / self.initial_capital

        return {
            'initial_capital': self.initial_capital,
            'current_equity': current_equity,
            'cash': self._cash,
            'total_return': total_return,
            'realized_pnl': self._realized_pnl,
            'num_positions': len(self._positions)
        }


if __name__ == "__main__":
    """
    Validation block for portfolio management.
    Tests portfolio operations with real trading scenarios.
    """
    import sys
    from datetime import timedelta

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating portfolio.py with real trading scenarios...\n")

    # Test 1: PortfolioManager initialization
    total_tests += 1
    print("Test 1: PortfolioManager initialization")
    try:
        manager = PortfolioManager(
            initial_capital=10000.0,
            max_position_size=0.95,
            enable_short_selling=False
        )

        if manager.cash != 10000.0:
            all_validation_failures.append(f"Initial cash: Expected 10000.0, got {manager.cash}")
        if manager.max_position_size != 0.95:
            all_validation_failures.append(f"Max position: Expected 0.95, got {manager.max_position_size}")
        if len(manager.positions) != 0:
            all_validation_failures.append(f"Initial positions: Expected 0, got {len(manager.positions)}")

        print(f"  ✓ Initial capital: ${manager.cash:,.2f}")
        print(f"  ✓ Max position size: {manager.max_position_size:.1%}")
    except Exception as e:
        all_validation_failures.append(f"Initialization test exception: {e}")

    # Test 2: Position size calculation
    total_tests += 1
    print("\nTest 2: Position size calculation")
    try:
        price = 50000.0
        equity = 10000.0

        quantity = manager.calculate_position_size(price, equity)
        expected_value = equity * 0.95  # max_position_size = 0.95
        expected_quantity = expected_value / price

        if abs(quantity - expected_quantity) > 0.000001:
            all_validation_failures.append(
                f"Position size: Expected {expected_quantity:.6f}, got {quantity:.6f}"
            )

        print(f"  ✓ Quantity for ${price:,.0f}: {quantity:.6f} BTC")
        print(f"  ✓ Position value: ${quantity * price:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Position size test exception: {e}")

    # Test 3: Open long position
    total_tests += 1
    print("\nTest 3: Open long position")
    try:
        timestamp = datetime(2025, 1, 1, 10, 0, 0)
        symbol = "BTCUSDT"
        price = 50000.0
        quantity = 0.1
        fees = 5.0

        success = manager.open_position(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            fees=fees
        )

        if not success:
            all_validation_failures.append("Open position: Expected success=True")

        expected_cash = 10000.0 - (quantity * price + fees)
        if abs(manager.cash - expected_cash) > 0.01:
            all_validation_failures.append(
                f"Cash after open: Expected {expected_cash:.2f}, got {manager.cash:.2f}"
            )

        if symbol not in manager.positions:
            all_validation_failures.append(f"Position not found: {symbol}")

        print(f"  ✓ Position opened: {symbol} qty={quantity}")
        print(f"  ✓ Remaining cash: ${manager.cash:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Open position test exception: {e}")

    # Test 4: Portfolio state and equity calculation
    total_tests += 1
    print("\nTest 4: Portfolio state and equity calculation")
    try:
        current_price = 51000.0
        current_prices = {symbol: current_price}

        state = manager.get_state(timestamp, current_prices)

        expected_unrealized = quantity * (current_price - price)
        if abs(state.unrealized_pnl - expected_unrealized) > 0.01:
            all_validation_failures.append(
                f"Unrealized PnL: Expected {expected_unrealized:.2f}, got {state.unrealized_pnl:.2f}"
            )

        expected_equity = manager.cash + (quantity * current_price)
        if abs(state.equity - expected_equity) > 0.01:
            all_validation_failures.append(
                f"Equity: Expected {expected_equity:.2f}, got {state.equity:.2f}"
            )

        print(f"  ✓ Unrealized PnL: ${state.unrealized_pnl:,.2f}")
        print(f"  ✓ Total equity: ${state.equity:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Portfolio state test exception: {e}")

    # Test 5: Update equity curve
    total_tests += 1
    print("\nTest 5: Update equity curve")
    try:
        manager.update_equity_curve(timestamp, current_prices)

        if len(manager.equity_curve) != 1:
            all_validation_failures.append(
                f"Equity curve length: Expected 1, got {len(manager.equity_curve)}"
            )

        curve_time, curve_equity = manager.equity_curve[0]
        if curve_time != timestamp:
            all_validation_failures.append(
                f"Equity curve timestamp mismatch: {curve_time} != {timestamp}"
            )

        print(f"  ✓ Equity curve updated: {len(manager.equity_curve)} points")
        print(f"  ✓ Current equity: ${curve_equity:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Equity curve test exception: {e}")

    # Test 6: Close position
    total_tests += 1
    print("\nTest 6: Close position")
    try:
        exit_price = 52000.0
        exit_timestamp = timestamp + timedelta(hours=2)
        exit_fees = 5.2

        pnl = manager.close_position(
            symbol=symbol,
            price=exit_price,
            timestamp=exit_timestamp,
            fees=exit_fees
        )

        if pnl is None:
            all_validation_failures.append("Close position: Expected PnL value, got None")
        else:
            expected_pnl = quantity * (exit_price - price) - exit_fees
            if abs(pnl - expected_pnl) > 0.01:
                all_validation_failures.append(
                    f"Realized PnL: Expected {expected_pnl:.2f}, got {pnl:.2f}"
                )

        if symbol in manager.positions:
            all_validation_failures.append(f"Position still exists: {symbol}")

        print(f"  ✓ Position closed with PnL: ${pnl:.2f}")
        print(f"  ✓ Cash after close: ${manager.cash:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Close position test exception: {e}")

    # Test 7: Get portfolio metrics
    total_tests += 1
    print("\nTest 7: Get portfolio metrics")
    try:
        metrics = manager.get_metrics()

        required_keys = {'initial_capital', 'current_equity', 'cash', 'total_return', 'realized_pnl', 'num_positions'}
        if not required_keys.issubset(metrics.keys()):
            all_validation_failures.append(
                f"Metrics keys: Missing {required_keys - set(metrics.keys())}"
            )

        if metrics['num_positions'] != 0:
            all_validation_failures.append(f"Open positions: Expected 0, got {metrics['num_positions']}")

        print(f"  ✓ Total return: {metrics['total_return']:.2%}")
        print(f"  ✓ Realized PnL: ${metrics['realized_pnl']:,.2f}")
        print(f"  ✓ Open positions: {metrics['num_positions']}")
    except Exception as e:
        all_validation_failures.append(f"Metrics test exception: {e}")

    # Test 8: Cannot open position without cash
    total_tests += 1
    print("\nTest 8: Insufficient cash handling")
    try:
        # Try to open position larger than available cash
        large_quantity = 1.0  # Would cost $52,000 but we only have ~$10,200

        success = manager.open_position(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            quantity=large_quantity,
            price=52000.0,
            timestamp=exit_timestamp,
            fees=5.0
        )

        if success:
            all_validation_failures.append("Insufficient cash: Expected failure, got success")

        print(f"  ✓ Correctly rejected trade with insufficient cash")
    except Exception as e:
        all_validation_failures.append(f"Insufficient cash test exception: {e}")

    # Test 9: Short selling disabled
    total_tests += 1
    print("\nTest 9: Short selling restrictions")
    try:
        success = manager.open_position(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            price=50000.0,
            timestamp=exit_timestamp,
            fees=5.0
        )

        if success:
            all_validation_failures.append("Short selling: Expected failure when disabled, got success")

        print(f"  ✓ Correctly rejected short sale when disabled")
    except Exception as e:
        all_validation_failures.append(f"Short selling test exception: {e}")

    # Test 10: Enable short selling
    total_tests += 1
    print("\nTest 10: Short selling enabled")
    try:
        short_manager = PortfolioManager(
            initial_capital=10000.0,
            max_position_size=0.95,
            enable_short_selling=True
        )

        success = short_manager.open_position(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.1,
            price=50000.0,
            timestamp=timestamp,
            fees=5.0
        )

        if not success:
            all_validation_failures.append("Short selling: Expected success when enabled, got failure")

        # Close short position with profit (price goes down)
        pnl = short_manager.close_position(
            symbol="BTCUSDT",
            price=48000.0,  # Price went down = profit on short
            timestamp=exit_timestamp,
            fees=5.0
        )

        if pnl is None:
            all_validation_failures.append("Short close: Expected PnL value, got None")
        elif pnl <= 0:
            all_validation_failures.append(f"Short PnL: Expected profit (price down), got {pnl:.2f}")

        print(f"  ✓ Short position opened and closed successfully")
        print(f"  ✓ Short PnL (price down): ${pnl:.2f}")
    except Exception as e:
        all_validation_failures.append(f"Short selling enabled test exception: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Portfolio management is validated and ready for backtesting")
        sys.exit(0)
