"""
Order Execution Simulation for Backtesting

This module simulates order execution with realistic fees, slippage, and market
impact. It tracks all executed trades and generates trade history for analysis.

**Purpose**: Provide realistic order execution simulation for backtesting that
accounts for real-world trading costs and market conditions.

**Key Components**:
- OrderExecutor: Simulates order execution with fees and slippage
- Trade tracking and history generation
- Realistic execution price calculation
- Support for market and limit orders

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```python
executor = OrderExecutor(fee_percent=0.001, slippage_percent=0.0005)
trade = executor.execute_order(
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.1,
    price=50000.0,
    timestamp=datetime.now()
)
```

**Expected Output**:
```python
{
    'symbol': 'BTCUSDT',
    'side': 'BUY',
    'quantity': 0.1,
    'execution_price': 50025.0,  # price + slippage
    'fees': 5.0,
    'total_cost': 5007.5,
    'timestamp': datetime(...)
}
```
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from crypto_trader.core.types import OrderSide, OrderType, Trade


@dataclass
class ExecutionResult:
    """
    Result of an order execution.

    Attributes:
        symbol: Trading pair
        side: BUY or SELL
        order_type: MARKET or LIMIT
        quantity: Amount traded
        requested_price: Original requested price
        execution_price: Actual execution price (after slippage)
        fees: Transaction fees
        total_cost: Total cost including fees
        timestamp: Execution timestamp
        slippage: Applied slippage amount
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    requested_price: float
    execution_price: float
    fees: float
    total_cost: float
    timestamp: datetime
    slippage: float


class OrderExecutor:
    """
    Simulates order execution with realistic market conditions.

    Applies fees, slippage, and tracks all executed trades. Supports both
    market and limit orders with configurable execution parameters.
    """

    def __init__(
        self,
        fee_percent: float = 0.001,
        slippage_percent: float = 0.0005,
        use_maker_taker_fees: bool = False,
        maker_fee: float = 0.0,
        taker_fee: float = 0.001
    ):
        """
        Initialize order executor.

        Args:
            fee_percent: Trading fee as percentage (0.001 = 0.1%)
            slippage_percent: Slippage as percentage (0.0005 = 0.05%)
            use_maker_taker_fees: Use different fees for maker/taker
            maker_fee: Maker fee percentage (used if use_maker_taker_fees=True)
            taker_fee: Taker fee percentage (used if use_maker_taker_fees=True)
        """
        self.fee_percent = fee_percent
        self.slippage_percent = slippage_percent
        self.use_maker_taker_fees = use_maker_taker_fees
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        self._trade_history: List[ExecutionResult] = []

        logger.info(
            f"OrderExecutor initialized: fee={fee_percent:.4%}, "
            f"slippage={slippage_percent:.4%}"
        )

    def calculate_slippage(self, price: float, side: OrderSide) -> float:
        """
        Calculate slippage amount for an order.

        Args:
            price: Base price
            side: BUY or SELL

        Returns:
            Slippage amount to add/subtract from price
        """
        # For buys, slippage increases price
        # For sells, slippage decreases price
        slippage = price * self.slippage_percent

        if side == OrderSide.BUY:
            return slippage
        else:
            return -slippage

    def calculate_fees(
        self,
        quantity: float,
        price: float,
        order_type: OrderType
    ) -> float:
        """
        Calculate transaction fees.

        Args:
            quantity: Trade quantity
            price: Execution price
            order_type: MARKET or LIMIT

        Returns:
            Fee amount in quote currency
        """
        trade_value = quantity * price

        if self.use_maker_taker_fees:
            # Market orders are takers, limit orders are makers
            fee_rate = self.taker_fee if order_type == OrderType.MARKET else self.maker_fee
        else:
            fee_rate = self.fee_percent

        return trade_value * fee_rate

    def execute_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        timestamp: datetime,
        order_type: OrderType = OrderType.MARKET
    ) -> ExecutionResult:
        """
        Execute an order with fees and slippage.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Amount to trade
            price: Base price
            timestamp: Execution time
            order_type: MARKET or LIMIT

        Returns:
            ExecutionResult with all execution details
        """
        # Calculate slippage
        slippage = self.calculate_slippage(price, side)
        execution_price = price + slippage

        # Calculate fees
        fees = self.calculate_fees(quantity, execution_price, order_type)

        # Calculate total cost
        if side == OrderSide.BUY:
            total_cost = (quantity * execution_price) + fees
        else:
            # For sells, fees reduce proceeds
            total_cost = (quantity * execution_price) - fees

        # Create execution result
        result = ExecutionResult(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            requested_price=price,
            execution_price=execution_price,
            fees=fees,
            total_cost=total_cost,
            timestamp=timestamp,
            slippage=slippage
        )

        # Record in history
        self._trade_history.append(result)

        logger.debug(
            f"Executed {side.value} {symbol}: qty={quantity:.6f} @ "
            f"${execution_price:.2f}, fees=${fees:.2f}"
        )

        return result

    def create_trade_record(
        self,
        entry_execution: ExecutionResult,
        exit_execution: ExecutionResult
    ) -> Trade:
        """
        Create a Trade record from entry and exit executions.

        Args:
            entry_execution: Entry order execution
            exit_execution: Exit order execution

        Returns:
            Trade object with complete trade information
        """
        # Calculate PnL
        if entry_execution.side == OrderSide.BUY:
            # Long trade
            pnl = (exit_execution.execution_price - entry_execution.execution_price) * entry_execution.quantity
        else:
            # Short trade
            pnl = (entry_execution.execution_price - exit_execution.execution_price) * entry_execution.quantity

        # Subtract fees
        total_fees = entry_execution.fees + exit_execution.fees
        pnl -= total_fees

        # Calculate PnL percentage
        pnl_percent = (pnl / entry_execution.total_cost) * 100

        trade = Trade(
            symbol=entry_execution.symbol,
            entry_time=entry_execution.timestamp,
            exit_time=exit_execution.timestamp,
            entry_price=entry_execution.execution_price,
            exit_price=exit_execution.execution_price,
            side=entry_execution.side,
            quantity=entry_execution.quantity,
            pnl=pnl,
            pnl_percent=pnl_percent,
            fees=total_fees,
            order_type=entry_execution.order_type
        )

        return trade

    def get_trade_history(self) -> List[ExecutionResult]:
        """
        Get history of all executed orders.

        Returns:
            List of ExecutionResult objects
        """
        return self._trade_history.copy()

    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Get trade history as a DataFrame.

        Returns:
            DataFrame with all execution details
        """
        if not self._trade_history:
            return pd.DataFrame()

        records = []
        for execution in self._trade_history:
            records.append({
                'timestamp': execution.timestamp,
                'symbol': execution.symbol,
                'side': execution.side.value,
                'order_type': execution.order_type.value,
                'quantity': execution.quantity,
                'requested_price': execution.requested_price,
                'execution_price': execution.execution_price,
                'slippage': execution.slippage,
                'fees': execution.fees,
                'total_cost': execution.total_cost
            })

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def get_total_fees(self) -> float:
        """
        Calculate total fees paid across all trades.

        Returns:
            Total fees in quote currency
        """
        return sum(execution.fees for execution in self._trade_history)

    def get_statistics(self) -> Dict[str, float]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution metrics
        """
        if not self._trade_history:
            return {
                'total_executions': 0,
                'total_fees': 0.0,
                'avg_slippage_percent': 0.0,
                'total_volume': 0.0
            }

        total_fees = self.get_total_fees()
        total_volume = sum(
            exec.quantity * exec.execution_price
            for exec in self._trade_history
        )

        # Calculate average slippage as percentage
        avg_slippage_pct = np.mean([
            abs(exec.slippage / exec.requested_price)
            for exec in self._trade_history
        ]) * 100

        return {
            'total_executions': len(self._trade_history),
            'total_fees': total_fees,
            'avg_slippage_percent': avg_slippage_pct,
            'total_volume': total_volume,
            'fees_percent_of_volume': (total_fees / total_volume * 100) if total_volume > 0 else 0.0
        }

    def reset(self) -> None:
        """Clear all trade history."""
        self._trade_history.clear()
        logger.debug("Order executor reset")


if __name__ == "__main__":
    """
    Validation block for order execution simulation.
    Tests order execution with real trading scenarios.
    """
    import sys
    from datetime import timedelta

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating executor.py with real order execution scenarios...\n")

    # Test 1: OrderExecutor initialization
    total_tests += 1
    print("Test 1: OrderExecutor initialization")
    try:
        executor = OrderExecutor(
            fee_percent=0.001,
            slippage_percent=0.0005
        )

        if executor.fee_percent != 0.001:
            all_validation_failures.append(f"Fee percent: Expected 0.001, got {executor.fee_percent}")
        if executor.slippage_percent != 0.0005:
            all_validation_failures.append(f"Slippage percent: Expected 0.0005, got {executor.slippage_percent}")

        print(f"  ✓ Fee: {executor.fee_percent:.4%}")
        print(f"  ✓ Slippage: {executor.slippage_percent:.4%}")
    except Exception as e:
        all_validation_failures.append(f"Initialization test exception: {e}")

    # Test 2: Calculate slippage
    total_tests += 1
    print("\nTest 2: Slippage calculation")
    try:
        price = 50000.0

        buy_slippage = executor.calculate_slippage(price, OrderSide.BUY)
        sell_slippage = executor.calculate_slippage(price, OrderSide.SELL)

        expected_amount = price * 0.0005  # 0.05%
        if abs(buy_slippage - expected_amount) > 0.01:
            all_validation_failures.append(
                f"Buy slippage: Expected {expected_amount:.2f}, got {buy_slippage:.2f}"
            )
        if abs(sell_slippage + expected_amount) > 0.01:  # Sell slippage is negative
            all_validation_failures.append(
                f"Sell slippage: Expected {-expected_amount:.2f}, got {sell_slippage:.2f}"
            )

        print(f"  ✓ Buy slippage: +${buy_slippage:.2f} ({buy_slippage/price:.4%})")
        print(f"  ✓ Sell slippage: ${sell_slippage:.2f} ({sell_slippage/price:.4%})")
    except Exception as e:
        all_validation_failures.append(f"Slippage calculation test exception: {e}")

    # Test 3: Calculate fees
    total_tests += 1
    print("\nTest 3: Fee calculation")
    try:
        quantity = 0.1
        price = 50000.0

        fees = executor.calculate_fees(quantity, price, OrderType.MARKET)
        trade_value = quantity * price
        expected_fees = trade_value * 0.001

        if abs(fees - expected_fees) > 0.01:
            all_validation_failures.append(
                f"Fees: Expected {expected_fees:.2f}, got {fees:.2f}"
            )

        print(f"  ✓ Trade value: ${trade_value:,.2f}")
        print(f"  ✓ Fees: ${fees:.2f} ({fees/trade_value:.4%})")
    except Exception as e:
        all_validation_failures.append(f"Fee calculation test exception: {e}")

    # Test 4: Execute buy order
    total_tests += 1
    print("\nTest 4: Execute buy order")
    try:
        timestamp = datetime(2025, 1, 1, 10, 0, 0)
        symbol = "BTCUSDT"
        quantity = 0.1
        price = 50000.0

        result = executor.execute_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            timestamp=timestamp
        )

        # Verify execution price includes slippage
        expected_exec_price = price + (price * 0.0005)
        if abs(result.execution_price - expected_exec_price) > 0.01:
            all_validation_failures.append(
                f"Execution price: Expected {expected_exec_price:.2f}, got {result.execution_price:.2f}"
            )

        # Verify total cost includes fees
        expected_cost = (quantity * result.execution_price) + result.fees
        if abs(result.total_cost - expected_cost) > 0.01:
            all_validation_failures.append(
                f"Total cost: Expected {expected_cost:.2f}, got {result.total_cost:.2f}"
            )

        print(f"  ✓ Requested price: ${price:,.2f}")
        print(f"  ✓ Execution price: ${result.execution_price:,.2f} (slippage: ${result.slippage:.2f})")
        print(f"  ✓ Fees: ${result.fees:.2f}")
        print(f"  ✓ Total cost: ${result.total_cost:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Buy order test exception: {e}")

    # Test 5: Execute sell order
    total_tests += 1
    print("\nTest 5: Execute sell order")
    try:
        exit_timestamp = timestamp + timedelta(hours=2)
        exit_price = 52000.0

        sell_result = executor.execute_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            price=exit_price,
            timestamp=exit_timestamp
        )

        # For sell orders, slippage reduces execution price
        expected_exec_price = exit_price - (exit_price * 0.0005)
        if abs(sell_result.execution_price - expected_exec_price) > 0.01:
            all_validation_failures.append(
                f"Sell execution price: Expected {expected_exec_price:.2f}, got {sell_result.execution_price:.2f}"
            )

        print(f"  ✓ Requested price: ${exit_price:,.2f}")
        print(f"  ✓ Execution price: ${sell_result.execution_price:,.2f} (slippage: ${sell_result.slippage:.2f})")
        print(f"  ✓ Fees: ${sell_result.fees:.2f}")
    except Exception as e:
        all_validation_failures.append(f"Sell order test exception: {e}")

    # Test 6: Create trade record
    total_tests += 1
    print("\nTest 6: Create complete trade record")
    try:
        trade = executor.create_trade_record(result, sell_result)

        if trade.symbol != symbol:
            all_validation_failures.append(f"Trade symbol: Expected {symbol}, got {trade.symbol}")
        if trade.side != OrderSide.BUY:
            all_validation_failures.append(f"Trade side: Expected BUY, got {trade.side}")
        if trade.quantity != quantity:
            all_validation_failures.append(f"Trade quantity: Expected {quantity}, got {trade.quantity}")

        # Verify PnL calculation
        expected_pnl = (sell_result.execution_price - result.execution_price) * quantity - (result.fees + sell_result.fees)
        if abs(trade.pnl - expected_pnl) > 0.01:
            all_validation_failures.append(
                f"Trade PnL: Expected {expected_pnl:.2f}, got {trade.pnl:.2f}"
            )

        print(f"  ✓ Symbol: {trade.symbol}")
        print(f"  ✓ Entry: ${trade.entry_price:,.2f} -> Exit: ${trade.exit_price:,.2f}")
        print(f"  ✓ PnL: ${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
        print(f"  ✓ Duration: {trade.duration_minutes:.0f} minutes")
    except Exception as e:
        all_validation_failures.append(f"Trade record test exception: {e}")

    # Test 7: Trade history
    total_tests += 1
    print("\nTest 7: Trade history tracking")
    try:
        history = executor.get_trade_history()

        if len(history) != 2:  # Should have 2 executions (buy + sell)
            all_validation_failures.append(f"History length: Expected 2, got {len(history)}")

        # Check first execution is the buy
        if history[0].side != OrderSide.BUY:
            all_validation_failures.append(f"First execution: Expected BUY, got {history[0].side}")

        # Check second execution is the sell
        if history[1].side != OrderSide.SELL:
            all_validation_failures.append(f"Second execution: Expected SELL, got {history[1].side}")

        print(f"  ✓ Total executions: {len(history)}")
        print(f"  ✓ First: {history[0].side.value} @ ${history[0].execution_price:,.2f}")
        print(f"  ✓ Second: {history[1].side.value} @ ${history[1].execution_price:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Trade history test exception: {e}")

    # Test 8: Execution statistics
    total_tests += 1
    print("\nTest 8: Execution statistics")
    try:
        stats = executor.get_statistics()

        required_keys = {'total_executions', 'total_fees', 'avg_slippage_percent', 'total_volume', 'fees_percent_of_volume'}
        if not required_keys.issubset(stats.keys()):
            all_validation_failures.append(
                f"Statistics keys: Missing {required_keys - set(stats.keys())}"
            )

        if stats['total_executions'] != 2:
            all_validation_failures.append(f"Total executions: Expected 2, got {stats['total_executions']}")

        total_fees = result.fees + sell_result.fees
        if abs(stats['total_fees'] - total_fees) > 0.01:
            all_validation_failures.append(
                f"Total fees: Expected {total_fees:.2f}, got {stats['total_fees']:.2f}"
            )

        print(f"  ✓ Total executions: {stats['total_executions']}")
        print(f"  ✓ Total fees: ${stats['total_fees']:,.2f}")
        print(f"  ✓ Average slippage: {stats['avg_slippage_percent']:.4f}%")
        print(f"  ✓ Total volume: ${stats['total_volume']:,.2f}")
        print(f"  ✓ Fees as % of volume: {stats['fees_percent_of_volume']:.4f}%")
    except Exception as e:
        all_validation_failures.append(f"Statistics test exception: {e}")

    # Test 9: Maker/Taker fees
    total_tests += 1
    print("\nTest 9: Maker/Taker fee structure")
    try:
        mt_executor = OrderExecutor(
            fee_percent=0.001,  # Not used when maker/taker enabled
            slippage_percent=0.0005,
            use_maker_taker_fees=True,
            maker_fee=0.0,
            taker_fee=0.002
        )

        # Market order should use taker fee
        market_result = mt_executor.execute_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            timestamp=timestamp,
            order_type=OrderType.MARKET
        )

        expected_taker_fee = 0.1 * market_result.execution_price * 0.002
        if abs(market_result.fees - expected_taker_fee) > 0.01:
            all_validation_failures.append(
                f"Taker fee: Expected {expected_taker_fee:.2f}, got {market_result.fees:.2f}"
            )

        # Limit order should use maker fee (0.0)
        limit_result = mt_executor.execute_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            timestamp=timestamp,
            order_type=OrderType.LIMIT
        )

        if limit_result.fees != 0.0:
            all_validation_failures.append(
                f"Maker fee: Expected 0.0, got {limit_result.fees:.2f}"
            )

        print(f"  ✓ Market order (taker): ${market_result.fees:.2f}")
        print(f"  ✓ Limit order (maker): ${limit_result.fees:.2f}")
    except Exception as e:
        all_validation_failures.append(f"Maker/Taker fees test exception: {e}")

    # Test 10: Trade history DataFrame
    total_tests += 1
    print("\nTest 10: Trade history DataFrame export")
    try:
        df = executor.get_trade_history_df()

        expected_columns = {'timestamp', 'symbol', 'side', 'order_type', 'quantity',
                          'requested_price', 'execution_price', 'slippage', 'fees', 'total_cost'}
        actual_columns = set(df.columns)

        if not expected_columns.issubset(actual_columns):
            all_validation_failures.append(
                f"DataFrame columns: Missing {expected_columns - actual_columns}"
            )

        if len(df) != 2:
            all_validation_failures.append(f"DataFrame rows: Expected 2, got {len(df)}")

        print(f"  ✓ DataFrame shape: {df.shape}")
        print(f"  ✓ Columns: {list(df.columns)}")
    except Exception as e:
        all_validation_failures.append(f"DataFrame export test exception: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Order execution simulation is validated and ready for backtesting")
        sys.exit(0)
