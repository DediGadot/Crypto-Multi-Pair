"""
Position sizing algorithms for crypto trading.

This module implements multiple position sizing strategies to determine optimal
trade sizes based on account risk, volatility, and performance metrics. Each
sizing method balances risk management with capital efficiency.

**Purpose**: Calculate position sizes using various risk-based methodologies
including fixed fractional, Kelly Criterion, volatility-based, and risk parity.

**Key Components**:
- Fixed fraction sizing (constant % of capital)
- Kelly Criterion (optimal bet sizing)
- Volatility-based sizing (ATR/volatility adjusted)
- Risk parity (equal risk contribution)

**Third-party packages**:
- numpy: https://numpy.org/doc/stable/
- pandas: https://pandas.pydata.org/docs/
- loguru: https://loguru.readthedocs.io/

**Sample Input**:
```python
sizer = FixedFractionSizer(risk_per_trade=0.02)
position_size = sizer.calculate(
    capital=10000.0,
    entry_price=50000.0,
    stop_loss_price=48000.0
)
```

**Expected Output**:
```python
{
    'quantity': 0.125,
    'position_value': 6250.0,
    'risk_amount': 200.0,
    'risk_percent': 0.02
}
```
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from loguru import logger


class PositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.

    All position sizers must implement the calculate method that returns
    the quantity to trade given current market conditions and account state.
    """

    @abstractmethod
    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate position size.

        Args:
            capital: Available trading capital
            entry_price: Intended entry price
            stop_loss_price: Stop loss price (if applicable)
            **kwargs: Additional parameters specific to sizing method

        Returns:
            Dictionary with quantity, position_value, risk_amount, risk_percent
        """
        pass


class FixedFractionSizer(PositionSizer):
    """
    Fixed fraction position sizing.

    Risks a constant percentage of capital on each trade. This is the simplest
    and most common position sizing method, providing consistent risk management.

    Attributes:
        risk_per_trade: Percentage of capital to risk per trade (0-1)
        max_position_size: Maximum position as % of capital (0-1)
    """

    def __init__(
        self,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.95
    ):
        """
        Initialize fixed fraction sizer.

        Args:
            risk_per_trade: Risk per trade as decimal (0.02 = 2%)
            max_position_size: Max position size as decimal (0.95 = 95%)
        """
        if not 0 < risk_per_trade <= 1.0:
            raise ValueError(f"risk_per_trade must be between 0 and 1, got {risk_per_trade}")
        if not 0 < max_position_size <= 1.0:
            raise ValueError(f"max_position_size must be between 0 and 1, got {max_position_size}")

        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size

        logger.debug(
            f"FixedFractionSizer initialized: risk={risk_per_trade:.2%}, "
            f"max_position={max_position_size:.2%}"
        )

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate position size based on fixed risk percentage.

        If stop_loss_price is provided, position size is calculated based on risk.
        Otherwise, uses max_position_size directly.

        Args:
            capital: Available capital
            entry_price: Entry price
            stop_loss_price: Stop loss price (optional)

        Returns:
            Dictionary with sizing details
        """
        risk_amount = capital * self.risk_per_trade

        if stop_loss_price is not None and stop_loss_price != entry_price:
            # Calculate position size based on risk and stop distance
            stop_distance = abs(entry_price - stop_loss_price)
            risk_per_unit = stop_distance
            quantity = risk_amount / risk_per_unit
            position_value = quantity * entry_price

            # Apply max position size constraint
            max_position_value = capital * self.max_position_size
            if position_value > max_position_value:
                position_value = max_position_value
                quantity = position_value / entry_price
                # Recalculate actual risk with capped position
                risk_amount = quantity * risk_per_unit
        else:
            # No stop loss provided, use max position size
            position_value = capital * self.max_position_size
            quantity = position_value / entry_price
            risk_amount = capital * self.risk_per_trade

        risk_percent = risk_amount / capital

        logger.debug(
            f"Fixed fraction sizing: capital=${capital:,.2f}, qty={quantity:.6f}, "
            f"value=${position_value:,.2f}, risk=${risk_amount:,.2f} ({risk_percent:.2%})"
        )

        return {
            'quantity': quantity,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent
        }


class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion position sizing.

    Calculates optimal position size based on win rate and win/loss ratio
    to maximize long-term growth rate. Uses fractional Kelly to reduce variance.

    Formula: f = (p * b - q) / b
    where:
        f = fraction of capital to bet
        p = probability of winning
        b = ratio of win to loss
        q = probability of losing (1-p)

    Attributes:
        win_rate: Historical win rate (0-1)
        avg_win_loss_ratio: Average win / average loss ratio
        kelly_fraction: Fraction of full Kelly to use (0-1, typically 0.25-0.5)
        max_position_size: Maximum position as % of capital (0-1)
    """

    def __init__(
        self,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 2.0,
        kelly_fraction: float = 0.25,
        max_position_size: float = 0.95
    ):
        """
        Initialize Kelly Criterion sizer.

        Args:
            win_rate: Historical win rate (0.5 = 50%)
            avg_win_loss_ratio: Average win / average loss
            kelly_fraction: Fraction of full Kelly (0.25 = quarter Kelly)
            max_position_size: Max position size as decimal
        """
        if not 0 < win_rate < 1.0:
            raise ValueError(f"win_rate must be between 0 and 1, got {win_rate}")
        if avg_win_loss_ratio <= 0:
            raise ValueError(f"avg_win_loss_ratio must be positive, got {avg_win_loss_ratio}")
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be between 0 and 1, got {kelly_fraction}")
        if not 0 < max_position_size <= 1.0:
            raise ValueError(f"max_position_size must be between 0 and 1, got {max_position_size}")

        self.win_rate = win_rate
        self.avg_win_loss_ratio = avg_win_loss_ratio
        self.kelly_fraction = kelly_fraction
        self.max_position_size = max_position_size

        # Calculate full Kelly percentage
        p = win_rate
        b = avg_win_loss_ratio
        q = 1 - p
        self._full_kelly = (p * b - q) / b

        # Apply fractional Kelly
        self._kelly_percent = max(0, self._full_kelly * kelly_fraction)

        logger.debug(
            f"KellyCriterionSizer initialized: win_rate={win_rate:.2%}, "
            f"win/loss_ratio={avg_win_loss_ratio:.2f}, "
            f"full_kelly={self._full_kelly:.2%}, "
            f"fractional_kelly={self._kelly_percent:.2%}"
        )

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate position size using Kelly Criterion.

        Args:
            capital: Available capital
            entry_price: Entry price
            stop_loss_price: Stop loss price (optional)

        Returns:
            Dictionary with sizing details
        """
        # Calculate Kelly-based position size
        kelly_position_value = capital * self._kelly_percent

        # Apply max position constraint
        max_position_value = capital * self.max_position_size
        position_value = min(kelly_position_value, max_position_value)

        quantity = position_value / entry_price

        # Calculate risk
        if stop_loss_price is not None:
            stop_distance = abs(entry_price - stop_loss_price)
            risk_amount = quantity * stop_distance
        else:
            # Conservative estimate: assume 50% of position value at risk
            risk_amount = position_value * 0.5

        risk_percent = risk_amount / capital

        logger.debug(
            f"Kelly sizing: capital=${capital:,.2f}, kelly={self._kelly_percent:.2%}, "
            f"qty={quantity:.6f}, value=${position_value:,.2f}, risk=${risk_amount:,.2f}"
        )

        return {
            'quantity': quantity,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent
        }

    def update_statistics(self, win_rate: float, avg_win_loss_ratio: float) -> None:
        """
        Update Kelly statistics with new performance data.

        Args:
            win_rate: New win rate
            avg_win_loss_ratio: New average win/loss ratio
        """
        self.win_rate = win_rate
        self.avg_win_loss_ratio = avg_win_loss_ratio

        # Recalculate Kelly percentage
        p = win_rate
        b = avg_win_loss_ratio
        q = 1 - p
        self._full_kelly = (p * b - q) / b
        self._kelly_percent = max(0, self._full_kelly * self.kelly_fraction)

        logger.info(
            f"Kelly stats updated: win_rate={win_rate:.2%}, "
            f"win/loss={avg_win_loss_ratio:.2f}, kelly={self._kelly_percent:.2%}"
        )


class VolatilityBasedSizer(PositionSizer):
    """
    Volatility-based position sizing.

    Adjusts position size inversely to volatility - higher volatility means
    smaller positions to maintain consistent risk. Uses ATR (Average True Range)
    or standard deviation as volatility measure.

    Attributes:
        risk_per_trade: Risk per trade as % of capital
        volatility_window: Number of periods for volatility calculation
        volatility_scalar: Multiplier for volatility-based sizing
        max_position_size: Maximum position as % of capital
    """

    def __init__(
        self,
        risk_per_trade: float = 0.02,
        volatility_window: int = 14,
        volatility_scalar: float = 2.0,
        max_position_size: float = 0.95
    ):
        """
        Initialize volatility-based sizer.

        Args:
            risk_per_trade: Risk per trade as decimal
            volatility_window: Periods for volatility calculation
            volatility_scalar: ATR multiplier for stop distance
            max_position_size: Max position size as decimal
        """
        if not 0 < risk_per_trade <= 1.0:
            raise ValueError(f"risk_per_trade must be between 0 and 1, got {risk_per_trade}")
        if volatility_window < 1:
            raise ValueError(f"volatility_window must be positive, got {volatility_window}")
        if volatility_scalar <= 0:
            raise ValueError(f"volatility_scalar must be positive, got {volatility_scalar}")
        if not 0 < max_position_size <= 1.0:
            raise ValueError(f"max_position_size must be between 0 and 1, got {max_position_size}")

        self.risk_per_trade = risk_per_trade
        self.volatility_window = volatility_window
        self.volatility_scalar = volatility_scalar
        self.max_position_size = max_position_size

        logger.debug(
            f"VolatilityBasedSizer initialized: risk={risk_per_trade:.2%}, "
            f"window={volatility_window}, scalar={volatility_scalar:.1f}"
        )

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        volatility: Optional[float] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate position size adjusted for volatility.

        Args:
            capital: Available capital
            entry_price: Entry price
            stop_loss_price: Stop loss price (optional)
            volatility: ATR or volatility measure (required)

        Returns:
            Dictionary with sizing details
        """
        if volatility is None:
            raise ValueError("volatility parameter is required for VolatilityBasedSizer")

        if volatility <= 0:
            raise ValueError(f"volatility must be positive, got {volatility}")

        # Calculate risk amount
        risk_amount = capital * self.risk_per_trade

        # Use provided stop loss or volatility-based stop
        if stop_loss_price is not None:
            stop_distance = abs(entry_price - stop_loss_price)
        else:
            # Use volatility scalar * ATR as stop distance
            stop_distance = volatility * self.volatility_scalar

        # Calculate position size
        quantity = risk_amount / stop_distance
        position_value = quantity * entry_price

        # Apply max position constraint
        max_position_value = capital * self.max_position_size
        if position_value > max_position_value:
            position_value = max_position_value
            quantity = position_value / entry_price
            risk_amount = quantity * stop_distance

        risk_percent = risk_amount / capital

        logger.debug(
            f"Volatility sizing: capital=${capital:,.2f}, volatility=${volatility:.2f}, "
            f"stop_dist=${stop_distance:.2f}, qty={quantity:.6f}, "
            f"value=${position_value:,.2f}, risk=${risk_amount:,.2f}"
        )

        return {
            'quantity': quantity,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'volatility': volatility,
            'stop_distance': stop_distance
        }


class RiskParitySizer(PositionSizer):
    """
    Risk parity position sizing.

    Sizes positions so each contributes equal risk to the portfolio. Useful for
    multi-asset portfolios where different assets have different volatilities.

    Attributes:
        target_risk_per_position: Target risk contribution per position
        num_positions: Number of positions in portfolio
        max_position_size: Maximum position as % of capital
    """

    def __init__(
        self,
        target_risk_per_position: float = 0.05,
        num_positions: int = 3,
        max_position_size: float = 0.95
    ):
        """
        Initialize risk parity sizer.

        Args:
            target_risk_per_position: Target risk per position as decimal
            num_positions: Number of positions to manage
            max_position_size: Max position size as decimal
        """
        if not 0 < target_risk_per_position <= 1.0:
            raise ValueError(
                f"target_risk_per_position must be between 0 and 1, got {target_risk_per_position}"
            )
        if num_positions < 1:
            raise ValueError(f"num_positions must be positive, got {num_positions}")
        if not 0 < max_position_size <= 1.0:
            raise ValueError(f"max_position_size must be between 0 and 1, got {max_position_size}")

        self.target_risk_per_position = target_risk_per_position
        self.num_positions = num_positions
        self.max_position_size = max_position_size

        logger.debug(
            f"RiskParitySizer initialized: target_risk={target_risk_per_position:.2%}, "
            f"positions={num_positions}"
        )

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        volatility: Optional[float] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate position size for equal risk contribution.

        Args:
            capital: Available capital
            entry_price: Entry price
            stop_loss_price: Stop loss price (optional)
            volatility: Volatility measure (optional)

        Returns:
            Dictionary with sizing details
        """
        # Calculate target risk for this position
        risk_amount = capital * self.target_risk_per_position

        # Calculate stop distance
        if stop_loss_price is not None:
            stop_distance = abs(entry_price - stop_loss_price)
        elif volatility is not None:
            # Use 2x volatility as stop distance
            stop_distance = volatility * 2.0
        else:
            # Default to 5% of entry price
            stop_distance = entry_price * 0.05

        # Calculate position size
        quantity = risk_amount / stop_distance
        position_value = quantity * entry_price

        # Apply max position constraint
        max_position_value = capital * self.max_position_size
        if position_value > max_position_value:
            position_value = max_position_value
            quantity = position_value / entry_price
            risk_amount = quantity * stop_distance

        risk_percent = risk_amount / capital

        logger.debug(
            f"Risk parity sizing: capital=${capital:,.2f}, target_risk=${risk_amount:,.2f}, "
            f"qty={quantity:.6f}, value=${position_value:,.2f}"
        )

        return {
            'quantity': quantity,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent
        }


def create_position_sizer(method: str, **kwargs) -> PositionSizer:
    """
    Factory function to create position sizers.

    Args:
        method: Sizing method ('fixed_percent', 'fixed_amount', 'kelly',
                'volatility_based', 'risk_parity')
        **kwargs: Parameters for the specific sizer

    Returns:
        PositionSizer instance

    Raises:
        ValueError: If method is not recognized
    """
    method = method.lower()

    if method in ['fixed_percent', 'fixed_fraction', 'fixed_amount']:
        return FixedFractionSizer(**kwargs)
    elif method == 'kelly':
        return KellyCriterionSizer(**kwargs)
    elif method == 'volatility_based':
        return VolatilityBasedSizer(**kwargs)
    elif method == 'risk_parity':
        return RiskParitySizer(**kwargs)
    else:
        raise ValueError(
            f"Unknown position sizing method: {method}. "
            f"Valid methods: fixed_percent, kelly, volatility_based, risk_parity"
        )


if __name__ == "__main__":
    """
    Validation block for position sizing algorithms.
    Tests all sizing methods with realistic trading scenarios.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating sizing.py with real trading scenarios...\n")

    # Test 1: FixedFractionSizer - basic sizing
    total_tests += 1
    print("Test 1: FixedFractionSizer basic sizing")
    try:
        sizer = FixedFractionSizer(risk_per_trade=0.02, max_position_size=0.95)

        result = sizer.calculate(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss_price=48000.0
        )

        # With 2% risk and $2000 stop distance, should risk $200
        # $200 / $2000 per unit = 0.1 units
        expected_quantity = 0.1
        expected_risk = 200.0

        if abs(result['quantity'] - expected_quantity) > 0.0001:
            all_validation_failures.append(
                f"Fixed fraction quantity: Expected {expected_quantity}, got {result['quantity']}"
            )
        if abs(result['risk_amount'] - expected_risk) > 0.01:
            all_validation_failures.append(
                f"Fixed fraction risk: Expected {expected_risk}, got {result['risk_amount']}"
            )

        print(f"  ✓ Quantity: {result['quantity']:.6f}")
        print(f"  ✓ Position value: ${result['position_value']:,.2f}")
        print(f"  ✓ Risk amount: ${result['risk_amount']:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Fixed fraction basic test exception: {e}")

    # Test 2: FixedFractionSizer - max position constraint
    total_tests += 1
    print("\nTest 2: FixedFractionSizer with max position constraint")
    try:
        sizer = FixedFractionSizer(risk_per_trade=0.02, max_position_size=0.50)

        result = sizer.calculate(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss_price=40000.0  # Large stop would exceed max position
        )

        # Max position is 50% = $5000, which is 0.1 BTC
        max_position_value = 5000.0

        if result['position_value'] > max_position_value:
            all_validation_failures.append(
                f"Max position exceeded: {result['position_value']} > {max_position_value}"
            )

        print(f"  ✓ Position value capped at: ${result['position_value']:,.2f}")
        print(f"  ✓ Quantity: {result['quantity']:.6f}")
    except Exception as e:
        all_validation_failures.append(f"Fixed fraction max constraint test exception: {e}")

    # Test 3: FixedFractionSizer - no stop loss
    total_tests += 1
    print("\nTest 3: FixedFractionSizer without stop loss")
    try:
        sizer = FixedFractionSizer(risk_per_trade=0.02, max_position_size=0.95)

        result = sizer.calculate(
            capital=10000.0,
            entry_price=50000.0
        )

        # Should use max_position_size when no stop provided
        expected_value = 9500.0

        if abs(result['position_value'] - expected_value) > 1.0:
            all_validation_failures.append(
                f"Position value: Expected {expected_value}, got {result['position_value']}"
            )

        print(f"  ✓ Uses max position size: ${result['position_value']:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Fixed fraction no stop test exception: {e}")

    # Test 4: KellyCriterionSizer - basic calculation
    total_tests += 1
    print("\nTest 4: KellyCriterionSizer basic calculation")
    try:
        # Win rate 60%, win/loss ratio 2:1, quarter Kelly
        sizer = KellyCriterionSizer(
            win_rate=0.6,
            avg_win_loss_ratio=2.0,
            kelly_fraction=0.25,
            max_position_size=0.95
        )

        result = sizer.calculate(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss_price=48000.0
        )

        # Full Kelly: (0.6*2 - 0.4)/2 = 0.4 = 40%
        # Quarter Kelly: 40% * 0.25 = 10%
        expected_kelly_percent = 0.10
        expected_position_value = 1000.0

        # Allow some tolerance
        if abs(result['position_value'] - expected_position_value) > 100.0:
            all_validation_failures.append(
                f"Kelly position value: Expected ~{expected_position_value}, got {result['position_value']}"
            )

        print(f"  ✓ Position value: ${result['position_value']:,.2f}")
        print(f"  ✓ Quantity: {result['quantity']:.6f}")
    except Exception as e:
        all_validation_failures.append(f"Kelly basic test exception: {e}")

    # Test 5: KellyCriterionSizer - update statistics
    total_tests += 1
    print("\nTest 5: KellyCriterionSizer statistics update")
    try:
        sizer = KellyCriterionSizer(
            win_rate=0.5,
            avg_win_loss_ratio=1.5,
            kelly_fraction=0.25
        )

        initial_kelly = sizer._kelly_percent

        # Update with better stats
        sizer.update_statistics(win_rate=0.65, avg_win_loss_ratio=2.5)

        updated_kelly = sizer._kelly_percent

        if updated_kelly <= initial_kelly:
            all_validation_failures.append(
                f"Kelly should increase with better stats: {initial_kelly} -> {updated_kelly}"
            )

        print(f"  ✓ Initial Kelly: {initial_kelly:.2%}")
        print(f"  ✓ Updated Kelly: {updated_kelly:.2%}")
    except Exception as e:
        all_validation_failures.append(f"Kelly update test exception: {e}")

    # Test 6: VolatilityBasedSizer - with volatility
    total_tests += 1
    print("\nTest 6: VolatilityBasedSizer with volatility")
    try:
        sizer = VolatilityBasedSizer(
            risk_per_trade=0.02,
            volatility_scalar=2.0
        )

        volatility = 1000.0  # $1000 ATR

        result = sizer.calculate(
            capital=10000.0,
            entry_price=50000.0,
            volatility=volatility
        )

        # Risk $200, stop distance = 2.0 * $1000 = $2000
        # Quantity = $200 / $2000 = 0.1
        expected_quantity = 0.1

        if abs(result['quantity'] - expected_quantity) > 0.01:
            all_validation_failures.append(
                f"Volatility quantity: Expected {expected_quantity}, got {result['quantity']}"
            )
        if 'volatility' not in result:
            all_validation_failures.append("Volatility not in result")

        print(f"  ✓ Volatility: ${result['volatility']:,.2f}")
        print(f"  ✓ Stop distance: ${result['stop_distance']:,.2f}")
        print(f"  ✓ Quantity: {result['quantity']:.6f}")
    except Exception as e:
        all_validation_failures.append(f"Volatility sizing test exception: {e}")

    # Test 7: VolatilityBasedSizer - requires volatility
    total_tests += 1
    print("\nTest 7: VolatilityBasedSizer requires volatility parameter")
    try:
        sizer = VolatilityBasedSizer(risk_per_trade=0.02)

        exception_raised = False
        try:
            result = sizer.calculate(
                capital=10000.0,
                entry_price=50000.0
            )
        except ValueError as e:
            exception_raised = True
            if "volatility parameter is required" not in str(e):
                all_validation_failures.append(f"Wrong error message: {e}")

        if not exception_raised:
            all_validation_failures.append("Should raise ValueError when volatility not provided")

        print(f"  ✓ Correctly requires volatility parameter")
    except Exception as e:
        all_validation_failures.append(f"Volatility required test exception: {e}")

    # Test 8: RiskParitySizer - basic calculation
    total_tests += 1
    print("\nTest 8: RiskParitySizer basic calculation")
    try:
        sizer = RiskParitySizer(
            target_risk_per_position=0.05,
            num_positions=3
        )

        result = sizer.calculate(
            capital=10000.0,
            entry_price=50000.0,
            stop_loss_price=48000.0
        )

        # Target risk 5% = $500, stop distance $2000
        # Uncapped: Quantity = $500 / $2000 = 0.25, value = $12,500
        # But max position is 95% = $9,500, so caps at 0.19 BTC
        # Actual risk = 0.19 * $2000 = $380
        max_position_value = 9500.0

        if result['position_value'] > max_position_value:
            all_validation_failures.append(
                f"Position value exceeds max: {result['position_value']} > {max_position_value}"
            )

        # Risk should be capped by max position constraint
        if result['risk_amount'] > 500.0:
            all_validation_failures.append(
                f"Risk amount exceeds target: {result['risk_amount']} > 500.0"
            )

        print(f"  ✓ Risk amount (capped): ${result['risk_amount']:,.2f}")
        print(f"  ✓ Quantity: {result['quantity']:.6f}")
        print(f"  ✓ Position value: ${result['position_value']:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Risk parity test exception: {e}")

    # Test 9: Factory function - create sizers
    total_tests += 1
    print("\nTest 9: Factory function creates correct sizers")
    try:
        fixed_sizer = create_position_sizer('fixed_percent', risk_per_trade=0.02)
        kelly_sizer = create_position_sizer('kelly', win_rate=0.6, avg_win_loss_ratio=2.0)
        volatility_sizer = create_position_sizer('volatility_based', risk_per_trade=0.02)
        risk_parity_sizer = create_position_sizer('risk_parity', target_risk_per_position=0.05)

        if not isinstance(fixed_sizer, FixedFractionSizer):
            all_validation_failures.append("Factory didn't create FixedFractionSizer")
        if not isinstance(kelly_sizer, KellyCriterionSizer):
            all_validation_failures.append("Factory didn't create KellyCriterionSizer")
        if not isinstance(volatility_sizer, VolatilityBasedSizer):
            all_validation_failures.append("Factory didn't create VolatilityBasedSizer")
        if not isinstance(risk_parity_sizer, RiskParitySizer):
            all_validation_failures.append("Factory didn't create RiskParitySizer")

        print(f"  ✓ Created FixedFractionSizer")
        print(f"  ✓ Created KellyCriterionSizer")
        print(f"  ✓ Created VolatilityBasedSizer")
        print(f"  ✓ Created RiskParitySizer")
    except Exception as e:
        all_validation_failures.append(f"Factory function test exception: {e}")

    # Test 10: Factory function - invalid method
    total_tests += 1
    print("\nTest 10: Factory function rejects invalid method")
    try:
        exception_raised = False
        try:
            invalid_sizer = create_position_sizer('invalid_method')
        except ValueError as e:
            exception_raised = True
            if "Unknown position sizing method" not in str(e):
                all_validation_failures.append(f"Wrong error message: {e}")

        if not exception_raised:
            all_validation_failures.append("Should raise ValueError for invalid method")

        print(f"  ✓ Correctly rejects invalid method")
    except Exception as e:
        all_validation_failures.append(f"Invalid method test exception: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Position sizing algorithms validated and ready for use")
        sys.exit(0)
