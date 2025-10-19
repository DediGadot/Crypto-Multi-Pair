"""
Hold Signal Mixin

Provides common hold signal generation behavior for strategies.
Extracts duplicated logic found in 10+ strategy implementations.

**Purpose**: Eliminate the duplicated pattern of filling missing signals with
'HOLD', which appears in nearly every strategy with slight variations.

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/

**Sample Usage**:
```python
from crypto_trader.strategies.base import BaseStrategy
from crypto_trader.strategies.mixins import HoldSignalMixin

class MyStrategy(HoldSignalMixin, BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = self._create_signal_dataframe(data)

        # Generate BUY/SELL signals
        buy_mask = (data['close'] > data['sma_fast'])
        sell_mask = (data['close'] < data['sma_slow'])

        signals.loc[buy_mask, 'signal'] = 'BUY'
        signals.loc[sell_mask, 'signal'] = 'SELL'

        # Automatically fill remaining rows with 'HOLD'
        return self.apply_hold_signals(data, signals)
```

**Expected Output**:
- DataFrame with 'signal' column containing BUY/SELL/HOLD values
- No NaN values in signal column
- Consistent signal format across strategies
"""

from typing import Optional
import pandas as pd
import numpy as np


class HoldSignalMixin:
    """
    Mixin providing hold signal generation functionality.

    Use this mixin when your strategy generates BUY/SELL signals
    and needs to fill the rest with HOLD.
    """

    def _create_signal_dataframe(
        self,
        data: pd.DataFrame,
        default_signal: str = "HOLD"
    ) -> pd.DataFrame:
        """
        Create a signal DataFrame initialized with default signals.

        Args:
            data: Market data DataFrame with DatetimeIndex
            default_signal: Default signal value ('HOLD', 'BUY', 'SELL')

        Returns:
            DataFrame with same index as data, signal column initialized

        Example:
            >>> signals = self._create_signal_dataframe(data)
            >>> signals.loc[buy_condition, 'signal'] = 'BUY'
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = default_signal
        return signals

    def apply_hold_signals(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Fill missing/NaN signals with 'HOLD'.

        Args:
            data: Original market data (for validation)
            signals: DataFrame with 'signal' column
            inplace: Whether to modify signals in place

        Returns:
            DataFrame with all signals filled (no NaN values)

        Raises:
            ValueError: If signal column is missing

        Example:
            >>> signals = self.apply_hold_signals(data, signals)
            >>> assert not signals['signal'].isna().any()
        """
        if 'signal' not in signals.columns:
            raise ValueError("signals DataFrame must have 'signal' column")

        if not inplace:
            signals = signals.copy()

        # Fill NaN values with HOLD
        signals['signal'] = signals['signal'].fillna('HOLD')

        # Replace empty strings with HOLD
        signals.loc[signals['signal'] == '', 'signal'] = 'HOLD'

        # Validate signal values
        valid_signals = {'BUY', 'SELL', 'HOLD'}
        invalid_mask = ~signals['signal'].isin(valid_signals)
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            invalid_values = signals.loc[invalid_mask, 'signal'].unique()
            raise ValueError(
                f"Found {invalid_count} invalid signals: {invalid_values}. "
                f"Valid signals are: {valid_signals}"
            )

        return signals

    def add_signal_metadata(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame,
        strategy_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add metadata columns to signal DataFrame for analysis.

        Args:
            signals: DataFrame with signal column
            data: Market data with price information
            strategy_name: Name of strategy generating signals

        Returns:
            DataFrame with additional metadata columns

        Example:
            >>> signals = self.add_signal_metadata(signals, data, "SMA_Crossover")
            >>> print(signals.columns)
            ['signal', 'price', 'strategy']
        """
        signals = signals.copy()

        # Add price at signal time
        if 'close' in data.columns:
            signals['price'] = data['close']

        # Add strategy name
        if strategy_name:
            signals['strategy'] = strategy_name

        return signals

    def count_signals(self, signals: pd.DataFrame) -> dict[str, int]:
        """
        Count each signal type for reporting.

        Args:
            signals: DataFrame with signal column

        Returns:
            Dictionary mapping signal type to count

        Example:
            >>> counts = self.count_signals(signals)
            >>> print(counts)
            {'BUY': 15, 'SELL': 12, 'HOLD': 673}
        """
        if 'signal' not in signals.columns:
            return {}

        return signals['signal'].value_counts().to_dict()

    def signal_transition_matrix(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Create transition matrix showing signal changes.

        Useful for analyzing strategy behavior and patterns.

        Args:
            signals: DataFrame with signal column

        Returns:
            DataFrame showing transitions between signal states

        Example:
            >>> transitions = self.signal_transition_matrix(signals)
            >>> print(transitions)
                    BUY  SELL  HOLD
            BUY       0     5    10
            SELL      8     0     7
            HOLD     10     8   652
        """
        if 'signal' not in signals.columns or len(signals) < 2:
            return pd.DataFrame()

        # Create previous signal column
        prev_signals = signals['signal'].shift(1)
        curr_signals = signals['signal']

        # Build transition counts
        transitions = pd.crosstab(
            prev_signals,
            curr_signals,
            rownames=['From'],
            colnames=['To']
        )

        return transitions


if __name__ == "__main__":
    """
    Validation block for HoldSignalMixin.
    Tests all mixin methods with real data.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating HoldSignalMixin...\n")

    # Create test instance
    class TestStrategy(HoldSignalMixin):
        """Test class with mixin."""
        pass

    strategy = TestStrategy()

    # Test 1: Create signal dataframe
    total_tests += 1
    print("Test 1: Create signal DataFrame")
    try:
        test_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1050, 1200, 1150]
        }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))

        signals = strategy._create_signal_dataframe(test_data)

        if not isinstance(signals, pd.DataFrame):
            all_validation_failures.append(
                f"Expected DataFrame, got {type(signals)}"
            )
        elif 'signal' not in signals.columns:
            all_validation_failures.append("signal column missing")
        elif len(signals) != len(test_data):
            all_validation_failures.append(
                f"Expected {len(test_data)} rows, got {len(signals)}"
            )
        elif not (signals['signal'] == 'HOLD').all():
            all_validation_failures.append("Not all signals initialized to HOLD")
        else:
            print("  ✓ Signal DataFrame created correctly")
            print(f"  ✓ Shape: {signals.shape}")
            print(f"  ✓ Default signal: {signals['signal'].iloc[0]}")
    except Exception as e:
        all_validation_failures.append(f"Test 1 failed: {e}")

    # Test 2: Apply hold signals
    total_tests += 1
    print("\nTest 2: Apply hold signals")
    try:
        signals = pd.DataFrame({
            'signal': ['BUY', np.nan, 'SELL', '', 'HOLD']
        }, index=test_data.index)

        result = strategy.apply_hold_signals(test_data, signals)

        if result['signal'].isna().any():
            all_validation_failures.append("NaN values still present after apply")
        elif (result['signal'] == '').any():
            all_validation_failures.append("Empty strings still present")
        elif result['signal'].iloc[1] != 'HOLD':
            all_validation_failures.append(
                f"Expected HOLD at index 1, got {result['signal'].iloc[1]}"
            )
        else:
            print("  ✓ Hold signals applied correctly")
            print(f"  ✓ Signals: {result['signal'].tolist()}")
    except Exception as e:
        all_validation_failures.append(f"Test 2 failed: {e}")

    # Test 3: Invalid signal detection
    total_tests += 1
    print("\nTest 3: Invalid signal detection")
    try:
        signals = pd.DataFrame({
            'signal': ['BUY', 'INVALID', 'SELL']
        }, index=test_data.index[:3])

        error_raised = False
        try:
            strategy.apply_hold_signals(test_data.iloc[:3], signals)
        except ValueError as e:
            if 'invalid signals' in str(e).lower():
                error_raised = True

        if not error_raised:
            all_validation_failures.append("Expected ValueError for invalid signal")
        else:
            print("  ✓ Invalid signals detected correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 3 failed: {e}")

    # Test 4: Add signal metadata
    total_tests += 1
    print("\nTest 4: Add signal metadata")
    try:
        signals = strategy._create_signal_dataframe(test_data)
        signals.loc[test_data.index[0], 'signal'] = 'BUY'

        result = strategy.add_signal_metadata(
            signals,
            test_data,
            strategy_name="TestStrategy"
        )

        if 'price' not in result.columns:
            all_validation_failures.append("price column missing")
        elif 'strategy' not in result.columns:
            all_validation_failures.append("strategy column missing")
        elif result['price'].iloc[0] != 100:
            all_validation_failures.append(
                f"Expected price 100, got {result['price'].iloc[0]}"
            )
        else:
            print("  ✓ Metadata added correctly")
            print(f"  ✓ Columns: {list(result.columns)}")
    except Exception as e:
        all_validation_failures.append(f"Test 4 failed: {e}")

    # Test 5: Count signals
    total_tests += 1
    print("\nTest 5: Count signals")
    try:
        signals = pd.DataFrame({
            'signal': ['BUY', 'BUY', 'SELL', 'HOLD', 'HOLD', 'HOLD']
        })

        counts = strategy.count_signals(signals)

        if counts.get('BUY') != 2:
            all_validation_failures.append(f"Expected 2 BUY, got {counts.get('BUY')}")
        elif counts.get('SELL') != 1:
            all_validation_failures.append(f"Expected 1 SELL, got {counts.get('SELL')}")
        elif counts.get('HOLD') != 3:
            all_validation_failures.append(f"Expected 3 HOLD, got {counts.get('HOLD')}")
        else:
            print("  ✓ Signal counts correct")
            print(f"  ✓ Counts: {counts}")
    except Exception as e:
        all_validation_failures.append(f"Test 5 failed: {e}")

    # Test 6: Transition matrix
    total_tests += 1
    print("\nTest 6: Signal transition matrix")
    try:
        signals = pd.DataFrame({
            'signal': ['HOLD', 'BUY', 'HOLD', 'SELL', 'HOLD']
        })

        matrix = strategy.signal_transition_matrix(signals)

        if not isinstance(matrix, pd.DataFrame):
            all_validation_failures.append(
                f"Expected DataFrame, got {type(matrix)}"
            )
        elif matrix.empty:
            all_validation_failures.append("Transition matrix is empty")
        else:
            print("  ✓ Transition matrix created")
            print(f"  ✓ Shape: {matrix.shape}")
    except Exception as e:
        all_validation_failures.append(f"Test 6 failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("HoldSignalMixin is validated and ready for use")
        sys.exit(0)
