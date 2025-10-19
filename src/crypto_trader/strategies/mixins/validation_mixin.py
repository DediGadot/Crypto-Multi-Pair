"""
Validation Mixin

Provides common validation functionality for trading strategies.
Extracts duplicated validation logic found across many strategies.

**Purpose**: Eliminate code duplication in strategy validation by providing
reusable methods for common checks (required columns, data quality, parameters).

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/

**Sample Usage**:
```python
from crypto_trader.strategies.base import BaseStrategy
from crypto_trader.strategies.mixins import ValidationMixin

class MyStrategy(ValidationMixin, BaseStrategy):
    def initialize(self, config: dict):
        # Validate parameters
        self.fast_period = self.validate_parameter(
            config.get('fast_period', 10),
            param_name='fast_period',
            min_value=1,
            max_value=100
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Validate input data
        self.validate_required_columns(data, ['open', 'high', 'low', 'close'])
        self.validate_data_quality(data, min_rows=50)

        # Generate signals...
```

**Expected Output**:
- Clear ValueError messages on validation failures
- Consistent validation across all strategies
- Reduced duplication of validation logic
"""

from typing import Any, Optional
import pandas as pd
import numpy as np


class ValidationMixin:
    """
    Mixin providing data and parameter validation functionality.

    Use this mixin when your strategy needs to validate inputs,
    parameters, or data quality before processing.
    """

    def validate_required_columns(
        self,
        data: pd.DataFrame,
        required_columns: list[str],
        data_name: str = "data"
    ) -> None:
        """
        Validate that required columns exist in DataFrame.

        Args:
            data: DataFrame to validate
            required_columns: List of column names that must exist
            data_name: Name of data for error message

        Raises:
            ValueError: If any required columns are missing

        Example:
            >>> self.validate_required_columns(data, ['close', 'volume'])
        """
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(
                f"{data_name} missing required columns: {missing}. "
                f"Available columns: {list(data.columns)}"
            )

    def validate_data_quality(
        self,
        data: pd.DataFrame,
        min_rows: Optional[int] = None,
        max_nan_pct: float = 5.0,
        data_name: str = "data"
    ) -> None:
        """
        Validate data quality (length, missing values).

        Args:
            data: DataFrame to validate
            min_rows: Minimum required rows (None = no check)
            max_nan_pct: Maximum allowed NaN percentage (default 5%)
            data_name: Name of data for error message

        Raises:
            ValueError: If data quality is insufficient

        Example:
            >>> self.validate_data_quality(data, min_rows=100, max_nan_pct=2.0)
        """
        # Check minimum rows
        if min_rows is not None and len(data) < min_rows:
            raise ValueError(
                f"{data_name} has only {len(data)} rows, "
                f"minimum required: {min_rows}"
            )

        # Check for excessive NaN values
        if not data.empty:
            nan_pct = (data.isna().sum().sum() / (len(data) * len(data.columns))) * 100
            if nan_pct > max_nan_pct:
                raise ValueError(
                    f"{data_name} has {nan_pct:.1f}% NaN values, "
                    f"maximum allowed: {max_nan_pct}%"
                )

    def validate_parameter(
        self,
        value: Any,
        param_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allowed_values: Optional[set] = None,
        param_type: Optional[type] = None
    ) -> Any:
        """
        Validate a strategy parameter value.

        Args:
            value: Parameter value to validate
            param_name: Name of parameter for error messages
            min_value: Minimum allowed value (for numeric params)
            max_value: Maximum allowed value (for numeric params)
            allowed_values: Set of allowed values (for enum-like params)
            param_type: Expected type of parameter

        Returns:
            The validated value (unchanged)

        Raises:
            ValueError: If validation fails

        Example:
            >>> period = self.validate_parameter(
            ...     config.get('period', 14),
            ...     param_name='period',
            ...     min_value=1,
            ...     max_value=200
            ... )
        """
        # Check type
        if param_type is not None and not isinstance(value, param_type):
            raise ValueError(
                f"{param_name} must be type {param_type.__name__}, "
                f"got {type(value).__name__}"
            )

        # Check allowed values
        if allowed_values is not None and value not in allowed_values:
            raise ValueError(
                f"{param_name} must be one of {allowed_values}, got {value}"
            )

        # Check numeric range
        if min_value is not None and value < min_value:
            raise ValueError(
                f"{param_name} must be >= {min_value}, got {value}"
            )
        if max_value is not None and value > max_value:
            raise ValueError(
                f"{param_name} must be <= {max_value}, got {value}"
            )

        return value

    def validate_datetime_index(
        self,
        data: pd.DataFrame,
        data_name: str = "data"
    ) -> None:
        """
        Validate that DataFrame has proper DatetimeIndex.

        Args:
            data: DataFrame to validate
            data_name: Name of data for error message

        Raises:
            ValueError: If index is not DatetimeIndex

        Example:
            >>> self.validate_datetime_index(data)
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                f"{data_name} must have DatetimeIndex, "
                f"got {type(data.index).__name__}"
            )

    def validate_positive_values(
        self,
        data: pd.DataFrame,
        columns: list[str],
        allow_zero: bool = False,
        data_name: str = "data"
    ) -> None:
        """
        Validate that specified columns contain only positive values.

        Useful for price, volume, and other metrics that must be positive.

        Args:
            data: DataFrame to validate
            columns: Columns that must be positive
            allow_zero: Whether zero values are acceptable
            data_name: Name of data for error message

        Raises:
            ValueError: If non-positive values found

        Example:
            >>> self.validate_positive_values(data, ['close', 'volume'])
        """
        for col in columns:
            if col not in data.columns:
                continue

            min_value = data[col].min()
            if allow_zero:
                if min_value < 0:
                    raise ValueError(
                        f"{data_name}['{col}'] contains negative values (min: {min_value})"
                    )
            else:
                if min_value <= 0:
                    raise ValueError(
                        f"{data_name}['{col}'] contains non-positive values (min: {min_value})"
                    )

    def validate_no_duplicates(
        self,
        data: pd.DataFrame,
        data_name: str = "data"
    ) -> None:
        """
        Validate that DataFrame has no duplicate index values.

        Args:
            data: DataFrame to validate
            data_name: Name of data for error message

        Raises:
            ValueError: If duplicate index values found

        Example:
            >>> self.validate_no_duplicates(data)
        """
        if data.index.duplicated().any():
            dup_count = data.index.duplicated().sum()
            raise ValueError(
                f"{data_name} has {dup_count} duplicate index values"
            )

    def validate_sorted_index(
        self,
        data: pd.DataFrame,
        data_name: str = "data"
    ) -> None:
        """
        Validate that DataFrame index is sorted.

        Args:
            data: DataFrame to validate
            data_name: Name of data for error message

        Raises:
            ValueError: If index is not sorted

        Example:
            >>> self.validate_sorted_index(data)
        """
        if not data.index.is_monotonic_increasing:
            raise ValueError(f"{data_name} index is not sorted")


if __name__ == "__main__":
    """
    Validation block for ValidationMixin.
    Tests all validation methods with real data.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating ValidationMixin...\n")

    # Create test instance
    class TestStrategy(ValidationMixin):
        """Test class with mixin."""
        pass

    strategy = TestStrategy()

    # Test 1: Validate required columns (success)
    total_tests += 1
    print("Test 1: Validate required columns (success)")
    try:
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1050]
        })

        strategy.validate_required_columns(test_data, ['open', 'close', 'volume'])
        print("  ✓ Required columns validation passed")
    except Exception as e:
        all_validation_failures.append(f"Test 1 failed: {e}")

    # Test 2: Validate required columns (failure)
    total_tests += 1
    print("\nTest 2: Validate required columns (failure)")
    try:
        error_raised = False
        try:
            strategy.validate_required_columns(test_data, ['missing_col'])
        except ValueError as e:
            if 'missing required columns' in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append("Expected ValueError for missing column")
        else:
            print("  ✓ Missing column detected correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 2 failed: {e}")

    # Test 3: Validate data quality (min rows)
    total_tests += 1
    print("\nTest 3: Validate data quality (min rows)")
    try:
        small_data = pd.DataFrame({'value': [1, 2]})

        error_raised = False
        try:
            strategy.validate_data_quality(small_data, min_rows=10)
        except ValueError as e:
            if 'minimum required' in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append("Expected ValueError for insufficient rows")
        else:
            print("  ✓ Minimum rows validation working")
    except Exception as e:
        all_validation_failures.append(f"Test 3 failed: {e}")

    # Test 4: Validate parameter (numeric range)
    total_tests += 1
    print("\nTest 4: Validate parameter (numeric range)")
    try:
        # Valid value
        result = strategy.validate_parameter(
            50,
            param_name='period',
            min_value=1,
            max_value=100
        )

        if result != 50:
            all_validation_failures.append(f"Expected 50, got {result}")

        # Invalid value (too low)
        error_raised = False
        try:
            strategy.validate_parameter(
                0,
                param_name='period',
                min_value=1,
                max_value=100
            )
        except ValueError as e:
            if 'must be >=' in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append("Expected ValueError for value too low")
        else:
            print("  ✓ Parameter range validation working")
    except Exception as e:
        all_validation_failures.append(f"Test 4 failed: {e}")

    # Test 5: Validate datetime index
    total_tests += 1
    print("\nTest 5: Validate datetime index")
    try:
        # Valid datetime index
        valid_data = pd.DataFrame(
            {'value': [1, 2, 3]},
            index=pd.date_range('2025-01-01', periods=3, freq='1h')
        )
        strategy.validate_datetime_index(valid_data)

        # Invalid integer index
        invalid_data = pd.DataFrame({'value': [1, 2, 3]})
        error_raised = False
        try:
            strategy.validate_datetime_index(invalid_data)
        except ValueError as e:
            if 'DatetimeIndex' in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append("Expected ValueError for non-datetime index")
        else:
            print("  ✓ Datetime index validation working")
    except Exception as e:
        all_validation_failures.append(f"Test 5 failed: {e}")

    # Test 6: Validate positive values
    total_tests += 1
    print("\nTest 6: Validate positive values")
    try:
        # Valid positive values
        positive_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1050]
        })
        strategy.validate_positive_values(positive_data, ['close', 'volume'])

        # Invalid negative values
        negative_data = pd.DataFrame({
            'close': [100, -10, 102],
            'volume': [1000, 1100, 1050]
        })
        error_raised = False
        try:
            strategy.validate_positive_values(negative_data, ['close'])
        except ValueError as e:
            if 'non-positive values' in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append("Expected ValueError for negative values")
        else:
            print("  ✓ Positive value validation working")
    except Exception as e:
        all_validation_failures.append(f"Test 6 failed: {e}")

    # Test 7: Validate no duplicates
    total_tests += 1
    print("\nTest 7: Validate no duplicates")
    try:
        # Valid unique index
        unique_data = pd.DataFrame(
            {'value': [1, 2, 3]},
            index=pd.date_range('2025-01-01', periods=3, freq='1h')
        )
        strategy.validate_no_duplicates(unique_data)

        # Invalid duplicate index
        dup_index = pd.DatetimeIndex(['2025-01-01', '2025-01-01', '2025-01-02'])
        dup_data = pd.DataFrame({'value': [1, 2, 3]}, index=dup_index)

        error_raised = False
        try:
            strategy.validate_no_duplicates(dup_data)
        except ValueError as e:
            if 'duplicate index' in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append("Expected ValueError for duplicate index")
        else:
            print("  ✓ Duplicate detection working")
    except Exception as e:
        all_validation_failures.append(f"Test 7 failed: {e}")

    # Test 8: Validate sorted index
    total_tests += 1
    print("\nTest 8: Validate sorted index")
    try:
        # Valid sorted index
        sorted_data = pd.DataFrame(
            {'value': [1, 2, 3]},
            index=pd.date_range('2025-01-01', periods=3, freq='1h')
        )
        strategy.validate_sorted_index(sorted_data)

        # Invalid unsorted index
        unsorted_index = pd.DatetimeIndex(['2025-01-03', '2025-01-01', '2025-01-02'])
        unsorted_data = pd.DataFrame({'value': [1, 2, 3]}, index=unsorted_index)

        error_raised = False
        try:
            strategy.validate_sorted_index(unsorted_data)
        except ValueError as e:
            if 'not sorted' in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append("Expected ValueError for unsorted index")
        else:
            print("  ✓ Sorted index validation working")
    except Exception as e:
        all_validation_failures.append(f"Test 8 failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("ValidationMixin is validated and ready for use")
        sys.exit(0)
