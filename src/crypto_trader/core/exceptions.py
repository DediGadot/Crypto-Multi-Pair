"""
Custom exceptions for the crypto trading system.

This module defines a hierarchy of custom exceptions used throughout the
trading system for clear error handling and debugging. All exceptions
inherit from a base CryptoTraderError class.

Documentation:
- Python Exceptions: https://docs.python.org/3/library/exceptions.html
- Exception Best Practices: https://docs.python.org/3/tutorial/errors.html

Sample Input:
    raise DataFetchError("Failed to fetch OHLCV data", symbol="BTCUSDT")
    raise ValidationError("Invalid timeframe", details={"timeframe": "2h"})

Expected Output:
    Properly formatted exception messages with context for debugging
"""

from typing import Any, Optional


class CryptoTraderError(Exception):
    """
    Base exception class for all crypto trading system errors.

    All custom exceptions in the system should inherit from this class
    to allow for centralized error handling and logging.

    Attributes:
        message: Human-readable error description
        details: Additional context as key-value pairs
        original_error: Original exception if this wraps another error
    """

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with details and original error if present."""
        parts = [self.message]

        if self.details:
            # Format details as key=value pairs
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")

        if self.original_error:
            parts.append(f"Caused by: {type(self.original_error).__name__}: {str(self.original_error)}")

        return " | ".join(parts)


class DataFetchError(CryptoTraderError):
    """
    Exception raised when data fetching fails.

    Common scenarios:
    - API rate limits exceeded
    - Network connectivity issues
    - Invalid API credentials
    - Symbol or exchange not found
    - Data not available for requested timeframe

    Example:
        raise DataFetchError(
            "Failed to fetch OHLCV data from Binance",
            details={"symbol": "BTCUSDT", "timeframe": "1h"},
            original_error=api_exception
        )
    """
    pass


class StrategyError(CryptoTraderError):
    """
    Exception raised when strategy execution or initialization fails.

    Common scenarios:
    - Invalid strategy parameters
    - Missing required indicators
    - Strategy logic errors
    - Signal generation failures
    - Indicator calculation errors

    Example:
        raise StrategyError(
            "RSI calculation failed due to insufficient data",
            details={"required_periods": 14, "available_periods": 10}
        )
    """
    pass


class BacktestError(CryptoTraderError):
    """
    Exception raised during backtesting operations.

    Common scenarios:
    - Insufficient historical data
    - Invalid backtest configuration
    - Position sizing errors
    - Portfolio calculation errors
    - Trade execution simulation failures

    Example:
        raise BacktestError(
            "Insufficient data for backtest period",
            details={
                "requested_start": "2024-01-01",
                "data_available_from": "2024-06-01"
            }
        )
    """
    pass


class ConfigurationError(CryptoTraderError):
    """
    Exception raised for configuration-related errors.

    Common scenarios:
    - Missing required configuration keys
    - Invalid configuration values
    - YAML parsing errors
    - Environment variable not set
    - Type validation failures

    Example:
        raise ConfigurationError(
            "Missing required API key in configuration",
            details={"missing_key": "binance_api_key", "config_file": "config.yaml"}
        )
    """
    pass


class ValidationError(CryptoTraderError):
    """
    Exception raised when data validation fails.

    Common scenarios:
    - Invalid input parameters
    - Data type mismatches
    - Range validation failures
    - Schema validation errors
    - Pydantic model validation errors

    Example:
        raise ValidationError(
            "Order quantity exceeds maximum allowed",
            details={"quantity": 100, "max_quantity": 50, "symbol": "BTCUSDT"}
        )
    """
    pass


class OrderExecutionError(CryptoTraderError):
    """
    Exception raised when order execution fails.

    Common scenarios:
    - Insufficient balance
    - Order rejection by exchange
    - Invalid order parameters
    - Market closed or halted
    - Position limit reached

    Example:
        raise OrderExecutionError(
            "Insufficient balance for order",
            details={
                "required": 1000.0,
                "available": 500.0,
                "symbol": "ETHUSDT"
            }
        )
    """
    pass


class RiskManagementError(CryptoTraderError):
    """
    Exception raised when risk management rules are violated.

    Common scenarios:
    - Position size exceeds risk limits
    - Maximum drawdown threshold reached
    - Daily loss limit exceeded
    - Concentration risk violation
    - Leverage limits exceeded

    Example:
        raise RiskManagementError(
            "Daily loss limit exceeded",
            details={
                "daily_loss": 0.05,
                "max_daily_loss": 0.03,
                "current_capital": 9500.0
            }
        )
    """
    pass


if __name__ == "__main__":
    """
    Validation function to test all exception classes.
    Verifies proper initialization, message formatting, and inheritance.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating exceptions.py with real data...\n")

    # Test 1: Base CryptoTraderError
    total_tests += 1
    print("Test 1: Base CryptoTraderError")
    try:
        base_error = CryptoTraderError("Base error occurred")
        if "Base error occurred" not in str(base_error):
            all_validation_failures.append(f"Base error message not found in: {str(base_error)}")

        error_with_details = CryptoTraderError(
            "Error with context",
            details={"key1": "value1", "key2": 42}
        )
        error_str = str(error_with_details)
        if "Details:" not in error_str:
            all_validation_failures.append(f"Details section not in error: {error_str}")
        if "key1=value1" not in error_str:
            all_validation_failures.append(f"Detail key1 not in error: {error_str}")

        print(f"  ✓ Basic error: {base_error}")
        print(f"  ✓ Error with details: {error_with_details}")
    except Exception as e:
        all_validation_failures.append(f"Base error test exception: {e}")

    # Test 2: Error with original exception
    total_tests += 1
    print("\nTest 2: Error with original exception")
    try:
        original = ValueError("Original error message")
        wrapped_error = CryptoTraderError(
            "Wrapped error",
            details={"context": "test"},
            original_error=original
        )
        error_str = str(wrapped_error)
        if "Caused by:" not in error_str:
            all_validation_failures.append(f"'Caused by' not in wrapped error: {error_str}")
        if "ValueError" not in error_str:
            all_validation_failures.append(f"ValueError type not in error: {error_str}")
        if "Original error message" not in error_str:
            all_validation_failures.append(f"Original message not in error: {error_str}")

        print(f"  ✓ Wrapped error: {wrapped_error}")
    except Exception as e:
        all_validation_failures.append(f"Wrapped error test exception: {e}")

    # Test 3: DataFetchError
    total_tests += 1
    print("\nTest 3: DataFetchError")
    try:
        data_error = DataFetchError(
            "Failed to fetch OHLCV data",
            details={"symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1h"}
        )
        if not isinstance(data_error, CryptoTraderError):
            all_validation_failures.append("DataFetchError should inherit from CryptoTraderError")
        if "BTCUSDT" not in str(data_error):
            all_validation_failures.append(f"Symbol not in error: {str(data_error)}")

        print(f"  ✓ DataFetchError: {data_error}")
    except Exception as e:
        all_validation_failures.append(f"DataFetchError test exception: {e}")

    # Test 4: StrategyError
    total_tests += 1
    print("\nTest 4: StrategyError")
    try:
        strategy_error = StrategyError(
            "RSI calculation failed",
            details={"required_periods": 14, "available_periods": 10}
        )
        if not isinstance(strategy_error, CryptoTraderError):
            all_validation_failures.append("StrategyError should inherit from CryptoTraderError")
        if "required_periods=14" not in str(strategy_error):
            all_validation_failures.append(f"Periods not in error: {str(strategy_error)}")

        print(f"  ✓ StrategyError: {strategy_error}")
    except Exception as e:
        all_validation_failures.append(f"StrategyError test exception: {e}")

    # Test 5: BacktestError
    total_tests += 1
    print("\nTest 5: BacktestError")
    try:
        backtest_error = BacktestError(
            "Insufficient data for backtest",
            details={"requested_start": "2024-01-01", "available_from": "2024-06-01"}
        )
        if not isinstance(backtest_error, CryptoTraderError):
            all_validation_failures.append("BacktestError should inherit from CryptoTraderError")
        if "2024-01-01" not in str(backtest_error):
            all_validation_failures.append(f"Date not in error: {str(backtest_error)}")

        print(f"  ✓ BacktestError: {backtest_error}")
    except Exception as e:
        all_validation_failures.append(f"BacktestError test exception: {e}")

    # Test 6: ConfigurationError
    total_tests += 1
    print("\nTest 6: ConfigurationError")
    try:
        config_error = ConfigurationError(
            "Missing API key",
            details={"key": "binance_api_key", "file": "config.yaml"}
        )
        if not isinstance(config_error, CryptoTraderError):
            all_validation_failures.append("ConfigurationError should inherit from CryptoTraderError")
        if "binance_api_key" not in str(config_error):
            all_validation_failures.append(f"API key not in error: {str(config_error)}")

        print(f"  ✓ ConfigurationError: {config_error}")
    except Exception as e:
        all_validation_failures.append(f"ConfigurationError test exception: {e}")

    # Test 7: ValidationError
    total_tests += 1
    print("\nTest 7: ValidationError")
    try:
        validation_error = ValidationError(
            "Quantity exceeds maximum",
            details={"quantity": 100, "max_quantity": 50}
        )
        if not isinstance(validation_error, CryptoTraderError):
            all_validation_failures.append("ValidationError should inherit from CryptoTraderError")
        if "quantity=100" not in str(validation_error):
            all_validation_failures.append(f"Quantity not in error: {str(validation_error)}")

        print(f"  ✓ ValidationError: {validation_error}")
    except Exception as e:
        all_validation_failures.append(f"ValidationError test exception: {e}")

    # Test 8: OrderExecutionError
    total_tests += 1
    print("\nTest 8: OrderExecutionError")
    try:
        order_error = OrderExecutionError(
            "Insufficient balance",
            details={"required": 1000.0, "available": 500.0}
        )
        if not isinstance(order_error, CryptoTraderError):
            all_validation_failures.append("OrderExecutionError should inherit from CryptoTraderError")
        if "required=1000.0" not in str(order_error):
            all_validation_failures.append(f"Balance not in error: {str(order_error)}")

        print(f"  ✓ OrderExecutionError: {order_error}")
    except Exception as e:
        all_validation_failures.append(f"OrderExecutionError test exception: {e}")

    # Test 9: RiskManagementError
    total_tests += 1
    print("\nTest 9: RiskManagementError")
    try:
        risk_error = RiskManagementError(
            "Daily loss limit exceeded",
            details={"daily_loss": 0.05, "max_daily_loss": 0.03}
        )
        if not isinstance(risk_error, CryptoTraderError):
            all_validation_failures.append("RiskManagementError should inherit from CryptoTraderError")
        if "daily_loss=0.05" not in str(risk_error):
            all_validation_failures.append(f"Loss not in error: {str(risk_error)}")

        print(f"  ✓ RiskManagementError: {risk_error}")
    except Exception as e:
        all_validation_failures.append(f"RiskManagementError test exception: {e}")

    # Test 10: Inheritance chain validation
    total_tests += 1
    print("\nTest 10: Inheritance chain validation")
    try:
        all_exception_classes = [
            DataFetchError,
            StrategyError,
            BacktestError,
            ConfigurationError,
            ValidationError,
            OrderExecutionError,
            RiskManagementError,
        ]

        for exc_class in all_exception_classes:
            test_exc = exc_class("Test message")
            if not isinstance(test_exc, CryptoTraderError):
                all_validation_failures.append(
                    f"{exc_class.__name__} does not inherit from CryptoTraderError"
                )
            if not isinstance(test_exc, Exception):
                all_validation_failures.append(
                    f"{exc_class.__name__} does not inherit from Exception"
                )

        print(f"  ✓ All {len(all_exception_classes)} exception classes inherit correctly")
    except Exception as e:
        all_validation_failures.append(f"Inheritance test exception: {e}")

    # Test 11: Exception can be raised and caught
    total_tests += 1
    print("\nTest 11: Exception raising and catching")
    try:
        exception_raised = False
        exception_caught = False

        try:
            raise DataFetchError("Test raise", details={"test": True})
        except CryptoTraderError as e:
            exception_raised = True
            exception_caught = True
            if "test=True" not in str(e):
                all_validation_failures.append("Raised exception details not preserved")

        if not exception_raised:
            all_validation_failures.append("Exception was not raised")
        if not exception_caught:
            all_validation_failures.append("Exception was not caught")

        print(f"  ✓ Exception raised and caught successfully")
    except Exception as e:
        all_validation_failures.append(f"Raise/catch test exception: {e}")

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
