"""
Core module for the crypto trading system.

This module provides the foundational components including type definitions,
exceptions, and configuration management. All public APIs are exported here
for convenient access.

Documentation:
- Project structure follows modern Python packaging standards
- All exports are explicitly defined for better IDE support
- Type definitions use Python 3.12+ features

Sample Usage:
    from crypto_trader.core import (
        Timeframe,
        OrderType,
        OrderSide,
        PerformanceMetrics,
        BacktestResult,
        TradingConfig,
        DataFetchError
    )

    # Use enums
    tf = Timeframe.HOUR_1

    # Create config
    config = TradingConfig.from_yaml("config.yaml")

    # Raise exceptions
    raise DataFetchError("Connection failed", details={"exchange": "binance"})

Expected Output:
    Clean imports with proper type hints and IDE autocomplete support
"""

# Type definitions
from crypto_trader.core.types import (
    # Enums
    Timeframe,
    OrderType,
    OrderSide,
    Signal,
    # Dataclasses
    PerformanceMetrics,
    Trade,
    BacktestResult,
)

# Exceptions
from crypto_trader.core.exceptions import (
    CryptoTraderError,
    DataFetchError,
    StrategyError,
    BacktestError,
    ConfigurationError,
    ValidationError,
    OrderExecutionError,
    RiskManagementError,
)

# Configuration
from crypto_trader.core.config import (
    DataConfig,
    StrategyConfig,
    BacktestConfig,
    RiskConfig,
    TradingConfig,
)

# Version information
__version__ = "0.1.0"
__author__ = "Crypto Trading Team"

# Public API - explicitly define what gets exported with *
__all__ = [
    # Version
    "__version__",
    "__author__",
    # Types - Enums
    "Timeframe",
    "OrderType",
    "OrderSide",
    "Signal",
    # Types - Dataclasses
    "PerformanceMetrics",
    "Trade",
    "BacktestResult",
    # Exceptions
    "CryptoTraderError",
    "DataFetchError",
    "StrategyError",
    "BacktestError",
    "ConfigurationError",
    "ValidationError",
    "OrderExecutionError",
    "RiskManagementError",
    # Configuration
    "DataConfig",
    "StrategyConfig",
    "BacktestConfig",
    "RiskConfig",
    "TradingConfig",
]


if __name__ == "__main__":
    """
    Validation function to verify all exports are accessible.
    Tests that all public APIs can be imported and used correctly.
    """
    import sys
    from datetime import datetime

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating __init__.py exports with real data...\n")

    # Test 1: Enum exports
    total_tests += 1
    print("Test 1: Enum exports")
    try:
        # Test Timeframe
        tf = Timeframe.HOUR_1
        if tf.value != "1h":
            all_validation_failures.append(f"Timeframe export: Expected '1h', got '{tf.value}'")

        # Test OrderType
        ot = OrderType.MARKET
        if ot.value != "market":
            all_validation_failures.append(f"OrderType export: Expected 'market', got '{ot.value}'")

        # Test OrderSide
        os = OrderSide.BUY
        if os.value != "buy":
            all_validation_failures.append(f"OrderSide export: Expected 'buy', got '{os.value}'")

        # Test Signal
        sig: Signal = 1
        if sig != 1:
            all_validation_failures.append(f"Signal export: Expected 1, got {sig}")

        print(f"  ✓ Timeframe: {tf.value}")
        print(f"  ✓ OrderType: {ot.value}")
        print(f"  ✓ OrderSide: {os.value}")
        print(f"  ✓ Signal: {sig}")
    except Exception as e:
        all_validation_failures.append(f"Enum exports test exception: {e}")

    # Test 2: Dataclass exports
    total_tests += 1
    print("\nTest 2: Dataclass exports")
    try:
        # Test PerformanceMetrics
        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.8,
            max_drawdown=0.12
        )
        if metrics.total_return != 0.15:
            all_validation_failures.append(f"PerformanceMetrics: Expected 0.15, got {metrics.total_return}")

        # Test Trade
        trade = Trade(
            symbol="BTCUSDT",
            entry_time=datetime(2025, 1, 1),
            exit_time=datetime(2025, 1, 2),
            entry_price=45000.0,
            exit_price=46000.0,
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=100.0,
            pnl_percent=2.22,
            fees=10.0
        )
        if trade.symbol != "BTCUSDT":
            all_validation_failures.append(f"Trade: Expected 'BTCUSDT', got '{trade.symbol}'")

        # Test BacktestResult
        result = BacktestResult(
            strategy_name="Test",
            symbol="ETHUSDT",
            timeframe=Timeframe.HOUR_4,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=10000.0,
            metrics=metrics
        )
        if result.strategy_name != "Test":
            all_validation_failures.append(f"BacktestResult: Expected 'Test', got '{result.strategy_name}'")

        print(f"  ✓ PerformanceMetrics created")
        print(f"  ✓ Trade created")
        print(f"  ✓ BacktestResult created")
    except Exception as e:
        all_validation_failures.append(f"Dataclass exports test exception: {e}")

    # Test 3: Exception exports
    total_tests += 1
    print("\nTest 3: Exception exports")
    try:
        exception_classes = [
            (CryptoTraderError, "Base error"),
            (DataFetchError, "Data fetch failed"),
            (StrategyError, "Strategy error"),
            (BacktestError, "Backtest error"),
            (ConfigurationError, "Config error"),
            (ValidationError, "Validation error"),
            (OrderExecutionError, "Order error"),
            (RiskManagementError, "Risk error"),
        ]

        for exc_class, message in exception_classes:
            exc = exc_class(message)
            if not isinstance(exc, Exception):
                all_validation_failures.append(f"{exc_class.__name__} is not an Exception")
            if message not in str(exc):
                all_validation_failures.append(f"Message not in {exc_class.__name__}: {exc}")

        print(f"  ✓ All {len(exception_classes)} exception classes exported")
    except Exception as e:
        all_validation_failures.append(f"Exception exports test exception: {e}")

    # Test 4: Configuration exports
    total_tests += 1
    print("\nTest 4: Configuration exports")
    try:
        # Test DataConfig
        data_config = DataConfig(exchange="binance")
        if data_config.exchange != "binance":
            all_validation_failures.append(f"DataConfig: Expected 'binance', got '{data_config.exchange}'")

        # Test StrategyConfig
        strategy_config = StrategyConfig(name="MA")
        if strategy_config.name != "MA":
            all_validation_failures.append(f"StrategyConfig: Expected 'MA', got '{strategy_config.name}'")

        # Test BacktestConfig
        backtest_config = BacktestConfig(initial_capital=20000.0)
        if backtest_config.initial_capital != 20000.0:
            all_validation_failures.append(f"BacktestConfig: Expected 20000.0, got {backtest_config.initial_capital}")

        # Test RiskConfig
        risk_config = RiskConfig(max_open_positions=5)
        if risk_config.max_open_positions != 5:
            all_validation_failures.append(f"RiskConfig: Expected 5, got {risk_config.max_open_positions}")

        # Test TradingConfig
        trading_config = TradingConfig(
            data=data_config,
            strategy=strategy_config,
            backtest=backtest_config,
            risk=risk_config
        )
        if trading_config.data.exchange != "binance":
            all_validation_failures.append(f"TradingConfig: Exchange mismatch")

        print(f"  ✓ DataConfig created")
        print(f"  ✓ StrategyConfig created")
        print(f"  ✓ BacktestConfig created")
        print(f"  ✓ RiskConfig created")
        print(f"  ✓ TradingConfig created")
    except Exception as e:
        all_validation_failures.append(f"Configuration exports test exception: {e}")

    # Test 5: __all__ completeness
    total_tests += 1
    print("\nTest 5: __all__ completeness")
    try:
        # Verify all expected exports are in __all__
        expected_exports = {
            "Timeframe", "OrderType", "OrderSide", "Signal",
            "PerformanceMetrics", "Trade", "BacktestResult",
            "CryptoTraderError", "DataFetchError", "StrategyError",
            "BacktestError", "ConfigurationError", "ValidationError",
            "OrderExecutionError", "RiskManagementError",
            "DataConfig", "StrategyConfig", "BacktestConfig",
            "RiskConfig", "TradingConfig",
            "__version__", "__author__"
        }

        missing_exports = expected_exports - set(__all__)
        if missing_exports:
            all_validation_failures.append(f"Missing from __all__: {missing_exports}")

        print(f"  ✓ __all__ contains {len(__all__)} exports")
        print(f"  ✓ All expected exports present")
    except Exception as e:
        all_validation_failures.append(f"__all__ test exception: {e}")

    # Test 6: Version information
    total_tests += 1
    print("\nTest 6: Version information")
    try:
        if not isinstance(__version__, str):
            all_validation_failures.append(f"__version__ should be string, got {type(__version__)}")
        if not __version__:
            all_validation_failures.append("__version__ should not be empty")

        if not isinstance(__author__, str):
            all_validation_failures.append(f"__author__ should be string, got {type(__author__)}")
        if not __author__:
            all_validation_failures.append("__author__ should not be empty")

        print(f"  ✓ Version: {__version__}")
        print(f"  ✓ Author: {__author__}")
    except Exception as e:
        all_validation_failures.append(f"Version info test exception: {e}")

    # Test 7: Import pattern validation
    total_tests += 1
    print("\nTest 7: Import pattern validation")
    try:
        # Verify that star import would work
        from crypto_trader.core import *

        # Check a few key exports are accessible
        if Timeframe.HOUR_1.value != "1h":
            all_validation_failures.append("Star import: Timeframe not accessible")
        if not issubclass(DataFetchError, CryptoTraderError):
            all_validation_failures.append("Star import: Exception hierarchy broken")

        print(f"  ✓ Star import works correctly")
        print(f"  ✓ All public APIs accessible")
    except Exception as e:
        all_validation_failures.append(f"Import pattern test exception: {e}")

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
