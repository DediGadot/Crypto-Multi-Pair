"""
CLI Commands Module

This module exports all CLI command functions for data management, strategy
operations, and backtesting.

**Purpose**: Provide a centralized import point for all CLI command modules,
making it easy to register commands with the main Typer app.

**Exports**:
- data: Data management commands (fetch, update, list, validate)
- strategy: Strategy commands (list, info, test, validate)
- backtest: Backtest commands (run, compare, optimize, report)

**Third-party packages**:
- None (pure Python module for organizing imports)

**Sample Usage**:
```python
from crypto_trader.cli.commands import data, strategy, backtest

# Register with Typer app
data_app.command("fetch")(data.fetch)
strategy_app.command("list")(strategy.list_strategies)
backtest_app.command("run")(backtest.run)
```

**Expected Output**:
All command modules are available for import and registration.
"""

from crypto_trader.cli.commands import backtest, data, strategy

__all__ = ["data", "strategy", "backtest"]

if __name__ == "__main__":
    """
    Validation block for CLI commands module.
    Tests that all command modules can be imported.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating CLI commands module...\n")

    # Test 1: Verify data module is imported
    total_tests += 1
    print("Test 1: Data module import")
    try:
        if not hasattr(data, 'fetch'):
            all_validation_failures.append("data module missing 'fetch' function")
        if not hasattr(data, 'update'):
            all_validation_failures.append("data module missing 'update' function")
        if not hasattr(data, 'list_data'):
            all_validation_failures.append("data module missing 'list_data' function")
        if not hasattr(data, 'validate'):
            all_validation_failures.append("data module missing 'validate' function")

        if len(all_validation_failures) == 0:
            print("  ✓ data.fetch available")
            print("  ✓ data.update available")
            print("  ✓ data.list_data available")
            print("  ✓ data.validate available")
    except Exception as e:
        all_validation_failures.append(f"Data module test failed: {e}")

    # Test 2: Verify strategy module is imported
    total_tests += 1
    print("\nTest 2: Strategy module import")
    try:
        if not hasattr(strategy, 'list_strategies_cmd'):
            all_validation_failures.append("strategy module missing 'list_strategies_cmd' function")
        if not hasattr(strategy, 'info'):
            all_validation_failures.append("strategy module missing 'info' function")
        if not hasattr(strategy, 'test'):
            all_validation_failures.append("strategy module missing 'test' function")
        if not hasattr(strategy, 'validate'):
            all_validation_failures.append("strategy module missing 'validate' function")

        if len(all_validation_failures) == 0:
            print("  ✓ strategy.list_strategies_cmd available")
            print("  ✓ strategy.info available")
            print("  ✓ strategy.test available")
            print("  ✓ strategy.validate available")
    except Exception as e:
        all_validation_failures.append(f"Strategy module test failed: {e}")

    # Test 3: Verify backtest module is imported
    total_tests += 1
    print("\nTest 3: Backtest module import")
    try:
        if not hasattr(backtest, 'run'):
            all_validation_failures.append("backtest module missing 'run' function")
        if not hasattr(backtest, 'compare'):
            all_validation_failures.append("backtest module missing 'compare' function")
        if not hasattr(backtest, 'optimize'):
            all_validation_failures.append("backtest module missing 'optimize' function")
        if not hasattr(backtest, 'report'):
            all_validation_failures.append("backtest module missing 'report' function")

        if len(all_validation_failures) == 0:
            print("  ✓ backtest.run available")
            print("  ✓ backtest.compare available")
            print("  ✓ backtest.optimize available")
            print("  ✓ backtest.report available")
    except Exception as e:
        all_validation_failures.append(f"Backtest module test failed: {e}")

    # Test 4: Verify __all__ exports
    total_tests += 1
    print("\nTest 4: Module __all__ exports")
    try:
        expected_exports = {"data", "strategy", "backtest"}
        actual_exports = set(__all__)

        if actual_exports != expected_exports:
            all_validation_failures.append(
                f"__all__ mismatch: expected {expected_exports}, got {actual_exports}"
            )
        else:
            print(f"  ✓ __all__ contains {len(__all__)} exports")
            for export in __all__:
                print(f"    - {export}")
    except Exception as e:
        all_validation_failures.append(f"__all__ test failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("CLI commands module is validated and ready for use")
        sys.exit(0)
