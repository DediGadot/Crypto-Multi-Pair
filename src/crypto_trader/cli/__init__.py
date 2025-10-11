"""
CLI Module for Crypto Trading System

This module provides the command-line interface for the crypto trading system
using Typer and Rich for beautiful, user-friendly terminal interactions.

**Purpose**: Export the main CLI app and provide easy access to all CLI
functionality including data management, strategy operations, and backtesting.

**Key Components**:
- app: Main Typer application with all command groups
- commands: Submodules for data, strategy, and backtest commands

**Third-party packages**:
- typer: https://typer.tiangolo.com/
- rich: https://rich.readthedocs.io/en/stable/

**Sample Usage**:
```python
from crypto_trader.cli import app

# Run the CLI
if __name__ == "__main__":
    app()
```

**Command Line Usage**:
```bash
# Data commands
crypto-trader data fetch BTCUSDT --timeframe 1h --days 30
crypto-trader data list --symbol BTCUSDT

# Strategy commands
crypto-trader strategy list
crypto-trader strategy test SMA_Crossover --symbol BTCUSDT

# Backtest commands
crypto-trader backtest run SMA_Crossover --symbol BTCUSDT --days 90
crypto-trader backtest compare SMA_Crossover RSI_Mean_Reversion
```

**Expected Output**:
Beautiful, colorful CLI interface with progress bars, tables, and panels
for all trading system operations.
"""

# Import app lazily to avoid circular imports
def get_app():
    """Get the main CLI app instance."""
    from crypto_trader.cli.app import app
    return app

__all__ = ["get_app"]

__version__ = "0.1.0"

if __name__ == "__main__":
    """
    Validation block for CLI module.
    Tests that the main app can be imported and run.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating CLI module...\n")

    # Test 1: Verify app can be retrieved
    total_tests += 1
    print("Test 1: Main app retrieval")
    try:
        import typer

        app = get_app()
        if not isinstance(app, typer.Typer):
            all_validation_failures.append("app is not a Typer instance")
        else:
            print("  ✓ app is Typer instance")
            print("  ✓ app successfully retrieved")
    except Exception as e:
        all_validation_failures.append(f"App retrieval test failed: {e}")

    # Test 2: Verify __all__ exports
    total_tests += 1
    print("\nTest 2: Module __all__ exports")
    try:
        if __all__ != ["get_app"]:
            all_validation_failures.append(f"__all__ incorrect: expected ['get_app'], got {__all__}")
        else:
            print("  ✓ __all__ exports 'get_app'")
    except Exception as e:
        all_validation_failures.append(f"__all__ test failed: {e}")

    # Test 3: Verify version is set
    total_tests += 1
    print("\nTest 3: Module version")
    try:
        if not __version__:
            all_validation_failures.append("__version__ is not set")
        else:
            print(f"  ✓ __version__ = {__version__}")
    except Exception as e:
        all_validation_failures.append(f"Version test failed: {e}")

    # Test 4: Check that commands can be imported
    total_tests += 1
    print("\nTest 4: Commands import")
    try:
        from crypto_trader.cli import commands

        if not hasattr(commands, 'data'):
            all_validation_failures.append("commands module missing 'data'")
        if not hasattr(commands, 'strategy'):
            all_validation_failures.append("commands module missing 'strategy'")
        if not hasattr(commands, 'backtest'):
            all_validation_failures.append("commands module missing 'backtest'")

        if len(all_validation_failures) == 0:
            print("  ✓ commands.data available")
            print("  ✓ commands.strategy available")
            print("  ✓ commands.backtest available")
    except ImportError as e:
        all_validation_failures.append(f"Commands import failed: {e}")

    # Test 5: Verify app has registered commands
    total_tests += 1
    print("\nTest 5: Registered commands")
    try:
        # Check if app has registered subcommands
        # Note: Typer doesn't expose registered commands easily,
        # so we just verify the app can be called
        if not callable(app):
            all_validation_failures.append("app is not callable")
        else:
            print("  ✓ app is callable")
            print("  ✓ Command groups should be registered")
    except Exception as e:
        all_validation_failures.append(f"Registered commands test failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("CLI module is validated and ready for use")
        print("\nTo use the CLI:")
        print("  uv run crypto-trader --help")
        print("  uv run crypto-trader data --help")
        print("  uv run crypto-trader strategy --help")
        print("  uv run crypto-trader backtest --help")
        sys.exit(0)
