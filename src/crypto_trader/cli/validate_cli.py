"""
CLI Validation Script

This script validates the CLI structure without requiring all implementation details
to be complete. It tests the CLI module structure, command registration, and
basic functionality.

**Purpose**: Validate that the CLI module is correctly structured and can be
imported without errors once implementation dependencies are satisfied.

**Usage**:
```bash
uv run python src/crypto_trader/cli/validate_cli.py
```
"""

import sys
from pathlib import Path

# Track all validation failures
all_validation_failures = []
total_tests = 0

print("🔍 Validating CLI module structure...\n")

# Test 1: Check that CLI directory exists
total_tests += 1
print("Test 1: CLI directory structure")
try:
    cli_dir = Path(__file__).parent
    commands_dir = cli_dir / "commands"

    if not cli_dir.exists():
        all_validation_failures.append("CLI directory does not exist")
    if not commands_dir.exists():
        all_validation_failures.append("commands directory does not exist")

    # Check for required files
    required_files = [
        cli_dir / "__init__.py",
        cli_dir / "app.py",
        commands_dir / "__init__.py",
        commands_dir / "data.py",
        commands_dir / "strategy.py",
        commands_dir / "backtest.py"
    ]

    for file_path in required_files:
        if not file_path.exists():
            all_validation_failures.append(f"Missing required file: {file_path.name}")
        else:
            print(f"  ✓ {file_path.relative_to(cli_dir)}")

except Exception as e:
    all_validation_failures.append(f"Directory structure test failed: {e}")

# Test 2: Verify module docstrings
total_tests += 1
print("\nTest 2: Module documentation")
try:
    import crypto_trader.cli
    import crypto_trader.cli.app
    import crypto_trader.cli.commands

    modules = [
        (crypto_trader.cli, "cli.__init__"),
        (crypto_trader.cli.app, "cli.app"),
        (crypto_trader.cli.commands, "cli.commands.__init__")
    ]

    for module, name in modules:
        if not module.__doc__:
            all_validation_failures.append(f"{name} missing docstring")
        else:
            print(f"  ✓ {name} has docstring")

except Exception as e:
    all_validation_failures.append(f"Module documentation test failed: {e}")

# Test 3: Check that Typer app structure is correct
total_tests += 1
print("\nTest 3: Typer app structure")
try:
    import typer
    from crypto_trader.cli.app import app, data_app, strategy_app, backtest_app

    apps = [
        (app, "main app"),
        (data_app, "data_app"),
        (strategy_app, "strategy_app"),
        (backtest_app, "backtest_app")
    ]

    for app_obj, name in apps:
        if not isinstance(app_obj, typer.Typer):
            all_validation_failures.append(f"{name} is not a Typer instance")
        else:
            print(f"  ✓ {name} is Typer instance")

except Exception as e:
    all_validation_failures.append(f"Typer app structure test failed: {e}")

# Test 4: Check Rich console usage
total_tests += 1
print("\nTest 4: Rich console integration")
try:
    from rich.console import Console
    from crypto_trader.cli.commands import data, strategy, backtest

    for module, name in [(data, "data"), (strategy, "strategy"), (backtest, "backtest")]:
        if hasattr(module, "console"):
            if isinstance(module.console, Console):
                print(f"  ✓ {name}.console is Rich Console")
            else:
                all_validation_failures.append(f"{name}.console is not Rich Console")
        else:
            all_validation_failures.append(f"{name} module missing console")

except Exception as e:
    all_validation_failures.append(f"Rich console test failed: {e}")

# Test 5: Check command function existence
total_tests += 1
print("\nTest 5: Command functions")
try:
    from crypto_trader.cli.commands import data, strategy, backtest

    # Data commands
    data_commands = ['fetch', 'update', 'list_data', 'validate']
    for cmd in data_commands:
        if not hasattr(data, cmd):
            all_validation_failures.append(f"data.{cmd} function missing")
        elif not callable(getattr(data, cmd)):
            all_validation_failures.append(f"data.{cmd} is not callable")
        else:
            print(f"  ✓ data.{cmd} exists")

    # Strategy commands
    strategy_commands = ['list_strategies_cmd', 'info', 'test', 'validate']
    for cmd in strategy_commands:
        if not hasattr(strategy, cmd):
            all_validation_failures.append(f"strategy.{cmd} function missing")
        elif not callable(getattr(strategy, cmd)):
            all_validation_failures.append(f"strategy.{cmd} is not callable")
        else:
            print(f"  ✓ strategy.{cmd} exists")

    # Backtest commands
    backtest_commands = ['run', 'compare', 'optimize', 'report']
    for cmd in backtest_commands:
        if not hasattr(backtest, cmd):
            all_validation_failures.append(f"backtest.{cmd} function missing")
        elif not callable(getattr(backtest, cmd)):
            all_validation_failures.append(f"backtest.{cmd} is not callable")
        else:
            print(f"  ✓ backtest.{cmd} exists")

except Exception as e:
    all_validation_failures.append(f"Command functions test failed: {e}")

# Test 6: Check __all__ exports
total_tests += 1
print("\nTest 6: Module exports")
try:
    from crypto_trader.cli import __all__ as cli_all
    from crypto_trader.cli.commands import __all__ as commands_all

    if "get_app" not in cli_all:
        all_validation_failures.append("cli.__all__ missing 'get_app'")
    else:
        print("  ✓ cli.__all__ exports 'get_app'")

    expected_commands = {"data", "strategy", "backtest"}
    actual_commands = set(commands_all)

    if actual_commands != expected_commands:
        all_validation_failures.append(
            f"commands.__all__ mismatch: expected {expected_commands}, got {actual_commands}"
        )
    else:
        print(f"  ✓ commands.__all__ exports {len(commands_all)} modules")

except Exception as e:
    all_validation_failures.append(f"Module exports test failed: {e}")

# Test 7: Check entry point script location
total_tests += 1
print("\nTest 7: Entry point configuration")
try:
    import toml

    pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"

    if pyproject_path.exists():
        with open(pyproject_path, 'r') as f:
            config = toml.load(f)

        scripts = config.get('project', {}).get('scripts', {})

        if 'crypto-trader' in scripts:
            entry_point = scripts['crypto-trader']
            if entry_point == "crypto_trader.cli.app:app":
                print(f"  ✓ Entry point configured: crypto-trader")
            else:
                all_validation_failures.append(
                    f"Entry point incorrect: {entry_point}"
                )
        else:
            all_validation_failures.append("crypto-trader entry point not configured")
    else:
        print("  ⚠ pyproject.toml not found, skipping entry point check")

except ImportError:
    print("  ⚠ toml module not available, skipping entry point check")
except Exception as e:
    all_validation_failures.append(f"Entry point check failed: {e}")

# Test 8: Check for validation blocks in all files
total_tests += 1
print("\nTest 8: Validation blocks")
try:
    cli_dir = Path(__file__).parent

    python_files = [
        cli_dir / "app.py",
        cli_dir / "commands" / "__init__.py",
        cli_dir / "commands" / "data.py",
        cli_dir / "commands" / "strategy.py",
        cli_dir / "commands" / "backtest.py"
    ]

    for file_path in python_files:
        with open(file_path, 'r') as f:
            content = f.read()

        if 'if __name__ == "__main__":' in content:
            if "all_validation_failures" in content and "total_tests" in content:
                print(f"  ✓ {file_path.name} has validation block")
            else:
                all_validation_failures.append(
                    f"{file_path.name} validation block incomplete"
                )
        else:
            all_validation_failures.append(
                f"{file_path.name} missing validation block"
            )

except Exception as e:
    all_validation_failures.append(f"Validation blocks test failed: {e}")

# Final validation result
print("\n" + "="*60)
if all_validation_failures:
    print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
    for failure in all_validation_failures:
        print(f"  - {failure}")
    print("\n📝 Note: Some validation failures may be due to missing dependencies")
    print("   that will be satisfied by the implementation modules.")
    sys.exit(1)
else:
    print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
    print("\nCLI Structure Summary:")
    print("  • Main app with command groups ✓")
    print("  • Data management commands ✓")
    print("  • Strategy operation commands ✓")
    print("  • Backtesting commands ✓")
    print("  • Rich console integration ✓")
    print("  • Comprehensive documentation ✓")
    print("\nCLI is validated and ready for use!")
    print("\nUsage:")
    print("  uv run crypto-trader --help")
    print("  uv run crypto-trader data --help")
    print("  uv run crypto-trader strategy --help")
    print("  uv run crypto-trader backtest --help")
    sys.exit(0)
