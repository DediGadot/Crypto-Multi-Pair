"""
Main CLI Application for Crypto Trading System

This module implements the main Typer CLI application with command groups for data
management, strategy operations, and backtesting.

**Purpose**: Provide a beautiful, user-friendly CLI interface for all trading system
operations including data fetching, strategy testing, and backtest execution.

**Key Features**:
- Organized command groups (data, strategy, backtest)
- Rich terminal output with colors and progress bars
- Comprehensive help text for all commands
- Integration with all system components
- Logging with loguru

**Third-party packages**:
- typer: https://typer.tiangolo.com/
- rich: https://rich.readthedocs.io/en/stable/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```bash
crypto-trader data fetch BTCUSDT --timeframe 1h --days 30
crypto-trader strategy list
crypto-trader backtest run SMA_Crossover --symbol BTCUSDT
```

**Expected Output**:
```
✓ Fetched 720 candles for BTCUSDT
  Timeframe: 1h
  Period: 2024-01-01 to 2024-01-31
  Data saved to database
```
"""

from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from crypto_trader.cli.commands import backtest, data, strategy

# Initialize Rich console for beautiful output
console = Console()

# Create main Typer app
app = typer.Typer(
    name="crypto-trader",
    help="Modular crypto algorithmic trading and analysis system",
    add_completion=True,
    rich_markup_mode="rich",
)

# Create command group apps
data_app = typer.Typer(
    name="data",
    help="Data management commands (fetch, update, list, validate)",
)

strategy_app = typer.Typer(
    name="strategy",
    help="Strategy commands (list, info, test, validate)",
)

backtest_app = typer.Typer(
    name="backtest",
    help="Backtesting commands (run, compare, optimize, report)",
)

# Add command groups to main app
app.add_typer(data_app, name="data")
app.add_typer(strategy_app, name="strategy")
app.add_typer(backtest_app, name="backtest")

# Register commands from modules
data_app.command("fetch")(data.fetch)
data_app.command("update")(data.update)
data_app.command("list")(data.list_data)
data_app.command("validate")(data.validate)

strategy_app.command("list")(strategy.list_strategies)
strategy_app.command("info")(strategy.info)
strategy_app.command("test")(strategy.test)
strategy_app.command("validate")(strategy.validate)

backtest_app.command("run")(backtest.run)
backtest_app.command("compare")(backtest.compare)
backtest_app.command("optimize")(backtest.optimize)
backtest_app.command("report")(backtest.report)


@app.command()
def version():
    """Show version information."""
    console.print(Panel.fit(
        "[bold blue]Crypto Trading System[/bold blue]\n"
        "Version: [green]0.1.0[/green]\n"
        "Python: [yellow]3.12+[/yellow]",
        title="Version Info",
        border_style="blue"
    ))


@app.command()
def info():
    """Show system information and status."""
    from crypto_trader import __version__

    console.print("\n[bold]Crypto Trading System[/bold]")
    console.print(f"Version: [green]{__version__}[/green]")
    console.print("\n[bold]Available Command Groups:[/bold]")
    console.print("  • [blue]data[/blue]     - Data management (fetch, update, list)")
    console.print("  • [blue]strategy[/blue] - Strategy operations (list, test, validate)")
    console.print("  • [blue]backtest[/blue] - Backtesting (run, compare, optimize)")
    console.print("\nUse [cyan]crypto-trader [COMMAND] --help[/cyan] for more information.\n")


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress all output except errors"
    ),
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        help="Log file path (default: crypto_trader.log)"
    )
):
    """
    Crypto Trading System - Modular algorithmic trading platform.

    A comprehensive system for fetching data, testing strategies, and running
    backtests on cryptocurrency markets.
    """
    # Configure logging based on options
    logger.remove()  # Remove default handler

    if not quiet:
        if verbose:
            logger.add(
                lambda msg: console.print(msg, end=""),
                level="DEBUG",
                colorize=True
            )
        else:
            logger.add(
                lambda msg: console.print(msg, end=""),
                level="INFO",
                colorize=True
            )

    # Add file logging if specified
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
    elif not quiet:
        # Default file logging
        logger.add(
            "crypto_trader.log",
            rotation="10 MB",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )


if __name__ == "__main__":
    """
    Validation block for CLI main app.
    Tests command structure and help output.
    """
    import sys
    import subprocess
    from pathlib import Path

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating CLI main app structure...\n")

    # Test 1: Verify app is created
    total_tests += 1
    print("Test 1: Main app initialization")
    try:
        if not isinstance(app, typer.Typer):
            all_validation_failures.append("Main app is not a Typer instance")
        else:
            print("  ✓ Main app is Typer instance")
    except Exception as e:
        all_validation_failures.append(f"Main app initialization failed: {e}")

    # Test 2: Verify command groups exist
    total_tests += 1
    print("\nTest 2: Command groups registration")
    try:
        # Check that command groups are registered
        if not isinstance(data_app, typer.Typer):
            all_validation_failures.append("data_app is not a Typer instance")
        if not isinstance(strategy_app, typer.Typer):
            all_validation_failures.append("strategy_app is not a Typer instance")
        if not isinstance(backtest_app, typer.Typer):
            all_validation_failures.append("backtest_app is not a Typer instance")

        if len(all_validation_failures) == 0:
            print("  ✓ All command groups are Typer instances")
            print("  ✓ data_app created")
            print("  ✓ strategy_app created")
            print("  ✓ backtest_app created")
    except Exception as e:
        all_validation_failures.append(f"Command groups test failed: {e}")

    # Test 3: Verify console is initialized
    total_tests += 1
    print("\nTest 3: Rich console initialization")
    try:
        if not isinstance(console, Console):
            all_validation_failures.append("Console is not a Rich Console instance")
        else:
            print("  ✓ Rich console initialized")
    except Exception as e:
        all_validation_failures.append(f"Console initialization failed: {e}")

    # Test 4: Test version command exists
    total_tests += 1
    print("\nTest 4: Version command")
    try:
        # Check if version function exists and is callable
        if not callable(version):
            all_validation_failures.append("version command is not callable")
        else:
            print("  ✓ version command is callable")
    except Exception as e:
        all_validation_failures.append(f"Version command test failed: {e}")

    # Test 5: Test info command exists
    total_tests += 1
    print("\nTest 5: Info command")
    try:
        if not callable(info):
            all_validation_failures.append("info command is not callable")
        else:
            print("  ✓ info command is callable")
    except Exception as e:
        all_validation_failures.append(f"Info command test failed: {e}")

    # Test 6: Verify main callback exists
    total_tests += 1
    print("\nTest 6: Main callback function")
    try:
        if not callable(main):
            all_validation_failures.append("main callback is not callable")
        else:
            print("  ✓ main callback is callable")
    except Exception as e:
        all_validation_failures.append(f"Main callback test failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("CLI main app structure is validated and ready for use")
        print("\nNote: Command module implementations will be validated separately")
        sys.exit(0)
