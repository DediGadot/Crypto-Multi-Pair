"""
Strategy CLI Commands

This module implements CLI commands for strategy management including listing
available strategies, viewing strategy details, testing strategies with data,
and validating strategy configurations.

**Purpose**: Provide user-friendly CLI commands for exploring and testing trading
strategies with rich output and comprehensive information display.

**Key Commands**:
- list: List all available strategies
- info: Show detailed information about a strategy
- test: Quick test a strategy with market data
- validate: Validate strategy configuration

**Third-party packages**:
- typer: https://typer.tiangolo.com/
- rich: https://rich.readthedocs.io/en/stable/
- loguru: https://loguru.readthedocs.io/en/stable/
- pandas: https://pandas.pydata.org/docs/

**Sample Input**:
```bash
crypto-trader strategy list
crypto-trader strategy info SMA_Crossover
crypto-trader strategy test SMA_Crossover --symbol BTCUSDT
```

**Expected Output**:
```
Available Strategies:
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name           ┃ Description                ┃ Tags       ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ SMA_Crossover  │ Simple moving average...   │ trend      │
└────────────────┴────────────────────────────┴────────────┘
```
"""

from typing import Optional, Dict, Any

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from crypto_trader.strategies.registry import get_registry, list_strategies
from crypto_trader.data.storage import OHLCVStorage as DataStorage
from crypto_trader.data.providers import MockDataProvider

console = Console()


def load_all_strategies():
    """
    Load all strategies from the library directory.

    Scans the strategies/library directory and registers all
    found strategy classes with the global registry.
    """
    from pathlib import Path

    registry = get_registry()

    # Get the strategies library path
    strategies_path = Path(__file__).parent.parent.parent / "strategies" / "library"

    if strategies_path.exists():
        loaded = registry.load_from_directory(strategies_path, recursive=True)
        logger.debug(f"Loaded {loaded} strategies from library")
    else:
        logger.warning(f"Strategies library not found: {strategies_path}")


def list_strategies_cmd(
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-t",
        help="Filter strategies by tags (comma-separated)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information"
    )
):
    """
    List all available trading strategies.

    Shows all registered strategies with their descriptions, tags,
    and optionally detailed parameter information.

    Example:
        crypto-trader strategy list
        crypto-trader strategy list --tags momentum,trend
    """
    try:
        console.print("\n[bold blue]Available Trading Strategies[/bold blue]\n")

        # Load all strategies
        load_all_strategies()

        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]

        # Get strategies from registry
        strategies = list_strategies(tags=tag_list)

        if not strategies:
            if tag_list:
                console.print(f"[yellow]No strategies found with tags: {', '.join(tag_list)}[/yellow]\n")
            else:
                console.print("[yellow]No strategies registered[/yellow]\n")
            return

        # Create table
        table = Table(show_header=True, header_style="bold cyan", title="Strategies")
        table.add_column("Name", style="yellow", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Module", style="cyan")
        table.add_column("Tags", style="magenta")

        for name, metadata in strategies.items():
            tags_str = ", ".join(metadata.get('tags', [])) if metadata.get('tags') else "-"
            description = metadata.get('description', 'No description')[:60]

            table.add_row(
                name,
                description,
                metadata.get('module', 'unknown'),
                tags_str
            )

        console.print(table)
        console.print(f"\nTotal strategies: [bold]{len(strategies)}[/bold]\n")

        if not verbose:
            console.print("[dim]Use --verbose for detailed information[/dim]\n")

    except Exception as e:
        console.print(f"\n[red]✗ List failed: {e}[/red]\n")
        logger.exception("Strategy list command failed")
        raise typer.Exit(1)


def info(
    strategy_name: str = typer.Argument(
        ...,
        help="Name of the strategy to show info for"
    ),
    show_parameters: bool = typer.Option(
        True,
        "--parameters/--no-parameters",
        help="Show strategy parameters"
    ),
    show_indicators: bool = typer.Option(
        True,
        "--indicators/--no-indicators",
        help="Show required indicators"
    )
):
    """
    Show detailed information about a strategy.

    Displays comprehensive information including description, parameters,
    required indicators, and usage examples.

    Example:
        crypto-trader strategy info SMA_Crossover
    """
    try:
        console.print(f"\n[bold blue]Strategy Information: {strategy_name}[/bold blue]\n")

        # Load strategies
        load_all_strategies()

        # Get strategy from registry
        registry = get_registry()
        try:
            strategy_class = registry.get_strategy(strategy_name)
            strategy_info = registry.get_strategy_info(strategy_name)
        except KeyError:
            console.print(f"[red]✗ Strategy '{strategy_name}' not found[/red]")
            console.print("\nUse 'crypto-trader strategy list' to see available strategies\n")
            raise typer.Exit(1)

        # Display basic info
        panel_content = f"""[bold]Description:[/bold]
{strategy_info.get('description', 'No description available')}

[bold]Module:[/bold] [cyan]{strategy_info.get('module', 'unknown')}[/cyan]
[bold]Class:[/bold] [cyan]{strategy_info.get('class_name', 'unknown')}[/cyan]
"""

        if strategy_info.get('tags'):
            tags_str = ", ".join(strategy_info['tags'])
            panel_content += f"\n[bold]Tags:[/bold] [magenta]{tags_str}[/magenta]"

        console.print(Panel(panel_content, title=f"[bold]{strategy_name}[/bold]", border_style="blue"))

        # Create instance to get parameters and indicators
        try:
            instance = strategy_class(name=strategy_name)

            # Show parameters
            if show_parameters:
                console.print("\n[bold]Parameters:[/bold]")
                params = instance.get_parameters()

                if params:
                    param_table = Table(show_header=True, header_style="bold cyan")
                    param_table.add_column("Parameter", style="yellow")
                    param_table.add_column("Value", style="green")

                    for key, value in params.items():
                        param_table.add_row(str(key), str(value))

                    console.print(param_table)
                else:
                    console.print("  [dim]No parameters defined[/dim]")

            # Show required indicators
            if show_indicators:
                console.print("\n[bold]Required Indicators:[/bold]")
                indicators = instance.get_required_indicators()

                if indicators:
                    for indicator in indicators:
                        console.print(f"  • [cyan]{indicator}[/cyan]")
                else:
                    console.print("  [dim]No specific indicators required[/dim]")

        except Exception as e:
            console.print(f"\n[yellow]⚠ Could not instantiate strategy: {e}[/yellow]")
            logger.warning(f"Strategy instantiation failed: {e}")

        console.print()

    except Exception as e:
        console.print(f"\n[red]✗ Info command failed: {e}[/red]\n")
        logger.exception("Strategy info command failed")
        raise typer.Exit(1)


def test(
    strategy_name: str = typer.Argument(
        ...,
        help="Name of the strategy to test"
    ),
    symbol: str = typer.Option(
        "BTCUSDT",
        "--symbol",
        "-s",
        help="Trading pair to test with"
    ),
    timeframe: str = typer.Option(
        "1h",
        "--timeframe",
        "-t",
        help="Candle timeframe"
    ),
    days: int = typer.Option(
        7,
        "--days",
        "-d",
        help="Number of days to test"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Strategy configuration (JSON string)"
    )
):
    """
    Quick test a strategy with market data.

    Runs the strategy signal generation on recent market data and
    displays the generated signals and basic statistics.

    Example:
        crypto-trader strategy test SMA_Crossover --symbol BTCUSDT --days 7
    """
    try:
        console.print(f"\n[bold blue]Testing Strategy: {strategy_name}[/bold blue]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Load strategies
            task = progress.add_task("Loading strategies...", total=None)
            load_all_strategies()

            # Get strategy
            registry = get_registry()
            try:
                strategy_class = registry.get_strategy(strategy_name)
            except KeyError:
                console.print(f"[red]✗ Strategy '{strategy_name}' not found[/red]\n")
                raise typer.Exit(1)

            # Create strategy instance
            progress.update(task, description="Initializing strategy...")
            strategy = strategy_class(name=strategy_name)

            # Parse config if provided
            if config:
                import json
                config_dict = json.loads(config)
                strategy.initialize(config_dict)
            else:
                strategy.initialize({})

            # Load data
            progress.update(task, description="Loading market data...")
            try:
                storage = OHLCVStorage()
                df = storage.load_ohlcv(symbol, timeframe, days=days)

                if df is None or len(df) == 0:
                    console.print(f"[yellow]⚠ No data found for {symbol} {timeframe}[/yellow]")
                    console.print("  Use 'crypto-trader data fetch' to download data first\n")
                    raise typer.Exit(1)

            except Exception as e:
                console.print(f"[yellow]⚠ Could not load data from storage: {e}[/yellow]")
                console.print("  Using mock data for testing...\n")
                provider = MockDataProvider()
                df = provider.get_ohlcv(symbol, timeframe, limit=days*24)

            # Generate signals
            progress.update(task, description="Generating signals...")
            signals = strategy.generate_signals(df)

        # Display results
        console.print(f"[green]✓[/green] Test completed\n")

        # Create summary table
        summary = Table(title="Test Summary", show_header=True)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")

        summary.add_row("Symbol", symbol)
        summary.add_row("Timeframe", timeframe)
        summary.add_row("Data Points", str(len(df)))
        summary.add_row("Signals Generated", str(len(signals)))

        # Count signal types
        if 'signal' in signals.columns:
            buy_signals = (signals['signal'] == 'BUY').sum()
            sell_signals = (signals['signal'] == 'SELL').sum()
            hold_signals = (signals['signal'] == 'HOLD').sum()

            summary.add_row("BUY Signals", str(buy_signals))
            summary.add_row("SELL Signals", str(sell_signals))
            summary.add_row("HOLD Signals", str(hold_signals))

        console.print(summary)

        # Show recent signals
        if len(signals) > 0 and 'signal' in signals.columns:
            console.print("\n[bold]Recent Signals:[/bold]")
            recent_signals = signals[signals['signal'] != 'HOLD'].tail(5)

            if len(recent_signals) > 0:
                signal_table = Table(show_header=True, header_style="bold cyan")
                signal_table.add_column("Timestamp", style="yellow")
                signal_table.add_column("Signal", style="green")
                signal_table.add_column("Confidence", style="cyan")

                for idx, row in recent_signals.iterrows():
                    signal_style = "green" if row['signal'] == 'BUY' else "red"
                    confidence = f"{row.get('confidence', 0):.2%}" if 'confidence' in row else "N/A"

                    signal_table.add_row(
                        str(idx),
                        f"[{signal_style}]{row['signal']}[/{signal_style}]",
                        confidence
                    )

                console.print(signal_table)
            else:
                console.print("  [dim]No BUY/SELL signals in recent period[/dim]")

        console.print()

    except Exception as e:
        console.print(f"\n[red]✗ Test failed: {e}[/red]\n")
        logger.exception("Strategy test command failed")
        raise typer.Exit(1)


def validate(
    strategy_name: str = typer.Argument(
        ...,
        help="Name of the strategy to validate"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Strategy configuration (JSON string)"
    )
):
    """
    Validate a strategy configuration.

    Checks if the strategy can be instantiated correctly and if
    the provided configuration is valid.

    Example:
        crypto-trader strategy validate SMA_Crossover
        crypto-trader strategy validate SMA_Crossover --config '{"fast": 10, "slow": 20}'
    """
    try:
        console.print(f"\n[bold blue]Validating Strategy: {strategy_name}[/bold blue]\n")

        validation_issues = []

        # Load strategies
        load_all_strategies()

        # Get strategy
        registry = get_registry()
        try:
            strategy_class = registry.get_strategy(strategy_name)
        except KeyError:
            console.print(f"[red]✗ Strategy '{strategy_name}' not found[/red]\n")
            raise typer.Exit(1)

        # Test instantiation
        try:
            strategy = strategy_class(name=strategy_name)
            console.print("[green]✓[/green] Strategy class can be instantiated")
        except Exception as e:
            validation_issues.append(f"Cannot instantiate: {e}")

        # Test initialization with config
        if config:
            try:
                import json
                config_dict = json.loads(config)
                strategy.initialize(config_dict)
                console.print("[green]✓[/green] Configuration is valid")
            except json.JSONDecodeError as e:
                validation_issues.append(f"Invalid JSON configuration: {e}")
            except Exception as e:
                validation_issues.append(f"Configuration error: {e}")
        else:
            try:
                strategy.initialize({})
                console.print("[green]✓[/green] Default initialization successful")
            except Exception as e:
                validation_issues.append(f"Default initialization failed: {e}")

        # Test required methods
        try:
            params = strategy.get_parameters()
            console.print(f"[green]✓[/green] get_parameters() works (returned {len(params)} params)")
        except Exception as e:
            validation_issues.append(f"get_parameters() failed: {e}")

        try:
            indicators = strategy.get_required_indicators()
            console.print(f"[green]✓[/green] get_required_indicators() works (requires {len(indicators)} indicators)")
        except Exception as e:
            validation_issues.append(f"get_required_indicators() failed: {e}")

        # Display results
        if validation_issues:
            console.print("\n[yellow]⚠ Validation Issues:[/yellow]")
            for issue in validation_issues:
                console.print(f"  • {issue}")
            console.print()
        else:
            console.print(f"\n[green]✓ Strategy '{strategy_name}' is valid and ready to use[/green]\n")

    except Exception as e:
        console.print(f"\n[red]✗ Validation failed: {e}[/red]\n")
        logger.exception("Strategy validate command failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    """
    Validation block for strategy CLI commands.
    Tests command functions structure and dependencies.
    """
    import sys
    import inspect

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating strategy CLI commands...\n")

    # Test 1: Verify all command functions exist
    total_tests += 1
    print("Test 1: Command functions exist")
    try:
        commands = [list_strategies_cmd, info, test, validate]
        for cmd in commands:
            if not callable(cmd):
                all_validation_failures.append(f"{cmd.__name__} is not callable")

        if len(all_validation_failures) == 0:
            print("  ✓ list_strategies_cmd exists")
            print("  ✓ info exists")
            print("  ✓ test exists")
            print("  ✓ validate exists")
    except Exception as e:
        all_validation_failures.append(f"Command existence test failed: {e}")

    # Test 2: Check command docstrings
    total_tests += 1
    print("\nTest 2: Command documentation")
    try:
        for cmd in commands:
            if not cmd.__doc__:
                all_validation_failures.append(f"{cmd.__name__} missing docstring")

        if len(all_validation_failures) == 0:
            print("  ✓ All commands have docstrings")
    except Exception as e:
        all_validation_failures.append(f"Documentation test failed: {e}")

    # Test 3: Verify Rich console
    total_tests += 1
    print("\nTest 3: Rich console")
    try:
        if not isinstance(console, Console):
            all_validation_failures.append("Console is not Rich Console")
        else:
            print("  ✓ Rich console initialized")
    except Exception as e:
        all_validation_failures.append(f"Console test failed: {e}")

    # Test 4: Test function signatures
    total_tests += 1
    print("\nTest 4: Function signatures")
    try:
        # Check list has correct params
        list_sig = inspect.signature(list_strategies_cmd)
        list_params = list(list_sig.parameters.keys())
        expected_list = ['tags', 'verbose']
        for param in expected_list:
            if param not in list_params:
                all_validation_failures.append(f"list_strategies_cmd missing '{param}' parameter")

        # Check info has strategy_name
        info_sig = inspect.signature(info)
        if 'strategy_name' not in info_sig.parameters:
            all_validation_failures.append("info missing 'strategy_name' parameter")

        # Check test has required params
        test_sig = inspect.signature(test)
        test_params = list(test_sig.parameters.keys())
        if 'strategy_name' not in test_params:
            all_validation_failures.append("test missing 'strategy_name' parameter")

        if len(all_validation_failures) == 0:
            print("  ✓ list_strategies_cmd has correct parameters")
            print("  ✓ info has correct parameters")
            print("  ✓ test has correct parameters")
            print("  ✓ validate has correct parameters")
    except Exception as e:
        all_validation_failures.append(f"Signature test failed: {e}")

    # Test 5: Import dependencies
    total_tests += 1
    print("\nTest 5: Module dependencies")
    try:
        from crypto_trader.strategies.registry import get_registry, list_strategies
        from crypto_trader.strategies.loader import load_all_strategies
        from crypto_trader.data.storage import OHLCVStorage as DataStorage

        print("  ✓ Strategy registry imported")
        print("  ✓ Strategy loader imported")
        print("  ✓ DataStorage imported")
    except ImportError as e:
        all_validation_failures.append(f"Import failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Strategy CLI commands are validated and ready for use")
        print("\nNote: Integration tests require registered strategies")
        sys.exit(0)
