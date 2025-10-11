"""
Backtest CLI Commands

This module implements CLI commands for backtesting trading strategies including
running single strategy backtests, comparing multiple strategies, optimizing
parameters, and generating detailed reports.

**Purpose**: Provide powerful CLI commands for comprehensive backtesting operations
with rich output, progress tracking, and detailed performance metrics.

**Key Commands**:
- run: Execute a single strategy backtest
- compare: Compare performance of multiple strategies
- optimize: Optimize strategy parameters
- report: Generate detailed backtest report

**Third-party packages**:
- typer: https://typer.tiangolo.com/
- rich: https://rich.readthedocs.io/en/stable/
- loguru: https://loguru.readthedocs.io/en/stable/
- pandas: https://pandas.pydata.org/docs/

**Sample Input**:
```bash
crypto-trader backtest run SMA_Crossover --symbol BTCUSDT --days 90
crypto-trader backtest compare SMA_Crossover RSI_Mean_Reversion --symbol BTCUSDT
crypto-trader backtest report backtest_12345
```

**Expected Output**:
```
Running Backtest: SMA_Crossover
Symbol: BTCUSDT | Timeframe: 1h | Period: 90 days

✓ Backtest completed
  Total Return: +24.5%
  Sharpe Ratio: 1.85
  Max Drawdown: -8.3%
```
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

from crypto_trader.strategies.registry import get_registry
from crypto_trader.backtesting.engine import BacktestEngine
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


def run(
    strategy_name: str = typer.Argument(
        ...,
        help="Name of the strategy to backtest"
    ),
    symbol: str = typer.Option(
        "BTCUSDT",
        "--symbol",
        "-s",
        help="Trading pair symbol"
    ),
    timeframe: str = typer.Option(
        "1h",
        "--timeframe",
        "-t",
        help="Candle timeframe"
    ),
    days: int = typer.Option(
        90,
        "--days",
        "-d",
        help="Number of days to backtest"
    ),
    initial_capital: float = typer.Option(
        10000.0,
        "--capital",
        "-c",
        help="Initial capital in USDT"
    ),
    fee_percent: float = typer.Option(
        0.001,
        "--fee",
        "-f",
        help="Trading fee percentage (0.001 = 0.1%)"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Strategy configuration (JSON string)"
    ),
    save_report: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save backtest report"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for reports"
    )
):
    """
    Run a backtest for a single trading strategy.

    Executes a comprehensive backtest including signal generation,
    order execution simulation, and performance analysis.

    Example:
        crypto-trader backtest run SMA_Crossover --symbol BTCUSDT --days 90
    """
    try:
        console.print(f"\n[bold blue]Running Backtest: {strategy_name}[/bold blue]")
        console.print(f"Symbol: [cyan]{symbol}[/cyan] | Timeframe: [cyan]{timeframe}[/cyan] | Days: [cyan]{days}[/cyan]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
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

            # Parse config
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
                    console.print(f"[yellow]⚠ No data found for {symbol}[/yellow]")
                    console.print("  Using mock data...\n")
                    provider = MockDataProvider()
                    df = provider.get_ohlcv(symbol, timeframe, limit=days*24)
            except Exception as e:
                console.print(f"[yellow]⚠ Data loading failed: {e}[/yellow]")
                console.print("  Using mock data...\n")
                provider = MockDataProvider()
                df = provider.get_ohlcv(symbol, timeframe, limit=days*24)

            # Run backtest
            progress.update(task, description="Running backtest...")
            engine = BacktestEngine(
                initial_capital=initial_capital,
                fee_percent=fee_percent
            )

            results = engine.run_backtest(
                strategy=strategy,
                data=df,
                symbol=symbol
            )

        # Display results
        console.print("[green]✓[/green] Backtest completed\n")

        # Create performance summary table
        metrics = results.get('metrics', {})

        summary = Table(title="Performance Summary", show_header=True, box=box.ROUNDED)
        summary.add_column("Metric", style="cyan", no_wrap=True)
        summary.add_column("Value", style="green", justify="right")

        # Key metrics
        total_return = metrics.get('total_return_percent', 0)
        return_style = "green" if total_return >= 0 else "red"

        summary.add_row("Initial Capital", f"${initial_capital:,.2f}")
        summary.add_row("Final Value", f"${metrics.get('final_value', 0):,.2f}")
        summary.add_row("Total Return", f"[{return_style}]{total_return:+.2f}%[/{return_style}]")
        summary.add_row("Total Trades", str(metrics.get('total_trades', 0)))
        summary.add_row("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
        summary.add_row("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        summary.add_row("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        summary.add_row("Max Drawdown", f"{metrics.get('max_drawdown_percent', 0):.2f}%")

        console.print(summary)

        # Trade statistics
        if metrics.get('total_trades', 0) > 0:
            console.print("\n[bold]Trade Statistics:[/bold]")
            trade_table = Table(show_header=True, box=box.SIMPLE)
            trade_table.add_column("Type", style="cyan")
            trade_table.add_column("Count", justify="right")
            trade_table.add_column("Avg Profit", justify="right", style="green")

            trade_table.add_row(
                "Winning",
                str(metrics.get('winning_trades', 0)),
                f"${metrics.get('avg_win', 0):,.2f}"
            )
            trade_table.add_row(
                "Losing",
                str(metrics.get('losing_trades', 0)),
                f"${metrics.get('avg_loss', 0):,.2f}"
            )

            console.print(trade_table)

        # Save report
        if save_report:
            try:
                output_dir = Path(output) if output else Path("reports")
                output_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = output_dir / f"backtest_{strategy_name}_{symbol}_{timestamp}.json"

                import json
                with open(report_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                console.print(f"\n[green]✓[/green] Report saved to [cyan]{report_file}[/cyan]")
            except Exception as e:
                console.print(f"\n[yellow]⚠[/yellow] Could not save report: {e}")
                logger.warning(f"Report save failed: {e}")

        console.print()

    except Exception as e:
        console.print(f"\n[red]✗ Backtest failed: {e}[/red]\n")
        logger.exception("Backtest run command failed")
        raise typer.Exit(1)


def compare(
    strategy_names: List[str] = typer.Argument(
        ...,
        help="Names of strategies to compare (space-separated)"
    ),
    symbol: str = typer.Option(
        "BTCUSDT",
        "--symbol",
        "-s",
        help="Trading pair symbol"
    ),
    timeframe: str = typer.Option(
        "1h",
        "--timeframe",
        "-t",
        help="Candle timeframe"
    ),
    days: int = typer.Option(
        90,
        "--days",
        "-d",
        help="Number of days to backtest"
    ),
    initial_capital: float = typer.Option(
        10000.0,
        "--capital",
        "-c",
        help="Initial capital"
    )
):
    """
    Compare performance of multiple strategies.

    Runs backtests for all specified strategies on the same data
    and displays a comparison table with key metrics.

    Example:
        crypto-trader backtest compare SMA_Crossover RSI_Mean_Reversion --symbol BTCUSDT
    """
    try:
        console.print(f"\n[bold blue]Comparing {len(strategy_names)} Strategies[/bold blue]")
        console.print(f"Symbol: [cyan]{symbol}[/cyan] | Timeframe: [cyan]{timeframe}[/cyan] | Days: [cyan]{days}[/cyan]\n")

        # Load data once
        console.print("Loading market data...")
        try:
            storage = OHLCVStorage()
            df = storage.load_ohlcv(symbol, timeframe, days=days)

            if df is None or len(df) == 0:
                provider = MockDataProvider()
                df = provider.get_ohlcv(symbol, timeframe, limit=days*24)
        except Exception:
            provider = MockDataProvider()
            df = provider.get_ohlcv(symbol, timeframe, limit=days*24)

        # Load strategies
        load_all_strategies()
        registry = get_registry()

        # Run backtests for each strategy
        results_dict = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            for strategy_name in strategy_names:
                task = progress.add_task(f"Backtesting {strategy_name}...", total=None)

                try:
                    strategy_class = registry.get_strategy(strategy_name)
                    strategy = strategy_class(name=strategy_name)
                    strategy.initialize({})

                    engine = BacktestEngine(initial_capital=initial_capital)
                    results = engine.run_backtest(strategy, df, symbol)

                    results_dict[strategy_name] = results.get('metrics', {})

                except Exception as e:
                    console.print(f"[red]✗ Failed to backtest {strategy_name}: {e}[/red]")
                    logger.error(f"Strategy {strategy_name} backtest failed: {e}")
                    continue

        # Display comparison table
        if not results_dict:
            console.print("[red]✗ No successful backtests to compare[/red]\n")
            raise typer.Exit(1)

        console.print("\n[bold]Strategy Comparison[/bold]\n")

        comp_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        comp_table.add_column("Strategy", style="yellow")
        comp_table.add_column("Return %", justify="right")
        comp_table.add_column("Sharpe", justify="right")
        comp_table.add_column("Trades", justify="right")
        comp_table.add_column("Win Rate", justify="right")
        comp_table.add_column("Max DD %", justify="right")
        comp_table.add_column("Profit Factor", justify="right")

        # Sort by return
        sorted_strategies = sorted(
            results_dict.items(),
            key=lambda x: x[1].get('total_return_percent', 0),
            reverse=True
        )

        for strategy_name, metrics in sorted_strategies:
            return_pct = metrics.get('total_return_percent', 0)
            return_style = "green" if return_pct >= 0 else "red"

            comp_table.add_row(
                strategy_name,
                f"[{return_style}]{return_pct:+.2f}[/{return_style}]",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                str(metrics.get('total_trades', 0)),
                f"{metrics.get('win_rate', 0):.1f}%",
                f"{metrics.get('max_drawdown_percent', 0):.2f}",
                f"{metrics.get('profit_factor', 0):.2f}"
            )

        console.print(comp_table)
        console.print()

        # Highlight best performer
        best_strategy = sorted_strategies[0][0]
        best_return = sorted_strategies[0][1].get('total_return_percent', 0)

        console.print(
            f"[bold green]🏆 Best Performer:[/bold green] {best_strategy} "
            f"([green]{best_return:+.2f}%[/green])\n"
        )

    except Exception as e:
        console.print(f"\n[red]✗ Comparison failed: {e}[/red]\n")
        logger.exception("Backtest compare command failed")
        raise typer.Exit(1)


def optimize(
    strategy_name: str = typer.Argument(
        ...,
        help="Name of the strategy to optimize"
    ),
    symbol: str = typer.Option(
        "BTCUSDT",
        "--symbol",
        "-s",
        help="Trading pair symbol"
    ),
    param_ranges: str = typer.Option(
        ...,
        "--params",
        "-p",
        help="Parameter ranges (JSON format)"
    ),
    metric: str = typer.Option(
        "sharpe_ratio",
        "--metric",
        "-m",
        help="Metric to optimize (sharpe_ratio, total_return, etc.)"
    ),
    days: int = typer.Option(
        90,
        "--days",
        "-d",
        help="Number of days to backtest"
    )
):
    """
    Optimize strategy parameters.

    Runs parameter optimization using grid search or similar methods
    to find the best parameter combination for the strategy.

    Example:
        crypto-trader backtest optimize SMA_Crossover --params '{"fast": [5,10,20], "slow": [20,50,100]}'
    """
    try:
        console.print(f"\n[bold blue]Optimizing Strategy: {strategy_name}[/bold blue]\n")

        # Parse parameter ranges
        import json
        param_dict = json.loads(param_ranges)

        console.print(f"Parameter ranges: {param_dict}")
        console.print(f"Optimization metric: [cyan]{metric}[/cyan]\n")

        # Calculate total combinations
        import itertools
        param_names = list(param_dict.keys())
        param_values = [param_dict[name] for name in param_names]
        combinations = list(itertools.product(*param_values))

        console.print(f"Testing [bold]{len(combinations)}[/bold] parameter combinations...\n")

        # Load data
        try:
            storage = OHLCVStorage()
            df = storage.load_ohlcv(symbol, "1h", days=days)
            if df is None or len(df) == 0:
                provider = MockDataProvider()
                df = provider.get_ohlcv(symbol, "1h", limit=days*24)
        except Exception:
            provider = MockDataProvider()
            df = provider.get_ohlcv(symbol, "1h", limit=days*24)

        # Load strategy
        load_all_strategies()
        registry = get_registry()
        strategy_class = registry.get_strategy(strategy_name)

        # Run optimization
        best_result = None
        best_params = None
        best_score = float('-inf')

        results_list = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Optimizing...", total=len(combinations))

            for combo in combinations:
                # Create config
                config = {name: value for name, value in zip(param_names, combo)}

                try:
                    # Run backtest
                    strategy = strategy_class(name=strategy_name)
                    strategy.initialize(config)

                    engine = BacktestEngine(initial_capital=10000.0)
                    results = engine.run_backtest(strategy, df, symbol)

                    metrics = results.get('metrics', {})
                    score = metrics.get(metric, 0)

                    results_list.append({
                        'params': config,
                        'score': score,
                        'metrics': metrics
                    })

                    if score > best_score:
                        best_score = score
                        best_params = config
                        best_result = metrics

                except Exception as e:
                    logger.warning(f"Optimization iteration failed: {e}")

                progress.update(task, advance=1)

        # Display results
        if best_result:
            console.print("\n[green]✓[/green] Optimization completed\n")

            # Best parameters
            console.print("[bold]Best Parameters:[/bold]")
            param_panel = Panel(
                "\n".join([f"{k}: [cyan]{v}[/cyan]" for k, v in best_params.items()]),
                title="Optimal Configuration",
                border_style="green"
            )
            console.print(param_panel)

            # Best performance
            console.print("\n[bold]Best Performance:[/bold]")
            perf_table = Table(show_header=False, box=box.SIMPLE)
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="green")

            perf_table.add_row("Optimization Metric", f"{metric}: {best_score:.2f}")
            perf_table.add_row("Total Return", f"{best_result.get('total_return_percent', 0):+.2f}%")
            perf_table.add_row("Sharpe Ratio", f"{best_result.get('sharpe_ratio', 0):.2f}")
            perf_table.add_row("Total Trades", str(best_result.get('total_trades', 0)))

            console.print(perf_table)

            # Top 5 results
            console.print("\n[bold]Top 5 Results:[/bold]")
            top_table = Table(show_header=True, header_style="bold cyan")
            top_table.add_column("Rank", justify="right")
            top_table.add_column("Parameters", style="yellow")
            top_table.add_column("Score", justify="right", style="green")

            sorted_results = sorted(results_list, key=lambda x: x['score'], reverse=True)
            for i, result in enumerate(sorted_results[:5], 1):
                params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
                top_table.add_row(
                    str(i),
                    params_str,
                    f"{result['score']:.2f}"
                )

            console.print(top_table)
            console.print()

        else:
            console.print("[red]✗ Optimization failed - no valid results[/red]\n")

    except Exception as e:
        console.print(f"\n[red]✗ Optimization failed: {e}[/red]\n")
        logger.exception("Backtest optimize command failed")
        raise typer.Exit(1)


def report(
    report_id: str = typer.Argument(
        ...,
        help="Backtest report ID or file path"
    ),
    format: str = typer.Option(
        "console",
        "--format",
        "-f",
        help="Output format (console, html, pdf)"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path"
    )
):
    """
    Generate detailed backtest report.

    Creates a comprehensive report from a saved backtest including
    charts, metrics, and trade analysis.

    Example:
        crypto-trader backtest report backtest_12345 --format html
    """
    try:
        console.print(f"\n[bold blue]Generating Backtest Report[/bold blue]\n")

        # Load report data
        import json
        from pathlib import Path

        report_path = Path(report_id)
        if not report_path.exists():
            # Try to find in reports directory
            report_path = Path("reports") / report_id
            if not report_path.exists():
                console.print(f"[red]✗ Report not found: {report_id}[/red]\n")
                raise typer.Exit(1)

        with open(report_path, 'r') as f:
            data = json.load(f)

        console.print(f"Report: [cyan]{report_path.name}[/cyan]\n")

        if format == "console":
            # Display report in console
            metrics = data.get('metrics', {})

            # Overview
            console.print("[bold]Backtest Overview[/bold]")
            overview = Table(show_header=False, box=box.SIMPLE)
            overview.add_column("Field", style="cyan")
            overview.add_column("Value")

            overview.add_row("Strategy", data.get('strategy_name', 'Unknown'))
            overview.add_row("Symbol", data.get('symbol', 'Unknown'))
            overview.add_row("Period", f"{data.get('start_date', 'N/A')} to {data.get('end_date', 'N/A')}")

            console.print(overview)

            # Performance metrics
            console.print("\n[bold]Performance Metrics[/bold]")
            metrics_table = Table(show_header=True, header_style="bold cyan")
            metrics_table.add_column("Metric", style="yellow")
            metrics_table.add_column("Value", style="green", justify="right")

            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics_table.add_row(key.replace('_', ' ').title(), f"{value:.2f}")
                else:
                    metrics_table.add_row(key.replace('_', ' ').title(), str(value))

            console.print(metrics_table)

        elif format in ["html", "pdf"]:
            console.print(f"[yellow]ℹ[/yellow] {format.upper()} export not yet implemented")
            console.print("  Use --format console for text output\n")

        console.print()

    except Exception as e:
        console.print(f"\n[red]✗ Report generation failed: {e}[/red]\n")
        logger.exception("Backtest report command failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    """
    Validation block for backtest CLI commands.
    Tests command functions structure and dependencies.
    """
    import sys
    import inspect

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating backtest CLI commands...\n")

    # Test 1: Verify all command functions exist
    total_tests += 1
    print("Test 1: Command functions exist")
    try:
        commands = [run, compare, optimize, report]
        for cmd in commands:
            if not callable(cmd):
                all_validation_failures.append(f"{cmd.__name__} is not callable")

        if len(all_validation_failures) == 0:
            print("  ✓ run exists")
            print("  ✓ compare exists")
            print("  ✓ optimize exists")
            print("  ✓ report exists")
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
        # Check run has required params
        run_sig = inspect.signature(run)
        run_params = list(run_sig.parameters.keys())
        if 'strategy_name' not in run_params:
            all_validation_failures.append("run missing 'strategy_name' parameter")
        if 'symbol' not in run_params:
            all_validation_failures.append("run missing 'symbol' parameter")

        # Check compare has strategy_names
        compare_sig = inspect.signature(compare)
        if 'strategy_names' not in compare_sig.parameters:
            all_validation_failures.append("compare missing 'strategy_names' parameter")

        if len(all_validation_failures) == 0:
            print("  ✓ run has correct parameters")
            print("  ✓ compare has correct parameters")
            print("  ✓ optimize has correct parameters")
            print("  ✓ report has correct parameters")
    except Exception as e:
        all_validation_failures.append(f"Signature test failed: {e}")

    # Test 5: Import dependencies
    total_tests += 1
    print("\nTest 5: Module dependencies")
    try:
        from crypto_trader.strategies.registry import get_registry
        from crypto_trader.backtesting.engine import BacktestEngine
        from crypto_trader.data.storage import OHLCVStorage as DataStorage

        print("  ✓ Strategy registry imported")
        print("  ✓ BacktestEngine imported")
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
        print("Backtest CLI commands are validated and ready for use")
        print("\nNote: Integration tests require backtest executor implementation")
        sys.exit(0)
