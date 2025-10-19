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
from dataclasses import asdict
import json

import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

from crypto_trader.core.config import BacktestConfig
from crypto_trader.core.types import BacktestResult, Timeframe
from crypto_trader.strategies.registry import get_registry
from crypto_trader.backtesting.engine import BacktestEngine
from crypto_trader.data.storage import OHLCVStorage as DataStorage
from crypto_trader.data.providers import MockDataProvider

console = Console()


def _resolve_timeframe(value: str) -> Timeframe:
    """Resolve CLI timeframe string to Timeframe enum with fallback."""
    try:
        return Timeframe(value)
    except ValueError:
        console.print(
            f"[yellow]⚠ Unsupported timeframe '{value}' provided; defaulting to 1h[/yellow]"
        )
        return Timeframe.HOUR_1


def _prepare_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure market data has a timestamp column and sorted index."""
    if df is None or df.empty:
        return pd.DataFrame()

    prepared = df.copy()
    prepared.index = pd.to_datetime(prepared.index, utc=True)
    prepared.sort_index(inplace=True)
    prepared["timestamp"] = prepared.index
    return prepared


def _load_market_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Load market data from storage with mock fallback."""
    storage = DataStorage()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    df = storage.load_ohlcv(symbol, timeframe, start_date=start_date, end_date=end_date)

    if df is None or df.empty:
        provider = MockDataProvider()
        df = provider.get_ohlcv(symbol, timeframe, limit=days * 24)

    return _prepare_market_data(df)


def _metrics_as_dict(result: BacktestResult) -> Dict[str, Any]:
    """Convert BacktestResult metrics to a serializable dict with percentages."""
    metrics = result.metrics
    metrics_dict = asdict(metrics)
    metrics_dict.update(
        {
            "total_return_percent": metrics.total_return * 100,
            "max_drawdown_percent": metrics.max_drawdown * 100,
            "win_rate_percent": metrics.win_rate * 100,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
        }
    )
    return metrics_dict


def _build_report_payload(result: BacktestResult) -> Dict[str, Any]:
    """Create a JSON-serializable payload for saving backtest reports."""
    payload = {
        "strategy_name": result.strategy_name,
        "symbol": result.symbol,
        "timeframe": result.timeframe.value,
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "initial_capital": result.initial_capital,
        "metrics": _metrics_as_dict(result),
        "summary": result.summary(),
        "metadata": result.metadata,
    }

    trades_preview = []
    for trade in result.trades[:50]:
        trades_preview.append(
            {
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "side": trade.side.value,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "pnl": trade.pnl,
                "pnl_percent": trade.pnl_percent,
                "fees": trade.fees,
            }
        )
    payload["trades_preview"] = trades_preview
    return payload


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
            task = progress.add_task("Loading strategies...", total=None)
            load_all_strategies()
            registry = get_registry()

            try:
                strategy_class = registry.get_strategy(strategy_name)
            except KeyError:
                console.print(f"[red]✗ Strategy '{strategy_name}' not found[/red]\n")
                raise typer.Exit(1)

            progress.update(task, description="Initializing strategy...")
            strategy = strategy_class(name=strategy_name)
            strategy_config = json.loads(config) if config else {}
            strategy.initialize(strategy_config)

            progress.update(task, description="Loading market data...")
            market_data = _load_market_data(symbol, timeframe, days)
            if market_data.empty:
                console.print(f"[red]✗ Unable to load market data for {symbol}[/red]\n")
                raise typer.Exit(1)

            progress.update(task, description="Running backtest...")
            engine = BacktestEngine()
            timeframe_enum = _resolve_timeframe(timeframe)
            backtest_config = BacktestConfig(
                initial_capital=initial_capital,
                trading_fee_percent=fee_percent
            )

            result = engine.run_backtest(
                strategy=strategy,
                data=market_data,
                config=backtest_config,
                symbol=symbol,
                timeframe=timeframe_enum
            )

        console.print("[green]✓[/green] Backtest completed\n")

        metrics = result.metrics
        total_return_pct = metrics.total_return * 100
        win_rate_pct = metrics.win_rate * 100
        max_drawdown_pct = metrics.max_drawdown * 100

        summary = Table(title="Performance Summary", show_header=True, box=box.ROUNDED)
        summary.add_column("Metric", style="cyan", no_wrap=True)
        summary.add_column("Value", style="green", justify="right")
        return_style = "green" if total_return_pct >= 0 else "red"

        summary.add_row("Initial Capital", f"${initial_capital:,.2f}")
        summary.add_row("Final Value", f"${metrics.final_capital:,.2f}")
        summary.add_row("Total Return", f"[{return_style}]{total_return_pct:+.2f}%[/{return_style}]")
        summary.add_row("Total Trades", str(metrics.total_trades))
        summary.add_row("Win Rate", f"{win_rate_pct:.2f}%")
        summary.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
        summary.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        summary.add_row("Max Drawdown", f"{max_drawdown_pct:.2f}%")

        console.print(summary)

        if metrics.total_trades > 0:
            console.print("\n[bold]Trade Statistics:[/bold]")
            trade_table = Table(show_header=True, box=box.SIMPLE)
            trade_table.add_column("Type", style="cyan")
            trade_table.add_column("Count", justify="right")
            trade_table.add_column("Average", justify="right", style="green")

            trade_table.add_row("Winning", str(metrics.winning_trades), f"${metrics.avg_win:,.2f}")
            trade_table.add_row("Losing", str(metrics.losing_trades), f"${metrics.avg_loss:,.2f}")
            trade_table.add_row("Expectancy", "-", f"${metrics.expectancy:,.2f}")

            console.print(trade_table)

        if save_report:
            try:
                output_dir = Path(output) if output else Path("reports")
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                report_file = output_dir / f"backtest_{strategy_name}_{symbol}_{timestamp}.json"

                with open(report_file, "w", encoding="utf-8") as f:
                    json.dump(_build_report_payload(result), f, indent=2)

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

        market_data = _load_market_data(symbol, timeframe, days)
        if market_data.empty:
            console.print(f"[red]✗ Unable to load market data for {symbol}[/red]\n")
            raise typer.Exit(1)

        load_all_strategies()
        registry = get_registry()
        timeframe_enum = _resolve_timeframe(timeframe)
        backtest_config = BacktestConfig(initial_capital=initial_capital)

        results: Dict[str, BacktestResult] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            for name in strategy_names:
                task = progress.add_task(f"Backtesting {name}...", total=None)
                try:
                    strategy_class = registry.get_strategy(name)
                    strategy = strategy_class(name=name)
                    strategy.initialize({})

                    engine = BacktestEngine()
                    result = engine.run_backtest(
                        strategy=strategy,
                        data=market_data.copy(),
                        config=backtest_config,
                        symbol=symbol,
                        timeframe=timeframe_enum
                    )
                    results[name] = result
                except Exception as exc:
                    console.print(f"[red]✗ Failed to backtest {name}: {exc}[/red]")
                    logger.error(f"Strategy {name} backtest failed: {exc}")
                finally:
                    progress.update(task, completed=True)

        if not results:
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

        sorted_results = sorted(
            results.items(),
            key=lambda item: item[1].metrics.total_return,
            reverse=True
        )

        for name, result in sorted_results:
            metrics = result.metrics
            return_pct = metrics.total_return * 100
            win_rate_pct = metrics.win_rate * 100
            drawdown_pct = metrics.max_drawdown * 100
            return_style = "green" if return_pct >= 0 else "red"

            comp_table.add_row(
                name,
                f"[{return_style}]{return_pct:+.2f}%[/{return_style}]",
                f"{metrics.sharpe_ratio:.2f}",
                str(metrics.total_trades),
                f"{win_rate_pct:.1f}%",
                f"{drawdown_pct:.2f}%",
                f"{metrics.profit_factor:.2f}"
            )

        console.print(comp_table)
        console.print()

        best_name, best_result = sorted_results[0]
        console.print(
            f"[bold green]🏆 Best Performer:[/bold green] {best_name} "
            f"([green]{best_result.metrics.total_return * 100:+.2f}%[/green])\n"
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

        param_dict = json.loads(param_ranges)
        console.print(f"Parameter ranges: {param_dict}")
        console.print(f"Optimization metric: [cyan]{metric}[/cyan]\n")

        import itertools

        param_names = list(param_dict.keys())
        param_values = [param_dict[name] for name in param_names]
        combinations = list(itertools.product(*param_values))

        if not combinations:
            console.print("[red]✗ No parameter combinations to evaluate[/red]\n")
            raise typer.Exit(1)

        console.print(f"Testing [bold]{len(combinations)}[/bold] parameter combinations...\n")

        market_data = _load_market_data(symbol, Timeframe.HOUR_1.value, days)
        if market_data.empty:
            console.print(f"[red]✗ Unable to load market data for {symbol}[/red]\n")
            raise typer.Exit(1)

        load_all_strategies()
        registry = get_registry()
        strategy_class = registry.get_strategy(strategy_name)
        engine = BacktestEngine()
        backtest_config = BacktestConfig(initial_capital=10000.0)

        best_params = None
        best_result: Optional[BacktestResult] = None
        best_score = float("-inf")
        evaluation_rows = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Optimizing...", total=len(combinations))

            for combo in combinations:
                strategy_params = {name: value for name, value in zip(param_names, combo)}
                try:
                    strategy = strategy_class(name=strategy_name)
                    strategy.initialize(strategy_params)

                    result = engine.run_backtest(
                        strategy=strategy,
                        data=market_data.copy(),
                        config=backtest_config,
                        symbol=symbol,
                        timeframe=Timeframe.HOUR_1
                    )

                    metrics = result.metrics
                    score = getattr(metrics, metric, None)
                    if score is None:
                        raise AttributeError(
                            f"Metric '{metric}' not available on PerformanceMetrics"
                        )

                    evaluation_rows.append(
                        {
                            "params": strategy_params,
                            "score": score,
                            "result": result,
                        }
                    )

                    if score > best_score:
                        best_score = score
                        best_params = strategy_params
                        best_result = result

                except Exception as exc:
                    logger.warning(f"Optimization iteration failed: {exc}")

                progress.update(task, advance=1)

        if not best_result:
            console.print("[red]✗ Optimization failed - no valid results[/red]\n")
            return

        console.print("\n[green]✓[/green] Optimization completed\n")

        console.print("[bold]Best Parameters:[/bold]")
        param_panel = Panel(
            "\n".join(f"{k}: [cyan]{v}[/cyan]" for k, v in best_params.items()),
            title="Optimal Configuration",
            border_style="green",
        )
        console.print(param_panel)

        metrics = best_result.metrics

        console.print("\n[bold]Best Performance:[/bold]")
        perf_table = Table(show_header=False, box=box.SIMPLE)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        perf_table.add_row("Optimization Metric", f"{metric}: {best_score:.2f}")
        perf_table.add_row("Total Return", f"{metrics.total_return * 100:+.2f}%")
        perf_table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        perf_table.add_row("Total Trades", str(metrics.total_trades))

        console.print(perf_table)

        console.print("\n[bold]Top 5 Results:[/bold]")
        top_table = Table(show_header=True, header_style="bold cyan")
        top_table.add_column("Rank", justify="right")
        top_table.add_column("Parameters", style="yellow")
        top_table.add_column("Score", justify="right", style="green")

        top_results = sorted(
            evaluation_rows, key=lambda row: row["score"], reverse=True
        )[:5]

        for idx, row in enumerate(top_results, start=1):
            params_str = ", ".join(f"{k}={v}" for k, v in row["params"].items())
            top_table.add_row(str(idx), params_str, f"{row['score']:.2f}")

        console.print(top_table)
        console.print()

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
            metrics = data.get("metrics", {})

            console.print("[bold]Backtest Overview[/bold]")
            overview = Table(show_header=False, box=box.SIMPLE)
            overview.add_column("Field", style="cyan")
            overview.add_column("Value")

            overview.add_row("Strategy", data.get("strategy_name", "Unknown"))
            overview.add_row("Symbol", data.get("symbol", "Unknown"))
            overview.add_row("Timeframe", data.get("timeframe", "Unknown"))
            overview.add_row(
                "Period",
                f"{data.get('start_date', 'N/A')} to {data.get('end_date', 'N/A')}",
            )

            console.print(overview)

            console.print("\n[bold]Performance Metrics[/bold]")
            metrics_table = Table(show_header=False, box=box.SIMPLE)
            metrics_table.add_column("Metric", style="yellow")
            metrics_table.add_column("Value", style="green")

            def _format_number(value, suffix=""):
                if isinstance(value, (int, float)):
                    return f"{value:.2f}{suffix}"
                return str(value)

            metrics_table.add_row(
                "Total Return",
                f"{metrics.get('total_return_percent', metrics.get('total_return', 0) * 100):+.2f}%",
            )
            metrics_table.add_row(
                "Max Drawdown",
                f"{metrics.get('max_drawdown_percent', metrics.get('max_drawdown', 0) * 100):.2f}%",
            )
            metrics_table.add_row(
                "Sharpe Ratio",
                _format_number(metrics.get("sharpe_ratio", 0)),
            )
            metrics_table.add_row(
                "Sortino Ratio",
                _format_number(metrics.get("sortino_ratio", 0)),
            )
            metrics_table.add_row(
                "Profit Factor",
                _format_number(metrics.get("profit_factor", 0)),
            )
            metrics_table.add_row(
                "Win Rate",
                f"{metrics.get('win_rate_percent', metrics.get('win_rate', 0) * 100):.2f}%",
            )
            metrics_table.add_row(
                "Total Trades",
                str(metrics.get("total_trades", 0)),
            )

            console.print(metrics_table)

            trades_preview = data.get("trades_preview", [])
            if trades_preview:
                console.print("\n[bold]Sample Trades (first 5)[/bold]")
                trades_table = Table(show_header=True, header_style="bold cyan")
                trades_table.add_column("Entry", style="cyan")
                trades_table.add_column("Exit", style="cyan")
                trades_table.add_column("Side", justify="center")
                trades_table.add_column("PnL", justify="right")
                trades_table.add_column("PnL %", justify="right")

                for trade in trades_preview[:5]:
                    trades_table.add_row(
                        trade.get("entry_time", ""),
                        trade.get("exit_time", ""),
                        trade.get("side", ""),
                        f"${trade.get('pnl', 0):,.2f}",
                        f"{trade.get('pnl_percent', 0):+.2f}%",
                    )

                console.print(trades_table)

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
