"""
CLI Commands for Crypto Trader Master Analysis

This module provides the command-line interface for running comprehensive
strategy analysis and ranking.

**Purpose**: Clean CLI layer for master strategy analysis

**Key Commands**:
- analyze: Run comprehensive strategy testing and ranking

**Third-party packages**:
- typer: https://typer.tiangolo.com/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```bash
python master.py --symbol BTC/USDT
python master.py --symbol ETH/USDT --quick
python master.py --workers 8 --horizons 30 90 180 365
```

**Expected Output**:
Comprehensive analysis reports with strategy rankings.

Extracted from master.py during Phase 4 refactoring.
"""

from typing import List, Optional
import sys

import typer
from loguru import logger

from crypto_trader.orchestration import MasterStrategyAnalyzer


app = typer.Typer(
    name="crypto-trader",
    help="Master strategy analysis and ranking system",
    no_args_is_help=True,
)


@app.command()
def analyze(
    symbol: str = typer.Option(
        "BTC/USDT",
        "--symbol", "-s",
        help="Trading pair symbol (e.g., BTC/USDT, ETH/USDT)"
    ),
    timeframe: str = typer.Option(
        "1h",
        "--timeframe", "-t",
        help="Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)"
    ),
    horizons: Optional[List[int]] = typer.Option(
        None,
        "--horizons", "-h",
        help="Custom time horizons in days (e.g., 30 90 180 365)"
    ),
    workers: int = typer.Option(
        4,
        "--workers", "-w",
        help="Number of parallel workers (1-32)"
    ),
    quick: bool = typer.Option(
        False,
        "--quick", "-q",
        help="Quick mode (fewer horizons for faster testing)"
    ),
    multi_pair: bool = typer.Option(
        False,
        "--multi-pair", "-m",
        help="Test multi-pair strategies (Portfolio, StatArb, etc.)"
    ),
    output_dir: str = typer.Option(
        "master_results",
        "--output", "-o",
        help="Output directory base name"
    ),
):
    """
    Run comprehensive master strategy analysis.

    Tests all strategies across multiple time horizons, ranks them by
    composite score, and generates detailed comparison reports.

    Examples:

        # Standard analysis
        python master.py --symbol BTC/USDT

        # Quick analysis with fewer horizons
        python master.py --symbol ETH/USDT --quick

        # Custom horizons and workers
        python master.py --workers 8 --horizons 30 90 180 365

        # Test multi-pair portfolio strategies
        python master.py --multi-pair --quick
    """
    # Input validation
    valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
    if timeframe not in valid_timeframes:
        logger.error(
            f"Invalid timeframe: {timeframe}. "
            f"Must be one of: {valid_timeframes}"
        )
        raise typer.Exit(1)

    if workers < 1 or workers > 32:
        logger.error(
            f"Invalid workers: {workers}. "
            f"Must be between 1 and 32."
        )
        raise typer.Exit(1)

    if horizons:
        for h in horizons:
            if h < 7 or h > 3650:
                logger.error(
                    f"Invalid horizon: {h} days. "
                    f"Must be between 7 and 3650."
                )
                raise typer.Exit(1)

    if not symbol or '/' not in symbol:
        logger.error(
            f"Invalid symbol: {symbol}. "
            f"Must be in format BASE/QUOTE (e.g., BTC/USDT)"
        )
        raise typer.Exit(1)

    # Create and run analyzer
    analyzer = MasterStrategyAnalyzer(
        symbol=symbol,
        timeframe=timeframe,
        horizons=horizons,
        workers=workers,
        quick_mode=quick,
        multi_pair=multi_pair,
        output_dir=output_dir,
    )

    analyzer.run()


if __name__ == "__main__":
    """
    Validation block for CLI commands.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    # Test 1: Verify command is importable
    total_tests += 1
    print("Test 1: Verify analyze command is importable")
    try:
        if analyze is None:
            all_validation_failures.append("analyze command not defined")
        else:
            print(f"  ✓ analyze command: {analyze.__name__}")
    except Exception as e:
        all_validation_failures.append(f"Command import failed: {e}")

    # Test 2: Verify Typer app is configured
    total_tests += 1
    print("\nTest 2: Verify Typer app configuration")
    try:
        if app is None:
            all_validation_failures.append("Typer app not defined")
        else:
            print(f"  ✓ Typer app name: {app.info.name}")
            print(f"  ✓ Typer app help: {app.info.help[:50]}...")
    except Exception as e:
        all_validation_failures.append(f"Typer app configuration failed: {e}")

    # Test 3: Verify command parameters
    total_tests += 1
    print("\nTest 3: Verify analyze command parameters")
    try:
        import inspect
        sig = inspect.signature(analyze)
        params = list(sig.parameters.keys())

        expected_params = ['symbol', 'timeframe', 'horizons', 'workers', 'quick', 'multi_pair', 'output_dir']
        if params != expected_params:
            all_validation_failures.append(
                f"Parameter mismatch: expected {expected_params}, got {params}"
            )
        else:
            print(f"  ✓ All {len(params)} parameters present: {params}")
    except Exception as e:
        all_validation_failures.append(f"Parameter verification failed: {e}")

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
        print("\nNOTE: CLI commands extracted from master.py during Phase 4 refactoring")
        sys.exit(0)
