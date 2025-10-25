#!/usr/bin/env python3
"""
Master Windowed Analysis - Train/Test Split with Scientific Rigor

This script implements proper train/test methodology for strategy evaluation:
- Training set: Data before cutoff (for parameter tuning)
- Test set: Recent data (for final evaluation)
- Non-overlapping windows within each set
- Comprehensive statistics: mean, median, std, percentiles
- Result caching to avoid recomputation
- Distribution visualization

Usage:
    python master_windowed.py --symbol BTC/USDT --timeframe 1h
    python master_windowed.py --quick  # Fast mode with fewer horizons

**Purpose**: Scientific strategy evaluation with train/test split

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- loguru: https://loguru.readthedocs.io/en/stable/

**Expected Output**:
HTML report with train/test results, distributions, and generalization analysis.

**Methodology**:
Proper ML approach: train on historical data, test on recent unseen data.
No lookahead bias, temporal separation enforced.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.features.factory import augment_with_features, DEFAULT_JOIN_CONFIG
from crypto_trader.strategies import get_registry
from crypto_trader.orchestration.window_manager import TrainTestSplitter, WindowSpec
from crypto_trader.analysis.aggregator import ResultsAggregator, WindowedMetrics
from crypto_trader.analysis.windowed_cache import WindowedResultsCache
from crypto_trader.execution.workers import run_backtest_worker

app = typer.Typer()


def fetch_full_dataset(
    symbol: str,
    timeframe: str,
    max_days: int = 1095  # 3 years default
) -> pd.DataFrame:
    """
    Fetch full historical dataset.

    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        max_days: Maximum days to fetch

    Returns:
        DataFrame with OHLCV + features
    """
    logger.info(f"📡 Fetching {max_days} days of {symbol} data at {timeframe} timeframe")

    from datetime import timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=max_days)

    fetcher = BinanceDataFetcher()
    data = fetcher.get_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )

    logger.info(f"✅ Fetched {len(data):,} candles "
               f"({data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})")

    # Augment with features
    logger.info("🔧 Computing technical indicators...")
    data = augment_with_features(data, symbol, timeframe, config=DEFAULT_JOIN_CONFIG)

    logger.info(f"✅ Dataset ready: {len(data):,} rows × {len(data.columns)} columns")

    return data


def run_windowed_backtest(
    strategy_name: str,
    window: WindowSpec,
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cache: WindowedResultsCache
) -> Optional[Dict[str, Any]]:
    """
    Run backtest for a single window, checking cache first.

    Args:
        strategy_name: Strategy to test
        window: Window specification
        data: Full dataset
        symbol: Trading symbol
        timeframe: Timeframe string
        cache: Results cache

    Returns:
        Result dictionary or None if error
    """
    # Check cache first
    cached = cache.get_result(
        strategy=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        horizon=window.horizon_name,
        window_id=window.window_id,
        dataset_type=window.dataset_type,
        start_date=window.start_date.isoformat(),
        end_date=window.end_date.isoformat()
    )

    if cached is not None:
        return cached

    # Not cached - run backtest
    logger.debug(f"Running {strategy_name} on {window.horizon_name} "
                f"window {window.window_id} ({window.dataset_type})")

    # Slice data for this window
    window_data = data.iloc[window.start_idx:window.end_idx].copy()

    # Convert to dict for worker (required for multiprocessing)
    # Drop timestamp column if it exists to avoid conflicts with reset_index
    if 'timestamp' in window_data.columns:
        window_data = window_data.drop(columns=['timestamp'])

    data_dict = window_data.reset_index().to_dict('list')

    # Run backtest via worker
    result = run_backtest_worker(
        strategy_name=strategy_name,
        data_dict=data_dict,
        horizon_name=window.horizon_name,
        horizon_days=window.horizon_days,
        symbol=symbol,
        timeframe=timeframe,
        default_params={}
    )

    if result and 'error' not in result:
        # Store in cache
        cache.store_result(
            strategy=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            horizon=window.horizon_name,
            window_id=window.window_id,
            dataset_type=window.dataset_type,
            start_date=window.start_date.isoformat(),
            end_date=window.end_date.isoformat(),
            result=result
        )

    return result


@app.command()
def analyze(
    symbol: str = typer.Option("BTC/USDT", "--symbol", "-s", help="Trading pair"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    test_years: float = typer.Option(2.0, "--test-years", help="Years reserved for test set"),
    horizons: Optional[List[int]] = typer.Option(None, "--horizons", "-h", help="Custom horizons in days"),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel workers"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick mode (fewer horizons)"),
    max_days: int = typer.Option(1095, "--max-days", help="Maximum days of data to fetch"),
    output_dir: str = typer.Option("windowed_results", "--output", "-o", help="Output directory")
):
    """
    Run windowed train/test analysis on all strategies.

    Implements scientific train/test split:
    - Training set: Historical data before cutoff
    - Test set: Recent data (last N years)
    - Non-overlapping windows in each set
    - Comprehensive statistics across windows
    """
    logger.info("=" * 80)
    logger.info("🧪 WINDOWED TRAIN/TEST ANALYSIS")
    logger.info("=" * 80)

    start_time = time.perf_counter()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"{output_dir}_{timestamp}")
    output_path.mkdir(exist_ok=True, parents=True)

    # Configure logging
    log_file = output_path / "windowed_analysis.log"
    logger.add(log_file, level="DEBUG", rotation="100 MB")

    # Define horizons
    if horizons:
        horizon_list = horizons
    elif quick:
        horizon_list = [30, 90, 180]
    else:
        horizon_list = [30, 90, 180, 365]

    logger.info(f"📊 Configuration:")
    logger.info(f"   Symbol: {symbol}")
    logger.info(f"   Timeframe: {timeframe}")
    logger.info(f"   Test Set: {test_years} years")
    logger.info(f"   Horizons: {horizon_list} days")
    logger.info(f"   Workers: {workers}")
    logger.info(f"   Max Data: {max_days} days")

    # Validate configuration BEFORE doing any work
    min_days_required = int(test_years * 365 * 1.5)  # Need at least 1.5x test set for train data
    if max_days < min_days_required:
        logger.error(f"❌ CONFIGURATION ERROR")
        logger.error(f"   max_days ({max_days}) is insufficient for test_years ({test_years})")
        logger.error(f"   Minimum required: {min_days_required} days")
        logger.error(f"   Solution: Use --max-days {min_days_required} or --test-years {max_days / 365 / 1.5:.1f}")
        raise ValueError(
            f"Insufficient data: need at least {min_days_required} days "
            f"for {test_years}-year test set with training data"
        )

    # Fetch data
    data = fetch_full_dataset(symbol, timeframe, max_days)

    # Validate we got data
    if len(data) == 0:
        logger.error(f"❌ No data fetched for {symbol}")
        raise ValueError(f"Failed to fetch data for {symbol}")

    logger.info(f"✅ Fetched {len(data):,} candles")

    # Initialize components
    runtime_date = datetime.now()
    splitter = TrainTestSplitter(runtime_date=runtime_date, test_set_years=test_years)
    aggregator = ResultsAggregator(recent_weight=0.6)
    cache = WindowedResultsCache()

    # Discover strategies
    import crypto_trader.strategies.library  # noqa: F401
    registry = get_registry()
    strategy_names = [name for name in registry.get_strategy_names()
                     if "Portfolio" not in name and "Statistical" not in name]

    logger.info(f"🎯 Testing {len(strategy_names)} strategies")

    # Generate windows for all horizons
    all_windows: Dict[str, Dict[str, List[WindowSpec]]] = {}

    for horizon_days in horizon_list:
        horizon_name = f"{horizon_days}d"
        train_wins, test_wins = splitter.generate_windows(
            data, horizon_days, horizon_name, timeframe
        )
        all_windows[horizon_name] = {
            'train': train_wins,
            'test': test_wins
        }

    # Calculate total jobs
    total_windows = sum(
        len(all_windows[h]['train']) + len(all_windows[h]['test'])
        for h in all_windows
    )
    total_jobs = len(strategy_names) * total_windows

    logger.info(f"📈 Total backtest jobs: {total_jobs:,}")
    logger.info(f"   ({len(strategy_names)} strategies × {total_windows} windows)")

    # Run all backtests in parallel
    logger.info("\n" + "=" * 80)
    logger.info("EXECUTING WINDOWED BACKTESTS")
    logger.info("=" * 80)

    all_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []

        # Submit all jobs
        for strategy_name in strategy_names:
            all_results[strategy_name] = {}

            for horizon_name, windows_dict in all_windows.items():
                all_results[strategy_name][horizon_name] = {
                    'train': [],
                    'test': []
                }

                for dataset_type in ['train', 'test']:
                    windows = windows_dict[dataset_type]

                    for window in windows:
                        future = executor.submit(
                            run_windowed_backtest,
                            strategy_name,
                            window,
                            data,
                            symbol,
                            timeframe,
                            cache
                        )
                        futures.append((future, strategy_name, horizon_name, dataset_type))

        # Collect results with progress bar
        successful = 0
        failed = 0

        with tqdm(total=len(futures), desc="Running backtests") as pbar:
            for future, strategy_name, horizon_name, dataset_type in futures:
                try:
                    result = future.result(timeout=300)
                    if result and 'error' not in result:
                        all_results[strategy_name][horizon_name][dataset_type].append(result)
                        successful += 1
                    elif result and 'error' in result:
                        logger.warning(f"❌ {strategy_name}/{horizon_name}/{dataset_type}: {result['error']}")
                        failed += 1
                    else:
                        logger.warning(f"❌ {strategy_name}/{horizon_name}/{dataset_type}: No result returned")
                        failed += 1
                except TimeoutError:
                    logger.error(f"⏱️  {strategy_name}/{horizon_name}/{dataset_type}: Timeout after 300s")
                    failed += 1
                except Exception as e:
                    import traceback
                    logger.error(f"💥 {strategy_name}/{horizon_name}/{dataset_type}: {type(e).__name__}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    failed += 1

                pbar.update(1)

        logger.info(f"\n📊 Backtest Results: {successful} successful, {failed} failed out of {len(futures)} total")

    # Save cache
    cache.save()

    # Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATING STATISTICS")
    logger.info("=" * 80)

    aggregated: Dict[str, Dict[str, WindowedMetrics]] = {}

    for strategy_name in strategy_names:
        aggregated[strategy_name] = {}

        for horizon_name in all_windows.keys():
            for dataset_type in ['train', 'test']:
                results = all_results[strategy_name][horizon_name][dataset_type]

                metrics = aggregator.aggregate_windows(
                    results=results,
                    strategy_name=strategy_name,
                    horizon_name=horizon_name,
                    dataset_type=dataset_type
                )

                key = f"{horizon_name}_{dataset_type}"
                aggregated[strategy_name][key] = metrics

                # Log summary
                if metrics.num_windows > 0:
                    logger.info(
                        f"{strategy_name} ({horizon_name}, {dataset_type}): "
                        f"{metrics.num_windows} windows, "
                        f"Return={metrics.mean_return:.2%}±{metrics.std_return:.2%}, "
                        f"Sharpe={metrics.mean_sharpe:.2f}±{metrics.std_sharpe:.2f}"
                    )

    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING REPORT")
    logger.info("=" * 80)

    # Save aggregated results as CSV
    results_file = output_path / "aggregated_results.csv"

    records = []
    for strategy_name, horizon_metrics in aggregated.items():
        for key, metrics in horizon_metrics.items():
            record = {'strategy': strategy_name, **metrics.to_dict()}
            records.append(record)

    results_df = pd.DataFrame(records)
    results_df.to_csv(results_file, index=False)

    logger.info(f"✅ Saved aggregated results to {results_file}")

    # Generate simple text report
    report_file = output_path / "REPORT.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("WINDOWED TRAIN/TEST ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Symbol: {symbol}\n")
        f.write(f"Timeframe: {timeframe}\n")
        f.write(f"Test Set: {test_years} years\n")
        f.write(f"Horizons: {horizon_list}\n")
        f.write(f"Strategies Tested: {len(strategy_names)}\n")
        f.write(f"Total Windows: {total_windows}\n\n")

        f.write("=" * 80 + "\n")
        f.write("TRAIN/TEST SPLIT METHODOLOGY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Set: All data before {splitter.cutoff_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Test Set: {splitter.cutoff_date.strftime('%Y-%m-%d')} to {runtime_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Non-overlapping windows within each set\n")
        f.write(f"Statistics: mean, median, std dev, percentiles (25th, 75th)\n\n")

        f.write("=" * 80 + "\n")
        f.write("TOP STRATEGIES BY TEST SET PERFORMANCE\n")
        f.write("=" * 80 + "\n\n")

        # Rank by test set Sharpe (average across horizons)
        rankings = []
        for strategy_name, horizon_metrics in aggregated.items():
            test_sharpes = [
                metrics.mean_sharpe
                for key, metrics in horizon_metrics.items()
                if 'test' in key and metrics.num_windows > 0
            ]

            if test_sharpes:
                avg_test_sharpe = sum(test_sharpes) / len(test_sharpes)
                rankings.append((strategy_name, avg_test_sharpe))

        rankings.sort(key=lambda x: x[1], reverse=True)

        for i, (strategy_name, avg_sharpe) in enumerate(rankings[:10], 1):
            f.write(f"{i}. {strategy_name}: Sharpe = {avg_sharpe:.2f}\n")

            # Show details for each horizon
            for horizon_name in all_windows.keys():
                train_key = f"{horizon_name}_train"
                test_key = f"{horizon_name}_test"

                if train_key in aggregated[strategy_name] and test_key in aggregated[strategy_name]:
                    train_m = aggregated[strategy_name][train_key]
                    test_m = aggregated[strategy_name][test_key]

                    f.write(f"   {horizon_name}: Train Sharpe={train_m.mean_sharpe:.2f}±{train_m.std_sharpe:.2f}, "
                           f"Test Sharpe={test_m.mean_sharpe:.2f}±{test_m.std_sharpe:.2f}\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("GENERALIZATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Strategies with best train-to-test consistency (low overfitting):\n\n")

        # Calculate train/test gap
        consistency_scores = []
        for strategy_name, horizon_metrics in aggregated.items():
            gaps = []
            for horizon_name in all_windows.keys():
                train_key = f"{horizon_name}_train"
                test_key = f"{horizon_name}_test"

                if train_key in horizon_metrics and test_key in horizon_metrics:
                    train_m = horizon_metrics[train_key]
                    test_m = horizon_metrics[test_key]

                    if train_m.num_windows > 0 and test_m.num_windows > 0:
                        # Calculate gap (positive = test better than train = no overfit)
                        gap = test_m.mean_sharpe - train_m.mean_sharpe
                        gaps.append(gap)

            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                consistency_scores.append((strategy_name, avg_gap))

        consistency_scores.sort(key=lambda x: -x[1])  # Sort by smallest gap (most negative = overfit)

        for i, (strategy_name, gap) in enumerate(consistency_scores[:10], 1):
            if gap >= 0:
                status = "✓ Generalizes well"
            elif gap > -0.5:
                status = "○ Slight overfit"
            else:
                status = "✗ Significant overfit"

            f.write(f"{i}. {strategy_name}: Gap = {gap:+.2f} {status}\n")

    logger.success(f"✅ Report saved to {report_file}")

    duration = time.perf_counter() - start_time
    logger.success(f"\n✅ Analysis complete in {duration:.1f}s")
    logger.info(f"📁 Results saved to {output_path}")


if __name__ == "__main__":
    app()
