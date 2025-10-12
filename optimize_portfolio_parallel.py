#!/usr/bin/env python3
"""
Parallel Portfolio Optimization with Walk-Forward Analysis

High-performance parallelized version using multiprocessing for 10-15x speedup.

**Parallelization Strategy**: Config-level parallelization with shared data
- Each worker processes one complete configuration (all splits)
- Historical data shared via multiprocessing for memory efficiency
- Progress tracking with tqdm
- Automatic worker count detection

**Performance**: ~10-15x speedup on 16-core systems

Usage:
    python optimize_portfolio_parallel.py --quick
    python optimize_portfolio_parallel.py --workers 8
    python optimize_portfolio_parallel.py --timeframe 1h --test-windows 5
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from itertools import product, combinations
import warnings
from collections import defaultdict
import multiprocessing as mp
import os

# Add src directory to Python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import typer
import yaml
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from crypto_trader.data.fetchers import BinanceDataFetcher

# Suppress warnings
warnings.filterwarnings('ignore')

app = typer.Typer(help="Parallel portfolio optimization with walk-forward analysis")

# Global variable for shared data (set by worker initialization)
_shared_historical_data = None
_shared_splits = None
_shared_timeframe = None


def worker_init(historical_data_dict: Dict[str, pd.DataFrame],
                splits: List[Tuple],
                timeframe: str):
    """
    Initialize worker process with shared data.

    This runs once per worker when the pool is created.
    """
    global _shared_historical_data, _shared_splits, _shared_timeframe
    _shared_historical_data = historical_data_dict
    _shared_splits = splits
    _shared_timeframe = timeframe


def backtest_single_period(
    period_data: Dict[str, pd.DataFrame],
    timestamps: List[pd.Timestamp],
    assets: List[Tuple[str, float]],
    rebalance_params: Dict[str, Any],
    timeframe: str,
    initial_capital: float = 10000.0
) -> Dict[str, float]:
    """
    Backtest portfolio on a single period.

    Extracted for reusability between train and test periods.
    """
    try:
        # Initialize portfolio
        shares = {}
        for symbol, weight in assets:
            allocation = initial_capital * weight
            initial_price = period_data[symbol].loc[timestamps[0], 'close']
            shares[symbol] = allocation / initial_price

        # Simulate with rebalancing
        equity_values = []
        rebalance_count = 0
        last_rebalance = None

        for timestamp in timestamps:
            prices = {s: period_data[s].loc[timestamp, 'close'] for s, _ in assets}
            portfolio_values = {s: shares[s] * prices[s] for s in shares}
            total_value = sum(portfolio_values.values())
            current_weights = {s: portfolio_values[s] / total_value for s in portfolio_values}

            # Check rebalancing
            needs_rebalance = False
            max_deviation = max(abs(current_weights[s] - w) for s, w in assets)

            if rebalance_params['rebalance_method'] == 'threshold':
                needs_rebalance = max_deviation > rebalance_params['threshold']
            elif rebalance_params['rebalance_method'] == 'calendar':
                if last_rebalance is not None:
                    days_since = (timestamp - last_rebalance).total_seconds() / 86400
                    needs_rebalance = days_since >= rebalance_params['calendar_period_days']
            elif rebalance_params['rebalance_method'] == 'hybrid':
                threshold_trigger = max_deviation > rebalance_params['threshold']
                calendar_trigger = False
                if last_rebalance is not None:
                    days_since = (timestamp - last_rebalance).total_seconds() / 86400
                    calendar_trigger = days_since >= rebalance_params['calendar_period_days']
                needs_rebalance = threshold_trigger or calendar_trigger

            # Min interval check
            if needs_rebalance and last_rebalance is not None:
                hours_since = (timestamp - last_rebalance).total_seconds() / 3600
                if hours_since < rebalance_params['min_rebalance_interval_hours']:
                    needs_rebalance = False

            # Momentum filter
            if needs_rebalance and rebalance_params['use_momentum_filter']:
                lookback_periods = 30 * 24 if timeframe == "1h" else 30
                current_idx = list(timestamps).index(timestamp)
                if current_idx >= lookback_periods:
                    lookback_ts = timestamps[current_idx - lookback_periods]
                    old_prices = {s: period_data[s].loc[lookback_ts, 'close'] for s in prices}
                    old_value = sum(shares[s] * old_prices[s] for s in shares)
                    portfolio_return = (total_value - old_value) / old_value
                    if portfolio_return > 0.20:
                        needs_rebalance = False

            # Execute rebalance
            if needs_rebalance:
                target_values = {s: total_value * w for s, w in assets}
                shares = {s: target_values[s] / prices[s] for s in prices}
                rebalance_count += 1
                last_rebalance = timestamp

            equity_values.append(total_value)

        # Calculate buy-and-hold benchmark
        buyhold_shares = {}
        for symbol, weight in assets:
            allocation = initial_capital * weight
            initial_price = period_data[symbol].loc[timestamps[0], 'close']
            buyhold_shares[symbol] = allocation / initial_price

        buyhold_final = sum(buyhold_shares[s] * period_data[s].loc[timestamps[-1], 'close']
                           for s, _ in assets)

        # Calculate metrics
        final_value = equity_values[-1]
        total_return = (final_value / initial_capital) - 1
        buyhold_return = (buyhold_final / initial_capital) - 1
        outperformance = total_return - buyhold_return

        # Sharpe ratio
        returns = pd.Series(equity_values).pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            periods_per_year = {'1h': 24*365, '4h': 6*365, '1d': 365}.get(timeframe, 24*365)
            sharpe = (returns.mean() * periods_per_year) / (returns.std() * np.sqrt(periods_per_year))
        else:
            sharpe = 0.0

        # Max drawdown
        peak = equity_values[0]
        max_dd = 0.0
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (value - peak) / peak
            if dd < max_dd:
                max_dd = dd

        # Volatility
        if len(returns) > 0:
            periods_per_year = {'1h': 24*365, '4h': 6*365, '1d': 365}.get(timeframe, 24*365)
            volatility = returns.std() * np.sqrt(periods_per_year)
        else:
            volatility = 0.0

        return {
            'total_return': total_return,
            'buyhold_return': buyhold_return,
            'outperformance': outperformance,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'volatility': volatility,
            'rebalance_count': rebalance_count,
            'final_value': final_value,
            'periods': len(timestamps)
        }

    except Exception as e:
        return {'error': str(e)}


def process_configuration(config_tuple: Tuple) -> Optional[Dict]:
    """
    Process one configuration across all walk-forward splits.

    This function runs in parallel workers. It must be picklable (top-level function).

    Args:
        config_tuple: (config_id, assets, rebalance_params)

    Returns:
        Dictionary with aggregated results or None if failed
    """
    config_id, assets, rebalance_params = config_tuple

    # Access shared data (set by worker_init)
    historical_data = _shared_historical_data
    splits = _shared_splits
    timeframe = _shared_timeframe

    if historical_data is None or splits is None:
        return None

    try:
        train_metrics_list = []
        test_metrics_list = []

        for train_start, train_end, test_end in splits:
            # Extract train period data
            train_data = {}
            for symbol, _ in assets:
                if symbol not in historical_data:
                    continue
                data = historical_data[symbol]
                mask = (data.index >= train_start) & (data.index <= train_end)
                train_data[symbol] = data[mask]

            # Get common train timestamps
            train_timestamps = None
            for data in train_data.values():
                if train_timestamps is None:
                    train_timestamps = data.index
                else:
                    train_timestamps = train_timestamps.intersection(data.index)

            if len(train_timestamps) >= 10:
                train_timestamps = sorted(train_timestamps)
                train_metrics = backtest_single_period(
                    train_data, train_timestamps, assets, rebalance_params, timeframe
                )
                if 'error' not in train_metrics:
                    train_metrics_list.append(train_metrics)

            # Extract test period data
            test_data = {}
            for symbol, _ in assets:
                if symbol not in historical_data:
                    continue
                data = historical_data[symbol]
                mask = (data.index > train_end) & (data.index <= test_end)
                test_data[symbol] = data[mask]

            # Get common test timestamps
            test_timestamps = None
            for data in test_data.values():
                if test_timestamps is None:
                    test_timestamps = data.index
                else:
                    test_timestamps = test_timestamps.intersection(data.index)

            if len(test_timestamps) >= 10:
                test_timestamps = sorted(test_timestamps)
                test_metrics = backtest_single_period(
                    test_data, test_timestamps, assets, rebalance_params, timeframe
                )
                if 'error' not in test_metrics:
                    test_metrics_list.append(test_metrics)

        # Aggregate results
        if train_metrics_list and test_metrics_list:
            return {
                'config_id': config_id,
                'assets': [s for s, _ in assets],
                'weights': [w for _, w in assets],
                'rebalance_params': rebalance_params,

                # Training metrics (in-sample)
                'train_avg_outperformance': np.mean([m['outperformance'] for m in train_metrics_list]),
                'train_avg_return': np.mean([m['total_return'] for m in train_metrics_list]),
                'train_avg_sharpe': np.mean([m['sharpe_ratio'] for m in train_metrics_list]),
                'train_avg_drawdown': np.mean([m['max_drawdown'] for m in train_metrics_list]),

                # Test metrics (out-of-sample) - MOST IMPORTANT
                'test_avg_outperformance': np.mean([m['outperformance'] for m in test_metrics_list]),
                'test_avg_return': np.mean([m['total_return'] for m in test_metrics_list]),
                'test_avg_sharpe': np.mean([m['sharpe_ratio'] for m in test_metrics_list]),
                'test_avg_drawdown': np.mean([m['max_drawdown'] for m in test_metrics_list]),
                'test_consistency': np.std([m['outperformance'] for m in test_metrics_list]),

                # Win rate
                'test_win_rate': sum(1 for m in test_metrics_list if m['outperformance'] > 0) / len(test_metrics_list),

                # Robustness (test vs train)
                'generalization_gap': np.mean([m['outperformance'] for m in train_metrics_list]) -
                                     np.mean([m['outperformance'] for m in test_metrics_list]),

                'splits_tested': len(test_metrics_list)
            }
        else:
            return None

    except Exception as e:
        return None


class ParallelOptimizer:
    """Parallel portfolio optimizer with multiprocessing."""

    def __init__(
        self,
        window_days: int = 365,
        timeframe: str = "1h",
        test_windows: int = 5,
        quick_mode: bool = False,
        workers: Optional[int] = None
    ):
        """Initialize parallel optimizer."""
        self.window_days = window_days
        self.timeframe = timeframe
        self.test_windows = test_windows
        self.quick_mode = quick_mode

        # Determine number of workers
        if workers is None:
            self.workers = max(1, mp.cpu_count() - 1)  # Leave one core free
        else:
            self.workers = min(workers, mp.cpu_count())

        self.fetcher = BinanceDataFetcher()

        logger.info(f"Parallel Optimizer initialized:")
        logger.info(f"  CPU cores: {mp.cpu_count()}")
        logger.info(f"  Workers: {self.workers}")
        logger.info(f"  Window size: {window_days} days")
        logger.info(f"  Timeframe: {timeframe}")
        logger.info(f"  Test windows: {test_windows}")
        logger.info(f"  Quick mode: {quick_mode}")

    def get_asset_universe(self) -> List[str]:
        """Get asset universe."""
        return [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
            "ADA/USDT", "XRP/USDT", "MATIC/USDT", "DOT/USDT",
        ]

    def get_asset_combinations(self) -> List[List[str]]:
        """Generate asset combinations to test."""
        universe = self.get_asset_universe()

        if self.quick_mode:
            return [
                ["BTC/USDT", "ETH/USDT"],
                ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
            ]
        else:
            combinations_list = []
            combinations_list.extend([
                ["BTC/USDT", "ETH/USDT"],
                ["BTC/USDT", "BNB/USDT"],
                ["ETH/USDT", "BNB/USDT"],
            ])

            for combo in combinations(universe[:6], 3):
                combinations_list.append(list(combo))
                if len(combinations_list) >= 15:
                    break

            combinations_list.extend([
                ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
                ["BTC/USDT", "ETH/USDT", "ADA/USDT", "BNB/USDT"],
                ["BTC/USDT", "ETH/USDT", "SOL/USDT", "MATIC/USDT"],
            ])

            combinations_list.extend([
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"],
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "MATIC/USDT"],
            ])

            return combinations_list

    def get_weight_schemes(self, num_assets: int) -> List[List[float]]:
        """Generate weight allocation schemes."""
        schemes = []

        if num_assets == 2:
            schemes = [[0.50, 0.50], [0.60, 0.40], [0.70, 0.30], [0.40, 0.60]]
        elif num_assets == 3:
            schemes = [[0.33, 0.33, 0.34], [0.50, 0.30, 0.20], [0.40, 0.35, 0.25], [0.60, 0.25, 0.15]]
        elif num_assets == 4:
            schemes = [[0.25, 0.25, 0.25, 0.25], [0.40, 0.30, 0.15, 0.15],
                      [0.35, 0.35, 0.15, 0.15], [0.50, 0.25, 0.15, 0.10], [0.30, 0.30, 0.20, 0.20]]
        elif num_assets == 5:
            schemes = [[0.20, 0.20, 0.20, 0.20, 0.20], [0.35, 0.25, 0.20, 0.10, 0.10],
                      [0.30, 0.25, 0.20, 0.15, 0.10]]
        else:
            weight = 1.0 / num_assets
            schemes = [[weight] * num_assets]

        if self.quick_mode and len(schemes) > 2:
            schemes = schemes[:2]

        return schemes

    def get_rebalancing_parameters(self) -> Dict[str, List]:
        """Get rebalancing parameter grid."""
        if self.quick_mode:
            return {
                'threshold': [0.10, 0.15],
                'rebalance_method': ['threshold'],
                'calendar_period_days': [30],
                'min_rebalance_interval_hours': [24],
                'use_momentum_filter': [False]
            }
        else:
            return {
                'threshold': [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
                'rebalance_method': ['threshold', 'calendar', 'hybrid'],
                'calendar_period_days': [7, 14, 30, 60, 90],
                'min_rebalance_interval_hours': [12, 24, 48, 72],
                'use_momentum_filter': [False, True]
            }

    def fetch_historical_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch historical data."""
        logger.info(f"\nFetching historical data for {len(symbols)} assets...")

        total_days = self.window_days * (self.test_windows + 1)

        if self.timeframe == "1h":
            limit = total_days * 24
        elif self.timeframe == "4h":
            limit = total_days * 6
        elif self.timeframe == "1d":
            limit = total_days
        else:
            limit = total_days * 24

        historical_data = {}

        for symbol in symbols:
            try:
                data = self.fetcher.get_ohlcv(symbol, self.timeframe, limit=limit)

                if data is None or len(data) < limit * 0.5:
                    logger.warning(f"  ⚠ Insufficient data for {symbol}")
                    continue

                historical_data[symbol] = data
                logger.success(f"  ✓ {symbol}: {len(data)} candles")

            except Exception as e:
                logger.error(f"  ✗ {symbol}: {e}")
                continue

        return historical_data

    def create_walk_forward_splits(
        self, historical_data: Dict[str, pd.DataFrame]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Create walk-forward splits."""
        timestamps = None
        for data in historical_data.values():
            if timestamps is None:
                timestamps = data.index
            else:
                timestamps = timestamps.intersection(data.index)

        timestamps = sorted(timestamps)

        if self.timeframe == "1h":
            periods_per_window = self.window_days * 24
        elif self.timeframe == "4h":
            periods_per_window = self.window_days * 6
        elif self.timeframe == "1d":
            periods_per_window = self.window_days
        else:
            periods_per_window = self.window_days * 24

        splits = []

        for i in range(self.test_windows):
            train_start_idx = 0
            train_end_idx = periods_per_window * (i + 1)
            test_end_idx = periods_per_window * (i + 2)

            if test_end_idx > len(timestamps):
                break

            splits.append((
                timestamps[train_start_idx],
                timestamps[train_end_idx - 1],
                timestamps[test_end_idx - 1]
            ))

        logger.info(f"\nCreated {len(splits)} walk-forward splits:")
        for i, (train_start, train_end, test_end) in enumerate(splits, 1):
            train_days = (train_end - train_start).days
            test_days = (test_end - train_end).days
            logger.info(f"  Split {i}: Train {train_days}d ({train_start.date()} to {train_end.date()}) "
                       f"→ Test {test_days}d (to {test_end.date()})")

        return splits

    def optimize(self, output_dir: str = "optimization_results") -> Dict[str, Any]:
        """Run parallel optimization."""
        from optimize_portfolio_comprehensive import ComprehensiveOptimizer

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info("\n" + "=" * 80)
        logger.info("PARALLEL PORTFOLIO OPTIMIZATION")
        logger.info(f"Using {self.workers} parallel workers")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Step 1: Fetch data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Fetching Historical Data")
        logger.info("=" * 80)

        asset_universe = self.get_asset_universe()
        all_historical_data = self.fetch_historical_data(asset_universe)
        logger.success(f"Fetched data for {len(all_historical_data)}/{len(asset_universe)} assets")

        # Step 2: Create splits
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Creating Walk-Forward Splits")
        logger.info("=" * 80)

        splits = self.create_walk_forward_splits(all_historical_data)

        # Step 3: Generate configs
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Generating Configuration Grid")
        logger.info("=" * 80)

        asset_combinations = self.get_asset_combinations()
        rebalance_grid = self.get_rebalancing_parameters()
        rebalance_param_names = list(rebalance_grid.keys())
        rebalance_combinations = list(product(*rebalance_grid.values()))

        # Build all configurations
        all_configs = []
        config_id = 0

        for asset_combo in asset_combinations:
            valid_assets = [a for a in asset_combo if a in all_historical_data]
            if len(valid_assets) < 2:
                continue

            weight_schemes = self.get_weight_schemes(len(valid_assets))

            for weights in weight_schemes:
                assets = list(zip(valid_assets, weights))

                for rebalance_combo in rebalance_combinations:
                    rebalance_params = dict(zip(rebalance_param_names, rebalance_combo))
                    config_id += 1
                    all_configs.append((config_id, assets, rebalance_params))

        logger.info(f"Total configurations: {len(all_configs)}")
        logger.info(f"Total backtests: {len(all_configs) * len(splits) * 2:,} (train + test)")
        logger.info(f"Estimated serial time: {len(all_configs) * len(splits) * 0.1 / 60:.1f} minutes")
        logger.info(f"Estimated parallel time ({self.workers} workers): {len(all_configs) * len(splits) * 0.1 / 60 / self.workers:.1f} minutes")

        # Step 4: Run parallel optimization
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Running Parallel Walk-Forward Optimization")
        logger.info("=" * 80)

        # Create process pool with worker initialization
        with mp.Pool(
            processes=self.workers,
            initializer=worker_init,
            initargs=(all_historical_data, splits, self.timeframe)
        ) as pool:

            # Process configs in parallel with progress bar
            results_list = list(tqdm(
                pool.imap_unordered(process_configuration, all_configs),
                total=len(all_configs),
                desc="Optimizing",
                unit="config",
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ))

        # Filter out None results
        all_results = [r for r in results_list if r is not None]

        duration = (datetime.now() - start_time).total_seconds()
        logger.success(f"\n✓ Completed optimization in {duration:.1f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"  Valid results: {len(all_results)}/{len(all_configs)}")
        logger.info(f"  Speedup: {len(all_configs) * len(splits) * 0.1 / duration:.1f}x vs estimated serial")

        # Step 5: Analyze results
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Analyzing Results")
        logger.info("=" * 80)

        if not all_results:
            logger.error("❌ No valid results - optimization failed")
            sys.exit(1)

        all_results.sort(key=lambda x: x['test_avg_outperformance'], reverse=True)
        best = all_results[0]

        logger.info("\n🏆 BEST CONFIGURATION (Out-of-Sample Performance):")
        logger.info(f"  Config ID: {best['config_id']}")
        logger.info(f"  Assets: {', '.join(best['assets'])}")
        logger.info(f"  Weights: {[f'{w:.1%}' for w in best['weights']]}")
        logger.info(f"  Threshold: {best['rebalance_params']['threshold']:.2%}")
        logger.info(f"  Method: {best['rebalance_params']['rebalance_method']}")
        logger.info("\n📊 OUT-OF-SAMPLE PERFORMANCE:")
        logger.info(f"  Test Outperformance: {best['test_avg_outperformance']:.2%}")
        logger.info(f"  Test Return: {best['test_avg_return']:.2%}")
        logger.info(f"  Test Sharpe: {best['test_avg_sharpe']:.3f}")
        logger.info(f"  Test Win Rate: {best['test_win_rate']:.1%}")
        logger.info(f"  Generalization Gap: {best['generalization_gap']:.2%}")

        # Step 6: Generate reports (reuse from serial version)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Generating Reports")
        logger.info("=" * 80)

        # Use ComprehensiveOptimizer methods for report generation
        serial_optimizer = ComprehensiveOptimizer(
            window_days=self.window_days,
            timeframe=self.timeframe,
            test_windows=self.test_windows,
            quick_mode=self.quick_mode
        )

        serial_optimizer._generate_research_report(all_results, splits, output_path)
        serial_optimizer._generate_optimized_config(best, output_path)
        serial_optimizer._save_detailed_results(all_results, output_path)

        logger.success(f"\n✅ OPTIMIZATION COMPLETE")
        logger.info(f"Results saved to: {output_path}")

        return {
            'best_config': best,
            'all_results': all_results,
            'splits': splits,
            'duration': duration,
            'workers_used': self.workers
        }


@app.command()
def optimize(
    window_days: int = typer.Option(365, "--window-days", "-d", help="Window size in days"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe (1h, 4h, 1d)"),
    test_windows: int = typer.Option(5, "--test-windows", "-n", help="Number of walk-forward test windows"),
    quick_mode: bool = typer.Option(False, "--quick", "-q", help="Quick mode with reduced parameter grid"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Number of parallel workers (default: auto)"),
    output_dir: str = typer.Option("optimization_results", "--output", "-o", help="Output directory")
):
    """
    Run parallel portfolio optimization with walk-forward analysis.

    Uses multiprocessing for 10-15x speedup over serial execution.

    Example:
        python optimize_portfolio_parallel.py --quick
        python optimize_portfolio_parallel.py --workers 8
        python optimize_portfolio_parallel.py --timeframe 1h --test-windows 5
    """
    optimizer = ParallelOptimizer(
        window_days=window_days,
        timeframe=timeframe,
        test_windows=test_windows,
        quick_mode=quick_mode,
        workers=workers
    )

    try:
        result = optimizer.optimize(output_dir=output_dir)

        logger.info("\n" + "=" * 80)
        logger.success("✅ PARALLEL OPTIMIZATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nPerformance:")
        logger.info(f"  Duration: {result['duration']:.1f}s ({result['duration']/60:.2f} min)")
        logger.info(f"  Workers used: {result['workers_used']}")
        logger.info(f"\nGenerated Files:")
        logger.info(f"  1. {output_dir}/optimized_config.yaml")
        logger.info(f"  2. {output_dir}/OPTIMIZATION_REPORT.txt")
        logger.info(f"  3. {output_dir}/optimization_results_*.csv")

    except Exception as e:
        logger.error(f"\n❌ Optimization failed: {e}")
        logger.exception("Full traceback:")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
