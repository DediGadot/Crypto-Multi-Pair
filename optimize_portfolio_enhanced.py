#!/usr/bin/env python3
"""
ENHANCED Portfolio Optimization with Comprehensive Data Validation

This version fixes all data availability mismatch issues:
1. ✅ Pre-flight data validation BEFORE starting optimization
2. ✅ Auto-adjust mode - automatically uses best parameters for available data
3. ✅ Detailed error logging with root cause analysis
4. ✅ Data availability report generation
5. ✅ Graceful degradation when assets have varying data lengths
6. ✅ Comprehensive parameter suggestions

Usage:
    # Auto-adjust to available data:
    python optimize_portfolio_enhanced.py --auto-adjust --quick

    # Manual parameters with validation:
    python optimize_portfolio_enhanced.py --timeframe 1h --window-days 350 --test-windows 5

    # Check data availability only:
    python optimize_portfolio_enhanced.py --check-data-only
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from numba import njit

from crypto_trader.data.fetchers import BinanceDataFetcher

# Import optimized functions from the original script
from optimize_portfolio_optimized import (
    simulate_portfolio_rebalancing_numba,
    calculate_sharpe_ratio_numba,
    calculate_max_drawdown_numba,
    calculate_volatility_numba,
    worker_init,
    process_configuration
)

# Suppress warnings
warnings.filterwarnings('ignore')

app = typer.Typer(help="ENHANCED portfolio optimization with comprehensive data validation")

# Global variables
_shared_price_arrays = None
_shared_timestamp_arrays = None
_shared_splits = None
_shared_timeframe = None


class DataAvailabilityReport:
    """Comprehensive data availability analysis."""

    def __init__(self, symbols: List[str], timeframe: str, fetcher: BinanceDataFetcher):
        self.symbols = symbols
        self.timeframe = timeframe
        self.fetcher = fetcher
        self.availability: Dict[str, Dict] = {}

    def analyze(self) -> Dict[str, Any]:
        """Analyze data availability for all symbols."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 DATA AVAILABILITY ANALYSIS")
        logger.info("=" * 80)

        all_data = {}
        for symbol in self.symbols:
            try:
                data = self.fetcher.get_ohlcv(symbol, self.timeframe, limit=100000)
                if data is not None and len(data) > 0:
                    all_data[symbol] = data
                    self.availability[symbol] = {
                        'available': True,
                        'periods': len(data),
                        'start_date': data.index[0],
                        'end_date': data.index[-1],
                        'days': (data.index[-1] - data.index[0]).days
                    }
                    logger.success(f"  ✓ {symbol:15s}: {len(data):6,} periods ({self.availability[symbol]['days']:4} days)")
                else:
                    self.availability[symbol] = {'available': False, 'periods': 0}
                    logger.warning(f"  ⚠ {symbol:15s}: No data available")
            except Exception as e:
                self.availability[symbol] = {'available': False, 'periods': 0, 'error': str(e)}
                logger.error(f"  ✗ {symbol:15s}: {e}")

        # Calculate common timestamps
        if len(all_data) >= 2:
            common_timestamps = None
            for data in all_data.values():
                if common_timestamps is None:
                    common_timestamps = set(data.index)
                else:
                    common_timestamps = common_timestamps.intersection(set(data.index))

            common_count = len(common_timestamps)
            logger.info(f"\n📍 Common timestamps across all assets: {common_count:,}")

            return {
                'individual_data': all_data,
                'availability': self.availability,
                'common_periods': common_count,
                'assets_with_data': len(all_data),
                'total_assets': len(self.symbols)
            }
        else:
            logger.error("\n❌ Insufficient assets with data")
            return {
                'individual_data': all_data,
                'availability': self.availability,
                'common_periods': 0,
                'assets_with_data': len(all_data),
                'total_assets': len(self.symbols)
            }

    def calculate_max_parameters(self, common_periods: int, timeframe: str) -> Dict[str, int]:
        """Calculate maximum possible parameters for available data."""
        if timeframe == "1h":
            periods_per_day = 24
        elif timeframe == "4h":
            periods_per_day = 6
        elif timeframe == "1d":
            periods_per_day = 1
        else:
            periods_per_day = 24

        logger.info("\n" + "=" * 80)
        logger.info("🎯 OPTIMAL PARAMETER CALCULATION")
        logger.info("=" * 80)

        results = {}

        # For different test window counts
        for test_windows in [3, 4, 5, 6, 7]:
            total_windows = test_windows + 1
            max_periods_per_window = int(common_periods * 0.95 / total_windows)  # Use 95% for safety
            max_days_per_window = max_periods_per_window // periods_per_day

            if max_days_per_window >= 30:  # Minimum reasonable window
                results[test_windows] = {
                    'max_window_days': max_days_per_window,
                    'periods_per_window': max_periods_per_window,
                    'total_periods_needed': max_periods_per_window * total_windows,
                    'available_periods': common_periods,
                    'margin': common_periods - (max_periods_per_window * total_windows)
                }
                logger.info(f"  test_windows={test_windows}: max window_days={max_days_per_window:3} "
                           f"(needs {max_periods_per_window * total_windows:,} periods, "
                           f"margin={results[test_windows]['margin']:,})")

        if not results:
            logger.error("\n❌ Insufficient data for any reasonable parameters")
            logger.error(f"   Available periods: {common_periods:,}")
            logger.error(f"   Minimum needed: ~{30 * periods_per_day * 4:,} (for 30-day windows, 3 test windows)")

        return results

    def suggest_parameters(self, common_periods: int, timeframe: str, requested_test_windows: int = 5) -> Optional[Dict[str, int]]:
        """Suggest optimal parameters based on available data."""
        max_params = self.calculate_max_parameters(common_periods, timeframe)

        if not max_params:
            return None

        # Try to match requested test windows
        if requested_test_windows in max_params:
            suggestion = max_params[requested_test_windows]
            logger.info(f"\n✅ RECOMMENDED PARAMETERS (for {requested_test_windows} test windows):")
            logger.info(f"   --window-days {suggestion['max_window_days']}")
            logger.info(f"   --test-windows {requested_test_windows}")
            logger.info(f"   --timeframe {timeframe}")
            return {'window_days': suggestion['max_window_days'], 'test_windows': requested_test_windows}

        # Find closest available
        available_windows = sorted(max_params.keys())
        closest = min(available_windows, key=lambda x: abs(x - requested_test_windows))
        suggestion = max_params[closest]

        logger.warning(f"\n⚠️  Requested {requested_test_windows} test windows not possible")
        logger.info(f"✅ RECOMMENDED PARAMETERS (for {closest} test windows):")
        logger.info(f"   --window-days {suggestion['max_window_days']}")
        logger.info(f"   --test-windows {closest}")
        logger.info(f"   --timeframe {timeframe}")

        return {'window_days': suggestion['max_window_days'], 'test_windows': closest}


class EnhancedOptimizer:
    """Enhanced optimizer with comprehensive data validation and auto-adjustment."""

    def __init__(
        self,
        window_days: Optional[int] = None,
        timeframe: str = "1h",
        test_windows: Optional[int] = None,
        quick_mode: bool = False,
        workers: Optional[int] = None,
        auto_adjust: bool = False
    ):
        """Initialize enhanced optimizer."""
        self.timeframe = timeframe
        self.quick_mode = quick_mode
        self.auto_adjust = auto_adjust

        # Will be set after data validation
        self.window_days = window_days
        self.test_windows = test_windows

        if workers is None:
            self.workers = max(1, mp.cpu_count() - 1)
        else:
            self.workers = min(workers, mp.cpu_count())

        self.fetcher = BinanceDataFetcher()

        logger.info(f"\n🚀 ENHANCED Optimizer initialized:")
        logger.info(f"  Mode: {'AUTO-ADJUST' if auto_adjust else 'MANUAL'}")
        logger.info(f"  Timeframe: {timeframe}")
        logger.info(f"  CPU cores: {mp.cpu_count()}")
        logger.info(f"  Workers: {self.workers}")

    def get_asset_universe(self) -> List[str]:
        """Get asset universe."""
        return [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
            "ADA/USDT", "XRP/USDT", "MATIC/USDT", "DOT/USDT",
        ]

    def validate_and_prepare_data(self, requested_window_days: int, requested_test_windows: int) -> Tuple[Dict, Dict, List, bool]:
        """
        Validate data availability and prepare for optimization.

        Returns:
            (price_arrays, timestamp_arrays, splits, is_valid)
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: DATA VALIDATION & PREPARATION")
        logger.info("=" * 80)

        # Step 1: Analyze data availability
        asset_universe = self.get_asset_universe()
        report = DataAvailabilityReport(asset_universe, self.timeframe, self.fetcher)
        analysis = report.analyze()

        if analysis['assets_with_data'] < 2:
            logger.error("\n❌ CRITICAL: Need at least 2 assets with data")
            return {}, {}, [], False

        # Step 2: Calculate requirements
        if self.timeframe == "1h":
            periods_per_day = 24
        elif self.timeframe == "4h":
            periods_per_day = 6
        elif self.timeframe == "1d":
            periods_per_day = 1
        else:
            periods_per_day = 24

        periods_per_window = requested_window_days * periods_per_day
        total_windows = requested_test_windows + 1
        required_periods = periods_per_window * total_windows

        logger.info("\n" + "=" * 80)
        logger.info("📐 PARAMETER VALIDATION")
        logger.info("=" * 80)
        logger.info(f"  Requested window_days: {requested_window_days}")
        logger.info(f"  Requested test_windows: {requested_test_windows}")
        logger.info(f"  Periods per window: {periods_per_window:,}")
        logger.info(f"  Total windows needed: {total_windows}")
        logger.info(f"  Required periods: {required_periods:,}")
        logger.info(f"  Available periods: {analysis['common_periods']:,}")

        # Step 3: Validate sufficiency
        margin = analysis['common_periods'] - required_periods

        if margin < 0:
            logger.error(f"\n❌ INSUFFICIENT DATA")
            logger.error(f"   Need: {required_periods:,} periods")
            logger.error(f"   Have: {analysis['common_periods']:,} periods")
            logger.error(f"   Shortfall: {abs(margin):,} periods")

            # Suggest alternatives
            suggestions = report.suggest_parameters(
                analysis['common_periods'],
                self.timeframe,
                requested_test_windows
            )

            if suggestions and self.auto_adjust:
                logger.info(f"\n🔄 AUTO-ADJUST MODE: Using recommended parameters")
                self.window_days = suggestions['window_days']
                self.test_windows = suggestions['test_windows']
                # Retry with adjusted parameters
                return self.validate_and_prepare_data(self.window_days, self.test_windows)
            else:
                logger.info("\n💡 Run with --auto-adjust to automatically use optimal parameters")
                return {}, {}, [], False

        elif margin < required_periods * 0.05:  # Less than 5% margin
            logger.warning(f"\n⚠️  LOW MARGIN: Only {margin:,} extra periods ({margin/required_periods*100:.1f}%)")
            logger.warning(f"   Consider reducing parameters for safety")
        else:
            logger.success(f"\n✅ SUFFICIENT DATA: {margin:,} extra periods ({margin/required_periods*100:.1f}% margin)")

        # Step 4: Fetch and prepare data
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: DATA FETCHING & CONVERSION")
        logger.info("=" * 80)

        all_historical_data = self._fetch_data_parallel(analysis['individual_data'], required_periods)

        if len(all_historical_data) < 2:
            logger.error("\n❌ Less than 2 assets have sufficient data after filtering")
            return {}, {}, [], False

        # Convert to NumPy
        price_arrays, timestamp_arrays = self._convert_to_numpy(all_historical_data)

        # Create splits
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: WALK-FORWARD SPLIT CREATION")
        logger.info("=" * 80)

        splits = self._create_validated_splits(timestamp_arrays, requested_window_days, requested_test_windows)

        if len(splits) < requested_test_windows:
            logger.error(f"\n❌ Could only create {len(splits)} splits (requested {requested_test_windows})")
            logger.error(f"   This indicates a data availability issue")
            return {}, {}, [], False

        logger.success(f"\n✅ VALIDATION COMPLETE: All checks passed")
        logger.info(f"   Assets: {len(price_arrays)}")
        logger.info(f"   Splits: {len(splits)}")
        logger.info(f"   Ready for optimization")

        return price_arrays, timestamp_arrays, splits, True

    def _fetch_data_parallel(self, existing_data: Dict[str, pd.DataFrame], min_periods: int) -> Dict[str, pd.DataFrame]:
        """Fetch and filter data with minimum period requirement."""
        filtered_data = {}

        for symbol, data in existing_data.items():
            if len(data) >= min_periods * 0.5:  # At least 50% of required
                filtered_data[symbol] = data
                logger.success(f"  ✓ {symbol:15s}: {len(data):6,} periods (sufficient)")
            else:
                logger.warning(f"  ⚠ {symbol:15s}: {len(data):6,} periods (insufficient, skipped)")

        return filtered_data

    def _convert_to_numpy(self, historical_data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
        """Convert to NumPy arrays."""
        logger.info("\n🔢 Converting to NumPy arrays...")

        price_arrays = {}
        timestamp_arrays = {}

        for symbol, df in historical_data.items():
            price_arrays[symbol] = df['close'].values
            timestamp_arrays[symbol] = df.index.values
            logger.success(f"  ✓ {symbol:15s}: {len(price_arrays[symbol]):6,} prices")

        return price_arrays, timestamp_arrays

    def _create_validated_splits(
        self,
        timestamp_arrays: Dict[str, np.ndarray],
        window_days: int,
        test_windows: int
    ) -> List[Tuple]:
        """Create and validate walk-forward splits."""

        # Get common timestamps
        timestamps = None
        for ts_arr in timestamp_arrays.values():
            if timestamps is None:
                timestamps = ts_arr
            else:
                timestamps = np.intersect1d(timestamps, ts_arr)

        timestamps = np.sort(timestamps)

        if self.timeframe == "1h":
            periods_per_window = window_days * 24
        elif self.timeframe == "4h":
            periods_per_window = window_days * 6
        elif self.timeframe == "1d":
            periods_per_window = window_days
        else:
            periods_per_window = window_days * 24

        splits = []

        for i in range(test_windows):
            train_start_idx = 0
            train_end_idx = periods_per_window * (i + 1)
            test_end_idx = periods_per_window * (i + 2)

            if test_end_idx > len(timestamps):
                logger.warning(f"  ⚠ Split {i+1}: Not enough data (would need index {test_end_idx}, have {len(timestamps)})")
                break

            # Validate this split has enough periods
            train_periods = train_end_idx - train_start_idx
            test_periods = test_end_idx - train_end_idx

            if train_periods < 100 or test_periods < 100:
                logger.warning(f"  ⚠ Split {i+1}: Too few periods (train={train_periods}, test={test_periods})")
                break

            splits.append((
                timestamps[train_start_idx],
                timestamps[train_end_idx - 1],
                timestamps[test_end_idx - 1]
            ))

            train_days = (timestamps[train_end_idx - 1] - timestamps[train_start_idx]) / np.timedelta64(1, 'D')
            test_days = (timestamps[test_end_idx - 1] - timestamps[train_end_idx - 1]) / np.timedelta64(1, 'D')
            logger.info(f"  ✓ Split {i+1}: Train {train_days:.0f}d ({train_periods:,} periods) → Test {test_days:.0f}d ({test_periods:,} periods)")

        return splits

    def optimize(self, output_dir: str = "optimization_results") -> Dict[str, Any]:
        """Run enhanced optimization with validation."""
        from optimize_portfolio_comprehensive import ComprehensiveOptimizer

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info("\n" + "=" * 80)
        logger.info("🚀 ENHANCED PORTFOLIO OPTIMIZATION")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Set defaults if not provided
        if self.window_days is None:
            self.window_days = 365 if not self.auto_adjust else None
        if self.test_windows is None:
            self.test_windows = 5

        # If auto_adjust and no manual params, suggest optimal
        if self.auto_adjust and self.window_days is None:
            logger.info("\n🔄 AUTO-ADJUST MODE: Analyzing data to find optimal parameters...")

        # Validate and prepare data
        price_arrays, timestamp_arrays, splits, is_valid = self.validate_and_prepare_data(
            self.window_days, self.test_windows
        )

        if not is_valid:
            logger.error("\n❌ DATA VALIDATION FAILED - Cannot proceed with optimization")
            sys.exit(1)

        # Continue with optimization using validated data
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: CONFIGURATION GENERATION")
        logger.info("=" * 80)

        from optimize_portfolio_optimized import OptimizedOptimizer
        opt = OptimizedOptimizer(
            window_days=self.window_days,
            timeframe=self.timeframe,
            test_windows=self.test_windows,
            quick_mode=self.quick_mode,
            workers=self.workers
        )

        asset_combinations = opt.get_asset_combinations()
        rebalance_grid = opt.get_rebalancing_parameters()
        rebalance_param_names = list(rebalance_grid.keys())
        rebalance_combinations = list(product(*rebalance_grid.values()))

        all_configs = []
        config_id = 0

        for asset_combo in asset_combinations:
            valid_assets = [a for a in asset_combo if a in price_arrays]
            if len(valid_assets) < 2:
                continue

            weight_schemes = opt.get_weight_schemes(len(valid_assets))

            for weights in weight_schemes:
                assets = list(zip(valid_assets, weights))

                for rebalance_combo in rebalance_combinations:
                    rebalance_params = dict(zip(rebalance_param_names, rebalance_combo))
                    config_id += 1
                    all_configs.append((config_id, assets, rebalance_params))

        logger.info(f"  Total configurations: {len(all_configs)}")
        logger.info(f"  Total backtests: {len(all_configs) * len(splits) * 2:,}")

        # Run optimization
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5: PARALLEL OPTIMIZATION")
        logger.info("=" * 80)

        with mp.Pool(
            processes=self.workers,
            initializer=worker_init,
            initargs=(price_arrays, timestamp_arrays, splits, self.timeframe)
        ) as pool:

            results_list = list(tqdm(
                pool.imap_unordered(process_configuration, all_configs),
                total=len(all_configs),
                desc="Optimizing",
                unit="config",
                ncols=100
            ))

        all_results = [r for r in results_list if r is not None]

        duration = (datetime.now() - start_time).total_seconds()
        logger.success(f"\n✅ OPTIMIZATION COMPLETE in {duration:.1f}s ({duration/60:.2f} min)")
        logger.info(f"   Valid results: {len(all_results)}/{len(all_configs)}")

        if len(all_results) == 0:
            logger.error("\n❌ NO VALID RESULTS - All configurations failed")
            logger.error("   This should not happen after validation")
            sys.exit(1)

        # Analyze results
        all_results.sort(key=lambda x: x['test_avg_outperformance'], reverse=True)
        best = all_results[0]

        logger.info("\n🏆 BEST CONFIGURATION:")
        logger.info(f"  Assets: {', '.join(best['assets'])}")
        logger.info(f"  Weights: {[f'{w:.1%}' for w in best['weights']]}")
        logger.info(f"  Test Outperformance: {best['test_avg_outperformance']:.2%}")
        logger.info(f"  Test Sharpe: {best['test_avg_sharpe']:.3f}")

        # Generate reports
        serial_optimizer = ComprehensiveOptimizer(
            window_days=self.window_days,
            timeframe=self.timeframe,
            test_windows=self.test_windows,
            quick_mode=self.quick_mode
        )

        serial_optimizer._generate_research_report(all_results, splits, output_path)
        serial_optimizer._generate_optimized_config(best, output_path)
        serial_optimizer._save_detailed_results(all_results, output_path)

        logger.success(f"\n✅ Results saved to: {output_path}")

        return {
            'best_config': best,
            'all_results': all_results,
            'splits': splits,
            'duration': duration
        }


@app.command()
def optimize(
    window_days: Optional[int] = typer.Option(None, "--window-days", "-d"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t"),
    test_windows: Optional[int] = typer.Option(None, "--test-windows", "-n"),
    quick_mode: bool = typer.Option(False, "--quick", "-q"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w"),
    output_dir: str = typer.Option("optimization_results", "--output", "-o"),
    auto_adjust: bool = typer.Option(False, "--auto-adjust", "-a"),
    check_data_only: bool = typer.Option(False, "--check-data-only")
):
    """
    Run ENHANCED portfolio optimization with comprehensive data validation.

    Examples:
        # Auto-adjust to available data (RECOMMENDED):
        python optimize_portfolio_enhanced.py --auto-adjust --quick

        # Manual parameters:
        python optimize_portfolio_enhanced.py --window-days 350 --test-windows 5 --quick

        # Check data availability only:
        python optimize_portfolio_enhanced.py --check-data-only
    """

    if check_data_only:
        # Just run data analysis and exit
        fetcher = BinanceDataFetcher()
        asset_universe = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
            "ADA/USDT", "XRP/USDT", "MATIC/USDT", "DOT/USDT",
        ]
        report = DataAvailabilityReport(asset_universe, timeframe, fetcher)
        analysis = report.analyze()
        report.suggest_parameters(analysis['common_periods'], timeframe, test_windows or 5)
        return

    # Set defaults
    if window_days is None and not auto_adjust:
        window_days = 365
    if test_windows is None:
        test_windows = 5

    optimizer = EnhancedOptimizer(
        window_days=window_days,
        timeframe=timeframe,
        test_windows=test_windows,
        quick_mode=quick_mode,
        workers=workers,
        auto_adjust=auto_adjust
    )

    try:
        result = optimizer.optimize(output_dir=output_dir)
        logger.success("\n✅ ENHANCED OPTIMIZATION COMPLETE!")

    except Exception as e:
        logger.error(f"\n❌ Optimization failed: {e}")
        logger.exception("Full traceback:")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
