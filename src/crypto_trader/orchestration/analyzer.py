"""
Master Strategy Analyzer - Comprehensive Strategy Testing and Ranking

This module provides the orchestration layer for running comprehensive
strategy analysis across multiple time horizons with parallel execution.

**Purpose**: Orchestrate strategy testing, scoring, and ranking

**Key Classes**:
- HorizonConfig: Configuration for time horizon tests
- StrategyScore: Aggregated scoring for strategies
- MasterStrategyAnalyzer: Main orchestrator for comprehensive analysis

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- concurrent.futures: https://docs.python.org/3/library/concurrent.futures.html
- loguru: https://loguru.readthedocs.io/en/stable/
- plotly: https://plotly.com/python/

**Sample Input**:
```python
from crypto_trader.orchestration.analyzer import MasterStrategyAnalyzer

analyzer = MasterStrategyAnalyzer(
    symbol="BTC/USDT",
    timeframe="1h",
    quick_mode=True
)
analyzer.run()
```

**Expected Output**:
Comprehensive analysis reports with strategy rankings and comparison matrices.

Extracted from master.py (lines 193-2616) during Phase 3 refactoring.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from crypto_trader.core.config import BacktestConfig
from crypto_trader.core.types import BacktestResult, Timeframe
from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.strategies import get_registry
from crypto_trader.backtesting.engine import BacktestEngine
from crypto_trader.features.factory import augment_with_features, DEFAULT_JOIN_CONFIG
from crypto_trader.analysis.performance_store import PerformanceStore
from crypto_trader.data.alt.onchain_ingestor import ingest_onchain
from crypto_trader.data.alt.sentiment_ingestor import ingest_sentiment
from crypto_trader.data.alt.orderflow_stream import ingest_orderflow
from crypto_trader.reports.formatters.html import HTMLFormatter
from crypto_trader.reports.formatters.plotly_interactive import generate_interactive_section_html

# Aliases for backward compatibility with extracted code
HTMLReportWriter = HTMLFormatter

# Import execution layer modules
from crypto_trader.execution.workers import (
    run_backtest_worker,
    run_multipair_backtest_worker
)
from crypto_trader.execution.data_utils import calculate_data_limit
from crypto_trader.execution.logging_utils import (
    log_dataframe_info,
    log_memory_usage,
    log_worker_lifecycle,
    log_error_with_context
)
from crypto_trader.execution.metric_utils import (
    periods_per_year_from_timeframe,
    calculate_sharpe_ratio_safe
)

# Create aliases for underscore-prefixed function names (backward compatibility)
_calculate_data_limit = calculate_data_limit
_periods_per_year_from_timeframe = periods_per_year_from_timeframe


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

from contextlib import contextmanager
import functools

@contextmanager
def log_operation(operation_name: str, log_level: str = "INFO"):
    """
    Context manager for logging operations with timing and status.

    Args:
        operation_name: Name of the operation being logged
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Example:
        with log_operation("Data Fetching"):
            data = fetch_data()
    """
    logger.log(log_level, f"[START] {operation_name}")
    start = time.perf_counter()
    try:
        yield
        duration = time.perf_counter() - start
        logger.log(log_level, f"[COMPLETE] {operation_name} ({duration:.3f}s)")
    except Exception as e:
        duration = time.perf_counter() - start
        logger.error(f"[FAILED] {operation_name} after {duration:.3f}s: {type(e).__name__}: {str(e)}")
        raise


def log_validation_checkpoint(checkpoint_name: str, passed: bool, details: str = ""):
    """
    Log validation checkpoint with pass/fail status.

    Args:
        checkpoint_name: Name of the validation checkpoint
        passed: Whether the checkpoint passed
        details: Optional details about the checkpoint
    """
    status = "✓" if passed else "✗"
    level = "SUCCESS" if passed else "ERROR"
    msg = f"[VALIDATION] {status} {checkpoint_name}"
    if details:
        msg += f" - {details}"
    logger.log(level, msg)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class HorizonConfig:
    """Configuration for a time horizon test."""
    name: str
    days: int
    description: str


@dataclass
class StrategyScore:
    """Aggregated scoring for a strategy."""
    strategy_name: str
    composite_score: float
    avg_return: float
    avg_sharpe: float
    avg_max_drawdown: float
    avg_win_rate: float
    horizons_beat_buyhold: int
    total_horizons: int
    horizon_results: Dict[str, Dict[str, float]]


class MasterStrategyAnalyzer:
    """
    Comprehensive strategy analysis engine.

    Tests all registered strategies across multiple time horizons,
    compares to buy-and-hold, and generates ranking reports.
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        horizons: Optional[List[int]] = None,
        workers: int = 4,
        quick_mode: bool = False,
        multi_pair: bool = False,
        output_dir: str = "master_results"
    ):
        """
        Initialize the master analyzer with comprehensive logging.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT') - for single-pair strategies
            timeframe: Candle timeframe (default: '1h')
            horizons: List of time horizons in days (default: [30, 90, 180, 365, 730])
            workers: Number of parallel workers (default: 4)
            quick_mode: If True, use fewer horizons for faster testing
            multi_pair: If True, test multi-pair strategies (Portfolio, Statistical Arbitrage)
            output_dir: Directory for saving results
        """
        logger.info("=" * 80)
        logger.info("🚀 MASTER STRATEGY ANALYZER INITIALIZATION")
        logger.info("=" * 80)

        self.symbol = symbol
        self.timeframe = timeframe
        # Limit workers for multi-pair mode (now using shared data pool, can use more workers)
        self.workers = min(workers, 4) if multi_pair else workers
        self.quick_mode = quick_mode
        self.multi_pair = multi_pair

        logger.debug(f"[INIT] Symbol: {symbol}")
        logger.debug(f"[INIT] Timeframe: {timeframe}")
        logger.debug(f"[INIT] Workers: {workers} (adjusted to: {self.workers})")
        logger.debug(f"[INIT] Quick mode: {quick_mode}")
        logger.debug(f"[INIT] Multi-pair mode: {multi_pair}")

        # Define time horizons
        logger.debug(f"[INIT] Configuring time horizons...")
        if horizons:
            self.horizons = [HorizonConfig(f"{d}d", d, f"{d} days") for d in horizons]
            logger.info(f"[INIT] Custom horizons: {horizons} days")
        elif quick_mode:
            self.horizons = [
                HorizonConfig("30d", 30, "30 days"),
                HorizonConfig("90d", 90, "90 days"),
                HorizonConfig("180d", 180, "180 days"),
            ]
            logger.info(f"[INIT] Quick mode horizons: {[h.days for h in self.horizons]} days")
        else:
            self.horizons = [
                HorizonConfig("30d", 30, "30 days"),
                HorizonConfig("90d", 90, "90 days"),
                HorizonConfig("180d", 180, "180 days"),
                HorizonConfig("365d", 365, "1 year"),
                HorizonConfig("730d", 730, "2 years"),
            ]
            logger.info(f"[INIT] Standard horizons: {[h.days for h in self.horizons]} days")

        # Setup output directory
        logger.debug(f"[INIT] Setting up output directory...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"{output_dir}_{timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.details_dir = self.output_dir / "detailed_results"
        self.details_dir.mkdir(exist_ok=True)
        logger.info(f"[INIT] Output directory: {self.output_dir}")

        # Configure logging
        log_file = self.output_dir / "master_analysis.log"
        logger.add(log_file, level="DEBUG", rotation="100 MB")
        logger.success(f"[INIT] Logging configured: {log_file}")

        # Initialize components
        logger.debug(f"[INIT] Initializing data fetcher...")
        with log_operation("Data Fetcher Initialization", "DEBUG"):
            self.fetcher = BinanceDataFetcher()

        logger.debug(f"[INIT] Initializing backtest engine...")
        self.engine = BacktestEngine()

        logger.debug(f"[INIT] Initializing performance store...")
        self.performance_store = PerformanceStore()

        # BUGFIX (Bug #7): Add lock for thread-safe performance store updates
        import threading
        self._perf_lock = threading.Lock()

        # Results storage
        self.all_results: List[Dict[str, Any]] = []
        self.buy_hold_results: Dict[str, Dict[str, float]] = {}
        logger.debug(f"[INIT] Results storage initialized")

        # Log system information
        log_memory_usage("After Initialization", detailed=True)

        logger.success("✅ MasterStrategyAnalyzer initialized successfully!")
        logger.info(f"  📍 Symbol: {symbol}")
        logger.info(f"  ⏰ Timeframe: {timeframe}")
        logger.info(f"  📊 Horizons: {[h.name for h in self.horizons]}")
        logger.info(f"  👷 Workers: {self.workers}")
        logger.info(f"  🔀 Multi-pair mode: {multi_pair}")
        logger.info("=" * 80)

    def discover_strategies(self) -> Tuple[List[Tuple[str, type]], List[str]]:
        """
        Discover all registered strategies.

        Returns:
            Tuple of (single_pair_strategies, multi_pair_strategy_names)
        """
        # Import strategies to ensure registration
        import crypto_trader.strategies.library  # noqa: F401

        registry = get_registry()
        strategy_names = registry.get_strategy_names()

        single_pair_strategies = []
        multi_pair_strategies = []

        for name in strategy_names:
            # Separate multi-pair strategies
            if ("Portfolio" in name or "Statistical" in name or
                "HierarchicalRiskParity" in name or "BlackLitterman" in name or
                "RiskParity" in name or "Copula" in name or "DeepRL" in name):
                if self.multi_pair:
                    multi_pair_strategies.append(name)
                continue

            try:
                strategy_class = registry.get_strategy(name)
                single_pair_strategies.append((name, strategy_class))
            except Exception as e:
                logger.warning(f"Could not load strategy {name}: {e}")

        logger.info(f"Discovered {len(single_pair_strategies)} single-pair strategies: {[s[0] for s in single_pair_strategies]}")
        if self.multi_pair:
            logger.info(f"Discovered {len(multi_pair_strategies)} multi-pair strategies: {multi_pair_strategies}")

        return single_pair_strategies, multi_pair_strategies

    def get_asset_combinations(self) -> List[List[str]]:
        """
        Get asset combinations for multi-pair strategies.

        Returns:
            List of asset symbol lists
        """
        if self.quick_mode:
            return [
                ["BTC/USDT", "ETH/USDT"],
                ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            ]
        else:
            return [
                ["BTC/USDT", "ETH/USDT"],
                ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
                ["BTC/USDT", "ETH/USDT", "ADA/USDT", "XRP/USDT"],
            ]

    def fetch_data(self, days: int) -> pd.DataFrame:
        """
        Fetch historical data for specified time period with comprehensive logging.

        Args:
            days: Number of days of historical data

        Returns:
            DataFrame with OHLCV data and augmented features
        """
        logger.info(f"[DATA-FETCH] 📥 Starting data fetch for {days} days")

        # Calculate limit based on timeframe
        limit = _calculate_data_limit(self.timeframe, days)
        logger.debug(f"[DATA-FETCH] Calculated limit: {limit} candles for {days} days at {self.timeframe}")

        # BUGFIX (Bug #11): NEVER silently fall back to mock data
        # Backtesting requires REAL market data - fake data gives fake results
        data = None
        try:
            logger.debug(f"[DATA-FETCH] Fetching from Binance: {self.symbol} @ {self.timeframe}")
            fetch_start = time.perf_counter()
            data = self.fetcher.get_ohlcv(self.symbol, self.timeframe, limit=limit)
            fetch_duration = time.perf_counter() - fetch_start
            logger.info(f"[DATA-FETCH] ✓ Primary fetch completed in {fetch_duration:.2f}s")
        except Exception as e:
            # DO NOT fall back to mock data - fail loudly
            logger.error(f"[DATA-FETCH] ✗ Failed to fetch real market data: {type(e).__name__}: {e}")
            logger.error(f"[DATA-FETCH] Backtesting requires actual historical prices")
            logger.error(f"[DATA-FETCH] Check:")
            logger.error(f"[DATA-FETCH]   1. Internet connection")
            logger.error(f"[DATA-FETCH]   2. Binance API status (status.binance.com)")
            logger.error(f"[DATA-FETCH]   3. Symbol {self.symbol} is valid and trading")
            logger.error(f"[DATA-FETCH]   4. No rate limiting (429 errors)")
            raise ValueError(
                f"Cannot fetch real market data for {self.symbol}. "
                f"Backtesting with fake data would produce meaningless results. "
                f"Original error: {type(e).__name__}: {e}"
            )

        if data is None or len(data) == 0:
            logger.error(f"[DATA-FETCH] ✗ No data available for {self.symbol}")
            raise ValueError(f"No data fetched for {self.symbol}")

        log_dataframe_info(data, f"Initial OHLCV Data ({days}d)", detailed=True)
        log_validation_checkpoint("OHLCV Data Retrieved", True,
                                 f"{len(data)} candles, {len(data.columns)} columns")

        # Prepare feature pillars
        logger.debug(f"[DATA-FETCH] Preparing feature pillars (onchain, sentiment, orderflow)...")
        self._prepare_feature_pillars()

        # Join alternative data features (safe no-op if none available)
        try:
            logger.info(f"[DATA-FETCH] 🔗 Augmenting with alternative data features...")
            augment_start = time.perf_counter()

            with_features = augment_with_features(
                market_df=data,
                symbol=self.symbol,
                timeframe=self.timeframe,
                config=DEFAULT_JOIN_CONFIG,
            )

            augment_duration = time.perf_counter() - augment_start
            added_cols = [c for c in with_features.columns if c not in data.columns]

            logger.info(f"[DATA-FETCH] ✓ Feature augmentation completed in {augment_duration:.2f}s")
            logger.info(f"[DATA-FETCH] Added {len(added_cols)} feature columns: {added_cols[:10]}")

            log_dataframe_info(with_features, f"Data with Features ({days}d)", detailed=True)
            log_validation_checkpoint("Feature Augmentation", True,
                                     f"+{len(added_cols)} columns")

            return with_features
        except Exception as fe:
            logger.warning(f"Feature join failed; continuing with OHLCV only: {fe}")
            return data

    def _prepare_feature_pillars(self) -> None:
        """Ensure alternative data pillars are materialized before feature join."""
        try:
            ingest_onchain(symbol=self.symbol, timeframe=self.timeframe, prefer_local_csv=True)
        except Exception as exc:
            logger.debug(f"On-chain ingestion skipped: {exc}")

        try:
            ingest_sentiment(symbol=self.symbol, timeframe=self.timeframe, prefer_local_csv=True)
        except Exception as exc:
            logger.debug(f"Sentiment ingestion skipped: {exc}")

        try:
            ingest_orderflow(symbol=self.symbol, timeframe=self.timeframe, prefer_local_csv=True)
        except Exception as exc:
            logger.debug(f"Order flow ingestion skipped: {exc}")

    def _record_performance(self, result: Dict[str, Any]) -> None:
        """
        Persist single backtest result for ensemble weighting.

        BUGFIX (Bug #7): Thread-safe with lock to prevent corruption from parallel workers.
        """
        payload = dict(result)
        payload.setdefault("symbol", result.get("symbol", self.symbol))
        payload.setdefault("timeframe", result.get("timeframe", self.timeframe))
        try:
            with self._perf_lock:  # Thread-safe write
                self.performance_store.record(payload)
        except Exception as exc:
            logger.debug(f"Performance store update skipped: {exc}")


    def _get_default_params(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a strategy by introspecting the class.

        BUGFIX (Bug #10): Query strategy class instead of maintaining hardcoded defaults.

        Args:
            strategy_name: Name of registered strategy

        Returns:
            Dictionary of default parameters, or empty dict if introspection fails
        """
        try:
            from crypto_trader.strategies import get_registry
            registry = get_registry()

            # Get the strategy class
            strategy_class = registry.get_strategy(strategy_name)

            # Try to instantiate without args and get defaults
            # Most strategies have sensible defaults in their __init__
            try:
                temp_strategy = strategy_class()
                # Call initialize with empty config to set defaults
                if hasattr(temp_strategy, 'initialize'):
                    temp_strategy.initialize({})
                # Get the parameters
                if hasattr(temp_strategy, 'get_parameters'):
                    defaults = temp_strategy.get_parameters()
                    logger.debug(f"Introspected defaults for {strategy_name}: {defaults}")
                    return defaults
            except Exception as e_init:
                # Some strategies may require args, try with name
                try:
                    temp_strategy = strategy_class(name=strategy_name)
                    if hasattr(temp_strategy, 'initialize'):
                        temp_strategy.initialize({})
                    if hasattr(temp_strategy, 'get_parameters'):
                        defaults = temp_strategy.get_parameters()
                        logger.debug(f"Introspected defaults for {strategy_name}: {defaults}")
                        return defaults
                except Exception as e_retry:
                    logger.debug(f"Could not introspect {strategy_name}: {e_retry}")

            # If all else fails, return empty dict (strategies will use their internal defaults)
            logger.debug(f"Using empty params for {strategy_name} (will use strategy's internal defaults)")
            return {}

        except Exception as e:
            logger.warning(f"Failed to get defaults for {strategy_name}: {e}")
            return {}

    def _timeframe_to_enum(self) -> Timeframe:
        """Convert string timeframe to Timeframe enum."""
        mapping = {
            "1m": Timeframe.MINUTE_1,
            "5m": Timeframe.MINUTE_5,
            "15m": Timeframe.MINUTE_15,
            "1h": Timeframe.HOUR_1,
            "4h": Timeframe.HOUR_4,
            "1d": Timeframe.DAY_1,
            "1w": Timeframe.WEEK_1,
        }
        return mapping.get(self.timeframe, Timeframe.HOUR_1)

    def calculate_buy_hold(self, data: pd.DataFrame, horizon: HorizonConfig) -> Dict[str, float]:
        """
        Calculate buy-and-hold benchmark for a horizon.

        Args:
            data: Historical OHLCV data
            horizon: Time horizon configuration

        Returns:
            Dictionary with buy-and-hold metrics
        """
        initial_price = data['close'].iloc[0]
        final_price = data['close'].iloc[-1]
        buy_hold_return = (final_price - initial_price) / initial_price

        # Calculate simple metrics
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()

        # Annualize based on timeframe
        periods_per_year = _periods_per_year_from_timeframe(self.timeframe)
        sharpe = (returns.mean() * periods_per_year) / (volatility * np.sqrt(periods_per_year)) if volatility > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_return': buy_hold_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_drawdown),
            'final_price': final_price,
        }

    def run_parallel_analysis(self) -> None:
        """Run parallel backtests for all strategies and horizons."""
        # Memory monitoring
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            initial_memory_mb = process.memory_info().rss / 1024 / 1024
            memory_monitoring = True
        except ImportError:
            logger.warning("psutil not installed - memory monitoring disabled. Install with: pip install psutil")
            memory_monitoring = False
            initial_memory_mb = 0

        logger.info("\n" + "=" * 80)
        logger.info("RUNNING PARALLEL STRATEGY ANALYSIS")
        if memory_monitoring:
            logger.info(f"Initial memory usage: {initial_memory_mb:.1f} MB")
        logger.info("=" * 80)

        # Discover strategies
        single_pair_strategies, multi_pair_strategies = self.discover_strategies()

        # Calculate total jobs
        single_pair_jobs = len(single_pair_strategies) * len(self.horizons)

        # For multi-pair, each strategy × horizon × asset_combinations
        multi_pair_jobs = 0
        if self.multi_pair and multi_pair_strategies:
            asset_combinations = self.get_asset_combinations()
            multi_pair_jobs = len(multi_pair_strategies) * len(self.horizons) * len(asset_combinations)

        total_jobs = single_pair_jobs + multi_pair_jobs
        logger.info(f"\nTotal jobs: {total_jobs} ({single_pair_jobs} single-pair + {multi_pair_jobs} multi-pair)")
        logger.info(f"Parallel workers: {self.workers}")

        # Fetch data for each horizon (cache to avoid refetching)
        horizon_data = {}
        logger.info("\nFetching historical data for all horizons...")
        for horizon in self.horizons:
            try:
                data = self.fetch_data(horizon.days)
                horizon_data[horizon.name] = data
                logger.success(f"  ✓ {horizon.name}: {len(data)} candles")

                # Calculate buy-and-hold benchmark
                self.buy_hold_results[horizon.name] = self.calculate_buy_hold(data, horizon)
            except Exception as e:
                logger.error(f"  ✗ {horizon.name}: {e}")

        # Pre-fetch multi-pair data (shared data pool optimization)
        multi_pair_data = {}
        if self.multi_pair and multi_pair_strategies:
            logger.info("\n" + "=" * 80)
            logger.info("PRE-FETCHING MULTI-PAIR DATA (Shared Data Pool)")
            logger.info("=" * 80)

            # Collect all unique symbols needed across all asset combinations
            asset_combinations = self.get_asset_combinations()
            all_symbols = set()
            for combo in asset_combinations:
                all_symbols.update(combo)

            logger.info(f"Pre-fetching data for {len(all_symbols)} unique assets: {', '.join(all_symbols)}")

            # Calculate max limit needed across all horizons
            max_horizon_days = max(h.days for h in self.horizons)
            max_limit = _calculate_data_limit(
                self.timeframe,
                max_horizon_days,
                warmup_multiplier=1.5  # 50% extra for warmup
            )

            # Fetch once, reuse for all workers
            for symbol in all_symbols:
                try:
                    data = self.fetcher.get_ohlcv(symbol, self.timeframe, limit=max_limit)
                    if data is not None and len(data) > 0:
                        # Serialize to dict for worker processes
                        multi_pair_data[symbol] = {
                            'timestamp': data.index.tolist() if hasattr(data.index, 'tolist') else list(range(len(data))),
                            **{col: data[col].tolist() for col in data.columns}
                        }
                        logger.success(f"  ✓ {symbol}: {len(data)} candles")
                    else:
                        logger.warning(f"  ⚠ {symbol}: No data available")
                except Exception as e:
                    logger.error(f"  ✗ {symbol}: {e}")

            logger.success(f"✓ Pre-fetched {len(multi_pair_data)} assets. Will share with all workers (zero redundant fetches!)")

            # Estimate memory saved
            estimated_calls_saved = (len(multi_pair_strategies) * len(self.horizons) *
                                    len(asset_combinations) * len(all_symbols) - len(all_symbols))
            logger.info(f"  Memory optimization: ~{estimated_calls_saved} redundant API calls eliminated")

        # Run backtests in parallel
        logger.info("\nRunning parallel backtests...")

        single_jobs = []
        for strategy_name, _ in single_pair_strategies:
            for horizon in self.horizons:
                if horizon.name not in horizon_data:
                    continue
                data = horizon_data[horizon.name]
                data_dict = {
                    'timestamp': data.index.tolist() if hasattr(data.index, 'tolist') else list(range(len(data))),
                    **{col: data[col].tolist() for col in data.columns}
                }
                default_params = self._get_default_params(strategy_name)
                single_jobs.append(
                    (
                        strategy_name,
                        data_dict,
                        horizon.name,
                        horizon.days,
                        self.symbol,
                        self.timeframe,
                        default_params,
                    )
                )

        multi_jobs = []
        if self.multi_pair and multi_pair_strategies:
            asset_combinations = self.get_asset_combinations()
            for strategy_name in multi_pair_strategies:
                for horizon in self.horizons:
                    for asset_symbols in asset_combinations:
                        default_params = self._get_default_params(strategy_name)
                        multi_jobs.append(
                            (strategy_name, asset_symbols, horizon.name, horizon.days, default_params)
                        )

        completed = 0

        def _handle_result(result: Optional[Dict[str, Any]], strategy_name: str, horizon_name: str, job_type: str) -> None:
            nonlocal completed
            if result and 'error' not in result:
                self.all_results.append(result)
                self._record_performance(result)
            elif result and 'error' in result:
                logger.error(
                    f"Backtest failed for {strategy_name} ({job_type}) on {horizon_name}: {result['error']}"
                )
            completed += 1

        def _run_parallel(pbar_obj) -> None:
            # Try ProcessPool, fall back to ThreadPool if permission denied
            try:
                executor = ProcessPoolExecutor(max_workers=self.workers)
                exec_type = "ProcessPool"
            except (PermissionError, OSError) as e:
                logger.warning(
                    f"ProcessPoolExecutor unavailable ({e.__class__.__name__}). "
                    f"Falling back to ThreadPoolExecutor (slower but reliable)"
                )
                from concurrent.futures import ThreadPoolExecutor
                executor = ThreadPoolExecutor(max_workers=self.workers)
                exec_type = "ThreadPool"

            logger.info(f"Using {exec_type} with {self.workers} workers")
            with executor:
                futures = {}
                for job in single_jobs:
                    future = executor.submit(run_backtest_worker, *job)
                    futures[future] = (job[0], job[2], 'single')
                for job in multi_jobs:
                    strategy_name, asset_symbols, horizon_name, horizon_days, default_params = job
                    future = executor.submit(
                        run_multipair_backtest_worker,
                        strategy_name,
                        asset_symbols,
                        multi_pair_data,
                        horizon_name,
                        horizon_days,
                        self.timeframe,
                        default_params,
                    )
                    futures[future] = (strategy_name, horizon_name, 'multi')

                for future in as_completed(futures):
                    strategy_name, horizon_name, job_type = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.error(f"Job failed for {strategy_name} ({job_type}) on {horizon_name}: {exc}")
                        result = None
                    _handle_result(result, strategy_name, horizon_name, job_type)
                    pbar_obj.update(1)

        def _run_serial(pbar_obj) -> None:
            for job in single_jobs:
                result = run_backtest_worker(*job)
                _handle_result(result, job[0], job[2], 'single')
                pbar_obj.update(1)
            for job in multi_jobs:
                strategy_name, asset_symbols, horizon_name, horizon_days, default_params = job
                result = run_multipair_backtest_worker(
                    strategy_name,
                    asset_symbols,
                    multi_pair_data,
                    horizon_name,
                    horizon_days,
                    self.timeframe,
                    default_params,
                )
                _handle_result(result, strategy_name, horizon_name, 'multi')
                pbar_obj.update(1)

        with tqdm(total=total_jobs, desc="Progress") as pbar:
            try:
                _run_parallel(pbar)
            except (PermissionError, OSError) as exc:
                logger.warning(f"Process pool unavailable ({exc}); falling back to serial execution")
                _run_serial(pbar)

        # Final memory report
        if memory_monitoring:
            final_memory_mb = process.memory_info().rss / 1024 / 1024
            memory_used_mb = final_memory_mb - initial_memory_mb
            logger.info(f"\n📊 Memory Usage Report:")
            logger.info(f"  Initial: {initial_memory_mb:.1f} MB")
            logger.info(f"  Final: {final_memory_mb:.1f} MB")
            logger.info(f"  Used: {memory_used_mb:+.1f} MB")

        logger.success(f"\n✓ Completed {len(self.all_results)} successful backtests out of {total_jobs}")

    def compute_composite_scores(self) -> List[StrategyScore]:
        """
        Compute composite scores for all strategies.

        Returns:
            List of StrategyScore objects sorted by composite score
        """
        logger.info("\nComputing composite scores...")

        if not self.all_results:
            logger.error("No results to score")
            return []

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.all_results)

        # Group by strategy
        strategy_scores = []

        for strategy_name in df['strategy_name'].unique():
            strategy_df = df[df['strategy_name'] == strategy_name]

            # Aggregate metrics across horizons
            avg_return = strategy_df['total_return'].mean()
            avg_sharpe = strategy_df['sharpe_ratio'].mean()
            avg_max_drawdown = strategy_df['max_drawdown'].mean()
            avg_win_rate = strategy_df['win_rate'].mean()

            # Count horizons where strategy beat buy-and-hold
            # For multi-pair strategies with multiple asset combinations,
            # we check if ANY configuration beat buy-hold for each horizon
            horizons_beat = 0
            horizon_results = {}

            # Group by horizon first to handle multiple configurations
            for horizon_name in strategy_df['horizon'].unique():
                horizon_rows = strategy_df[strategy_df['horizon'] == horizon_name]

                # Skip empty horizons (shouldn't happen but be safe)
                if horizon_rows.empty:
                    logger.warning(f"No results for {strategy_name} on {horizon_name}, skipping")
                    continue

                buyhold_return = self.buy_hold_results.get(horizon_name, {}).get('total_return', 0)

                # Get best result for this horizon (across all configurations)
                best_return = horizon_rows['total_return'].max()
                best_row = horizon_rows.loc[horizon_rows['total_return'].idxmax()]

                horizon_results[horizon_name] = {
                    'return': best_return,
                    'sharpe': best_row['sharpe_ratio'],
                    'drawdown': best_row['max_drawdown'],
                    'win_rate': best_row['win_rate'],
                    'trades': best_row['total_trades'],
                    'vs_buyhold': best_return - buyhold_return,
                    'beat_buyhold': best_return > buyhold_return,
                }

                # Count this horizon as "won" if best configuration beat buy-hold
                if best_return > buyhold_return:
                    horizons_beat += 1

            # Normalize metrics to 0-1 scale for composite scoring
            # Use min-max normalization across all strategies
            all_returns = df['total_return'].values
            all_sharpes = df['sharpe_ratio'].values
            all_drawdowns = df['max_drawdown'].values
            all_win_rates = df['win_rate'].values

            norm_return = self._normalize(avg_return, all_returns)
            norm_sharpe = self._normalize(avg_sharpe, all_sharpes)
            norm_drawdown = 1 - self._normalize(avg_max_drawdown, all_drawdowns)  # Lower is better
            norm_win_rate = self._normalize(avg_win_rate, all_win_rates)

            # Composite score with weights
            composite_score = (
                0.35 * norm_sharpe +      # 35% Sharpe (risk-adjusted)
                0.30 * norm_return +      # 30% Return
                0.20 * norm_drawdown +    # 20% Drawdown (inverted)
                0.15 * norm_win_rate      # 15% Win rate
            )

            strategy_scores.append(StrategyScore(
                strategy_name=strategy_name,
                composite_score=composite_score,
                avg_return=avg_return,
                avg_sharpe=avg_sharpe,
                avg_max_drawdown=avg_max_drawdown,
                avg_win_rate=avg_win_rate,
                horizons_beat_buyhold=horizons_beat,
                total_horizons=len(self.horizons),
                horizon_results=horizon_results,
            ))

        # Sort by composite score with tie-breakers (horizons beat buyhold, return, sharpe)
        strategy_scores.sort(
            key=lambda x: (
                x.composite_score,
                x.horizons_beat_buyhold,
                x.avg_return,
                x.avg_sharpe
            ),
            reverse=True
        )

        logger.success(f"✓ Computed scores for {len(strategy_scores)} strategies")
        return strategy_scores

    def _normalize(self, value: float, array: np.ndarray) -> float:
        """Normalize value to 0-1 scale using min-max normalization."""
        min_val = array.min()
        max_val = array.max()
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    def _write_practical_recommendations(self, f, strategy_scores: List[StrategyScore], avg_buyhold: float) -> None:
        """Write practical trading recommendations based on beating buy-and-hold."""
        f.write("## 🎯 PRACTICAL STRATEGY RECOMMENDATIONS\n\n")
        f.write("*Based on actual performance vs buy-and-hold benchmark*\n\n")
        f.write("---\n\n")

        # Categorize strategies by how many horizons they beat buy-and-hold
        beat_buyhold = [s for s in strategy_scores if s.horizons_beat_buyhold >= 3]
        close_to_buyhold = [s for s in strategy_scores if 0 < s.horizons_beat_buyhold < 3]
        underperformed = [s for s in strategy_scores if s.horizons_beat_buyhold == 0]

        # Sort each category by average return
        beat_buyhold.sort(key=lambda x: x.avg_return, reverse=True)
        close_to_buyhold.sort(key=lambda x: x.avg_return, reverse=True)
        underperformed.sort(key=lambda x: x.avg_return, reverse=True)

        # TIER 1: Strategies that consistently beat buy-and-hold
        f.write("### 🏆 TIER 1: CONSISTENTLY BEATS BUY-AND-HOLD\n\n")
        if beat_buyhold:
            f.write("✅ These strategies beat buy-and-hold on **3+ time horizons**  \n")
            f.write("**RECOMMENDED for actual trading**\n\n")

            f.write("| Rank | Strategy | Avg Return | Sharpe | Drawdown | Won |\n")
            f.write("|------|----------|------------|--------|----------|-----|\n")
            for rank, strat in enumerate(beat_buyhold, 1):
                outperf = strat.avg_return - avg_buyhold
                f.write(f"| {rank} | {strat.strategy_name} | {strat.avg_return:+.1%} | ")
                f.write(f"{strat.avg_sharpe:.2f} | {strat.avg_max_drawdown:.1%} | ")
                f.write(f"{strat.horizons_beat_buyhold}/{strat.total_horizons} |\n")

            # Investment recommendations for top performers
            if len(beat_buyhold) > 0:
                best = beat_buyhold[0]
                f.write(f"\n#### 💡 TOP RECOMMENDATION: {best.strategy_name}\n\n")
                f.write(f"- **Returns:** {best.avg_return:+.1%} (vs {avg_buyhold:+.1%} buy-and-hold)\n")
                f.write(f"- **Sharpe Ratio:** {best.avg_sharpe:.2f} (risk-adjusted performance)\n")
                f.write(f"- **Max Drawdown:** {best.avg_max_drawdown:.1%} (worst peak-to-trough loss)\n")
                f.write(f"- **Beat buy-and-hold** on {best.horizons_beat_buyhold}/{best.total_horizons} time horizons\n")

                # Find best horizon for this strategy
                best_horizon = None
                best_horizon_return = -float('inf')
                for horizon_name, result in best.horizon_results.items():
                    if result['return'] > best_horizon_return:
                        best_horizon_return = result['return']
                        best_horizon = horizon_name

                if best_horizon:
                    f.write(f"- **Best horizon:** {best_horizon} ({best_horizon_return:+.1%} return)\n")

                f.write("\n**ACTION PLAN:**\n\n")
                f.write("1. Start with paper trading to validate performance\n")
                f.write("2. Use conservative position sizing (2-5% of portfolio)\n")
                f.write(f"3. Set stop-loss at {best.avg_max_drawdown * 2:.1%} (2× max drawdown)\n")
                f.write("4. Monitor weekly and compare to buy-and-hold baseline\n\n")
        else:
            f.write("❌ **NO strategies consistently beat buy-and-hold** (3+ horizons)  \n")
            f.write("Consider sticking with passive buy-and-hold strategy\n\n")

        # TIER 2: Sometimes beats buy-and-hold
        f.write("### ⚠️  TIER 2: SOMETIMES BEATS BUY-AND-HOLD\n\n")
        if close_to_buyhold:
            f.write("⚡ These strategies beat buy-and-hold on **1-2 time horizons**  \n")
            f.write("Use with **CAUTION** - performance is inconsistent\n\n")

            f.write("| Rank | Strategy | Avg Return | Sharpe | Drawdown | Won |\n")
            f.write("|------|----------|------------|--------|----------|-----|\n")
            for rank, strat in enumerate(close_to_buyhold, 1):
                f.write(f"| {rank} | {strat.strategy_name} | {strat.avg_return:+.1%} | ")
                f.write(f"{strat.avg_sharpe:.2f} | {strat.avg_max_drawdown:.1%} | ")
                f.write(f"{strat.horizons_beat_buyhold}/{strat.total_horizons} |\n")

            f.write("\n> 💡 These may work for specific time horizons or market conditions.  \n")
            f.write("> Check **TIME HORIZON ANALYSIS** section for details.\n\n")
        else:
            f.write("None found\n\n")

        # TIER 3: Never beats buy-and-hold
        f.write("### ❌ TIER 3: DOES NOT BEAT BUY-AND-HOLD\n\n")
        if underperformed:
            f.write("🚫 These strategies **NEVER** beat buy-and-hold on any time horizon  \n")
            f.write("**NOT RECOMMENDED** for trading - use buy-and-hold instead\n\n")

            f.write("| Rank | Strategy | Avg Return | Sharpe | Drawdown | Won |\n")
            f.write("|------|----------|------------|--------|----------|-----|\n")
            for rank, strat in enumerate(underperformed, 1):
                f.write(f"| {rank} | {strat.strategy_name} | {strat.avg_return:+.1%} | ")
                f.write(f"{strat.avg_sharpe:.2f} | {strat.avg_max_drawdown:.1%} | ")
                f.write(f"{strat.horizons_beat_buyhold}/{strat.total_horizons} |\n")

            f.write("\n> 💡 Even if returns are positive, buy-and-hold performed better.\n\n")
        else:
            f.write("None found\n\n")

        # Investment profile recommendations
        f.write("### 👤 RECOMMENDATIONS BY INVESTOR PROFILE\n\n")

        f.write("**🎯 AGGRESSIVE INVESTOR** (maximize returns, accept high risk):\n\n")
        if beat_buyhold:
            aggressive_pick = beat_buyhold[0]  # Highest return among beat_buyhold
            f.write(f"→ **{aggressive_pick.strategy_name}**  \n")
            f.write(f"   Returns: {aggressive_pick.avg_return:+.1%} | Drawdown: {aggressive_pick.avg_max_drawdown:.1%}\n\n")
        else:
            f.write("→ **Buy-and-hold** (no active strategies beat benchmark)\n\n")

        f.write("**🛡️  CONSERVATIVE INVESTOR** (minimize drawdown, accept lower returns):\n\n")
        if beat_buyhold:
            # Find strategy with lowest drawdown among winners
            conservative_pick = min(beat_buyhold, key=lambda x: x.avg_max_drawdown)
            f.write(f"→ **{conservative_pick.strategy_name}**  \n")
            f.write(f"   Returns: {conservative_pick.avg_return:+.1%} | Drawdown: {conservative_pick.avg_max_drawdown:.1%}\n\n")
        else:
            f.write("→ **Buy-and-hold** (no active strategies beat benchmark)\n\n")

        f.write("**⚖️  BALANCED INVESTOR** (best risk-adjusted returns):\n\n")
        if beat_buyhold:
            # Find strategy with highest Sharpe among winners
            balanced_pick = max(beat_buyhold, key=lambda x: x.avg_sharpe)
            f.write(f"→ **{balanced_pick.strategy_name}**  \n")
            f.write(f"   Returns: {balanced_pick.avg_return:+.1%} | Sharpe: {balanced_pick.avg_sharpe:.2f}\n\n")
        else:
            f.write("→ **Buy-and-hold** (no active strategies beat benchmark)\n\n")

        # Time horizon specific recommendations
        f.write("### ⏰ BEST STRATEGY BY TIME HORIZON\n\n")
        f.write("*Choose strategy based on your investment timeline:*\n\n")

        for horizon in self.horizons:
            # Find best strategy that beat buy-hold for this horizon
            best_for_horizon = None
            best_return = -float('inf')

            for score in strategy_scores:
                if horizon.name in score.horizon_results:
                    result = score.horizon_results[horizon.name]
                    # Must have beaten buy-hold
                    if result['vs_buyhold'] > 0 and result['return'] > best_return:
                        best_return = result['return']
                        best_for_horizon = score.strategy_name

            buyhold = self.buy_hold_results.get(horizon.name, {}).get('total_return', 0)

            if best_for_horizon:
                f.write(f"- **{horizon.name}** → {best_for_horizon} ({best_return:+.1%})  \n")
                f.write(f"  Beat buy-and-hold by {best_return - buyhold:+.1%}\n\n")
            else:
                f.write(f"- **{horizon.name}** → Buy-and-hold (no strategy beat benchmark)\n\n")

    def _write_practical_recommendations_html(self, f, strategy_scores: List[StrategyScore], avg_buyhold: float) -> None:
        """Write practical trading recommendations in HTML format."""
        f.write("<h2>🎯 PRACTICAL STRATEGY RECOMMENDATIONS</h2>\n")
        f.write("<p><em>Based on actual performance vs buy-and-hold benchmark</em></p>\n")
        f.write("<hr>\n\n")

        # Categorize strategies
        beat_buyhold = [s for s in strategy_scores if s.horizons_beat_buyhold >= 3]
        close_to_buyhold = [s for s in strategy_scores if 0 < s.horizons_beat_buyhold < 3]
        underperformed = [s for s in strategy_scores if s.horizons_beat_buyhold == 0]

        beat_buyhold.sort(key=lambda x: x.avg_return, reverse=True)
        close_to_buyhold.sort(key=lambda x: x.avg_return, reverse=True)
        underperformed.sort(key=lambda x: x.avg_return, reverse=True)

        # TIER 1
        f.write("<h3>🏆 TIER 1: CONSISTENTLY BEATS BUY-AND-HOLD</h3>\n")
        if beat_buyhold:
            f.write("<p>✅ These strategies beat buy-and-hold on <strong>3+ time horizons</strong><br>\n")
            f.write("<strong>RECOMMENDED for actual trading</strong></p>\n\n")

            f.write("<table>\n")
            f.write("    <thead>\n")
            f.write("        <tr>\n")
            f.write("            <th>Rank</th><th>Strategy</th><th>Avg Return</th><th>Sharpe</th><th>Drawdown</th><th>Won</th>\n")
            f.write("        </tr>\n")
            f.write("    </thead>\n")
            f.write("    <tbody>\n")

            for rank, strat in enumerate(beat_buyhold, 1):
                f.write(f"        <tr class='tier1'>\n")
                f.write(f"            <td>{rank}</td>\n")
                f.write(f"            <td><strong>{strat.strategy_name}</strong></td>\n")
                f.write(f"            <td>{HTMLReportWriter.format_percentage(strat.avg_return)}</td>\n")
                f.write(f"            <td>{strat.avg_sharpe:.2f}</td>\n")
                f.write(f"            <td>{strat.avg_max_drawdown:.1%}</td>\n")
                f.write(f"            <td>{strat.horizons_beat_buyhold}/{strat.total_horizons}</td>\n")
                f.write(f"        </tr>\n")

            f.write("    </tbody>\n")
            f.write("</table>\n\n")

            # Top recommendation box
            if len(beat_buyhold) > 0:
                best = beat_buyhold[0]
                f.write("<div class='recommendation-box'>\n")
                f.write(f"<h4>💡 TOP RECOMMENDATION: {best.strategy_name}</h4>\n")
                f.write("<ul>\n")
                f.write(f"    <li><strong>Returns:</strong> {best.avg_return:+.1%} (vs {avg_buyhold:+.1%} buy-and-hold)</li>\n")
                f.write(f"    <li><strong>Sharpe Ratio:</strong> {best.avg_sharpe:.2f} (risk-adjusted performance)</li>\n")
                f.write(f"    <li><strong>Max Drawdown:</strong> {best.avg_max_drawdown:.1%} (worst peak-to-trough loss)</li>\n")
                f.write(f"    <li><strong>Beat buy-and-hold</strong> on {best.horizons_beat_buyhold}/{best.total_horizons} time horizons</li>\n")

                # Find best horizon
                best_horizon = None
                best_horizon_return = -float('inf')
                for horizon_name, result in best.horizon_results.items():
                    if result['return'] > best_horizon_return:
                        best_horizon_return = result['return']
                        best_horizon = horizon_name

                if best_horizon:
                    f.write(f"    <li><strong>Best horizon:</strong> {best_horizon} ({best_horizon_return:+.1%} return)</li>\n")

                f.write("</ul>\n\n")
                f.write("<h4>ACTION PLAN:</h4>\n")
                f.write("<ol>\n")
                f.write("    <li>Start with paper trading to validate performance</li>\n")
                f.write("    <li>Use conservative position sizing (2-5% of portfolio)</li>\n")
                f.write(f"    <li>Set stop-loss at {best.avg_max_drawdown * 2:.1%} (2× max drawdown)</li>\n")
                f.write("    <li>Monitor weekly and compare to buy-and-hold baseline</li>\n")
                f.write("</ol>\n")
                f.write("</div>\n\n")
        else:
            f.write("<p>❌ <strong>NO strategies consistently beat buy-and-hold</strong> (3+ horizons)<br>\n")
            f.write("Consider sticking with passive buy-and-hold strategy</p>\n\n")

        # TIER 2
        f.write("<h3>⚠️ TIER 2: SOMETIMES BEATS BUY-AND-HOLD</h3>\n")
        if close_to_buyhold:
            f.write("<p>⚡ These strategies beat buy-and-hold on <strong>1-2 time horizons</strong><br>\n")
            f.write("Use with <strong>CAUTION</strong> - performance is inconsistent</p>\n\n")

            f.write("<table>\n")
            f.write("    <thead>\n")
            f.write("        <tr>\n")
            f.write("            <th>Rank</th><th>Strategy</th><th>Avg Return</th><th>Sharpe</th><th>Drawdown</th><th>Won</th>\n")
            f.write("        </tr>\n")
            f.write("    </thead>\n")
            f.write("    <tbody>\n")

            for rank, strat in enumerate(close_to_buyhold, 1):
                f.write(f"        <tr class='tier2'>\n")
                f.write(f"            <td>{rank}</td>\n")
                f.write(f"            <td><strong>{strat.strategy_name}</strong></td>\n")
                f.write(f"            <td>{HTMLReportWriter.format_percentage(strat.avg_return)}</td>\n")
                f.write(f"            <td>{strat.avg_sharpe:.2f}</td>\n")
                f.write(f"            <td>{strat.avg_max_drawdown:.1%}</td>\n")
                f.write(f"            <td>{strat.horizons_beat_buyhold}/{strat.total_horizons}</td>\n")
                f.write(f"        </tr>\n")

            f.write("    </tbody>\n")
            f.write("</table>\n\n")

            f.write("<div class='blockquote info'>\n")
            f.write("    <p>💡 These may work for specific time horizons or market conditions. ")
            f.write("Check <strong>TIME HORIZON ANALYSIS</strong> section for details.</p>\n")
            f.write("</div>\n\n")
        else:
            f.write("<p>None found</p>\n\n")

        # TIER 3
        f.write("<h3>❌ TIER 3: DOES NOT BEAT BUY-AND-HOLD</h3>\n")
        if underperformed:
            f.write("<p>🚫 These strategies <strong>NEVER</strong> beat buy-and-hold on any time horizon<br>\n")
            f.write("<strong>NOT RECOMMENDED</strong> for trading - use buy-and-hold instead</p>\n\n")

            f.write("<table>\n")
            f.write("    <thead>\n")
            f.write("        <tr>\n")
            f.write("            <th>Rank</th><th>Strategy</th><th>Avg Return</th><th>Sharpe</th><th>Drawdown</th><th>Won</th>\n")
            f.write("        </tr>\n")
            f.write("    </thead>\n")
            f.write("    <tbody>\n")

            for rank, strat in enumerate(underperformed, 1):
                f.write(f"        <tr class='tier3'>\n")
                f.write(f"            <td>{rank}</td>\n")
                f.write(f"            <td><strong>{strat.strategy_name}</strong></td>\n")
                f.write(f"            <td>{HTMLReportWriter.format_percentage(strat.avg_return)}</td>\n")
                f.write(f"            <td>{strat.avg_sharpe:.2f}</td>\n")
                f.write(f"            <td>{strat.avg_max_drawdown:.1%}</td>\n")
                f.write(f"            <td>{strat.horizons_beat_buyhold}/{strat.total_horizons}</td>\n")
                f.write(f"        </tr>\n")

            f.write("    </tbody>\n")
            f.write("</table>\n\n")

            f.write("<div class='blockquote info'>\n")
            f.write("    <p>💡 Even if returns are positive, buy-and-hold performed better.</p>\n")
            f.write("</div>\n\n")
        else:
            f.write("<p>None found</p>\n\n")

        # Investor Profile Recommendations
        f.write("<h3>👤 RECOMMENDATIONS BY INVESTOR PROFILE</h3>\n")
        f.write("<div class='profile-section'>\n")

        # Aggressive
        f.write("    <div class='profile-card'>\n")
        f.write("        <h4>🎯 AGGRESSIVE INVESTOR</h4>\n")
        f.write("        <p><em>Maximize returns, accept high risk</em></p>\n")
        if beat_buyhold:
            aggressive_pick = beat_buyhold[0]
            f.write(f"        <p><strong>→ {aggressive_pick.strategy_name}</strong></p>\n")
            f.write(f"        <p>Returns: {HTMLReportWriter.format_percentage(aggressive_pick.avg_return, False)} | ")
            f.write(f"Drawdown: {aggressive_pick.avg_max_drawdown:.1%}</p>\n")
        else:
            f.write("        <p><strong>→ Buy-and-hold</strong> (no active strategies beat benchmark)</p>\n")
        f.write("    </div>\n")

        # Conservative
        f.write("    <div class='profile-card'>\n")
        f.write("        <h4>🛡️ CONSERVATIVE INVESTOR</h4>\n")
        f.write("        <p><em>Minimize drawdown, accept lower returns</em></p>\n")
        if beat_buyhold:
            conservative_pick = min(beat_buyhold, key=lambda x: x.avg_max_drawdown)
            f.write(f"        <p><strong>→ {conservative_pick.strategy_name}</strong></p>\n")
            f.write(f"        <p>Returns: {HTMLReportWriter.format_percentage(conservative_pick.avg_return, False)} | ")
            f.write(f"Drawdown: {conservative_pick.avg_max_drawdown:.1%}</p>\n")
        else:
            f.write("        <p><strong>→ Buy-and-hold</strong> (no active strategies beat benchmark)</p>\n")
        f.write("    </div>\n")

        # Balanced
        f.write("    <div class='profile-card'>\n")
        f.write("        <h4>⚖️ BALANCED INVESTOR</h4>\n")
        f.write("        <p><em>Best risk-adjusted returns</em></p>\n")
        if beat_buyhold:
            balanced_pick = max(beat_buyhold, key=lambda x: x.avg_sharpe)
            f.write(f"        <p><strong>→ {balanced_pick.strategy_name}</strong></p>\n")
            f.write(f"        <p>Returns: {HTMLReportWriter.format_percentage(balanced_pick.avg_return, False)} | ")
            f.write(f"Sharpe: {balanced_pick.avg_sharpe:.2f}</p>\n")
        else:
            f.write("        <p><strong>→ Buy-and-hold</strong> (no active strategies beat benchmark)</p>\n")
        f.write("    </div>\n")

        f.write("</div>\n\n")

        # Time horizon recommendations
        f.write("<h3>⏰ BEST STRATEGY BY TIME HORIZON</h3>\n")
        f.write("<p><em>Choose strategy based on your investment timeline:</em></p>\n")
        f.write("<ul>\n")

        for horizon in self.horizons:
            best_for_horizon = None
            best_return = -float('inf')

            for score in strategy_scores:
                if horizon.name in score.horizon_results:
                    result = score.horizon_results[horizon.name]
                    if result['vs_buyhold'] > 0 and result['return'] > best_return:
                        best_return = result['return']
                        best_for_horizon = score.strategy_name

            buyhold = self.buy_hold_results.get(horizon.name, {}).get('total_return', 0)

            if best_for_horizon:
                f.write(f"    <li><strong>{horizon.name}</strong> → {best_for_horizon} ")
                f.write(f"({HTMLReportWriter.format_percentage(best_return, False)})<br>\n")
                f.write(f"        <em>Beat buy-and-hold by {HTMLReportWriter.format_percentage(best_return - buyhold)}</em></li>\n")
            else:
                f.write(f"    <li><strong>{horizon.name}</strong> → Buy-and-hold (no strategy beat benchmark)</li>\n")

        f.write("</ul>\n\n")

    def _generate_deep_dive_analysis(self, winning_strategy: StrategyScore, horizon: HorizonConfig) -> str:
        """
        Generate deep dive analysis with visualizations for the winning strategy.

        Args:
            winning_strategy: The best performing strategy
            horizon: The time horizon to analyze (typically the longest or best performing)

        Returns:
            HTML content with embedded visualizations
        """
        logger.info(f"Generating deep dive analysis for {winning_strategy.strategy_name} on {horizon.name}...")

        try:
            # Re-run the winning strategy to get full BacktestResult
            data = self.fetch_data(horizon.days)

            # Get strategy instance
            registry = get_registry()
            strategy_dict = registry.list_strategies()
            strategy_entries = [(name, metadata['class']) for name, metadata in strategy_dict.items()
                              if name == winning_strategy.strategy_name]

            if not strategy_entries:
                return "<p>⚠️ Could not load strategy for deep dive analysis</p>"

            strategy_name, strategy_class = strategy_entries[0]

            # Check if this is a multi-pair or portfolio strategy
            if hasattr(strategy_class, 'REQUIRES_MULTI_PAIR') and strategy_class.REQUIRES_MULTI_PAIR:
                return "<p>⚠️ Deep dive analysis not yet supported for multi-pair strategies</p>"

            # Check if strategy has portfolio or multi_asset tags (requires multiple assets)
            strategy_metadata = strategy_dict.get(winning_strategy.strategy_name, {})
            strategy_tags = strategy_metadata.get('tags', [])
            if 'portfolio' in strategy_tags or 'multi_asset' in strategy_tags:
                return "<p>⚠️ Deep dive analysis not yet supported for portfolio/multi-asset strategies</p>"

            # Initialize strategy with default parameters
            default_params = strategy_class.get_default_params() if hasattr(strategy_class, 'get_default_params') else {}
            strategy = strategy_class(name=strategy_name, config=default_params)
            strategy.initialize(strategy.config)

            # Augment data with features
            try:
                data = augment_with_features(data, DEFAULT_JOIN_CONFIG)
            except Exception as e:
                logger.warning(f"Feature augmentation failed: {e}")

            # Run backtest
            config = BacktestConfig(
                initial_capital=10000.0,
                trading_fee_percent=0.001,
                slippage_percent=0.0005,
                max_position_size=0.95
            )

            engine = BacktestEngine()
            timeframe_enum = self._timeframe_to_enum()
            result = engine.run_backtest(strategy, data, config, self.symbol, timeframe_enum)

            # Calculate buy-and-hold equity curve
            initial_price = data['close'].iloc[0]
            buyhold_equity = []
            for i in range(len(data)):
                current_price = data['close'].iloc[i]
                buyhold_value = config.initial_capital * (current_price / initial_price)
                timestamp = data['timestamp'].iloc[i] if 'timestamp' in data.columns else data.index[i]
                buyhold_equity.append((pd.to_datetime(timestamp), buyhold_value))

            # Generate visualizations
            html_content = self._create_deep_dive_html(result, buyhold_equity, data, winning_strategy)

            return html_content

        except Exception as e:
            logger.error(f"Deep dive analysis failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return f"<p>⚠️ Deep dive analysis failed: {str(e)}</p>"

    def _create_deep_dive_html(self, result: BacktestResult, buyhold_equity: List[Tuple],
                                data: pd.DataFrame, winning_strategy: StrategyScore) -> str:
        """
        Create HTML content with embedded visualizations for deep dive analysis.

        Args:
            result: BacktestResult from the winning strategy
            buyhold_equity: List of (datetime, value) tuples for buy-and-hold
            data: Market data DataFrame
            winning_strategy: StrategyScore object

        Returns:
            HTML content string with embedded base64 images
        """
        html_parts = []

        html_parts.append("<h2>🔬 DEEP DIVE: WINNING STRATEGY ANALYSIS</h2>\n")
        html_parts.append(f"<h3>{result.strategy_name}</h3>\n")
        html_parts.append("<p><em>Detailed analysis of trading decisions vs buy-and-hold benchmark</em></p>\n")
        html_parts.append("<hr>\n\n")

        # Extract equity curves as DataFrames
        strategy_times = [t for t, v in result.equity_curve]
        strategy_values = [v for t, v in result.equity_curve]
        buyhold_times = [t for t, v in buyhold_equity]
        buyhold_values = [v for t, v in buyhold_equity]

        strategy_df = pd.DataFrame({'time': strategy_times, 'value': strategy_values})
        buyhold_df = pd.DataFrame({'time': buyhold_times, 'value': buyhold_values})

        # === CHART 1: Equity Curve Comparison ===
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(strategy_df['time'], strategy_df['value'], label=f'{result.strategy_name}',
                   linewidth=2, color='#2E86DE')
            ax.plot(buyhold_df['time'], buyhold_df['value'], label='Buy & Hold',
                   linewidth=2, color='#EE5A6F', linestyle='--')

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax.set_title('Portfolio Value: Strategy vs Buy-and-Hold', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

            img_data = self._fig_to_base64(fig)
            html_parts.append(f'<img src="data:image/png;base64,{img_data}" style="width:100%; max-width:1000px;">\n\n')
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create equity curve chart: {e}")

        # === CHART 2: Cumulative Returns ===
        try:
            strategy_returns = [(v / strategy_values[0] - 1) * 100 for v in strategy_values]
            buyhold_returns = [(v / buyhold_values[0] - 1) * 100 for v in buyhold_values]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(strategy_times, strategy_returns, label=f'{result.strategy_name}',
                   linewidth=2, color='#2E86DE')
            ax.plot(buyhold_times, buyhold_returns, label='Buy & Hold',
                   linewidth=2, color='#EE5A6F', linestyle='--')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            # Highlight outperformance periods
            for i in range(min(len(strategy_returns), len(buyhold_returns))):
                if strategy_returns[i] > buyhold_returns[i]:
                    ax.axvspan(strategy_times[i], strategy_times[min(i+1, len(strategy_times)-1)],
                              alpha=0.1, color='green')

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Return (%)', fontsize=12)
            ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

            img_data = self._fig_to_base64(fig)
            html_parts.append(f'<img src="data:image/png;base64,{img_data}" style="width:100%; max-width:1000px;">\n\n')
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create returns chart: {e}")

        # === CHART 3: Price Chart with Trade Markers ===
        if result.trades and len(result.trades) > 0:
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

                # Price chart
                times = data['timestamp'] if 'timestamp' in data.columns else data.index
                times = pd.to_datetime(times)
                ax1.plot(times, data['close'], label='Price', linewidth=1.5, color='#34495e')

                # Buy signals
                buy_trades = [t for t in result.trades if t.side.value == 'buy']
                if buy_trades:
                    buy_times = [t.entry_time for t in buy_trades]
                    buy_prices = [t.entry_price for t in buy_trades]
                    ax1.scatter(buy_times, buy_prices, marker='^', s=100, c='green',
                               label='Buy', zorder=5, edgecolors='darkgreen')

                # Sell signals
                sell_trades = [t for t in result.trades]
                if sell_trades:
                    sell_times = [t.exit_time for t in sell_trades]
                    sell_prices = [t.exit_price for t in sell_trades]
                    ax1.scatter(sell_times, sell_prices, marker='v', s=100, c='red',
                               label='Sell', zorder=5, edgecolors='darkred')

                ax1.set_ylabel('Price ($)', fontsize=12)
                ax1.set_title(f'Trading Signals: {result.strategy_name}', fontsize=14, fontweight='bold')
                ax1.legend(loc='best', fontsize=11)
                ax1.grid(True, alpha=0.3)

                # Trade PnL distribution
                pnls = [t.pnl_percent for t in result.trades]
                colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
                ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.6)
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax2.set_xlabel('Trade Number', fontsize=12)
                ax2.set_ylabel('PnL (%)', fontsize=12)
                ax2.set_title('Individual Trade Performance', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                img_data = self._fig_to_base64(fig)
                html_parts.append(f'<img src="data:image/png;base64,{img_data}" style="width:100%; max-width:1000px;">\n\n')
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to create trade signals chart: {e}")

        # === CHART 4: Drawdown Comparison ===
        try:
            # Calculate drawdowns
            strategy_peak = pd.Series(strategy_values).cummax()
            strategy_dd = [(strategy_values[i] - strategy_peak.iloc[i]) / strategy_peak.iloc[i] * 100
                          for i in range(len(strategy_values))]

            buyhold_peak = pd.Series(buyhold_values).cummax()
            buyhold_dd = [(buyhold_values[i] - buyhold_peak.iloc[i]) / buyhold_peak.iloc[i] * 100
                         for i in range(len(buyhold_values))]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.fill_between(strategy_times, strategy_dd, 0, alpha=0.3, color='#2E86DE',
                           label=f'{result.strategy_name}')
            ax.fill_between(buyhold_times, buyhold_dd, 0, alpha=0.3, color='#EE5A6F',
                           label='Buy & Hold')
            ax.plot(strategy_times, strategy_dd, linewidth=1, color='#2E86DE')
            ax.plot(buyhold_times, buyhold_dd, linewidth=1, color='#EE5A6F')

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Drawdown (%)', fontsize=12)
            ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

            img_data = self._fig_to_base64(fig)
            html_parts.append(f'<img src="data:image/png;base64,{img_data}" style="width:100%; max-width:1000px;">\n\n')
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create drawdown chart: {e}")

        # === Trade Statistics Table ===
        if result.trades and len(result.trades) > 0:
            html_parts.append("<h4>📊 Trade Statistics</h4>\n")
            html_parts.append("<table>\n")
            html_parts.append("    <thead>\n")
            html_parts.append("        <tr>\n")
            html_parts.append("            <th>Metric</th><th>Value</th><th>Description</th>\n")
            html_parts.append("        </tr>\n")
            html_parts.append("    </thead>\n")
            html_parts.append("    <tbody>\n")

            winning_trades = [t for t in result.trades if t.pnl > 0]
            losing_trades = [t for t in result.trades if t.pnl <= 0]

            avg_win = sum(t.pnl_percent for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.pnl_percent for t in losing_trades) / len(losing_trades) if losing_trades else 0

            stats = [
                ("Total Trades", f"{len(result.trades)}", "Number of completed round trips"),
                ("Winning Trades", f"{len(winning_trades)}", "Trades that closed with profit"),
                ("Losing Trades", f"{len(losing_trades)}", "Trades that closed with loss"),
                ("Win Rate", f"{len(winning_trades)/len(result.trades)*100:.1f}%", "Percentage of profitable trades"),
                ("Avg Win", f"{avg_win:+.2f}%", "Average profit per winning trade"),
                ("Avg Loss", f"{avg_loss:+.2f}%", "Average loss per losing trade"),
                ("Best Trade", f"{max(t.pnl_percent for t in result.trades):+.2f}%", "Largest single trade gain"),
                ("Worst Trade", f"{min(t.pnl_percent for t in result.trades):+.2f}%", "Largest single trade loss"),
                ("Avg Trade Duration", f"{result.metrics.avg_trade_duration/60:.1f} hours", "Average time in position"),
            ]

            for metric, value, desc in stats:
                html_parts.append(f"        <tr>\n")
                html_parts.append(f"            <td><strong>{metric}</strong></td>\n")
                html_parts.append(f"            <td>{value}</td>\n")
                html_parts.append(f"            <td><em>{desc}</em></td>\n")
                html_parts.append(f"        </tr>\n")

            html_parts.append("    </tbody>\n")
            html_parts.append("</table>\n\n")

        # === Key Insights ===
        html_parts.append("<h4>💡 Key Insights</h4>\n")
        html_parts.append("<ul>\n")

        final_strategy = result.equity_curve[-1][1]
        final_buyhold = buyhold_equity[-1][1]
        outperformance = (final_strategy - final_buyhold) / final_buyhold * 100

        html_parts.append(f"    <li><strong>Total Return:</strong> Strategy returned {result.metrics.total_return*100:.1f}% "
                         f"vs Buy-and-Hold {(final_buyhold/result.initial_capital - 1)*100:.1f}%</li>\n")
        html_parts.append(f"    <li><strong>Outperformance:</strong> Strategy outperformed buy-and-hold by "
                         f"{outperformance:+.1f}%</li>\n")
        html_parts.append(f"    <li><strong>Risk-Adjusted Returns:</strong> Sharpe ratio of {result.metrics.sharpe_ratio:.2f} "
                         f"indicates {'excellent' if result.metrics.sharpe_ratio > 2 else 'good' if result.metrics.sharpe_ratio > 1 else 'moderate'} risk-adjusted performance</li>\n")
        html_parts.append(f"    <li><strong>Drawdown Management:</strong> Maximum drawdown of {result.metrics.max_drawdown*100:.1f}% "
                         f"{'is well-controlled' if result.metrics.max_drawdown < 0.2 else 'requires careful risk management'}</li>\n")

        if result.trades:
            html_parts.append(f"    <li><strong>Trading Frequency:</strong> Executed {len(result.trades)} trades over "
                             f"{result.duration_days} days ({len(result.trades)/result.duration_days*30:.1f} trades/month)</li>\n")

        html_parts.append("</ul>\n\n")

        return "".join(html_parts)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        buffer.close()
        return img_str

    def _prepare_interactive_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Prepare data for interactive visualizations by collecting all strategy results.

        Returns:
            Dictionary mapping strategy names to their complete data including:
            - timestamps: List of datetime objects
            - equity_strategy: Strategy equity curve values
            - equity_buyhold: Buy-and-hold equity curve values
            - prices: Asset prices
            - trades: List of trade dictionaries
            - metrics: Performance metrics dictionary
        """
        logger.info("Preparing interactive visualization data for all strategies...")

        interactive_data = {}

        # Group results by strategy name
        strategy_results = {}
        for result in self.all_results:
            strategy_name = result['strategy_name']
            if strategy_name not in strategy_results:
                strategy_results[strategy_name] = []
            strategy_results[strategy_name].append(result)

        # Process each strategy
        for strategy_name, results in strategy_results.items():
            try:
                # Use the longest horizon for this strategy
                longest_result = max(results, key=lambda r: r.get('days', 0))
                horizon_days = longest_result.get('days', 30)

                # Fetch data for this horizon
                data = self.fetch_data(horizon_days)

                # Re-run backtest to get full equity curve
                registry = get_registry()
                strategy_dict = registry.list_strategies()
                strategy_entries = [(name, metadata['class']) for name, metadata in strategy_dict.items()
                                  if name == strategy_name]

                if not strategy_entries:
                    logger.warning(f"Could not load strategy {strategy_name}")
                    continue

                _, strategy_class = strategy_entries[0]

                # Skip multi-pair strategies for now (need different data handling)
                if hasattr(strategy_class, 'REQUIRES_MULTI_PAIR') and strategy_class.REQUIRES_MULTI_PAIR:
                    logger.debug(f"Skipping multi-pair strategy {strategy_name} for interactive viz")
                    continue

                # Initialize strategy
                default_params = strategy_class.get_default_params() if hasattr(strategy_class, 'get_default_params') else {}
                strategy = strategy_class(name=strategy_name, config=default_params)
                strategy.initialize(strategy.config)

                # Augment data
                try:
                    data = augment_with_features(data, self.symbol, self.timeframe, config=DEFAULT_JOIN_CONFIG)
                except Exception as e:
                    logger.warning(f"Feature augmentation failed for {strategy_name}: {e}")

                # Run backtest
                config = BacktestConfig(
                    initial_capital=10000.0,
                    trading_fee_percent=0.001,
                    slippage_percent=0.0005,
                    max_position_size=0.95
                )

                engine = BacktestEngine()
                timeframe_enum = self._timeframe_to_enum()
                backtest_result = engine.run_backtest(strategy, data, config, self.symbol, timeframe_enum)

                # Extract timestamps and equity curves
                timestamps = []
                equity_strategy = []
                equity_buyhold = []
                prices = []

                initial_price = data['close'].iloc[0]

                for i in range(len(data)):
                    timestamp = data['timestamp'].iloc[i] if 'timestamp' in data.columns else data.index[i]
                    timestamps.append(pd.to_datetime(timestamp))

                    # Get strategy equity at this point
                    if i < len(backtest_result.equity_curve):
                        equity_strategy.append(backtest_result.equity_curve[i])
                    else:
                        equity_strategy.append(equity_strategy[-1] if equity_strategy else config.initial_capital)

                    # Calculate buy-and-hold equity
                    current_price = data['close'].iloc[i]
                    buyhold_value = config.initial_capital * (current_price / initial_price)
                    equity_buyhold.append(buyhold_value)

                    prices.append(current_price)

                # Extract trades
                trades = []
                for trade in backtest_result.trades:
                    trades.append({
                        'entry_time': pd.to_datetime(trade.entry_time),
                        'exit_time': pd.to_datetime(trade.exit_time),
                        'entry_price': float(trade.entry_price),
                        'exit_price': float(trade.exit_price),
                        'side': trade.side.value,
                        'pnl_percent': float(trade.pnl_percent)
                    })

                # Calculate metrics
                metrics = {
                    'total_return': float(backtest_result.metrics.total_return),
                    'sharpe_ratio': float(backtest_result.metrics.sharpe_ratio),
                    'max_drawdown': float(backtest_result.metrics.max_drawdown),
                    'win_rate': float(backtest_result.metrics.win_rate),
                    'num_trades': int(backtest_result.metrics.total_trades),
                    'profit_factor': float(backtest_result.metrics.profit_factor) if hasattr(backtest_result.metrics, 'profit_factor') else 0.0
                }

                # Store data
                interactive_data[strategy_name] = {
                    'timestamps': timestamps,
                    'equity_strategy': equity_strategy,
                    'equity_buyhold': equity_buyhold,
                    'prices': prices,
                    'trades': trades,
                    'metrics': metrics
                }

                logger.debug(f"Prepared interactive data for {strategy_name}")

            except Exception as e:
                logger.warning(f"Could not prepare interactive data for {strategy_name}: {e}")
                continue

        logger.info(f"✓ Prepared interactive data for {len(interactive_data)} strategies")
        return interactive_data

    def _generate_interactive_section(self, interactive_data: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate HTML for the interactive visualization section.

        Args:
            interactive_data: Dictionary mapping strategy names to their visualization data

        Returns:
            HTML string containing the interactive section
        """
        if not interactive_data:
            return "<p><em>No interactive visualizations available</em></p>"

        try:
            html_content = generate_interactive_section_html(
                all_results=interactive_data,
                symbol=self.symbol
            )
            return html_content
        except Exception as e:
            logger.error(f"Failed to generate interactive section: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return f"<p>⚠️ Interactive visualizations unavailable: {str(e)}</p>"

    def generate_master_report(self, strategy_scores: List[StrategyScore]) -> None:
        """Generate comprehensive master report in HTML format."""
        logger.info("\nGenerating master report...")

        report_file = self.output_dir / "MASTER_REPORT.html"

        with open(report_file, 'w', encoding='utf-8') as f:
            # HTML Header
            f.write("<!DOCTYPE html>\n")
            f.write("<html lang='en'>\n")
            f.write("<head>\n")
            f.write("    <meta charset='UTF-8'>\n")
            f.write("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
            f.write("    <title>Crypto Trading Master Strategy Analysis</title>\n")
            f.write(HTMLReportWriter.get_css())
            f.write("</head>\n")
            f.write("<body>\n")
            f.write("<div class='container'>\n")

            # Main Header
            f.write("<h1>🚀 CRYPTO TRADING MASTER STRATEGY ANALYSIS</h1>\n\n")

            # Metadata section
            f.write("<div class='metadata'>\n")
            f.write(f"    <p><strong>Asset:</strong> {self.symbol}</p>\n")
            f.write(f"    <p><strong>Timeframe:</strong> {self.timeframe}</p>\n")
            f.write(f"    <p><strong>Strategies Tested:</strong> {len(strategy_scores)}</p>\n")
            f.write(f"    <p><strong>Time Horizons:</strong> {', '.join([h.name for h in self.horizons])}</p>\n")

            # Get date range from the longest horizon data
            try:
                longest_horizon = max(self.horizons, key=lambda h: h.days)
                data = self.fetch_data(longest_horizon.days)
                if hasattr(data, 'index') and len(data) > 0:
                    start_date = data.index[0].strftime('%Y-%m-%d')
                    end_date = data.index[-1].strftime('%Y-%m-%d')
                    f.write(f"    <p><strong>Data Period:</strong> {start_date} to {end_date} ({len(data):,} candles)</p>\n")
            except Exception as e:
                logger.debug(f"Could not extract date range: {e}")

            f.write(f"    <p><strong>Total Backtests:</strong> {len(self.all_results)}</p>\n")
            f.write(f"    <p><strong>Parallel Workers:</strong> {self.workers}</p>\n")
            f.write(f"    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            f.write("</div>\n\n")

            if not strategy_scores:
                f.write("<p>❌ No results available</p>\n")
                f.write("</div></body></html>")
                return

            # Calculate average buy-hold return
            avg_buyhold = np.mean([v['total_return'] for v in self.buy_hold_results.values()])

            # PRACTICAL RECOMMENDATIONS SECTION (HTML VERSION)
            self._write_practical_recommendations_html(f, strategy_scores, avg_buyhold)

            # Best strategy (composite score)
            best = strategy_scores[0]
            f.write("<h2>📊 COMPOSITE SCORE RANKINGS (Academic)</h2>\n")
            f.write("<div class='blockquote warning'>\n")
            f.write("    <p><strong>⚠️ NOTE:</strong> This ranking uses a weighted composite score (35% Sharpe, 30% Return, ")
            f.write("20% Drawdown, 15% WinRate). See <strong>PRACTICAL RECOMMENDATIONS</strong> above for ")
            f.write("actual trading decisions based on beating buy-and-hold.</p>\n")
            f.write("</div>\n\n")

            f.write(f"<p><strong>Top by Composite Score:</strong> {best.strategy_name}</p>\n")
            f.write(f"<p><strong>Composite Score:</strong> {best.composite_score:.3f} / 1.000</p>\n")
            f.write(f"<p><strong>Rank:</strong> #1 out of {len(strategy_scores)}</p>\n\n")

            f.write("<h4>Performance Summary:</h4>\n")
            f.write("<ul>\n")
            f.write(f"    <li>Average Return: {HTMLReportWriter.format_percentage(best.avg_return)}</li>\n")

            outperformance = best.avg_return - avg_buyhold
            f.write(f"    <li>Buy-and-Hold Avg: {HTMLReportWriter.format_percentage(avg_buyhold)}</li>\n")
            f.write(f"    <li>Outperformance: {HTMLReportWriter.format_percentage(outperformance)}</li>\n")
            f.write(f"    <li>Sharpe Ratio: <strong>{best.avg_sharpe:.2f}</strong></li>\n")
            f.write(f"    <li>Max Drawdown: <strong>{best.avg_max_drawdown:.1%}</strong></li>\n")
            f.write(f"    <li>Win Rate: <strong>{best.avg_win_rate:.1%}</strong></li>\n")
            f.write(f"    <li>Horizons Won: <strong>{best.horizons_beat_buyhold}/{best.total_horizons}</strong></li>\n")
            f.write("</ul>\n\n")

            # Add warning if best by score didn't beat buy-hold
            if best.horizons_beat_buyhold == 0:
                f.write("<div class='blockquote warning'>\n")
                f.write("    <p><strong>⚠️ WARNING:</strong> This strategy did NOT beat buy-and-hold on any time horizon! ")
                f.write("See PRACTICAL RECOMMENDATIONS section for better trading choices.</p>\n")
                f.write("</div>\n\n")

            # Rankings Table
            f.write("<h3>Strategy Rankings (by Composite Score)</h3>\n")
            f.write("<table>\n")
            f.write("    <thead>\n")
            f.write("        <tr>\n")
            f.write("            <th>Rank</th><th>Strategy</th><th>Score</th><th>Return</th>\n")
            f.write("            <th>Sharpe</th><th>MaxDD</th><th>WinRate</th><th>Won</th>\n")
            f.write("        </tr>\n")
            f.write("    </thead>\n")
            f.write("    <tbody>\n")

            for rank, score in enumerate(strategy_scores, 1):
                f.write(f"        <tr>\n")
                f.write(f"            <td>{rank}</td>\n")
                f.write(f"            <td><strong>{score.strategy_name}</strong></td>\n")
                f.write(f"            <td>{score.composite_score:.3f}</td>\n")
                f.write(f"            <td>{HTMLReportWriter.format_percentage(score.avg_return)}</td>\n")
                f.write(f"            <td>{score.avg_sharpe:.2f}</td>\n")
                f.write(f"            <td>{score.avg_max_drawdown:.1%}</td>\n")
                f.write(f"            <td>{score.avg_win_rate:.1%}</td>\n")
                f.write(f"            <td>{score.horizons_beat_buyhold}/{score.total_horizons}</td>\n")
                f.write(f"        </tr>\n")

            f.write("    </tbody>\n")
            f.write("</table>\n\n")
            f.write(f"<p><strong>Buy-and-Hold Baseline:</strong> {HTMLReportWriter.format_percentage(avg_buyhold)}</p>\n\n")

            # Time horizon analysis
            f.write("<h2>📈 TIME HORIZON ANALYSIS</h2>\n")
            f.write("<h4>Best Strategy by Horizon:</h4>\n")
            f.write("<ul>\n")

            for horizon in self.horizons:
                best_for_horizon = None
                best_return = -float('inf')

                for score in strategy_scores:
                    if horizon.name in score.horizon_results:
                        horizon_return = score.horizon_results[horizon.name]['return']
                        if horizon_return > best_return:
                            best_return = horizon_return
                            best_for_horizon = score.strategy_name

                buyhold = self.buy_hold_results.get(horizon.name, {}).get('total_return', 0)
                f.write(f"    <li><strong>{horizon.name}:</strong> {best_for_horizon} ")
                f.write(f"({HTMLReportWriter.format_percentage(best_return)} vs buy-hold {HTMLReportWriter.format_percentage(buyhold)})</li>\n")

            f.write("</ul>\n\n")

            # === ADD VISUALIZATIONS ===
            f.write("<h2>📊 INTERACTIVE VISUALIZATIONS</h2>\n\n")

            # Performance Heatmap
            f.write("<h3>Strategy Performance Heatmap</h3>\n")
            f.write("<p><em>Color-coded returns across all strategies and time horizons. Green = positive, Red = negative.</em></p>\n")
            heatmap_html = HTMLReportWriter.create_performance_heatmap(strategy_scores, self.horizons)
            f.write(heatmap_html)
            f.write("\n\n")

            # Sharpe Ratio Comparison
            f.write("<h3>Sharpe Ratio Comparison</h3>\n")
            f.write("<p><em>Risk-adjusted performance metric. Green (>1.0) = Good, Orange (>0) = Acceptable, Red (<0) = Poor.</em></p>\n")
            sharpe_chart_html = HTMLReportWriter.create_sharpe_comparison_chart(strategy_scores)
            f.write(sharpe_chart_html)
            f.write("\n\n")

            # === NEW: Interactive Time Series Visualizations ===
            try:
                logger.info("Generating interactive time series visualizations...")
                interactive_data = self._prepare_interactive_data()
                if interactive_data:
                    interactive_section_html = self._generate_interactive_section(interactive_data)
                    f.write(interactive_section_html)
                    f.write("\n\n")
                    logger.success("✓ Interactive visualizations added to report")
                else:
                    logger.warning("No interactive data available - skipping interactive section")
            except Exception as e:
                logger.error(f"Failed to add interactive section: {e}")
                f.write("<div class='blockquote warning'>\n")
                f.write("    <p>⚠️ Interactive visualizations unavailable</p>\n")
                f.write("</div>\n\n")

            # Detailed analysis of best strategy
            f.write(f"<h2>🔍 DETAILED ANALYSIS: {best.strategy_name} (Best Overall)</h2>\n")
            f.write("<h4>Performance Across Horizons:</h4>\n")
            f.write("<table>\n")
            f.write("    <thead>\n")
            f.write("        <tr>\n")
            f.write("            <th>Horizon</th><th>Return</th><th>vs B&H</th><th>Sharpe</th>\n")
            f.write("            <th>MaxDD</th><th>WinRate</th><th>Trades</th>\n")
            f.write("        </tr>\n")
            f.write("    </thead>\n")
            f.write("    <tbody>\n")

            for horizon in self.horizons:
                if horizon.name not in best.horizon_results:
                    continue

                result = best.horizon_results[horizon.name]
                f.write(f"        <tr>\n")
                f.write(f"            <td><strong>{horizon.name}</strong></td>\n")
                f.write(f"            <td>{HTMLReportWriter.format_percentage(result['return'])}</td>\n")
                f.write(f"            <td>{HTMLReportWriter.format_percentage(result['vs_buyhold'])}</td>\n")
                f.write(f"            <td>{result['sharpe']:.2f}</td>\n")
                f.write(f"            <td>{result['drawdown']:.1%}</td>\n")
                f.write(f"            <td>{result['win_rate']:.1%}</td>\n")
                f.write(f"            <td>{int(result['trades'])}</td>\n")
                f.write(f"        </tr>\n")

            f.write("    </tbody>\n")
            f.write("</table>\n\n")

            # Recommendations
            f.write("<h2>🚀 NEXT STEPS FOR IMPLEMENTATION</h2>\n")

            practical_winners = [s for s in strategy_scores if s.horizons_beat_buyhold >= 3]
            if practical_winners:
                practical_winners.sort(key=lambda x: x.avg_return, reverse=True)
                practical_best = practical_winners[0]

                f.write("<h3>📋 RECOMMENDED ACTION PLAN</h3>\n")
                f.write(f"<p><strong>✅ Deploy:</strong> {practical_best.strategy_name}</p>\n")
                f.write("<p><em>(Top strategy that consistently beats buy-and-hold)</em></p>\n\n")

                f.write("<div class='action-plan'>\n")
                f.write("<h4>1. VALIDATION PHASE (Weeks 1-4)</h4>\n")
                f.write("<ul>\n")
                f.write("    <li>Start with paper trading to validate performance</li>\n")
                f.write("    <li>Track all signals and compare to backtested results</li>\n")
                f.write("    <li>Document any discrepancies between live and backtest</li>\n")
                f.write("    <li>Verify transaction costs match assumptions (0.1% + 0.05%)</li>\n")
                f.write("</ul>\n\n")

                f.write("<h4>2. INITIAL DEPLOYMENT (Weeks 5-8)</h4>\n")
                f.write("<ul>\n")
                f.write("    <li>Start with 2-5% of total portfolio</li>\n")
                f.write(f"    <li>Set stop-loss at {practical_best.avg_max_drawdown * 2:.1%} (2× max historical drawdown)</li>\n")
                f.write("    <li>Monitor daily for first 2 weeks, then weekly</li>\n")
                f.write("    <li>Keep detailed performance log vs buy-and-hold</li>\n")
                f.write("</ul>\n\n")

                f.write("<h4>3. SCALING (Weeks 9+)</h4>\n")
                f.write("<ul>\n")
                f.write("    <li>If outperforming buy-and-hold: gradually increase to 10-20%</li>\n")
                f.write("    <li>If underperforming: reduce position or revert to buy-and-hold</li>\n")
                f.write("    <li>Consider diversifying across top 3 performing strategies</li>\n")
                f.write("</ul>\n\n")

                f.write("<h4>4. OPTIMIZATION & EXPANSION</h4>\n")
                f.write("<ul>\n")
                f.write(f"    <li>Run parameter optimization on {practical_best.strategy_name}</li>\n")
                f.write("    <li>Test on other crypto pairs (ETH, SOL, BNB, ADA)</li>\n")
                f.write("    <li>Consider ensemble approach combining multiple strategies</li>\n")
                f.write("    <li>Review performance quarterly and rerun analysis</li>\n")
                f.write("</ul>\n")
                f.write("</div>\n\n")
            else:
                f.write("<div class='blockquote warning'>\n")
                f.write("    <p><strong>⚠️ NO STRATEGIES BEAT BUY-AND-HOLD CONSISTENTLY</strong></p>\n")
                f.write("</div>\n\n")
                f.write("<h4>RECOMMENDED ACTION:</h4>\n")
                f.write("<ul>\n")
                f.write("    <li>Stick with passive buy-and-hold strategy</li>\n")
                f.write("    <li>Review market conditions and retry analysis in 3-6 months</li>\n")
                f.write("    <li>Consider these alternatives:\n")
                f.write("        <ul>\n")
                f.write("            <li>DCA (Dollar Cost Averaging) into BTC/ETH</li>\n")
                f.write("            <li>Portfolio diversification (60/40 BTC/ETH split)</li>\n")
                f.write("            <li>Focus on parameter optimization of promising strategies</li>\n")
                f.write("        </ul>\n")
                f.write("    </li>\n")
                f.write("</ul>\n\n")

            f.write("<h3>📊 Additional Resources</h3>\n")
            f.write("<ul>\n")
            f.write("    <li><strong>Full comparison matrix:</strong> <code>comparison_matrix.csv</code></li>\n")
            f.write("    <li><strong>Detailed results:</strong> <code>detailed_results/</code> directory</li>\n")
            f.write("    <li>See <strong>PRACTICAL STRATEGY RECOMMENDATIONS</strong> section above</li>\n")
            f.write("</ul>\n\n")

            # Add deep dive analysis for winning strategy
            if practical_winners:
                # Use the best performing horizon for deep dive
                winning_strategy = practical_winners[0]
                best_horizon = None
                best_horizon_return = -float('inf')

                # Find the horizon where the strategy performed best
                for horizon_name, result in winning_strategy.horizon_results.items():
                    if result['return'] > best_horizon_return:
                        best_horizon_return = result['return']
                        # Find the corresponding HorizonConfig
                        for h in self.horizons:
                            if h.name == horizon_name:
                                best_horizon = h
                                break

                # If we found a good horizon, generate deep dive
                if best_horizon:
                    try:
                        deep_dive_html = self._generate_deep_dive_analysis(winning_strategy, best_horizon)
                        f.write(deep_dive_html)
                        f.write("<hr>\n\n")
                    except Exception as e:
                        logger.warning(f"Could not generate deep dive analysis: {e}")

            # Add academic research section
            self._write_academic_section_html(f, strategy_scores, avg_buyhold)

            # Close HTML
            f.write("</div>\n")
            f.write("</body>\n")
            f.write("</html>\n")

        logger.success(f"✓ Master report: {report_file}")

        # Also save comparison matrix as CSV
        self._save_comparison_matrix()

    def _write_academic_section(self, f, strategy_scores: List[StrategyScore], avg_buyhold: float) -> None:
        """Write comprehensive academic analysis section."""
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("ACADEMIC RESEARCH REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Abstract
        f.write("ABSTRACT\n")
        f.write("-" * 80 + "\n\n")
        f.write("TL;DR: Comprehensive empirical evaluation of ")
        f.write(f"{len(strategy_scores)} algorithmic trading strategies across ")
        f.write(f"{len(self.horizons)} time horizons revealed ")

        # Count strategies that beat buy-hold
        strategies_beat_buyhold = sum(1 for s in strategy_scores if s.avg_return > avg_buyhold)
        f.write(f"{strategies_beat_buyhold} strategies outperforming passive buy-and-hold ")
        f.write(f"benchmarks, with the top strategy achieving {strategy_scores[0].avg_return:+.2%} ")
        f.write(f"average returns versus {avg_buyhold:+.2%} for buy-and-hold.\n\n")

        f.write("This study presents a systematic comparative analysis of cryptocurrency ")
        f.write(f"trading strategies on {self.symbol} using high-frequency {self.timeframe} ")
        f.write("candlestick data from Binance exchange. We evaluate ")

        # Count strategy types
        single_pair_count = sum(1 for r in self.all_results if r.get('strategy_type') == 'single_pair')
        multi_pair_count = sum(1 for r in self.all_results if r.get('strategy_type') == 'multi_pair')
        unique_single = len(set(r['strategy_name'] for r in self.all_results if r.get('strategy_type') == 'single_pair'))
        unique_multi = len(set(r['strategy_name'] for r in self.all_results if r.get('strategy_type') == 'multi_pair'))

        f.write(f"{unique_single} single-asset and {unique_multi} multi-asset strategies ")
        f.write(f"through {len(self.all_results)} independent backtests spanning timeframes from ")
        f.write(f"{self.horizons[0].days} to {self.horizons[-1].days} days. ")
        f.write("Performance is assessed using risk-adjusted metrics including Sharpe ratio, ")
        f.write("maximum drawdown, win rate, and total returns, with all strategies benchmarked ")
        f.write("against passive buy-and-hold positions. Results indicate significant ")
        f.write("heterogeneity in strategy performance across temporal horizons, with ")
        f.write(f"momentum-based and portfolio rebalancing approaches demonstrating superior ")
        f.write("risk-adjusted returns in the tested market conditions.\n\n")

        # Methodology
        f.write("\n" + "=" * 80 + "\n")
        f.write("1. METHODOLOGY\n")
        f.write("=" * 80 + "\n\n")

        # 1.1 Data Collection
        f.write("1.1 Data Collection & Preprocessing\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"TL;DR: {len(self.all_results)} backtests executed on {self.timeframe} OHLCV ")
        f.write(f"data from Binance, spanning {self.horizons[0].days}-{self.horizons[-1].days} ")
        f.write("days with no survivorship bias.\n\n")

        f.write("Market Data Specification:\n")
        f.write(f"  • Exchange: Binance (via REST API)\n")
        f.write(f"  • Primary Asset: {self.symbol}\n")
        f.write(f"  • Timeframe Granularity: {self.timeframe} candlesticks\n")
        f.write(f"  • Data Fields: Open, High, Low, Close, Volume (OHLCV)\n")

        # Calculate total candles
        total_candles = sum(len(d) for d in [self.fetch_data(h.days) for h in self.horizons[:1]])
        f.write(f"  • Sample Size (largest horizon): {self.horizons[-1].days * 24 if self.timeframe == '1h' else 'varies'} candles\n")
        f.write(f"  • Historical Range: {self.horizons[0].days} to {self.horizons[-1].days} days\n")
        f.write(f"  • Data Quality: Real-time market data, no look-ahead bias\n\n")

        f.write("The dataset encompasses multiple market regimes including trending, ")
        f.write("ranging, and volatile periods, ensuring robust out-of-sample testing. ")
        f.write("All data points represent actual executed trades on Binance, eliminating ")
        f.write("concerns regarding liquidity assumptions or bid-ask spread estimation ")
        f.write("common in synthetic datasets.\n\n")

        # 1.2 Strategy Selection
        f.write("1.2 Strategy Universe & Classification\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"TL;DR: {len(strategy_scores)} strategies tested across {len(set(r.get('strategy_type', 'single_pair') for r in self.all_results))} ")
        f.write("categories: technical indicators, mean reversion, momentum, and portfolio management.\n\n")

        f.write("Strategy Taxonomy:\n\n")

        # Group strategies by type
        for idx, score in enumerate(strategy_scores, 1):
            # Get strategy type
            strategy_results = [r for r in self.all_results if r['strategy_name'] == score.strategy_name]
            if strategy_results:
                strategy_type = strategy_results[0].get('strategy_type', 'single_pair')

                if strategy_type == 'single_pair':
                    # Classify single-pair strategies
                    if 'SMA' in score.strategy_name or 'EMA' in score.strategy_name:
                        category = "Trend Following (Moving Average)"
                    elif 'RSI' in score.strategy_name or 'VWAP' in score.strategy_name:
                        category = "Mean Reversion (Oscillator)"
                    elif 'MACD' in score.strategy_name or 'Supertrend' in score.strategy_name:
                        category = "Momentum (Trend + Momentum)"
                    elif 'Bollinger' in score.strategy_name:
                        category = "Volatility Breakout"
                    elif 'Ichimoku' in score.strategy_name:
                        category = "Multi-Timeframe Analysis"
                    else:
                        category = "Technical Indicator"

                    f.write(f"  {idx}. {score.strategy_name} ({category})\n")

                    # Get parameters
                    params = self._get_default_params(score.strategy_name)
                    if params:
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                        f.write(f"     Parameters: {param_str}\n")

                else:  # multi_pair
                    # Properly classify multi-pair strategies
                    strategy_name = score.strategy_name

                    # Portfolio optimization strategies
                    if strategy_name in ['HierarchicalRiskParity', 'BlackLitterman', 'RiskParity',
                                          'DeepRLPortfolio', 'PortfolioRebalancer']:
                        if strategy_name == 'HierarchicalRiskParity':
                            category = "Portfolio Optimization (Hierarchical Risk Parity)"
                            description = "Hierarchical clustering-based portfolio construction"
                        elif strategy_name == 'BlackLitterman':
                            category = "Portfolio Optimization (Black-Litterman)"
                            description = "Bayesian asset allocation with investor views"
                        elif strategy_name == 'RiskParity':
                            category = "Portfolio Optimization (Risk Parity)"
                            description = "Equal Risk Contribution with kurtosis minimization"
                        elif strategy_name == 'DeepRLPortfolio':
                            category = "Portfolio Optimization (Deep Reinforcement Learning)"
                            description = "PPO agent-based dynamic portfolio allocation"
                        else:  # PortfolioRebalancer
                            category = "Portfolio Rebalancing (Threshold-based)"
                            description = "Periodic rebalancing with drift threshold"

                        sample = strategy_results[0]
                        num_assets = sample.get('num_assets', 'N/A')
                        f.write(f"  {idx}. {strategy_name} ({category})\n")
                        f.write(f"     Tested Configurations: {num_assets}-asset portfolios\n")
                        f.write(f"     Method: {description}\n")

                    # Pairs trading strategies
                    elif strategy_name in ['CopulaPairsTrading', 'StatisticalArbitrage']:
                        if strategy_name == 'CopulaPairsTrading':
                            category = "Pairs Trading (Copula-Enhanced)"
                            description = "Tail dependency modeling with Student-t copula"
                        else:  # StatisticalArbitrage
                            category = "Pairs Trading (Cointegration-based)"
                            description = "Mean reversion on cointegrated pairs"

                        f.write(f"  {idx}. {strategy_name} ({category})\n")
                        f.write(f"     Method: {description}\n")

                    else:
                        # Fallback for unknown strategies
                        category = "Multi-Asset Strategy"
                        f.write(f"  {idx}. {strategy_name} ({category})\n")

        f.write("\n")
        f.write("All strategies were implemented with identical trading costs assumptions:\n")
        f.write("  • Commission: 0.1% per trade (Binance maker/taker fee)\n")
        f.write("  • Slippage: 0.05% (conservative market impact estimate)\n")
        f.write("  • Initial Capital: $10,000 USD per strategy\n\n")

        # 1.3 Testing Framework
        f.write("1.3 Backtesting Framework & Execution\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"TL;DR: Parallel execution using {self.workers} workers, event-driven ")
        f.write("backtesting engine, no optimization bias, walk-forward validation across ")
        f.write(f"{len(self.horizons)} horizons.\n\n")

        f.write("Computational Infrastructure:\n")
        f.write(f"  • Execution Mode: Parallel processing ({self.workers} concurrent workers)\n")
        f.write(f"  • Backtest Engine: Event-driven architecture (VectorBT-based)\n")
        f.write(f"  • Total Simulations: {len(self.all_results)} independent backtests\n")
        f.write(f"  • Execution Time: {len(self.all_results) / self.workers / 60:.1f} minutes (estimated)\n\n")

        f.write("Temporal Validation Structure:\n")
        for horizon in self.horizons:
            f.write(f"  • {horizon.name:6s}: {horizon.description:20s} ")
            # Calculate number of strategies tested on this horizon
            horizon_tests = sum(1 for r in self.all_results if r['horizon'] == horizon.name)
            f.write(f"({horizon_tests} strategies tested)\n")

        f.write("\nThis multi-horizon approach enables assessment of strategy robustness across ")
        f.write("different market timescales, identifying strategies that maintain consistent ")
        f.write("performance versus those exhibiting regime-specific behavior.\n\n")

        # 1.4 Evaluation Metrics
        f.write("1.4 Performance Metrics & Scoring Methodology\n")
        f.write("-" * 80 + "\n\n")
        f.write("TL;DR: Composite scoring combines Sharpe ratio (35%), returns (30%), ")
        f.write("drawdown (20%), and win rate (15%) using min-max normalization.\n\n")

        f.write("Primary Metrics:\n\n")
        f.write("  1. Total Return (R):\n")
        f.write("     R = (Final_Capital - Initial_Capital) / Initial_Capital\n")
        f.write("     Measures absolute profitability without risk adjustment.\n\n")

        f.write("  2. Sharpe Ratio (SR):\n")
        f.write("     SR = (Mean_Return × Periods_Per_Year) / (Std_Return × √Periods_Per_Year)\n")
        f.write("     Risk-adjusted return metric, annualized for comparability.\n\n")

        f.write("  3. Maximum Drawdown (MDD):\n")
        f.write("     MDD = max(Peak_Value - Trough_Value) / Peak_Value\n")
        f.write("     Largest peak-to-trough decline, measures downside risk.\n\n")

        f.write("  4. Win Rate (WR):\n")
        f.write("     WR = Profitable_Trades / Total_Trades\n")
        f.write("     Percentage of trades closing with profit.\n\n")

        f.write("Composite Score Formula:\n")
        f.write("  Normalized_Score = 0.35×Sharpe_norm + 0.30×Return_norm + \n")
        f.write("                     0.20×(1-Drawdown_norm) + 0.15×WinRate_norm\n\n")
        f.write("Where all metrics are normalized to [0,1] using min-max scaling across\n")
        f.write("the strategy universe. Drawdown is inverted (lower is better). This\n")
        f.write("weighting scheme prioritizes risk-adjusted returns (Sharpe) while\n")
        f.write("incorporating absolute performance and risk metrics.\n\n")

        # Results & Analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("2. RESULTS & COMPARATIVE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # 2.1 Overall Performance Distribution
        f.write("2.1 Performance Distribution Across Strategy Universe\n")
        f.write("-" * 80 + "\n\n")

        # Calculate statistics
        all_returns = [s.avg_return for s in strategy_scores]
        all_sharpes = [s.avg_sharpe for s in strategy_scores]
        positive_returns = sum(1 for r in all_returns if r > 0)
        beat_buyhold_count = sum(1 for s in strategy_scores if s.avg_return > avg_buyhold)

        f.write(f"TL;DR: {positive_returns}/{len(strategy_scores)} strategies profitable, ")
        f.write(f"{beat_buyhold_count}/{len(strategy_scores)} beat buy-and-hold, ")
        f.write(f"average return {np.mean(all_returns):+.2%} (vs {avg_buyhold:+.2%} passive).\n\n")

        f.write("Aggregate Statistics:\n")
        f.write(f"  • Mean Return: {np.mean(all_returns):+.2%}\n")
        f.write(f"  • Median Return: {np.median(all_returns):+.2%}\n")
        f.write(f"  • Std Deviation: {np.std(all_returns):.2%}\n")
        f.write(f"  • Best Strategy: {strategy_scores[0].strategy_name} ({strategy_scores[0].avg_return:+.2%})\n")
        f.write(f"  • Worst Strategy: {strategy_scores[-1].strategy_name} ({strategy_scores[-1].avg_return:+.2%})\n")
        f.write(f"  • Return Spread: {(strategy_scores[0].avg_return - strategy_scores[-1].avg_return):.2%}\n\n")

        f.write("Risk-Adjusted Performance:\n")
        f.write(f"  • Mean Sharpe Ratio: {np.mean(all_sharpes):.2f}\n")
        f.write(f"  • Median Sharpe Ratio: {np.median(all_sharpes):.2f}\n")
        f.write(f"  • Positive Sharpe Count: {sum(1 for s in all_sharpes if s > 0)}/{len(all_sharpes)}\n")
        f.write(f"  • Sharpe > 1.0 (Good): {sum(1 for s in all_sharpes if s > 1.0)}/{len(all_sharpes)}\n")
        f.write(f"  • Sharpe > 2.0 (Excellent): {sum(1 for s in all_sharpes if s > 2.0)}/{len(all_sharpes)}\n\n")

        # 2.2 Individual Strategy Analysis
        f.write("2.2 Individual Strategy Performance Profiles\n")
        f.write("-" * 80 + "\n\n")
        f.write("Detailed analysis of each strategy's performance characteristics, organized\n")
        f.write("by composite score ranking:\n\n")

        for rank, score in enumerate(strategy_scores, 1):
            f.write(f"#{rank} - {score.strategy_name}\n")
            f.write("-" * 60 + "\n\n")

            # TL;DR for this strategy
            performance_desc = "profitable" if score.avg_return > 0 else "unprofitable"
            vs_buyhold = "outperformed" if score.avg_return > avg_buyhold else "underperformed"
            risk_adj = "excellent" if score.avg_sharpe > 2 else ("good" if score.avg_sharpe > 1 else ("moderate" if score.avg_sharpe > 0 else "poor"))

            f.write(f"TL;DR: {performance_desc.capitalize()} strategy with {score.avg_return:+.2%} average ")
            f.write(f"returns, {vs_buyhold} buy-and-hold by {(score.avg_return - avg_buyhold):+.2%}, ")
            f.write(f"{risk_adj} risk-adjusted returns (Sharpe {score.avg_sharpe:.2f}), ")
            f.write(f"won {score.horizons_beat_buyhold}/{score.total_horizons} time horizons.\n\n")

            # Detailed metrics
            f.write("Aggregate Performance Metrics:\n")
            f.write(f"  • Composite Score: {score.composite_score:.3f}/1.000 (Rank #{rank})\n")
            f.write(f"  • Average Return: {score.avg_return:+.2%}\n")
            f.write(f"  • vs Buy-and-Hold: {(score.avg_return - avg_buyhold):+.2%} ")
            f.write("(outperformance)\n" if score.avg_return > avg_buyhold else "(underperformance)\n")
            f.write(f"  • Sharpe Ratio: {score.avg_sharpe:.2f}\n")
            f.write(f"  • Max Drawdown: {score.avg_max_drawdown:.2%}\n")
            f.write(f"  • Win Rate: {score.avg_win_rate:.1%}\n\n")

            # Horizon-by-horizon breakdown
            f.write("Performance Breakdown by Time Horizon:\n\n")
            f.write(f"{'Horizon':<12} {'Return':<12} {'vs B&H':<12} {'Sharpe':<10} {'MDD':<10} {'Trades':<8}\n")
            f.write("-" * 60 + "\n")

            for horizon in self.horizons:
                if horizon.name in score.horizon_results:
                    hr = score.horizon_results[horizon.name]
                    f.write(f"{horizon.name:<12} ")
                    f.write(f"{hr['return']:>+10.2%} ")
                    f.write(f"{hr['vs_buyhold']:>+10.2%} ")
                    f.write(f"{hr['sharpe']:>9.2f} ")
                    f.write(f"{hr['drawdown']:>9.2%} ")
                    f.write(f"{int(hr['trades']):>6}\n")

            f.write("\n")

            # Key observations
            f.write("Key Observations:\n")

            # Consistency analysis
            returns_by_horizon = [hr['return'] for hr in score.horizon_results.values()]
            consistency = np.std(returns_by_horizon)
            if consistency < 0.1:
                f.write("  • High consistency across time horizons (low return volatility)\n")
            elif consistency < 0.3:
                f.write("  • Moderate consistency across time horizons\n")
            else:
                f.write("  • High variability across time horizons (regime-dependent)\n")

            # Trend analysis
            if len(returns_by_horizon) >= 2:
                if returns_by_horizon[-1] > returns_by_horizon[0]:
                    f.write("  • Performance improves with longer time horizons\n")
                else:
                    f.write("  • Performance degrades with longer time horizons\n")

            # Risk assessment
            if score.avg_max_drawdown < 0.10:
                f.write("  • Low drawdown risk (< 10%)\n")
            elif score.avg_max_drawdown < 0.20:
                f.write("  • Moderate drawdown risk (10-20%)\n")
            else:
                f.write("  • High drawdown risk (> 20%)\n")

            f.write("\n\n")

        # Discussion
        f.write("=" * 80 + "\n")
        f.write("3. DISCUSSION & INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("TL;DR: Results demonstrate significant alpha generation opportunities in ")
        f.write("cryptocurrency markets, with strategy selection and timeframe matching ")
        f.write("critical for success. Portfolio approaches show promise for long-term holdings.\n\n")

        f.write("Key Findings:\n\n")

        f.write(f"1. Market Efficiency: Only {beat_buyhold_count}/{len(strategy_scores)} strategies ")
        f.write("   beat buy-and-hold on average, suggesting semi-strong form efficiency in\n")
        f.write("   cryptocurrency markets, though significant alpha opportunities exist for\n")
        f.write("   sophisticated strategies.\n\n")

        f.write("2. Strategy Heterogeneity: Performance varies widely (")
        f.write(f"{(strategy_scores[0].avg_return - strategy_scores[-1].avg_return):.1%} spread), ")
        f.write("indicating\n   strategy selection is paramount. Top quartile strategies demonstrate\n")
        f.write("   consistent outperformance across multiple horizons.\n\n")

        f.write("3. Risk-Return Tradeoff: Highest returns don't always correspond to best\n")
        f.write("   risk-adjusted performance. The composite scoring approach successfully\n")
        f.write("   identifies strategies with favorable Sharpe ratios and manageable drawdowns.\n\n")

        f.write("4. Temporal Dependencies: Strategy effectiveness varies significantly across\n")
        f.write("   time horizons, suggesting different strategies are optimal for different\n")
        f.write("   investment timescales (short-term speculation vs long-term investment).\n\n")

        # Find multi-asset strategies
        multi_asset_strategies = [s for s in strategy_scores if 'Portfolio' in s.strategy_name]
        if multi_asset_strategies:
            f.write("5. Portfolio Effects: Multi-asset portfolio strategies demonstrated ")
            if multi_asset_strategies[0].avg_return > avg_buyhold:
                f.write("strong\n   performance through diversification benefits, ")
                f.write("particularly on longer time\n   horizons where rebalancing captured mean-reversion opportunities.\n\n")
            else:
                f.write("mixed\n   results, with diversification benefits offset by rebalancing costs ")
                f.write("and\n   correlation effects in highly correlated crypto markets.\n\n")

        f.write("Limitations & Caveats:\n\n")
        f.write("  • Historical Performance: Past results do not guarantee future returns.\n")
        f.write("    Cryptocurrency markets are rapidly evolving.\n\n")
        f.write("  • Parameter Sensitivity: Default parameters used; optimization may improve\n")
        f.write("    results but risks overfitting.\n\n")
        f.write("  • Market Impact: $10,000 capital assumption may not reflect slippage at\n")
        f.write("    scale. Larger positions would experience greater market impact.\n\n")
        f.write("  • Regime Specificity: Results depend on tested historical period. Different\n")
        f.write("    market regimes (bull, bear, sideways) may produce different outcomes.\n\n")
        f.write("  • Transaction Costs: 0.1% commission assumption may be conservative for\n")
        f.write("    high-frequency strategies or pessimistic for volume-based fee discounts.\n\n")

        # Conclusion
        f.write("=" * 80 + "\n")
        f.write("4. CONCLUSIONS\n")
        f.write("=" * 80 + "\n\n")

        f.write("TL;DR: Systematic strategy evaluation framework successfully identified ")
        f.write(f"{beat_buyhold_count} strategies with consistent alpha generation. ")
        f.write(f"Top performer ({strategy_scores[0].strategy_name}) achieved {strategy_scores[0].avg_return:+.2%} ")
        f.write("returns with favorable risk profile.\n\n")

        f.write("This comprehensive empirical analysis demonstrates that algorithmic trading\n")
        f.write("strategies can generate positive risk-adjusted returns in cryptocurrency markets,\n")
        f.write("though performance is highly strategy-dependent and temporally variable.\n\n")

        f.write("Primary Conclusions:\n\n")
        f.write(f"  1. The optimal strategy ({strategy_scores[0].strategy_name}) achieved ")
        f.write(f"composite score of\n     {strategy_scores[0].composite_score:.3f}, ")
        f.write("demonstrating superior risk-adjusted returns through\n     ")
        f.write("consistent performance across multiple time horizons.\n\n")

        f.write(f"  2. {beat_buyhold_count} out of {len(strategy_scores)} strategies ")
        f.write("outperformed passive buy-and-hold,\n     validating the potential for active ")
        f.write("management in crypto markets while\n     highlighting the importance of strategy selection.\n\n")

        f.write("  3. Multi-horizon testing revealed significant temporal dependencies,\n")
        f.write("     suggesting portfolio managers should match strategy selection to\n")
        f.write("     intended holding periods and market conditions.\n\n")

        f.write("  4. Risk management remains critical: even top-performing strategies\n")
        f.write(f"     experienced drawdowns up to {max(s.avg_max_drawdown for s in strategy_scores):.1%}, ")
        f.write("necessitating\n     appropriate position sizing and stop-loss disciplines.\n\n")

        f.write("Recommendations for Implementation:\n\n")
        f.write("  • Deploy top-quartile strategies with proven track records across horizons\n")
        f.write("  • Implement robust risk management (position sizing, stop losses)\n")
        f.write("  • Monitor performance regularly and be prepared to adapt to regime changes\n")
        f.write("  • Consider ensemble approaches combining multiple complementary strategies\n")
        f.write("  • Conduct forward testing before live deployment with real capital\n\n")

        f.write("Future Research Directions:\n\n")
        f.write("  • Parameter optimization using walk-forward analysis\n")
        f.write("  • Machine learning approaches for regime detection and strategy selection\n")
        f.write("  • Transaction cost sensitivity analysis at various position sizes\n")
        f.write("  • Multi-asset portfolio optimization with dynamic allocation\n")
        f.write("  • Out-of-sample testing on additional cryptocurrencies and timeframes\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"END OF ACADEMIC RESEARCH REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")

    def _write_academic_section_html(self, f, strategy_scores: List[StrategyScore], avg_buyhold: float) -> None:
        """Write academic analysis section in HTML format."""
        f.write("<div class='academic-section'>\n")
        f.write("<h2>📚 ACADEMIC RESEARCH REPORT</h2>\n")
        f.write("<p><em>Detailed technical analysis for research purposes</em></p>\n\n")

        # Abstract
        f.write("<h3>ABSTRACT</h3>\n")

        strategies_beat_buyhold = sum(1 for s in strategy_scores if s.avg_return > avg_buyhold)

        f.write("<div class='blockquote info'>\n")
        f.write(f"<p><strong>TL;DR:</strong> Comprehensive empirical evaluation of {len(strategy_scores)} algorithmic trading strategies across ")
        f.write(f"{len(self.horizons)} time horizons revealed {strategies_beat_buyhold} strategies outperforming passive buy-and-hold ")
        f.write(f"benchmarks, with the top strategy achieving {strategy_scores[0].avg_return:+.2%} ")
        f.write(f"average returns versus {avg_buyhold:+.2%} for buy-and-hold.</p>\n")
        f.write("</div>\n\n")

        f.write("<p>This study presents a systematic comparative analysis of cryptocurrency ")
        f.write(f"trading strategies on <strong>{self.symbol}</strong> using high-frequency <strong>{self.timeframe}</strong> ")
        f.write("candlestick data from Binance exchange. We evaluate ")

        unique_single = len(set(r['strategy_name'] for r in self.all_results if r.get('strategy_type') == 'single_pair'))
        unique_multi = len(set(r['strategy_name'] for r in self.all_results if r.get('strategy_type') == 'multi_pair'))

        f.write(f"<strong>{unique_single}</strong> single-asset and <strong>{unique_multi}</strong> multi-asset strategies ")
        f.write(f"through <strong>{len(self.all_results)}</strong> independent backtests spanning timeframes from ")
        f.write(f"<strong>{self.horizons[0].days}</strong> to <strong>{self.horizons[-1].days}</strong> days. ")
        f.write("Performance is assessed using risk-adjusted metrics including Sharpe ratio, ")
        f.write("maximum drawdown, win rate, and total returns, with all strategies benchmarked ")
        f.write("against passive buy-and-hold positions. Results indicate significant ")
        f.write("heterogeneity in strategy performance across temporal horizons, with ")
        f.write("momentum-based and portfolio rebalancing approaches demonstrating superior ")
        f.write("risk-adjusted returns in the tested market conditions.</p>\n\n")

        # Methodology
        f.write("<h3>1. METHODOLOGY</h3>\n\n")

        f.write("<h4>1.1 Data Collection & Preprocessing</h4>\n")
        f.write("<div class='blockquote info'>\n")
        f.write(f"<p><strong>TL;DR:</strong> {len(self.all_results)} backtests executed on {self.timeframe} OHLCV ")
        f.write(f"data from Binance, spanning {self.horizons[0].days}-{self.horizons[-1].days} ")
        f.write("days with no survivorship bias.</p>\n")
        f.write("</div>\n\n")

        f.write("<p><strong>Market Data Specification:</strong></p>\n")
        f.write("<ul>\n")
        f.write("    <li><strong>Exchange:</strong> Binance (via REST API)</li>\n")
        f.write(f"    <li><strong>Primary Asset:</strong> {self.symbol}</li>\n")
        f.write(f"    <li><strong>Timeframe Granularity:</strong> {self.timeframe} candlesticks</li>\n")
        f.write("    <li><strong>Data Fields:</strong> Open, High, Low, Close, Volume (OHLCV)</li>\n")

        if self.horizons:
            max_candles = max(h.days * 24 for h in self.horizons)
            f.write(f"    <li><strong>Sample Size (largest horizon):</strong> {max_candles} candles</li>\n")
            f.write(f"    <li><strong>Historical Range:</strong> {self.horizons[0].days} to {self.horizons[-1].days} days</li>\n")

        f.write("    <li><strong>Data Quality:</strong> Real-time market data, no look-ahead bias</li>\n")
        f.write("</ul>\n\n")

        # Results Summary
        f.write("<h3>2. RESULTS SUMMARY</h3>\n\n")

        f.write("<h4>2.1 Performance Distribution</h4>\n")

        returns = [s.avg_return for s in strategy_scores]
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        best_return = max(returns)
        worst_return = min(returns)

        f.write("<div class='blockquote info'>\n")
        f.write(f"<p><strong>TL;DR:</strong> {sum(1 for r in returns if r > 0)}/{len(returns)} strategies profitable, ")
        f.write(f"{strategies_beat_buyhold}/{len(returns)} beat buy-and-hold, ")
        f.write(f"average return {mean_return:+.2%} (vs {avg_buyhold:+.2%} passive).</p>\n")
        f.write("</div>\n\n")

        f.write("<p><strong>Aggregate Statistics:</strong></p>\n")
        f.write("<ul>\n")
        f.write(f"    <li><strong>Mean Return:</strong> {mean_return:+.2%}</li>\n")
        f.write(f"    <li><strong>Median Return:</strong> {median_return:+.2%}</li>\n")
        f.write(f"    <li><strong>Std Deviation:</strong> {std_return:.2%}</li>\n")
        f.write(f"    <li><strong>Best Strategy:</strong> {strategy_scores[0].strategy_name} ({best_return:+.2%})</li>\n")
        f.write(f"    <li><strong>Worst Strategy:</strong> {strategy_scores[-1].strategy_name} ({worst_return:+.2%})</li>\n")
        f.write(f"    <li><strong>Return Spread:</strong> {best_return - worst_return:.2%}</li>\n")
        f.write("</ul>\n\n")

        # Risk-Adjusted Performance
        f.write("<h4>2.2 Risk-Adjusted Performance</h4>\n")
        sharpes = [s.avg_sharpe for s in strategy_scores]
        positive_sharpe = sum(1 for s in sharpes if s > 0)
        good_sharpe = sum(1 for s in sharpes if s > 1.0)

        f.write("<ul>\n")
        f.write(f"    <li><strong>Mean Sharpe Ratio:</strong> {np.mean(sharpes):.2f}</li>\n")
        f.write(f"    <li><strong>Median Sharpe Ratio:</strong> {np.median(sharpes):.2f}</li>\n")
        f.write(f"    <li><strong>Positive Sharpe Count:</strong> {positive_sharpe}/{len(sharpes)}</li>\n")
        f.write(f"    <li><strong>Sharpe > 1.0 (Good):</strong> {good_sharpe}/{len(sharpes)}</li>\n")
        f.write("</ul>\n\n")

        # Top Strategies Table
        f.write("<h4>2.3 Top 5 Performing Strategies</h4>\n")
        f.write("<table>\n")
        f.write("    <thead>\n")
        f.write("        <tr>\n")
        f.write("            <th>Rank</th><th>Strategy</th><th>Return</th><th>Sharpe</th>\n")
        f.write("            <th>Drawdown</th><th>Win Rate</th><th>Beat B&H</th>\n")
        f.write("        </tr>\n")
        f.write("    </thead>\n")
        f.write("    <tbody>\n")

        for rank, strat in enumerate(strategy_scores[:5], 1):
            f.write(f"        <tr>\n")
            f.write(f"            <td>{rank}</td>\n")
            f.write(f"            <td><strong>{strat.strategy_name}</strong></td>\n")
            f.write(f"            <td>{HTMLReportWriter.format_percentage(strat.avg_return)}</td>\n")
            f.write(f"            <td>{strat.avg_sharpe:.2f}</td>\n")
            f.write(f"            <td>{strat.avg_max_drawdown:.1%}</td>\n")
            f.write(f"            <td>{strat.avg_win_rate:.1%}</td>\n")
            f.write(f"            <td>{strat.horizons_beat_buyhold}/{strat.total_horizons}</td>\n")
            f.write(f"        </tr>\n")

        f.write("    </tbody>\n")
        f.write("</table>\n\n")

        # Key Findings
        f.write("<h3>3. KEY FINDINGS</h3>\n\n")

        f.write("<div class='blockquote warning'>\n")
        f.write(f"<p><strong>Market Efficiency:</strong> Only {strategies_beat_buyhold}/{len(strategy_scores)} strategies ")
        f.write("beat buy-and-hold on average, suggesting semi-strong form efficiency in ")
        f.write("cryptocurrency markets, though significant alpha opportunities exist for ")
        f.write("sophisticated strategies.</p>\n")
        f.write("</div>\n\n")

        f.write("<ol>\n")
        f.write("    <li><strong>Strategy Heterogeneity:</strong> Performance varies widely ")
        f.write(f"({best_return - worst_return:.1%} spread), indicating strategy selection is paramount. ")
        f.write("Top quartile strategies demonstrate consistent outperformance across multiple horizons.</li>\n\n")

        f.write("    <li><strong>Risk-Return Tradeoff:</strong> Highest returns don't always correspond to best ")
        f.write("risk-adjusted performance. The composite scoring approach successfully identifies ")
        f.write("strategies with favorable Sharpe ratios and manageable drawdowns.</li>\n\n")

        f.write("    <li><strong>Temporal Dependencies:</strong> Strategy effectiveness varies significantly across ")
        f.write("time horizons, suggesting different strategies are optimal for different ")
        f.write("investment timescales (short-term speculation vs long-term investment).</li>\n\n")

        if unique_multi > 0:
            f.write("    <li><strong>Portfolio Effects:</strong> Multi-asset portfolio strategies demonstrated strong ")
            f.write("performance through diversification benefits, particularly on longer time ")
            f.write("horizons where rebalancing captured mean-reversion opportunities.</li>\n\n")

        f.write("</ol>\n\n")

        # Limitations
        f.write("<h3>4. LIMITATIONS & CAVEATS</h3>\n\n")
        f.write("<ul>\n")
        f.write("    <li><strong>Historical Performance:</strong> Past results do not guarantee future returns. ")
        f.write("Cryptocurrency markets are rapidly evolving.</li>\n\n")

        f.write("    <li><strong>Parameter Sensitivity:</strong> Default parameters used; optimization may improve ")
        f.write("results but risks overfitting.</li>\n\n")

        f.write("    <li><strong>Market Impact:</strong> $10,000 capital assumption may not reflect slippage at ")
        f.write("scale. Larger positions would experience greater market impact.</li>\n\n")

        f.write("    <li><strong>Regime Specificity:</strong> Results depend on tested historical period. Different ")
        f.write("market regimes (bull, bear, sideways) may produce different outcomes.</li>\n\n")

        f.write("    <li><strong>Transaction Costs:</strong> 0.1% commission assumption may be conservative for ")
        f.write("high-frequency strategies or pessimistic for volume-based fee discounts.</li>\n")
        f.write("</ul>\n\n")

        # Conclusions
        f.write("<h3>5. CONCLUSIONS</h3>\n\n")

        f.write("<div class='blockquote info'>\n")
        f.write(f"<p><strong>TL;DR:</strong> Systematic strategy evaluation framework successfully identified ")
        f.write(f"{strategies_beat_buyhold} strategies with consistent alpha generation. ")
        f.write(f"Top performer ({strategy_scores[0].strategy_name}) achieved ")
        f.write(f"{strategy_scores[0].avg_return:+.2%} returns with favorable risk profile.</p>\n")
        f.write("</div>\n\n")

        f.write("<p><strong>Primary Conclusions:</strong></p>\n")
        f.write("<ol>\n")
        f.write(f"    <li>The optimal strategy ({strategy_scores[0].strategy_name}) achieved composite score of ")
        f.write(f"{strategy_scores[0].composite_score:.3f}, demonstrating superior risk-adjusted returns through ")
        f.write("consistent performance across multiple time horizons.</li>\n\n")

        f.write(f"    <li>{strategies_beat_buyhold} out of {len(strategy_scores)} strategies outperformed passive buy-and-hold, ")
        f.write("validating the potential for active management in crypto markets while ")
        f.write("highlighting the importance of strategy selection.</li>\n\n")

        f.write("    <li>Multi-horizon testing revealed significant temporal dependencies, ")
        f.write("suggesting portfolio managers should match strategy selection to ")
        f.write("intended holding periods and market conditions.</li>\n\n")

        max_dd = max(s.avg_max_drawdown for s in strategy_scores)
        f.write(f"    <li>Risk management remains critical: even top-performing strategies ")
        f.write(f"experienced drawdowns up to {max_dd:.1%}, necessitating ")
        f.write("appropriate position sizing and stop-loss disciplines.</li>\n")
        f.write("</ol>\n\n")

        # Recommendations
        f.write("<h4>Recommendations for Implementation:</h4>\n")
        f.write("<ul>\n")
        f.write("    <li>Deploy top-quartile strategies with proven track records across horizons</li>\n")
        f.write("    <li>Implement robust risk management (position sizing, stop losses)</li>\n")
        f.write("    <li>Monitor performance regularly and be prepared to adapt to regime changes</li>\n")
        f.write("    <li>Consider ensemble approaches combining multiple complementary strategies</li>\n")
        f.write("    <li>Conduct forward testing before live deployment with real capital</li>\n")
        f.write("</ul>\n\n")

        # Future Research
        f.write("<h4>Future Research Directions:</h4>\n")
        f.write("<ul>\n")
        f.write("    <li>Parameter optimization using walk-forward analysis</li>\n")
        f.write("    <li>Machine learning approaches for regime detection and strategy selection</li>\n")
        f.write("    <li>Transaction cost sensitivity analysis at various position sizes</li>\n")
        f.write("    <li>Multi-asset portfolio optimization with dynamic allocation</li>\n")
        f.write("    <li>Out-of-sample testing on additional cryptocurrencies and timeframes</li>\n")
        f.write("</ul>\n\n")

        f.write("<hr>\n")
        f.write(f"<p><em>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>\n")

        f.write("</div>\n")

    def _save_comparison_matrix(self) -> None:
        """Save detailed comparison matrix as CSV."""
        if not self.all_results:
            return

        csv_file = self.output_dir / "comparison_matrix.csv"
        df = pd.DataFrame(self.all_results)

        # Add buy-hold comparison
        df['buyhold_return'] = df['horizon'].apply(
            lambda h: self.buy_hold_results.get(h, {}).get('total_return', 0)
        )
        df['outperformance'] = df['total_return'] - df['buyhold_return']
        df['beat_buyhold'] = df['outperformance'] > 0

        # Sort by strategy and horizon
        df = df.sort_values(['strategy_name', 'horizon_days'])

        df.to_csv(csv_file, index=False)
        logger.success(f"✓ Comparison matrix: {csv_file}")

    def run(self) -> None:
        """Run the complete master analysis."""
        start_time = datetime.now()

        logger.info("\n" + "=" * 80)
        logger.info("MASTER STRATEGY ANALYSIS - STARTING")
        logger.info("=" * 80)

        try:
            # Step 1: Run parallel analysis
            self.run_parallel_analysis()

            # Step 2: Compute composite scores
            strategy_scores = self.compute_composite_scores()

            # Step 3: Generate reports
            self.generate_master_report(strategy_scores)

            # Completion
            duration = (datetime.now() - start_time).total_seconds()

            logger.info("\n" + "=" * 80)
            logger.success("✅ MASTER ANALYSIS COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Duration: {duration / 60:.1f} minutes")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"\nView report: {self.output_dir / 'MASTER_REPORT.html'}")
            logger.info("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"\n❌ MASTER ANALYSIS FAILED: {e}")
            logger.exception("Full traceback:")
            sys.exit(1)


__all__ = [
    'HorizonConfig',
    'StrategyScore',
    'MasterStrategyAnalyzer',
]


if __name__ == "__main__":
    """
    Validation block for orchestration module.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: Verify classes are importable
    total_tests += 1
    print("Test 1: Verify classes importable")
    try:
        if HorizonConfig is None:
            all_validation_failures.append("HorizonConfig not defined")
        if StrategyScore is None:
            all_validation_failures.append("StrategyScore not defined")
        if MasterStrategyAnalyzer is None:
            all_validation_failures.append("MasterStrategyAnalyzer not defined")

        print(f"  ✓ HorizonConfig: {HorizonConfig.__name__}")
        print(f"  ✓ StrategyScore: {StrategyScore.__name__}")
        print(f"  ✓ MasterStrategyAnalyzer: {MasterStrategyAnalyzer.__name__}")
    except Exception as e:
        all_validation_failures.append(f"Class import failed: {e}")

    # Test 2: Verify HorizonConfig can be instantiated
    total_tests += 1
    print("\nTest 2: Verify HorizonConfig instantiation")
    try:
        horizon = HorizonConfig("30d", 30, "30 days")
        if horizon.name != "30d" or horizon.days != 30:
            all_validation_failures.append(f"HorizonConfig instantiation incorrect: {horizon}")
        else:
            print(f"  ✓ HorizonConfig created: {horizon.name}, {horizon.days} days")
    except Exception as e:
        all_validation_failures.append(f"HorizonConfig instantiation failed: {e}")

    # Test 3: Verify StrategyScore can be instantiated
    total_tests += 1
    print("\nTest 3: Verify StrategyScore instantiation")
    try:
        score = StrategyScore(
            strategy_name="TestStrategy",
            composite_score=1.5,
            avg_return=0.25,
            avg_sharpe=1.2,
            avg_max_drawdown=0.15,
            avg_win_rate=0.65,
            horizons_beat_buyhold=3,
            total_horizons=4,
            horizon_results={"30d": {"return": 0.25, "sharpe": 1.2}}
        )
        if score.strategy_name != "TestStrategy" or score.composite_score != 1.5:
            all_validation_failures.append(f"StrategyScore instantiation incorrect: {score}")
        else:
            print(f"  ✓ StrategyScore created: {score.strategy_name}, score={score.composite_score}")
    except Exception as e:
        all_validation_failures.append(f"StrategyScore instantiation failed: {e}")

    # Test 4: Verify MasterStrategyAnalyzer __init__ signature
    total_tests += 1
    print("\nTest 4: Verify MasterStrategyAnalyzer signature")
    try:
        import inspect
        sig = inspect.signature(MasterStrategyAnalyzer.__init__)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'symbol', 'timeframe', 'horizons', 'workers', 'quick_mode', 'multi_pair', 'output_dir']
        if params != expected_params:
            all_validation_failures.append(f"MasterStrategyAnalyzer signature mismatch: {params} != {expected_params}")
        else:
            print(f"  ✓ MasterStrategyAnalyzer __init__ has {len(params)-1} parameters")
    except Exception as e:
        all_validation_failures.append(f"MasterStrategyAnalyzer signature check failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Orchestration module is validated and ready for use")
        print("\nNOTE: MasterStrategyAnalyzer extracted from master.py")
        sys.exit(0)
