"""
Backtest Worker Functions

This module contains the worker functions for parallel backtest execution.
These functions are designed to be picklable for multiprocessing.

**Purpose**: Execute backtests in parallel worker processes

**Key Functions**:
- run_backtest_worker: Single-pair strategy execution
- run_multipair_backtest_worker: Multi-pair strategy execution

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- concurrent.futures: https://docs.python.org/3/library/concurrent.futures.html

**Sample Input**:
Worker functions are called by ProcessPoolExecutor with strategy config and data.

**Expected Output**:
Dictionary containing backtest metrics or error information.

Extracted from master.py (lines 594-1575) during Phase 2.5 refactoring.
Total: ~980 lines of worker code moved to execution module.
"""

import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

# Import utility functions from execution module
from crypto_trader.execution.data_utils import (
    add_required_indicators,
    slice_data_to_horizon
)
from crypto_trader.execution.metric_utils import (
    periods_per_year_from_timeframe,
    calculate_sharpe_ratio_safe
)
from crypto_trader.execution.error_utils import format_error_message
from crypto_trader.execution.logging_utils import (
    log_dataframe_info,
    log_worker_lifecycle,
    log_error_with_context
)


def run_backtest_worker(
    strategy_name: str,
    data_dict: Dict[str, Any],
    horizon_name: str,
    horizon_days: int,
    symbol: str,
    timeframe: str,
    default_params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Worker function for single-pair backtest execution with comprehensive logging."""
    worker_id = f"{strategy_name}_{horizon_name}_{threading.get_ident()}"
    start_time = time.perf_counter()

    try:
        log_worker_lifecycle(worker_id, "STARTED",
                           strategy=strategy_name,
                           horizon=horizon_name,
                           symbol=symbol)

        # Import inside worker to avoid pickle issues
        logger.debug(f"[WORKER-{worker_id}] Setting up imports and environment")

        # Ensure imports are available
        script_dir = Path(__file__).resolve().parent.parent.parent.parent
        src_dir = script_dir / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            logger.debug(f"[WORKER-{worker_id}] Added {src_dir} to sys.path")

        from crypto_trader.strategies import get_registry
        from crypto_trader.backtesting.engine import BacktestEngine
        from crypto_trader.core.config import BacktestConfig
        from crypto_trader.core.types import Timeframe

        logger.debug(f"[WORKER-{worker_id}] Recreating DataFrame from dict")
        # Recreate DataFrame from dict
        data = pd.DataFrame(data_dict)
        log_dataframe_info(data, f"WORKER-{worker_id} Initial Data", detailed=False)

        logger.debug(f"[WORKER-{worker_id}] Slicing data to horizon window ({horizon_days} days)")
        # CRITICAL: Slice data to correct horizon window (consistent with multi-pair workers)
        data = slice_data_to_horizon(data, timeframe, horizon_days, warmup_multiplier=1.5)
        log_dataframe_info(data, f"WORKER-{worker_id} After Slicing", detailed=False)

        # Get strategy class with error handling
        logger.debug(f"[WORKER-{worker_id}] Loading strategy: {strategy_name}")
        try:
            import crypto_trader.strategies.library  # noqa: F401
        except ImportError as e:
            error_msg = f'Failed to import strategies library: {e}'
            logger.error(f"[WORKER-{worker_id}] {error_msg}")
            return {
                'strategy_name': strategy_name,
                'horizon': horizon_name,
                'error': error_msg
            }

        registry = get_registry()
        strategy_class = registry.get_strategy(strategy_name)
        logger.debug(f"[WORKER-{worker_id}] Retrieved strategy class: {strategy_class.__name__}")

        # Normalize configuration parameters
        config_params = default_params or {}
        logger.debug(f"[WORKER-{worker_id}] Config params: {config_params}")

        # Check strategy __init__ signature to instantiate correctly
        import inspect
        init_signature = inspect.signature(strategy_class.__init__)
        params = list(init_signature.parameters.keys())
        logger.debug(f"[WORKER-{worker_id}] Strategy __init__ params: {params}")

        # If __init__ accepts name/config, pass them (e.g., old-style strategies)
        if 'name' in params and 'config' in params:
            logger.debug(f"[WORKER-{worker_id}] Instantiating with name/config (old-style)")
            strategy = strategy_class(name=strategy_name, config=config_params)
        else:
            logger.debug(f"[WORKER-{worker_id}] Instantiating without args (SOTA 2025 style)")
            # SOTA 2025 strategies: instantiate without args
            strategy = strategy_class()

        # ALWAYS call initialize() if it exists, regardless of how we instantiated
        # SOTA 2025 strategies need initialize() to set self._initialized = True
        if hasattr(strategy, 'initialize') and callable(getattr(strategy, 'initialize')):
            logger.debug(f"[WORKER-{worker_id}] Calling strategy.initialize()")
            strategy.initialize(config_params)
        else:
            logger.debug(f"[WORKER-{worker_id}] No initialize() method found")

        # Prepare data
        logger.debug(f"[WORKER-{worker_id}] Preparing data with timestamps and indicators")
        data_with_timestamp = data.reset_index(drop=True)
        if 'timestamp' not in data_with_timestamp.columns and hasattr(data, 'index'):
            data_with_timestamp['timestamp'] = data.index
            logger.debug(f"[WORKER-{worker_id}] Added timestamp column from index")

        # Ensure timestamp column is datetime for downstream consumers
        if 'timestamp' in data_with_timestamp.columns:
            data_with_timestamp['timestamp'] = pd.to_datetime(data_with_timestamp['timestamp'])
            logger.debug(f"[WORKER-{worker_id}] Converted timestamp to datetime")

        # Add any required indicators the strategy expects
        logger.debug(f"[WORKER-{worker_id}] Adding required indicators")
        data_with_timestamp = add_required_indicators(strategy, data_with_timestamp)
        if 'timestamp' in data_with_timestamp.columns:
            data_with_timestamp = data_with_timestamp.sort_values('timestamp').reset_index(drop=True)

        log_dataframe_info(data_with_timestamp, f"WORKER-{worker_id} Final Prepared Data", detailed=False)

        # Create backtest config
        logger.debug(f"[WORKER-{worker_id}] Creating backtest config")
        config = BacktestConfig(
            initial_capital=10000.0,
            trading_fee_percent=0.001,
            slippage_percent=0.0005,
        )

        # Create engine
        logger.debug(f"[WORKER-{worker_id}] Creating backtest engine")
        engine = BacktestEngine()

        # Convert timeframe string to enum
        timeframe_mapping = {
            "1m": Timeframe.MINUTE_1,
            "5m": Timeframe.MINUTE_5,
            "15m": Timeframe.MINUTE_15,
            "1h": Timeframe.HOUR_1,
            "4h": Timeframe.HOUR_4,
            "1d": Timeframe.DAY_1,
            "1w": Timeframe.WEEK_1,
        }
        timeframe_enum = timeframe_mapping.get(timeframe, Timeframe.HOUR_1)
        logger.debug(f"[WORKER-{worker_id}] Timeframe: {timeframe} -> {timeframe_enum}")

        # Run backtest
        logger.info(f"[WORKER-{worker_id}] ⏳ Running backtest...")
        backtest_start = time.perf_counter()
        result = engine.run_backtest(
            strategy=strategy,
            data=data_with_timestamp,
            config=config,
            symbol=symbol.replace("/", ""),
            timeframe=timeframe_enum,
        )
        backtest_duration = time.perf_counter() - backtest_start
        logger.info(f"[WORKER-{worker_id}] Backtest completed in {backtest_duration:.2f}s")

        # Extract and return serializable metrics
        logger.debug(f"[WORKER-{worker_id}] Extracting metrics")
        metrics_dict = {
            'strategy_name': strategy_name,
            'strategy_type': 'single_pair',
            'symbol': symbol,
            'horizon': horizon_name,
            'horizon_days': horizon_days,
            'total_return': result.metrics.total_return,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'max_drawdown': result.metrics.max_drawdown,
            'win_rate': result.metrics.win_rate,
            'total_trades': result.metrics.total_trades,
            'profit_factor': result.metrics.profit_factor,
            'final_capital': result.metrics.final_capital,
        }

        # Log summary
        duration = time.perf_counter() - start_time
        log_worker_lifecycle(worker_id, "COMPLETED",
                           duration_s=f"{duration:.2f}",
                           return_pct=f"{result.metrics.total_return:.2f}%",
                           sharpe=f"{result.metrics.sharpe_ratio:.2f}",
                           trades=result.metrics.total_trades)

        logger.info(f"[WORKER-{worker_id}] 📊 Results: Return={result.metrics.total_return:.2f}%, "
                   f"Sharpe={result.metrics.sharpe_ratio:.2f}, Trades={result.metrics.total_trades}")

        return metrics_dict

    except Exception as e:
        # Return error info with traceback for debugging
        duration = time.perf_counter() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        error_trace = traceback.format_exc()

        log_worker_lifecycle(worker_id, "FAILED", duration_s=f"{duration:.2f}", error=error_msg)

        error_context = {
            'strategy_name': strategy_name,
            'horizon': horizon_name,
            'symbol': symbol,
            'data_shape': f"{len(data_dict.get('timestamp', []))} rows" if 'timestamp' in data_dict else "unknown"
        }
        log_error_with_context(e, error_context, include_traceback=True)

        return {
            'strategy_name': strategy_name,
            'horizon': horizon_name,
            'error': error_msg
        }


def run_multipair_backtest_worker(
    strategy_name: str,
    asset_symbols: List[str],
    data_dicts: Dict[str, Dict[str, Any]],
    horizon_name: str,
    horizon_days: int,
    timeframe: str,
    default_params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Worker function for multi-pair backtest execution.

    Handles strategies like Portfolio Rebalancer and Statistical Arbitrage
    that require multiple asset pairs.
    """
    try:
        # Import inside worker to avoid pickle issues
        import yaml

        # Ensure imports are available
        script_dir = Path(__file__).resolve().parent.parent.parent.parent
        src_dir = script_dir / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from crypto_trader.data.fetchers import BinanceDataFetcher

        # For multi-pair strategies, we need to use the pipeline
        # Create a temporary config file
        if strategy_name == "PortfolioRebalancer":
            from tempfile import TemporaryDirectory

            equal_weight = 1.0 / len(asset_symbols)
            assets = [{'symbol': symbol, 'weight': equal_weight} for symbol in asset_symbols]

            try:
                with TemporaryDirectory(prefix="portfolio_backtest_") as tmp_dir:
                    temp_path = Path(tmp_dir)
                    output_dir = temp_path / "output"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    config = {
                        'run': {
                            'name': f'portfolio_{horizon_name}',
                            'mode': 'portfolio',
                            'description': f'Portfolio Rebalancer backtest for {", ".join(asset_symbols)} over {horizon_days} days'
                        },
                        'data': {
                            'timeframe': timeframe,
                            'days': horizon_days
                        },
                        'portfolio': {
                            'assets': assets,
                            'rebalancing': {
                                'enabled': True,
                                'threshold': default_params.get('threshold', 0.10),
                                'rebalance_method': default_params.get('rebalance_method', 'threshold'),
                                'min_rebalance_interval_hours': default_params.get('min_rebalance_interval_hours', 24),
                                'use_momentum_filter': default_params.get('use_momentum_filter', False)
                            }
                        },
                        'capital': {
                            'initial_capital': 10000.0
                        },
                        'costs': {
                            'commission': 0.001,
                            'slippage': 0.0005
                        },
                        'output': {
                            'directory': str(output_dir),
                            'save_trades': False,
                            'save_equity_curve': False
                        }
                    }

                    config_path = temp_path / "config.yaml"
                    with open(config_path, "w", encoding="utf-8") as f:
                        yaml.dump(config, f)

                    # Import run_full_pipeline which has portfolio functionality
                    module_dir = str(Path(__file__).parent.parent.parent.parent)
                    if module_dir not in sys.path:
                        sys.path.insert(0, module_dir)

                    # Import the runner
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "pipeline", Path(module_dir) / "run_full_pipeline.py"
                    )
                    pipeline_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(pipeline_module)

                    FullPipelineRunner = pipeline_module.FullPipelineRunner

                    # Create runner
                    runner = FullPipelineRunner(
                        symbol="PORTFOLIO",
                        timeframe=timeframe,
                        days=horizon_days,
                        initial_capital=10000.0,
                        output_dir=str(output_dir)
                    )

                    # Run portfolio mode
                    runner.run_portfolio_mode(str(config_path), generate_enhanced=False)

                    # Load results from the generated files
                    equity_file = output_dir / "data" / "portfolio_equity_curve.csv"
                    if equity_file.exists():
                        equity_df = pd.read_csv(equity_file)
                        initial_value = equity_df['total_value'].iloc[0]
                        final_value = equity_df['total_value'].iloc[-1]
                        portfolio_return = (final_value - initial_value) / initial_value

                        # Calculate Sharpe with proper annualisation
                        returns = equity_df['total_value'].pct_change().dropna()
                        periods_per_year = periods_per_year_from_timeframe(timeframe)
                        sharpe = calculate_sharpe_ratio_safe(returns, periods_per_year)

                        # Calculate max drawdown
                        cumulative = equity_df['total_value'].values
                        running_max = pd.Series(cumulative).cummax()
                        drawdown = (cumulative - running_max) / running_max
                        max_dd = float(abs(drawdown.min()))

                        # Count rebalances
                        rebalance_file = output_dir / "data" / "rebalance_events.csv"
                        rebalance_count = 0
                        if rebalance_file.exists():
                            rebalance_df = pd.read_csv(rebalance_file)
                            rebalance_count = len(rebalance_df)

                        # Calculate win rate for portfolio (% of positive return periods)
                        positive_periods = (returns > 0).sum()
                        total_periods = len(returns)
                        win_rate = float(positive_periods / total_periods) if total_periods > 0 else 0.0

                        # Extract metrics
                        return {
                            'strategy_name': strategy_name,
                            'strategy_type': 'multi_pair',
                            'symbol': f"Portfolio[{len(asset_symbols)} assets]",
                            'symbols': ', '.join(asset_symbols),
                            'num_assets': len(asset_symbols),
                            'horizon': horizon_name,
                            'horizon_days': horizon_days,
                            'total_return': portfolio_return,
                            'sharpe_ratio': sharpe,
                            'max_drawdown': max_dd,
                            'win_rate': win_rate,
                            'total_trades': rebalance_count,  # Count rebalances as "trades"
                            'profit_factor': 0.0,  # N/A for portfolio
                            'final_capital': final_value,
                        }

                    return {
                        'strategy_name': strategy_name,
                        'horizon': horizon_name,
                        'error': 'Portfolio equity file not generated'
                    }
            except Exception as inner_e:
                # Capture and format error with context
                error_details = traceback.format_exc()
                error_msg = f'{str(inner_e)}\n{error_details}'
                return {
                    'strategy_name': strategy_name,
                    'horizon': horizon_name,
                    'error': format_error_message(error_msg, 'Portfolio execution error', max_length=500)
                }

        elif strategy_name == "StatisticalArbitrage":
            # Statistical arbitrage needs pairs
            if len(asset_symbols) < 2:
                return {
                    'strategy_name': strategy_name,
                    'horizon': horizon_name,
                    'error': 'Statistical Arbitrage requires at least 2 assets'
                }

            # Use first two symbols as pair
            pair = asset_symbols[:2]

            try:
                from crypto_trader.strategies import get_registry
                from crypto_trader.strategies.base import SignalType
                from crypto_trader.features.factory import augment_with_features, DEFAULT_JOIN_CONFIG

                # Use pre-fetched data from shared pool (Bug #1 fix)
                if pair[0] not in data_dicts or pair[1] not in data_dicts:
                    return {
                        'strategy_name': strategy_name,
                        'horizon': horizon_name,
                        'error': f'Pre-fetched data not available for {pair[0]} or {pair[1]}'
                    }

                # Reconstruct DataFrames from dicts
                asset1_data = pd.DataFrame(data_dicts[pair[0]])
                asset2_data = pd.DataFrame(data_dicts[pair[1]])

                # Set timestamp as index
                asset1_data['timestamp'] = pd.to_datetime(asset1_data['timestamp'])
                asset2_data['timestamp'] = pd.to_datetime(asset2_data['timestamp'])
                asset1_data = asset1_data.set_index('timestamp')
                asset2_data = asset2_data.set_index('timestamp')

                # CRITICAL: Slice data to correct horizon window
                # Without this, all horizons test on the same full dataset!
                asset1_data = slice_data_to_horizon(asset1_data, timeframe, horizon_days, warmup_multiplier=1.5)
                asset2_data = slice_data_to_horizon(asset2_data, timeframe, horizon_days, warmup_multiplier=1.5)

                if len(asset1_data) < 100 or len(asset2_data) < 100:
                    return {
                        'strategy_name': strategy_name,
                        'horizon': horizon_name,
                        'error': f'Insufficient data for {pair[0]} ({len(asset1_data)}) or {pair[1]} ({len(asset2_data)})'
                    }

                # Apply feature augmentation (Bug #3 fix)
                asset1_data = augment_with_features(asset1_data, pair[0], timeframe, config=DEFAULT_JOIN_CONFIG)
                asset2_data = augment_with_features(asset2_data, pair[1], timeframe, config=DEFAULT_JOIN_CONFIG)

                # Align data on timestamps (Bug #2 fix - proper alignment with logging)
                common_index = asset1_data.index.intersection(asset2_data.index)

                # Log data loss if significant
                initial_length = max(len(asset1_data), len(asset2_data))
                data_loss_pct = (1 - len(common_index) / initial_length) * 100

                if data_loss_pct > 5:  # More than 5% data loss
                    logger.warning(
                        f"StatisticalArbitrage {pair[0]}/{pair[1]}: Data alignment lost {initial_length - len(common_index)} rows "
                        f"({data_loss_pct:.1f}% of data). {pair[0]} had {len(asset1_data)} rows, "
                        f"{pair[1]} had {len(asset2_data)} rows, aligned: {len(common_index)} rows"
                    )

                combined_data = pd.DataFrame({
                    'timestamp': common_index,
                    f'{pair[0].replace("/", "_")}_close': asset1_data.loc[common_index, 'close'].values,
                    f'{pair[1].replace("/", "_")}_close': asset2_data.loc[common_index, 'close'].values
                })

                if len(combined_data) < 100:
                    return {
                        'strategy_name': strategy_name,
                        'horizon': horizon_name,
                        'error': 'Insufficient aligned data after merge'
                    }

                # Get strategy class and instantiate
                try:
                    import crypto_trader.strategies.library  # noqa: F401
                except ImportError as e:
                    return {
                        'strategy_name': strategy_name,
                        'horizon': horizon_name,
                        'error': f'Failed to import strategies library: {e}'
                    }

                registry = get_registry()
                strategy_class = registry.get_strategy(strategy_name)
                config_params = default_params or {}

                # Check strategy __init__ signature to instantiate correctly
                import inspect
                init_signature = inspect.signature(strategy_class.__init__)
                params = list(init_signature.parameters.keys())

                # If __init__ accepts name/config, pass them (e.g., StatisticalArbitrage)
                if 'name' in params and 'config' in params:
                    strategy = strategy_class(name=strategy_name, config=config_params)
                    # Old-style strategies already configured via __init__
                else:
                    # SOTA 2025 strategies: instantiate without args
                    strategy = strategy_class()

                # Initialize with parameters (ensure minimums: lookback >= 50, z_score_window >= 20)
                # Only call initialize for strategies that have it
                if hasattr(strategy, 'initialize') and callable(getattr(strategy, 'initialize')):
                    strategy.initialize({
                        'pair1_symbol': pair[0],
                        'pair2_symbol': pair[1],
                        'lookback_period': max(50, min(180, horizon_days)),
                        'entry_threshold': config_params.get('entry_threshold', 2.0),
                        'exit_threshold': config_params.get('exit_threshold', 0.5),
                        'z_score_window': max(20, min(90, horizon_days // 2))
                    })

                # Generate signals
                signals = strategy.generate_signals(combined_data)

                # Validate signals DataFrame
                if signals is None or signals.empty:
                    return {
                        'strategy_name': strategy_name,
                        'horizon': horizon_name,
                        'error': 'Strategy generated empty signals'
                    }

                if 'signal' not in signals.columns:
                    return {
                        'strategy_name': strategy_name,
                        'horizon': horizon_name,
                        'error': 'No signal column generated'
                    }

                # Check if all signals are HOLD (indicates pairs not cointegrated)
                if (signals['signal'] == SignalType.HOLD.value).all():
                    return {
                        'strategy_name': strategy_name,
                        'horizon': horizon_name,
                        'error': f'Pairs {pair[0]}/{pair[1]} not cointegrated - no trading opportunity'
                    }

                # Simulate backtest from signals
                initial_capital = 10000.0
                capital = initial_capital
                position = None  # 'LONG' or 'SHORT'
                entry_price_ratio = None
                trades = []
                equity_curve = [capital]

                commission = 0.001  # 0.1%
                slippage = 0.0005  # 0.05%

                # Ensure signals and combined_data are aligned
                min_length = min(len(signals), len(combined_data))
                if len(signals) != len(combined_data):
                    logger.warning(
                        f"StatisticalArbitrage: signals length ({len(signals)}) != "
                        f"data length ({len(combined_data)}). Using min length {min_length}."
                    )

                for i in range(min_length):
                    signal = signals['signal'].iloc[i]
                    price1 = combined_data.iloc[i][f'{pair[0].replace("/", "_")}_close']
                    price2 = combined_data.iloc[i][f'{pair[1].replace("/", "_")}_close']

                    if pd.isna(price1) or pd.isna(price2):
                        equity_curve.append(capital)
                        continue

                    current_ratio = price1 / price2

                    # Entry logic
                    if position is None:
                        if signal == SignalType.BUY.value:
                            # Long spread (buy asset1, sell asset2)
                            position = 'LONG'
                            entry_price_ratio = current_ratio
                            # Apply costs
                            capital *= (1 - commission - slippage)
                        elif signal == SignalType.SELL.value:
                            # Short spread (sell asset1, buy asset2)
                            position = 'SHORT'
                            entry_price_ratio = current_ratio
                            # Apply costs
                            capital *= (1 - commission - slippage)

                    # Exit logic
                    elif position is not None:
                        should_exit = False
                        pnl_pct = 0.0

                        if signal == SignalType.SELL.value and position == 'LONG':
                            # Exit long position
                            should_exit = True
                            pnl_pct = (current_ratio - entry_price_ratio) / entry_price_ratio
                        elif signal == SignalType.BUY.value and position == 'SHORT':
                            # Exit short position
                            should_exit = True
                            pnl_pct = (entry_price_ratio - current_ratio) / entry_price_ratio

                        if should_exit:
                            # Apply PnL
                            capital *= (1 + pnl_pct)
                            # Apply exit costs
                            capital *= (1 - commission - slippage)

                            trades.append({
                                'entry_ratio': entry_price_ratio,
                                'exit_ratio': current_ratio,
                                'type': position,
                                'pnl_pct': pnl_pct,
                                'profitable': pnl_pct > 0
                            })

                            position = None
                            entry_price_ratio = None

                    equity_curve.append(capital)

                # Calculate metrics
                final_capital = capital
                total_return = (final_capital - initial_capital) / initial_capital

                # Calculate Sharpe ratio
                if len(equity_curve) > 1:
                    returns = pd.Series(equity_curve).pct_change().dropna()
                    periods_per_year = periods_per_year_from_timeframe(timeframe)
                    sharpe_ratio = calculate_sharpe_ratio_safe(returns, periods_per_year)
                else:
                    sharpe_ratio = 0.0

                # Calculate max drawdown
                equity_series = pd.Series(equity_curve)
                running_max = equity_series.cummax()
                drawdown = (equity_series - running_max) / running_max
                max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

                # Calculate win rate and profit factor
                if trades:
                    winning_trades = [t for t in trades if t['profitable']]
                    losing_trades = [t for t in trades if not t['profitable']]

                    win_rate = len(winning_trades) / len(trades)

                    gross_profit = sum(t['pnl_pct'] for t in winning_trades) if winning_trades else 0.0
                    gross_loss = abs(sum(t['pnl_pct'] for t in losing_trades)) if losing_trades else 0.0
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit > 0 else 0.0)
                else:
                    win_rate = 0.0
                    profit_factor = 0.0

                # Return results
                return {
                    'strategy_name': strategy_name,
                    'strategy_type': 'multi_pair',
                    'symbol': f"Pair[{pair[0]}/{pair[1]}]",
                    'symbols': f'{pair[0]} / {pair[1]}',
                    'num_assets': 2,
                    'horizon': horizon_name,
                    'horizon_days': horizon_days,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'total_trades': len(trades),
                    'profit_factor': profit_factor,
                    'final_capital': final_capital,
                }

            except Exception as inner_e:
                # Capture and format error with context
                error_details = traceback.format_exc()
                error_msg = f'{str(inner_e)}\n{error_details}'
                return {
                    'strategy_name': strategy_name,
                    'horizon': horizon_name,
                    'error': format_error_message(error_msg, 'Statistical Arbitrage execution error', max_length=500)
                }

        elif strategy_name in ["HierarchicalRiskParity", "BlackLitterman", "RiskParity", "CopulaPairsTrading", "DeepRLPortfolio"]:
            # New SOTA 2025 portfolio strategies
            # These strategies return portfolio weights (not buy/sell signals)
            if len(asset_symbols) < 2:
                return {
                    'strategy_name': strategy_name,
                    'horizon': horizon_name,
                    'error': f'{strategy_name} requires at least 2 assets'
                }

            try:
                from crypto_trader.strategies import get_registry
                from crypto_trader.features.factory import augment_with_features, DEFAULT_JOIN_CONFIG
                import crypto_trader.strategies.library  # noqa: F401

                # Use pre-fetched data from shared pool (Bug #1 fix)
                asset_data = {}
                for symbol in asset_symbols:
                    if symbol not in data_dicts:
                        return {
                            'strategy_name': strategy_name,
                            'horizon': horizon_name,
                            'error': f'Pre-fetched data not available for {symbol}'
                        }

                    # Reconstruct DataFrame from dict
                    data = pd.DataFrame(data_dicts[symbol])
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data = data.set_index('timestamp')

                    # CRITICAL: Slice data to correct horizon window
                    # Without this, all horizons test on the same full dataset!
                    data = slice_data_to_horizon(data, timeframe, horizon_days, warmup_multiplier=1.5)

                    if len(data) < 100:
                        return {
                            'strategy_name': strategy_name,
                            'horizon': horizon_name,
                            'error': f'Insufficient data for {symbol} ({len(data)} rows)'
                        }

                    # Apply feature augmentation (Bug #3 fix)
                    data = augment_with_features(data, symbol, timeframe, config=DEFAULT_JOIN_CONFIG)
                    asset_data[symbol] = data

                # Combine data into single DataFrame using proper index alignment (Bug #2 fix)
                # Find common timestamps across ALL assets
                common_index = asset_data[asset_symbols[0]].index
                for symbol in asset_symbols[1:]:
                    common_index = common_index.intersection(asset_data[symbol].index)

                # Log data loss if significant
                initial_lengths = [len(asset_data[s]) for s in asset_symbols]
                max_length = max(initial_lengths)
                data_loss_pct = (1 - len(common_index) / max_length) * 100

                if data_loss_pct > 5:  # More than 5% data loss
                    lengths_str = ', '.join([f"{s}:{len(asset_data[s])}" for s in asset_symbols])
                    logger.warning(
                        f"{strategy_name}: Data alignment lost {max_length - len(common_index)} rows "
                        f"({data_loss_pct:.1f}% of data). Asset lengths: {lengths_str}, aligned: {len(common_index)}"
                    )

                # Build combined DataFrame with aligned data
                combined_data = pd.DataFrame(index=common_index)
                combined_data['timestamp'] = common_index

                # Align all asset data to the common index
                for symbol in asset_symbols:
                    col_name = symbol.replace('/', '_') + '_close'
                    combined_data[col_name] = asset_data[symbol].loc[common_index, 'close'].values

                if len(combined_data) < 100:
                    return {
                        'strategy_name': strategy_name,
                        'horizon': horizon_name,
                        'error': 'Insufficient aligned data after merge'
                    }

                # Get strategy class and instantiate
                registry = get_registry()
                strategy_class = registry.get_strategy(strategy_name)
                config_params = default_params or {}

                # Check strategy __init__ signature to instantiate correctly
                import inspect
                init_signature = inspect.signature(strategy_class.__init__)
                params = list(init_signature.parameters.keys())

                # If __init__ accepts name/config, pass them (e.g., StatisticalArbitrage)
                if 'name' in params and 'config' in params:
                    strategy = strategy_class(name=strategy_name, config=config_params)
                else:
                    # SOTA 2025 strategies: instantiate without args
                    strategy = strategy_class()

                # Initialize with appropriate parameters
                # Only call initialize for strategies that have it
                if hasattr(strategy, 'initialize') and callable(getattr(strategy, 'initialize')):
                    if strategy_name == "CopulaPairsTrading":
                        # Copula pairs trading uses first two assets
                        # Ensure minimum lookback of 30 for stable copula estimation
                        strategy.initialize({
                            'asset_pairs': [(asset_symbols[0], asset_symbols[1])],
                            'lookback_period': max(30, min(90, horizon_days)),
                            'entry_threshold': config_params.get('entry_threshold', 2.0),
                            'exit_threshold': config_params.get('exit_threshold', 0.5),
                            'position_size': 0.5
                        })
                    else:
                        # Portfolio strategies (HRP, Black-Litterman, Risk Parity, Deep RL)
                        # Ensure minimum lookback of 30 for stable covariance estimation
                        strategy.initialize({
                            'asset_symbols': asset_symbols,
                            'lookback_period': max(30, min(90, horizon_days)),
                            'rebalance_freq': 7
                        })

                # Generate signals (returns weights or positions)
                signals = strategy.generate_signals(combined_data)

                # Simulate portfolio performance
                initial_capital = 10000.0
                capital = initial_capital

                if strategy_name == "CopulaPairsTrading":
                    # Pairs trading uses position columns (long/short)
                    position_cols = [col for col in signals.columns if col.startswith('position_')]

                    if len(position_cols) == 0:
                        return {
                            'strategy_name': strategy_name,
                            'horizon': horizon_name,
                            'error': 'No position columns generated'
                        }

                    # Track equity from position changes
                    equity_curve = [capital]
                    commission = 0.001
                    previous_positions = {col: 0.0 for col in position_cols}

                    # Ensure signals and combined_data are aligned
                    min_length = min(len(signals), len(combined_data))
                    if len(signals) != len(combined_data):
                        logger.warning(
                            f"CopulaPairsTrading: signals length ({len(signals)}) != "
                            f"data length ({len(combined_data)}). Using min length {min_length}."
                        )

                    for i in range(1, min_length):
                        period_commission = 0.0

                        # Calculate P&L from positions
                        for pos_col in position_cols:
                            asset_col = pos_col.replace('position_', '')
                            if asset_col in combined_data.columns:
                                price_curr = combined_data.iloc[i][asset_col]
                                price_prev = combined_data.iloc[i-1][asset_col]

                                # Check for NaN or invalid prices
                                if pd.isna(price_curr) or pd.isna(price_prev) or price_prev <= 0:
                                    continue

                                position = signals.iloc[i][pos_col]
                                prev_position = previous_positions[pos_col]

                                # Only apply commission when position CHANGES
                                if position != prev_position:
                                    position_change = abs(position - prev_position)
                                    period_commission += commission * position_change

                                # Calculate P&L from holding position
                                if position != 0:
                                    pnl_pct = (price_curr - price_prev) / price_prev * position
                                    capital *= (1 + pnl_pct)

                                previous_positions[pos_col] = position

                        # Apply commission once per period (sum of all position changes)
                        capital *= (1 - period_commission)
                        equity_curve.append(capital)

                else:
                    # Portfolio strategies use weight columns
                    weight_cols = [col for col in signals.columns if col.startswith('weight_')]

                    if len(weight_cols) == 0:
                        return {
                            'strategy_name': strategy_name,
                            'horizon': horizon_name,
                            'error': 'No weight columns generated'
                        }

                    # Calculate portfolio returns from weights
                    equity_curve = [capital]
                    commission = 0.001  # 0.1%
                    previous_weights = {col: signals.iloc[0][col] for col in weight_cols}

                    # Ensure signals and combined_data are aligned
                    min_length = min(len(signals), len(combined_data))
                    if len(signals) != len(combined_data):
                        logger.warning(
                            f"{strategy_name}: signals length ({len(signals)}) != "
                            f"data length ({len(combined_data)}). Using min length {min_length}."
                        )

                    for i in range(1, min_length):
                        # Calculate weighted portfolio return
                        portfolio_return = 0.0
                        for weight_col in weight_cols:
                            asset_col = weight_col.replace('weight_', '')
                            if asset_col in combined_data.columns:
                                price_curr = combined_data.iloc[i][asset_col]
                                price_prev = combined_data.iloc[i-1][asset_col]

                                # Check for NaN or invalid prices
                                if pd.isna(price_curr) or pd.isna(price_prev) or price_prev <= 0:
                                    continue

                                weight = signals.iloc[i][weight_col]
                                asset_return = (price_curr - price_prev) / price_prev
                                portfolio_return += weight * asset_return

                        # Apply portfolio return
                        capital *= (1 + portfolio_return)

                        # Check if rebalancing occurred (weights changed)
                        weights_changed = False
                        rebalance_cost = 0.0
                        for weight_col in weight_cols:
                            curr_weight = signals.iloc[i][weight_col]
                            prev_weight = previous_weights[weight_col]
                            weight_change = abs(curr_weight - prev_weight)
                            if weight_change > 0.01:  # Significant change (>1%)
                                weights_changed = True
                                rebalance_cost += commission * weight_change
                                previous_weights[weight_col] = curr_weight

                        # Apply rebalancing cost only when weights change
                        if weights_changed:
                            capital *= (1 - rebalance_cost)

                        equity_curve.append(capital)

                # Calculate metrics
                final_capital = capital
                total_return = (final_capital - initial_capital) / initial_capital

                # Calculate Sharpe ratio
                if len(equity_curve) > 1:
                    returns = pd.Series(equity_curve).pct_change().dropna()
                    periods_per_year = periods_per_year_from_timeframe(timeframe)
                    sharpe_ratio = calculate_sharpe_ratio_safe(returns, periods_per_year)
                else:
                    sharpe_ratio = 0.0

                # Calculate max drawdown
                equity_series = pd.Series(equity_curve)
                running_max = equity_series.cummax()
                drawdown = (equity_series - running_max) / running_max
                max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

                # Calculate win rate for portfolio (% of positive return periods)
                positive_periods = (returns > 0).sum()
                total_periods = len(returns)
                win_rate = float(positive_periods / total_periods) if total_periods > 0 else 0.0

                # Return results
                return {
                    'strategy_name': strategy_name,
                    'strategy_type': 'multi_pair',
                    'symbol': f"Portfolio[{len(asset_symbols)} assets]",
                    'symbols': ', '.join(asset_symbols),
                    'num_assets': len(asset_symbols),
                    'horizon': horizon_name,
                    'horizon_days': horizon_days,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'total_trades': 0,  # Count rebalances if needed
                    'profit_factor': 0.0,  # N/A for portfolio
                    'final_capital': final_capital,
                }

            except Exception as inner_e:
                # Capture and format error with context
                error_details = traceback.format_exc()
                error_msg = f'{str(inner_e)}\n{error_details}'
                return {
                    'strategy_name': strategy_name,
                    'horizon': horizon_name,
                    'error': format_error_message(error_msg, f'{strategy_name} execution error', max_length=500)
                }

        else:
            return {
                'strategy_name': strategy_name,
                'horizon': horizon_name,
                'error': f'Unknown multi-pair strategy: {strategy_name}'
            }

    except Exception as e:
        # Return error info with traceback for debugging
        error_msg = f"{type(e).__name__}: {str(e)}"
        error_trace = traceback.format_exc()
        logger.debug(f"Multi-pair worker error for {strategy_name} on {horizon_name}:\n{error_trace}")
        return {
            'strategy_name': strategy_name,
            'horizon': horizon_name,
            'error': error_msg
        }


__all__ = [
    'run_backtest_worker',
    'run_multipair_backtest_worker',
]


if __name__ == "__main__":
    """
    Validation block for worker module.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: Verify worker functions are importable
    total_tests += 1
    print("Test 1: Verify worker functions importable")
    try:
        if run_backtest_worker is None:
            all_validation_failures.append("run_backtest_worker not imported")
        if run_multipair_backtest_worker is None:
            all_validation_failures.append("run_multipair_backtest_worker not imported")

        print(f"  ✓ run_backtest_worker: {run_backtest_worker.__name__}")
        print(f"  ✓ run_multipair_backtest_worker: {run_multipair_backtest_worker.__name__}")
    except Exception as e:
        all_validation_failures.append(f"Worker import failed: {e}")

    # Test 2: Verify function signatures
    total_tests += 1
    print("\nTest 2: Verify function signatures")
    try:
        import inspect

        sig1 = inspect.signature(run_backtest_worker)
        params1 = list(sig1.parameters.keys())
        expected1 = ['strategy_name', 'data_dict', 'horizon_name', 'horizon_days', 'symbol', 'timeframe', 'default_params']
        if params1 != expected1:
            all_validation_failures.append(f"run_backtest_worker signature mismatch: {params1} != {expected1}")
        else:
            print(f"  ✓ run_backtest_worker signature: {len(params1)} parameters")

        sig2 = inspect.signature(run_multipair_backtest_worker)
        params2 = list(sig2.parameters.keys())
        expected2 = ['strategy_name', 'asset_symbols', 'data_dicts', 'horizon_name', 'horizon_days', 'timeframe', 'default_params']
        if params2 != expected2:
            all_validation_failures.append(f"run_multipair_backtest_worker signature mismatch: {params2} != {expected2}")
        else:
            print(f"  ✓ run_multipair_backtest_worker signature: {len(params2)} parameters")

    except Exception as e:
        all_validation_failures.append(f"Signature validation failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Worker functions fully extracted and ready for use")
        print("\nNOTE: Workers are now using utility modules from execution layer")
        sys.exit(0)
