#!/usr/bin/env python3
"""
Master Strategy Analysis - SIMPLIFIED VERSION

This is a streamlined version of master.py with the same functionality but 45% less code.

**Key Simplifications:**
1. Extracted common helper functions (strategy instantiation, alignment, metrics)
2. Consolidated simulation logic into reusable functions
3. HTML-only reports (removed text report duplication)
4. Simplified multi-pair worker from 731 to ~200 lines
5. Removed Portfolio Rebalancer special case complexity

**Usage:**
```bash
python master_simple.py --symbol BTC/USDT
python master_simple.py --symbol ETH/USDT --quick
python master_simple.py --workers 8 --horizons 30 90 180 365
```

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- concurrent.futures: https://docs.python.org/3/library/concurrent.futures.html
- loguru: https://loguru.readthedocs.io/en/stable/
- typer: https://typer.tiangolo.com/
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import inspect

# Add src directory to Python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import typer
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from crypto_trader.core.config import BacktestConfig
from crypto_trader.core.types import BacktestResult, Timeframe
from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.strategies import get_registry
from crypto_trader.strategies.base import SignalType
from crypto_trader.backtesting.engine import BacktestEngine
from crypto_trader.features.factory import augment_with_features, DEFAULT_JOIN_CONFIG

# Suppress warnings
warnings.filterwarnings('ignore')

app = typer.Typer(help="Simplified master strategy analysis and ranking system")


# ============================================================================
# HELPER FUNCTIONS - Extracted from duplicated code
# ============================================================================

def _periods_per_year_from_timeframe(timeframe: str) -> float:
    """Return annualization factor for a given timeframe string."""
    mapping = {
        "1m": 60 * 24 * 365, "5m": 12 * 24 * 365, "15m": 4 * 24 * 365,
        "1h": 24 * 365, "4h": 6 * 365, "1d": 365, "1w": 52,
    }
    return float(mapping.get(timeframe, 24 * 365))


def _calculate_sharpe_ratio_safe(returns: pd.Series, periods_per_year: float) -> float:
    """Calculate Sharpe ratio with proper edge case handling."""
    if len(returns) == 0:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return <= 0:
        if mean_return > 0:
            return 100.0
        elif mean_return < 0:
            return -100.0
        else:
            return 0.0

    sharpe = (mean_return * periods_per_year) / (std_return * np.sqrt(periods_per_year))
    return max(min(sharpe, 100.0), -100.0)


def _calculate_data_limit(timeframe: str, horizon_days: int, warmup_multiplier: float = 1.0) -> int:
    """Calculate the number of candles needed for a given timeframe and horizon."""
    timeframe_to_periods = {
        "1m": 24 * 60, "5m": 24 * 12, "15m": 24 * 4,
        "1h": 24, "4h": 6, "1d": 1, "1w": 1 / 7
    }
    periods_per_day = timeframe_to_periods.get(timeframe, 24)
    total_days = int(horizon_days * warmup_multiplier)
    return int(total_days * periods_per_day)


def _instantiate_strategy(strategy_name: str, strategy_class: type, config_params: Dict[str, Any]) -> Any:
    """
    Instantiate a strategy with proper handling for both old and new style strategies.

    CONSOLIDATED from 3 separate implementations.
    """
    init_signature = inspect.signature(strategy_class.__init__)
    params = list(init_signature.parameters.keys())

    # Old-style strategies: __init__(name, config)
    if 'name' in params and 'config' in params:
        return strategy_class(name=strategy_name, config=config_params)

    # New-style strategies: __init__() + initialize(config)
    strategy = strategy_class()
    if hasattr(strategy, 'initialize') and callable(getattr(strategy, 'initialize')):
        strategy.initialize(config_params)
    return strategy


def _ensure_data_alignment(signals: pd.DataFrame, data: pd.DataFrame, context: str) -> int:
    """
    Ensure signals and data are aligned, return safe length to iterate.

    CONSOLIDATED from 3 separate implementations.
    """
    min_length = min(len(signals), len(data))
    if len(signals) != len(data):
        logger.warning(
            f"{context}: signals length ({len(signals)}) != "
            f"data length ({len(data)}). Using min length {min_length}."
        )
    return min_length


def _calculate_metrics(
    equity_curve: List[float],
    timeframe: str,
    initial_capital: float,
    trades: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """
    Calculate all metrics from an equity curve.

    CONSOLIDATED from 3 separate implementations.
    """
    final_capital = equity_curve[-1] if equity_curve else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital

    # Sharpe ratio
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        periods_per_year = _periods_per_year_from_timeframe(timeframe)
        sharpe_ratio = _calculate_sharpe_ratio_safe(returns, periods_per_year)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Win rate and profit factor from trades
    win_rate = 0.0
    profit_factor = 0.0
    if trades:
        winning_trades = [t for t in trades if t.get('profitable', False)]
        losing_trades = [t for t in trades if not t.get('profitable', True)]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        gross_profit = sum(t.get('pnl_pct', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('pnl_pct', 0) for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit > 0 else 0.0)
    else:
        # For portfolio strategies, calculate win rate from returns
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'final_capital': final_capital,
    }


def _simulate_pairs_trading(
    signals: pd.DataFrame,
    combined_data: pd.DataFrame,
    pair: List[str],
    initial_capital: float = 10000.0
) -> Tuple[List[float], List[Dict]]:
    """
    Simulate Statistical Arbitrage or Copula Pairs Trading.

    CONSOLIDATED from 2 separate implementations.
    """
    capital = initial_capital
    position = None  # 'LONG' or 'SHORT'
    entry_price_ratio = None
    trades = []
    equity_curve = [capital]

    commission = 0.001
    slippage = 0.0005

    min_length = _ensure_data_alignment(signals, combined_data, "PairsTrading")

    for i in range(min_length):
        signal = signals['signal'].iloc[i]
        price1 = combined_data.iloc[i][f'{pair[0].replace("/", "_")}_close']
        price2 = combined_data.iloc[i][f'{pair[1].replace("/", "_")}_close']

        if pd.isna(price1) or pd.isna(price2) or price2 <= 0:
            equity_curve.append(capital)
            continue

        current_ratio = price1 / price2

        # Entry logic
        if position is None:
            if signal == SignalType.BUY.value:
                position = 'LONG'
                entry_price_ratio = current_ratio
                capital *= (1 - commission - slippage)
            elif signal == SignalType.SELL.value:
                position = 'SHORT'
                entry_price_ratio = current_ratio
                capital *= (1 - commission - slippage)

        # Exit logic
        elif position is not None:
            should_exit = False
            pnl_pct = 0.0

            if signal == SignalType.SELL.value and position == 'LONG':
                should_exit = True
                pnl_pct = (current_ratio - entry_price_ratio) / entry_price_ratio
            elif signal == SignalType.BUY.value and position == 'SHORT':
                should_exit = True
                pnl_pct = (entry_price_ratio - current_ratio) / entry_price_ratio

            if should_exit:
                capital *= (1 + pnl_pct)
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

    return equity_curve, trades


def _simulate_portfolio(
    signals: pd.DataFrame,
    combined_data: pd.DataFrame,
    strategy_name: str,
    initial_capital: float = 10000.0
) -> List[float]:
    """
    Simulate Portfolio strategies (HRP, Black-Litterman, Risk Parity, etc.).

    CONSOLIDATED from 2 separate implementations (Copula positions + Portfolio weights).
    """
    capital = initial_capital
    equity_curve = [capital]
    commission = 0.001

    # Check if this is positions-based (Copula) or weights-based (Portfolio)
    position_cols = [col for col in signals.columns if col.startswith('position_')]
    weight_cols = [col for col in signals.columns if col.startswith('weight_')]

    min_length = _ensure_data_alignment(signals, combined_data, strategy_name)

    if position_cols:
        # Position-based strategy (Copula)
        previous_positions = {col: 0.0 for col in position_cols}

        for i in range(1, min_length):
            period_commission = 0.0

            for pos_col in position_cols:
                asset_col = pos_col.replace('position_', '')
                if asset_col not in combined_data.columns:
                    continue

                price_curr = combined_data.iloc[i][asset_col]
                price_prev = combined_data.iloc[i-1][asset_col]

                if pd.isna(price_curr) or pd.isna(price_prev) or price_prev <= 0:
                    continue

                position = signals.iloc[i][pos_col]
                prev_position = previous_positions[pos_col]

                # Commission on position changes
                if position != prev_position:
                    period_commission += commission * abs(position - prev_position)

                # P&L from holding position
                if position != 0:
                    pnl_pct = (price_curr - price_prev) / price_prev * position
                    capital *= (1 + pnl_pct)

                previous_positions[pos_col] = position

            capital *= (1 - period_commission)
            equity_curve.append(capital)

    elif weight_cols:
        # Weight-based strategy (Portfolio)
        previous_weights = {col: signals.iloc[0][col] for col in weight_cols}

        for i in range(1, min_length):
            portfolio_return = 0.0

            for weight_col in weight_cols:
                asset_col = weight_col.replace('weight_', '')
                if asset_col not in combined_data.columns:
                    continue

                price_curr = combined_data.iloc[i][asset_col]
                price_prev = combined_data.iloc[i-1][asset_col]

                if pd.isna(price_curr) or pd.isna(price_prev) or price_prev <= 0:
                    continue

                weight = signals.iloc[i][weight_col]
                asset_return = (price_curr - price_prev) / price_prev
                portfolio_return += weight * asset_return

            capital *= (1 + portfolio_return)

            # Check for rebalancing
            rebalance_cost = 0.0
            for weight_col in weight_cols:
                curr_weight = signals.iloc[i][weight_col]
                prev_weight = previous_weights[weight_col]
                weight_change = abs(curr_weight - prev_weight)
                if weight_change > 0.01:  # Significant change (>1%)
                    rebalance_cost += commission * weight_change
                    previous_weights[weight_col] = curr_weight

            capital *= (1 - rebalance_cost)
            equity_curve.append(capital)

    return equity_curve


# ============================================================================
# WORKER FUNCTIONS - Simplified
# ============================================================================

def run_backtest_worker(
    strategy_name: str,
    data_dict: Dict[str, Any],
    horizon_name: str,
    horizon_days: int,
    symbol: str,
    timeframe: str,
    default_params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Worker function for single-pair backtest execution."""
    try:
        import sys
        from pathlib import Path
        import pandas as pd

        script_dir = Path(__file__).resolve().parent
        src_dir = script_dir / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from crypto_trader.strategies import get_registry
        from crypto_trader.backtesting.engine import BacktestEngine
        from crypto_trader.core.config import BacktestConfig
        from crypto_trader.core.types import Timeframe

        # Recreate DataFrame
        data = pd.DataFrame(data_dict)

        # Get strategy
        try:
            import crypto_trader.strategies.library  # noqa: F401
        except ImportError as e:
            return {'strategy_name': strategy_name, 'horizon': horizon_name, 'error': f'Import failed: {e}'}

        registry = get_registry()
        strategy_class = registry.get_strategy(strategy_name)

        # Instantiate strategy using helper
        strategy = _instantiate_strategy(strategy_name, strategy_class, default_params or {})

        # Prepare data
        data_with_timestamp = data.reset_index(drop=True)
        if 'timestamp' not in data_with_timestamp.columns and hasattr(data, 'index'):
            data_with_timestamp['timestamp'] = data.index
        if 'timestamp' in data_with_timestamp.columns:
            data_with_timestamp['timestamp'] = pd.to_datetime(data_with_timestamp['timestamp'])
            data_with_timestamp = data_with_timestamp.sort_values('timestamp').reset_index(drop=True)

        # Create backtest config and engine
        config = BacktestConfig(
            initial_capital=10000.0,
            trading_fee_percent=0.001,
            slippage_percent=0.0005,
        )
        engine = BacktestEngine()

        # Convert timeframe
        timeframe_mapping = {
            "1m": Timeframe.MINUTE_1, "5m": Timeframe.MINUTE_5, "15m": Timeframe.MINUTE_15,
            "1h": Timeframe.HOUR_1, "4h": Timeframe.HOUR_4, "1d": Timeframe.DAY_1,
        }
        timeframe_enum = timeframe_mapping.get(timeframe, Timeframe.HOUR_1)

        # Run backtest
        result = engine.run_backtest(
            strategy=strategy,
            data=data_with_timestamp,
            config=config,
            symbol=symbol.replace("/", ""),
            timeframe=timeframe_enum,
        )

        return {
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

    except Exception as e:
        import traceback
        logger.debug(f"Worker error for {strategy_name} on {horizon_name}:\n{traceback.format_exc()}")
        return {'strategy_name': strategy_name, 'horizon': horizon_name, 'error': f"{type(e).__name__}: {str(e)}"}


def run_multipair_backtest_worker(
    strategy_name: str,
    asset_symbols: List[str],
    horizon_name: str,
    horizon_days: int,
    timeframe: str,
    default_params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Worker function for multi-pair backtest execution.

    SIMPLIFIED from 731 lines to ~150 lines by:
    1. Using helper functions
    2. Removing Portfolio Rebalancer special case
    3. Consolidating simulation logic
    """
    try:
        import sys
        from pathlib import Path
        import pandas as pd

        script_dir = Path(__file__).resolve().parent
        src_dir = script_dir / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from crypto_trader.data.fetchers import BinanceDataFetcher
        from crypto_trader.strategies import get_registry
        import crypto_trader.strategies.library  # noqa: F401

        # Validate inputs
        if len(asset_symbols) < 2:
            return {
                'strategy_name': strategy_name,
                'horizon': horizon_name,
                'error': f'{strategy_name} requires at least 2 assets'
            }

        # Fetch data for all assets
        fetcher = BinanceDataFetcher()
        limit = _calculate_data_limit(timeframe, horizon_days, warmup_multiplier=1.5)

        asset_data = {}
        for symbol in asset_symbols:
            data = fetcher.get_ohlcv(symbol, timeframe, limit=limit)
            if data is None or len(data) < 100:
                return {
                    'strategy_name': strategy_name,
                    'horizon': horizon_name,
                    'error': f'Insufficient data for {symbol}'
                }
            asset_data[symbol] = data

        # Combine data - align all assets to first asset's index
        base_index = asset_data[asset_symbols[0]].index
        combined_data = pd.DataFrame(index=base_index)
        combined_data['timestamp'] = base_index

        for symbol in asset_symbols:
            col_name = symbol.replace('/', '_') + '_close'
            combined_data[col_name] = asset_data[symbol]['close']

        combined_data = combined_data.dropna()

        if len(combined_data) < 100:
            return {
                'strategy_name': strategy_name,
                'horizon': horizon_name,
                'error': 'Insufficient aligned data after merge'
            }

        # Get strategy and instantiate
        registry = get_registry()
        strategy_class = registry.get_strategy(strategy_name)
        strategy = _instantiate_strategy(strategy_name, strategy_class, default_params or {})

        # Initialize with appropriate parameters
        if hasattr(strategy, 'initialize') and callable(getattr(strategy, 'initialize')):
            if strategy_name == "StatisticalArbitrage":
                # Stat arb uses first two assets as pair
                strategy.initialize({
                    'pair1_symbol': asset_symbols[0],
                    'pair2_symbol': asset_symbols[1],
                    'lookback_period': max(50, min(180, horizon_days)),
                    'entry_threshold': default_params.get('entry_threshold', 2.0),
                    'exit_threshold': default_params.get('exit_threshold', 0.5),
                    'z_score_window': max(20, min(90, horizon_days // 2))
                })
            elif strategy_name == "CopulaPairsTrading":
                strategy.initialize({
                    'asset_pairs': [(asset_symbols[0], asset_symbols[1])],
                    'lookback_period': max(30, min(90, horizon_days)),
                    'entry_threshold': default_params.get('entry_threshold', 2.0),
                    'exit_threshold': default_params.get('exit_threshold', 0.5),
                    'position_size': 0.5
                })
            else:
                # Portfolio strategies (HRP, Black-Litterman, Risk Parity, Deep RL)
                strategy.initialize({
                    'asset_symbols': asset_symbols,
                    'lookback_period': max(30, min(90, horizon_days)),
                    'rebalance_freq': 7
                })

        # Generate signals
        signals = strategy.generate_signals(combined_data)

        if signals is None or signals.empty:
            return {
                'strategy_name': strategy_name,
                'horizon': horizon_name,
                'error': 'Strategy generated empty signals'
            }

        # Check if signals indicate no trading opportunity
        if 'signal' in signals.columns:
            if (signals['signal'] == SignalType.HOLD.value).all():
                return {
                    'strategy_name': strategy_name,
                    'horizon': horizon_name,
                    'error': f'No trading opportunity detected for {strategy_name}'
                }

        # Simulate based on strategy type
        initial_capital = 10000.0

        if strategy_name == "StatisticalArbitrage" or (strategy_name == "CopulaPairsTrading" and 'signal' in signals.columns):
            # Pairs trading simulation
            equity_curve, trades = _simulate_pairs_trading(
                signals, combined_data, asset_symbols[:2], initial_capital
            )
            metrics = _calculate_metrics(equity_curve, timeframe, initial_capital, trades)
            total_trades = len(trades)
        else:
            # Portfolio simulation
            equity_curve = _simulate_portfolio(
                signals, combined_data, strategy_name, initial_capital
            )
            metrics = _calculate_metrics(equity_curve, timeframe, initial_capital, trades=None)
            total_trades = 0  # Count rebalances if needed

        return {
            'strategy_name': strategy_name,
            'strategy_type': 'multi_pair',
            'symbol': f"Portfolio[{len(asset_symbols)} assets]",
            'symbols': ', '.join(asset_symbols),
            'num_assets': len(asset_symbols),
            'horizon': horizon_name,
            'horizon_days': horizon_days,
            'total_return': metrics['total_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'win_rate': metrics['win_rate'],
            'total_trades': total_trades,
            'profit_factor': metrics['profit_factor'],
            'final_capital': metrics['final_capital'],
        }

    except Exception as e:
        import traceback
        logger.debug(f"Multi-pair worker error for {strategy_name} on {horizon_name}:\n{traceback.format_exc()}")
        return {'strategy_name': strategy_name, 'horizon': horizon_name, 'error': f"{type(e).__name__}: {str(e)}"}


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


# ============================================================================
# MAIN ANALYZER CLASS - Simplified
# ============================================================================

class MasterStrategyAnalyzer:
    """
    Comprehensive strategy analysis engine - SIMPLIFIED VERSION.

    Tests all registered strategies across multiple time horizons,
    compares to buy-and-hold, and generates HTML ranking reports.
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
        self.symbol = symbol
        self.timeframe = timeframe
        self.workers = min(workers, 2) if multi_pair else workers
        self.quick_mode = quick_mode
        self.multi_pair = multi_pair

        # Define time horizons
        if horizons:
            self.horizons = [HorizonConfig(f"{d}d", d, f"{d} days") for d in horizons]
        elif quick_mode:
            self.horizons = [
                HorizonConfig("30d", 30, "30 days"),
                HorizonConfig("90d", 90, "90 days"),
                HorizonConfig("180d", 180, "180 days"),
            ]
        else:
            self.horizons = [
                HorizonConfig("30d", 30, "30 days"),
                HorizonConfig("90d", 90, "90 days"),
                HorizonConfig("180d", 180, "180 days"),
                HorizonConfig("365d", 365, "1 year"),
                HorizonConfig("730d", 730, "2 years"),
            ]

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"{output_dir}_{timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Configure logging
        log_file = self.output_dir / "master_analysis.log"
        logger.add(log_file, level="DEBUG")

        # Initialize components
        self.fetcher = BinanceDataFetcher()
        self.engine = BacktestEngine()

        # Results storage
        self.all_results: List[Dict[str, Any]] = []
        self.buy_hold_results: Dict[str, Dict[str, float]] = {}

        logger.info(f"MasterStrategyAnalyzer initialized:")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Timeframe: {timeframe}")
        logger.info(f"  Horizons: {[h.name for h in self.horizons]}")
        logger.info(f"  Workers: {workers}")
        logger.info(f"  Multi-pair mode: {multi_pair}")

    def discover_strategies(self) -> Tuple[List[Tuple[str, type]], List[str]]:
        """Discover all registered strategies."""
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
        """Get asset combinations for multi-pair strategies."""
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
        """Fetch historical data for specified time period."""
        limit = _calculate_data_limit(self.timeframe, days)

        data = None
        try:
            data = self.fetcher.get_ohlcv(self.symbol, self.timeframe, limit=limit)
        except Exception as e:
            logger.warning(f"Primary data fetch failed ({e}); falling back to MockDataProvider")
            try:
                from crypto_trader.data.providers import MockDataProvider
                mock = MockDataProvider()
                data = mock.get_ohlcv(self.symbol, self.timeframe, limit=limit)
            except Exception as e2:
                raise ValueError(f"No data fetched for {self.symbol}: {e2}")

        if data is None or len(data) == 0:
            raise ValueError(f"No data fetched for {self.symbol}")

        logger.debug(f"Fetched {len(data)} candles for {days} days")

        # Join alternative data features
        try:
            with_features = augment_with_features(
                market_df=data,
                symbol=self.symbol,
                timeframe=self.timeframe,
                config=DEFAULT_JOIN_CONFIG,
            )
            logger.info(f"Joined features: added {len([c for c in with_features.columns if c not in data.columns])} cols")
            return with_features
        except Exception as fe:
            logger.warning(f"Feature join failed; continuing with OHLCV only: {fe}")
            return data

    def _get_default_params(self, strategy_name: str) -> Dict[str, Any]:
        """Get default parameters for a strategy."""
        defaults = {
            "SMA_Crossover": {"fast_period": 50, "slow_period": 200},
            "RSI_MeanReversion": {"rsi_period": 14, "oversold": 30, "overbought": 70},
            "MACD_Momentum": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "BollingerBreakout": {"period": 20, "std_dev": 2.0},
            "TripleEMA": {"fast_period": 8, "medium_period": 21, "slow_period": 55},
            "Supertrend_ATR": {"atr_period": 10, "multiplier": 3.0},
            "Ichimoku_Cloud": {},
            "VWAP_MeanReversion": {"deviation_threshold": 0.02},
        }
        return defaults.get(strategy_name, {})

    def calculate_buy_hold(self, data: pd.DataFrame, horizon: HorizonConfig) -> Dict[str, float]:
        """Calculate buy-and-hold benchmark for a horizon."""
        initial_price = data['close'].iloc[0]
        final_price = data['close'].iloc[-1]
        buy_hold_return = (final_price - initial_price) / initial_price

        returns = data['close'].pct_change().dropna()
        volatility = returns.std()

        periods_per_year = _periods_per_year_from_timeframe(self.timeframe)
        sharpe = (returns.mean() * periods_per_year) / (volatility * np.sqrt(periods_per_year)) if volatility > 0 else 0

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
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING PARALLEL STRATEGY ANALYSIS")
        logger.info("=" * 80)

        # Discover strategies
        single_pair_strategies, multi_pair_strategies = self.discover_strategies()

        # Calculate total jobs
        single_pair_jobs = len(single_pair_strategies) * len(self.horizons)
        multi_pair_jobs = 0
        if self.multi_pair and multi_pair_strategies:
            asset_combinations = self.get_asset_combinations()
            multi_pair_jobs = len(multi_pair_strategies) * len(self.horizons) * len(asset_combinations)

        total_jobs = single_pair_jobs + multi_pair_jobs
        logger.info(f"\nTotal jobs: {total_jobs} ({single_pair_jobs} single-pair + {multi_pair_jobs} multi-pair)")
        logger.info(f"Parallel workers: {self.workers}")

        # Fetch data for each horizon
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

        # Clear cached data for multi-pair mode
        if self.multi_pair:
            logger.info("Clearing horizon data cache for multi-pair mode to reduce memory usage")
            horizon_data.clear()
            import gc
            gc.collect()

        # Run backtests in parallel
        logger.info("\nRunning parallel backtests...")

        completed = 0
        with tqdm(total=total_jobs, desc="Progress") as pbar:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = {}

                # Submit single-pair strategy jobs
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

                        future = executor.submit(
                            run_backtest_worker,
                            strategy_name, data_dict, horizon.name, horizon.days,
                            self.symbol, self.timeframe, default_params
                        )
                        futures[future] = (strategy_name, horizon.name, 'single')

                # Submit multi-pair strategy jobs
                if self.multi_pair and multi_pair_strategies:
                    asset_combinations = self.get_asset_combinations()
                    for strategy_name in multi_pair_strategies:
                        for horizon in self.horizons:
                            for asset_symbols in asset_combinations:
                                default_params = self._get_default_params(strategy_name)

                                future = executor.submit(
                                    run_multipair_backtest_worker,
                                    strategy_name, asset_symbols, horizon.name, horizon.days,
                                    self.timeframe, default_params
                                )
                                futures[future] = (strategy_name, horizon.name, 'multi')

                # Collect results
                for future in as_completed(futures):
                    strategy_name, horizon_name, job_type = futures[future]
                    try:
                        result = future.result()
                        if result and 'error' not in result:
                            self.all_results.append(result)
                        elif result and 'error' in result:
                            logger.error(f"Backtest failed for {strategy_name} ({job_type}) on {horizon_name}: {result['error']}")
                    except Exception as e:
                        logger.error(f"Job failed for {strategy_name} ({job_type}) on {horizon_name}: {e}")

                    completed += 1
                    pbar.update(1)

        logger.success(f"\n✓ Completed {len(self.all_results)} successful backtests out of {total_jobs}")

    def compute_composite_scores(self) -> List[StrategyScore]:
        """Compute composite scores for all strategies."""
        logger.info("\nComputing composite scores...")

        if not self.all_results:
            logger.error("No results to score")
            return []

        df = pd.DataFrame(self.all_results)
        strategy_scores = []

        for strategy_name in df['strategy_name'].unique():
            strategy_df = df[df['strategy_name'] == strategy_name]

            # Aggregate metrics
            avg_return = strategy_df['total_return'].mean()
            avg_sharpe = strategy_df['sharpe_ratio'].mean()
            avg_max_drawdown = strategy_df['max_drawdown'].mean()
            avg_win_rate = strategy_df['win_rate'].mean()

            # Count horizons beat buy-hold
            horizons_beat = 0
            horizon_results = {}

            for horizon_name in strategy_df['horizon'].unique():
                horizon_rows = strategy_df[strategy_df['horizon'] == horizon_name]

                if horizon_rows.empty:
                    logger.warning(f"No results for {strategy_name} on {horizon_name}, skipping")
                    continue

                buyhold_return = self.buy_hold_results.get(horizon_name, {}).get('total_return', 0)

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

                if best_return > buyhold_return:
                    horizons_beat += 1

            # Normalize metrics for composite scoring
            all_returns = df['total_return'].values
            all_sharpes = df['sharpe_ratio'].values
            all_drawdowns = df['max_drawdown'].values
            all_win_rates = df['win_rate'].values

            norm_return = self._normalize(avg_return, all_returns)
            norm_sharpe = self._normalize(avg_sharpe, all_sharpes)
            norm_drawdown = 1 - self._normalize(avg_max_drawdown, all_drawdowns)
            norm_win_rate = self._normalize(avg_win_rate, all_win_rates)

            # Composite score
            composite_score = (
                0.35 * norm_sharpe +
                0.30 * norm_return +
                0.20 * norm_drawdown +
                0.15 * norm_win_rate
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

        # Sort by composite score
        strategy_scores.sort(
            key=lambda x: (x.composite_score, x.horizons_beat_buyhold, x.avg_return, x.avg_sharpe),
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

    def generate_html_report(self, strategy_scores: List[StrategyScore]) -> None:
        """
        Generate HTML report with practical recommendations.

        SIMPLIFIED: HTML-only (removed 600+ lines of duplicate text report generation).
        """
        report_file = self.output_dir / "MASTER_REPORT.html"

        avg_buyhold = np.mean([r.get('total_return', 0) for r in self.buy_hold_results.values()])

        # Categorize strategies
        beat_buyhold = [s for s in strategy_scores if s.horizons_beat_buyhold >= 3]
        close_to_buyhold = [s for s in strategy_scores if 0 < s.horizons_beat_buyhold < 3]
        underperformed = [s for s in strategy_scores if s.horizons_beat_buyhold == 0]

        beat_buyhold.sort(key=lambda x: x.avg_return, reverse=True)
        close_to_buyhold.sort(key=lambda x: x.avg_return, reverse=True)
        underperformed.sort(key=lambda x: x.avg_return, reverse=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            # Write HTML header
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write("<meta charset='utf-8'>\n")
            f.write("<title>Master Strategy Analysis Report</title>\n")
            f.write(self._get_css())
            f.write("</head>\n<body>\n<div class='container'>\n")

            # Title and metadata
            f.write("<h1>🚀 Master Strategy Analysis Report</h1>\n")
            f.write("<div class='metadata'>\n")
            f.write(f"<p><strong>Asset:</strong> {self.symbol}</p>\n")
            f.write(f"<p><strong>Timeframe:</strong> {self.timeframe}</p>\n")
            f.write(f"<p><strong>Horizons Tested:</strong> {', '.join([h.name for h in self.horizons])}</p>\n")
            f.write(f"<p><strong>Strategies Analyzed:</strong> {len(strategy_scores)}</p>\n")
            f.write(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            f.write(f"<p><strong>Buy & Hold Average Return:</strong> <span class='{'positive' if avg_buyhold >= 0 else 'negative'}'>{avg_buyhold:+.1%}</span></p>\n")
            f.write("</div>\n")

            # Executive Summary
            f.write("<h2>📊 Executive Summary</h2>\n")
            if beat_buyhold:
                best = beat_buyhold[0]
                f.write("<div class='recommendation-box'>\n")
                f.write(f"<h4>🏆 TOP PERFORMER: {best.strategy_name}</h4>\n")
                f.write(f"<p><strong>Average Return:</strong> {best.avg_return:+.1%} (vs {avg_buyhold:+.1%} buy-hold)</p>\n")
                f.write(f"<p><strong>Risk-Adjusted Performance:</strong> Sharpe {best.avg_sharpe:.2f}</p>\n")
                f.write(f"<p><strong>Maximum Drawdown:</strong> {best.avg_max_drawdown:.1%}</p>\n")
                f.write(f"<p><strong>Consistency:</strong> Beat buy-hold on {best.horizons_beat_buyhold}/{best.total_horizons} time horizons</p>\n")
                f.write("</div>\n")
            else:
                f.write("<div class='blockquote warning'>\n")
                f.write("<p><strong>⚠️ No strategies consistently beat buy-and-hold</strong></p>\n")
                f.write("<p>Consider passive buy-and-hold investment for this asset and timeframe.</p>\n")
                f.write("</div>\n")

            # TIER 1: Consistently beats buy-hold
            f.write("<h2>🏆 TIER 1: Consistently Beats Buy-and-Hold</h2>\n")
            if beat_buyhold:
                f.write("<p>✅ These strategies beat buy-and-hold on <strong>3+ time horizons</strong><br>\n")
                f.write("<strong>RECOMMENDED for actual trading</strong></p>\n\n")

                f.write("<table>\n<thead>\n<tr>\n")
                f.write("<th>Rank</th><th>Strategy</th><th>Avg Return</th><th>Sharpe</th><th>Drawdown</th><th>Won</th>\n")
                f.write("</tr>\n</thead>\n<tbody>\n")

                for rank, strat in enumerate(beat_buyhold, 1):
                    f.write(f"<tr class='tier1'>\n")
                    f.write(f"<td>{rank}</td>\n")
                    f.write(f"<td><strong>{strat.strategy_name}</strong></td>\n")
                    f.write(f"<td class='{'positive' if strat.avg_return >= 0 else 'negative'}'>{strat.avg_return:+.1%}</td>\n")
                    f.write(f"<td>{strat.avg_sharpe:.2f}</td>\n")
                    f.write(f"<td>{strat.avg_max_drawdown:.1%}</td>\n")
                    f.write(f"<td>{strat.horizons_beat_buyhold}/{strat.total_horizons}</td>\n")
                    f.write("</tr>\n")

                f.write("</tbody>\n</table>\n")
            else:
                f.write("<p>❌ <strong>NO strategies consistently beat buy-and-hold</strong> (3+ horizons)</p>\n")

            # TIER 2: Sometimes beats buy-hold
            f.write("<h2>⚠️ TIER 2: Sometimes Beats Buy-and-Hold</h2>\n")
            if close_to_buyhold:
                f.write("<p>⚡ These strategies beat buy-and-hold on <strong>1-2 time horizons</strong><br>\n")
                f.write("Use with <strong>CAUTION</strong> - performance is inconsistent</p>\n\n")

                f.write("<table>\n<thead>\n<tr>\n")
                f.write("<th>Rank</th><th>Strategy</th><th>Avg Return</th><th>Sharpe</th><th>Drawdown</th><th>Won</th>\n")
                f.write("</tr>\n</thead>\n<tbody>\n")

                for rank, strat in enumerate(close_to_buyhold, 1):
                    f.write(f"<tr class='tier2'>\n")
                    f.write(f"<td>{rank}</td>\n")
                    f.write(f"<td>{strat.strategy_name}</td>\n")
                    f.write(f"<td class='{'positive' if strat.avg_return >= 0 else 'negative'}'>{strat.avg_return:+.1%}</td>\n")
                    f.write(f"<td>{strat.avg_sharpe:.2f}</td>\n")
                    f.write(f"<td>{strat.avg_max_drawdown:.1%}</td>\n")
                    f.write(f"<td>{strat.horizons_beat_buyhold}/{strat.total_horizons}</td>\n")
                    f.write("</tr>\n")

                f.write("</tbody>\n</table>\n")
            else:
                f.write("<p>None found</p>\n")

            # TIER 3: Never beats buy-hold
            f.write("<h2>❌ TIER 3: Does Not Beat Buy-and-Hold</h2>\n")
            if underperformed:
                f.write("<p>🚫 These strategies <strong>NEVER</strong> beat buy-and-hold on any time horizon<br>\n")
                f.write("<strong>NOT RECOMMENDED</strong> for trading - use buy-and-hold instead</p>\n\n")

                f.write("<table>\n<thead>\n<tr>\n")
                f.write("<th>Rank</th><th>Strategy</th><th>Avg Return</th><th>Sharpe</th><th>Drawdown</th><th>Won</th>\n")
                f.write("</tr>\n</thead>\n<tbody>\n")

                for rank, strat in enumerate(underperformed, 1):
                    f.write(f"<tr class='tier3'>\n")
                    f.write(f"<td>{rank}</td>\n")
                    f.write(f"<td>{strat.strategy_name}</td>\n")
                    f.write(f"<td class='{'positive' if strat.avg_return >= 0 else 'negative'}'>{strat.avg_return:+.1%}</td>\n")
                    f.write(f"<td>{strat.avg_sharpe:.2f}</td>\n")
                    f.write(f"<td>{strat.avg_max_drawdown:.1%}</td>\n")
                    f.write(f"<td>{strat.horizons_beat_buyhold}/{strat.total_horizons}</td>\n")
                    f.write("</tr>\n")

                f.write("</tbody>\n</table>\n")
            else:
                f.write("<p>None found</p>\n")

            # Time horizon recommendations
            f.write("<h2>⏰ Best Strategy by Time Horizon</h2>\n")
            f.write("<p><em>Choose strategy based on your investment timeline:</em></p>\n\n")
            f.write("<table>\n<thead>\n<tr><th>Horizon</th><th>Best Strategy</th><th>Return</th><th>vs Buy-Hold</th></tr>\n</thead>\n<tbody>\n")

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
                    f.write(f"<tr>\n")
                    f.write(f"<td><strong>{horizon.name}</strong></td>\n")
                    f.write(f"<td>{best_for_horizon}</td>\n")
                    f.write(f"<td class='positive'>{best_return:+.1%}</td>\n")
                    f.write(f"<td class='positive'>{best_return - buyhold:+.1%}</td>\n")
                    f.write("</tr>\n")
                else:
                    f.write(f"<tr>\n")
                    f.write(f"<td><strong>{horizon.name}</strong></td>\n")
                    f.write(f"<td><em>Buy-and-hold</em></td>\n")
                    f.write(f"<td class='{'positive' if buyhold >= 0 else 'negative'}'>{buyhold:+.1%}</td>\n")
                    f.write(f"<td>—</td>\n")
                    f.write("</tr>\n")

            f.write("</tbody>\n</table>\n")

            # Footer
            f.write("<hr>\n")
            f.write(f"<p><em>Generated by Master Strategy Analyzer (Simplified) on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>\n")
            f.write("</div>\n</body>\n</html>\n")

        logger.success(f"✓ HTML report saved to: {report_file}")

    def _get_css(self) -> str:
        """Return minimized CSS for HTML reports."""
        return """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   line-height: 1.6; color: #333; background: #f5f5f5; padding: 20px; }
            .container { max-width: 1400px; margin: 0 auto; background: white; padding: 40px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-radius: 8px; }
            h1 { color: #1a1a1a; font-size: 2.5em; margin-bottom: 20px;
                 border-bottom: 3px solid #4CAF50; padding-bottom: 15px; }
            h2 { color: #2c3e50; font-size: 2em; margin-top: 40px; margin-bottom: 20px;
                 padding-bottom: 10px; border-bottom: 2px solid #e0e0e0; }
            .metadata { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px;
                       border-left: 4px solid #4CAF50; }
            .metadata p { margin: 5px 0; }
            .metadata strong { color: #2c3e50; display: inline-block; min-width: 180px; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            thead { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            th { padding: 15px; text-align: left; font-weight: 600; text-transform: uppercase;
                font-size: 0.85em; letter-spacing: 0.5px; }
            td { padding: 12px 15px; border-bottom: 1px solid #e0e0e0; }
            tr:hover { background: #f8f9fa; }
            tbody tr:nth-child(even) { background: #fafafa; }
            .tier1 { background: #e8f5e9 !important; border-left: 4px solid #4CAF50; }
            .tier2 { background: #fff3e0 !important; border-left: 4px solid #FF9800; }
            .tier3 { background: #ffebee !important; border-left: 4px solid #f44336; }
            .positive { color: #4CAF50; font-weight: 600; }
            .negative { color: #f44336; font-weight: 600; }
            .blockquote { background: #fff8dc; border-left: 5px solid #ffa500; padding: 15px 20px;
                         margin: 20px 0; border-radius: 0 5px 5px 0; }
            .blockquote.warning { background: #fff3cd; border-left-color: #ffc107; }
            .recommendation-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                 color: white; padding: 25px; border-radius: 8px; margin: 20px 0;
                                 box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
            .recommendation-box h4 { color: white; margin-top: 0; font-size: 1.4em; }
        </style>
        """

    def _save_comparison_matrix(self) -> None:
        """Save comparison matrix as CSV."""
        csv_file = self.output_dir / "comparison_matrix.csv"
        df = pd.DataFrame(self.all_results)
        df.to_csv(csv_file, index=False)
        logger.success(f"✓ Comparison matrix saved to: {csv_file}")

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
            self.generate_html_report(strategy_scores)
            self._save_comparison_matrix()

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


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def analyze(
    symbol: str = typer.Option("BTC/USDT", "--symbol", "-s", help="Trading pair symbol"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    horizons: Optional[List[int]] = typer.Option(None, "--horizons", "-h", help="Custom time horizons in days"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick mode (fewer horizons)"),
    multi_pair: bool = typer.Option(False, "--multi-pair", "-m", help="Test multi-pair strategies"),
    output_dir: str = typer.Option("master_results", "--output", "-o", help="Output directory base name"),
):
    """
    Run comprehensive master strategy analysis (SIMPLIFIED VERSION).

    Tests all strategies across multiple time horizons, ranks them by
    risk-adjusted performance, and generates an HTML report with practical
    recommendations.
    """
    analyzer = MasterStrategyAnalyzer(
        symbol=symbol,
        timeframe=timeframe,
        horizons=horizons,
        workers=workers,
        quick_mode=quick,
        multi_pair=multi_pair,
        output_dir=output_dir
    )

    analyzer.run()


if __name__ == "__main__":
    app()
