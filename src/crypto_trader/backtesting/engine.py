"""
Backtesting Engine with VectorBT Integration

This module provides the main backtesting engine using VectorBT for vectorized
backtesting. It supports single strategy testing, multiple strategy comparison,
and parameter optimization.

**Purpose**: High-performance backtesting engine that evaluates trading strategies
on historical data with comprehensive performance metrics.

**Key Components**:
- BacktestEngine: Main backtesting orchestrator
- VectorBT Portfolio integration
- Performance metrics calculation
- Parameter optimization support
- Multiple strategy comparison

**Third-party packages**:
- vectorbt: https://vectorbt.dev/
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```python
engine = BacktestEngine()
result = engine.run_backtest(
    strategy=ma_crossover_strategy,
    data=btc_data,
    config=backtest_config
)
```

**Expected Output**:
```python
BacktestResult(
    strategy_name="MA Crossover",
    metrics=PerformanceMetrics(total_return=0.15, sharpe_ratio=1.8, ...),
    trades=[...],
    equity_curve=[...]
)
```
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

from crypto_trader.backtesting.executor import OrderExecutor
from crypto_trader.backtesting.portfolio import PortfolioManager
from crypto_trader.core.config import BacktestConfig
from crypto_trader.core.types import (
    BacktestResult,
    OrderSide,
    PerformanceMetrics,
    Signal,
    Timeframe,
    Trade,
)
from crypto_trader.strategies.base import BaseStrategy


class BacktestEngine:
    """
    Main backtesting engine using VectorBT for vectorized backtesting.

    Supports single strategy testing, multiple strategy comparison,
    and parameter optimization with comprehensive metrics calculation.
    """

    def __init__(self):
        """Initialize the backtesting engine."""
        logger.info("BacktestEngine initialized")

    def _signals_to_entries_exits(
        self, signals: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Convert strategy signals to entry/exit boolean series for VectorBT.

        Args:
            signals: DataFrame with 'signal' column containing BUY/SELL/HOLD

        Returns:
            Tuple of (entries, exits) boolean series
        """
        # Initialize as all False
        entries = pd.Series(False, index=signals.index)
        exits = pd.Series(False, index=signals.index)

        # Map signals to entries and exits
        for idx, row in signals.iterrows():
            signal = row['signal']
            if signal == 'BUY':
                entries.loc[idx] = True
            elif signal == 'SELL':
                exits.loc[idx] = True

        return entries, exits

    def _calculate_metrics(
        self,
        portfolio: vbt.Portfolio,
        trades_df: pd.DataFrame,
        initial_capital: float
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics from VectorBT portfolio.

        Args:
            portfolio: VectorBT Portfolio object
            trades_df: DataFrame with trade records
            initial_capital: Starting capital

        Returns:
            PerformanceMetrics object with all metrics
        """
        # Basic return metrics
        total_return = portfolio.total_return()
        final_value = portfolio.final_value()

        # Risk metrics
        sharpe_ratio = portfolio.sharpe_ratio()
        sortino_ratio = portfolio.sortino_ratio()
        max_drawdown = portfolio.max_drawdown()
        calmar_ratio = portfolio.calmar_ratio()

        # Trade metrics
        if len(trades_df) > 0:
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            # Average win/loss
            wins = trades_df[trades_df['pnl'] > 0]['pnl']
            losses = trades_df[trades_df['pnl'] < 0]['pnl']
            avg_win = wins.mean() if len(wins) > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0

            # Profit factor
            gross_profit = wins.sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

            # Consecutive wins/losses
            trade_results = (trades_df['pnl'] > 0).astype(int)
            trade_results = trade_results.replace(0, -1)  # Convert 0 to -1 for losses

            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_streak = 0
            current_type = 0

            for result in trade_results:
                if result == current_type:
                    current_streak += 1
                else:
                    if current_type == 1:
                        max_consecutive_wins = max(max_consecutive_wins, current_streak)
                    elif current_type == -1:
                        max_consecutive_losses = max(max_consecutive_losses, current_streak)
                    current_type = result
                    current_streak = 1

            # Final check for last streak
            if current_type == 1:
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            elif current_type == -1:
                max_consecutive_losses = max(max_consecutive_losses, current_streak)

            # Average trade duration
            avg_duration = trades_df['duration_minutes'].mean() if 'duration_minutes' in trades_df.columns else 0.0

            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

            # Total fees
            total_fees = trades_df['fees'].sum() if 'fees' in trades_df.columns else 0.0
        else:
            # No trades executed
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            avg_duration = 0.0
            expectancy = 0.0
            total_fees = 0.0

        # Recovery factor
        recovery_factor = (final_value - initial_capital) / abs(max_drawdown * initial_capital) if max_drawdown != 0 else 0.0

        return PerformanceMetrics(
            total_return=float(total_return),
            sharpe_ratio=float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0,
            max_drawdown=float(abs(max_drawdown)),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            total_trades=int(total_trades),
            winning_trades=int(winning_trades),
            losing_trades=int(losing_trades),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            max_consecutive_wins=int(max_consecutive_wins),
            max_consecutive_losses=int(max_consecutive_losses),
            avg_trade_duration=float(avg_duration),
            sortino_ratio=float(sortino_ratio) if not np.isnan(sortino_ratio) else 0.0,
            calmar_ratio=float(calmar_ratio) if not np.isnan(calmar_ratio) else 0.0,
            recovery_factor=float(recovery_factor),
            expectancy=float(expectancy),
            total_fees=float(total_fees),
            final_capital=float(final_value)
        )

    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        config: BacktestConfig,
        symbol: str = "BTCUSDT",
        timeframe: Timeframe = Timeframe.HOUR_1
    ) -> BacktestResult:
        """
        Run backtest for a single strategy on historical data.

        Args:
            strategy: Trading strategy to backtest
            data: Historical OHLCV data with required indicators
            config: Backtest configuration
            symbol: Trading pair symbol
            timeframe: Data timeframe

        Returns:
            BacktestResult with complete backtest results

        Raises:
            ValueError: If data validation fails
        """
        logger.info(
            f"Starting backtest: {strategy.name} on {symbol} "
            f"({timeframe.value}), capital=${config.initial_capital:,.2f}"
        )

        # Validate data
        if not strategy.validate_data(data):
            raise ValueError("Strategy data validation failed")

        # Generate signals
        signals = strategy.generate_signals(data)

        # Ensure all series share the actual timestamp index so downstream consumers
        # (reports/exports) receive real datetimes instead of integer positions.
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
        else:
            timestamps = data.index

        close_series = pd.Series(data['close'].values, index=timestamps, name='close')

        if 'timestamp' in signals.columns:
            signal_index = pd.to_datetime(signals['timestamp'])
            signals = signals.copy()
            signals.index = signal_index
        else:
            signals = signals.copy()
            signals.index = timestamps

        signals = signals.sort_index()

        # Convert signals to entries/exits for VectorBT
        entries, exits = self._signals_to_entries_exits(signals)
        entries = entries.reindex(close_series.index, fill_value=False)
        exits = exits.reindex(close_series.index, fill_value=False)

        # Create VectorBT portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=close_series,
            entries=entries,
            exits=exits,
            init_cash=config.initial_capital,
            fees=config.trading_fee_percent,
            slippage=config.slippage_percent,
            size=np.inf,  # Use all available cash for each trade
            size_type='value',  # Size in value terms (VectorBT SizeType.Value)
            freq='1h'  # Adjust based on timeframe
        )

        # Extract trade records
        trades_list: List[Trade] = []
        try:
            vbt_trades = portfolio.trades.records_readable

            if len(vbt_trades) > 0:
                for _, trade_row in vbt_trades.iterrows():
                    # Convert VectorBT trade to our Trade type
                    entry_time = trade_row['Entry Timestamp']
                    exit_time = trade_row['Exit Timestamp']
                    entry_price = trade_row['Avg Entry Price']  # VectorBT uses 'Avg Entry Price'
                    exit_price = trade_row['Avg Exit Price']    # VectorBT uses 'Avg Exit Price'
                    size = trade_row['Size']
                    pnl = trade_row['PnL']
                    return_pct = trade_row['Return'] * 100

                    # Calculate duration
                    duration = (exit_time - entry_time).total_seconds() / 60

                    # Estimate fees (VectorBT includes them in PnL)
                    trade_value = size * entry_price
                    fees = trade_value * config.trading_fee_percent * 2  # Entry + exit

                    trade = Trade(
                        symbol=symbol,
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        side=OrderSide.BUY,  # VectorBT default is long
                        quantity=size,
                        pnl=pnl,
                        pnl_percent=return_pct,
                        fees=fees
                    )
                    trades_list.append(trade)

                    # Create DataFrame for metrics calculation
                    trades_df = pd.DataFrame([{
                        'pnl': t.pnl,
                        'fees': t.fees,
                        'duration_minutes': t.duration_minutes
                    } for t in trades_list])
            else:
                trades_df = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Could not extract trade records: {e}")
            trades_df = pd.DataFrame()

        # Calculate metrics
        metrics = self._calculate_metrics(portfolio, trades_df, config.initial_capital)

        # Extract equity curve
        equity_curve = []
        equity_series = portfolio.value()
        for timestamp, value in equity_series.items():
            equity_curve.append((timestamp, float(value)))

        # Get date range
        start_date = data['timestamp'].iloc[0] if 'timestamp' in data.columns else data.index[0]
        end_date = data['timestamp'].iloc[-1] if 'timestamp' in data.columns else data.index[-1]

        if not isinstance(start_date, datetime):
            start_date = pd.to_datetime(start_date)
        if not isinstance(end_date, datetime):
            end_date = pd.to_datetime(end_date)

        # Create result
        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=config.initial_capital,
            metrics=metrics,
            trades=trades_list,
            equity_curve=equity_curve,
            metadata={
                'strategy_params': strategy.get_parameters(),
                'backtest_config': {
                    'fee_percent': config.trading_fee_percent,
                    'slippage_percent': config.slippage_percent,
                    'max_position_size': config.max_position_size
                }
            }
        )

        logger.info(
            f"Backtest complete: {metrics.total_trades} trades, "
            f"{metrics.total_return:.2%} return, Sharpe={metrics.sharpe_ratio:.2f}"
        )

        return result

    def run_multiple(
        self,
        strategies: List[BaseStrategy],
        data: pd.DataFrame,
        config: BacktestConfig,
        symbol: str = "BTCUSDT",
        timeframe: Timeframe = Timeframe.HOUR_1
    ) -> List[BacktestResult]:
        """
        Run backtests for multiple strategies on the same data.

        Args:
            strategies: List of strategies to test
            data: Historical OHLCV data
            config: Backtest configuration
            symbol: Trading pair symbol
            timeframe: Data timeframe

        Returns:
            List of BacktestResult objects, one per strategy
        """
        logger.info(f"Running backtest for {len(strategies)} strategies")

        results = []
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, data, config, symbol, timeframe)
                results.append(result)
            except Exception as e:
                logger.error(f"Backtest failed for {strategy.name}: {e}")
                continue

        # Sort by total return
        results.sort(key=lambda r: r.metrics.total_return, reverse=True)

        logger.info(f"Completed {len(results)} backtests successfully")
        return results

    def optimize_parameters(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        config: BacktestConfig,
        param_grid: Dict[str, List],
        symbol: str = "BTCUSDT",
        timeframe: Timeframe = Timeframe.HOUR_1
    ) -> Tuple[BacktestResult, Dict]:
        """
        Optimize strategy parameters using grid search.

        Args:
            strategy: Strategy to optimize
            data: Historical OHLCV data
            config: Backtest configuration
            param_grid: Dictionary mapping parameter names to lists of values
            symbol: Trading pair symbol
            timeframe: Data timeframe

        Returns:
            Tuple of (best_result, best_params)
        """
        logger.info(f"Starting parameter optimization for {strategy.name}")
        logger.info(f"Parameter grid: {param_grid}")

        # Generate all parameter combinations
        import itertools

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(f"Testing {len(combinations)} parameter combinations")

        best_result = None
        best_params = None
        best_sharpe = -np.inf

        for combo in combinations:
            # Create parameter dict
            params = dict(zip(param_names, combo))

            # Update strategy parameters
            strategy.set_parameters(params)
            strategy.initialize(strategy.config)

            try:
                # Run backtest
                result = self.run_backtest(strategy, data, config, symbol, timeframe)

                # Check if this is the best result
                # Use Sharpe ratio as optimization metric
                if result.metrics.sharpe_ratio > best_sharpe:
                    best_sharpe = result.metrics.sharpe_ratio
                    best_result = result
                    best_params = params.copy()

                logger.debug(
                    f"Params {params}: Return={result.metrics.total_return:.2%}, "
                    f"Sharpe={result.metrics.sharpe_ratio:.2f}"
                )

            except Exception as e:
                logger.warning(f"Backtest failed for params {params}: {e}")
                continue

        if best_result is None:
            raise ValueError("All parameter combinations failed")

        logger.info(
            f"Optimization complete. Best Sharpe: {best_sharpe:.2f}, "
            f"Best params: {best_params}"
        )

        return best_result, best_params


if __name__ == "__main__":
    """
    Validation block for backtesting engine.
    Tests with real MA crossover strategy on synthetic BTC data.
    """
    import sys

    from crypto_trader.core.config import BacktestConfig
    from crypto_trader.strategies.library.moving_average_crossover import MovingAverageCrossover

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating engine.py with real strategy and data...\n")

    # Test 1: Create synthetic market data
    total_tests += 1
    print("Test 1: Create synthetic market data")
    try:
        # Generate realistic price data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1h')
        np.random.seed(42)

        # Simulate price with trend
        prices = 40000 + np.cumsum(np.random.randn(len(dates)) * 200)

        # Calculate moving averages
        prices_series = pd.Series(prices)
        sma_10 = prices_series.rolling(window=10).mean()
        sma_20 = prices_series.rolling(window=20).mean()

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(100, 1000, len(dates)),
            'SMA_10': sma_10,
            'SMA_20': sma_20
        })

        # Remove NaN rows
        data = data.dropna()

        if len(data) < 100:
            all_validation_failures.append(f"Insufficient data: Expected >100 rows, got {len(data)}")

        print(f"  ✓ Generated {len(data)} candles")
        print(f"  ✓ Date range: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
        print(f"  ✓ Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    except Exception as e:
        all_validation_failures.append(f"Data creation test exception: {e}")

    # Test 2: Initialize strategy
    total_tests += 1
    print("\nTest 2: Initialize MA Crossover strategy")
    try:
        strategy = MovingAverageCrossover(
            name="MA_Cross_10_20",
            config={
                "fast_period": 10,
                "slow_period": 20,
                "signal_threshold": 0.5  # Lower threshold for more signals
            }
        )
        strategy.initialize(strategy.config)

        if not strategy._initialized:
            all_validation_failures.append("Strategy not initialized")

        print(f"  ✓ Strategy: {strategy.name}")
        print(f"  ✓ Parameters: {strategy.get_parameters()}")
    except Exception as e:
        all_validation_failures.append(f"Strategy initialization test exception: {e}")

    # Test 3: Create backtest config
    total_tests += 1
    print("\nTest 3: Create backtest configuration")
    try:
        config = BacktestConfig(
            initial_capital=10000.0,
            trading_fee_percent=0.001,
            slippage_percent=0.0005,
            max_position_size=0.95
        )

        if config.initial_capital != 10000.0:
            all_validation_failures.append(f"Initial capital: Expected 10000.0, got {config.initial_capital}")

        print(f"  ✓ Initial capital: ${config.initial_capital:,.2f}")
        print(f"  ✓ Trading fee: {config.trading_fee_percent:.4%}")
        print(f"  ✓ Slippage: {config.slippage_percent:.4%}")
    except Exception as e:
        all_validation_failures.append(f"Config creation test exception: {e}")

    # Test 4: Initialize backtesting engine
    total_tests += 1
    print("\nTest 4: Initialize backtesting engine")
    try:
        engine = BacktestEngine()
        print(f"  ✓ Engine initialized successfully")
    except Exception as e:
        all_validation_failures.append(f"Engine initialization test exception: {e}")

    # Test 5: Run backtest
    total_tests += 1
    print("\nTest 5: Run complete backtest")
    try:
        result = engine.run_backtest(
            strategy=strategy,
            data=data,
            config=config,
            symbol="BTCUSDT",
            timeframe=Timeframe.HOUR_1
        )

        if not isinstance(result, BacktestResult):
            all_validation_failures.append(f"Result type: Expected BacktestResult, got {type(result)}")

        if result.strategy_name != strategy.name:
            all_validation_failures.append(f"Strategy name: Expected {strategy.name}, got {result.strategy_name}")

        if result.symbol != "BTCUSDT":
            all_validation_failures.append(f"Symbol: Expected BTCUSDT, got {result.symbol}")

        print(f"  ✓ Backtest completed successfully")
        print(f"  ✓ Strategy: {result.strategy_name}")
        print(f"  ✓ Period: {result.duration_days} days")
    except Exception as e:
        all_validation_failures.append(f"Run backtest test exception: {e}")

    # Test 6: Verify metrics
    total_tests += 1
    print("\nTest 6: Verify performance metrics")
    try:
        metrics = result.metrics

        if not isinstance(metrics, PerformanceMetrics):
            all_validation_failures.append(f"Metrics type: Expected PerformanceMetrics, got {type(metrics)}")

        if metrics.final_capital <= 0:
            all_validation_failures.append(f"Final capital: Expected >0, got {metrics.final_capital}")

        # Total trades might be 0 if no crossovers meet threshold
        if metrics.total_trades > 0:
            if metrics.winning_trades + metrics.losing_trades != metrics.total_trades:
                all_validation_failures.append(
                    f"Trade count mismatch: {metrics.winning_trades} + {metrics.losing_trades} != {metrics.total_trades}"
                )

        print(f"  ✓ Total return: {metrics.total_return:.2%}")
        print(f"  ✓ Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  ✓ Max drawdown: {metrics.max_drawdown:.2%}")
        print(f"  ✓ Total trades: {metrics.total_trades}")
        print(f"  ✓ Win rate: {metrics.win_rate:.2%}")
        print(f"  ✓ Final capital: ${metrics.final_capital:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Metrics verification test exception: {e}")

    # Test 7: Verify equity curve
    total_tests += 1
    print("\nTest 7: Verify equity curve")
    try:
        equity_curve = result.equity_curve

        if not isinstance(equity_curve, list):
            all_validation_failures.append(f"Equity curve type: Expected list, got {type(equity_curve)}")
        elif len(equity_curve) == 0:
            all_validation_failures.append("Equity curve is empty")
        else:
            # Check first and last values
            first_time, first_value = equity_curve[0]
            last_time, last_value = equity_curve[-1]

            if first_value != config.initial_capital:
                # Allow small difference due to VectorBT calculations
                if abs(first_value - config.initial_capital) > 1.0:
                    all_validation_failures.append(
                        f"First equity: Expected ~{config.initial_capital:.2f}, got {first_value:.2f}"
                    )

            print(f"  ✓ Equity curve length: {len(equity_curve)} points")
            print(f"  ✓ Start: ${first_value:,.2f}")
            print(f"  ✓ End: ${last_value:,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Equity curve test exception: {e}")

    # Test 8: Result summary
    total_tests += 1
    print("\nTest 8: Result summary generation")
    try:
        summary = result.summary()

        required_keys = {'strategy', 'symbol', 'timeframe', 'duration_days',
                        'total_return', 'sharpe_ratio', 'max_drawdown',
                        'win_rate', 'total_trades', 'final_capital', 'quality'}

        if not required_keys.issubset(summary.keys()):
            all_validation_failures.append(
                f"Summary keys: Missing {required_keys - set(summary.keys())}"
            )

        print(f"  ✓ Summary generated with {len(summary)} fields")
        print(f"  ✓ Quality rating: {summary['quality']}")
    except Exception as e:
        all_validation_failures.append(f"Summary test exception: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Backtesting engine is validated and ready for production use")
        print(f"\nSample backtest result:")
        print(f"  • Strategy: {result.strategy_name}")
        print(f"  • Return: {result.metrics.total_return:.2%}")
        print(f"  • Sharpe: {result.metrics.sharpe_ratio:.2f}")
        print(f"  • Trades: {result.metrics.total_trades}")
        print(f"  • Quality: {result.metrics.risk_adjusted_quality()}")
        sys.exit(0)
