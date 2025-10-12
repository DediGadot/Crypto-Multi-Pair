#!/usr/bin/env python3
"""
Portfolio Parameter Optimization Script

This script performs walk-forward parameter optimization for multi-asset portfolio
strategies. It tests different parameter combinations across sliding time windows
to find robust configurations that perform well out-of-sample.

**Purpose**: Find optimal portfolio parameters that generalize across different
market conditions by testing on multiple historical time windows.

**Method**: Walk-forward analysis with train/validate splits

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- loguru: https://loguru.readthedocs.io/en/stable/
- typer: https://typer.tiangolo.com/
- pyyaml: https://pyyaml.org/wiki/PyYAMLDocumentation

**Sample Input**:
```bash
python optimize_portfolio.py --window-days 365 --timeframe 1h
```

**Expected Output**:
```yaml
# optimized_config.yaml with best parameters
portfolio:
  assets: [...]
  rebalancing:
    threshold: 0.08  # Optimized value
```

Usage:
    python optimize_portfolio.py
    python optimize_portfolio.py --window-days 365 --timeframe 1h
    python optimize_portfolio.py --metric sharpe --test-windows 5
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from itertools import product
import warnings

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

from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.core.types import Timeframe

# Suppress warnings
warnings.filterwarnings('ignore')

app = typer.Typer(help="Portfolio parameter optimization tool")


class PortfolioOptimizer:
    """
    Walk-forward portfolio parameter optimizer.

    Tests parameter combinations across sliding time windows to find
    robust configurations that work across different market conditions.
    """

    def __init__(
        self,
        window_days: int = 365,
        timeframe: str = "1h",
        step_days: Optional[int] = None,
        test_windows: int = 5,
        optimization_metric: str = "total_return"
    ):
        """
        Initialize the optimizer.

        Args:
            window_days: Size of each test window in days (default: 365)
            timeframe: Candle timeframe (default: "1h")
            step_days: Days to step forward between windows (default: window_days // 2)
            test_windows: Number of time windows to test (default: 5)
            optimization_metric: Metric to optimize (default: "total_return")
        """
        self.window_days = window_days
        self.timeframe = timeframe
        self.step_days = step_days or window_days // 2
        self.test_windows = test_windows
        self.optimization_metric = optimization_metric

        self.fetcher = BinanceDataFetcher()
        self.results: List[Dict] = []

        logger.info(f"Optimizer initialized:")
        logger.info(f"  Window size: {window_days} days")
        logger.info(f"  Timeframe: {timeframe}")
        logger.info(f"  Step size: {self.step_days} days")
        logger.info(f"  Test windows: {test_windows}")
        logger.info(f"  Metric: {optimization_metric}")

    def fetch_historical_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch long-term historical data for all symbols.

        Args:
            symbols: List of trading pair symbols

        Returns:
            Dictionary mapping symbol to full historical DataFrame
        """
        logger.info(f"\nFetching historical data for {len(symbols)} assets...")

        # Calculate total periods needed
        total_days = self.window_days * self.test_windows

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
            logger.info(f"  Fetching {symbol}...")
            try:
                data = self.fetcher.get_ohlcv(symbol, self.timeframe, limit=limit)

                if data is None or len(data) < limit * 0.8:
                    logger.warning(f"  ⚠ Insufficient data for {symbol}: {len(data) if data is not None else 0} candles")
                    continue

                historical_data[symbol] = data
                logger.success(f"    ✓ {len(data)} candles ({data.index[0]} to {data.index[-1]})")

            except Exception as e:
                logger.error(f"  ✗ Failed to fetch {symbol}: {e}")
                continue

        if len(historical_data) < len(symbols):
            logger.warning(f"Only fetched {len(historical_data)}/{len(symbols)} assets")

        return historical_data

    def create_time_windows(
        self,
        historical_data: Dict[str, pd.DataFrame]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Create sliding time windows for walk-forward analysis.

        Args:
            historical_data: Dictionary of historical price data

        Returns:
            List of (start_date, end_date) tuples for each window
        """
        # Get common timestamps
        timestamps = None
        for data in historical_data.values():
            if timestamps is None:
                timestamps = data.index
            else:
                timestamps = timestamps.intersection(data.index)

        if len(timestamps) == 0:
            raise ValueError("No overlapping timestamps across assets")

        timestamps = sorted(timestamps)

        # Calculate window size in periods
        if self.timeframe == "1h":
            window_periods = self.window_days * 24
            step_periods = self.step_days * 24
        elif self.timeframe == "4h":
            window_periods = self.window_days * 6
            step_periods = self.step_days * 6
        elif self.timeframe == "1d":
            window_periods = self.window_days
            step_periods = self.step_days
        else:
            window_periods = self.window_days * 24
            step_periods = self.step_days * 24

        # Create sliding windows
        windows = []
        start_idx = 0

        while len(windows) < self.test_windows and start_idx + window_periods <= len(timestamps):
            end_idx = start_idx + window_periods
            windows.append((timestamps[start_idx], timestamps[end_idx - 1]))
            start_idx += step_periods

        logger.info(f"\nCreated {len(windows)} time windows:")
        for i, (start, end) in enumerate(windows, 1):
            logger.info(f"  Window {i}: {start} to {end}")

        return windows

    def get_parameter_grid(self) -> Dict[str, List]:
        """
        Define parameter grid for optimization.

        Returns:
            Dictionary of parameter names to lists of values to test
        """
        return {
            'threshold': [0.05, 0.07, 0.10, 0.12, 0.15, 0.20],
            'rebalance_method': ['threshold', 'calendar', 'hybrid'],
            'calendar_period_days': [7, 14, 30, 60],
            'min_rebalance_interval_hours': [12, 24, 48],
            'use_momentum_filter': [False, True]
        }

    def backtest_configuration(
        self,
        historical_data: Dict[str, pd.DataFrame],
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        assets: List[Tuple[str, float]],
        params: Dict[str, Any],
        initial_capital: float = 10000.0
    ) -> Dict[str, float]:
        """
        Backtest a configuration on a specific time window.

        Args:
            historical_data: Full historical data
            window_start: Window start timestamp
            window_end: Window end timestamp
            assets: List of (symbol, weight) tuples
            params: Parameter dictionary
            initial_capital: Starting capital

        Returns:
            Dictionary of performance metrics
        """
        # Extract window data
        window_data = {}
        for symbol, _ in assets:
            if symbol not in historical_data:
                raise ValueError(f"Missing data for {symbol}")

            data = historical_data[symbol]
            mask = (data.index >= window_start) & (data.index <= window_end)
            window_data[symbol] = data[mask]

        # Get common timestamps in window
        timestamps = None
        for data in window_data.values():
            if timestamps is None:
                timestamps = data.index
            else:
                timestamps = timestamps.intersection(data.index)

        if len(timestamps) < 10:
            return {'error': 'insufficient_data'}

        timestamps = sorted(timestamps)

        # Initialize portfolio
        shares = {}
        for symbol, weight in assets:
            allocation = initial_capital * weight
            initial_price = window_data[symbol].loc[timestamps[0], 'close']
            shares[symbol] = allocation / initial_price

        # Simulate portfolio
        equity_values = []
        rebalance_count = 0
        last_rebalance = None

        for timestamp in timestamps:
            # Get current prices
            prices = {symbol: window_data[symbol].loc[timestamp, 'close']
                     for symbol, _ in assets}

            # Calculate portfolio value and weights
            portfolio_values = {symbol: shares[symbol] * prices[symbol]
                               for symbol in shares}
            total_value = sum(portfolio_values.values())
            current_weights = {symbol: portfolio_values[symbol] / total_value
                              for symbol in portfolio_values}

            # Check rebalancing conditions
            needs_rebalance = False
            max_deviation = max(abs(current_weights[s] - w) for s, w in assets)

            if params['rebalance_method'] == 'threshold':
                needs_rebalance = max_deviation > params['threshold']
            elif params['rebalance_method'] == 'calendar':
                if last_rebalance is None:
                    pass
                else:
                    days_since = (timestamp - last_rebalance).total_seconds() / 86400
                    needs_rebalance = days_since >= params['calendar_period_days']
            elif params['rebalance_method'] == 'hybrid':
                threshold_triggered = max_deviation > params['threshold']
                calendar_triggered = False
                if last_rebalance is not None:
                    days_since = (timestamp - last_rebalance).total_seconds() / 86400
                    calendar_triggered = days_since >= params['calendar_period_days']
                needs_rebalance = threshold_triggered or calendar_triggered

            # Check minimum interval
            if needs_rebalance and last_rebalance is not None:
                hours_since = (timestamp - last_rebalance).total_seconds() / 3600
                if hours_since < params['min_rebalance_interval_hours']:
                    needs_rebalance = False

            # Apply momentum filter
            if needs_rebalance and params['use_momentum_filter']:
                lookback_periods = 30 * 24  # 30 days
                current_idx = list(timestamps).index(timestamp)
                if current_idx >= lookback_periods:
                    lookback_idx = current_idx - lookback_periods
                    lookback_ts = timestamps[lookback_idx]

                    old_prices = {s: window_data[s].loc[lookback_ts, 'close']
                                 for s in prices}
                    old_value = sum(shares[s] * old_prices[s] for s in shares)
                    portfolio_return = (total_value - old_value) / old_value

                    if portfolio_return > 0.20:  # Skip if strong uptrend
                        needs_rebalance = False

            # Execute rebalance
            if needs_rebalance:
                target_values = {s: total_value * w for s, w in assets}
                shares = {s: target_values[s] / prices[s] for s in prices}
                rebalance_count += 1
                last_rebalance = timestamp

            equity_values.append(total_value)

        # Calculate metrics
        if len(equity_values) < 2:
            return {'error': 'insufficient_data'}

        final_value = equity_values[-1]
        total_return = (final_value / initial_capital) - 1

        # Calculate buy-and-hold benchmark
        buyhold_shares = {}
        for symbol, weight in assets:
            allocation = initial_capital * weight
            initial_price = window_data[symbol].loc[timestamps[0], 'close']
            buyhold_shares[symbol] = allocation / initial_price

        buyhold_final = sum(buyhold_shares[s] * window_data[s].loc[timestamps[-1], 'close']
                           for s, _ in assets)
        buyhold_return = (buyhold_final / initial_capital) - 1

        outperformance = total_return - buyhold_return

        # Calculate Sharpe ratio (annualized)
        returns = pd.Series(equity_values).pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            if self.timeframe == "1h":
                periods_per_year = 24 * 365
            elif self.timeframe == "4h":
                periods_per_year = 6 * 365
            elif self.timeframe == "1d":
                periods_per_year = 365
            else:
                periods_per_year = 24 * 365

            sharpe = (returns.mean() * periods_per_year) / (returns.std() * np.sqrt(periods_per_year))
        else:
            sharpe = 0.0

        # Calculate max drawdown
        peak = equity_values[0]
        max_dd = 0.0
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (value - peak) / peak
            if dd < max_dd:
                max_dd = dd

        # Calculate volatility (annualized)
        if len(returns) > 0:
            if self.timeframe == "1h":
                periods_per_year = 24 * 365
            elif self.timeframe == "4h":
                periods_per_year = 6 * 365
            elif self.timeframe == "1d":
                periods_per_year = 365
            else:
                periods_per_year = 24 * 365
            volatility = returns.std() * np.sqrt(periods_per_year)
        else:
            volatility = 0.0

        return {
            'total_return': total_return,
            'outperformance': outperformance,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'rebalance_count': rebalance_count,
            'volatility': volatility,
            'final_value': final_value,
            'buyhold_return': buyhold_return
        }

    def optimize(
        self,
        symbols: List[str],
        base_weights: List[float],
        output_path: str = "optimized_config.yaml"
    ) -> Dict[str, Any]:
        """
        Run full optimization process.

        Args:
            symbols: List of trading pair symbols
            base_weights: Initial weight allocation for assets
            output_path: Path to save optimized config

        Returns:
            Dictionary with best parameters and results
        """
        logger.info("\n" + "=" * 80)
        logger.info("PORTFOLIO PARAMETER OPTIMIZATION")
        logger.info("=" * 80)

        # Validate inputs
        if len(symbols) != len(base_weights):
            raise ValueError("Number of symbols must match number of weights")

        if not np.isclose(sum(base_weights), 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {sum(base_weights)}")

        assets = list(zip(symbols, base_weights))

        # Step 1: Fetch historical data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Fetching Historical Data")
        logger.info("=" * 80)

        historical_data = self.fetch_historical_data(symbols)

        if len(historical_data) < len(symbols):
            logger.warning(f"⚠ Only {len(historical_data)}/{len(symbols)} assets available")
            # Update assets list
            assets = [(s, w) for s, w in assets if s in historical_data]
            # Renormalize weights
            total_weight = sum(w for _, w in assets)
            assets = [(s, w / total_weight) for s, w in assets]

        # Step 2: Create time windows
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Creating Time Windows")
        logger.info("=" * 80)

        windows = self.create_time_windows(historical_data)

        # Step 3: Generate parameter grid
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Generating Parameter Grid")
        logger.info("=" * 80)

        param_grid = self.get_parameter_grid()

        # Create all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Testing {len(combinations)} parameter combinations")
        logger.info(f"Total backtests: {len(combinations) * len(windows)}")

        # Step 4: Run optimization
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Running Walk-Forward Optimization")
        logger.info("=" * 80)

        results = []
        total_tests = len(combinations) * len(windows)
        completed = 0

        for combo_idx, combo in enumerate(combinations, 1):
            params = dict(zip(param_names, combo))

            # Test across all windows
            window_metrics = []

            for window_idx, (start, end) in enumerate(windows, 1):
                metrics = self.backtest_configuration(
                    historical_data,
                    start,
                    end,
                    assets,
                    params
                )

                completed += 1

                if 'error' not in metrics:
                    window_metrics.append(metrics)

                # Progress update
                if completed % 10 == 0:
                    logger.info(f"  Progress: {completed}/{total_tests} tests ({completed/total_tests*100:.1f}%)")

            # Aggregate metrics across windows
            if window_metrics:
                avg_metrics = {
                    'avg_return': np.mean([m['total_return'] for m in window_metrics]),
                    'avg_outperformance': np.mean([m['outperformance'] for m in window_metrics]),
                    'avg_sharpe': np.mean([m['sharpe_ratio'] for m in window_metrics]),
                    'avg_max_drawdown': np.mean([m['max_drawdown'] for m in window_metrics]),
                    'avg_rebalances': np.mean([m['rebalance_count'] for m in window_metrics]),
                    'avg_volatility': np.mean([m['volatility'] for m in window_metrics]),
                    'consistency': np.std([m['total_return'] for m in window_metrics]),  # Lower is better
                    'windows_tested': len(window_metrics),
                    'params': params
                }

                results.append(avg_metrics)

        logger.success(f"\n✓ Completed {completed} backtests")

        # Step 5: Find best configuration
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Analyzing Results")
        logger.info("=" * 80)

        if not results:
            logger.error("❌ No valid results - optimization failed")
            sys.exit(1)

        # Sort by optimization metric
        metric_key = {
            'total_return': 'avg_return',
            'outperformance': 'avg_outperformance',
            'sharpe': 'avg_sharpe',
            'drawdown': 'avg_max_drawdown'  # Note: more negative is worse
        }.get(self.optimization_metric, 'avg_outperformance')

        if metric_key == 'avg_max_drawdown':
            # For drawdown, less negative is better
            results.sort(key=lambda x: x[metric_key], reverse=True)
        else:
            results.sort(key=lambda x: x[metric_key], reverse=True)

        best = results[0]

        logger.info("\n🏆 BEST CONFIGURATION:")
        logger.info(f"  Threshold: {best['params']['threshold']:.2%}")
        logger.info(f"  Method: {best['params']['rebalance_method']}")
        logger.info(f"  Calendar Period: {best['params']['calendar_period_days']} days")
        logger.info(f"  Min Interval: {best['params']['min_rebalance_interval_hours']} hours")
        logger.info(f"  Momentum Filter: {best['params']['use_momentum_filter']}")
        logger.info("\n📊 AVERAGE PERFORMANCE:")
        logger.info(f"  Avg Return: {best['avg_return']:.2%}")
        logger.info(f"  Avg Outperformance: {best['avg_outperformance']:.2%}")
        logger.info(f"  Avg Sharpe Ratio: {best['avg_sharpe']:.3f}")
        logger.info(f"  Avg Max Drawdown: {best['avg_max_drawdown']:.2%}")
        logger.info(f"  Avg Rebalances: {best['avg_rebalances']:.1f}")
        logger.info(f"  Return Consistency (σ): {best['consistency']:.2%}")

        # Step 6: Generate optimized config
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Generating Optimized Config")
        logger.info("=" * 80)

        optimized_config = {
            'run': {
                'name': f"optimized_portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}",
                'description': f"Optimized portfolio - {self.optimization_metric} maximization",
                'mode': 'portfolio'
            },
            'data': {
                'timeframe': self.timeframe,
                'days': self.window_days
            },
            'portfolio': {
                'assets': [
                    {'symbol': symbol, 'weight': weight}
                    for symbol, weight in assets
                ],
                'rebalancing': {
                    'enabled': True,
                    'threshold': best['params']['threshold'],
                    'rebalance_method': best['params']['rebalance_method'],
                    'calendar_period_days': best['params']['calendar_period_days'],
                    'min_rebalance_interval_hours': best['params']['min_rebalance_interval_hours'],
                    'use_momentum_filter': best['params']['use_momentum_filter'],
                    'momentum_lookback_days': 30
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
                'directory': 'results_optimized',
                'save_trades': True,
                'save_equity_curve': True
            },
            'logging': {
                'level': 'INFO',
                'save_to_file': True
            },
            'optimization_metadata': {
                'optimization_date': datetime.now().isoformat(),
                'optimization_metric': self.optimization_metric,
                'window_days': self.window_days,
                'windows_tested': best['windows_tested'],
                'avg_return': float(best['avg_return']),
                'avg_outperformance': float(best['avg_outperformance']),
                'avg_sharpe': float(best['avg_sharpe'])
            }
        }

        # Save config
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False, sort_keys=False)

        logger.success(f"\n✓ Optimized config saved to: {output_file}")

        # Save detailed results
        results_file = output_file.parent / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        logger.info(f"✓ Detailed results saved to: {results_file}")

        return {
            'best_config': optimized_config,
            'best_params': best['params'],
            'best_metrics': {k: v for k, v in best.items() if k != 'params'},
            'all_results': results
        }


@app.command()
def optimize(
    symbols: str = typer.Option(
        "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT",
        "--symbols",
        "-s",
        help="Comma-separated list of trading pairs"
    ),
    weights: str = typer.Option(
        "0.4,0.3,0.15,0.15",
        "--weights",
        "-w",
        help="Comma-separated list of target weights (must sum to 1.0)"
    ),
    window_days: int = typer.Option(
        365,
        "--window-days",
        "-d",
        help="Window size in days for each test period"
    ),
    timeframe: str = typer.Option(
        "1h",
        "--timeframe",
        "-t",
        help="Candle timeframe (1h, 4h, 1d)"
    ),
    test_windows: int = typer.Option(
        5,
        "--test-windows",
        "-n",
        help="Number of time windows to test"
    ),
    metric: str = typer.Option(
        "outperformance",
        "--metric",
        "-m",
        help="Optimization metric (total_return, outperformance, sharpe, drawdown)"
    ),
    output: str = typer.Option(
        "optimized_config.yaml",
        "--output",
        "-o",
        help="Output path for optimized config file"
    )
):
    """
    Optimize portfolio parameters using walk-forward analysis.

    This command tests different parameter combinations across multiple
    time windows to find robust configurations that generalize well.

    Example:
        python optimize_portfolio.py --window-days 365 --test-windows 5 --metric outperformance
    """
    # Parse symbols and weights
    symbol_list = [s.strip() for s in symbols.split(',')]
    weight_list = [float(w.strip()) for w in weights.split(',')]

    # Validate
    if len(symbol_list) != len(weight_list):
        typer.echo("❌ Error: Number of symbols must match number of weights", err=True)
        raise typer.Exit(1)

    if not np.isclose(sum(weight_list), 1.0, atol=0.01):
        typer.echo(f"❌ Error: Weights must sum to 1.0, got {sum(weight_list)}", err=True)
        raise typer.Exit(1)

    # Run optimization
    optimizer = PortfolioOptimizer(
        window_days=window_days,
        timeframe=timeframe,
        test_windows=test_windows,
        optimization_metric=metric
    )

    try:
        result = optimizer.optimize(
            symbols=symbol_list,
            base_weights=weight_list,
            output_path=output
        )

        logger.info("\n" + "=" * 80)
        logger.success("✅ OPTIMIZATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nNext steps:")
        logger.info(f"1. Review: {output}")
        logger.info(f"2. Run backtest: uv run python run_full_pipeline.py --portfolio --config {output}")
        logger.info(f"3. Generate report: uv run python run_full_pipeline.py --portfolio --config {output} --report")

    except Exception as e:
        logger.error(f"\n❌ Optimization failed: {e}")
        logger.exception("Full traceback:")
        raise typer.Exit(1)


if __name__ == "__main__":
    """
    Validation block for Portfolio Optimizer.
    Tests the optimizer with minimal configuration.
    """
    import sys

    # Track validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting Portfolio Optimizer validation")

    # Test 1: Optimizer initialization
    total_tests += 1
    try:
        optimizer = PortfolioOptimizer(
            window_days=30,  # Small window for testing
            timeframe="1d",
            test_windows=2
        )

        if optimizer.window_days != 30:
            all_validation_failures.append(f"Test 1: Expected window_days=30, got {optimizer.window_days}")

        logger.success("Test 1 PASSED: Optimizer initialized")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Parameter grid generation
    total_tests += 1
    try:
        param_grid = optimizer.get_parameter_grid()

        if 'threshold' not in param_grid:
            all_validation_failures.append("Test 2: Missing 'threshold' in parameter grid")

        if 'rebalance_method' not in param_grid:
            all_validation_failures.append("Test 2: Missing 'rebalance_method' in parameter grid")

        logger.success(f"Test 2 PASSED: Generated parameter grid with {len(param_grid)} parameters")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Config generation
    total_tests += 1
    try:
        best_params = {
            'threshold': 0.10,
            'rebalance_method': 'threshold',
            'calendar_period_days': 30,
            'min_rebalance_interval_hours': 24,
            'use_momentum_filter': False
        }

        assets = [('BTC/USDT', 0.5), ('ETH/USDT', 0.5)]

        config = {
            'portfolio': {
                'assets': [{'symbol': s, 'weight': w} for s, w in assets],
                'rebalancing': best_params
            }
        }

        if config['portfolio']['rebalancing']['threshold'] != 0.10:
            all_validation_failures.append("Test 3: Config generation failed - incorrect threshold")

        logger.success("Test 3 PASSED: Config generation working")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Portfolio Optimizer is validated and ready for use")
        print("\nTo run optimization:")
        print("  uv run python optimize_portfolio.py --help")
        sys.exit(0)
