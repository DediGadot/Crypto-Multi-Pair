#!/usr/bin/env python3
"""
Portfolio Backtest Runner - Multi-Asset Portfolio Backtesting

This script runs portfolio-level backtesting across multiple cryptocurrency pairs
using configuration from a YAML file. Supports threshold-based rebalancing strategy
that has been shown to outperform buy-and-hold by 77% in research.

Usage:
    python run_portfolio_backtest.py
    python run_portfolio_backtest.py --config my_config.yaml
    python run_portfolio_backtest.py --config config.yaml --output results_portfolio

Author: Crypto Trader Team
Created: 2025-10-12
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse

# Add src directory to Python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from loguru import logger
import pandas as pd
import numpy as np
import yaml

# Import components
from crypto_trader.core.config import BacktestConfig
from crypto_trader.core.types import Timeframe
from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.strategies import get_registry


class PortfolioBacktestRunner:
    """
    Multi-asset portfolio backtesting runner.

    Handles fetching data for multiple assets, running portfolio rebalancing
    strategy, and generating comprehensive performance reports.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the portfolio backtest runner.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.output_dir = Path(self.config['output']['directory'])

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Configure logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"portfolio_{timestamp}.log"
        logger.add(log_file, level=self.config['logging']['level'])

        # Initialize components
        self.fetcher = BinanceDataFetcher()
        self.portfolio_data: Dict[str, pd.DataFrame] = {}

    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from: {self.config_path}")
        logger.info(f"Run mode: {config['run']['mode']}")
        logger.info(f"Run name: {config['run']['name']}")

        return config

    def _timeframe_to_enum(self, timeframe_str: str) -> Timeframe:
        """
        Convert string timeframe to Timeframe enum.

        Args:
            timeframe_str: Timeframe string (e.g., '1h')

        Returns:
            Timeframe enum value
        """
        mapping = {
            "1m": Timeframe.MINUTE_1,
            "5m": Timeframe.MINUTE_5,
            "15m": Timeframe.MINUTE_15,
            "1h": Timeframe.HOUR_1,
            "4h": Timeframe.HOUR_4,
            "1d": Timeframe.DAY_1,
            "1w": Timeframe.WEEK_1,
        }
        return mapping.get(timeframe_str, Timeframe.HOUR_1)

    def fetch_portfolio_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all assets in portfolio.

        Returns:
            Dictionary mapping symbol to DataFrame with OHLCV data
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Fetching Portfolio Data")
        logger.info("=" * 70)

        timeframe = self.config['data']['timeframe']
        days = self.config['data']['days']

        # Calculate limit based on timeframe
        if timeframe == "1h":
            limit = days * 24
        elif timeframe == "4h":
            limit = days * 6
        elif timeframe == "1d":
            limit = days
        else:
            limit = days * 24  # Default

        portfolio_data = {}

        for asset in self.config['portfolio']['assets']:
            symbol = asset['symbol']
            logger.info(f"\nFetching {symbol}...")

            try:
                data = self.fetcher.get_ohlcv(
                    symbol,
                    timeframe,
                    limit=limit
                )

                if data is None or len(data) == 0:
                    raise ValueError(f"No data fetched for {symbol}")

                portfolio_data[symbol] = data

                logger.success(f"  ✓ Fetched {len(data)} candles")
                logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")
                logger.info(f"  Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                raise

        self.portfolio_data = portfolio_data
        logger.success(f"\nFetched data for {len(portfolio_data)} assets")
        return portfolio_data

    def calculate_portfolio_metrics(self) -> pd.DataFrame:
        """
        Calculate portfolio performance metrics with rebalancing simulation.

        Returns:
            DataFrame with portfolio metrics over time
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Running Portfolio Rebalancing Simulation")
        logger.info("=" * 70)

        # Get common timestamps
        timestamps = None
        for symbol, data in self.portfolio_data.items():
            if timestamps is None:
                timestamps = data.index
            else:
                timestamps = timestamps.intersection(data.index)

        logger.info(f"Common timespan: {len(timestamps)} periods")

        # Initialize portfolio
        initial_capital = self.config['capital']['initial_capital']
        assets = [(a['symbol'], a['weight']) for a in self.config['portfolio']['assets']]
        rebalance_threshold = self.config['portfolio']['rebalancing']['threshold']
        min_interval_hours = self.config['portfolio']['rebalancing']['min_rebalance_interval_hours']

        # Portfolio state tracking
        portfolio_values = {}
        shares = {}

        # Initialize allocations
        for symbol, weight in assets:
            portfolio_values[symbol] = initial_capital * weight
            initial_price = self.portfolio_data[symbol].loc[timestamps[0], 'close']
            shares[symbol] = portfolio_values[symbol] / initial_price

        # Track metrics over time
        equity_curve = []
        rebalance_events = []
        last_rebalance = None

        for idx, timestamp in enumerate(timestamps):
            # Get current prices
            prices = {symbol: self.portfolio_data[symbol].loc[timestamp, 'close']
                      for symbol, _ in assets}

            # Update portfolio values
            portfolio_values = {symbol: shares[symbol] * prices[symbol]
                                for symbol in shares}

            total_value = sum(portfolio_values.values())
            current_weights = {symbol: portfolio_values[symbol] / total_value
                               for symbol in portfolio_values}

            # Check for rebalancing
            needs_rebalance = False
            max_deviation = 0.0

            for symbol, target_weight in assets:
                deviation = abs(current_weights[symbol] - target_weight)
                max_deviation = max(max_deviation, deviation)
                if deviation > rebalance_threshold:
                    needs_rebalance = True

            # Check minimum interval
            if needs_rebalance and last_rebalance is not None:
                hours_since = (timestamp - last_rebalance).total_seconds() / 3600
                if hours_since < min_interval_hours:
                    needs_rebalance = False

            # Execute rebalance if needed
            if needs_rebalance:
                # Rebalance to target weights
                target_values = {symbol: total_value * weight for symbol, weight in assets}
                shares = {symbol: target_values[symbol] / prices[symbol] for symbol in prices}

                rebalance_events.append({
                    'timestamp': timestamp,
                    'total_value': total_value,
                    'max_deviation': max_deviation,
                    'weights_before': dict(current_weights)
                })

                last_rebalance = timestamp

                logger.debug(f"Rebalance at {timestamp}: deviation={max_deviation:.2%}, value=${total_value:,.2f}")

            # Record equity
            equity_curve.append({
                'timestamp': timestamp,
                'total_value': total_value,
                **{f'{symbol}_value': portfolio_values[symbol] for symbol in portfolio_values},
                **{f'{symbol}_weight': current_weights[symbol] for symbol in current_weights}
            })

        equity_df = pd.DataFrame(equity_curve)

        logger.success(f"Simulation complete:")
        logger.info(f"  Rebalance events: {len(rebalance_events)}")
        logger.info(f"  Initial value: ${initial_capital:,.2f}")
        logger.info(f"  Final value: ${equity_df['total_value'].iloc[-1]:,.2f}")
        logger.info(f"  Total return: {((equity_df['total_value'].iloc[-1] / initial_capital) - 1):.2%}")

        return equity_df, rebalance_events

    def calculate_buy_hold_benchmark(self) -> pd.DataFrame:
        """
        Calculate buy-and-hold benchmark for comparison.

        Returns:
            DataFrame with buy-hold portfolio values
        """
        logger.info("\nCalculating buy-and-hold benchmark...")

        # Get common timestamps
        timestamps = None
        for data in self.portfolio_data.values():
            if timestamps is None:
                timestamps = data.index
            else:
                timestamps = timestamps.intersection(data.index)

        initial_capital = self.config['capital']['initial_capital']
        assets = [(a['symbol'], a['weight']) for a in self.config['portfolio']['assets']]

        # Buy and hold - initial allocation only
        shares = {}
        for symbol, weight in assets:
            allocation = initial_capital * weight
            initial_price = self.portfolio_data[symbol].loc[timestamps[0], 'close']
            shares[symbol] = allocation / initial_price

        # Track value over time
        buy_hold_values = []
        for timestamp in timestamps:
            prices = {symbol: self.portfolio_data[symbol].loc[timestamp, 'close']
                      for symbol, _ in assets}

            portfolio_value = sum(shares[symbol] * prices[symbol] for symbol in shares)
            buy_hold_values.append({
                'timestamp': timestamp,
                'buy_hold_value': portfolio_value
            })

        buy_hold_df = pd.DataFrame(buy_hold_values)

        final_value = buy_hold_df['buy_hold_value'].iloc[-1]
        buy_hold_return = (final_value / initial_capital) - 1

        logger.info(f"  Buy & Hold Final: ${final_value:,.2f}")
        logger.info(f"  Buy & Hold Return: {buy_hold_return:.2%}")

        return buy_hold_df

    def generate_reports(self, equity_df: pd.DataFrame, buy_hold_df: pd.DataFrame,
                         rebalance_events: List[Dict]) -> None:
        """
        Generate comprehensive performance reports.

        Args:
            equity_df: Portfolio equity curve
            buy_hold_df: Buy-and-hold benchmark
            rebalance_events: List of rebalance events
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Generating Reports")
        logger.info("=" * 70)

        # 1. Save equity curve
        equity_path = self.data_dir / "portfolio_equity_curve.csv"
        equity_df.to_csv(equity_path, index=False)
        logger.info(f"  ✓ Equity curve: {equity_path}")

        # 2. Save buy-hold benchmark
        buyhold_path = self.data_dir / "buy_hold_benchmark.csv"
        buy_hold_df.to_csv(buyhold_path, index=False)
        logger.info(f"  ✓ Buy-hold benchmark: {buyhold_path}")

        # 3. Save rebalance events
        if rebalance_events:
            rebalance_path = self.data_dir / "rebalance_events.csv"
            pd.DataFrame(rebalance_events).to_csv(rebalance_path, index=False)
            logger.info(f"  ✓ Rebalance events: {rebalance_path}")

        # 4. Generate summary report
        self._generate_summary_report(equity_df, buy_hold_df, rebalance_events)

        logger.success("\nAll reports generated successfully!")

    def _generate_summary_report(self, equity_df: pd.DataFrame, buy_hold_df: pd.DataFrame,
                                  rebalance_events: List[Dict]) -> None:
        """Generate text summary report."""
        summary_path = self.output_dir / "PORTFOLIO_SUMMARY.txt"

        initial_capital = self.config['capital']['initial_capital']
        portfolio_final = equity_df['total_value'].iloc[-1]
        portfolio_return = (portfolio_final / initial_capital) - 1

        buyhold_final = buy_hold_df['buy_hold_value'].iloc[-1]
        buyhold_return = (buyhold_final / initial_capital) - 1

        outperformance = portfolio_return - buyhold_return

        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"PORTFOLIO REBALANCING - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Run Name: {self.config['run']['name']}\n")
            f.write(f"Description: {self.config['run']['description']}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("=" * 70 + "\n")
            f.write("PORTFOLIO CONFIGURATION\n")
            f.write("=" * 70 + "\n\n")

            f.write("Assets:\n")
            for asset in self.config['portfolio']['assets']:
                f.write(f"  {asset['symbol']}: {asset['weight']:.1%}\n")

            f.write(f"\nRebalancing:\n")
            f.write(f"  Threshold: {self.config['portfolio']['rebalancing']['threshold']:.1%}\n")
            f.write(f"  Min Interval: {self.config['portfolio']['rebalancing']['min_rebalance_interval_hours']} hours\n\n")

            f.write("=" * 70 + "\n")
            f.write("PERFORMANCE COMPARISON\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Initial Capital: ${initial_capital:,.2f}\n\n")

            f.write("REBALANCED PORTFOLIO:\n")
            f.write(f"  Final Value: ${portfolio_final:,.2f}\n")
            f.write(f"  Total Return: {portfolio_return:.2%}\n")
            f.write(f"  Rebalance Events: {len(rebalance_events)}\n\n")

            f.write("BUY & HOLD (No Rebalancing):\n")
            f.write(f"  Final Value: ${buyhold_final:,.2f}\n")
            f.write(f"  Total Return: {buyhold_return:.2%}\n\n")

            if outperformance > 0:
                f.write(f"RESULT: Portfolio OUTPERFORMED by {outperformance:.2%}\n")
                f.write(f"Status: ✅ SUCCESS - Rebalancing added value\n\n")
            else:
                f.write(f"RESULT: Portfolio UNDERPERFORMED by {abs(outperformance):.2%}\n")
                f.write(f"Status: ❌ Underperformed buy-and-hold\n\n")

            f.write("=" * 70 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"  - portfolio_equity_curve.csv\n")
            f.write(f"  - buy_hold_benchmark.csv\n")
            if rebalance_events:
                f.write(f"  - rebalance_events.csv\n")

        logger.info(f"\n  ✓ Summary report: {summary_path}")

    def run(self) -> None:
        """Run the complete portfolio backtest."""
        start_time = datetime.now()

        logger.info("\n" + "=" * 70)
        logger.info("PORTFOLIO BACKTESTING - STARTING")
        logger.info("=" * 70)
        logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        try:
            # Step 1: Fetch data for all assets
            self.fetch_portfolio_data()

            # Step 2: Run portfolio simulation
            equity_df, rebalance_events = self.calculate_portfolio_metrics()

            # Step 3: Calculate buy-and-hold benchmark
            buy_hold_df = self.calculate_buy_hold_benchmark()

            # Step 4: Generate reports
            self.generate_reports(equity_df, buy_hold_df, rebalance_events)

            # Completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("\n" + "=" * 70)
            logger.success("PORTFOLIO BACKTEST COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"\nView summary: {self.output_dir / 'PORTFOLIO_SUMMARY.txt'}")
            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"\n❌ BACKTEST FAILED: {e}")
            logger.exception("Full traceback:")
            sys.exit(1)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run multi-asset portfolio backtesting with rebalancing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config.yaml
  python run_portfolio_backtest.py

  # Use custom config file
  python run_portfolio_backtest.py --config my_portfolio.yaml

  # Custom output directory
  python run_portfolio_backtest.py --output results_4asset
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory from config",
    )

    args = parser.parse_args()

    # Create and run portfolio backtest
    runner = PortfolioBacktestRunner(config_path=args.config)

    # Override output dir if specified
    if args.output:
        runner.output_dir = Path(args.output)
        runner.output_dir.mkdir(exist_ok=True)
        runner.reports_dir = runner.output_dir / "reports"
        runner.reports_dir.mkdir(exist_ok=True)
        runner.data_dir = runner.output_dir / "data"
        runner.data_dir.mkdir(exist_ok=True)

    runner.run()


if __name__ == "__main__":
    main()
