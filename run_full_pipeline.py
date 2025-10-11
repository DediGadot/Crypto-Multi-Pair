#!/usr/bin/env python3
"""
Full Pipeline Runner - Comprehensive Strategy Backtesting

This script runs the complete trading pipeline for a single symbol pair:
1. Fetches all available historical data from Binance
2. Runs all 5 strategies with optimized parameters
3. Generates comprehensive reports and comparisons
4. Saves results to multiple formats (HTML, JSON, CSV)

Usage:
    python run_full_pipeline.py BTC/USDT
    python run_full_pipeline.py ETH/USDT --timeframe 1h --days 365
    python run_full_pipeline.py BTC/USDT --capital 10000 --output-dir results

Author: Crypto Trader Team
Created: 2025-10-11
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add src directory to Python path (allows running without installing package)
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from loguru import logger
import pandas as pd

# Import all components
from crypto_trader.core.config import BacktestConfig, RiskConfig
from crypto_trader.core.types import BacktestResult, Timeframe
from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.data.storage import OHLCVStorage
from crypto_trader.strategies import get_registry
from crypto_trader.backtesting.engine import BacktestEngine
from crypto_trader.analysis.comparison import StrategyComparison
from crypto_trader.analysis.reporting import ReportGenerator
from crypto_trader.risk.manager import RiskManager


class FullPipelineRunner:
    """
    Comprehensive pipeline runner for crypto trading strategies.

    This class orchestrates the entire backtesting workflow:
    - Data fetching and validation
    - Multi-strategy backtesting
    - Performance analysis
    - Report generation
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 365,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        output_dir: str = "results",
        max_position_risk: float = 0.02,
        max_portfolio_risk: float = 0.10,
    ):
        """
        Initialize the pipeline runner.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (default: 1h)
            days: Number of days of historical data (default: 365)
            initial_capital: Starting capital for backtests (default: 10000)
            commission: Trading commission rate (default: 0.1%)
            slippage: Slippage rate (default: 0.05%)
            output_dir: Directory for saving results (default: results)
            max_position_risk: Max risk per position (default: 2%)
            max_portfolio_risk: Max total portfolio risk (default: 10%)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.days = days
        self.initial_capital = initial_capital
        self.output_dir = Path(output_dir)

        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Configure logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"pipeline_{timestamp}.log"
        logger.add(log_file, level="DEBUG")

        # Initialize components
        self.fetcher = BinanceDataFetcher()
        self.storage = OHLCVStorage()
        self.backtest_config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
        )
        self.risk_config = RiskConfig(
            max_position_risk=max_position_risk,
            max_portfolio_risk=max_portfolio_risk,
        )
        self.engine = BacktestEngine()
        self.comparison = StrategyComparison()
        self.reporter = ReportGenerator()

        # Strategy configurations with optimized parameters
        self.strategy_configs = {
            "SMA_Crossover": {
                "fast_period": 50,
                "slow_period": 200,
            },
            "RSI_MeanReversion": {
                "rsi_period": 14,
                "oversold": 30,
                "overbought": 70,
            },
            "MACD_Momentum": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
            },
            "BollingerBreakout": {
                "period": 20,
                "std_dev": 2.0,
            },
            "TripleEMA": {
                "fast_period": 8,
                "medium_period": 21,
                "slow_period": 55,
            },
        }

        self.results: List[BacktestResult] = []
        self.price_data: Optional[pd.DataFrame] = None

    def _timeframe_to_enum(self) -> Timeframe:
        """
        Convert string timeframe to Timeframe enum.

        Returns:
            Timeframe enum value

        Raises:
            ValueError: If timeframe string is not valid
        """
        timeframe_mapping = {
            "1m": Timeframe.MINUTE_1,
            "5m": Timeframe.MINUTE_5,
            "15m": Timeframe.MINUTE_15,
            "1h": Timeframe.HOUR_1,
            "4h": Timeframe.HOUR_4,
            "1d": Timeframe.DAY_1,
            "1w": Timeframe.WEEK_1,
        }

        if self.timeframe not in timeframe_mapping:
            raise ValueError(f"Invalid timeframe: {self.timeframe}. Must be one of: {list(timeframe_mapping.keys())}")

        return timeframe_mapping[self.timeframe]

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical data from Binance.

        Returns:
            DataFrame with OHLCV data
        """
        logger.info("=" * 70)
        logger.info(f"STEP 1: Fetching Data")
        logger.info("=" * 70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Timeframe: {self.timeframe}")

        if self.days == -1:
            logger.info("Days: ALL AVAILABLE (fetching maximum historical data)")
        else:
            logger.info(f"Days: {self.days}")

        try:
            # Calculate limit based on days
            if self.days == -1:
                # Fetch all available data
                limit = -1  # Special value for fetch_all
            else:
                # Calculate limit based on timeframe
                if self.timeframe == "1h":
                    limit = self.days * 24
                elif self.timeframe == "4h":
                    limit = self.days * 6
                elif self.timeframe == "1d":
                    limit = self.days
                elif self.timeframe == "1m":
                    limit = self.days * 24 * 60
                elif self.timeframe == "5m":
                    limit = self.days * 24 * 12
                elif self.timeframe == "15m":
                    limit = self.days * 24 * 4
                else:
                    limit = self.days * 24  # Default to hourly

            # Fetch data with smart caching
            data = self.fetcher.get_ohlcv(
                self.symbol,
                self.timeframe,
                limit=limit,
            )

            if data is None or len(data) == 0:
                raise ValueError("No data fetched from exchange")

            logger.success(f"Fetched {len(data)} candles")
            logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
            logger.info(f"Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")

            # Save data
            self.storage.save_ohlcv(data, self.symbol, self.timeframe)

            # Export to CSV
            csv_path = self.data_dir / f"{self.symbol.replace('/', '_')}_{self.timeframe}.csv"
            data.to_csv(csv_path)
            logger.info(f"Data saved to: {csv_path}")

            return data

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise

    def run_strategies(self, data: pd.DataFrame) -> List[BacktestResult]:
        """
        Run all strategies on the data.

        Args:
            data: OHLCV data

        Returns:
            List of backtest results
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"STEP 2: Running Strategies")
        logger.info("=" * 70)

        # Import strategies to ensure registration
        import crypto_trader.strategies.library  # noqa: F401

        registry = get_registry()
        results = []

        # Prepare data (reset index for strategies)
        data_with_timestamp = data.reset_index()

        for strategy_name, config in self.strategy_configs.items():
            logger.info(f"\n{'─' * 70}")
            logger.info(f"Running: {strategy_name}")
            logger.info(f"Parameters: {config}")
            logger.info(f"{'─' * 70}")

            try:
                # Get strategy class and instantiate
                StrategyClass = registry.get_strategy(strategy_name)
                strategy = StrategyClass()
                strategy.initialize(config)

                # Run backtest
                result = self.engine.run_backtest(
                    strategy=strategy,
                    data=data_with_timestamp,
                    config=self.backtest_config,
                    symbol=self.symbol.replace("/", ""),
                    timeframe=self._timeframe_to_enum(),
                )

                results.append(result)

                # Log summary
                logger.success(f"Completed: {strategy_name}")
                logger.info(f"  Total Return: {result.metrics.total_return:.2%}")
                logger.info(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
                logger.info(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
                logger.info(f"  Win Rate: {result.metrics.win_rate:.2%}")
                logger.info(f"  Total Trades: {result.metrics.total_trades}")

            except Exception as e:
                logger.error(f"Failed to run {strategy_name}: {e}")
                continue

        self.results = results
        logger.success(f"\nCompleted {len(results)} out of {len(self.strategy_configs)} strategies")
        return results

    def generate_reports(self) -> None:
        """Generate comprehensive reports for all strategies."""
        logger.info("\n" + "=" * 70)
        logger.info(f"STEP 3: Generating Reports")
        logger.info("=" * 70)

        if len(self.results) == 0:
            logger.warning("No results to generate reports for")
            return

        try:
            # 1. Individual strategy reports
            logger.info("\nGenerating individual strategy reports...")
            for result in self.results:
                # HTML report
                html_path = self.reports_dir / f"{result.strategy_name}_report.html"
                self.reporter.generate_html_report(result, str(html_path))
                logger.info(f"  ✓ HTML: {html_path}")

                # JSON export
                json_path = self.data_dir / f"{result.strategy_name}_result.json"
                self.reporter.export_to_json(result, str(json_path))
                logger.info(f"  ✓ JSON: {json_path}")

                # CSV export (trades)
                csv_path = self.data_dir / f"{result.strategy_name}_trades.csv"
                self.reporter.export_to_csv(result, str(csv_path))
                logger.info(f"  ✓ CSV: {csv_path}")

            # 2. Comparison report
            logger.info("\nGenerating comparison report...")
            comparison_df = self.comparison.compare_strategies(self.results)

            # Save comparison CSV
            comparison_path = self.data_dir / "strategy_comparison.csv"
            comparison_df.to_csv(comparison_path)
            logger.info(f"  ✓ Comparison CSV: {comparison_path}")

            # 3. Buy and hold benchmark
            if self.price_data is not None and len(self.price_data) > 0:
                initial_price = self.price_data['close'].iloc[0]
                final_price = self.price_data['close'].iloc[-1]
                buy_hold_return = (final_price - initial_price) / initial_price
                buy_hold_final = self.initial_capital * (1 + buy_hold_return)

                logger.info("\nBuy & Hold Benchmark:")
                logger.info(f"  Entry: ${initial_price:,.2f}")
                logger.info(f"  Exit: ${final_price:,.2f}")
                logger.info(f"  Return: {buy_hold_return:.2%}")
                logger.info(f"  Final Capital: ${buy_hold_final:,.2f}")

            # 4. Best performer analysis
            logger.info("\nBest Performers:")
            best_sharpe = self.comparison.best_performer(self.results, "sharpe_ratio")
            best_return = self.comparison.best_performer(self.results, "total_return")
            best_drawdown = self.comparison.best_performer(self.results, "max_drawdown")

            logger.info(f"  Best Sharpe: {best_sharpe.strategy_name} ({best_sharpe.metrics.sharpe_ratio:.2f})")
            logger.info(f"  Best Return: {best_return.strategy_name} ({best_return.metrics.total_return:.2%})")
            logger.info(f"  Best Drawdown: {best_drawdown.strategy_name} ({best_drawdown.metrics.max_drawdown:.2%})")

            # 5. Comparison charts
            logger.info("\nGenerating comparison charts...")
            for metric in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]:
                chart = self.reporter.create_comparison_chart(self.results, metric)
                chart_path = self.reports_dir / f"comparison_{metric}.html"
                chart.write_html(str(chart_path))
                logger.info(f"  ✓ {metric}: {chart_path}")

            # 6. Generate summary report
            self._generate_summary_report(comparison_df)

            logger.success("\nAll reports generated successfully!")

        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            raise

    def _generate_summary_report(self, comparison_df: pd.DataFrame) -> None:
        """Generate a summary text report."""
        summary_path = self.output_dir / "SUMMARY.txt"

        # Calculate buy and hold return
        buy_hold_return = 0.0
        buy_hold_final = self.initial_capital
        if self.price_data is not None and len(self.price_data) > 0:
            initial_price = self.price_data['close'].iloc[0]
            final_price = self.price_data['close'].iloc[-1]
            buy_hold_return = (final_price - initial_price) / initial_price
            buy_hold_final = self.initial_capital * (1 + buy_hold_return)

        with open(summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"CRYPTO TRADING PIPELINE - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Timeframe: {self.timeframe}\n")
            f.write(f"Initial Capital: ${self.initial_capital:,.2f}\n")
            f.write(f"Strategies Tested: {len(self.results)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Buy and Hold Benchmark
            f.write("=" * 70 + "\n")
            f.write("BUY & HOLD BENCHMARK\n")
            f.write("=" * 70 + "\n\n")

            if self.price_data is not None and len(self.price_data) > 0:
                f.write(f"Strategy: Buy & Hold\n")
                f.write(f"Entry Price: ${self.price_data['close'].iloc[0]:,.2f}\n")
                f.write(f"Exit Price: ${self.price_data['close'].iloc[-1]:,.2f}\n")
                f.write(f"Total Return: {buy_hold_return:.2%}\n")
                f.write(f"Final Capital: ${buy_hold_final:,.2f}\n")
                f.write(f"Date Range: {self.price_data.index[0]} to {self.price_data.index[-1]}\n\n")
            else:
                f.write("No price data available for buy & hold calculation\n\n")

            f.write("=" * 70 + "\n")
            f.write("STRATEGY COMPARISON\n")
            f.write("=" * 70 + "\n\n")

            # Format comparison table
            f.write(comparison_df.to_string())
            f.write("\n\n")

            f.write("=" * 70 + "\n")
            f.write("BEST PERFORMERS\n")
            f.write("=" * 70 + "\n\n")

            best_sharpe = self.comparison.best_performer(self.results, "sharpe_ratio")
            best_return = self.comparison.best_performer(self.results, "total_return")
            best_drawdown = self.comparison.best_performer(self.results, "max_drawdown")

            f.write(f"Best Sharpe Ratio:\n")
            f.write(f"  Strategy: {best_sharpe.strategy_name}\n")
            f.write(f"  Sharpe: {best_sharpe.metrics.sharpe_ratio:.2f}\n")
            f.write(f"  Return: {best_sharpe.metrics.total_return:.2%}\n\n")

            f.write(f"Best Total Return:\n")
            f.write(f"  Strategy: {best_return.strategy_name}\n")
            f.write(f"  Return: {best_return.metrics.total_return:.2%}\n")
            f.write(f"  Sharpe: {best_return.metrics.sharpe_ratio:.2f}\n")
            # Compare to buy and hold
            if best_return.metrics.total_return > buy_hold_return:
                outperformance = best_return.metrics.total_return - buy_hold_return
                f.write(f"  vs Buy & Hold: +{outperformance:.2%} (OUTPERFORMED)\n\n")
            else:
                underperformance = buy_hold_return - best_return.metrics.total_return
                f.write(f"  vs Buy & Hold: -{underperformance:.2%} (UNDERPERFORMED)\n\n")

            f.write(f"Best Max Drawdown (lowest):\n")
            f.write(f"  Strategy: {best_drawdown.strategy_name}\n")
            f.write(f"  Drawdown: {best_drawdown.metrics.max_drawdown:.2%}\n")
            f.write(f"  Return: {best_drawdown.metrics.total_return:.2%}\n\n")

            f.write("=" * 70 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Reports Directory: {self.reports_dir}\n")
            f.write(f"Data Directory: {self.data_dir}\n\n")

            f.write("Individual Reports:\n")
            for result in self.results:
                f.write(f"  - {result.strategy_name}_report.html\n")

            f.write("\nComparison Files:\n")
            f.write(f"  - strategy_comparison.csv\n")
            f.write(f"  - comparison_*.html (charts)\n\n")

        logger.info(f"Summary report: {summary_path}")

    def run(self) -> None:
        """Run the complete pipeline."""
        start_time = datetime.now()

        logger.info("\n" + "=" * 70)
        logger.info("CRYPTO TRADING PIPELINE - FULL RUN")
        logger.info("=" * 70)
        logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        try:
            # Step 1: Fetch data
            data = self.fetch_data()
            self.price_data = data  # Store for buy and hold calculation

            # Step 2: Run all strategies
            results = self.run_strategies(data)

            # Step 3: Generate reports
            self.generate_reports()

            # Completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("\n" + "=" * 70)
            logger.success("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"\nView summary: {self.output_dir / 'SUMMARY.txt'}")
            logger.info(f"View reports: {self.reports_dir}")
            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"\n❌ PIPELINE FAILED: {e}")
            logger.exception("Full traceback:")
            sys.exit(1)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive backtesting pipeline for a crypto trading pair",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (BTC/USDT, 1h, 365 days, $10k capital)
  python run_full_pipeline.py BTC/USDT

  # Custom timeframe and period
  python run_full_pipeline.py ETH/USDT --timeframe 4h --days 180

  # Custom capital and output directory
  python run_full_pipeline.py BTC/USDT --capital 50000 --output-dir my_results

  # Full customization
  python run_full_pipeline.py SOL/USDT --timeframe 1d --days 730 \\
      --capital 100000 --commission 0.001 --output-dir sol_backtest
        """
    )

    parser.add_argument(
        "symbol",
        type=str,
        help="Trading pair symbol (e.g., BTC/USDT, ETH/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (default: 1h). Options: 1m, 5m, 15m, 1h, 4h, 1d",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data (default: 365, use -1 for ALL available data)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital for backtests (default: 10000)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Trading commission rate (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="Slippage rate (default: 0.0005 = 0.05%%)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for saving results (default: results)",
    )
    parser.add_argument(
        "--max-position-risk",
        type=float,
        default=0.02,
        help="Max risk per position (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--max-portfolio-risk",
        type=float,
        default=0.10,
        help="Max total portfolio risk (default: 0.10 = 10%%)",
    )

    args = parser.parse_args()

    # Validate symbol format
    if "/" not in args.symbol:
        logger.error(f"Invalid symbol format: {args.symbol}")
        logger.info("Symbol should be in format: BASE/QUOTE (e.g., BTC/USDT)")
        sys.exit(1)

    # Create and run pipeline
    runner = FullPipelineRunner(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
        output_dir=args.output_dir,
        max_position_risk=args.max_position_risk,
        max_portfolio_risk=args.max_portfolio_risk,
    )

    runner.run()


if __name__ == "__main__":
    main()
