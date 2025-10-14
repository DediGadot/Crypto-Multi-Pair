#!/usr/bin/env python3
"""
Full Pipeline Runner - Comprehensive Strategy Backtesting

This script runs the complete trading pipeline in two modes:

SINGLE-PAIR MODE (default):
1. Fetches all available historical data from Binance for one pair
2. Runs all 5 strategies with optimized parameters
3. Generates comprehensive reports and comparisons
4. Saves results to multiple formats (HTML, JSON, CSV)

PORTFOLIO MODE (--portfolio or --config):
1. Loads portfolio configuration from YAML file
2. Fetches data for multiple crypto assets
3. Runs portfolio rebalancing strategy
4. Generates portfolio performance reports and comparisons

Usage:
    # Single-pair mode
    python run_full_pipeline.py BTC/USDT
    python run_full_pipeline.py ETH/USDT --timeframe 1h --days 365
    python run_full_pipeline.py BTC/USDT --capital 10000 --output-dir results

    # Portfolio mode
    python run_full_pipeline.py --portfolio --config config_improved_10pct.yaml
    python run_full_pipeline.py --portfolio --config my_portfolio.yaml

Author: Crypto Trader Team
Created: 2025-10-11
Updated: 2025-10-12 (Added portfolio mode)
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
import numpy as np
import yaml

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
            trading_fee_percent=commission,
            slippage_percent=slippage,
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
            "StatisticalArbitrage": {
                "pair1_symbol": "BTC/USDT",
                "pair2_symbol": "ETH/USDT",
                "lookback_period": 180,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "z_score_window": 90,
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

    def generate_enhanced_report(self) -> None:
        """
        Generate an enhanced comparison report with:
        1. Strategy comparison table vs buy-and-hold with metric explanations
        2. Deep-dive analysis of the best performing strategy
        """
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING ENHANCED COMPARISON REPORT")
        logger.info("=" * 70)

        if len(self.results) == 0:
            logger.warning("No results available for enhanced report")
            return

        report_path = self.output_dir / "ENHANCED_REPORT.txt"

        # Calculate buy-and-hold benchmark
        buy_hold_return = 0.0
        buy_hold_final = self.initial_capital
        if self.price_data is not None and len(self.price_data) > 0:
            initial_price = self.price_data['close'].iloc[0]
            final_price = self.price_data['close'].iloc[-1]
            buy_hold_return = (final_price - initial_price) / initial_price
            buy_hold_final = self.initial_capital * (1 + buy_hold_return)

        # Find best performer by total return
        best_result = self.comparison.best_performer(self.results, "total_return")

        with open(report_path, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write("ENHANCED STRATEGY COMPARISON REPORT\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Timeframe: {self.timeframe}\n")
            f.write(f"Initial Capital: ${self.initial_capital:,.2f}\n")
            f.write(f"Period: {self.price_data.index[0]} to {self.price_data.index[-1]}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # ============================================================
            # PART 1: METRIC EXPLANATIONS
            # ============================================================
            f.write("=" * 90 + "\n")
            f.write("FINANCIAL METRICS EXPLAINED\n")
            f.write("=" * 90 + "\n\n")

            f.write("Total Return: The percentage gain or loss on initial capital over the entire period\n")
            f.write("Final Capital: The ending portfolio value after all trades\n")
            f.write("Sharpe Ratio: Risk-adjusted return (higher is better, >1 is good, >2 is excellent)\n")
            f.write("Max Drawdown: Largest peak-to-trough decline (lower absolute value is better)\n")
            f.write("Win Rate: Percentage of profitable trades (higher is better, >50% is good)\n")
            f.write("Total Trades: Number of complete buy/sell cycles executed\n")
            f.write("Profit Factor: Gross profit divided by gross loss (>1 is profitable, >2 is excellent)\n")
            f.write("Avg Win: Average profit per winning trade\n")
            f.write("Avg Loss: Average loss per losing trade\n")
            f.write("Win/Loss Ratio: Average win divided by average loss (>1 means wins are larger than losses)\n\n")

            # ============================================================
            # PART 2: STRATEGY COMPARISON TABLE
            # ============================================================
            f.write("=" * 90 + "\n")
            f.write("STRATEGY PERFORMANCE COMPARISON (vs Buy & Hold)\n")
            f.write("=" * 90 + "\n\n")

            # Header
            f.write(f"{'Strategy':<25} {'Return':<12} {'vs B&H':<12} {'Sharpe':<10} {'MaxDD':<12} {'WinRate':<10} {'Trades':<8}\n")
            f.write("-" * 90 + "\n")

            # Buy & Hold row
            f.write(f"{'Buy & Hold (Benchmark)':<25} ")
            f.write(f"{buy_hold_return:>10.2%}  ")
            f.write(f"{'--':>10}  ")
            f.write(f"{'N/A':>8}  ")
            f.write(f"{'N/A':>10}  ")
            f.write(f"{'N/A':>8}  ")
            f.write(f"{'0':>6}\n")

            # Strategy rows sorted by return
            sorted_results = sorted(self.results, key=lambda r: r.metrics.total_return, reverse=True)
            for result in sorted_results:
                vs_buyhold = result.metrics.total_return - buy_hold_return
                vs_indicator = "+" if vs_buyhold > 0 else ""

                f.write(f"{result.strategy_name:<25} ")
                f.write(f"{result.metrics.total_return:>10.2%}  ")
                f.write(f"{vs_indicator}{vs_buyhold:>9.2%}  ")
                f.write(f"{result.metrics.sharpe_ratio:>8.2f}  ")
                f.write(f"{result.metrics.max_drawdown:>10.2%}  ")
                f.write(f"{result.metrics.win_rate:>8.2%}  ")
                f.write(f"{result.metrics.total_trades:>6}\n")

            f.write("\n")

            # Summary insights
            f.write("KEY INSIGHTS:\n")
            f.write(f"- Best Performing Strategy: {best_result.strategy_name}\n")
            best_vs_buyhold = best_result.metrics.total_return - buy_hold_return
            if best_vs_buyhold > 0:
                f.write(f"- Outperformed Buy & Hold by: {best_vs_buyhold:.2%}\n")
            else:
                f.write(f"- Underperformed Buy & Hold by: {abs(best_vs_buyhold):.2%}\n")

            profitable_strats = [r for r in self.results if r.metrics.total_return > buy_hold_return]
            f.write(f"- Strategies Beating Buy & Hold: {len(profitable_strats)} of {len(self.results)}\n")

            high_sharpe = [r for r in self.results if r.metrics.sharpe_ratio > 1.0]
            f.write(f"- Strategies with Good Risk-Adjusted Returns (Sharpe>1): {len(high_sharpe)}\n\n")

            # ============================================================
            # PART 3: DETAILED METRICS TABLE
            # ============================================================
            f.write("=" * 90 + "\n")
            f.write("DETAILED PERFORMANCE METRICS\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"{'Strategy':<25} {'ProfitFactor':<14} {'AvgWin':<12} {'AvgLoss':<12} {'Win/Loss':<10}\n")
            f.write("-" * 90 + "\n")

            for result in sorted_results:
                f.write(f"{result.strategy_name:<25} ")
                f.write(f"{result.metrics.profit_factor:>12.2f}  ")
                f.write(f"${result.metrics.avg_win:>9.2f}  ")
                f.write(f"${abs(result.metrics.avg_loss):>9.2f}  ")

                if result.metrics.avg_loss != 0:
                    win_loss_ratio = abs(result.metrics.avg_win / result.metrics.avg_loss)
                    f.write(f"{win_loss_ratio:>8.2f}\n")
                else:
                    f.write(f"{'N/A':>8}\n")

            f.write("\n")

            # ============================================================
            # PART 4: BEST STRATEGY DEEP DIVE
            # ============================================================
            f.write("=" * 90 + "\n")
            f.write(f"DEEP DIVE: {best_result.strategy_name} (Best Performer)\n")
            f.write("=" * 90 + "\n\n")

            f.write("STRATEGY OVERVIEW:\n")
            f.write(f"- Total Return: {best_result.metrics.total_return:.2%}\n")
            f.write(f"- Final Capital: ${best_result.metrics.total_return * self.initial_capital + self.initial_capital:,.2f}\n")
            f.write(f"- Total Trades: {best_result.metrics.total_trades}\n")
            f.write(f"- Winning Trades: {int(best_result.metrics.total_trades * best_result.metrics.win_rate)}\n")
            f.write(f"- Losing Trades: {int(best_result.metrics.total_trades * (1 - best_result.metrics.win_rate))}\n")
            f.write(f"- Win Rate: {best_result.metrics.win_rate:.2%}\n")
            f.write(f"- Profit Factor: {best_result.metrics.profit_factor:.2f}\n")
            f.write(f"- Sharpe Ratio: {best_result.metrics.sharpe_ratio:.2f}\n")
            f.write(f"- Max Drawdown: {best_result.metrics.max_drawdown:.2%}\n\n")

            f.write("TRADE-BY-TRADE ANALYSIS:\n")
            f.write("-" * 90 + "\n\n")

            if best_result.trades and len(best_result.trades) > 0:
                # Analyze all trades
                winning_trades = [t for t in best_result.trades if t.pnl > 0]
                losing_trades = [t for t in best_result.trades if t.pnl <= 0]

                f.write(f"Total Trades Executed: {len(best_result.trades)}\n")
                f.write(f"Winning Trades: {len(winning_trades)}\n")
                f.write(f"Losing Trades: {len(losing_trades)}\n\n")

                # Show top 5 winning trades
                f.write("TOP 5 WINNING TRADES:\n")
                f.write(f"{'#':<4} {'Entry Date':<20} {'Exit Date':<20} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'Return':<10}\n")
                f.write("-" * 90 + "\n")

                top_winners = sorted(winning_trades, key=lambda t: t.pnl, reverse=True)[:5]
                for idx, trade in enumerate(top_winners, 1):
                    trade_return = (trade.exit_price - trade.entry_price) / trade.entry_price
                    f.write(f"{idx:<4} ")
                    f.write(f"{trade.entry_time.strftime('%Y-%m-%d %H:%M'):<20} ")
                    f.write(f"{trade.exit_time.strftime('%Y-%m-%d %H:%M'):<20} ")
                    f.write(f"${trade.entry_price:>8.2f}  ")
                    f.write(f"${trade.exit_price:>8.2f}  ")
                    f.write(f"${trade.pnl:>10.2f}  ")
                    f.write(f"{trade_return:>8.2%}\n")

                f.write("\n")

                # Show top 5 losing trades
                if losing_trades:
                    f.write("TOP 5 LOSING TRADES (To Learn From):\n")
                    f.write(f"{'#':<4} {'Entry Date':<20} {'Exit Date':<20} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'Return':<10}\n")
                    f.write("-" * 90 + "\n")

                    top_losers = sorted(losing_trades, key=lambda t: t.pnl)[:5]
                    for idx, trade in enumerate(top_losers, 1):
                        trade_return = (trade.exit_price - trade.entry_price) / trade.entry_price
                        f.write(f"{idx:<4} ")
                        f.write(f"{trade.entry_time.strftime('%Y-%m-%d %H:%M'):<20} ")
                        f.write(f"{trade.exit_time.strftime('%Y-%m-%d %H:%M'):<20} ")
                        f.write(f"${trade.entry_price:>8.2f}  ")
                        f.write(f"${trade.exit_price:>8.2f}  ")
                        f.write(f"${trade.pnl:>10.2f}  ")
                        f.write(f"{trade_return:>8.2%}\n")

                    f.write("\n")

                # Trade statistics
                f.write("TRADE STATISTICS:\n")
                f.write(f"- Average Trade Duration: {sum((t.exit_time - t.entry_time).total_seconds() for t in best_result.trades) / len(best_result.trades) / 3600:.1f} hours\n")
                f.write(f"- Longest Trade: {max((t.exit_time - t.entry_time).total_seconds() for t in best_result.trades) / 3600:.1f} hours\n")
                f.write(f"- Shortest Trade: {min((t.exit_time - t.entry_time).total_seconds() for t in best_result.trades) / 3600:.1f} hours\n")
                f.write(f"- Average Win: ${best_result.metrics.avg_win:.2f} ({(best_result.metrics.avg_win / self.initial_capital * 100):.2f}% of capital)\n")
                if best_result.metrics.avg_loss != 0:
                    f.write(f"- Average Loss: ${abs(best_result.metrics.avg_loss):.2f} ({abs(best_result.metrics.avg_loss / self.initial_capital * 100):.2f}% of capital)\n")
                f.write("\n")

                # Strategy-specific insights
                f.write(f"STRATEGY-SPECIFIC INSIGHTS FOR {best_result.strategy_name}:\n")

                if "SMA" in best_result.strategy_name or "Crossover" in best_result.strategy_name:
                    f.write("- This is a trend-following strategy using moving average crossovers\n")
                    f.write("- Trades are triggered when fast MA crosses above (buy) or below (sell) slow MA\n")
                    f.write("- Works best in trending markets, struggles in choppy/sideways markets\n")
                    f.write("- Consider using during strong directional moves\n")

                elif "RSI" in best_result.strategy_name:
                    f.write("- This is a mean-reversion strategy using Relative Strength Index\n")
                    f.write("- Buys when RSI indicates oversold conditions (typically <30)\n")
                    f.write("- Sells when RSI indicates overbought conditions (typically >70)\n")
                    f.write("- Works best in range-bound markets with clear support/resistance\n")
                    f.write("- Can generate false signals during strong trends\n")

                elif "MACD" in best_result.strategy_name:
                    f.write("- This is a momentum strategy using MACD indicator\n")
                    f.write("- Trades on MACD line crossing signal line (bullish/bearish crossovers)\n")
                    f.write("- Also considers histogram for momentum strength\n")
                    f.write("- Effective for catching medium to long-term trends\n")
                    f.write("- May lag at trend reversals due to indicator smoothing\n")

                elif "Bollinger" in best_result.strategy_name:
                    f.write("- This is a volatility breakout strategy using Bollinger Bands\n")
                    f.write("- Buys when price breaks above upper band (continuation)\n")
                    f.write("- Can also be used mean-reversion style (buy at lower band)\n")
                    f.write("- Adapts to volatility - bands widen in volatile markets\n")
                    f.write("- Watch for band squeezes which often precede breakouts\n")

                elif "EMA" in best_result.strategy_name or "Triple" in best_result.strategy_name:
                    f.write("- This is an advanced trend-following strategy using multiple EMAs\n")
                    f.write("- Uses fast, medium, and slow EMAs for trade confirmation\n")
                    f.write("- Requires alignment of all EMAs for strongest signals\n")
                    f.write("- More reliable than simple crossovers but may have fewer trades\n")
                    f.write("- Best used in sustained trending environments\n")

                f.write("\n")

                # Risk analysis
                f.write("RISK ANALYSIS:\n")
                drawdowns = []
                # Extract just the values from (timestamp, value) tuples
                equity_values = [value for _, value in best_result.equity_curve] if best_result.equity_curve else [self.initial_capital]
                peak = equity_values[0] if equity_values else self.initial_capital
                for value in equity_values:
                    if value > peak:
                        peak = value
                    drawdown = (value - peak) / peak
                    if drawdown < 0:
                        drawdowns.append(abs(drawdown))

                if drawdowns:
                    f.write(f"- Maximum Drawdown: {best_result.metrics.max_drawdown:.2%}\n")
                    f.write(f"- Average Drawdown: {sum(drawdowns) / len(drawdowns):.2%}\n")
                    f.write(f"- Number of Drawdown Periods: {len(drawdowns)}\n")
                f.write(f"- Risk-Adjusted Return (Sharpe): {best_result.metrics.sharpe_ratio:.2f}\n")
                f.write(f"- Profit Factor (Reward/Risk): {best_result.metrics.profit_factor:.2f}\n\n")

            else:
                f.write("No trade data available for detailed analysis.\n\n")

            # ============================================================
            # PART 5: RECOMMENDATIONS
            # ============================================================
            f.write("=" * 90 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 90 + "\n\n")

            if best_vs_buyhold > 0:
                f.write(f"✓ The {best_result.strategy_name} strategy outperformed buy-and-hold\n")
                f.write(f"  Consider using this strategy for active trading\n\n")
            else:
                f.write(f"✗ The best strategy still underperformed buy-and-hold\n")
                f.write(f"  Consider:\n")
                f.write(f"  1. Adjusting strategy parameters\n")
                f.write(f"  2. Testing on different timeframes\n")
                f.write(f"  3. Combining strategies for better risk-adjusted returns\n")
                f.write(f"  4. Sticking with buy-and-hold for this asset\n\n")

            if best_result.metrics.sharpe_ratio > 2.0:
                f.write(f"✓ Excellent risk-adjusted returns (Sharpe > 2.0)\n\n")
            elif best_result.metrics.sharpe_ratio > 1.0:
                f.write(f"✓ Good risk-adjusted returns (Sharpe > 1.0)\n\n")
            else:
                f.write(f"⚠ Risk-adjusted returns could be improved (Sharpe < 1.0)\n")
                f.write(f"  Consider reducing position sizes or tightening stop losses\n\n")

            if abs(best_result.metrics.max_drawdown) < 0.20:
                f.write(f"✓ Well-controlled drawdown (<20%)\n\n")
            else:
                f.write(f"⚠ Significant drawdown risk (>{abs(best_result.metrics.max_drawdown):.0%})\n")
                f.write(f"  Consider implementing stricter risk management\n\n")

            f.write("=" * 90 + "\n")
            f.write("END OF ENHANCED REPORT\n")
            f.write("=" * 90 + "\n")

        logger.success(f"\nEnhanced report generated: {report_path}")
        logger.info(f"View the report for detailed strategy analysis and insights")

    # ========================================================================
    # PORTFOLIO MODE METHODS
    # ========================================================================

    def run_portfolio_mode(self, config_path: str, generate_enhanced: bool = False) -> None:
        """
        Run in portfolio rebalancing mode.

        Args:
            config_path: Path to YAML configuration file
            generate_enhanced: Whether to generate enhanced report
        """
        logger.info("\n" + "=" * 70)
        logger.info("PORTFOLIO REBALANCING MODE")
        logger.info("=" * 70)

        # Load portfolio configuration
        config = self._load_portfolio_config(config_path)

        # Update output directory from config
        self.output_dir = Path(config['output']['directory'])
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Fetch multi-asset data
        portfolio_data = self._fetch_portfolio_data(config)

        # Run portfolio simulation
        equity_df, rebalance_events = self._calculate_portfolio_metrics(config, portfolio_data)

        # Calculate buy-and-hold benchmark
        buy_hold_df = self._calculate_buy_hold_benchmark(config, portfolio_data)

        # Generate portfolio reports
        self._generate_portfolio_reports(config, equity_df, buy_hold_df, rebalance_events)

        # Generate enhanced report if requested
        if generate_enhanced:
            self.generate_enhanced_portfolio_report(config, equity_df, buy_hold_df, rebalance_events, portfolio_data)

        logger.success("\nPortfolio backtest completed successfully!")

    def generate_enhanced_portfolio_report(self, config: Dict, equity_df: pd.DataFrame,
                                           buy_hold_df: pd.DataFrame, rebalance_events: List[Dict],
                                           portfolio_data: Dict[str, pd.DataFrame]) -> None:
        """
        Generate enhanced portfolio analysis report with:
        1. Portfolio metrics explained
        2. Individual asset performance breakdown
        3. Detailed rebalancing event analysis
        4. Recommendations
        """
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING ENHANCED PORTFOLIO REPORT")
        logger.info("=" * 70)

        report_path = self.output_dir / "ENHANCED_PORTFOLIO_REPORT.txt"

        initial_capital = config['capital']['initial_capital']
        portfolio_final = equity_df['total_value'].iloc[-1]
        portfolio_return = (portfolio_final / initial_capital) - 1

        buyhold_final = buy_hold_df['buy_hold_value'].iloc[-1]
        buyhold_return = (buyhold_final / initial_capital) - 1

        outperformance = portfolio_return - buyhold_return

        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("ENHANCED PORTFOLIO REBALANCING REPORT\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Portfolio Name: {config['run']['name']}\n")
            f.write(f"Description: {config['run']['description']}\n")
            f.write(f"Initial Capital: ${initial_capital:,.2f}\n")
            f.write(f"Period: {equity_df['timestamp'].iloc[0]} to {equity_df['timestamp'].iloc[-1]}\n")
            f.write(f"Duration: {(equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days} days\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # ============================================================
            # PART 1: PORTFOLIO METRICS EXPLAINED
            # ============================================================
            f.write("=" * 100 + "\n")
            f.write("PORTFOLIO METRICS EXPLAINED\n")
            f.write("=" * 100 + "\n\n")

            f.write("Portfolio Return: Total percentage gain/loss from rebalancing strategy\n")
            f.write("Buy-and-Hold Return: Baseline return with no rebalancing (maintain initial allocation)\n")
            f.write("Outperformance: Additional return gained from rebalancing vs buy-and-hold\n")
            f.write("Rebalance Events: Number of times portfolio was rebalanced to target weights\n")
            f.write("Rebalance Threshold: Maximum weight deviation allowed before triggering rebalance\n")
            f.write("Asset Weight: Percentage of portfolio allocated to each cryptocurrency\n")
            f.write("Weight Deviation: How far current weight has drifted from target weight\n\n")

            # ============================================================
            # PART 2: PERFORMANCE SUMMARY
            # ============================================================
            f.write("=" * 100 + "\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"{'Metric':<30} {'Rebalanced Portfolio':<25} {'Buy & Hold':<25} {'Difference':<20}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Final Value':<30} ${portfolio_final:>23,.2f} ${buyhold_final:>23,.2f} ${portfolio_final - buyhold_final:>18,.2f}\n")
            f.write(f"{'Total Return':<30} {portfolio_return:>23.2%} {buyhold_return:>23.2%} {outperformance:>18.2%}\n")
            f.write(f"{'Rebalance Events':<30} {len(rebalance_events):>23} {'0':>23} {len(rebalance_events):>18}\n\n")

            if outperformance > 0:
                f.write(f"✓ RESULT: Rebalancing OUTPERFORMED buy-and-hold by {outperformance:.2%}\n")
                f.write(f"  Value added from rebalancing: ${portfolio_final - buyhold_final:,.2f}\n\n")
            else:
                f.write(f"✗ RESULT: Rebalancing UNDERPERFORMED buy-and-hold by {abs(outperformance):.2%}\n")
                f.write(f"  Value lost from rebalancing: ${abs(portfolio_final - buyhold_final):,.2f}\n\n")

            # ============================================================
            # PART 3: INDIVIDUAL ASSET PERFORMANCE
            # ============================================================
            f.write("=" * 100 + "\n")
            f.write("INDIVIDUAL ASSET PERFORMANCE\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"{'Asset':<15} {'Target Weight':<15} {'Initial Price':<15} {'Final Price':<15} {'Return':<15} {'Contribution':<15}\n")
            f.write("-" * 100 + "\n")

            assets = [(a['symbol'], a['weight']) for a in config['portfolio']['assets']]
            for symbol, target_weight in assets:
                data = portfolio_data[symbol]
                initial_price = data.loc[equity_df['timestamp'].iloc[0], 'close']
                final_price = data.loc[equity_df['timestamp'].iloc[-1], 'close']
                asset_return = (final_price - initial_price) / initial_price
                contribution = target_weight * asset_return * 100  # Rough contribution estimate

                f.write(f"{symbol:<15} {target_weight:<14.1%} ${initial_price:>13,.2f} ${final_price:>13,.2f} {asset_return:>13.2%} {contribution:>13.2f}%\n")

            f.write("\n")

            # Asset return comparison
            f.write("ASSET PERFORMANCE RANKING (Best to Worst):\n")
            asset_returns = []
            for symbol, target_weight in assets:
                data = portfolio_data[symbol]
                initial_price = data.loc[equity_df['timestamp'].iloc[0], 'close']
                final_price = data.loc[equity_df['timestamp'].iloc[-1], 'close']
                asset_return = (final_price - initial_price) / initial_price
                asset_returns.append((symbol, asset_return))

            asset_returns.sort(key=lambda x: x[1], reverse=True)
            for idx, (symbol, ret) in enumerate(asset_returns, 1):
                emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
                f.write(f"{emoji} {idx}. {symbol:<12} {ret:>10.2%}\n")
            f.write("\n")

            # ============================================================
            # PART 4: REBALANCING EVENT ANALYSIS
            # ============================================================
            f.write("=" * 100 + "\n")
            f.write("DETAILED REBALANCING EVENT ANALYSIS\n")
            f.write("=" * 100 + "\n\n")

            if rebalance_events:
                f.write(f"Total Rebalancing Events: {len(rebalance_events)}\n")
                f.write(f"Average Days Between Rebalances: {(equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days / max(len(rebalance_events), 1):.1f}\n\n")

                f.write("REBALANCING EVENTS (Chronological):\n\n")

                for idx, event in enumerate(rebalance_events, 1):
                    f.write(f"Event #{idx} - {event['timestamp'].strftime('%Y-%m-%d %H:%M')}\n")
                    f.write(f"Portfolio Value: ${event['total_value']:,.2f}\n")
                    f.write(f"Max Weight Deviation: {event['max_deviation']:.2%}\n")
                    f.write(f"Trigger: {'Threshold exceeded' if event['max_deviation'] > config['portfolio']['rebalancing']['threshold'] else 'Calendar-based'}\n\n")

                    f.write("Weight Changes:\n")
                    for symbol, target_weight in assets:
                        before_weight = event['weights_before'].get(symbol, 0.0)
                        weight_change = target_weight - before_weight
                        action = "BUY" if weight_change > 0 else "SELL" if weight_change < 0 else "HOLD"
                        f.write(f"  {symbol:<12} {before_weight:>7.2%} → {target_weight:>7.2%} ({weight_change:>+7.2%})  [{action}]\n")

                    f.write("\n" + "-" * 100 + "\n\n")
            else:
                f.write("No rebalancing events occurred during this period.\n")
                f.write("Consider lowering the rebalance threshold or using calendar-based rebalancing.\n\n")

            # ============================================================
            # PART 5: REBALANCING STRATEGY ANALYSIS
            # ============================================================
            f.write("=" * 100 + "\n")
            f.write("REBALANCING STRATEGY ANALYSIS\n")
            f.write("=" * 100 + "\n\n")

            f.write("STRATEGY CONFIGURATION:\n")
            rebal_config = config['portfolio']['rebalancing']
            f.write(f"- Method: {rebal_config.get('rebalance_method', 'threshold').upper()}\n")
            f.write(f"- Threshold: {rebal_config['threshold']:.1%}\n")
            f.write(f"- Min Interval: {rebal_config['min_rebalance_interval_hours']} hours\n")

            if rebal_config.get('use_momentum_filter'):
                f.write(f"- Momentum Filter: ENABLED (lookback: {rebal_config.get('momentum_lookback_days', 30)} days)\n")
            else:
                f.write(f"- Momentum Filter: DISABLED\n")
            f.write("\n")

            f.write("HOW REBALANCING WORKS:\n")
            if rebal_config.get('rebalance_method', 'threshold') == 'threshold':
                f.write("- THRESHOLD METHOD: Rebalances when any asset weight deviates from target by more than threshold\n")
                f.write(f"- Your threshold: {rebal_config['threshold']:.1%}\n")
                f.write("- Lower threshold = More frequent rebalancing (more mean reversion)\n")
                f.write("- Higher threshold = Less frequent rebalancing (more trend following)\n")
            elif rebal_config.get('rebalance_method') == 'calendar':
                f.write("- CALENDAR METHOD: Rebalances on fixed schedule regardless of deviation\n")
                f.write(f"- Your schedule: Every {rebal_config.get('calendar_period_days', 30)} days\n")
                f.write("- Pros: Predictable, systematic\n")
                f.write("- Cons: May rebalance unnecessarily or miss opportunities\n")
            elif rebal_config.get('rebalance_method') == 'hybrid':
                f.write("- HYBRID METHOD: Combines threshold and calendar methods\n")
                f.write(f"- Rebalances if: Deviation > {rebal_config['threshold']:.1%} OR {rebal_config.get('calendar_period_days', 30)} days passed\n")
                f.write("- Best of both worlds: Responsive to volatility + systematic timing\n")
            f.write("\n")

            # ============================================================
            # PART 6: RECOMMENDATIONS
            # ============================================================
            f.write("=" * 100 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 100 + "\n\n")

            if outperformance > 0:
                f.write(f"✓ Your rebalancing strategy is working! It added {outperformance:.2%} vs buy-and-hold\n\n")

                if len(rebalance_events) < 3:
                    f.write("⚠ Low Rebalancing Frequency:\n")
                    f.write(f"  - Only {len(rebalance_events)} rebalances in {(equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days} days\n")
                    f.write(f"  - Consider lowering threshold from {rebal_config['threshold']:.1%} to {max(0.05, rebal_config['threshold'] * 0.7):.1%}\n")
                    f.write(f"  - Or try calendar-based rebalancing (monthly/quarterly)\n\n")
                elif len(rebalance_events) > len(equity_df) * 0.05:  # More than 5% of days
                    f.write("⚠ High Rebalancing Frequency:\n")
                    f.write(f"  - {len(rebalance_events)} rebalances may incur significant transaction costs\n")
                    f.write(f"  - Consider raising threshold from {rebal_config['threshold']:.1%} to {min(0.20, rebal_config['threshold'] * 1.5):.1%}\n")
                    f.write(f"  - Or increase min_rebalance_interval_hours\n\n")
                else:
                    f.write(f"✓ Rebalancing frequency is optimal ({len(rebalance_events)} events)\n\n")
            else:
                f.write(f"✗ Rebalancing underperformed buy-and-hold by {abs(outperformance):.2%}\n\n")
                f.write("POTENTIAL IMPROVEMENTS:\n")
                f.write("1. Adjust rebalancing threshold - Current may not suit market conditions\n")
                f.write(f"   - Try: {max(0.05, rebal_config['threshold'] * 0.5):.1%}, {rebal_config['threshold'] * 1.5:.1%}, or {rebal_config['threshold'] * 2:.1%}\n")
                f.write("2. Enable momentum filter to avoid rebalancing during strong trends\n")
                f.write("3. Try hybrid method (calendar + threshold)\n")
                f.write("4. Consider different asset allocations\n")
                f.write("5. Extend backtest period - Longer periods better demonstrate rebalancing benefits\n\n")

            # Risk analysis
            portfolio_volatility = equity_df['total_value'].pct_change().std() * (252 ** 0.5)  # Annualized
            buyhold_volatility = buy_hold_df['buy_hold_value'].pct_change().std() * (252 ** 0.5)

            f.write("RISK ANALYSIS:\n")
            f.write(f"- Portfolio Volatility (annualized): {portfolio_volatility:.2%}\n")
            f.write(f"- Buy-and-Hold Volatility (annualized): {buyhold_volatility:.2%}\n")
            if portfolio_volatility < buyhold_volatility:
                f.write(f"✓ Rebalancing reduced volatility by {(1 - portfolio_volatility/buyhold_volatility):.2%}\n")
            else:
                f.write(f"⚠ Rebalancing increased volatility by {(portfolio_volatility/buyhold_volatility - 1):.2%}\n")
            f.write("\n")

            f.write("=" * 100 + "\n")
            f.write("END OF ENHANCED PORTFOLIO REPORT\n")
            f.write("=" * 100 + "\n")

        logger.success(f"\nEnhanced portfolio report generated: {report_path}")
        logger.info("View the report for detailed portfolio analysis and insights")

    def _load_portfolio_config(self, config_path: str) -> Dict:
        """Load portfolio configuration from YAML file."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config: {config_path}")
        logger.info(f"Portfolio: {config['run']['name']}")
        logger.info(f"Assets: {len(config['portfolio']['assets'])}")

        return config

    def _fetch_portfolio_data(self, config: Dict) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all assets in portfolio."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Fetching Portfolio Data")
        logger.info("=" * 70)

        timeframe = config['data']['timeframe']
        days = config['data']['days']

        # Calculate limit
        if timeframe == "1h":
            limit = days * 24
        elif timeframe == "4h":
            limit = days * 6
        elif timeframe == "1d":
            limit = days
        else:
            limit = days * 24

        portfolio_data = {}

        for asset in config['portfolio']['assets']:
            symbol = asset['symbol']
            logger.info(f"\nFetching {symbol}...")

            try:
                data = self.fetcher.get_ohlcv(symbol, timeframe, limit=limit)

                if data is None or len(data) == 0:
                    raise ValueError(f"No data fetched for {symbol}")

                portfolio_data[symbol] = data

                logger.success(f"  ✓ Fetched {len(data)} candles")
                logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                raise

        logger.success(f"\nFetched data for {len(portfolio_data)} assets")
        return portfolio_data

    def _calculate_portfolio_metrics(self, config: Dict, portfolio_data: Dict[str, pd.DataFrame]) -> tuple:
        """Calculate portfolio performance with rebalancing simulation."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Running Portfolio Simulation")
        logger.info("=" * 70)

        # Get common timestamps
        timestamps = None
        for symbol, data in portfolio_data.items():
            if timestamps is None:
                timestamps = data.index
            else:
                timestamps = timestamps.intersection(data.index)

        logger.info(f"Common timespan: {len(timestamps)} periods")

        # Initialize portfolio
        initial_capital = config['capital']['initial_capital']
        assets = [(a['symbol'], a['weight']) for a in config['portfolio']['assets']]
        rebalance_threshold = config['portfolio']['rebalancing']['threshold']
        min_interval_hours = config['portfolio']['rebalancing']['min_rebalance_interval_hours']

        # Get rebalancing method if specified
        rebalance_method = config['portfolio']['rebalancing'].get('rebalance_method', 'threshold')
        calendar_period_days = config['portfolio']['rebalancing'].get('calendar_period_days', 30)
        use_momentum_filter = config['portfolio']['rebalancing'].get('use_momentum_filter', False)
        momentum_lookback_days = config['portfolio']['rebalancing'].get('momentum_lookback_days', 30)

        # Portfolio state tracking
        portfolio_values = {}
        shares = {}

        # Initialize allocations
        for symbol, weight in assets:
            portfolio_values[symbol] = initial_capital * weight
            initial_price = portfolio_data[symbol].loc[timestamps[0], 'close']
            shares[symbol] = portfolio_values[symbol] / initial_price

        # Track metrics
        equity_curve = []
        rebalance_events = []
        last_rebalance = None

        for idx, timestamp in enumerate(timestamps):
            # Get current prices
            prices = {symbol: portfolio_data[symbol].loc[timestamp, 'close']
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

            # Determine rebalancing based on method
            if rebalance_method == "threshold":
                if max_deviation > rebalance_threshold:
                    needs_rebalance = True

            elif rebalance_method == "calendar":
                if last_rebalance is not None:
                    days_since = (timestamp - last_rebalance).total_seconds() / (3600 * 24)
                    if days_since >= calendar_period_days:
                        needs_rebalance = True

            elif rebalance_method == "hybrid":
                threshold_triggered = max_deviation > rebalance_threshold
                calendar_triggered = False
                if last_rebalance is not None:
                    days_since = (timestamp - last_rebalance).total_seconds() / (3600 * 24)
                    calendar_triggered = days_since >= calendar_period_days

                if threshold_triggered or calendar_triggered:
                    needs_rebalance = True

            # Check minimum interval
            if needs_rebalance and last_rebalance is not None:
                hours_since = (timestamp - last_rebalance).total_seconds() / 3600
                if hours_since < min_interval_hours:
                    needs_rebalance = False

            # Apply momentum filter if enabled
            if needs_rebalance and use_momentum_filter and idx >= momentum_lookback_days * 24:
                lookback_idx = max(0, idx - momentum_lookback_days * 24)
                lookback_timestamp = timestamps[lookback_idx]
                old_prices = {symbol: portfolio_data[symbol].loc[lookback_timestamp, 'close']
                              for symbol in prices}
                old_total_value = sum(shares[symbol] * old_prices[symbol] for symbol in shares)
                portfolio_return = (total_value - old_total_value) / old_total_value

                if portfolio_return > 0.20:  # Skip if >20% gain
                    needs_rebalance = False

            # Execute rebalance if needed
            if needs_rebalance:
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
        logger.info(f"  Total return: {((equity_df['total_value'].iloc[-1] / initial_capital) - 1) * 100:.2f}%")

        return equity_df, rebalance_events

    def _calculate_buy_hold_benchmark(self, config: Dict, portfolio_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate buy-and-hold benchmark."""
        logger.info("\nCalculating buy-and-hold benchmark...")

        # Get common timestamps
        timestamps = None
        for data in portfolio_data.values():
            if timestamps is None:
                timestamps = data.index
            else:
                timestamps = timestamps.intersection(data.index)

        initial_capital = config['capital']['initial_capital']
        assets = [(a['symbol'], a['weight']) for a in config['portfolio']['assets']]

        # Buy and hold - initial allocation only
        shares = {}
        for symbol, weight in assets:
            allocation = initial_capital * weight
            initial_price = portfolio_data[symbol].loc[timestamps[0], 'close']
            shares[symbol] = allocation / initial_price

        # Track value over time
        buy_hold_values = []
        for timestamp in timestamps:
            prices = {symbol: portfolio_data[symbol].loc[timestamp, 'close']
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

    def _generate_portfolio_reports(self, config: Dict, equity_df: pd.DataFrame,
                                     buy_hold_df: pd.DataFrame, rebalance_events: List[Dict]) -> None:
        """Generate comprehensive portfolio reports."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Generating Reports")
        logger.info("=" * 70)

        # Save equity curve
        equity_path = self.data_dir / "portfolio_equity_curve.csv"
        equity_df.to_csv(equity_path, index=False)
        logger.info(f"  ✓ Equity curve: {equity_path}")

        # Save buy-hold benchmark
        buyhold_path = self.data_dir / "buy_hold_benchmark.csv"
        buy_hold_df.to_csv(buyhold_path, index=False)
        logger.info(f"  ✓ Buy-hold benchmark: {buyhold_path}")

        # Save rebalance events
        if rebalance_events:
            rebalance_path = self.data_dir / "rebalance_events.csv"
            pd.DataFrame(rebalance_events).to_csv(rebalance_path, index=False)
            logger.info(f"  ✓ Rebalance events: {rebalance_path}")

        # Generate summary report
        self._generate_portfolio_summary(config, equity_df, buy_hold_df, rebalance_events)

        logger.success("\nAll reports generated successfully!")

    def _generate_portfolio_summary(self, config: Dict, equity_df: pd.DataFrame,
                                     buy_hold_df: pd.DataFrame, rebalance_events: List[Dict]) -> None:
        """Generate text summary report for portfolio."""
        summary_path = self.output_dir / "PORTFOLIO_SUMMARY.txt"

        initial_capital = config['capital']['initial_capital']
        portfolio_final = equity_df['total_value'].iloc[-1]
        portfolio_return = (portfolio_final / initial_capital) - 1

        buyhold_final = buy_hold_df['buy_hold_value'].iloc[-1]
        buyhold_return = (buyhold_final / initial_capital) - 1

        outperformance = portfolio_return - buyhold_return

        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"PORTFOLIO REBALANCING - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Run Name: {config['run']['name']}\n")
            f.write(f"Description: {config['run']['description']}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("=" * 70 + "\n")
            f.write("PORTFOLIO CONFIGURATION\n")
            f.write("=" * 70 + "\n\n")

            f.write("Assets:\n")
            for asset in config['portfolio']['assets']:
                f.write(f"  {asset['symbol']}: {asset['weight']:.1%}\n")

            f.write(f"\nRebalancing:\n")
            rebal_config = config['portfolio']['rebalancing']
            f.write(f"  Method: {rebal_config.get('rebalance_method', 'threshold')}\n")
            f.write(f"  Threshold: {rebal_config['threshold']:.1%}\n")
            f.write(f"  Min Interval: {rebal_config['min_rebalance_interval_hours']} hours\n")

            if rebal_config.get('use_momentum_filter'):
                f.write(f"  Momentum Filter: Enabled (lookback: {rebal_config.get('momentum_lookback_days', 30)} days)\n")

            f.write("\n")

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
        description="Run comprehensive backtesting pipeline in single-pair or portfolio mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  SINGLE-PAIR MODE (default):
  # Run with defaults (BTC/USDT, 1h, 365 days, $10k capital)
  python run_full_pipeline.py BTC/USDT

  # Custom timeframe and period
  python run_full_pipeline.py ETH/USDT --timeframe 4h --days 180

  # Custom capital and output directory
  python run_full_pipeline.py BTC/USDT --capital 50000 --output-dir my_results

  # Full customization
  python run_full_pipeline.py SOL/USDT --timeframe 1d --days 730 \\
      --capital 100000 --commission 0.001 --output-dir sol_backtest

  PORTFOLIO MODE (multi-asset rebalancing):
  # Run optimized 10% threshold portfolio (recommended)
  python run_full_pipeline.py --portfolio --config config_improved_10pct.yaml

  # Run 5% threshold portfolio
  python run_full_pipeline.py --portfolio --config config_improved_5pct.yaml

  # Run custom portfolio configuration
  python run_full_pipeline.py --portfolio --config my_portfolio.yaml
        """
    )

    # Symbol argument (optional when using portfolio mode)
    parser.add_argument(
        "symbol",
        nargs='?',
        type=str,
        default=None,
        help="Trading pair symbol (e.g., BTC/USDT, ETH/USDT) - required for single-pair mode",
    )

    # Portfolio mode arguments
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Run in portfolio rebalancing mode (requires --config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to portfolio YAML configuration file (for portfolio mode)",
    )

    # Single-pair mode arguments
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
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate enhanced comparison report (single-pair mode only)",
    )

    args = parser.parse_args()

    # Determine execution mode
    portfolio_mode = args.portfolio or args.config is not None

    if portfolio_mode:
        # PORTFOLIO MODE
        if not args.config:
            logger.error("Portfolio mode requires --config argument")
            logger.info("Example: python run_full_pipeline.py --portfolio --config config_improved_10pct.yaml")
            sys.exit(1)

        # Create minimal runner instance for portfolio mode
        # The runner will reconfigure itself based on the config file
        runner = FullPipelineRunner(
            symbol="PORTFOLIO",  # Placeholder
            timeframe=args.timeframe,
            days=args.days,
            initial_capital=args.capital,
            commission=args.commission,
            slippage=args.slippage,
            output_dir=args.output_dir,
            max_position_risk=args.max_position_risk,
            max_portfolio_risk=args.max_portfolio_risk,
        )

        # Run portfolio mode
        runner.run_portfolio_mode(args.config, generate_enhanced=args.report)

    else:
        # SINGLE-PAIR MODE
        if not args.symbol:
            logger.error("Single-pair mode requires a symbol argument")
            logger.info("Example: python run_full_pipeline.py BTC/USDT")
            logger.info("Or use --portfolio --config for portfolio mode")
            sys.exit(1)

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

        # Generate enhanced report if requested
        if args.report:
            runner.generate_enhanced_report()


if __name__ == "__main__":
    main()
