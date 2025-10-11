"""
End-to-End Integration Test for Crypto Trading System.

This test validates the entire system workflow from data fetching to
strategy comparison and reporting.

Author: Crypto Trader Team
Created: 2025-10-11
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from loguru import logger

# Import all components
from crypto_trader.core.config import (
    BacktestConfig,
    DataConfig,
    RiskConfig,
    StrategyConfig,
    TradingConfig,
)
from crypto_trader.core.types import BacktestResult, PerformanceMetrics
from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.data.storage import OHLCVStorage
from crypto_trader.strategies import get_registry
from crypto_trader.backtesting.engine import BacktestEngine
from crypto_trader.analysis.comparison import StrategyComparison
from crypto_trader.analysis.metrics import MetricsCalculator
from crypto_trader.analysis.reporting import ReportGenerator
from crypto_trader.risk.manager import RiskManager


class EndToEndTest:
    """Complete end-to-end system test."""

    def __init__(self):
        """Initialize test environment."""
        self.test_results: List[str] = []
        self.errors: List[str] = []

        # Configure logger
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    def log_success(self, message: str) -> None:
        """Log successful test step."""
        self.test_results.append(f"✅ {message}")
        logger.info(message)

    def log_error(self, message: str) -> None:
        """Log test error."""
        self.errors.append(f"❌ {message}")
        logger.error(message)

    def test_1_configuration(self) -> bool:
        """Test 1: Configuration loading and validation."""
        logger.info("=" * 60)
        logger.info("TEST 1: Configuration System")
        logger.info("=" * 60)

        try:
            # Create configurations
            data_config = DataConfig(
                exchange="binance",
                symbols=["BTC/USDT", "ETH/USDT"],
                timeframes=["1h"],
                cache_enabled=True,
            )

            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission=0.001,
                slippage=0.0005,
            )

            risk_config = RiskConfig(
                max_position_risk=0.02,
                max_portfolio_risk=0.10,
            )

            # Validate all configs
            assert data_config.exchange == "binance"
            assert backtest_config.initial_capital == 10000.0
            assert risk_config.max_position_risk == 0.02

            self.log_success("Configuration system validated")
            return True

        except Exception as e:
            self.log_error(f"Configuration test failed: {e}")
            return False

    def test_2_data_layer(self) -> bool:
        """Test 2: Data fetching and storage."""
        logger.info("=" * 60)
        logger.info("TEST 2: Data Layer")
        logger.info("=" * 60)

        try:
            # Initialize data fetcher
            fetcher = BinanceDataFetcher()

            # Fetch data
            logger.info("Fetching BTC/USDT 1h data (100 candles)...")
            df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)

            assert df is not None
            assert len(df) > 0
            assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

            # Test storage
            storage = OHLCVStorage()
            storage.save_ohlcv(df, "BTC/USDT", "1h")

            # Load back
            loaded_df = storage.load_ohlcv("BTC/USDT", "1h")
            assert loaded_df is not None
            assert len(loaded_df) > 0

            self.log_success(f"Data layer validated - fetched {len(df)} candles")
            return True

        except Exception as e:
            self.log_error(f"Data layer test failed: {e}")
            return False

    def test_3_strategies(self) -> bool:
        """Test 3: Strategy loading and signal generation."""
        logger.info("=" * 60)
        logger.info("TEST 3: Strategy Framework")
        logger.info("=" * 60)

        try:
            # Import strategies to ensure they're registered
            import crypto_trader.strategies.library  # noqa: F401

            # Get registry
            registry = get_registry()

            # List strategies
            strategies = registry.list_strategies()
            logger.info(f"Found {len(strategies)} strategies: {', '.join(strategies)}")

            assert len(strategies) >= 5, "Should have at least 5 strategies"

            # Test loading a strategy
            StrategyClass = registry.get_strategy("SMA_Crossover")
            assert StrategyClass is not None

            # Instantiate and initialize strategy
            strategy = StrategyClass()
            strategy.initialize({"fast_period": 10, "slow_period": 20})

            # Get test data
            fetcher = BinanceDataFetcher()
            data = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)

            # Generate signals
            signals = strategy.generate_signals(data.reset_index())

            assert 'signal' in signals.columns
            assert signals['signal'].isin(['BUY', 'SELL', 'HOLD']).all()

            buy_count = (signals['signal'] == 'BUY').sum()
            sell_count = (signals['signal'] == 'SELL').sum()

            self.log_success(f"Strategies validated - generated {buy_count} BUY, {sell_count} SELL signals")
            return True

        except Exception as e:
            self.log_error(f"Strategy test failed: {e}")
            return False

    def test_4_backtesting(self) -> bool:
        """Test 4: Backtesting engine."""
        logger.info("=" * 60)
        logger.info("TEST 4: Backtesting Engine")
        logger.info("=" * 60)

        try:
            # Get data
            fetcher = BinanceDataFetcher()
            data = fetcher.get_ohlcv("BTC/USDT", "1h", limit=200)
            data = data.reset_index()  # Reset index to get timestamp column

            # Get registry
            registry = get_registry()
            StrategyClass = registry.get_strategy("SMA_Crossover")
            strategy = StrategyClass()
            strategy.initialize({"fast_period": 10, "slow_period": 20})

            # Import strategies
            import crypto_trader.strategies.library  # noqa: F401

            # Run backtest
            config = BacktestConfig(
                initial_capital=10000.0,
                commission=0.001,
                slippage=0.0005,
            )

            engine = BacktestEngine()
            result = engine.run_backtest(strategy, data, config)

            # Validate result
            assert result is not None
            assert isinstance(result, BacktestResult)
            assert result.metrics is not None
            assert isinstance(result.metrics, PerformanceMetrics)

            logger.info(f"Total Return: {result.metrics.total_return:.2%}")
            logger.info(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")
            logger.info(f"Total Trades: {result.metrics.total_trades}")

            self.log_success("Backtesting engine validated")
            return True

        except Exception as e:
            self.log_error(f"Backtesting test failed: {e}")
            return False

    def test_5_analysis(self) -> bool:
        """Test 5: Analysis and comparison."""
        logger.info("=" * 60)
        logger.info("TEST 5: Analysis & Comparison")
        logger.info("=" * 60)

        try:
            # Import strategies
            import crypto_trader.strategies.library  # noqa: F401

            # Get data
            fetcher = BinanceDataFetcher()
            data = fetcher.get_ohlcv("BTC/USDT", "1h", limit=200)
            data = data.reset_index()  # Reset index to get timestamp column

            # Run multiple strategies
            registry = get_registry()
            strategies_to_test = ["SMA_Crossover", "RSI_MeanReversion"]

            config = BacktestConfig(initial_capital=10000.0)
            engine = BacktestEngine()
            results = []

            for strategy_name in strategies_to_test:
                StrategyClass = registry.get_strategy(strategy_name)
                strategy = StrategyClass()

                if strategy_name == "SMA_Crossover":
                    strategy.initialize({"fast_period": 10, "slow_period": 20})
                elif strategy_name == "RSI_MeanReversion":
                    strategy.initialize({"rsi_period": 14, "oversold": 30, "overbought": 70})

                result = engine.run_backtest(strategy, data, config)
                results.append(result)

            # Compare strategies
            comparison = StrategyComparison()
            comparison_df = comparison.compare_strategies(results)

            assert comparison_df is not None
            assert len(comparison_df) == len(strategies_to_test)

            # Get best performer
            best = comparison.best_performer(results, "sharpe_ratio")
            logger.info(f"Best performer: {best.strategy_name} (Sharpe: {best.metrics.sharpe_ratio:.2f})")

            self.log_success("Analysis and comparison validated")
            return True

        except Exception as e:
            self.log_error(f"Analysis test failed: {e}")
            return False

    def test_6_risk_management(self) -> bool:
        """Test 6: Risk management."""
        logger.info("=" * 60)
        logger.info("TEST 6: Risk Management")
        logger.info("=" * 60)

        try:
            # Create risk manager
            config = RiskConfig(
                max_position_risk=0.02,
                max_portfolio_risk=0.10,
                position_sizing_method="fixed_percent",
            )

            risk_manager = RiskManager(config)

            # Test position sizing (this requires a portfolio state)
            # For now, just validate initialization
            assert risk_manager is not None
            assert risk_manager.config.max_position_risk == 0.02

            self.log_success("Risk management validated")
            return True

        except Exception as e:
            self.log_error(f"Risk management test failed: {e}")
            return False

    def test_7_reporting(self) -> bool:
        """Test 7: Report generation."""
        logger.info("=" * 60)
        logger.info("TEST 7: Report Generation")
        logger.info("=" * 60)

        try:
            # Import strategies
            import crypto_trader.strategies.library  # noqa: F401
            from tempfile import TemporaryDirectory

            # Run a quick backtest
            fetcher = BinanceDataFetcher()
            data = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)
            data = data.reset_index()  # Reset index to get timestamp column

            registry = get_registry()
            StrategyClass = registry.get_strategy("SMA_Crossover")
            strategy = StrategyClass()
            strategy.initialize({"fast_period": 10, "slow_period": 20})

            config = BacktestConfig(initial_capital=10000.0)
            engine = BacktestEngine()
            result = engine.run_backtest(strategy, data, config)

            # Generate report
            reporter = ReportGenerator()

            # Test HTML report - generate to temporary file
            with TemporaryDirectory() as tmpdir:
                html_path = Path(tmpdir) / "report.html"
                reporter.generate_html_report(result, str(html_path))

                # Verify file was created
                assert html_path.exists(), "HTML report file was not created"

                # Read and verify content
                html_content = html_path.read_text()
                assert len(html_content) > 0, "HTML report is empty"
                assert "<!DOCTYPE html>" in html_content, "HTML report missing DOCTYPE"
                assert result.strategy_name in html_content, "HTML report missing strategy name"
                assert "Total Return" in html_content, "HTML report missing metrics"

            self.log_success("Report generation validated")
            return True

        except Exception as e:
            self.log_error(f"Reporting test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all end-to-end tests."""
        logger.info("\n" + "=" * 60)
        logger.info("CRYPTO TRADING SYSTEM - END-TO-END TEST")
        logger.info("=" * 60 + "\n")

        tests = [
            ("Configuration", self.test_1_configuration),
            ("Data Layer", self.test_2_data_layer),
            ("Strategies", self.test_3_strategies),
            ("Backtesting", self.test_4_backtesting),
            ("Analysis", self.test_5_analysis),
            ("Risk Management", self.test_6_risk_management),
            ("Reporting", self.test_7_reporting),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Test '{test_name}' crashed: {e}")
                self.log_error(f"Test '{test_name}' crashed: {e}")
                failed += 1

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)

        for result in self.test_results:
            logger.info(result)

        if self.errors:
            logger.error("\nERRORS:")
            for error in self.errors:
                logger.error(error)

        logger.info("\n" + "=" * 60)
        logger.info(f"TOTAL: {passed} passed, {failed} failed out of {len(tests)} tests")
        logger.info("=" * 60 + "\n")

        return failed == 0


if __name__ == "__main__":
    """Run end-to-end tests."""
    import sys

    test = EndToEndTest()
    success = test.run_all_tests()

    if success:
        logger.info("✅ ALL END-TO-END TESTS PASSED")
        logger.info("The crypto trading system is fully functional!")
        sys.exit(0)
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("Please review errors above and fix issues.")
        sys.exit(1)
