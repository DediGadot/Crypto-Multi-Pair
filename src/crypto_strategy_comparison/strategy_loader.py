"""
Strategy Loader Module

Handles loading and managing strategy data:
- Load strategy definitions
- Load historical backtest results
- Validate strategy data
- Cache loaded strategies

Documentation:
- Pandas: https://pandas.pydata.org/docs/

Sample Input:
- strategy_names: List[str] = ["momentum_eth", "mean_reversion_btc"]

Expected Output:
- Dictionary of strategy objects with historical data and configuration
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger


class StrategyLoader:
    """Loads and manages strategy data."""

    def __init__(self):
        """Initialize the strategy loader."""
        self._cache: Dict[str, Dict] = {}
        logger.info("StrategyLoader initialized")

    @staticmethod
    def get_available_strategies() -> List[str]:
        """
        Get list of available strategies.

        Returns:
            List of strategy names
        """
        # In production, this would query a database or config file
        # For now, return mock data
        return [
            "Momentum ETH",
            "Mean Reversion BTC",
            "Grid Trading Multi",
            "DCA Bitcoin",
            "RSI Oversold",
            "MACD Crossover",
            "Bollinger Bands",
            "Arbitrage Bot",
        ]

    def load_strategies(self, strategy_names: List[str]) -> Dict[str, Dict]:
        """
        Load data for specified strategies.

        Args:
            strategy_names: List of strategy names to load

        Returns:
            Dictionary mapping strategy names to their data
        """
        strategies = {}

        for name in strategy_names:
            if name in self._cache:
                logger.info(f"Loading {name} from cache")
                strategies[name] = self._cache[name]
            else:
                logger.info(f"Loading {name} from source")
                strategy_data = self._load_strategy_data(name)
                self._cache[name] = strategy_data
                strategies[name] = strategy_data

        return strategies

    def _load_strategy_data(self, strategy_name: str) -> Dict[str, Any]:
        """
        Load data for a single strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy data dictionary
        """
        # In production, load from database/files
        # For now, generate mock data
        logger.info(f"Generating mock data for {strategy_name}")

        # Generate mock equity curve
        days = 365
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

        # Different strategies have different characteristics
        if "momentum" in strategy_name.lower():
            returns = np.random.normal(0.002, 0.025, days)
        elif "mean_reversion" in strategy_name.lower() or "mean reversion" in strategy_name.lower():
            returns = np.random.normal(0.0015, 0.018, days)
        elif "grid" in strategy_name.lower():
            returns = np.random.normal(0.001, 0.015, days)
        else:
            returns = np.random.normal(0.0012, 0.020, days)

        # Calculate equity curve
        equity = 10000 * (1 + returns).cumprod()

        # Generate mock trades
        num_trades = np.random.randint(50, 200)
        trades = self._generate_mock_trades(strategy_name, dates, num_trades)

        return {
            "name": strategy_name,
            "dates": dates.tolist(),
            "equity": equity.tolist(),
            "returns": returns.tolist(),
            "trades": trades,
            "config": {
                "asset": self._extract_asset(strategy_name),
                "type": self._extract_type(strategy_name),
            }
        }

    def _generate_mock_trades(
        self,
        strategy_name: str,
        dates: pd.DatetimeIndex,
        num_trades: int
    ) -> List[Dict]:
        """
        Generate mock trade data.

        Args:
            strategy_name: Strategy name
            dates: Date range
            num_trades: Number of trades to generate

        Returns:
            List of trade dictionaries
        """
        trades = []

        for i in range(num_trades):
            entry_date = dates[np.random.randint(0, len(dates) - 10)]
            exit_date = entry_date + timedelta(
                days=np.random.randint(1, 10)
            )

            entry_price = 1000 + np.random.normal(0, 100)
            pnl_pct = np.random.normal(1.5, 5.0)  # Average 1.5% with 5% std
            exit_price = entry_price * (1 + pnl_pct / 100)

            side = np.random.choice(["LONG", "SHORT"])

            trade = {
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "symbol": self._extract_asset(strategy_name),
                "side": side,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "duration_hours": round((exit_date - entry_date).total_seconds() / 3600, 1),
            }

            trades.append(trade)

        return trades

    def _extract_asset(self, strategy_name: str) -> str:
        """
        Extract asset from strategy name.

        Args:
            strategy_name: Strategy name

        Returns:
            Asset symbol
        """
        name_lower = strategy_name.lower()

        if "btc" in name_lower or "bitcoin" in name_lower:
            return "BTC"
        elif "eth" in name_lower or "ethereum" in name_lower:
            return "ETH"
        elif "multi" in name_lower:
            return "MULTI"
        else:
            return "BTC"  # Default

    def _extract_type(self, strategy_name: str) -> str:
        """
        Extract strategy type from name.

        Args:
            strategy_name: Strategy name

        Returns:
            Strategy type
        """
        name_lower = strategy_name.lower()

        type_keywords = {
            "momentum": "Momentum",
            "mean_reversion": "Mean Reversion",
            "mean reversion": "Mean Reversion",
            "grid": "Grid Trading",
            "dca": "DCA",
            "rsi": "RSI",
            "macd": "MACD",
            "bollinger": "Bollinger Bands",
            "arbitrage": "Arbitrage",
        }

        for keyword, strategy_type in type_keywords.items():
            if keyword in name_lower:
                return strategy_type

        return "Unknown"


if __name__ == "__main__":
    # Validation function
    import sys

    print("🔍 Validating strategy_loader.py...")

    all_validation_failures = []
    total_tests = 0

    # Test 1: Get available strategies
    total_tests += 1
    try:
        strategies = StrategyLoader.get_available_strategies()

        if not isinstance(strategies, list):
            all_validation_failures.append(
                f"Available strategies: Expected list, got {type(strategies)}"
            )

        if len(strategies) == 0:
            all_validation_failures.append(
                "Available strategies: Expected non-empty list"
            )

        # Check for expected strategies
        expected_strategies = ["Momentum ETH", "Mean Reversion BTC"]
        for expected in expected_strategies:
            if expected not in strategies:
                all_validation_failures.append(
                    f"Available strategies: Expected '{expected}' in list"
                )

    except Exception as e:
        all_validation_failures.append(f"Available strategies test failed: {e}")

    # Test 2: Load strategies
    total_tests += 1
    try:
        loader = StrategyLoader()
        loaded = loader.load_strategies(["Momentum ETH", "Mean Reversion BTC"])

        if not isinstance(loaded, dict):
            all_validation_failures.append(
                f"Load strategies: Expected dict, got {type(loaded)}"
            )

        if len(loaded) != 2:
            all_validation_failures.append(
                f"Load strategies: Expected 2 strategies, got {len(loaded)}"
            )

        for strategy_name in ["Momentum ETH", "Mean Reversion BTC"]:
            if strategy_name not in loaded:
                all_validation_failures.append(
                    f"Load strategies: Missing '{strategy_name}' in results"
                )

    except Exception as e:
        all_validation_failures.append(f"Load strategies test failed: {e}")

    # Test 3: Strategy data structure
    total_tests += 1
    try:
        loader = StrategyLoader()
        loaded = loader.load_strategies(["Momentum ETH"])
        strategy_data = loaded["Momentum ETH"]

        required_keys = ["name", "dates", "equity", "returns", "trades", "config"]
        for key in required_keys:
            if key not in strategy_data:
                all_validation_failures.append(
                    f"Strategy data: Missing required key '{key}'"
                )

        # Check data types
        if not isinstance(strategy_data["dates"], list):
            all_validation_failures.append(
                "Strategy data: 'dates' should be a list"
            )

        if not isinstance(strategy_data["equity"], list):
            all_validation_failures.append(
                "Strategy data: 'equity' should be a list"
            )

        if not isinstance(strategy_data["trades"], list):
            all_validation_failures.append(
                "Strategy data: 'trades' should be a list"
            )

        # Check equity curve has data
        if len(strategy_data["equity"]) == 0:
            all_validation_failures.append(
                "Strategy data: Equity curve should not be empty"
            )

    except Exception as e:
        all_validation_failures.append(f"Strategy data structure test failed: {e}")

    # Test 4: Trade data structure
    total_tests += 1
    try:
        loader = StrategyLoader()
        loaded = loader.load_strategies(["Momentum ETH"])
        trades = loaded["Momentum ETH"]["trades"]

        if len(trades) > 0:
            trade = trades[0]
            required_trade_keys = [
                "entry_date", "exit_date", "symbol", "side",
                "entry_price", "exit_price", "pnl_pct", "duration_hours"
            ]

            for key in required_trade_keys:
                if key not in trade:
                    all_validation_failures.append(
                        f"Trade data: Missing required key '{key}'"
                    )

            # Check side is valid
            if trade["side"] not in ["LONG", "SHORT"]:
                all_validation_failures.append(
                    f"Trade data: Invalid side '{trade['side']}', expected LONG or SHORT"
                )

    except Exception as e:
        all_validation_failures.append(f"Trade data structure test failed: {e}")

    # Test 5: Caching mechanism
    total_tests += 1
    try:
        loader = StrategyLoader()

        # Load first time
        loaded1 = loader.load_strategies(["Momentum ETH"])
        equity1 = loaded1["Momentum ETH"]["equity"]

        # Load second time (should come from cache)
        loaded2 = loader.load_strategies(["Momentum ETH"])
        equity2 = loaded2["Momentum ETH"]["equity"]

        # Check if data is the same (cached)
        if equity1 != equity2:
            all_validation_failures.append(
                "Caching: Expected same data from cache, got different data"
            )

    except Exception as e:
        all_validation_failures.append(f"Caching test failed: {e}")

    # Final validation result
    if all_validation_failures:
        print(
            f"❌ VALIDATION FAILED - {len(all_validation_failures)} "
            f"of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests successful")
        print("Strategy loader is validated and ready for use")
        sys.exit(0)
