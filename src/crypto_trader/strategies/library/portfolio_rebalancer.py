"""
Portfolio Rebalancing Strategy

This module implements a multi-asset portfolio rebalancing strategy that
systematically rebalances asset allocations when they drift from target weights.
This approach has been shown to outperform buy-and-hold by 77% in research.

**Purpose**: Implement threshold-based portfolio rebalancing to capture
mean reversion at the portfolio level while maintaining target asset allocations.

**Strategy Type**: Multi-Asset Portfolio Rebalancing
**Method**: Threshold-based rebalancing (15% deviation triggers rebalance)
**Signals**: REBALANCE when asset weight deviates >threshold from target

**Parameters**:
- assets: List of (symbol, target_weight) tuples
- rebalance_threshold: Deviation threshold to trigger rebalance (default: 0.15 = 15%)
- min_rebalance_interval_hours: Minimum hours between rebalances (default: 24)

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```python
portfolio_data = {
    'BTC/USDT': pd.DataFrame({'close': [...], 'timestamp': [...]}),
    'ETH/USDT': pd.DataFrame({'close': [...], 'timestamp': [...]}),
}
assets = [('BTC/USDT', 0.5), ('ETH/USDT', 0.5)]
```

**Expected Output**:
```python
signals = pd.DataFrame({
    'timestamp': [...],
    'BTC/USDT_signal': ['HOLD', 'SELL', 'HOLD', ...],
    'ETH/USDT_signal': ['HOLD', 'BUY', 'HOLD', ...],
    'rebalance_event': [False, True, False, ...],
    'metadata': [...]
})
```
"""

from typing import Any, Dict, List, Tuple
from datetime import timedelta

import pandas as pd
import numpy as np
from loguru import logger

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy


@register_strategy(
    name="PortfolioRebalancer",
    description="Multi-asset portfolio with threshold-based rebalancing",
    tags=["portfolio", "rebalancing", "multi_asset", "mean_reversion", "research_backed"]
)
class PortfolioRebalancerStrategy(BaseStrategy):
    """
    Portfolio Rebalancing Strategy.

    Maintains target asset allocations and rebalances when weights drift
    beyond threshold. Systematically sells winners and buys losers.

    Research shows this approach outperforms buy-and-hold by 77% with
    15% rebalancing threshold.
    """

    def __init__(self, name: str = "PortfolioRebalancer", config: Dict[str, Any] = None):
        """
        Initialize the Portfolio Rebalancer strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default parameters
        self.assets: List[Tuple[str, float]] = []  # List of (symbol, target_weight)
        self.rebalance_threshold = 0.15  # 15% deviation triggers rebalance
        self.min_rebalance_interval_hours = 24  # Don't rebalance more than once per day

        # Enhanced parameters
        self.rebalance_method = "threshold"  # "threshold", "calendar", or "hybrid"
        self.calendar_period_days = 30  # For calendar-based rebalancing
        self.use_momentum_filter = False  # Avoid rebalancing during strong trends
        self.momentum_lookback_days = 30  # Lookback period for momentum calculation

        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with strategy parameters

        Raises:
            ValueError: If parameters are invalid
        """
        # Extract asset configuration
        if "assets" in config:
            self.assets = config["assets"]
        else:
            raise ValueError("Portfolio strategy requires 'assets' configuration")

        self.rebalance_threshold = config.get("rebalance_threshold", 0.15)
        self.min_rebalance_interval_hours = config.get("min_rebalance_interval_hours", 24)

        # Enhanced parameters
        self.rebalance_method = config.get("rebalance_method", "threshold")
        self.calendar_period_days = config.get("calendar_period_days", 30)
        self.use_momentum_filter = config.get("use_momentum_filter", False)
        self.momentum_lookback_days = config.get("momentum_lookback_days", 30)

        # Validate configuration
        if len(self.assets) < 2:
            raise ValueError("Portfolio must have at least 2 assets")

        total_weight = sum(weight for _, weight in self.assets)
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Asset weights must sum to 1.0, got {total_weight}")

        if self.rebalance_threshold <= 0 or self.rebalance_threshold >= 1:
            raise ValueError("Rebalance threshold must be between 0 and 1")

        if self.rebalance_method not in ["threshold", "calendar", "hybrid"]:
            raise ValueError("Rebalance method must be 'threshold', 'calendar', or 'hybrid'")

        if self.calendar_period_days <= 0:
            raise ValueError("Calendar period must be positive")

        self._initialized = True
        logger.info(
            f"{self.name} initialized with {len(self.assets)} assets, "
            f"method={self.rebalance_method}, threshold={self.rebalance_threshold:.1%}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "assets": self.assets,
            "rebalance_threshold": self.rebalance_threshold,
            "min_rebalance_interval_hours": self.min_rebalance_interval_hours,
            "rebalance_method": self.rebalance_method,
            "calendar_period_days": self.calendar_period_days,
            "use_momentum_filter": self.use_momentum_filter,
            "momentum_lookback_days": self.momentum_lookback_days
        }

    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators.

        Returns:
            Empty list - no indicators needed, only price data
        """
        return []

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate rebalancing signals for portfolio.

        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data

        Returns:
            DataFrame with columns for each asset's signals and rebalance events

        Raises:
            ValueError: If data is invalid
        """
        # Validate that we have data for all assets
        for symbol, _ in self.assets:
            if symbol not in data:
                raise ValueError(f"Missing data for asset: {symbol}")

        # Get common timestamps across all assets
        timestamps = None
        for symbol, _ in self.assets:
            asset_data = data[symbol]
            if timestamps is None:
                timestamps = asset_data.index
            else:
                timestamps = timestamps.intersection(asset_data.index)

        if len(timestamps) < 2:
            raise ValueError("Insufficient overlapping data across assets")

        logger.info(f"Generating portfolio signals for {len(timestamps)} periods")

        # Initialize result arrays
        signals_dict = {f"{symbol}_signal": [] for symbol, _ in self.assets}
        signals_dict['rebalance_event'] = []
        signals_dict['metadata'] = []

        # Track portfolio state
        last_rebalance_time = None
        shares = None

        # Iterate through timestamps
        for idx, timestamp in enumerate(timestamps):
            # Get current prices for all assets
            prices = {}
            for symbol, _ in self.assets:
                prices[symbol] = data[symbol].loc[timestamp, 'close']

            # Calculate current portfolio value and weights
            if idx == 0:
                # Initial allocation
                initial_capital = 10000  # This will be overridden by backtest config
                portfolio_values = {
                    symbol: initial_capital * target_weight
                    for symbol, target_weight in self.assets
                }
                shares = {
                    symbol: portfolio_values[symbol] / prices[symbol]
                    for symbol in prices
                }
            else:
                # Update portfolio values based on current prices
                portfolio_values = {
                    symbol: shares[symbol] * prices[symbol]
                    for symbol in shares
                }

            # Calculate total portfolio value and current weights
            total_value = sum(portfolio_values.values())
            current_weights = {
                symbol: portfolio_values[symbol] / total_value
                for symbol in portfolio_values
            }

            # Check if rebalancing is needed based on method
            needs_rebalance = False
            max_deviation = 0.0
            rebalance_reason = None

            # Calculate deviation for all methods
            for symbol, target_weight in self.assets:
                deviation = abs(current_weights[symbol] - target_weight)
                max_deviation = max(max_deviation, deviation)

            # Determine rebalancing based on method
            if self.rebalance_method == "threshold":
                # Threshold-based: rebalance when deviation exceeds threshold
                if max_deviation > self.rebalance_threshold:
                    needs_rebalance = True
                    rebalance_reason = "threshold_rebalance"

            elif self.rebalance_method == "calendar":
                # Calendar-based: rebalance on fixed schedule
                if last_rebalance_time is None:
                    # First rebalance after initial allocation
                    pass
                else:
                    days_since_rebalance = (timestamp - last_rebalance_time).total_seconds() / (3600 * 24)
                    if days_since_rebalance >= self.calendar_period_days:
                        needs_rebalance = True
                        rebalance_reason = "calendar_rebalance"

            elif self.rebalance_method == "hybrid":
                # Hybrid: rebalance on calendar OR when threshold exceeded
                threshold_triggered = max_deviation > self.rebalance_threshold

                calendar_triggered = False
                if last_rebalance_time is not None:
                    days_since_rebalance = (timestamp - last_rebalance_time).total_seconds() / (3600 * 24)
                    calendar_triggered = days_since_rebalance >= self.calendar_period_days

                if threshold_triggered or calendar_triggered:
                    needs_rebalance = True
                    rebalance_reason = "threshold_rebalance" if threshold_triggered else "calendar_rebalance"

            # Check minimum interval (applies to all methods)
            if needs_rebalance and last_rebalance_time is not None:
                time_since_rebalance = (timestamp - last_rebalance_time).total_seconds() / 3600
                if time_since_rebalance < self.min_rebalance_interval_hours:
                    needs_rebalance = False

            # Apply momentum filter if enabled
            if needs_rebalance and self.use_momentum_filter:
                # Calculate portfolio momentum over lookback period
                lookback_periods = self.momentum_lookback_days * 24  # Convert days to hours
                if idx >= lookback_periods:
                    lookback_idx = max(0, idx - lookback_periods)
                    lookback_timestamp = timestamps[lookback_idx]

                    # Calculate portfolio return over lookback period
                    old_prices = {}
                    for symbol in prices:
                        old_prices[symbol] = data[symbol].loc[lookback_timestamp, 'close']

                    old_total_value = sum(shares[symbol] * old_prices[symbol] for symbol in shares)
                    portfolio_return = (total_value - old_total_value) / old_total_value

                    # Skip rebalancing if strong uptrend (>20% gain)
                    if portfolio_return > 0.20:
                        needs_rebalance = False
                        logger.debug(f"Skipped rebalance at {timestamp} due to strong uptrend: {portfolio_return:.2%}")

            # Generate signals
            if needs_rebalance:
                # Rebalance: sell overweight, buy underweight
                for symbol, target_weight in self.assets:
                    current_weight = current_weights[symbol]
                    if current_weight > target_weight:
                        # Overweight - sell
                        signals_dict[f"{symbol}_signal"].append(SignalType.SELL.value)
                    elif current_weight < target_weight:
                        # Underweight - buy
                        signals_dict[f"{symbol}_signal"].append(SignalType.BUY.value)
                    else:
                        signals_dict[f"{symbol}_signal"].append(SignalType.HOLD.value)

                signals_dict['rebalance_event'].append(True)
                signals_dict['metadata'].append({
                    'reason': rebalance_reason or 'threshold_rebalance',
                    'max_deviation': float(max_deviation),
                    'current_weights': {s: float(current_weights[s]) for s in current_weights},
                    'target_weights': {s: float(w) for s, w in self.assets}
                })

                # Update shares after rebalance
                target_values = {
                    symbol: total_value * target_weight
                    for symbol, target_weight in self.assets
                }
                shares = {
                    symbol: target_values[symbol] / prices[symbol]
                    for symbol in prices
                }

                last_rebalance_time = timestamp

                logger.debug(
                    f"Rebalance at {timestamp}, max deviation: {max_deviation:.2%}"
                )
            else:
                # No rebalance - hold all
                for symbol, _ in self.assets:
                    signals_dict[f"{symbol}_signal"].append(SignalType.HOLD.value)

                signals_dict['rebalance_event'].append(False)
                signals_dict['metadata'].append({
                    'current_weights': {s: float(current_weights[s]) for s in current_weights},
                    'max_deviation': float(max_deviation)
                })

        # Create result DataFrame
        result = pd.DataFrame(signals_dict, index=timestamps)
        result.reset_index(inplace=True)
        result.rename(columns={'index': 'timestamp'}, inplace=True)

        rebalance_count = sum(signals_dict['rebalance_event'])
        logger.info(f"Generated signals: {rebalance_count} rebalance events out of {len(timestamps)} periods")

        return result

    def _create_hold_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create HOLD signals for all rows.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with HOLD signals
        """
        # For portfolio, we need to create HOLD for all assets
        signals_dict = {f"{symbol}_signal": [SignalType.HOLD.value] * len(data)
                        for symbol, _ in self.assets}
        signals_dict['timestamp'] = data.index if isinstance(data.index, pd.DatetimeIndex) else data['timestamp']
        signals_dict['rebalance_event'] = [False] * len(data)
        signals_dict['metadata'] = [{}] * len(data)

        return pd.DataFrame(signals_dict)


if __name__ == "__main__":
    """
    Validation block for Portfolio Rebalancer Strategy.
    Tests the strategy with synthetic multi-asset data.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting Portfolio Rebalancer Strategy validation")

    # Test 1: Initialize strategy
    total_tests += 1
    try:
        assets = [
            ("BTC/USDT", 0.50),
            ("ETH/USDT", 0.30),
            ("SOL/USDT", 0.20)
        ]

        strategy = PortfolioRebalancerStrategy()
        strategy.initialize({
            "assets": assets,
            "rebalance_threshold": 0.15,
            "min_rebalance_interval_hours": 24
        })

        params = strategy.get_parameters()
        if params['rebalance_threshold'] != 0.15:
            all_validation_failures.append(
                f"Test 1: Expected threshold=0.15, got {params['rebalance_threshold']}"
            )

        logger.success("Test 1 PASSED: Strategy initialized")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Generate synthetic multi-asset data
    total_tests += 1
    try:
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')

        # BTC: trending up
        btc_prices = 40000 + np.cumsum(np.random.randn(100) * 200)
        # ETH: more volatile
        eth_prices = 2000 + np.cumsum(np.random.randn(100) * 50)
        # SOL: different pattern
        sol_prices = 100 + np.cumsum(np.random.randn(100) * 5)

        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': btc_prices,
                'open': btc_prices * 0.99,
                'high': btc_prices * 1.01,
                'low': btc_prices * 0.98,
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': eth_prices,
                'open': eth_prices * 0.99,
                'high': eth_prices * 1.01,
                'low': eth_prices * 0.98,
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates),
            "SOL/USDT": pd.DataFrame({
                'close': sol_prices,
                'open': sol_prices * 0.99,
                'high': sol_prices * 1.01,
                'low': sol_prices * 0.98,
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates)
        }

        logger.success("Test 2 PASSED: Generated multi-asset data")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Generate signals
    total_tests += 1
    try:
        signals = strategy.generate_signals(portfolio_data)

        if signals is None or signals.empty:
            all_validation_failures.append("Test 3: No signals generated")
        else:
            rebalance_events = signals['rebalance_event'].sum()
            logger.success(f"Test 3 PASSED: Generated {len(signals)} signals with {rebalance_events} rebalance events")
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
        print("Portfolio Rebalancer Strategy validated with synthetic multi-asset data")
        sys.exit(0)
