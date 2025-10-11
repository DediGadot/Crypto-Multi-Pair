"""
Simple Moving Average Crossover Strategy

This module implements a classic trend-following strategy using Simple Moving Average (SMA)
crossovers. The strategy generates buy signals when a fast SMA crosses above a slow SMA
(golden cross) and sell signals when the fast SMA crosses below the slow SMA (death cross).

**Purpose**: Implement a trend-following strategy based on SMA crossover signals for
cryptocurrency trading.

**Strategy Type**: Trend Following
**Indicators**: SMA(50), SMA(200)
**Entry Signal**: SMA(50) crosses above SMA(200) - Golden Cross
**Exit Signal**: SMA(50) crosses below SMA(200) - Death Cross

**Parameters**:
- fast_period: Period for fast SMA (default: 50)
- slow_period: Period for slow SMA (default: 200)

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- pandas_ta: https://github.com/twopirllc/pandas-ta
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```python
data = pd.DataFrame({
    'timestamp': [...],
    'open': [100, 101, 102, ...],
    'high': [105, 106, 107, ...],
    'low': [99, 100, 101, ...],
    'close': [103, 104, 105, ...],
    'volume': [1000, 1100, 1200, ...]
})
```

**Expected Output**:
```python
signals = pd.DataFrame({
    'timestamp': [...],
    'signal': ['HOLD', 'BUY', 'HOLD', 'SELL', ...],
    'confidence': [0.0, 0.85, 0.0, 0.82, ...],
    'metadata': [
        {},
        {'reason': 'golden_cross', 'fast_sma': 103.5, 'slow_sma': 102.0},
        {},
        {'reason': 'death_cross', 'fast_sma': 101.0, 'slow_sma': 102.5},
        ...
    ]
})
```
"""

from typing import Any, Dict, List

import pandas as pd
import pandas_ta as ta
from loguru import logger

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy


@register_strategy(
    name="SMA_Crossover",
    description="Simple Moving Average crossover strategy (Golden/Death Cross)",
    tags=["trend_following", "moving_average", "crossover"]
)
class SMACrossoverStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Generates buy signals on golden cross (fast SMA > slow SMA) and
    sell signals on death cross (fast SMA < slow SMA).

    Classic trend-following approach suitable for identifying major trend changes.
    """

    def __init__(self, name: str = "SMA_Crossover", config: Dict[str, Any] = None):
        """
        Initialize the SMA Crossover strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default parameters
        self.fast_period = 50
        self.slow_period = 200

        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with 'fast_period' and 'slow_period'

        Raises:
            ValueError: If parameters are invalid
        """
        self.fast_period = config.get("fast_period", 50)
        self.slow_period = config.get("slow_period", 200)

        # Validate parameters
        if self.fast_period <= 0 or self.slow_period <= 0:
            raise ValueError("SMA periods must be positive")

        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"Fast period ({self.fast_period}) must be less than "
                f"slow period ({self.slow_period})"
            )

        self._initialized = True
        logger.info(
            f"{self.name} initialized with fast_period={self.fast_period}, "
            f"slow_period={self.slow_period}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period
        }

    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators.

        Returns:
            Empty list - indicators are calculated internally
        """
        return []

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on SMA crossover.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with columns: ['timestamp', 'signal', 'confidence', 'metadata']

        Raises:
            ValueError: If data is invalid
        """
        # Validate data
        if not self.validate_data(data):
            raise ValueError("Invalid data provided to generate_signals")

        # Ensure we have enough data for slow SMA
        if len(data) < self.slow_period:
            logger.warning(
                f"Insufficient data: {len(data)} rows, need {self.slow_period} "
                f"for slow SMA"
            )
            # Return HOLD signals
            return self._create_hold_signals(data)

        # Calculate SMAs using pandas_ta
        df = data.copy()
        df['fast_sma'] = ta.sma(df['close'], length=self.fast_period)
        df['slow_sma'] = ta.sma(df['close'], length=self.slow_period)

        # Initialize signal arrays
        signals = []
        confidences = []
        metadata = []

        # Generate signals based on crossovers
        for i in range(len(df)):
            if i == 0:
                # First row - no previous data
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Check for NaN values (not enough history yet)
            if pd.isna(df['fast_sma'].iloc[i]) or pd.isna(df['slow_sma'].iloc[i]):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Get current and previous SMA values
            fast_current = df['fast_sma'].iloc[i]
            slow_current = df['slow_sma'].iloc[i]
            fast_previous = df['fast_sma'].iloc[i - 1]
            slow_previous = df['slow_sma'].iloc[i - 1]

            # Golden Cross: fast SMA crosses above slow SMA
            if fast_previous <= slow_previous and fast_current > slow_current:
                distance = abs(fast_current - slow_current)
                avg_price = (fast_current + slow_current) / 2
                confidence = min(0.5 + (distance / avg_price) * 100, 1.0)

                signals.append(SignalType.BUY.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'golden_cross',
                    'fast_sma': float(fast_current),
                    'slow_sma': float(slow_current),
                    'crossover_strength': float(distance / avg_price)
                })
                logger.debug(f"Golden Cross detected at {df.index[i]}")

            # Death Cross: fast SMA crosses below slow SMA
            elif fast_previous >= slow_previous and fast_current < slow_current:
                distance = abs(fast_current - slow_current)
                avg_price = (fast_current + slow_current) / 2
                confidence = min(0.5 + (distance / avg_price) * 100, 1.0)

                signals.append(SignalType.SELL.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'death_cross',
                    'fast_sma': float(fast_current),
                    'slow_sma': float(slow_current),
                    'crossover_strength': float(distance / avg_price)
                })
                logger.debug(f"Death Cross detected at {df.index[i]}")

            # No crossover - HOLD
            else:
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({
                    'fast_sma': float(fast_current),
                    'slow_sma': float(slow_current)
                })

        # Create result DataFrame
        result = pd.DataFrame({
            'timestamp': df.index if isinstance(df.index, pd.DatetimeIndex) else df['timestamp'],
            'signal': signals,
            'confidence': confidences,
            'metadata': metadata
        })

        logger.info(
            f"Generated {len(result)} signals: "
            f"{sum(s == SignalType.BUY.value for s in signals)} BUY, "
            f"{sum(s == SignalType.SELL.value for s in signals)} SELL, "
            f"{sum(s == SignalType.HOLD.value for s in signals)} HOLD"
        )

        return result

    def _create_hold_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create HOLD signals for all rows.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with HOLD signals
        """
        return pd.DataFrame({
            'timestamp': data.index if isinstance(data.index, pd.DatetimeIndex) else data['timestamp'],
            'signal': [SignalType.HOLD.value] * len(data),
            'confidence': [0.0] * len(data),
            'metadata': [{}] * len(data)
        })


if __name__ == "__main__":
    """
    Validation block for SMA Crossover Strategy.
    Tests the strategy with REAL BTC/USDT data from Binance.
    """
    import sys
    from datetime import datetime, timedelta

    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting SMA Crossover Strategy validation with REAL DATA")

    # Test 1: Initialize strategy with default parameters
    total_tests += 1
    try:
        strategy = SMACrossoverStrategy()
        strategy.initialize({})

        params = strategy.get_parameters()
        if params['fast_period'] != 50:
            all_validation_failures.append(
                f"Test 1: Expected fast_period=50, got {params['fast_period']}"
            )
        if params['slow_period'] != 200:
            all_validation_failures.append(
                f"Test 1: Expected slow_period=200, got {params['slow_period']}"
            )

        if not all_validation_failures or len(all_validation_failures) == 0:
            logger.success("Test 1 PASSED: Strategy initialized with default parameters")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Initialize with custom parameters
    total_tests += 1
    try:
        custom_strategy = SMACrossoverStrategy(name="CustomSMA")
        custom_strategy.initialize({"fast_period": 20, "slow_period": 50})

        params = custom_strategy.get_parameters()
        if params['fast_period'] != 20:
            all_validation_failures.append(
                f"Test 2: Expected fast_period=20, got {params['fast_period']}"
            )
        if params['slow_period'] != 50:
            all_validation_failures.append(
                f"Test 2: Expected slow_period=50, got {params['slow_period']}"
            )

        logger.success("Test 2 PASSED: Custom parameters set correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Test parameter validation
    total_tests += 1
    try:
        invalid_strategy = SMACrossoverStrategy()
        error_raised = False

        try:
            invalid_strategy.initialize({"fast_period": 100, "slow_period": 50})
        except ValueError as e:
            if "must be less than" in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append(
                "Test 3: Expected ValueError for fast_period >= slow_period"
            )
        else:
            logger.success("Test 3 PASSED: Parameter validation works")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Fetch real BTC/USDT data for signal generation
    total_tests += 1
    try:
        logger.info("Fetching BTC/USDT 1d data for last 300 days...")
        fetcher = BinanceDataFetcher(use_storage=False, use_cache=False)

        # Fetch 300 days to have enough data for 200-day SMA
        end_date = datetime.now()
        start_date = end_date - timedelta(days=300)

        data = fetcher.get_ohlcv(
            "BTC/USDT",
            "1d",
            start_date=start_date,
            end_date=end_date,
            limit=300
        )

        if data is None or data.empty:
            all_validation_failures.append("Test 4: Failed to fetch data")
        elif len(data) < 200:
            all_validation_failures.append(
                f"Test 4: Insufficient data - got {len(data)} rows, need 200+"
            )
        else:
            logger.success(f"Test 4 PASSED: Fetched {len(data)} days of BTC/USDT data")
            logger.info(f"Data range: {data.index.min()} to {data.index.max()}")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception raised: {e}")

    # Test 5: Generate signals with real data
    total_tests += 1
    try:
        if 'data' in locals() and data is not None and not data.empty:
            # Reset index to get timestamp column
            test_data = data.reset_index()

            signals = strategy.generate_signals(test_data)

            if signals is None or signals.empty:
                all_validation_failures.append("Test 5: No signals generated")
            elif len(signals) != len(test_data):
                all_validation_failures.append(
                    f"Test 5: Signal count mismatch - data: {len(test_data)}, "
                    f"signals: {len(signals)}"
                )
            elif not all(col in signals.columns for col in ['timestamp', 'signal', 'confidence', 'metadata']):
                all_validation_failures.append(
                    f"Test 5: Missing columns in signals. Got: {signals.columns.tolist()}"
                )
            else:
                buy_count = (signals['signal'] == SignalType.BUY.value).sum()
                sell_count = (signals['signal'] == SignalType.SELL.value).sum()
                hold_count = (signals['signal'] == SignalType.HOLD.value).sum()

                logger.success(f"Test 5 PASSED: Generated {len(signals)} signals")
                logger.info(f"  BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")

                # Show some actual signals
                buy_signals = signals[signals['signal'] == SignalType.BUY.value]
                sell_signals = signals[signals['signal'] == SignalType.SELL.value]

                if not buy_signals.empty:
                    logger.info(f"  Latest BUY signal: {buy_signals.iloc[-1]['timestamp']}")
                if not sell_signals.empty:
                    logger.info(f"  Latest SELL signal: {sell_signals.iloc[-1]['timestamp']}")
        else:
            all_validation_failures.append("Test 5: No data available from Test 4")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Verify signal structure and content
    total_tests += 1
    try:
        if 'signals' in locals() and signals is not None and not signals.empty:
            # Check for golden/death cross in metadata
            buy_signals = signals[signals['signal'] == SignalType.BUY.value]
            sell_signals = signals[signals['signal'] == SignalType.SELL.value]

            if not buy_signals.empty:
                first_buy = buy_signals.iloc[0]
                if 'reason' not in first_buy['metadata']:
                    all_validation_failures.append(
                        "Test 6: BUY signal metadata missing 'reason'"
                    )
                elif first_buy['metadata']['reason'] != 'golden_cross':
                    all_validation_failures.append(
                        f"Test 6: Expected reason='golden_cross', "
                        f"got '{first_buy['metadata']['reason']}'"
                    )
                elif first_buy['confidence'] <= 0:
                    all_validation_failures.append(
                        "Test 6: BUY signal should have positive confidence"
                    )

            if not sell_signals.empty:
                first_sell = sell_signals.iloc[0]
                if 'reason' not in first_sell['metadata']:
                    all_validation_failures.append(
                        "Test 6: SELL signal metadata missing 'reason'"
                    )
                elif first_sell['metadata']['reason'] != 'death_cross':
                    all_validation_failures.append(
                        f"Test 6: Expected reason='death_cross', "
                        f"got '{first_sell['metadata']['reason']}'"
                    )

            if not all_validation_failures or len([f for f in all_validation_failures if 'Test 6' in f]) == 0:
                logger.success("Test 6 PASSED: Signal structure and metadata correct")
        else:
            all_validation_failures.append("Test 6: No signals available from Test 5")
    except Exception as e:
        all_validation_failures.append(f"Test 6: Exception raised: {e}")

    # Test 7: Test with insufficient data
    total_tests += 1
    try:
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50),
            'open': range(100, 150),
            'high': range(105, 155),
            'low': range(99, 149),
            'close': range(103, 153),
            'volume': [1000] * 50
        })

        small_signals = strategy.generate_signals(small_data)

        if small_signals is None or small_signals.empty:
            all_validation_failures.append("Test 7: Should return signals even with insufficient data")
        elif len(small_signals) != 50:
            all_validation_failures.append(
                f"Test 7: Expected 50 signals, got {len(small_signals)}"
            )
        elif not all(s == SignalType.HOLD.value for s in small_signals['signal']):
            all_validation_failures.append(
                "Test 7: All signals should be HOLD with insufficient data"
            )
        else:
            logger.success("Test 7 PASSED: Handles insufficient data correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Test 8: Test faster crossover (20/50 instead of 50/200)
    total_tests += 1
    try:
        if 'data' in locals() and data is not None and not data.empty:
            fast_strategy = SMACrossoverStrategy(name="FastSMA")
            fast_strategy.initialize({"fast_period": 20, "slow_period": 50})

            test_data = data.reset_index()
            fast_signals = fast_strategy.generate_signals(test_data)

            fast_buy_count = (fast_signals['signal'] == SignalType.BUY.value).sum()
            fast_sell_count = (fast_signals['signal'] == SignalType.SELL.value).sum()

            # Faster crossover should have more signals
            original_buy_count = (signals['signal'] == SignalType.BUY.value).sum()
            original_sell_count = (signals['signal'] == SignalType.SELL.value).sum()

            if fast_buy_count + fast_sell_count <= original_buy_count + original_sell_count:
                logger.warning(
                    f"Test 8: Expected more signals with faster SMA, but got "
                    f"fast={fast_buy_count + fast_sell_count} vs "
                    f"slow={original_buy_count + original_sell_count}"
                )
                # This is a soft warning, not a failure

            logger.success(
                f"Test 8 PASSED: Fast SMA (20/50) generated "
                f"{fast_buy_count + fast_sell_count} signals"
            )
        else:
            all_validation_failures.append("Test 8: No data available")
    except Exception as e:
        all_validation_failures.append(f"Test 8: Exception raised: {e}")

    # Test 9: Verify confidence calculation
    total_tests += 1
    try:
        if 'signals' in locals() and signals is not None and not signals.empty:
            action_signals = signals[signals['signal'] != SignalType.HOLD.value]

            if action_signals.empty:
                logger.warning("Test 9: No BUY/SELL signals to verify confidence")
            else:
                confidences = action_signals['confidence']

                if (confidences < 0).any():
                    all_validation_failures.append("Test 9: Found negative confidence")
                elif (confidences > 1.0).any():
                    all_validation_failures.append("Test 9: Found confidence > 1.0")
                elif (confidences <= 0).any():
                    all_validation_failures.append("Test 9: Action signals should have positive confidence")
                else:
                    logger.success(
                        f"Test 9 PASSED: Confidence values in valid range "
                        f"[{confidences.min():.3f}, {confidences.max():.3f}]"
                    )
        else:
            all_validation_failures.append("Test 9: No signals available")
    except Exception as e:
        all_validation_failures.append(f"Test 9: Exception raised: {e}")

    # Test 10: Test strategy registration
    total_tests += 1
    try:
        from crypto_trader.strategies.registry import get_registry

        registry = get_registry()
        if "SMA_Crossover" not in registry:
            all_validation_failures.append(
                "Test 10: Strategy not registered in global registry"
            )
        else:
            registered_class = registry.get_strategy("SMA_Crossover")
            if registered_class is not SMACrossoverStrategy:
                all_validation_failures.append(
                    "Test 10: Wrong class registered in registry"
                )
            else:
                logger.success("Test 10 PASSED: Strategy registered correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 10: Exception raised: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("SMA Crossover Strategy validated with REAL BTC/USDT data")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
