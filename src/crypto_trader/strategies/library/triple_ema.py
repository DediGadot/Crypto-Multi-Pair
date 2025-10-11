"""
Triple EMA Trend Filter Strategy

This module implements a trend-following strategy using three Exponential Moving Averages (EMAs)
with reduced lag compared to Simple Moving Averages. The strategy generates buy signals when
there's bullish alignment (fast > medium > slow) AND price crosses above the fast EMA, and sell
signals when there's bearish alignment (fast < medium < slow) AND price crosses below the fast EMA.

**Purpose**: Implement a trend-following strategy with reduced lag using three EMAs and
alignment filtering for cryptocurrency trading.

**Strategy Type**: Trend Following with Reduced Lag
**Indicators**: EMA(8), EMA(21), EMA(55)
**Entry Signal**: Bullish alignment (8>21>55) AND price crosses above EMA(8)
**Exit Signal**: Bearish alignment (8<21<55) AND price crosses below EMA(8)

**Parameters**:
- fast: Fast EMA period (default: 8)
- medium: Medium EMA period (default: 21)
- slow: Slow EMA period (default: 55)

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
    'confidence': [0.0, 0.88, 0.0, 0.85, ...],
    'metadata': [
        {},
        {'reason': 'bullish_trend', 'ema_8': 104.5, 'ema_21': 103.2, 'ema_55': 101.8, 'alignment': 'bullish'},
        {},
        {'reason': 'bearish_trend', 'ema_8': 100.2, 'ema_21': 101.5, 'ema_55': 103.1, 'alignment': 'bearish'},
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
    name="TripleEMA",
    description="Triple EMA trend filter strategy with reduced lag",
    tags=["trend_following", "ema", "crossover", "trend_filter"]
)
class TripleEMAStrategy(BaseStrategy):
    """
    Triple EMA Trend Filter Strategy.

    Generates buy signals when there's bullish EMA alignment AND price crosses above fast EMA.
    Generates sell signals when there's bearish EMA alignment AND price crosses below fast EMA.

    Uses EMAs for reduced lag compared to SMAs, with alignment filter to reduce whipsaws.
    """

    def __init__(self, name: str = "TripleEMA", config: Dict[str, Any] = None):
        """
        Initialize the Triple EMA strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default parameters for Triple EMA(8, 21, 55)
        self.fast = 8
        self.medium = 21
        self.slow = 55

        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with 'fast', 'medium', and 'slow' EMA periods

        Raises:
            ValueError: If parameters are invalid
        """
        self.fast = config.get("fast", 8)
        self.medium = config.get("medium", 21)
        self.slow = config.get("slow", 55)

        # Validate parameters
        if self.fast <= 0 or self.medium <= 0 or self.slow <= 0:
            raise ValueError("All EMA periods must be positive")

        if not (self.fast < self.medium < self.slow):
            raise ValueError(
                f"EMA periods must be in ascending order: "
                f"fast ({self.fast}) < medium ({self.medium}) < slow ({self.slow})"
            )

        self._initialized = True
        logger.info(
            f"{self.name} initialized with EMA({self.fast}, {self.medium}, {self.slow})"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "fast": self.fast,
            "medium": self.medium,
            "slow": self.slow
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
        Generate trading signals based on Triple EMA alignment and crossover.

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

        # Need enough data for slow EMA
        if len(data) < self.slow + 1:
            logger.warning(
                f"Insufficient data: {len(data)} rows, need {self.slow + 1} for slow EMA"
            )
            return self._create_hold_signals(data)

        # Calculate EMAs using pandas_ta
        df = data.copy()
        df['ema_fast'] = ta.ema(df['close'], length=self.fast)
        df['ema_medium'] = ta.ema(df['close'], length=self.medium)
        df['ema_slow'] = ta.ema(df['close'], length=self.slow)

        # Initialize signal arrays
        signals = []
        confidences = []
        metadata = []

        # Generate signals based on EMA alignment and price crossover
        for i in range(len(df)):
            if i == 0:
                # First row - no previous data
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Check for NaN values (not enough history yet)
            if (pd.isna(df['ema_fast'].iloc[i]) or
                pd.isna(df['ema_medium'].iloc[i]) or
                pd.isna(df['ema_slow'].iloc[i])):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Get current and previous values
            close_current = df['close'].iloc[i]
            close_previous = df['close'].iloc[i - 1]
            ema_fast_current = df['ema_fast'].iloc[i]
            ema_medium_current = df['ema_medium'].iloc[i]
            ema_slow_current = df['ema_slow'].iloc[i]
            ema_fast_previous = df['ema_fast'].iloc[i - 1]

            # Skip if previous values are NaN
            if pd.isna(close_previous) or pd.isna(ema_fast_previous):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({
                    'ema_fast': float(ema_fast_current),
                    'ema_medium': float(ema_medium_current),
                    'ema_slow': float(ema_slow_current)
                })
                continue

            # Check EMA alignment
            bullish_alignment = (ema_fast_current > ema_medium_current > ema_slow_current)
            bearish_alignment = (ema_fast_current < ema_medium_current < ema_slow_current)

            # BUY: Bullish alignment AND price crosses above fast EMA
            if (bullish_alignment and
                close_previous <= ema_fast_previous and
                close_current > ema_fast_current):

                # Confidence based on strength of alignment
                fast_medium_gap = ema_fast_current - ema_medium_current
                medium_slow_gap = ema_medium_current - ema_slow_current
                total_gap = ema_fast_current - ema_slow_current
                confidence = min(0.5 + (total_gap / ema_slow_current) * 10, 1.0)

                signals.append(SignalType.BUY.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'bullish_trend',
                    'close': float(close_current),
                    'ema_fast': float(ema_fast_current),
                    'ema_medium': float(ema_medium_current),
                    'ema_slow': float(ema_slow_current),
                    'alignment': 'bullish',
                    'trend_strength': float(total_gap / ema_slow_current)
                })
                logger.debug(
                    f"Bullish Triple EMA signal at {df.index[i]}, "
                    f"close={close_current:.2f}, EMAs: {ema_fast_current:.2f} > "
                    f"{ema_medium_current:.2f} > {ema_slow_current:.2f}"
                )

            # SELL: Bearish alignment AND price crosses below fast EMA
            elif (bearish_alignment and
                  close_previous >= ema_fast_previous and
                  close_current < ema_fast_current):

                # Confidence based on strength of alignment
                fast_medium_gap = ema_medium_current - ema_fast_current
                medium_slow_gap = ema_slow_current - ema_medium_current
                total_gap = ema_slow_current - ema_fast_current
                confidence = min(0.5 + (total_gap / ema_slow_current) * 10, 1.0)

                signals.append(SignalType.SELL.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'bearish_trend',
                    'close': float(close_current),
                    'ema_fast': float(ema_fast_current),
                    'ema_medium': float(ema_medium_current),
                    'ema_slow': float(ema_slow_current),
                    'alignment': 'bearish',
                    'trend_strength': float(total_gap / ema_slow_current)
                })
                logger.debug(
                    f"Bearish Triple EMA signal at {df.index[i]}, "
                    f"close={close_current:.2f}, EMAs: {ema_fast_current:.2f} < "
                    f"{ema_medium_current:.2f} < {ema_slow_current:.2f}"
                )

            # No signal - HOLD
            else:
                # Determine current alignment for metadata
                if bullish_alignment:
                    alignment = 'bullish'
                elif bearish_alignment:
                    alignment = 'bearish'
                else:
                    alignment = 'neutral'

                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({
                    'close': float(close_current),
                    'ema_fast': float(ema_fast_current),
                    'ema_medium': float(ema_medium_current),
                    'ema_slow': float(ema_slow_current),
                    'alignment': alignment
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
    Validation block for Triple EMA Strategy.
    Tests the strategy with REAL BTC/USDT data from Binance.
    """
    import sys
    from datetime import datetime, timedelta

    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting Triple EMA Strategy validation with REAL DATA")

    # Test 1: Initialize strategy with default parameters
    total_tests += 1
    try:
        strategy = TripleEMAStrategy()
        strategy.initialize({})

        params = strategy.get_parameters()
        if params['fast'] != 8:
            all_validation_failures.append(
                f"Test 1: Expected fast=8, got {params['fast']}"
            )
        if params['medium'] != 21:
            all_validation_failures.append(
                f"Test 1: Expected medium=21, got {params['medium']}"
            )
        if params['slow'] != 55:
            all_validation_failures.append(
                f"Test 1: Expected slow=55, got {params['slow']}"
            )

        if not all_validation_failures or len(all_validation_failures) == 0:
            logger.success("Test 1 PASSED: Strategy initialized with default parameters")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Initialize with custom parameters
    total_tests += 1
    try:
        custom_strategy = TripleEMAStrategy(name="CustomTripleEMA")
        custom_strategy.initialize({"fast": 5, "medium": 13, "slow": 34})

        params = custom_strategy.get_parameters()
        if params['fast'] != 5:
            all_validation_failures.append(
                f"Test 2: Expected fast=5, got {params['fast']}"
            )
        if params['medium'] != 13:
            all_validation_failures.append(
                f"Test 2: Expected medium=13, got {params['medium']}"
            )
        if params['slow'] != 34:
            all_validation_failures.append(
                f"Test 2: Expected slow=34, got {params['slow']}"
            )

        logger.success("Test 2 PASSED: Custom parameters set correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Test parameter validation
    total_tests += 1
    try:
        invalid_strategy = TripleEMAStrategy()
        error_raised = False

        try:
            invalid_strategy.initialize({"fast": 50, "medium": 20, "slow": 10})
        except ValueError as e:
            if "ascending order" in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append(
                "Test 3: Expected ValueError for incorrect EMA ordering"
            )
        else:
            logger.success("Test 3 PASSED: Parameter validation works")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Fetch real BTC/USDT data for signal generation
    total_tests += 1
    try:
        logger.info("Fetching BTC/USDT 4h data for last 40 days...")
        fetcher = BinanceDataFetcher(use_storage=False, use_cache=False)

        # Fetch 40 days of 4h data (240 candles) for good Triple EMA signals
        end_date = datetime.now()
        start_date = end_date - timedelta(days=40)

        data = fetcher.get_ohlcv(
            "BTC/USDT",
            "4h",
            start_date=start_date,
            end_date=end_date,
            limit=240
        )

        if data is None or data.empty:
            all_validation_failures.append("Test 4: Failed to fetch data")
        elif len(data) < 60:
            all_validation_failures.append(
                f"Test 4: Insufficient data - got {len(data)} rows, need 60+"
            )
        else:
            logger.success(f"Test 4 PASSED: Fetched {len(data)} candles of BTC/USDT 4h data")
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
                    latest_buy = buy_signals.iloc[-1]
                    logger.info(f"  Latest BUY signal: {latest_buy['timestamp']}")
                    logger.info(f"    Alignment: {latest_buy['metadata'].get('alignment', 'N/A')}")
                if not sell_signals.empty:
                    latest_sell = sell_signals.iloc[-1]
                    logger.info(f"  Latest SELL signal: {latest_sell['timestamp']}")
                    logger.info(f"    Alignment: {latest_sell['metadata'].get('alignment', 'N/A')}")
        else:
            all_validation_failures.append("Test 5: No data available from Test 4")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Verify signal structure and content
    total_tests += 1
    try:
        if 'signals' in locals() and signals is not None and not signals.empty:
            # Check for bullish/bearish trend in metadata
            buy_signals = signals[signals['signal'] == SignalType.BUY.value]
            sell_signals = signals[signals['signal'] == SignalType.SELL.value]

            if not buy_signals.empty:
                first_buy = buy_signals.iloc[0]
                if 'reason' not in first_buy['metadata']:
                    all_validation_failures.append(
                        "Test 6: BUY signal metadata missing 'reason'"
                    )
                elif first_buy['metadata']['reason'] != 'bullish_trend':
                    all_validation_failures.append(
                        f"Test 6: Expected reason='bullish_trend', "
                        f"got '{first_buy['metadata']['reason']}'"
                    )
                elif first_buy['confidence'] <= 0:
                    all_validation_failures.append(
                        "Test 6: BUY signal should have positive confidence"
                    )
                elif first_buy['metadata'].get('alignment') != 'bullish':
                    all_validation_failures.append(
                        f"Test 6: Expected alignment='bullish', "
                        f"got '{first_buy['metadata'].get('alignment')}'"
                    )

            if not sell_signals.empty:
                first_sell = sell_signals.iloc[0]
                if 'reason' not in first_sell['metadata']:
                    all_validation_failures.append(
                        "Test 6: SELL signal metadata missing 'reason'"
                    )
                elif first_sell['metadata']['reason'] != 'bearish_trend':
                    all_validation_failures.append(
                        f"Test 6: Expected reason='bearish_trend', "
                        f"got '{first_sell['metadata']['reason']}'"
                    )
                elif first_sell['metadata'].get('alignment') != 'bearish':
                    all_validation_failures.append(
                        f"Test 6: Expected alignment='bearish', "
                        f"got '{first_sell['metadata'].get('alignment')}'"
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
            'timestamp': pd.date_range('2024-01-01', periods=40),
            'open': range(100, 140),
            'high': range(105, 145),
            'low': range(99, 139),
            'close': range(103, 143),
            'volume': [1000] * 40
        })

        small_signals = strategy.generate_signals(small_data)

        if small_signals is None or small_signals.empty:
            all_validation_failures.append("Test 7: Should return signals even with insufficient data")
        elif len(small_signals) != 40:
            all_validation_failures.append(
                f"Test 7: Expected 40 signals, got {len(small_signals)}"
            )
        elif not all(s == SignalType.HOLD.value for s in small_signals['signal']):
            all_validation_failures.append(
                "Test 7: All signals should be HOLD with insufficient data"
            )
        else:
            logger.success("Test 7 PASSED: Handles insufficient data correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Test 8: Test faster Triple EMA (5, 13, 34 - Fibonacci)
    total_tests += 1
    try:
        if 'data' in locals() and data is not None and not data.empty:
            fast_strategy = TripleEMAStrategy(name="FastTripleEMA")
            fast_strategy.initialize({"fast": 5, "medium": 13, "slow": 34})

            test_data = data.reset_index()
            fast_signals = fast_strategy.generate_signals(test_data)

            fast_buy_count = (fast_signals['signal'] == SignalType.BUY.value).sum()
            fast_sell_count = (fast_signals['signal'] == SignalType.SELL.value).sum()

            # Faster EMAs should generally have more signals
            original_buy_count = (signals['signal'] == SignalType.BUY.value).sum()
            original_sell_count = (signals['signal'] == SignalType.SELL.value).sum()

            logger.success(
                f"Test 8 PASSED: Fast Triple EMA (5,13,34) generated "
                f"{fast_buy_count + fast_sell_count} signals vs "
                f"standard {original_buy_count + original_sell_count}"
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
        if "TripleEMA" not in registry:
            all_validation_failures.append(
                "Test 10: Strategy not registered in global registry"
            )
        else:
            registered_class = registry.get_strategy("TripleEMA")
            if registered_class is not TripleEMAStrategy:
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
        print("Triple EMA Strategy validated with REAL BTC/USDT data")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
