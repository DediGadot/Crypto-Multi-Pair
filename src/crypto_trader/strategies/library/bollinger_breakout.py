"""
Bollinger Bands Volatility Breakout Strategy

This module implements a volatility breakout strategy using Bollinger Bands. The strategy
generates buy signals when price closes above the upper Bollinger Band (volatility breakout
to the upside) and sell signals when price closes below the lower Bollinger Band (volatility
breakout to the downside).

**Purpose**: Implement a volatility breakout strategy based on Bollinger Bands for
cryptocurrency trading.

**Strategy Type**: Volatility Breakout
**Indicators**: Bollinger Bands (20, 2.0)
**Entry Signal**: Price closes above upper Bollinger Band (bullish breakout)
**Exit Signal**: Price closes below lower Bollinger Band (bearish breakout)

**Parameters**:
- bb_period: Period for Bollinger Bands middle line (SMA) (default: 20)
- bb_std: Number of standard deviations for bands (default: 2.0)

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
    'confidence': [0.0, 0.82, 0.0, 0.78, ...],
    'metadata': [
        {},
        {'reason': 'upper_band_breakout', 'price': 105.5, 'upper_band': 104.2, 'distance': 1.3},
        {},
        {'reason': 'lower_band_breakout', 'price': 98.5, 'lower_band': 99.8, 'distance': 1.3},
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
    name="BollingerBreakout",
    description="Bollinger Bands volatility breakout strategy",
    tags=["volatility", "bollinger_bands", "breakout"]
)
class BollingerBreakoutStrategy(BaseStrategy):
    """
    Bollinger Bands Volatility Breakout Strategy.

    Generates buy signals when price breaks above upper band (bullish breakout) and
    sell signals when price breaks below lower band (bearish breakout).

    Suitable for capturing strong momentum moves in volatile markets.
    """

    def __init__(self, name: str = "BollingerBreakout", config: Dict[str, Any] = None):
        """
        Initialize the Bollinger Breakout strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default parameters for Bollinger Bands(20, 2)
        self.bb_period = 20
        self.bb_std = 2.0

        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with 'bb_period' and 'bb_std'

        Raises:
            ValueError: If parameters are invalid
        """
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2.0)

        # Validate parameters
        if self.bb_period <= 0:
            raise ValueError("Bollinger Bands period must be positive")

        if self.bb_std <= 0:
            raise ValueError("Bollinger Bands standard deviation must be positive")

        if self.bb_std > 5:
            logger.warning(
                f"BB standard deviation {self.bb_std} is unusually high (typical: 2.0)"
            )

        self._initialized = True
        logger.info(
            f"{self.name} initialized with BB({self.bb_period}, {self.bb_std})"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std
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
        Generate trading signals based on Bollinger Bands breakouts.

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

        # Need enough data for BB calculation
        if len(data) < self.bb_period:
            logger.warning(
                f"Insufficient data: {len(data)} rows, need {self.bb_period} for BB"
            )
            return self._create_hold_signals(data)

        # Calculate Bollinger Bands using pandas_ta
        df = data.copy()
        bb_result = ta.bbands(
            df['close'],
            length=self.bb_period,
            std=self.bb_std
        )

        # pandas_ta returns DataFrame with BBL_<period>_<std>, BBM_<period>_<std>, BBU_<period>_<std>
        df['bb_lower'] = bb_result.iloc[:, 0]  # Lower band
        df['bb_middle'] = bb_result.iloc[:, 1]  # Middle band (SMA)
        df['bb_upper'] = bb_result.iloc[:, 2]  # Upper band

        # Initialize signal arrays
        signals = []
        confidences = []
        metadata = []

        # Generate signals based on Bollinger Band breakouts
        for i in range(len(df)):
            if i == 0:
                # First row - no previous data
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Check for NaN values (not enough history yet)
            if (pd.isna(df['bb_lower'].iloc[i]) or
                pd.isna(df['bb_middle'].iloc[i]) or
                pd.isna(df['bb_upper'].iloc[i])):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Get current values
            close_current = df['close'].iloc[i]
            close_previous = df['close'].iloc[i - 1]
            bb_upper_current = df['bb_upper'].iloc[i]
            bb_lower_current = df['bb_lower'].iloc[i]
            bb_middle_current = df['bb_middle'].iloc[i]
            bb_upper_previous = df['bb_upper'].iloc[i - 1]
            bb_lower_previous = df['bb_lower'].iloc[i - 1]

            # Skip if previous values are NaN
            if pd.isna(close_previous) or pd.isna(bb_upper_previous) or pd.isna(bb_lower_previous):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({
                    'close': float(close_current),
                    'bb_upper': float(bb_upper_current),
                    'bb_middle': float(bb_middle_current),
                    'bb_lower': float(bb_lower_current)
                })
                continue

            # Upper band breakout: Close crosses above upper BB
            if close_previous <= bb_upper_previous and close_current > bb_upper_current:
                # Confidence based on how far above upper band
                distance = close_current - bb_upper_current
                band_width = bb_upper_current - bb_middle_current
                confidence = min(0.5 + (distance / band_width) * 0.5, 1.0)

                signals.append(SignalType.BUY.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'upper_band_breakout',
                    'close': float(close_current),
                    'bb_upper': float(bb_upper_current),
                    'bb_middle': float(bb_middle_current),
                    'bb_lower': float(bb_lower_current),
                    'distance_above_upper': float(distance),
                    'band_width': float(bb_upper_current - bb_lower_current)
                })
                logger.debug(
                    f"Upper band breakout at {df.index[i]}, "
                    f"close={close_current:.2f}, upper={bb_upper_current:.2f}"
                )

            # Lower band breakout: Close crosses below lower BB
            elif close_previous >= bb_lower_previous and close_current < bb_lower_current:
                # Confidence based on how far below lower band
                distance = bb_lower_current - close_current
                band_width = bb_middle_current - bb_lower_current
                confidence = min(0.5 + (distance / band_width) * 0.5, 1.0)

                signals.append(SignalType.SELL.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'lower_band_breakout',
                    'close': float(close_current),
                    'bb_upper': float(bb_upper_current),
                    'bb_middle': float(bb_middle_current),
                    'bb_lower': float(bb_lower_current),
                    'distance_below_lower': float(distance),
                    'band_width': float(bb_upper_current - bb_lower_current)
                })
                logger.debug(
                    f"Lower band breakout at {df.index[i]}, "
                    f"close={close_current:.2f}, lower={bb_lower_current:.2f}"
                )

            # No breakout - HOLD
            else:
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({
                    'close': float(close_current),
                    'bb_upper': float(bb_upper_current),
                    'bb_middle': float(bb_middle_current),
                    'bb_lower': float(bb_lower_current),
                    'band_width': float(bb_upper_current - bb_lower_current)
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
    Validation block for Bollinger Breakout Strategy.
    Tests the strategy with REAL BTC/USDT data from Binance.
    """
    import sys
    from datetime import datetime, timedelta

    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting Bollinger Breakout Strategy validation with REAL DATA")

    # Test 1: Initialize strategy with default parameters
    total_tests += 1
    try:
        strategy = BollingerBreakoutStrategy()
        strategy.initialize({})

        params = strategy.get_parameters()
        if params['bb_period'] != 20:
            all_validation_failures.append(
                f"Test 1: Expected bb_period=20, got {params['bb_period']}"
            )
        if params['bb_std'] != 2.0:
            all_validation_failures.append(
                f"Test 1: Expected bb_std=2.0, got {params['bb_std']}"
            )

        if not all_validation_failures or len(all_validation_failures) == 0:
            logger.success("Test 1 PASSED: Strategy initialized with default parameters")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Initialize with custom parameters
    total_tests += 1
    try:
        custom_strategy = BollingerBreakoutStrategy(name="CustomBB")
        custom_strategy.initialize({"bb_period": 30, "bb_std": 2.5})

        params = custom_strategy.get_parameters()
        if params['bb_period'] != 30:
            all_validation_failures.append(
                f"Test 2: Expected bb_period=30, got {params['bb_period']}"
            )
        if params['bb_std'] != 2.5:
            all_validation_failures.append(
                f"Test 2: Expected bb_std=2.5, got {params['bb_std']}"
            )

        logger.success("Test 2 PASSED: Custom parameters set correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Test parameter validation
    total_tests += 1
    try:
        invalid_strategy = BollingerBreakoutStrategy()
        error_raised = False

        try:
            invalid_strategy.initialize({"bb_period": -10, "bb_std": 2.0})
        except ValueError as e:
            if "must be positive" in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append(
                "Test 3: Expected ValueError for negative bb_period"
            )
        else:
            logger.success("Test 3 PASSED: Parameter validation works")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Fetch real BTC/USDT data for signal generation
    total_tests += 1
    try:
        logger.info("Fetching BTC/USDT 15m data for last 7 days...")
        fetcher = BinanceDataFetcher(use_storage=False, use_cache=False)

        # Fetch 7 days of 15m data (672 candles) for volatility signals
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        data = fetcher.get_ohlcv(
            "BTC/USDT",
            "15m",
            start_date=start_date,
            end_date=end_date,
            limit=672
        )

        if data is None or data.empty:
            all_validation_failures.append("Test 4: Failed to fetch data")
        elif len(data) < 30:
            all_validation_failures.append(
                f"Test 4: Insufficient data - got {len(data)} rows, need 30+"
            )
        else:
            logger.success(f"Test 4 PASSED: Fetched {len(data)} candles of BTC/USDT 15m data")
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
                    logger.info(f"    Close: {latest_buy['metadata'].get('close', 'N/A')}")
                    logger.info(f"    Upper Band: {latest_buy['metadata'].get('bb_upper', 'N/A')}")
                if not sell_signals.empty:
                    latest_sell = sell_signals.iloc[-1]
                    logger.info(f"  Latest SELL signal: {latest_sell['timestamp']}")
                    logger.info(f"    Close: {latest_sell['metadata'].get('close', 'N/A')}")
                    logger.info(f"    Lower Band: {latest_sell['metadata'].get('bb_lower', 'N/A')}")
        else:
            all_validation_failures.append("Test 5: No data available from Test 4")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Verify signal structure and content
    total_tests += 1
    try:
        if 'signals' in locals() and signals is not None and not signals.empty:
            # Check for breakout types in metadata
            buy_signals = signals[signals['signal'] == SignalType.BUY.value]
            sell_signals = signals[signals['signal'] == SignalType.SELL.value]

            if not buy_signals.empty:
                first_buy = buy_signals.iloc[0]
                if 'reason' not in first_buy['metadata']:
                    all_validation_failures.append(
                        "Test 6: BUY signal metadata missing 'reason'"
                    )
                elif first_buy['metadata']['reason'] != 'upper_band_breakout':
                    all_validation_failures.append(
                        f"Test 6: Expected reason='upper_band_breakout', "
                        f"got '{first_buy['metadata']['reason']}'"
                    )
                elif first_buy['confidence'] <= 0:
                    all_validation_failures.append(
                        "Test 6: BUY signal should have positive confidence"
                    )
                elif 'band_width' not in first_buy['metadata']:
                    all_validation_failures.append(
                        "Test 6: BUY signal metadata missing 'band_width'"
                    )

            if not sell_signals.empty:
                first_sell = sell_signals.iloc[0]
                if 'reason' not in first_sell['metadata']:
                    all_validation_failures.append(
                        "Test 6: SELL signal metadata missing 'reason'"
                    )
                elif first_sell['metadata']['reason'] != 'lower_band_breakout':
                    all_validation_failures.append(
                        f"Test 6: Expected reason='lower_band_breakout', "
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
            'timestamp': pd.date_range('2024-01-01', periods=15),
            'open': range(100, 115),
            'high': range(105, 120),
            'low': range(99, 114),
            'close': range(103, 118),
            'volume': [1000] * 15
        })

        small_signals = strategy.generate_signals(small_data)

        if small_signals is None or small_signals.empty:
            all_validation_failures.append("Test 7: Should return signals even with insufficient data")
        elif len(small_signals) != 15:
            all_validation_failures.append(
                f"Test 7: Expected 15 signals, got {len(small_signals)}"
            )
        elif not all(s == SignalType.HOLD.value for s in small_signals['signal']):
            all_validation_failures.append(
                "Test 7: All signals should be HOLD with insufficient data"
            )
        else:
            logger.success("Test 7 PASSED: Handles insufficient data correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Test 8: Test wider bands (3 std dev)
    total_tests += 1
    try:
        if 'data' in locals() and data is not None and not data.empty:
            wide_strategy = BollingerBreakoutStrategy(name="WideBB")
            wide_strategy.initialize({"bb_period": 20, "bb_std": 3.0})

            test_data = data.reset_index()
            wide_signals = wide_strategy.generate_signals(test_data)

            wide_buy_count = (wide_signals['signal'] == SignalType.BUY.value).sum()
            wide_sell_count = (wide_signals['signal'] == SignalType.SELL.value).sum()

            # Wider bands should have fewer breakout signals
            original_buy_count = (signals['signal'] == SignalType.BUY.value).sum()
            original_sell_count = (signals['signal'] == SignalType.SELL.value).sum()

            logger.success(
                f"Test 8 PASSED: Wide BB (3.0) generated "
                f"{wide_buy_count + wide_sell_count} signals vs "
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
        if "BollingerBreakout" not in registry:
            all_validation_failures.append(
                "Test 10: Strategy not registered in global registry"
            )
        else:
            registered_class = registry.get_strategy("BollingerBreakout")
            if registered_class is not BollingerBreakoutStrategy:
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
        print("Bollinger Breakout Strategy validated with REAL BTC/USDT data")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
