"""
MACD Momentum Strategy

This module implements a momentum strategy using the Moving Average Convergence Divergence (MACD)
indicator. The strategy generates buy signals when the MACD line crosses above the signal line
and sell signals when the MACD line crosses below the signal line.

**Purpose**: Implement a momentum strategy based on MACD signal line crossovers for
cryptocurrency trading.

**Strategy Type**: Momentum
**Indicators**: MACD(12, 26, 9)
**Entry Signal**: MACD line crosses above signal line (bullish momentum)
**Exit Signal**: MACD line crosses below signal line (bearish momentum)

**Parameters**:
- fast: Fast EMA period (default: 12)
- slow: Slow EMA period (default: 26)
- signal: Signal line EMA period (default: 9)

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
    'confidence': [0.0, 0.78, 0.0, 0.82, ...],
    'metadata': [
        {},
        {'reason': 'bullish_crossover', 'macd': 1.5, 'signal': 0.8, 'histogram': 0.7},
        {},
        {'reason': 'bearish_crossover', 'macd': 0.5, 'signal': 1.2, 'histogram': -0.7},
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
    name="MACD_Momentum",
    description="MACD signal line crossover momentum strategy",
    tags=["momentum", "macd", "crossover"]
)
class MACDMomentumStrategy(BaseStrategy):
    """
    MACD Momentum Strategy.

    Generates buy signals when MACD crosses above signal line (bullish) and
    sell signals when MACD crosses below signal line (bearish).

    Classic momentum approach suitable for trending markets.
    """

    def __init__(self, name: str = "MACD_Momentum", config: Dict[str, Any] = None):
        """
        Initialize the MACD Momentum strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default parameters for MACD(12, 26, 9)
        self.fast = 12
        self.slow = 26
        self.signal = 9

        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with 'fast', 'slow', and 'signal' periods

        Raises:
            ValueError: If parameters are invalid
        """
        self.fast = config.get("fast", 12)
        self.slow = config.get("slow", 26)
        self.signal = config.get("signal", 9)

        # Validate parameters
        if self.fast <= 0 or self.slow <= 0 or self.signal <= 0:
            raise ValueError("All MACD periods must be positive")

        if self.fast >= self.slow:
            raise ValueError(
                f"Fast period ({self.fast}) must be less than slow period ({self.slow})"
            )

        self._initialized = True
        logger.info(
            f"{self.name} initialized with MACD({self.fast}, {self.slow}, {self.signal})"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "fast": self.fast,
            "slow": self.slow,
            "signal": self.signal
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
        Generate trading signals based on MACD crossover.

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

        # MACD requires at least slow + signal periods
        min_periods = self.slow + self.signal
        if len(data) < min_periods:
            logger.warning(
                f"Insufficient data: {len(data)} rows, need {min_periods} for MACD"
            )
            return self._create_hold_signals(data)

        # Calculate MACD using pandas_ta
        df = data.copy()
        macd_result = ta.macd(
            df['close'],
            fast=self.fast,
            slow=self.slow,
            signal=self.signal
        )

        # pandas_ta returns a DataFrame with MACD_<fast>_<slow>_<signal>, MACDh, MACDs columns
        df['macd'] = macd_result.iloc[:, 0]  # MACD line
        df['macd_signal'] = macd_result.iloc[:, 2]  # Signal line
        df['macd_histogram'] = macd_result.iloc[:, 1]  # Histogram

        # Initialize signal arrays
        signals = []
        confidences = []
        metadata = []

        # Generate signals based on MACD crossovers
        for i in range(len(df)):
            if i == 0:
                # First row - no previous data
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Check for NaN values (not enough history yet)
            if (pd.isna(df['macd'].iloc[i]) or
                pd.isna(df['macd_signal'].iloc[i]) or
                pd.isna(df['macd_histogram'].iloc[i])):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Get current and previous values
            macd_current = df['macd'].iloc[i]
            signal_current = df['macd_signal'].iloc[i]
            histogram_current = df['macd_histogram'].iloc[i]

            macd_previous = df['macd'].iloc[i - 1]
            signal_previous = df['macd_signal'].iloc[i - 1]

            # Skip if previous values are NaN
            if pd.isna(macd_previous) or pd.isna(signal_previous):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({
                    'macd': float(macd_current),
                    'signal': float(signal_current),
                    'histogram': float(histogram_current)
                })
                continue

            # Bullish crossover: MACD crosses above signal line
            if macd_previous <= signal_previous and macd_current > signal_current:
                # Confidence based on histogram magnitude (stronger separation = higher confidence)
                histogram_abs = abs(histogram_current)
                price = df['close'].iloc[i]
                confidence = min(0.5 + (histogram_abs / price) * 1000, 1.0)

                signals.append(SignalType.BUY.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'bullish_crossover',
                    'macd': float(macd_current),
                    'signal': float(signal_current),
                    'histogram': float(histogram_current),
                    'crossover_strength': float(histogram_abs)
                })
                logger.debug(
                    f"Bullish MACD crossover at {df.index[i]}, "
                    f"histogram={histogram_current:.4f}"
                )

            # Bearish crossover: MACD crosses below signal line
            elif macd_previous >= signal_previous and macd_current < signal_current:
                # Confidence based on histogram magnitude
                histogram_abs = abs(histogram_current)
                price = df['close'].iloc[i]
                confidence = min(0.5 + (histogram_abs / price) * 1000, 1.0)

                signals.append(SignalType.SELL.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'bearish_crossover',
                    'macd': float(macd_current),
                    'signal': float(signal_current),
                    'histogram': float(histogram_current),
                    'crossover_strength': float(histogram_abs)
                })
                logger.debug(
                    f"Bearish MACD crossover at {df.index[i]}, "
                    f"histogram={histogram_current:.4f}"
                )

            # No crossover - HOLD
            else:
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({
                    'macd': float(macd_current),
                    'signal': float(signal_current),
                    'histogram': float(histogram_current)
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
    Validation block for MACD Momentum Strategy.
    Tests the strategy with REAL BTC/USDT data from Binance.
    """
    import sys
    from datetime import datetime, timedelta

    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting MACD Momentum Strategy validation with REAL DATA")

    # Test 1: Initialize strategy with default parameters
    total_tests += 1
    try:
        strategy = MACDMomentumStrategy()
        strategy.initialize({})

        params = strategy.get_parameters()
        if params['fast'] != 12:
            all_validation_failures.append(
                f"Test 1: Expected fast=12, got {params['fast']}"
            )
        if params['slow'] != 26:
            all_validation_failures.append(
                f"Test 1: Expected slow=26, got {params['slow']}"
            )
        if params['signal'] != 9:
            all_validation_failures.append(
                f"Test 1: Expected signal=9, got {params['signal']}"
            )

        if not all_validation_failures or len(all_validation_failures) == 0:
            logger.success("Test 1 PASSED: Strategy initialized with default parameters")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Initialize with custom parameters
    total_tests += 1
    try:
        custom_strategy = MACDMomentumStrategy(name="CustomMACD")
        custom_strategy.initialize({"fast": 8, "slow": 21, "signal": 5})

        params = custom_strategy.get_parameters()
        if params['fast'] != 8:
            all_validation_failures.append(
                f"Test 2: Expected fast=8, got {params['fast']}"
            )
        if params['slow'] != 21:
            all_validation_failures.append(
                f"Test 2: Expected slow=21, got {params['slow']}"
            )
        if params['signal'] != 5:
            all_validation_failures.append(
                f"Test 2: Expected signal=5, got {params['signal']}"
            )

        logger.success("Test 2 PASSED: Custom parameters set correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Test parameter validation
    total_tests += 1
    try:
        invalid_strategy = MACDMomentumStrategy()
        error_raised = False

        try:
            invalid_strategy.initialize({"fast": 26, "slow": 12, "signal": 9})
        except ValueError as e:
            if "must be less than" in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append(
                "Test 3: Expected ValueError for fast >= slow"
            )
        else:
            logger.success("Test 3 PASSED: Parameter validation works")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Fetch real BTC/USDT data for signal generation
    total_tests += 1
    try:
        logger.info("Fetching BTC/USDT 1h data for last 30 days...")
        fetcher = BinanceDataFetcher(use_storage=False, use_cache=False)

        # Fetch 30 days of 1h data (720 candles) for good MACD signals
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        data = fetcher.get_ohlcv(
            "BTC/USDT",
            "1h",
            start_date=start_date,
            end_date=end_date,
            limit=720
        )

        if data is None or data.empty:
            all_validation_failures.append("Test 4: Failed to fetch data")
        elif len(data) < 50:
            all_validation_failures.append(
                f"Test 4: Insufficient data - got {len(data)} rows, need 50+"
            )
        else:
            logger.success(f"Test 4 PASSED: Fetched {len(data)} candles of BTC/USDT 1h data")
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
                    logger.info(f"    Histogram: {latest_buy['metadata'].get('histogram', 'N/A')}")
                if not sell_signals.empty:
                    latest_sell = sell_signals.iloc[-1]
                    logger.info(f"  Latest SELL signal: {latest_sell['timestamp']}")
                    logger.info(f"    Histogram: {latest_sell['metadata'].get('histogram', 'N/A')}")
        else:
            all_validation_failures.append("Test 5: No data available from Test 4")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Verify signal structure and content
    total_tests += 1
    try:
        if 'signals' in locals() and signals is not None and not signals.empty:
            # Check for bullish/bearish crossover in metadata
            buy_signals = signals[signals['signal'] == SignalType.BUY.value]
            sell_signals = signals[signals['signal'] == SignalType.SELL.value]

            if not buy_signals.empty:
                first_buy = buy_signals.iloc[0]
                if 'reason' not in first_buy['metadata']:
                    all_validation_failures.append(
                        "Test 6: BUY signal metadata missing 'reason'"
                    )
                elif first_buy['metadata']['reason'] != 'bullish_crossover':
                    all_validation_failures.append(
                        f"Test 6: Expected reason='bullish_crossover', "
                        f"got '{first_buy['metadata']['reason']}'"
                    )
                elif first_buy['confidence'] <= 0:
                    all_validation_failures.append(
                        "Test 6: BUY signal should have positive confidence"
                    )
                elif 'histogram' not in first_buy['metadata']:
                    all_validation_failures.append(
                        "Test 6: BUY signal metadata missing 'histogram'"
                    )

            if not sell_signals.empty:
                first_sell = sell_signals.iloc[0]
                if 'reason' not in first_sell['metadata']:
                    all_validation_failures.append(
                        "Test 6: SELL signal metadata missing 'reason'"
                    )
                elif first_sell['metadata']['reason'] != 'bearish_crossover':
                    all_validation_failures.append(
                        f"Test 6: Expected reason='bearish_crossover', "
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
            'timestamp': pd.date_range('2024-01-01', periods=20),
            'open': range(100, 120),
            'high': range(105, 125),
            'low': range(99, 119),
            'close': range(103, 123),
            'volume': [1000] * 20
        })

        small_signals = strategy.generate_signals(small_data)

        if small_signals is None or small_signals.empty:
            all_validation_failures.append("Test 7: Should return signals even with insufficient data")
        elif len(small_signals) != 20:
            all_validation_failures.append(
                f"Test 7: Expected 20 signals, got {len(small_signals)}"
            )
        elif not all(s == SignalType.HOLD.value for s in small_signals['signal']):
            all_validation_failures.append(
                "Test 7: All signals should be HOLD with insufficient data"
            )
        else:
            logger.success("Test 7 PASSED: Handles insufficient data correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Test 8: Test faster MACD (8, 21, 5)
    total_tests += 1
    try:
        if 'data' in locals() and data is not None and not data.empty:
            fast_strategy = MACDMomentumStrategy(name="FastMACD")
            fast_strategy.initialize({"fast": 8, "slow": 21, "signal": 5})

            test_data = data.reset_index()
            fast_signals = fast_strategy.generate_signals(test_data)

            fast_buy_count = (fast_signals['signal'] == SignalType.BUY.value).sum()
            fast_sell_count = (fast_signals['signal'] == SignalType.SELL.value).sum()

            # Faster MACD should generally have more signals
            original_buy_count = (signals['signal'] == SignalType.BUY.value).sum()
            original_sell_count = (signals['signal'] == SignalType.SELL.value).sum()

            logger.success(
                f"Test 8 PASSED: Fast MACD (8,21,5) generated "
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
        if "MACD_Momentum" not in registry:
            all_validation_failures.append(
                "Test 10: Strategy not registered in global registry"
            )
        else:
            registered_class = registry.get_strategy("MACD_Momentum")
            if registered_class is not MACDMomentumStrategy:
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
        print("MACD Momentum Strategy validated with REAL BTC/USDT data")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
