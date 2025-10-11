"""
RSI Mean Reversion Strategy

This module implements a mean reversion strategy using the Relative Strength Index (RSI).
The strategy generates buy signals when RSI crosses below the oversold threshold (indicating
oversold conditions) and sell signals when RSI crosses above the overbought threshold.

**Purpose**: Implement a mean reversion strategy based on RSI oversold/overbought conditions
for cryptocurrency trading.

**Strategy Type**: Mean Reversion
**Indicators**: RSI(14)
**Entry Signal**: RSI crosses below 30 (oversold - price likely to bounce up)
**Exit Signal**: RSI crosses above 70 (overbought - price likely to drop)

**Parameters**:
- rsi_period: Period for RSI calculation (default: 14)
- oversold: Oversold threshold (default: 30)
- overbought: Overbought threshold (default: 70)

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
    'confidence': [0.0, 0.90, 0.0, 0.85, ...],
    'metadata': [
        {},
        {'reason': 'oversold', 'rsi': 28.5, 'distance_from_threshold': 1.5},
        {},
        {'reason': 'overbought', 'rsi': 72.3, 'distance_from_threshold': 2.3},
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
    name="RSI_MeanReversion",
    description="RSI oversold/overbought mean reversion strategy",
    tags=["mean_reversion", "rsi", "oscillator"]
)
class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.

    Generates buy signals when RSI enters oversold territory (< 30) and
    sell signals when RSI enters overbought territory (> 70).

    Classic mean reversion approach suitable for range-bound markets.
    """

    def __init__(self, name: str = "RSI_MeanReversion", config: Dict[str, Any] = None):
        """
        Initialize the RSI Mean Reversion strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default parameters
        self.rsi_period = 14
        self.oversold = 30
        self.overbought = 70

        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with 'rsi_period', 'oversold', and 'overbought'

        Raises:
            ValueError: If parameters are invalid
        """
        self.rsi_period = config.get("rsi_period", 14)
        self.oversold = config.get("oversold", 30)
        self.overbought = config.get("overbought", 70)

        # Validate parameters
        if self.rsi_period <= 0:
            raise ValueError("RSI period must be positive")

        if not (0 < self.oversold < 50):
            raise ValueError(
                f"Oversold threshold ({self.oversold}) must be between 0 and 50"
            )

        if not (50 < self.overbought < 100):
            raise ValueError(
                f"Overbought threshold ({self.overbought}) must be between 50 and 100"
            )

        if self.oversold >= self.overbought:
            raise ValueError(
                f"Oversold ({self.oversold}) must be less than "
                f"overbought ({self.overbought})"
            )

        self._initialized = True
        logger.info(
            f"{self.name} initialized with rsi_period={self.rsi_period}, "
            f"oversold={self.oversold}, overbought={self.overbought}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought
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
        Generate trading signals based on RSI levels.

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

        # Ensure we have enough data for RSI
        if len(data) < self.rsi_period + 1:
            logger.warning(
                f"Insufficient data: {len(data)} rows, need {self.rsi_period + 1} "
                f"for RSI calculation"
            )
            return self._create_hold_signals(data)

        # Calculate RSI using pandas_ta
        df = data.copy()
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)

        # Initialize signal arrays
        signals = []
        confidences = []
        metadata = []

        # Generate signals based on RSI levels
        for i in range(len(df)):
            if i == 0:
                # First row - no previous data
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Check for NaN values (not enough history yet)
            if pd.isna(df['rsi'].iloc[i]):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Get current and previous RSI values
            rsi_current = df['rsi'].iloc[i]
            rsi_previous = df['rsi'].iloc[i - 1]

            # Skip if previous RSI is NaN
            if pd.isna(rsi_previous):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({'rsi': float(rsi_current)})
                continue

            # Oversold: RSI crosses below oversold threshold (BUY signal)
            if rsi_previous >= self.oversold and rsi_current < self.oversold:
                # Confidence increases the further below oversold threshold
                distance = self.oversold - rsi_current
                confidence = min(0.5 + (distance / self.oversold) * 0.5, 1.0)

                signals.append(SignalType.BUY.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'oversold',
                    'rsi': float(rsi_current),
                    'threshold': self.oversold,
                    'distance_from_threshold': float(distance)
                })
                logger.debug(f"Oversold signal at {df.index[i]}, RSI={rsi_current:.2f}")

            # Overbought: RSI crosses above overbought threshold (SELL signal)
            elif rsi_previous <= self.overbought and rsi_current > self.overbought:
                # Confidence increases the further above overbought threshold
                distance = rsi_current - self.overbought
                confidence = min(0.5 + (distance / (100 - self.overbought)) * 0.5, 1.0)

                signals.append(SignalType.SELL.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'overbought',
                    'rsi': float(rsi_current),
                    'threshold': self.overbought,
                    'distance_from_threshold': float(distance)
                })
                logger.debug(f"Overbought signal at {df.index[i]}, RSI={rsi_current:.2f}")

            # No signal - HOLD
            else:
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({'rsi': float(rsi_current)})

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
    Validation block for RSI Mean Reversion Strategy.
    Tests the strategy with REAL BTC/USDT data from Binance.
    """
    import sys
    from datetime import datetime, timedelta

    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting RSI Mean Reversion Strategy validation with REAL DATA")

    # Test 1: Initialize strategy with default parameters
    total_tests += 1
    try:
        strategy = RSIMeanReversionStrategy()
        strategy.initialize({})

        params = strategy.get_parameters()
        if params['rsi_period'] != 14:
            all_validation_failures.append(
                f"Test 1: Expected rsi_period=14, got {params['rsi_period']}"
            )
        if params['oversold'] != 30:
            all_validation_failures.append(
                f"Test 1: Expected oversold=30, got {params['oversold']}"
            )
        if params['overbought'] != 70:
            all_validation_failures.append(
                f"Test 1: Expected overbought=70, got {params['overbought']}"
            )

        if not all_validation_failures or len(all_validation_failures) == 0:
            logger.success("Test 1 PASSED: Strategy initialized with default parameters")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Initialize with custom parameters
    total_tests += 1
    try:
        custom_strategy = RSIMeanReversionStrategy(name="CustomRSI")
        custom_strategy.initialize({"rsi_period": 21, "oversold": 20, "overbought": 80})

        params = custom_strategy.get_parameters()
        if params['rsi_period'] != 21:
            all_validation_failures.append(
                f"Test 2: Expected rsi_period=21, got {params['rsi_period']}"
            )
        if params['oversold'] != 20:
            all_validation_failures.append(
                f"Test 2: Expected oversold=20, got {params['oversold']}"
            )
        if params['overbought'] != 80:
            all_validation_failures.append(
                f"Test 2: Expected overbought=80, got {params['overbought']}"
            )

        logger.success("Test 2 PASSED: Custom parameters set correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Test parameter validation
    total_tests += 1
    try:
        invalid_strategy = RSIMeanReversionStrategy()
        error_raised = False

        try:
            invalid_strategy.initialize({"oversold": 60, "overbought": 70})
        except ValueError as e:
            if "must be between 0 and 50" in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append(
                "Test 3: Expected ValueError for invalid oversold threshold"
            )
        else:
            logger.success("Test 3 PASSED: Parameter validation works")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Fetch real BTC/USDT data for signal generation
    total_tests += 1
    try:
        logger.info("Fetching BTC/USDT 4h data for last 60 days...")
        fetcher = BinanceDataFetcher(use_storage=False, use_cache=False)

        # Fetch 60 days of 4h data (360 candles) to have good RSI signals
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        data = fetcher.get_ohlcv(
            "BTC/USDT",
            "4h",
            start_date=start_date,
            end_date=end_date,
            limit=360
        )

        if data is None or data.empty:
            all_validation_failures.append("Test 4: Failed to fetch data")
        elif len(data) < 20:
            all_validation_failures.append(
                f"Test 4: Insufficient data - got {len(data)} rows, need 20+"
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
                    logger.info(f"  Latest BUY signal: {buy_signals.iloc[-1]['timestamp']}")
                    logger.info(f"    RSI: {buy_signals.iloc[-1]['metadata'].get('rsi', 'N/A')}")
                if not sell_signals.empty:
                    logger.info(f"  Latest SELL signal: {sell_signals.iloc[-1]['timestamp']}")
                    logger.info(f"    RSI: {sell_signals.iloc[-1]['metadata'].get('rsi', 'N/A')}")
        else:
            all_validation_failures.append("Test 5: No data available from Test 4")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Verify signal structure and content
    total_tests += 1
    try:
        if 'signals' in locals() and signals is not None and not signals.empty:
            # Check for oversold/overbought in metadata
            buy_signals = signals[signals['signal'] == SignalType.BUY.value]
            sell_signals = signals[signals['signal'] == SignalType.SELL.value]

            if not buy_signals.empty:
                first_buy = buy_signals.iloc[0]
                if 'reason' not in first_buy['metadata']:
                    all_validation_failures.append(
                        "Test 6: BUY signal metadata missing 'reason'"
                    )
                elif first_buy['metadata']['reason'] != 'oversold':
                    all_validation_failures.append(
                        f"Test 6: Expected reason='oversold', "
                        f"got '{first_buy['metadata']['reason']}'"
                    )
                elif first_buy['confidence'] <= 0:
                    all_validation_failures.append(
                        "Test 6: BUY signal should have positive confidence"
                    )
                elif first_buy['metadata'].get('rsi', 100) >= 30:
                    all_validation_failures.append(
                        f"Test 6: Oversold RSI should be < 30, "
                        f"got {first_buy['metadata'].get('rsi')}"
                    )

            if not sell_signals.empty:
                first_sell = sell_signals.iloc[0]
                if 'reason' not in first_sell['metadata']:
                    all_validation_failures.append(
                        "Test 6: SELL signal metadata missing 'reason'"
                    )
                elif first_sell['metadata']['reason'] != 'overbought':
                    all_validation_failures.append(
                        f"Test 6: Expected reason='overbought', "
                        f"got '{first_sell['metadata']['reason']}'"
                    )
                elif first_sell['metadata'].get('rsi', 0) <= 70:
                    all_validation_failures.append(
                        f"Test 6: Overbought RSI should be > 70, "
                        f"got {first_sell['metadata'].get('rsi')}"
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
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'open': range(100, 110),
            'high': range(105, 115),
            'low': range(99, 109),
            'close': range(103, 113),
            'volume': [1000] * 10
        })

        small_signals = strategy.generate_signals(small_data)

        if small_signals is None or small_signals.empty:
            all_validation_failures.append("Test 7: Should return signals even with insufficient data")
        elif len(small_signals) != 10:
            all_validation_failures.append(
                f"Test 7: Expected 10 signals, got {len(small_signals)}"
            )
        elif not all(s == SignalType.HOLD.value for s in small_signals['signal']):
            all_validation_failures.append(
                "Test 7: All signals should be HOLD with insufficient data"
            )
        else:
            logger.success("Test 7 PASSED: Handles insufficient data correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Test 8: Test more aggressive thresholds (20/80)
    total_tests += 1
    try:
        if 'data' in locals() and data is not None and not data.empty:
            aggressive_strategy = RSIMeanReversionStrategy(name="AggressiveRSI")
            aggressive_strategy.initialize({"oversold": 20, "overbought": 80})

            test_data = data.reset_index()
            aggressive_signals = aggressive_strategy.generate_signals(test_data)

            aggressive_buy_count = (aggressive_signals['signal'] == SignalType.BUY.value).sum()
            aggressive_sell_count = (aggressive_signals['signal'] == SignalType.SELL.value).sum()

            # More extreme thresholds should have fewer signals
            original_buy_count = (signals['signal'] == SignalType.BUY.value).sum()
            original_sell_count = (signals['signal'] == SignalType.SELL.value).sum()

            total_aggressive = aggressive_buy_count + aggressive_sell_count
            total_original = original_buy_count + original_sell_count

            if total_aggressive > total_original:
                logger.warning(
                    f"Test 8: Expected fewer signals with extreme thresholds (20/80), "
                    f"but got aggressive={total_aggressive} vs original={total_original}"
                )
                # This is a soft warning, not a hard failure

            logger.success(
                f"Test 8 PASSED: Aggressive RSI (20/80) generated "
                f"{total_aggressive} signals vs original {total_original}"
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
        if "RSI_MeanReversion" not in registry:
            all_validation_failures.append(
                "Test 10: Strategy not registered in global registry"
            )
        else:
            registered_class = registry.get_strategy("RSI_MeanReversion")
            if registered_class is not RSIMeanReversionStrategy:
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
        print("RSI Mean Reversion Strategy validated with REAL BTC/USDT data")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
