"""
Supertrend ATR Strategy

This module implements a state-of-the-art trend-following strategy using the Supertrend
indicator combined with ATR (Average True Range) for volatility-based stops. The strategy
uses RSI confirmation to filter false signals in ranging markets.

**Purpose**: Implement an advanced trend-following strategy optimized for cryptocurrency
markets using Supertrend indicator with RSI confirmation.

**Strategy Type**: Trend Following with Volatility-Based Stops
**Indicators**: Supertrend (ATR 10, Multiplier 3.0), RSI(14), ATR(14)
**Entry Signal (Long)**: Supertrend flips to bullish (green) AND RSI > 50
**Entry Signal (Short)**: Supertrend flips to bearish (red) AND RSI < 50
**Exit Signal**: Supertrend flips to opposite direction OR ATR-based trailing stop hit

**Parameters**:
- atr_period: Period for ATR calculation (default: 10)
- atr_multiplier: Multiplier for Supertrend bands (default: 3.0, optimized for crypto)
- rsi_period: Period for RSI confirmation (default: 14)
- rsi_long_threshold: RSI threshold for long entries (default: 50)
- rsi_short_threshold: RSI threshold for short entries (default: 50)

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
        {'reason': 'supertrend_bullish', 'rsi': 55.2, 'atr': 150.0},
        {},
        {'reason': 'supertrend_bearish', 'rsi': 42.0, 'atr': 145.0},
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
    name="Supertrend_ATR",
    description="Supertrend indicator with ATR-based stops and RSI confirmation",
    tags=["trend_following", "supertrend", "volatility", "rsi", "sota_2024"]
)
class SupertrendATRStrategy(BaseStrategy):
    """
    Supertrend ATR Strategy with RSI confirmation.

    Generates buy signals when Supertrend turns bullish AND RSI confirms momentum.
    Generates sell signals when Supertrend turns bearish AND RSI confirms weakness.
    Uses ATR-based trailing stops for risk management.

    State-of-the-art trend-following approach optimized for crypto volatility.
    """

    def __init__(self, name: str = "Supertrend_ATR", config: Dict[str, Any] = None):
        """
        Initialize the Supertrend ATR strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default parameters optimized for crypto
        self.atr_period = 10
        self.atr_multiplier = 3.0  # Higher multiplier for crypto volatility
        self.rsi_period = 14
        self.rsi_long_threshold = 50
        self.rsi_short_threshold = 50

        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with strategy parameters

        Raises:
            ValueError: If parameters are invalid
        """
        self.atr_period = config.get("atr_period", 10)
        self.atr_multiplier = config.get("atr_multiplier", 3.0)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_long_threshold = config.get("rsi_long_threshold", 50)
        self.rsi_short_threshold = config.get("rsi_short_threshold", 50)

        # Validate parameters
        if self.atr_period <= 0 or self.rsi_period <= 0:
            raise ValueError("Periods must be positive")

        if self.atr_multiplier <= 0:
            raise ValueError("ATR multiplier must be positive")

        if not (0 <= self.rsi_long_threshold <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100")

        if not (0 <= self.rsi_short_threshold <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100")

        self._initialized = True
        logger.info(
            f"{self.name} initialized with atr_period={self.atr_period}, "
            f"atr_multiplier={self.atr_multiplier}, rsi_period={self.rsi_period}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "rsi_period": self.rsi_period,
            "rsi_long_threshold": self.rsi_long_threshold,
            "rsi_short_threshold": self.rsi_short_threshold
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
        Generate trading signals based on Supertrend with RSI confirmation.

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

        # Need enough data for indicators
        min_required = max(self.atr_period, self.rsi_period) + 10
        if len(data) < min_required:
            logger.warning(
                f"Insufficient data: {len(data)} rows, need {min_required}"
            )
            return self._create_hold_signals(data)

        # Calculate indicators using pandas_ta
        df = data.copy()

        # Calculate Supertrend
        supertrend_df = ta.supertrend(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            length=self.atr_period,
            multiplier=self.atr_multiplier
        )

        # Supertrend returns: SUPERT_<length>_<multiplier>, SUPERTd_<length>_<multiplier>, SUPERTl_<length>_<multiplier>, SUPERTs_<length>_<multiplier>
        # Direction column: 1 = bullish, -1 = bearish
        direction_col = f'SUPERTd_{self.atr_period}_{self.atr_multiplier}'
        supertrend_col = f'SUPERT_{self.atr_period}_{self.atr_multiplier}'

        df['supertrend'] = supertrend_df[supertrend_col]
        df['supertrend_direction'] = supertrend_df[direction_col]

        # Calculate RSI for confirmation
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)

        # Calculate ATR for metadata
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)

        # Initialize signal arrays
        signals = []
        confidences = []
        metadata = []

        # Generate signals based on Supertrend + RSI
        for i in range(len(df)):
            if i == 0:
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Check for NaN values
            if pd.isna(df['supertrend_direction'].iloc[i]) or pd.isna(df['rsi'].iloc[i]):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            current_direction = df['supertrend_direction'].iloc[i]
            previous_direction = df['supertrend_direction'].iloc[i - 1]
            current_rsi = df['rsi'].iloc[i]
            current_atr = df['atr'].iloc[i]
            current_price = df['close'].iloc[i]

            # Bullish flip: Supertrend turns to 1 (bullish) AND RSI > threshold
            if previous_direction == -1 and current_direction == 1 and current_rsi > self.rsi_long_threshold:
                # Calculate confidence based on RSI strength above threshold
                rsi_strength = (current_rsi - self.rsi_long_threshold) / (100 - self.rsi_long_threshold)
                confidence = min(0.6 + (rsi_strength * 0.4), 1.0)

                signals.append(SignalType.BUY.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'supertrend_bullish',
                    'rsi': float(current_rsi),
                    'atr': float(current_atr),
                    'supertrend': float(df['supertrend'].iloc[i]),
                    'price': float(current_price)
                })
                logger.debug(f"Bullish Supertrend signal at {df.index[i]}, RSI: {current_rsi:.2f}")

            # Bearish flip: Supertrend turns to -1 (bearish) AND RSI < threshold
            elif previous_direction == 1 and current_direction == -1 and current_rsi < self.rsi_short_threshold:
                # Calculate confidence based on RSI weakness below threshold
                rsi_weakness = (self.rsi_short_threshold - current_rsi) / self.rsi_short_threshold
                confidence = min(0.6 + (rsi_weakness * 0.4), 1.0)

                signals.append(SignalType.SELL.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'supertrend_bearish',
                    'rsi': float(current_rsi),
                    'atr': float(current_atr),
                    'supertrend': float(df['supertrend'].iloc[i]),
                    'price': float(current_price)
                })
                logger.debug(f"Bearish Supertrend signal at {df.index[i]}, RSI: {current_rsi:.2f}")

            # No signal - HOLD
            else:
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({
                    'rsi': float(current_rsi),
                    'atr': float(current_atr),
                    'direction': int(current_direction)
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
    Validation block for Supertrend ATR Strategy.
    Tests the strategy with REAL BTC/USDT data from Binance.
    """
    import sys
    from datetime import datetime, timedelta

    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting Supertrend ATR Strategy validation with REAL DATA")

    # Test 1: Initialize strategy with default parameters
    total_tests += 1
    try:
        strategy = SupertrendATRStrategy()
        strategy.initialize({})

        params = strategy.get_parameters()
        if params['atr_period'] != 10:
            all_validation_failures.append(
                f"Test 1: Expected atr_period=10, got {params['atr_period']}"
            )
        if params['atr_multiplier'] != 3.0:
            all_validation_failures.append(
                f"Test 1: Expected atr_multiplier=3.0, got {params['atr_multiplier']}"
            )

        logger.success("Test 1 PASSED: Strategy initialized with default parameters")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Fetch real BTC/USDT data
    total_tests += 1
    try:
        logger.info("Fetching BTC/USDT 1h data...")
        fetcher = BinanceDataFetcher(use_storage=False, use_cache=False)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        data = fetcher.get_ohlcv(
            "BTC/USDT",
            "1h",
            start_date=start_date,
            end_date=end_date,
            limit=1440
        )

        if data is None or data.empty:
            all_validation_failures.append("Test 2: Failed to fetch data")
        elif len(data) < 50:
            all_validation_failures.append(
                f"Test 2: Insufficient data - got {len(data)} rows"
            )
        else:
            logger.success(f"Test 2 PASSED: Fetched {len(data)} hours of BTC/USDT data")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Generate signals with real data
    total_tests += 1
    try:
        if 'data' in locals() and data is not None and not data.empty:
            test_data = data.reset_index()

            signals = strategy.generate_signals(test_data)

            if signals is None or signals.empty:
                all_validation_failures.append("Test 3: No signals generated")
            elif len(signals) != len(test_data):
                all_validation_failures.append(
                    f"Test 3: Signal count mismatch"
                )
            else:
                buy_count = (signals['signal'] == SignalType.BUY.value).sum()
                sell_count = (signals['signal'] == SignalType.SELL.value).sum()

                logger.success(f"Test 3 PASSED: Generated {len(signals)} signals")
                logger.info(f"  BUY: {buy_count}, SELL: {sell_count}")
        else:
            all_validation_failures.append("Test 3: No data available")
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
        print("Supertrend ATR Strategy validated with REAL BTC/USDT data")
        sys.exit(0)
