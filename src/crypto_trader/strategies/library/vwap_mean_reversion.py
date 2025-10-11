"""
VWAP Mean Reversion Strategy

This module implements a volume-weighted price action strategy that exploits mean reversion
around the VWAP (Volume Weighted Average Price). The strategy uses standard deviation bands
and RSI confirmation to identify price extremes likely to revert to the mean.

**Purpose**: Implement a volume-based mean reversion strategy optimized for intraday
cryptocurrency trading using institutional benchmark (VWAP).

**Strategy Type**: Volume-Weighted Mean Reversion
**Indicators**: VWAP (daily anchor), Standard Deviation Bands (±2 std dev), RSI(14)
**Entry Signal (Long)**: Price touches lower band (-2 std dev) AND RSI < 30 (oversold)
**Entry Signal (Short)**: Price touches upper band (+2 std dev) AND RSI > 70 (overbought)
**Exit Signal**: Price returns to VWAP OR RSI crosses back through thresholds

**Parameters**:
- vwap_anchor: VWAP calculation anchor period (default: 'D' for daily)
- std_dev_multiplier: Standard deviation multiplier for bands (default: 2.0)
- rsi_period: Period for RSI oscillator (default: 14)
- rsi_oversold: RSI threshold for oversold (default: 30)
- rsi_overbought: RSI threshold for overbought (default: 70)

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- pandas_ta: https://github.com/twopirllc/pandas-ta
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```python
data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
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
        {'reason': 'vwap_oversold', 'rsi': 28.5, 'distance_from_vwap': -2.1},
        {},
        {'reason': 'vwap_overbought', 'rsi': 72.0, 'distance_from_vwap': 2.3},
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
    name="VWAP_MeanReversion",
    description="VWAP-based mean reversion with standard deviation bands and RSI",
    tags=["mean_reversion", "vwap", "volume", "rsi", "sota_2024"]
)
class VWAPMeanReversionStrategy(BaseStrategy):
    """
    VWAP Mean Reversion Strategy.

    Generates buy signals when price hits lower band (undervalued) with RSI oversold.
    Generates sell signals when price hits upper band (overvalued) with RSI overbought.
    Uses VWAP as fair value benchmark and standard deviation bands for entry zones.

    Volume-weighted approach provides institutional-grade fair value estimation.
    """

    def __init__(self, name: str = "VWAP_MeanReversion", config: Dict[str, Any] = None):
        """
        Initialize the VWAP Mean Reversion strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default parameters
        self.vwap_anchor = 'D'  # Daily VWAP
        self.std_dev_multiplier = 2.0
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70

        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with strategy parameters

        Raises:
            ValueError: If parameters are invalid
        """
        self.vwap_anchor = config.get("vwap_anchor", 'D')
        self.std_dev_multiplier = config.get("std_dev_multiplier", 2.0)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)

        # Validate parameters
        if self.std_dev_multiplier <= 0:
            raise ValueError("Standard deviation multiplier must be positive")

        if self.rsi_period <= 0:
            raise ValueError("RSI period must be positive")

        if not (0 <= self.rsi_oversold <= 100) or not (0 <= self.rsi_overbought <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100")

        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("RSI oversold must be less than overbought")

        self._initialized = True
        logger.info(
            f"{self.name} initialized with std_dev={self.std_dev_multiplier}, "
            f"RSI oversold={self.rsi_oversold}, overbought={self.rsi_overbought}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "vwap_anchor": self.vwap_anchor,
            "std_dev_multiplier": self.std_dev_multiplier,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought
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
        Generate trading signals based on VWAP mean reversion.

        Args:
            data: DataFrame with OHLCV data and DatetimeIndex

        Returns:
            DataFrame with columns: ['timestamp', 'signal', 'confidence', 'metadata']

        Raises:
            ValueError: If data is invalid
        """
        # Validate data
        if not self.validate_data(data):
            raise ValueError("Invalid data provided to generate_signals")

        # Need enough data for indicators
        min_required = max(self.rsi_period + 10, 50)
        if len(data) < min_required:
            logger.warning(
                f"Insufficient data: {len(data)} rows, need {min_required}"
            )
            return self._create_hold_signals(data)

        # Ensure we have DatetimeIndex for VWAP calculation
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            else:
                logger.warning("Cannot calculate VWAP without DatetimeIndex")
                return self._create_hold_signals(data)

        # Calculate VWAP with daily anchor using pandas_ta
        try:
            df['vwap'] = ta.vwap(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                anchor=self.vwap_anchor
            )
        except Exception as e:
            logger.warning(f"VWAP calculation failed: {e}, using simple average")
            # Fallback to volume-weighted moving average
            df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

        # Calculate standard deviation of price around VWAP
        df['price_vwap_diff'] = df['close'] - df['vwap']
        df['vwap_std'] = df['price_vwap_diff'].rolling(window=20).std()

        # Calculate bands
        df['vwap_upper'] = df['vwap'] + (self.std_dev_multiplier * df['vwap_std'])
        df['vwap_lower'] = df['vwap'] - (self.std_dev_multiplier * df['vwap_std'])

        # Calculate RSI for confirmation
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)

        # Calculate distance from VWAP in standard deviations
        df['vwap_distance'] = df['price_vwap_diff'] / df['vwap_std'].replace(0, 1)

        # Initialize signal arrays
        signals = []
        confidences = []
        metadata = []

        # Generate signals based on VWAP mean reversion
        for i in range(len(df)):
            # Check for NaN values
            if any(pd.isna(df[col].iloc[i]) for col in ['vwap', 'vwap_lower', 'vwap_upper', 'rsi']):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            current_price = df['close'].iloc[i]
            vwap = df['vwap'].iloc[i]
            vwap_lower = df['vwap_lower'].iloc[i]
            vwap_upper = df['vwap_upper'].iloc[i]
            current_rsi = df['rsi'].iloc[i]
            vwap_distance = df['vwap_distance'].iloc[i]

            # Long Entry: Price at/below lower band AND RSI oversold
            # Price is "cheap" relative to volume-weighted average
            if current_price <= vwap_lower and current_rsi < self.rsi_oversold:
                # Calculate confidence based on RSI extremity and distance from VWAP
                rsi_strength = (self.rsi_oversold - current_rsi) / self.rsi_oversold
                distance_strength = min(abs(vwap_distance) / self.std_dev_multiplier, 1.0)
                confidence = min(0.6 + (rsi_strength * 0.2) + (distance_strength * 0.2), 1.0)

                signals.append(SignalType.BUY.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'vwap_oversold',
                    'rsi': float(current_rsi),
                    'vwap': float(vwap),
                    'price': float(current_price),
                    'distance_from_vwap': float(vwap_distance),
                    'lower_band': float(vwap_lower)
                })
                logger.debug(
                    f"VWAP Oversold signal at {df.index[i]}, "
                    f"RSI: {current_rsi:.2f}, Distance: {vwap_distance:.2f}σ"
                )

            # Short Entry: Price at/above upper band AND RSI overbought
            # Price is "expensive" relative to volume-weighted average
            elif current_price >= vwap_upper and current_rsi > self.rsi_overbought:
                # Calculate confidence based on RSI extremity and distance from VWAP
                rsi_strength = (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
                distance_strength = min(abs(vwap_distance) / self.std_dev_multiplier, 1.0)
                confidence = min(0.6 + (rsi_strength * 0.2) + (distance_strength * 0.2), 1.0)

                signals.append(SignalType.SELL.value)
                confidences.append(confidence)
                metadata.append({
                    'reason': 'vwap_overbought',
                    'rsi': float(current_rsi),
                    'vwap': float(vwap),
                    'price': float(current_price),
                    'distance_from_vwap': float(vwap_distance),
                    'upper_band': float(vwap_upper)
                })
                logger.debug(
                    f"VWAP Overbought signal at {df.index[i]}, "
                    f"RSI: {current_rsi:.2f}, Distance: {vwap_distance:.2f}σ"
                )

            # No signal - HOLD
            else:
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({
                    'rsi': float(current_rsi),
                    'vwap': float(vwap),
                    'distance_from_vwap': float(vwap_distance)
                })

        # Create result DataFrame
        result = pd.DataFrame({
            'timestamp': df.index,
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
    Validation block for VWAP Mean Reversion Strategy.
    Tests the strategy with REAL BTC/USDT data from Binance.
    """
    import sys
    from datetime import datetime, timedelta

    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting VWAP Mean Reversion Strategy validation with REAL DATA")

    # Test 1: Initialize strategy
    total_tests += 1
    try:
        strategy = VWAPMeanReversionStrategy()
        strategy.initialize({})

        params = strategy.get_parameters()
        if params['std_dev_multiplier'] != 2.0:
            all_validation_failures.append(
                f"Test 1: Expected std_dev_multiplier=2.0, got {params['std_dev_multiplier']}"
            )

        logger.success("Test 1 PASSED: Strategy initialized")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Fetch real data
    total_tests += 1
    try:
        logger.info("Fetching BTC/USDT 1h data...")
        fetcher = BinanceDataFetcher(use_storage=False, use_cache=False)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        data = fetcher.get_ohlcv(
            "BTC/USDT",
            "1h",
            start_date=start_date,
            end_date=end_date,
            limit=168
        )

        if data is None or data.empty:
            all_validation_failures.append("Test 2: Failed to fetch data")
        else:
            logger.success(f"Test 2 PASSED: Fetched {len(data)} hours of data")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Generate signals
    total_tests += 1
    try:
        if 'data' in locals() and data is not None and not data.empty:
            # Data already has DatetimeIndex from fetcher
            signals = strategy.generate_signals(data)

            if signals is None or signals.empty:
                all_validation_failures.append("Test 3: No signals generated")
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
        print("VWAP Mean Reversion Strategy validated with REAL BTC/USDT data")
        sys.exit(0)
