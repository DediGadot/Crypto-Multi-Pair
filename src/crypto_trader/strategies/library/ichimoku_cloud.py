"""
Ichimoku Cloud Strategy

This module implements a comprehensive Japanese indicator system combining trend,
momentum, and support/resistance analysis. Ichimoku provides multiple confirmation
layers to filter false signals and excellent risk management through cloud zones.

**Purpose**: Implement a multi-dimensional trading system optimized for trend
identification and risk management in cryptocurrency markets.

**Strategy Type**: Multi-Timeframe Trend & Momentum System
**Indicators**: Conversion Line (9), Base Line (26), Leading Span A, Leading Span B (52)
**Entry Signal (Long)**: Price > Cloud AND Conversion > Base AND Lagging Span confirms
**Entry Signal (Short)**: Price < Cloud AND Conversion < Base AND Lagging Span confirms
**Exit Signal**: Price crosses back through cloud OR Conversion/Base crossover reverses

**Parameters**:
- conversion_period: Tenkan-sen period (default: 9)
- base_period: Kijun-sen period (default: 26)
- span_b_period: Senkou Span B period (default: 52)
- displacement: Chikou Span displacement (default: 26)

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
        {'reason': 'ichimoku_bullish', 'above_cloud': True, 'conv_base': 'bullish'},
        {},
        {'reason': 'ichimoku_bearish', 'above_cloud': False, 'conv_base': 'bearish'},
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
    name="Ichimoku_Cloud",
    description="Ichimoku Cloud comprehensive trend and momentum system",
    tags=["trend_following", "ichimoku", "multi_timeframe", "sota_2024"]
)
class IchimokuCloudStrategy(BaseStrategy):
    """
    Ichimoku Cloud Strategy.

    Generates buy signals when price is above cloud, Conversion > Base, and Lagging Span confirms.
    Generates sell signals when price is below cloud, Conversion < Base, and Lagging Span confirms.
    Cloud provides dynamic support/resistance and trend visualization.

    Comprehensive Japanese indicator system optimized for crypto hourly timeframes.
    """

    def __init__(self, name: str = "Ichimoku_Cloud", config: Dict[str, Any] = None):
        """
        Initialize the Ichimoku Cloud strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default Ichimoku parameters
        self.conversion_period = 9   # Tenkan-sen
        self.base_period = 26        # Kijun-sen
        self.span_b_period = 52      # Senkou Span B
        self.displacement = 26       # Chikou Span

        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with strategy parameters

        Raises:
            ValueError: If parameters are invalid
        """
        self.conversion_period = config.get("conversion_period", 9)
        self.base_period = config.get("base_period", 26)
        self.span_b_period = config.get("span_b_period", 52)
        self.displacement = config.get("displacement", 26)

        # Validate parameters
        if any(p <= 0 for p in [self.conversion_period, self.base_period, self.span_b_period, self.displacement]):
            raise ValueError("All periods must be positive")

        if self.conversion_period >= self.base_period:
            raise ValueError("Conversion period must be less than base period")

        self._initialized = True
        logger.info(
            f"{self.name} initialized with conversion={self.conversion_period}, "
            f"base={self.base_period}, span_b={self.span_b_period}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "conversion_period": self.conversion_period,
            "base_period": self.base_period,
            "span_b_period": self.span_b_period,
            "displacement": self.displacement
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
        Generate trading signals based on Ichimoku Cloud system.

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

        # Need enough data for Ichimoku (including displacement for leading spans)
        min_required = self.span_b_period + self.displacement + 10
        if len(data) < min_required:
            logger.warning(
                f"Insufficient data: {len(data)} rows, need {min_required}"
            )
            return self._create_hold_signals(data)

        # Calculate Ichimoku indicator using pandas_ta
        df = data.copy()

        # Calculate all Ichimoku components
        ichimoku_df = ta.ichimoku(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            tenkan=self.conversion_period,
            kijun=self.base_period,
            senkou=self.span_b_period
        )

        # Ichimoku returns multiple columns with format: ISA_<tenkan>, ISB_<kijun>_<senkou>, ITS_<tenkan>, IKS_<kijun>, ICS_<kijun>
        conversion_col = f'ITS_{self.conversion_period}'     # Conversion Line (Tenkan-sen)
        base_col = f'IKS_{self.base_period}'                 # Base Line (Kijun-sen)
        span_a_col = f'ISA_{self.conversion_period}'         # Leading Span A (Senkou Span A)
        span_b_col = f'ISB_{self.base_period}'               # Leading Span B (Senkou Span B)
        lagging_col = f'ICS_{self.base_period}'              # Lagging Span (Chikou Span)

        df['conversion_line'] = ichimoku_df[0][conversion_col]
        df['base_line'] = ichimoku_df[0][base_col]
        df['span_a'] = ichimoku_df[0][span_a_col]
        df['span_b'] = ichimoku_df[0][span_b_col]
        df['lagging_span'] = ichimoku_df[0][lagging_col]

        # Calculate cloud top and bottom
        df['cloud_top'] = df[['span_a', 'span_b']].max(axis=1)
        df['cloud_bottom'] = df[['span_a', 'span_b']].min(axis=1)

        # Initialize signal arrays
        signals = []
        confidences = []
        metadata = []

        # Generate signals based on Ichimoku conditions
        for i in range(len(df)):
            if i < self.displacement:  # Need historical data for lagging span
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Check for NaN values
            if any(pd.isna(df[col].iloc[i]) for col in ['conversion_line', 'base_line', 'cloud_top', 'cloud_bottom']):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            current_price = df['close'].iloc[i]
            conversion = df['conversion_line'].iloc[i]
            base = df['base_line'].iloc[i]
            cloud_top = df['cloud_top'].iloc[i]
            cloud_bottom = df['cloud_bottom'].iloc[i]

            # Check lagging span (price 26 periods ago)
            lagging_idx = i - self.displacement
            if lagging_idx >= 0 and not pd.isna(df['close'].iloc[lagging_idx]):
                lagging_price = df['close'].iloc[lagging_idx]
                lagging_above = current_price > lagging_price
            else:
                lagging_above = None

            # Determine current position relative to cloud
            above_cloud = current_price > cloud_top
            below_cloud = current_price < cloud_bottom
            in_cloud = not above_cloud and not below_cloud

            # Check if previous candle was in different position (crossover)
            if i > 0 and not pd.isna(df['cloud_top'].iloc[i-1]):
                prev_price = df['close'].iloc[i - 1]
                prev_cloud_top = df['cloud_top'].iloc[i - 1]
                prev_cloud_bottom = df['cloud_bottom'].iloc[i - 1]

                # Strong Bullish: Price > Cloud AND Conversion > Base AND Lagging Span confirms
                if (above_cloud and conversion > base and lagging_above and
                    prev_price <= prev_cloud_top):

                    # Calculate confidence based on distance from cloud
                    cloud_distance = (current_price - cloud_top) / current_price
                    conv_base_strength = (conversion - base) / base
                    confidence = min(0.6 + cloud_distance * 10 + conv_base_strength * 10, 1.0)

                    signals.append(SignalType.BUY.value)
                    confidences.append(max(0.5, min(confidence, 1.0)))
                    metadata.append({
                        'reason': 'ichimoku_bullish',
                        'above_cloud': True,
                        'conv_base': 'bullish',
                        'lagging_confirms': True,
                        'cloud_distance': float(cloud_distance),
                        'price': float(current_price),
                        'cloud_top': float(cloud_top)
                    })
                    logger.debug(f"Ichimoku bullish signal at {df.index[i]}")

                # Strong Bearish: Price < Cloud AND Conversion < Base AND Lagging Span confirms
                elif (below_cloud and conversion < base and (lagging_above is False) and
                      prev_price >= prev_cloud_bottom):

                    # Calculate confidence based on distance from cloud
                    cloud_distance = (cloud_bottom - current_price) / current_price
                    conv_base_strength = (base - conversion) / base
                    confidence = min(0.6 + cloud_distance * 10 + conv_base_strength * 10, 1.0)

                    signals.append(SignalType.SELL.value)
                    confidences.append(max(0.5, min(confidence, 1.0)))
                    metadata.append({
                        'reason': 'ichimoku_bearish',
                        'above_cloud': False,
                        'conv_base': 'bearish',
                        'lagging_confirms': True,
                        'cloud_distance': float(cloud_distance),
                        'price': float(current_price),
                        'cloud_bottom': float(cloud_bottom)
                    })
                    logger.debug(f"Ichimoku bearish signal at {df.index[i]}")

                else:
                    signals.append(SignalType.HOLD.value)
                    confidences.append(0.0)
                    metadata.append({
                        'above_cloud': above_cloud,
                        'in_cloud': in_cloud,
                        'conv_vs_base': 'bullish' if conversion > base else 'bearish'
                    })
            else:
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})

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
    Validation block for Ichimoku Cloud Strategy.
    Tests the strategy with REAL BTC/USDT data from Binance.
    """
    import sys
    from datetime import datetime, timedelta

    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting Ichimoku Cloud Strategy validation with REAL DATA")

    # Test 1: Initialize strategy
    total_tests += 1
    try:
        strategy = IchimokuCloudStrategy()
        strategy.initialize({})

        params = strategy.get_parameters()
        if params['conversion_period'] != 9:
            all_validation_failures.append(
                f"Test 1: Expected conversion_period=9, got {params['conversion_period']}"
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
        start_date = end_date - timedelta(days=90)

        data = fetcher.get_ohlcv(
            "BTC/USDT",
            "1h",
            start_date=start_date,
            end_date=end_date,
            limit=2160
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
            test_data = data.reset_index()
            signals = strategy.generate_signals(test_data)

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
        print("Ichimoku Cloud Strategy validated with REAL BTC/USDT data")
        sys.exit(0)
