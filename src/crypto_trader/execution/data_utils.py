"""
Data Utilities for Backtest Execution

This module provides data manipulation and technical indicator utilities
for preparing data for backtest execution.

**Purpose**: Data slicing, indicator computation, and data preparation

**Key Functions**:
- calculate_data_limit: Calculate candles needed for timeframe/horizon
- slice_data_to_horizon: Slice data to appropriate window for horizon
- compute_indicator_series: Compute technical indicators
- add_required_indicators: Ensure strategy indicators are present

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- pandas_ta: https://github.com/twopirllc/pandas-ta

**Sample Input**:
```python
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})
sliced = slice_data_to_horizon(data, timeframe='1h', horizon_days=30)
```

**Expected Output**:
DataFrame sliced to appropriate window with required indicators computed.

Extracted from master.py (lines 424-590) during Phase 2.5 refactoring.
"""

from typing import Any, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger


def calculate_data_limit(
    timeframe: str,
    horizon_days: int,
    warmup_multiplier: float = 1.0
) -> int:
    """
    Calculate the number of candles needed for a given timeframe and horizon.

    Args:
        timeframe: Timeframe string (e.g., '1h', '1d')
        horizon_days: Number of days in the horizon
        warmup_multiplier: Multiplier for warmup period (default 1.0 = no warmup)
                          Use 3.0 for multi-pair strategies (2x warmup + 1x test)
                          Use 4.0 for advanced strategies (HRP, Statistical Arbitrage)

    Returns:
        Number of candles needed (includes warmup period)
    """
    timeframe_to_periods = {
        "1m": 24 * 60,
        "5m": 24 * 12,
        "15m": 24 * 4,
        "1h": 24,
        "4h": 6,
        "1d": 1,
        "1w": 1 / 7
    }
    periods_per_day = timeframe_to_periods.get(timeframe, 24)  # Default to hourly

    # Apply warmup multiplier for strategies that need historical context
    total_days = int(horizon_days * warmup_multiplier)
    return int(total_days * periods_per_day)


def slice_data_to_horizon(
    data: pd.DataFrame,
    timeframe: str,
    horizon_days: int,
    warmup_multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Slice data to the appropriate window for a given horizon.

    Takes the LAST N candles corresponding to horizon_days × warmup_multiplier.
    This ensures each horizon tests on the correct time period.

    CRITICAL: Without this, all horizons would test on the same full dataset!

    Args:
        data: Full DataFrame with all available data
        timeframe: Candle timeframe
        horizon_days: Target horizon in days
        warmup_multiplier: Multiplier for warmup period (e.g., 1.5 = 50% warmup)

    Returns:
        Sliced DataFrame with only the relevant window

    Example:
        Full data: 270 days (6480 candles at 1h)
        horizon_days=30, warmup=1.5
        Result: Last 45 days (1080 candles) - most recent data
    """
    required_candles = calculate_data_limit(timeframe, horizon_days, warmup_multiplier)

    if len(data) <= required_candles:
        # Already have the right amount or less
        return data

    # Take the LAST required_candles rows (most recent data)
    sliced = data.tail(required_candles).copy()

    logger.debug(
        f"Sliced data from {len(data)} to {len(sliced)} candles for {horizon_days}d horizon "
        f"(warmup={warmup_multiplier}x)"
    )

    return sliced


def compute_indicator_series(df: pd.DataFrame, indicator: str) -> Optional[pd.Series]:
    """
    Compute a technical indicator column for the provided DataFrame.
    Supports a limited set used by built-in strategies.

    Args:
        df: DataFrame with OHLCV data
        indicator: Indicator name (e.g., 'SMA_20', 'RSI_14', 'EMA_50', 'ATR_14')

    Returns:
        Series with computed indicator values, or None if unsupported
    """
    normalized = indicator.upper()

    try:
        if normalized.startswith("SMA_"):
            period = int(normalized.split("_")[1])
            return df["close"].rolling(window=period, min_periods=period).mean()
        if normalized.startswith("EMA_"):
            period = int(normalized.split("_")[1])
            return ta.ema(df["close"], length=period)
        if normalized.startswith("RSI_"):
            period = int(normalized.split("_")[1])
            return ta.rsi(df["close"], length=period)
        if normalized.startswith("ATR_"):
            period = int(normalized.split("_")[1])
            return ta.atr(df["high"], df["low"], df["close"], length=period)
    except Exception as exc:
        logger.warning(
            f"Failed to compute indicator '{indicator}': {exc}"
        )
        return None

    return None


def add_required_indicators(strategy: Any, data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all required indicators for the strategy are present on the DataFrame.

    Args:
        strategy: Strategy object with get_required_indicators() method
        data: DataFrame with OHLCV data

    Returns:
        DataFrame with all required indicators computed

    Raises:
        ValueError: If strategy requires unsupported indicators
    """
    get_indicators = getattr(strategy, "get_required_indicators", None)
    if get_indicators is None:
        return data

    try:
        required = get_indicators()
    except Exception as exc:
        logger.warning(
            f"Could not obtain required indicators for {strategy.name}: {exc}"
        )
        return data

    if not required:
        return data

    df = data.copy()
    for indicator in required:
        if indicator in df.columns:
            continue

        series = compute_indicator_series(df, indicator)
        if series is None:
            raise ValueError(
                f"Unsupported indicator '{indicator}' for strategy '{strategy.name}'"
            )
        df[indicator] = series

    return df


if __name__ == "__main__":
    """
    Validation block for data utilities.
    """
    import sys
    import numpy as np

    all_validation_failures = []
    total_tests = 0

    # Test 1: calculate_data_limit
    total_tests += 1
    print("Test 1: calculate_data_limit")
    try:
        # 1h timeframe, 30 days, no warmup
        result = calculate_data_limit("1h", 30, 1.0)
        expected = 30 * 24  # 720 candles
        if result != expected:
            all_validation_failures.append(f"calculate_data_limit: Expected {expected}, got {result}")
        else:
            print(f"  ✓ 1h timeframe, 30 days, 1.0x warmup = {result} candles")

        # 1d timeframe, 90 days, 1.5x warmup
        result2 = calculate_data_limit("1d", 90, 1.5)
        expected2 = int(90 * 1.5 * 1)  # 135 candles
        if result2 != expected2:
            all_validation_failures.append(f"calculate_data_limit: Expected {expected2}, got {result2}")
        else:
            print(f"  ✓ 1d timeframe, 90 days, 1.5x warmup = {result2} candles")

    except Exception as e:
        all_validation_failures.append(f"calculate_data_limit failed: {e}")

    # Test 2: slice_data_to_horizon
    total_tests += 1
    print("\nTest 2: slice_data_to_horizon")
    try:
        # Create sample data with 1000 rows
        sample_data = pd.DataFrame({
            'open': np.random.rand(1000),
            'high': np.random.rand(1000),
            'low': np.random.rand(1000),
            'close': np.random.rand(1000),
            'volume': np.random.rand(1000) * 1000
        })

        # Slice to 30 days (1h timeframe, 1.5x warmup = 1080 candles)
        sliced = slice_data_to_horizon(sample_data, "1h", 30, 1.5)
        expected_length = min(1080, len(sample_data))  # Should be 1000 (all data)

        if len(sliced) != expected_length:
            all_validation_failures.append(f"slice_data_to_horizon: Expected {expected_length} rows, got {len(sliced)}")
        else:
            print(f"  ✓ Sliced 1000 rows -> {len(sliced)} rows (30d horizon, 1.5x warmup)")

        # Verify it's the LAST rows
        if not sliced.equals(sample_data.tail(expected_length)):
            all_validation_failures.append("slice_data_to_horizon: Did not return LAST rows")
        else:
            print(f"  ✓ Correctly returned LAST {len(sliced)} rows")

    except Exception as e:
        all_validation_failures.append(f"slice_data_to_horizon failed: {e}")

    # Test 3: compute_indicator_series
    total_tests += 1
    print("\nTest 3: compute_indicator_series")
    try:
        # Create sample OHLCV data
        sample_df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 10,
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111] * 10,
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108] * 10,
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 10,
            'volume': [1000] * 100
        })

        # Test SMA_20
        sma = compute_indicator_series(sample_df, "SMA_20")
        if sma is None:
            all_validation_failures.append("compute_indicator_series: SMA_20 returned None")
        elif len(sma) != len(sample_df):
            all_validation_failures.append(f"compute_indicator_series: SMA_20 length mismatch")
        else:
            print(f"  ✓ SMA_20 computed: {len(sma)} values")

        # Test RSI_14
        rsi = compute_indicator_series(sample_df, "RSI_14")
        if rsi is None:
            all_validation_failures.append("compute_indicator_series: RSI_14 returned None")
        elif len(rsi) != len(sample_df):
            all_validation_failures.append(f"compute_indicator_series: RSI_14 length mismatch")
        else:
            print(f"  ✓ RSI_14 computed: {len(rsi)} values")

        # Test unsupported indicator
        unsupported = compute_indicator_series(sample_df, "UNSUPPORTED_10")
        if unsupported is not None:
            all_validation_failures.append("compute_indicator_series: Unsupported indicator should return None")
        else:
            print(f"  ✓ Unsupported indicator correctly returns None")

    except Exception as e:
        all_validation_failures.append(f"compute_indicator_series failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Data utilities are validated and ready for use")
        sys.exit(0)
