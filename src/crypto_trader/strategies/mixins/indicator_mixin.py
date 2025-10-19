"""
Indicator Mixin

Provides common technical indicator calculation methods.
Wraps pandas_ta to provide consistent interface and error handling.

**Purpose**: Standardize indicator calculations across strategies and
provide fallback handling when indicators fail to calculate.

**Third-party packages**:
- pandas_ta: https://github.com/twopirllc/pandas-ta
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/

**Sample Usage**:
```python
from crypto_trader.strategies.base import BaseStrategy
from crypto_trader.strategies.mixins import IndicatorMixin

class MyStrategy(IndicatorMixin, BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate indicators with automatic error handling
        data = self.add_sma(data, length=20, column='close')
        data = self.add_rsi(data, length=14)
        data = self.add_macd(data)

        # Use indicators for signal generation
        signals = self._create_signal_dataframe(data)
        # ... signal logic
        return signals
```

**Expected Output**:
- DataFrame with calculated indicator columns
- Graceful handling of calculation failures
- Consistent indicator naming across strategies
"""

from typing import Optional
import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False


class IndicatorMixin:
    """
    Mixin providing technical indicator calculation functionality.

    Wraps pandas_ta with consistent error handling and fallback behavior.
    """

    def _ensure_pandas_ta(self) -> None:
        """Ensure pandas_ta is available."""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError(
                "pandas_ta is required for indicator calculations. "
                "Install with: uv add pandas-ta"
            )

    def add_sma(
        self,
        data: pd.DataFrame,
        length: int = 20,
        column: str = 'close',
        name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add Simple Moving Average indicator.

        Args:
            data: DataFrame with OHLCV data
            length: Period for SMA calculation
            column: Column to calculate SMA on
            name: Custom name for SMA column (default: f'sma_{length}')

        Returns:
            DataFrame with SMA column added

        Example:
            >>> data = self.add_sma(data, length=50, column='close')
            >>> print(data['sma_50'].tail())
        """
        self._ensure_pandas_ta()

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        col_name = name or f'sma_{length}'

        try:
            data[col_name] = ta.sma(data[column], length=length)
        except Exception as e:
            # Fallback to simple rolling mean
            data[col_name] = data[column].rolling(window=length).mean()

        return data

    def add_ema(
        self,
        data: pd.DataFrame,
        length: int = 20,
        column: str = 'close',
        name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add Exponential Moving Average indicator.

        Args:
            data: DataFrame with OHLCV data
            length: Period for EMA calculation
            column: Column to calculate EMA on
            name: Custom name for EMA column (default: f'ema_{length}')

        Returns:
            DataFrame with EMA column added

        Example:
            >>> data = self.add_ema(data, length=12)
        """
        self._ensure_pandas_ta()

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        col_name = name or f'ema_{length}'

        try:
            data[col_name] = ta.ema(data[column], length=length)
        except Exception:
            # Fallback to pandas ewm
            data[col_name] = data[column].ewm(span=length, adjust=False).mean()

        return data

    def add_rsi(
        self,
        data: pd.DataFrame,
        length: int = 14,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Relative Strength Index indicator.

        Args:
            data: DataFrame with OHLCV data
            length: Period for RSI calculation
            column: Column to calculate RSI on

        Returns:
            DataFrame with 'rsi' column added

        Example:
            >>> data = self.add_rsi(data, length=14)
            >>> print(data['rsi'].tail())
        """
        self._ensure_pandas_ta()

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        try:
            data['rsi'] = ta.rsi(data[column], length=length)
        except Exception:
            # Fallback to manual calculation
            delta = data[column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))

        return data

    def add_macd(
        self,
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add MACD indicator.

        Args:
            data: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to calculate MACD on

        Returns:
            DataFrame with 'macd', 'macd_signal', 'macd_hist' columns

        Example:
            >>> data = self.add_macd(data)
            >>> print(data[['macd', 'macd_signal']].tail())
        """
        self._ensure_pandas_ta()

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        try:
            macd_result = ta.macd(data[column], fast=fast, slow=slow, signal=signal)
            data['macd'] = macd_result[f'MACD_{fast}_{slow}_{signal}']
            data['macd_signal'] = macd_result[f'MACDs_{fast}_{slow}_{signal}']
            data['macd_hist'] = macd_result[f'MACDh_{fast}_{slow}_{signal}']
        except Exception:
            # Fallback to manual calculation
            ema_fast = data[column].ewm(span=fast, adjust=False).mean()
            ema_slow = data[column].ewm(span=slow, adjust=False).mean()
            data['macd'] = ema_fast - ema_slow
            data['macd_signal'] = data['macd'].ewm(span=signal, adjust=False).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']

        return data

    def add_bollinger_bands(
        self,
        data: pd.DataFrame,
        length: int = 20,
        std: float = 2.0,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands indicator.

        Args:
            data: DataFrame with OHLCV data
            length: Period for moving average
            std: Standard deviation multiplier
            column: Column to calculate bands on

        Returns:
            DataFrame with 'bb_upper', 'bb_mid', 'bb_lower' columns

        Example:
            >>> data = self.add_bollinger_bands(data, length=20, std=2.0)
        """
        self._ensure_pandas_ta()

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        try:
            bb_result = ta.bbands(data[column], length=length, std=std)
            data['bb_lower'] = bb_result[f'BBL_{length}_{std}']
            data['bb_mid'] = bb_result[f'BBM_{length}_{std}']
            data['bb_upper'] = bb_result[f'BBU_{length}_{std}']
        except Exception:
            # Fallback to manual calculation
            sma = data[column].rolling(window=length).mean()
            rolling_std = data[column].rolling(window=length).std()
            data['bb_mid'] = sma
            data['bb_upper'] = sma + (rolling_std * std)
            data['bb_lower'] = sma - (rolling_std * std)

        return data

    def add_atr(
        self,
        data: pd.DataFrame,
        length: int = 14
    ) -> pd.DataFrame:
        """
        Add Average True Range indicator.

        Requires 'high', 'low', 'close' columns.

        Args:
            data: DataFrame with OHLCV data
            length: Period for ATR calculation

        Returns:
            DataFrame with 'atr' column added

        Example:
            >>> data = self.add_atr(data, length=14)
        """
        self._ensure_pandas_ta()

        required = ['high', 'low', 'close']
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"ATR requires columns: {missing}")

        try:
            data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=length)
        except Exception:
            # Fallback to manual calculation
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = tr.rolling(window=length).mean()

        return data


if __name__ == "__main__":
    """
    Validation block for IndicatorMixin.
    Tests all indicator methods with real data.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating IndicatorMixin...\n")

    if not PANDAS_TA_AVAILABLE:
        print("⚠️  pandas_ta not available - skipping indicator tests")
        print("Install with: uv add pandas-ta")
        sys.exit(0)

    # Create test instance
    class TestStrategy(IndicatorMixin):
        """Test class with mixin."""
        pass

    strategy = TestStrategy()

    # Create realistic OHLCV data
    dates = pd.date_range('2025-01-01', periods=100, freq='1h')
    np.random.seed(42)
    base_price = 100
    test_data = pd.DataFrame({
        'open': base_price + np.random.randn(100).cumsum(),
        'high': base_price + np.random.randn(100).cumsum() + 2,
        'low': base_price + np.random.randn(100).cumsum() - 2,
        'close': base_price + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)

    # Test 1: Add SMA
    total_tests += 1
    print("Test 1: Add SMA indicator")
    try:
        result = strategy.add_sma(test_data.copy(), length=20)

        if 'sma_20' not in result.columns:
            all_validation_failures.append("sma_20 column not added")
        elif result['sma_20'].isna().all():
            all_validation_failures.append("sma_20 is all NaN")
        else:
            print(f"  ✓ SMA added successfully")
            print(f"  ✓ Last value: {result['sma_20'].iloc[-1]:.2f}")
    except Exception as e:
        all_validation_failures.append(f"Test 1 failed: {e}")

    # Test 2: Add EMA
    total_tests += 1
    print("\nTest 2: Add EMA indicator")
    try:
        result = strategy.add_ema(test_data.copy(), length=12)

        if 'ema_12' not in result.columns:
            all_validation_failures.append("ema_12 column not added")
        elif result['ema_12'].isna().all():
            all_validation_failures.append("ema_12 is all NaN")
        else:
            print(f"  ✓ EMA added successfully")
            print(f"  ✓ Last value: {result['ema_12'].iloc[-1]:.2f}")
    except Exception as e:
        all_validation_failures.append(f"Test 2 failed: {e}")

    # Test 3: Add RSI
    total_tests += 1
    print("\nTest 3: Add RSI indicator")
    try:
        result = strategy.add_rsi(test_data.copy(), length=14)

        if 'rsi' not in result.columns:
            all_validation_failures.append("rsi column not added")
        elif result['rsi'].dropna().empty:
            all_validation_failures.append("rsi has no values")
        elif not (0 <= result['rsi'].dropna().iloc[-1] <= 100):
            all_validation_failures.append(
                f"RSI value {result['rsi'].iloc[-1]} outside valid range 0-100"
            )
        else:
            print(f"  ✓ RSI added successfully")
            print(f"  ✓ Last value: {result['rsi'].iloc[-1]:.2f}")
    except Exception as e:
        all_validation_failures.append(f"Test 3 failed: {e}")

    # Test 4: Add MACD
    total_tests += 1
    print("\nTest 4: Add MACD indicator")
    try:
        result = strategy.add_macd(test_data.copy())

        required_cols = ['macd', 'macd_signal', 'macd_hist']
        missing = [col for col in required_cols if col not in result.columns]

        if missing:
            all_validation_failures.append(f"MACD missing columns: {missing}")
        elif result['macd'].isna().all():
            all_validation_failures.append("MACD is all NaN")
        else:
            print(f"  ✓ MACD added successfully")
            print(f"  ✓ MACD: {result['macd'].iloc[-1]:.2f}")
            print(f"  ✓ Signal: {result['macd_signal'].iloc[-1]:.2f}")
    except Exception as e:
        all_validation_failures.append(f"Test 4 failed: {e}")

    # Test 5: Add Bollinger Bands
    total_tests += 1
    print("\nTest 5: Add Bollinger Bands")
    try:
        result = strategy.add_bollinger_bands(test_data.copy(), length=20, std=2.0)

        required_cols = ['bb_upper', 'bb_mid', 'bb_lower']
        missing = [col for col in required_cols if col not in result.columns]

        if missing:
            all_validation_failures.append(f"Bollinger Bands missing columns: {missing}")
        elif result['bb_mid'].isna().all():
            all_validation_failures.append("Bollinger Bands are all NaN")
        elif not (result['bb_lower'].iloc[-1] < result['bb_mid'].iloc[-1] < result['bb_upper'].iloc[-1]):
            all_validation_failures.append("Bollinger Bands not in correct order")
        else:
            print(f"  ✓ Bollinger Bands added successfully")
            print(f"  ✓ Upper: {result['bb_upper'].iloc[-1]:.2f}")
            print(f"  ✓ Mid: {result['bb_mid'].iloc[-1]:.2f}")
            print(f"  ✓ Lower: {result['bb_lower'].iloc[-1]:.2f}")
    except Exception as e:
        all_validation_failures.append(f"Test 5 failed: {e}")

    # Test 6: Add ATR
    total_tests += 1
    print("\nTest 6: Add ATR indicator")
    try:
        result = strategy.add_atr(test_data.copy(), length=14)

        if 'atr' not in result.columns:
            all_validation_failures.append("atr column not added")
        elif result['atr'].dropna().empty:
            all_validation_failures.append("ATR has no values")
        elif (result['atr'].dropna() < 0).any():
            all_validation_failures.append("ATR has negative values")
        else:
            print(f"  ✓ ATR added successfully")
            print(f"  ✓ Last value: {result['atr'].iloc[-1]:.2f}")
    except Exception as e:
        all_validation_failures.append(f"Test 6 failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("IndicatorMixin is validated and ready for use")
        sys.exit(0)
