"""
Data Provider Interface for Crypto Trading System

This module defines the abstract base class for data providers that fetch and manage
OHLCV (Open, High, Low, Close, Volume) data from various cryptocurrency exchanges.

Purpose:
- Define a standard interface for all data providers
- Ensure consistent API across different exchange implementations
- Support extensibility for adding new exchange integrations

Third-party documentation:
- pandas: https://pandas.pydata.org/docs/
- abc: https://docs.python.org/3/library/abc.html

Sample Input:
    provider.get_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-01-31")

Expected Output:
    DataFrame with columns: [timestamp, open, high, low, close, volume]
    Index: DatetimeIndex
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import pandas as pd
from loguru import logger


class DataProvider(ABC):
    """
    Abstract base class for cryptocurrency data providers.

    All data providers must implement these methods to ensure consistent
    behavior across different exchange integrations.
    """

    def __init__(self, name: str):
        """
        Initialize the data provider.

        Args:
            name: Human-readable name of the provider (e.g., "Binance", "Coinbase")
        """
        self.name = name
        logger.info(f"Initializing {self.name} data provider")

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol and timeframe.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe (e.g., "1m", "5m", "1h", "1d")
            start_date: Start date for data retrieval (optional)
            end_date: End date for data retrieval (optional)
            limit: Maximum number of candles to fetch (optional)

        Returns:
            DataFrame with OHLCV data:
                - Index: DatetimeIndex (UTC)
                - Columns: [open, high, low, close, volume]

        Raises:
            ValueError: If symbol or timeframe is invalid
            ConnectionError: If unable to connect to exchange
        """
        pass

    @abstractmethod
    def update_data(
        self,
        symbol: str,
        timeframe: str
    ) -> bool:
        """
        Update existing data with latest candles.

        Fetches the most recent data and appends it to existing storage.
        Only fetches data newer than what's already stored.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe (e.g., "1m", "5m", "1h", "1d")

        Returns:
            True if update was successful, False otherwise

        Raises:
            ValueError: If symbol or timeframe is invalid
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading pairs from the exchange.

        Returns:
            List of symbol strings (e.g., ["BTC/USDT", "ETH/USDT", ...])

        Raises:
            ConnectionError: If unable to connect to exchange
        """
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid and tradeable on the exchange.

        Args:
            symbol: Trading pair symbol to validate

        Returns:
            True if symbol is valid and tradeable, False otherwise
        """
        pass

    @abstractmethod
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Check if a timeframe is supported by the exchange.

        Args:
            timeframe: Candlestick timeframe to validate

        Returns:
            True if timeframe is supported, False otherwise
        """
        pass


class MockDataProvider(DataProvider):
    """
    Mock implementation of DataProvider for testing purposes.

    Returns synthetic data without connecting to real exchanges.
    Useful for testing and development when exchange access is not needed.
    """

    def __init__(self):
        super().__init__("Mock Provider")
        self._symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        self._timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Return synthetic OHLCV data for testing."""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Generate synthetic data
        import numpy as np

        num_candles = limit if limit else 100
        dates = pd.date_range(
            end=end_date or datetime.now(),
            periods=num_candles,
            freq='1h'
        )

        # Generate realistic-looking price data
        base_price = 50000 if "BTC" in symbol else 3000
        price_change = np.random.randn(num_candles).cumsum() * 100
        close_prices = base_price + price_change

        df = pd.DataFrame({
            'open': close_prices * (1 + np.random.randn(num_candles) * 0.001),
            'high': close_prices * (1 + np.abs(np.random.randn(num_candles)) * 0.002),
            'low': close_prices * (1 - np.abs(np.random.randn(num_candles)) * 0.002),
            'close': close_prices,
            'volume': np.random.uniform(100, 1000, num_candles)
        }, index=dates)

        return df

    def update_data(self, symbol: str, timeframe: str) -> bool:
        """Mock update always succeeds."""
        return True

    def get_available_symbols(self) -> List[str]:
        """Return mock list of symbols."""
        return self._symbols.copy()

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is in mock list."""
        return symbol in self._symbols

    def validate_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe is in mock list."""
        return timeframe in self._timeframes


if __name__ == "__main__":
    """
    Validation: Test the DataProvider interface with MockDataProvider.

    Tests:
    1. Fetch OHLCV data with default parameters
    2. Fetch OHLCV data with specific date range
    3. Fetch OHLCV data with limit
    4. Get available symbols
    5. Validate symbol and timeframe
    6. Error handling for invalid inputs
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting DataProvider validation with MockDataProvider")

    # Initialize mock provider
    provider = MockDataProvider()

    # Test 1: Fetch OHLCV data with limit
    total_tests += 1
    try:
        df = provider.get_ohlcv("BTC/USDT", "1h", limit=50)
        if not isinstance(df, pd.DataFrame):
            all_validation_failures.append("Test 1: Result is not a DataFrame")
        elif len(df) != 50:
            all_validation_failures.append(f"Test 1: Expected 50 candles, got {len(df)}")
        elif not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            all_validation_failures.append(f"Test 1: Missing required columns. Got: {df.columns.tolist()}")
        elif not isinstance(df.index, pd.DatetimeIndex):
            all_validation_failures.append("Test 1: Index is not DatetimeIndex")
        else:
            logger.success(f"Test 1 PASSED: Fetched {len(df)} candles for BTC/USDT")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Fetch OHLCV data with date range
    total_tests += 1
    try:
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        df = provider.get_ohlcv("ETH/USDT", "1h", start_date=start, end_date=end, limit=100)
        if len(df) != 100:
            all_validation_failures.append(f"Test 2: Expected 100 candles, got {len(df)}")
        else:
            logger.success(f"Test 2 PASSED: Fetched data with date range")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Get available symbols
    total_tests += 1
    try:
        symbols = provider.get_available_symbols()
        expected_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        if not isinstance(symbols, list):
            all_validation_failures.append("Test 3: Result is not a list")
        elif symbols != expected_symbols:
            all_validation_failures.append(f"Test 3: Expected {expected_symbols}, got {symbols}")
        else:
            logger.success(f"Test 3 PASSED: Got {len(symbols)} available symbols")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Validate symbol
    total_tests += 1
    try:
        valid = provider.validate_symbol("BTC/USDT")
        invalid = provider.validate_symbol("INVALID/PAIR")
        if not valid:
            all_validation_failures.append("Test 4: BTC/USDT should be valid")
        elif invalid:
            all_validation_failures.append("Test 4: INVALID/PAIR should be invalid")
        else:
            logger.success("Test 4 PASSED: Symbol validation works correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception raised: {e}")

    # Test 5: Validate timeframe
    total_tests += 1
    try:
        valid = provider.validate_timeframe("1h")
        invalid = provider.validate_timeframe("99x")
        if not valid:
            all_validation_failures.append("Test 5: 1h should be valid")
        elif invalid:
            all_validation_failures.append("Test 5: 99x should be invalid")
        else:
            logger.success("Test 5 PASSED: Timeframe validation works correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Error handling for invalid symbol
    total_tests += 1
    try:
        df = provider.get_ohlcv("INVALID/PAIR", "1h")
        all_validation_failures.append("Test 6: Expected ValueError for invalid symbol but no exception was raised")
    except ValueError:
        logger.success("Test 6 PASSED: ValueError raised for invalid symbol")
    except Exception as e:
        all_validation_failures.append(f"Test 6: Expected ValueError but got {type(e).__name__}: {e}")

    # Test 7: Error handling for invalid timeframe
    total_tests += 1
    try:
        df = provider.get_ohlcv("BTC/USDT", "99x")
        all_validation_failures.append("Test 7: Expected ValueError for invalid timeframe but no exception was raised")
    except ValueError:
        logger.success("Test 7 PASSED: ValueError raised for invalid timeframe")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Expected ValueError but got {type(e).__name__}: {e}")

    # Test 8: Update data
    total_tests += 1
    try:
        result = provider.update_data("BTC/USDT", "1h")
        if not isinstance(result, bool):
            all_validation_failures.append("Test 8: Result should be boolean")
        elif not result:
            all_validation_failures.append("Test 8: Update should return True for mock provider")
        else:
            logger.success("Test 8 PASSED: Update data returns True")
    except Exception as e:
        all_validation_failures.append(f"Test 8: Exception raised: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("DataProvider interface is validated and ready for production implementations")
        sys.exit(0)
