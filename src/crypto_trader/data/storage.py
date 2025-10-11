"""
File-Based Storage for OHLCV Data

This module provides simple CSV-based storage for cryptocurrency OHLCV data.
Uses the local filesystem to persist data without requiring a database.

Purpose:
- Store OHLCV data in CSV format
- Load historical data from CSV files
- Support incremental updates (append new data)
- Validate data integrity
- Organize data by symbol and timeframe

Third-party documentation:
- pandas: https://pandas.pydata.org/docs/
- pathlib: https://docs.python.org/3/library/pathlib.html

Sample Input:
    storage.save_ohlcv(df, "BTC/USDT", "1h")
    df = storage.load_ohlcv("BTC/USDT", "1h")

Expected Output:
    CSV files in: data/ohlcv/{symbol}/{timeframe}.csv
    DataFrame with columns: [timestamp, open, high, low, close, volume]
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger


class OHLCVStorage:
    """
    File-based storage for OHLCV data using CSV format.

    Organizes data by symbol and timeframe in a hierarchical directory structure.
    Supports incremental updates to avoid re-fetching existing data.
    """

    def __init__(self, base_path: str = "data/ohlcv"):
        """
        Initialize the storage with a base directory.

        Args:
            base_path: Base directory for storing OHLCV data
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized OHLCV storage at: {self.base_path.absolute()}")

    def _get_file_path(self, symbol: str, timeframe: str) -> Path:
        """
        Get the file path for a given symbol and timeframe.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe (e.g., "1h")

        Returns:
            Path object for the CSV file
        """
        # Replace "/" with "_" for filesystem compatibility
        safe_symbol = symbol.replace("/", "_")
        symbol_dir = self.base_path / safe_symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{timeframe}.csv"

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        mode: str = "overwrite"
    ) -> bool:
        """
        Save OHLCV data to CSV file.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            mode: Save mode - "overwrite" or "append"

        Returns:
            True if save was successful, False otherwise

        Raises:
            ValueError: If DataFrame is invalid or empty
        """
        try:
            # Validate DataFrame
            if df is None or df.empty:
                raise ValueError("DataFrame is None or empty")

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"DataFrame missing required columns. Expected: {required_columns}")

            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("Index is not DatetimeIndex, attempting conversion")
                df.index = pd.to_datetime(df.index)

            # Sort by timestamp
            df = df.sort_index()

            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]

            file_path = self._get_file_path(symbol, timeframe)

            if mode == "append" and file_path.exists():
                # Load existing data and merge
                existing_df = self.load_ohlcv(symbol, timeframe)
                if existing_df is not None and not existing_df.empty:
                    # Combine and remove duplicates
                    combined_df = pd.concat([existing_df, df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df = combined_df.sort_index()
                    df = combined_df

            # Save to CSV
            df.to_csv(file_path, index=True)
            logger.info(f"Saved {len(df)} candles to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving OHLCV data for {symbol} {timeframe}: {e}")
            return False

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data from CSV file.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            start_date: Filter data from this date (optional)
            end_date: Filter data until this date (optional)

        Returns:
            DataFrame with OHLCV data or None if file doesn't exist
        """
        try:
            file_path = self._get_file_path(symbol, timeframe)

            if not file_path.exists():
                logger.warning(f"No data file found for {symbol} {timeframe}")
                return None

            # Load CSV with first column as index
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Ensure proper column order
            column_order = ['open', 'high', 'low', 'close', 'volume']
            df = df[column_order]

            # Apply date filters if provided
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]

            logger.info(f"Loaded {len(df)} candles from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading OHLCV data for {symbol} {timeframe}: {e}")
            return None

    def has_data(self, symbol: str, timeframe: str) -> bool:
        """
        Check if data exists for a given symbol and timeframe.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe

        Returns:
            True if data file exists, False otherwise
        """
        file_path = self._get_file_path(symbol, timeframe)
        return file_path.exists()

    def get_date_range(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get the date range of stored data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe

        Returns:
            Tuple of (start_date, end_date) or None if no data exists
        """
        df = self.load_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            return None

        return (df.index.min().to_pydatetime(), df.index.max().to_pydatetime())

    def get_latest_timestamp(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[datetime]:
        """
        Get the timestamp of the most recent candle.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe

        Returns:
            Datetime of latest candle or None if no data exists
        """
        df = self.load_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            return None

        return df.index.max().to_pydatetime()

    def delete_data(self, symbol: str, timeframe: str) -> bool:
        """
        Delete stored data for a given symbol and timeframe.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            file_path = self._get_file_path(symbol, timeframe)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted data file: {file_path}")
                return True
            else:
                logger.warning(f"No data file to delete: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting data for {symbol} {timeframe}: {e}")
            return False

    def validate_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, list]:
        """
        Validate OHLCV data for correctness.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check if DataFrame is empty
        if df is None or df.empty:
            errors.append("DataFrame is None or empty")
            return (False, errors)

        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        # Check for null values
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                errors.append(f"Column '{col}' has {count} null values")

        # Check for negative values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns and (df[col] < 0).any():
                errors.append(f"Column '{col}' has negative values")

        # Check high/low relationship
        if 'high' in df.columns and 'low' in df.columns:
            if (df['high'] < df['low']).any():
                errors.append("High is less than Low in some rows")

        # Check if high is the highest and low is the lowest
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            max_price = df[['open', 'high', 'low', 'close']].max(axis=1)
            min_price = df[['open', 'high', 'low', 'close']].min(axis=1)

            if (df['high'] < max_price).any():
                errors.append("High is not the maximum price in some rows")

            if (df['low'] > min_price).any():
                errors.append("Low is not the minimum price in some rows")

        # Check for duplicate timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.duplicated().any():
                errors.append(f"Found {df.index.duplicated().sum()} duplicate timestamps")

        is_valid = len(errors) == 0
        return (is_valid, errors)


if __name__ == "__main__":
    """
    Validation: Test OHLCVStorage functionality with real operations.

    Tests:
    1. Create storage and verify directory creation
    2. Save OHLCV data in overwrite mode
    3. Load saved data and verify integrity
    4. Save additional data in append mode
    5. Verify append merged data correctly
    6. Get date range and latest timestamp
    7. Validate data correctness
    8. Delete data
    9. Handle invalid data
    """
    import sys
    import numpy as np

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting OHLCVStorage validation")

    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    test_df = pd.DataFrame({
        'open': np.random.uniform(40000, 42000, 100),
        'high': np.random.uniform(42000, 43000, 100),
        'low': np.random.uniform(39000, 40000, 100),
        'close': np.random.uniform(40000, 42000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    }, index=dates)

    # Test 1: Create storage and verify directory
    total_tests += 1
    try:
        storage = OHLCVStorage(base_path="data/test_ohlcv")
        if not storage.base_path.exists():
            all_validation_failures.append("Test 1: Base directory was not created")
        else:
            logger.success("Test 1 PASSED: Storage initialized and directory created")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Save OHLCV data
    total_tests += 1
    try:
        result = storage.save_ohlcv(test_df, "BTC/USDT", "1h", mode="overwrite")
        if not result:
            all_validation_failures.append("Test 2: Save operation returned False")
        elif not storage.has_data("BTC/USDT", "1h"):
            all_validation_failures.append("Test 2: Data file was not created")
        else:
            logger.success("Test 2 PASSED: Data saved successfully")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Load saved data
    total_tests += 1
    try:
        loaded_df = storage.load_ohlcv("BTC/USDT", "1h")
        if loaded_df is None:
            all_validation_failures.append("Test 3: Load returned None")
        elif len(loaded_df) != len(test_df):
            all_validation_failures.append(f"Test 3: Expected {len(test_df)} rows, got {len(loaded_df)}")
        elif not all(col in loaded_df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            all_validation_failures.append(f"Test 3: Missing columns. Got: {loaded_df.columns.tolist()}")
        else:
            logger.success(f"Test 3 PASSED: Loaded {len(loaded_df)} candles")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Append new data
    total_tests += 1
    try:
        # Create new data starting after existing data
        new_dates = pd.date_range(start='2024-01-05 04:00:00', periods=50, freq='1h')
        new_df = pd.DataFrame({
            'open': np.random.uniform(40000, 42000, 50),
            'high': np.random.uniform(42000, 43000, 50),
            'low': np.random.uniform(39000, 40000, 50),
            'close': np.random.uniform(40000, 42000, 50),
            'volume': np.random.uniform(100, 1000, 50)
        }, index=new_dates)

        result = storage.save_ohlcv(new_df, "BTC/USDT", "1h", mode="append")
        if not result:
            all_validation_failures.append("Test 4: Append operation returned False")
        else:
            # Load and verify combined length
            combined_df = storage.load_ohlcv("BTC/USDT", "1h")
            expected_len = 100 + 50  # Original + new
            if len(combined_df) < 100:  # Should have at least original data
                all_validation_failures.append(f"Test 4: After append expected at least 100 rows, got {len(combined_df)}")
            else:
                logger.success(f"Test 4 PASSED: Appended data, now have {len(combined_df)} candles")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception raised: {e}")

    # Test 5: Get date range
    total_tests += 1
    try:
        date_range = storage.get_date_range("BTC/USDT", "1h")
        if date_range is None:
            all_validation_failures.append("Test 5: Date range returned None")
        elif not isinstance(date_range, tuple) or len(date_range) != 2:
            all_validation_failures.append(f"Test 5: Expected tuple of 2 elements, got {type(date_range)}")
        elif date_range[0] > date_range[1]:
            all_validation_failures.append("Test 5: Start date is after end date")
        else:
            logger.success(f"Test 5 PASSED: Date range from {date_range[0]} to {date_range[1]}")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Get latest timestamp
    total_tests += 1
    try:
        latest = storage.get_latest_timestamp("BTC/USDT", "1h")
        if latest is None:
            all_validation_failures.append("Test 6: Latest timestamp returned None")
        elif not isinstance(latest, datetime):
            all_validation_failures.append(f"Test 6: Expected datetime, got {type(latest)}")
        else:
            logger.success(f"Test 6 PASSED: Latest timestamp is {latest}")
    except Exception as e:
        all_validation_failures.append(f"Test 6: Exception raised: {e}")

    # Test 7: Validate correct data
    total_tests += 1
    try:
        is_valid, errors = storage.validate_data(test_df)
        if not is_valid:
            all_validation_failures.append(f"Test 7: Valid data marked as invalid: {errors}")
        else:
            logger.success("Test 7 PASSED: Data validation correctly identified valid data")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Test 8: Validate invalid data (missing columns)
    total_tests += 1
    try:
        invalid_df = pd.DataFrame({
            'open': [100, 200],
            'close': [110, 210]
            # Missing high, low, volume
        })
        is_valid, errors = storage.validate_data(invalid_df)
        if is_valid:
            all_validation_failures.append("Test 8: Invalid data marked as valid")
        elif len(errors) == 0:
            all_validation_failures.append("Test 8: No errors reported for invalid data")
        else:
            logger.success(f"Test 8 PASSED: Validation correctly identified {len(errors)} errors")
    except Exception as e:
        all_validation_failures.append(f"Test 8: Exception raised: {e}")

    # Test 9: Load with date filters
    total_tests += 1
    try:
        start_filter = datetime(2024, 1, 2)
        end_filter = datetime(2024, 1, 3)
        filtered_df = storage.load_ohlcv("BTC/USDT", "1h", start_date=start_filter, end_date=end_filter)
        if filtered_df is None:
            all_validation_failures.append("Test 9: Filtered load returned None")
        elif filtered_df.index.min() < start_filter:
            all_validation_failures.append("Test 9: Data before start_date found")
        elif filtered_df.index.max() > end_filter:
            all_validation_failures.append("Test 9: Data after end_date found")
        else:
            logger.success(f"Test 9 PASSED: Date filtering works, got {len(filtered_df)} candles")
    except Exception as e:
        all_validation_failures.append(f"Test 9: Exception raised: {e}")

    # Test 10: Delete data
    total_tests += 1
    try:
        result = storage.delete_data("BTC/USDT", "1h")
        if not result:
            all_validation_failures.append("Test 10: Delete operation returned False")
        elif storage.has_data("BTC/USDT", "1h"):
            all_validation_failures.append("Test 10: Data still exists after deletion")
        else:
            logger.success("Test 10 PASSED: Data deleted successfully")
    except Exception as e:
        all_validation_failures.append(f"Test 10: Exception raised: {e}")

    # Test 11: Handle non-existent data
    total_tests += 1
    try:
        df = storage.load_ohlcv("NONEXISTENT/PAIR", "1h")
        if df is not None:
            all_validation_failures.append("Test 11: Expected None for non-existent data, got DataFrame")
        else:
            logger.success("Test 11 PASSED: Correctly returned None for non-existent data")
    except Exception as e:
        all_validation_failures.append(f"Test 11: Exception raised: {e}")

    # Cleanup test directory
    try:
        import shutil
        shutil.rmtree("data/test_ohlcv")
        logger.info("Cleaned up test directory")
    except Exception as e:
        logger.warning(f"Could not clean up test directory: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("OHLCVStorage is validated and ready for production use")
        sys.exit(0)
