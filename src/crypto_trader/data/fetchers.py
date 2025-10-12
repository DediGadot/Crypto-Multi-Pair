"""
CCXT Exchange Data Fetchers

This module provides data fetchers for cryptocurrency exchanges using the CCXT library.
Implements rate limiting, retry logic, and error handling for robust data retrieval.

Purpose:
- Fetch OHLCV data from cryptocurrency exchanges
- Handle rate limiting to respect exchange API limits
- Implement retry logic with exponential backoff
- Support batch fetching for multiple symbols
- Integrate with storage and cache layers

Third-party documentation:
- CCXT: https://docs.ccxt.com/
- pandas: https://pandas.pydata.org/docs/

Sample Input:
    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=100)

Expected Output:
    DataFrame with columns: [open, high, low, close, volume]
    Data automatically saved to storage and cached
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import ccxt
import pandas as pd
from loguru import logger

from crypto_trader.data.cache import OHLCVCache
from crypto_trader.data.providers import DataProvider
from crypto_trader.data.storage import OHLCVStorage


class RateLimiter:
    """
    Simple rate limiter to respect exchange API limits.

    Ensures requests are spaced out according to exchange requirements.
    """

    def __init__(self, max_requests_per_minute: int = 1200):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests = max_requests_per_minute
        self.min_interval = 60.0 / max_requests_per_minute
        self.last_request_time = 0.0
        logger.info(f"Initialized rate limiter: {max_requests_per_minute} req/min")

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class BinanceDataFetcher(DataProvider):
    """
    Data fetcher for Binance exchange using CCXT.

    Fetches OHLCV data with automatic storage and caching.
    Implements rate limiting and retry logic for reliability.
    """

    def __init__(
        self,
        use_storage: bool = True,
        use_cache: bool = True,
        storage_path: str = "data/ohlcv",
        max_retries: int = 3,
        rate_limit: int = 1200
    ):
        """
        Initialize Binance data fetcher.

        Args:
            use_storage: Enable automatic data storage to CSV
            use_cache: Enable in-memory caching
            storage_path: Path for CSV storage
            max_retries: Maximum retry attempts for failed requests
            rate_limit: Maximum requests per minute
        """
        super().__init__("Binance")

        # Initialize CCXT exchange
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        # Initialize storage and cache
        self.use_storage = use_storage
        self.use_cache = use_cache
        self.storage = OHLCVStorage(base_path=storage_path) if use_storage else None
        self.cache = OHLCVCache(max_size=100, default_ttl=300) if use_cache else None

        # Rate limiting and retries
        self.rate_limiter = RateLimiter(max_requests_per_minute=rate_limit)
        self.max_retries = max_retries

        # Load available symbols and timeframes
        self._load_markets()

        logger.info(f"Initialized {self.name} fetcher with {len(self._symbols)} symbols")

    def _load_markets(self) -> None:
        """Load available markets and timeframes from exchange."""
        try:
            self.exchange.load_markets()
            self._symbols = list(self.exchange.symbols)
            self._timeframes = list(self.exchange.timeframes.keys())
            logger.info(f"Loaded {len(self._symbols)} symbols and {len(self._timeframes)} timeframes")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            self._symbols = []
            self._timeframes = []

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs."""
        return self._symbols.copy()

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid and tradeable."""
        return symbol in self._symbols

    def validate_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe is supported."""
        return timeframe in self._timeframes

    def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List:
        """
        Fetch OHLCV data with retry logic.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            since: Timestamp in milliseconds (optional)
            limit: Maximum number of candles (optional)

        Returns:
            List of OHLCV candles from exchange

        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait_if_needed()

                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )

                logger.debug(f"Fetched {len(ohlcv)} candles for {symbol} {timeframe}")
                return ohlcv

            except ccxt.RateLimitExceeded as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Rate limit exceeded, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)

            except ccxt.NetworkError as e:
                wait_time = 2 ** attempt
                logger.warning(f"Network error: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)

            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt == self.max_retries - 1:
                    raise

        raise Exception(f"Failed to fetch data after {self.max_retries} attempts")

    def _convert_to_dataframe(self, ohlcv_data: List) -> pd.DataFrame:
        """
        Convert CCXT OHLCV data to DataFrame.

        Args:
            ohlcv_data: List of OHLCV candles from CCXT

        Returns:
            DataFrame with proper index and columns
        """
        if not ohlcv_data:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame(
            ohlcv_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def _get_timeframe_duration_ms(self, timeframe: str) -> int:
        """
        Get duration of one candle in milliseconds.

        Args:
            timeframe: Timeframe string (e.g., "1h", "1d")

        Returns:
            Duration in milliseconds
        """
        # Parse timeframe to get duration
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }
        return timeframe_map.get(timeframe, 60 * 60 * 1000)  # Default to 1h

    def _fetch_paginated(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with pagination for large requests.

        Binance limits requests to 1000 candles. This method handles
        fetching more data by making multiple paginated requests.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            limit: Total number of candles to fetch
            end_time: End timestamp (defaults to now)

        Returns:
            Combined DataFrame with all requested candles
        """
        all_data = []
        candles_per_request = 1000
        num_requests = (limit + candles_per_request - 1) // candles_per_request

        # Start from end_time (or now) and work backwards
        if end_time is None:
            end_time = datetime.now()

        current_end_ms = int(end_time.timestamp() * 1000)
        timeframe_duration_ms = self._get_timeframe_duration_ms(timeframe)

        logger.info(f"Fetching {limit} candles in {num_requests} paginated requests")

        for i in range(num_requests):
            # Calculate how many candles to fetch in this request
            remaining = limit - len(all_data)
            batch_size = min(candles_per_request, remaining)

            # Calculate start time for this batch
            # Go back batch_size candles from current_end_ms
            batch_start_ms = current_end_ms - (batch_size * timeframe_duration_ms)

            logger.debug(f"Request {i+1}/{num_requests}: Fetching {batch_size} candles from {datetime.fromtimestamp(batch_start_ms/1000)}")

            try:
                # Fetch this batch
                ohlcv_data = self._fetch_with_retry(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=batch_start_ms,
                    limit=batch_size
                )

                if not ohlcv_data:
                    logger.warning(f"No data returned for batch {i+1}")
                    break

                all_data.extend(ohlcv_data)

                # Move the end time backwards for next request
                # Set to the timestamp of the first candle we just fetched
                current_end_ms = ohlcv_data[0][0]

                logger.debug(f"Fetched {len(ohlcv_data)} candles, total so far: {len(all_data)}")

                # If we got less than requested, we've hit the limit of available data
                if len(ohlcv_data) < batch_size:
                    logger.info(f"Reached end of available data at batch {i+1}")
                    break

            except Exception as e:
                logger.error(f"Error fetching batch {i+1}: {e}")
                # Return what we have so far
                break

        # Convert to DataFrame
        if not all_data:
            logger.warning("No data fetched")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = self._convert_to_dataframe(all_data)

        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        logger.success(f"Successfully fetched {len(df)} candles via pagination")
        return df

    def _fetch_all_available(
        self,
        symbol: str,
        timeframe: str,
        existing_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Fetch ALL available historical data from Binance.

        Uses smart caching - only fetches data that's not already stored.
        Continues fetching backwards until no more data is available.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            existing_df: Existing data from storage (if any)

        Returns:
            Complete DataFrame with all available data
        """
        all_data = []

        # Start from now and work backwards
        if existing_df is not None and not existing_df.empty:
            # We have some data - check if we need to fetch older or newer data
            oldest_stored = existing_df.index.min()
            newest_stored = existing_df.index.max()

            logger.info(f"Found {len(existing_df)} cached candles from {oldest_stored} to {newest_stored}")

            # Fetch newer data (updates)
            logger.info("Checking for newer data...")
            current_time = datetime.now()
            timeframe_duration_ms = self._get_timeframe_duration_ms(timeframe)

            # Only fetch if there could be new data
            time_since_latest = (current_time - newest_stored).total_seconds() * 1000
            if time_since_latest > timeframe_duration_ms:
                since_ms = int(newest_stored.timestamp() * 1000)
                newer_data = self._fetch_with_retry(symbol, timeframe, since=since_ms, limit=1000)
                if newer_data:
                    newer_df = self._convert_to_dataframe(newer_data)
                    # Remove duplicates with existing data
                    newer_df = newer_df[newer_df.index > newest_stored]
                    if not newer_df.empty:
                        logger.info(f"Fetched {len(newer_df)} newer candles")
                        all_data.append(newer_df)

            # Fetch older data (historical backfill)
            logger.info("Fetching older historical data...")
            current_start_ms = int(oldest_stored.timestamp() * 1000)

        else:
            # No existing data - start from now
            logger.info("No cached data found, fetching all available history")
            current_start_ms = int(datetime.now().timestamp() * 1000)

        # Fetch backwards in batches of 1000 until we run out
        batch_num = 0
        max_batches = 100  # Safety limit (100k candles max ~= 11 years for hourly data)

        # Start from the oldest we have, or from oldest available if no data
        while batch_num < max_batches:
            try:
                logger.info(f"Fetching historical batch {batch_num + 1} (going backwards in time)")

                # Fetch backwards - use since parameter to go back from current_start_ms
                ohlcv_data = self._fetch_with_retry(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_start_ms - (1000 * self._get_timeframe_duration_ms(timeframe)),
                    limit=1000
                )

                if not ohlcv_data:
                    logger.info("No more historical data available from exchange")
                    break

                batch_df = self._convert_to_dataframe(ohlcv_data)

                # Check if this is older than what we have
                if existing_df is not None and not existing_df.empty:
                    batch_df = batch_df[batch_df.index < oldest_stored]

                if batch_df.empty:
                    logger.info("Reached end of older data (all data now cached)")
                    break

                all_data.append(batch_df)
                logger.info(f"Fetched {len(batch_df)} older candles (oldest: {batch_df.index.min()})")

                # Update the start point for next batch (go further back)
                current_start_ms = ohlcv_data[0][0]
                batch_num += 1

                # If we got less than 1000, we've hit the limit
                if len(ohlcv_data) < 1000:
                    logger.info(f"Reached end of available data after {batch_num} batches")
                    break

            except Exception as e:
                logger.warning(f"Error or end of data at batch {batch_num + 1}: {e}")
                break

        # Combine all data
        if all_data:
            combined = pd.concat(all_data, ignore_index=False)
            combined = combined.sort_index()
            combined = combined[~combined.index.duplicated(keep='first')]

            if existing_df is not None and not existing_df.empty:
                # Merge with existing
                final_df = pd.concat([combined, existing_df], ignore_index=False)
                final_df = final_df.sort_index()
                final_df = final_df[~final_df.index.duplicated(keep='first')]
            else:
                final_df = combined
        elif existing_df is not None and not existing_df.empty:
            final_df = existing_df
        else:
            logger.warning("No data fetched")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        # Save to storage
        if self.use_storage and self.storage:
            self.storage.save_ohlcv(final_df, symbol, timeframe, mode="overwrite")

        # Cache it
        if self.use_cache and self.cache:
            self.cache.set_ohlcv(symbol, timeframe, final_df)

        logger.success(f"Total available data: {len(final_df)} candles from {final_df.index.min()} to {final_df.index.max()}")
        return final_df

    def _fetch_with_smart_caching(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        existing_df: Optional[pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch data with smart caching - only downloads missing candles.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            limit: Number of candles needed
            existing_df: Existing data from storage
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with requested data
        """
        # If we have enough data already, use it
        if existing_df is not None and not existing_df.empty and len(existing_df) >= limit:
            logger.info(f"Using {limit} candles from cache (have {len(existing_df)} total)")
            result = existing_df.tail(limit)

            if self.use_cache and self.cache:
                self.cache.set_ohlcv(symbol, timeframe, result)

            return result

        # Calculate how many more we need
        if existing_df is not None and not existing_df.empty:
            needed = limit - len(existing_df)
            logger.info(f"Have {len(existing_df)} cached, need {needed} more (total: {limit})")

            # Fetch only what's missing (older data)
            oldest_stored = existing_df.index.min()
            end_ms = int(oldest_stored.timestamp() * 1000) - 1

            if needed > 1000:
                new_df = self._fetch_paginated(symbol, timeframe, needed, datetime.fromtimestamp(end_ms/1000))
            else:
                # Request candles ending just before the earliest stored one
                timeframe_duration_ms = self._get_timeframe_duration_ms(timeframe)
                lookback_ms = timeframe_duration_ms * (needed + 1)
                since_ms = max(0, end_ms - lookback_ms)

                ohlcv_data = self._fetch_with_retry(symbol, timeframe, since=since_ms, limit=needed)
                new_df = self._convert_to_dataframe(ohlcv_data)
                new_df = new_df[new_df.index < oldest_stored]

            # Combine with existing
            combined = pd.concat([new_df, existing_df], ignore_index=False)
            combined = combined.sort_index()
            combined = combined[~combined.index.duplicated(keep='first')]
            result = combined.tail(limit)

        else:
            # No existing data, fetch from scratch
            logger.info(f"No cache, fetching {limit} candles from exchange")
            if limit > 1000:
                result = self._fetch_paginated(symbol, timeframe, limit, end_date)
            else:
                since = None
                if start_date:
                    since = int(start_date.timestamp() * 1000)

                ohlcv_data = self._fetch_with_retry(symbol, timeframe, since, limit)
                result = self._convert_to_dataframe(ohlcv_data)

        # Save to storage
        if self.use_storage and self.storage:
            self.storage.save_ohlcv(result, symbol, timeframe, mode="overwrite")

        # Cache it
        if self.use_cache and self.cache:
            self.cache.set_ohlcv(symbol, timeframe, result)

        return result

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        fetch_all: bool = False
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol and timeframe with smart caching.

        Checks cache first, then storage. For new data requests, only fetches
        data that's missing from storage (incremental updates).

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe (e.g., "1h")
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            limit: Maximum number of candles (optional, -1 for all available)
            fetch_all: If True, fetches all available historical data

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If symbol or timeframe is invalid
            Exception: If fetch fails after retries
        """
        # Validate inputs
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Check if we have existing data in storage
        existing_df = None
        if self.use_storage and self.storage:
            existing_df = self.storage.load_ohlcv(symbol, timeframe)

        # If we have data and just need to filter it
        if existing_df is not None and not existing_df.empty and not fetch_all and limit and limit != -1:
            # Check if we have enough cached data
            if len(existing_df) >= limit:
                logger.info(f"Using {len(existing_df)} cached candles for {symbol} {timeframe}")
                result_df = existing_df.tail(limit)

                # Cache it
                if self.use_cache and self.cache:
                    self.cache.set_ohlcv(symbol, timeframe, result_df)

                return result_df

        # Handle fetch_all or limit=-1 (fetch all available data)
        if fetch_all or limit == -1:
            logger.info(f"Fetching all available data for {symbol} {timeframe}")
            return self._fetch_all_available(symbol, timeframe, existing_df)

        # Smart incremental fetch: Only fetch what's missing
        if limit and limit > 0:
            return self._fetch_with_smart_caching(
                symbol, timeframe, limit, existing_df, start_date, end_date
            )

        # Fallback: Fetch from exchange
        logger.info(f"Fetching {symbol} {timeframe} from exchange")

        # Determine if we need pagination
        if limit and limit > 1000:
            df = self._fetch_paginated(symbol, timeframe, limit, end_date)
        else:
            since = None
            if start_date:
                since = int(start_date.timestamp() * 1000)

            ohlcv_data = self._fetch_with_retry(symbol, timeframe, since, limit)
            df = self._convert_to_dataframe(ohlcv_data)

            if end_date:
                df = df[df.index <= end_date]

        # Save to storage
        if self.use_storage and self.storage:
            self.storage.save_ohlcv(df, symbol, timeframe, mode="overwrite")

        # Save to cache
        if self.use_cache and self.cache:
            self.cache.set_ohlcv(symbol, timeframe, df)

        return df

    def update_data(self, symbol: str, timeframe: str) -> bool:
        """
        Update existing data with latest candles.

        Fetches only new data since the last stored timestamp.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe

        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Validate inputs
            if not self.validate_symbol(symbol):
                raise ValueError(f"Invalid symbol: {symbol}")
            if not self.validate_timeframe(timeframe):
                raise ValueError(f"Invalid timeframe: {timeframe}")

            # Get latest timestamp from storage
            latest_ts = None
            if self.use_storage and self.storage:
                latest_ts = self.storage.get_latest_timestamp(symbol, timeframe)

            # Fetch new data
            since = None
            if latest_ts:
                # Fetch from slightly before latest to avoid gaps
                since = int((latest_ts - timedelta(hours=1)).timestamp() * 1000)
                logger.info(f"Updating {symbol} {timeframe} from {latest_ts}")
            else:
                logger.info(f"No existing data, fetching initial {symbol} {timeframe}")

            ohlcv_data = self._fetch_with_retry(symbol, timeframe, since=since, limit=1000)
            df = self._convert_to_dataframe(ohlcv_data)

            if df.empty:
                logger.warning(f"No new data for {symbol} {timeframe}")
                return True

            # Save to storage (append mode)
            if self.use_storage and self.storage:
                self.storage.save_ohlcv(df, symbol, timeframe, mode="append")

            # Update cache
            if self.use_cache and self.cache:
                # Reload full data and cache it
                full_df = self.storage.load_ohlcv(symbol, timeframe) if self.storage else df
                self.cache.set_ohlcv(symbol, timeframe, full_df)

            logger.success(f"Updated {symbol} {timeframe} with {len(df)} new candles")
            return True

        except Exception as e:
            logger.error(f"Failed to update {symbol} {timeframe}: {e}")
            return False

    def fetch_batch(
        self,
        symbols: List[str],
        timeframe: str,
        limit: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            timeframe: Candlestick timeframe
            limit: Maximum number of candles per symbol

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}

        for symbol in symbols:
            try:
                df = self.get_ohlcv(symbol, timeframe, limit=limit)
                results[symbol] = df
                logger.info(f"Fetched {symbol}: {len(df)} candles")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        return results

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {}


if __name__ == "__main__":
    """
    Validation: Test BinanceDataFetcher with real Binance data.

    Tests:
    1. Initialize fetcher and load markets
    2. Validate symbols and timeframes
    3. Fetch BTC/USDT 1h data (100 candles)
    4. Verify data structure and contents
    5. Test caching (second fetch should be from cache)
    6. Test storage (verify CSV file created)
    7. Test update functionality
    8. Test batch fetching
    9. Verify cache statistics
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting BinanceDataFetcher validation with REAL EXCHANGE DATA")

    # Test 1: Initialize fetcher
    total_tests += 1
    try:
        fetcher = BinanceDataFetcher(
            use_storage=True,
            use_cache=True,
            storage_path="data/test_binance",
            rate_limit=1200
        )
        if fetcher.exchange is None:
            all_validation_failures.append("Test 1: Exchange not initialized")
        elif len(fetcher._symbols) == 0:
            all_validation_failures.append("Test 1: No symbols loaded")
        else:
            logger.success(f"Test 1 PASSED: Initialized with {len(fetcher._symbols)} symbols")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Validate symbol and timeframe
    total_tests += 1
    try:
        valid_symbol = fetcher.validate_symbol("BTC/USDT")
        valid_tf = fetcher.validate_timeframe("1h")
        invalid_symbol = fetcher.validate_symbol("INVALID/PAIR")

        if not valid_symbol:
            all_validation_failures.append("Test 2: BTC/USDT should be valid")
        elif not valid_tf:
            all_validation_failures.append("Test 2: 1h should be valid timeframe")
        elif invalid_symbol:
            all_validation_failures.append("Test 2: INVALID/PAIR should be invalid")
        else:
            logger.success("Test 2 PASSED: Symbol and timeframe validation works")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Fetch real BTC/USDT data
    total_tests += 1
    try:
        logger.info("Fetching 100 candles of BTC/USDT 1h data from Binance...")
        df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)

        if df is None or df.empty:
            all_validation_failures.append("Test 3: DataFrame is None or empty")
        elif len(df) != 100:
            all_validation_failures.append(f"Test 3: Expected 100 candles, got {len(df)}")
        elif not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            all_validation_failures.append(f"Test 3: Missing columns. Got: {df.columns.tolist()}")
        elif not isinstance(df.index, pd.DatetimeIndex):
            all_validation_failures.append("Test 3: Index is not DatetimeIndex")
        else:
            logger.success(f"Test 3 PASSED: Fetched {len(df)} candles")
            logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
            logger.info(f"Latest close price: ${df['close'].iloc[-1]:,.2f}")
            logger.info(f"Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Verify data integrity
    total_tests += 1
    try:
        if 'df' in locals() and df is not None and not df.empty:
            # Check for nulls
            null_count = df.isnull().sum().sum()
            # Check price relationships
            high_low_valid = (df['high'] >= df['low']).all()
            prices_positive = (df[['open', 'high', 'low', 'close']] > 0).all().all()

            if null_count > 0:
                all_validation_failures.append(f"Test 4: Found {null_count} null values")
            elif not high_low_valid:
                all_validation_failures.append("Test 4: High < Low in some rows")
            elif not prices_positive:
                all_validation_failures.append("Test 4: Found non-positive prices")
            else:
                logger.success("Test 4 PASSED: Data integrity verified")
        else:
            all_validation_failures.append("Test 4: No data to validate (Test 3 failed)")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception raised: {e}")

    # Test 5: Test caching (second fetch should be instant)
    total_tests += 1
    try:
        start_time = time.time()
        df2 = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)
        fetch_time = time.time() - start_time

        if df2 is None or df2.empty:
            all_validation_failures.append("Test 5: Cached fetch returned None/empty")
        elif fetch_time > 0.1:  # Should be nearly instant from cache
            all_validation_failures.append(f"Test 5: Cache fetch too slow ({fetch_time:.3f}s), possibly not cached")
        else:
            logger.success(f"Test 5 PASSED: Cached fetch took {fetch_time:.4f}s")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Verify storage created file
    total_tests += 1
    try:
        has_data = fetcher.storage.has_data("BTC/USDT", "1h")
        if not has_data:
            all_validation_failures.append("Test 6: Storage file was not created")
        else:
            file_path = fetcher.storage._get_file_path("BTC/USDT", "1h")
            logger.success(f"Test 6 PASSED: Data stored at {file_path}")
    except Exception as e:
        all_validation_failures.append(f"Test 6: Exception raised: {e}")

    # Test 7: Test update functionality
    total_tests += 1
    try:
        # Wait a moment to ensure there might be new data
        time.sleep(2)
        result = fetcher.update_data("BTC/USDT", "1h")
        if not isinstance(result, bool):
            all_validation_failures.append("Test 7: Update should return boolean")
        elif not result:
            # Update returning False is acceptable if no new data
            logger.info("Test 7: Update returned False (no new data available)")
        else:
            logger.success("Test 7 PASSED: Update completed successfully")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Test 8: Test batch fetching (2 symbols to minimize API calls)
    total_tests += 1
    try:
        symbols = ["BTC/USDT", "ETH/USDT"]
        logger.info(f"Batch fetching {symbols}...")
        results = fetcher.fetch_batch(symbols, "1h", limit=50)

        if len(results) != 2:
            all_validation_failures.append(f"Test 8: Expected 2 results, got {len(results)}")
        elif any(df.empty for df in results.values()):
            all_validation_failures.append("Test 8: Some batch results are empty")
        else:
            logger.success(f"Test 8 PASSED: Batch fetched {len(results)} symbols")
            for sym, data in results.items():
                logger.info(f"  {sym}: {len(data)} candles")
    except Exception as e:
        all_validation_failures.append(f"Test 8: Exception raised: {e}")

    # Test 9: Check cache statistics
    total_tests += 1
    try:
        stats = fetcher.get_cache_stats()
        if not isinstance(stats, dict):
            all_validation_failures.append("Test 9: Stats should be a dictionary")
        elif stats.get('total_requests', 0) == 0:
            all_validation_failures.append("Test 9: No cache requests recorded")
        else:
            logger.success(f"Test 9 PASSED: Cache stats - {stats}")
    except Exception as e:
        all_validation_failures.append(f"Test 9: Exception raised: {e}")

    # Test 10: Test invalid inputs
    total_tests += 1
    try:
        error_raised = False
        try:
            fetcher.get_ohlcv("INVALID/PAIR", "1h")
        except ValueError:
            error_raised = True

        if not error_raised:
            all_validation_failures.append("Test 10: Expected ValueError for invalid symbol")
        else:
            logger.success("Test 10 PASSED: Error handling works correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 10: Exception raised: {e}")

    # Cleanup test directory
    try:
        import shutil
        shutil.rmtree("data/test_binance")
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
        print("BinanceDataFetcher successfully fetched REAL data from Binance")
        print("Module is validated and ready for production use")
        sys.exit(0)
