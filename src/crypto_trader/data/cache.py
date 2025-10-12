"""
In-Memory Cache for OHLCV and Indicator Data

This module provides simple in-memory caching to reduce redundant data fetches
and indicator calculations. Uses Python's built-in functools.lru_cache and
a custom TTL-based cache for time-sensitive data.

Purpose:
- Cache OHLCV DataFrames to reduce file I/O
- Cache indicator calculations to avoid recomputation
- Support TTL (Time To Live) for auto-expiration
- Thread-safe caching for concurrent access
- Memory-efficient with LRU eviction policy

Third-party documentation:
- functools: https://docs.python.org/3/library/functools.html
- threading: https://docs.python.org/3/library/threading.html

Sample Input:
    cache.set("BTC/USDT:1h:ohlcv", dataframe, ttl=300)
    df = cache.get("BTC/USDT:1h:ohlcv")

Expected Output:
    Cached data returned from memory instead of disk/network
"""

import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger


class TTLCache:
    """
    Thread-safe LRU cache with Time To Live (TTL) support.

    Stores key-value pairs with optional expiration times.
    Automatically evicts expired and least-recently-used items.
    """

    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        """
        Initialize the TTL cache.

        Args:
            max_size: Maximum number of items to store
            default_ttl: Default time to live in seconds (0 = no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._expiry = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        logger.info(f"Initialized TTLCache with max_size={max_size}, default_ttl={default_ttl}s")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            # Check if key exists
            if key not in self._cache:
                self._misses += 1
                return None

            # Check if expired
            if key in self._expiry:
                if time.time() > self._expiry[key]:
                    # Expired - remove and return None
                    del self._cache[key]
                    del self._expiry[key]
                    self._misses += 1
                    return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = use default, 0 = no expiration)
        """
        with self._lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl

            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
                if key in self._expiry:
                    del self._expiry[key]

            # Add to cache
            self._cache[key] = value

            # Set expiry if TTL > 0
            if ttl > 0:
                self._expiry[key] = time.time() + ttl

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Evict oldest if over capacity
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._expiry:
                    del self._expiry[oldest_key]
                logger.debug(f"Evicted cache key: {oldest_key}")

    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._expiry:
                    del self._expiry[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.2f}%",
                'total_requests': total_requests
            }

    def cleanup_expired(self) -> int:
        """
        Remove all expired items from cache.

        Returns:
            Number of items removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, expiry_time in self._expiry.items()
                if current_time > expiry_time
            ]

            for key in expired_keys:
                del self._cache[key]
                del self._expiry[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")

            return len(expired_keys)


class OHLCVCache:
    """
    Specialized cache for OHLCV and indicator data.

    Provides high-level interface for caching cryptocurrency data
    with appropriate TTL values and key naming conventions.
    """

    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        """
        Initialize OHLCV cache.

        Args:
            max_size: Maximum number of items to cache
            default_ttl: Default TTL in seconds for cached items
        """
        self._cache = TTLCache(max_size=max_size, default_ttl=default_ttl)
        logger.info("Initialized OHLCVCache")

    def _make_key(self, symbol: str, timeframe: str, data_type: str = "ohlcv") -> str:
        """
        Create standardized cache key.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            data_type: Type of data (e.g., "ohlcv", "indicators", "analysis")

        Returns:
            Cache key string
        """
        return f"{symbol}:{timeframe}:{data_type}"

    def get_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get cached OHLCV data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe

        Returns:
            Cached DataFrame or None
        """
        key = self._make_key(symbol, timeframe, "ohlcv")
        return self._cache.get(key)

    def set_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache OHLCV data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            df: DataFrame to cache
            ttl: Time to live in seconds (None = use default)
        """
        key = self._make_key(symbol, timeframe, "ohlcv")
        self._cache.set(key, df.copy(), ttl=ttl)
        logger.debug(f"Cached OHLCV data for {symbol} {timeframe}")

    def get_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str
    ) -> Optional[Any]:
        """
        Get cached indicator data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            indicator_name: Name of the indicator

        Returns:
            Cached indicator data or None
        """
        key = self._make_key(symbol, timeframe, f"indicator:{indicator_name}")
        return self._cache.get(key)

    def set_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        data: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache indicator data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            indicator_name: Name of the indicator
            data: Indicator data to cache
            ttl: Time to live in seconds (None = use default)
        """
        key = self._make_key(symbol, timeframe, f"indicator:{indicator_name}")
        self._cache.set(key, data, ttl=ttl)
        logger.debug(f"Cached indicator {indicator_name} for {symbol} {timeframe}")

    def invalidate_symbol(self, symbol: str, timeframe: Optional[str] = None) -> int:
        """
        Invalidate all cached data for a symbol.

        Args:
            symbol: Trading pair symbol
            timeframe: Optional specific timeframe (None = all timeframes)

        Returns:
            Number of cache entries invalidated
        """
        # Remove expired entries first so stats remain accurate
        self._cache.cleanup_expired()

        if timeframe:
            prefix = f"{symbol}:{timeframe}:"
        else:
            prefix = f"{symbol}:"

        removed = 0
        with self._cache._lock:
            keys_to_delete = [key for key in list(self._cache._cache.keys()) if key.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache._cache[key]
                if key in self._cache._expiry:
                    del self._cache._expiry[key]
                removed += 1

        if removed:
            logger.info(f"Invalidated {removed} cache entr{'y' if removed == 1 else 'ies'} for {symbol}{'' if not timeframe else f' {timeframe}'}")
        else:
            logger.debug(f"No cache entries found for {symbol}{'' if not timeframe else f' {timeframe}'}")

        return removed

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.get_stats()

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


def cached(ttl: int = 300, cache_instance: Optional[TTLCache] = None):
    """
    Decorator for caching function results with TTL.

    Args:
        ttl: Time to live in seconds
        cache_instance: Optional cache instance to use (creates new if None)

    Returns:
        Decorated function with caching

    Example:
        @cached(ttl=60)
        def expensive_calculation(x, y):
            return x * y + compute_complex_stuff()
    """
    # Create cache instance if not provided
    if cache_instance is None:
        cache_instance = TTLCache(max_size=128, default_ttl=ttl)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try to get from cache
            result = cache_instance.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl=ttl)
            logger.debug(f"Cache miss for {func.__name__}, computed and cached")
            return result

        # Add cache control methods
        wrapper.cache_clear = cache_instance.clear
        wrapper.cache_stats = cache_instance.get_stats

        return wrapper

    return decorator


if __name__ == "__main__":
    """
    Validation: Test cache functionality with real operations.

    Tests:
    1. TTLCache basic operations (get/set/delete)
    2. TTL expiration
    3. LRU eviction when max size exceeded
    4. Thread safety (concurrent access)
    5. Cache statistics
    6. OHLCVCache specialized operations
    7. Cached decorator functionality
    """
    import sys
    import numpy as np

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting cache validation")

    # Test 1: Basic TTLCache operations
    total_tests += 1
    try:
        cache = TTLCache(max_size=10, default_ttl=60)
        cache.set("key1", "value1")
        result = cache.get("key1")
        if result != "value1":
            all_validation_failures.append(f"Test 1: Expected 'value1', got {result}")
        else:
            logger.success("Test 1 PASSED: Basic get/set works")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: TTL expiration
    total_tests += 1
    try:
        cache = TTLCache(max_size=10, default_ttl=1)
        cache.set("expires", "soon", ttl=1)
        time.sleep(1.5)
        result = cache.get("expires")
        if result is not None:
            all_validation_failures.append(f"Test 2: Expected None for expired key, got {result}")
        else:
            logger.success("Test 2 PASSED: TTL expiration works")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: LRU eviction
    total_tests += 1
    try:
        cache = TTLCache(max_size=3, default_ttl=0)  # No TTL, only LRU
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1

        if cache.get("key1") is not None:
            all_validation_failures.append("Test 3: key1 should have been evicted")
        elif cache.get("key4") is None:
            all_validation_failures.append("Test 3: key4 should be in cache")
        else:
            logger.success("Test 3 PASSED: LRU eviction works")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Delete operation
    total_tests += 1
    try:
        cache = TTLCache(max_size=10, default_ttl=0)
        cache.set("delete_me", "value")
        result = cache.delete("delete_me")
        if not result:
            all_validation_failures.append("Test 4: Delete returned False for existing key")
        elif cache.get("delete_me") is not None:
            all_validation_failures.append("Test 4: Key still exists after deletion")
        else:
            logger.success("Test 4 PASSED: Delete operation works")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception raised: {e}")

    # Test 5: Cache statistics
    total_tests += 1
    try:
        cache = TTLCache(max_size=10, default_ttl=0)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Generate hits and misses
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("key3")  # miss

        stats = cache.get_stats()
        if stats['hits'] != 2:
            all_validation_failures.append(f"Test 5: Expected 2 hits, got {stats['hits']}")
        elif stats['misses'] != 1:
            all_validation_failures.append(f"Test 5: Expected 1 miss, got {stats['misses']}")
        elif stats['size'] != 2:
            all_validation_failures.append(f"Test 5: Expected size 2, got {stats['size']}")
        else:
            logger.success(f"Test 5 PASSED: Cache stats - {stats}")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Clear cache
    total_tests += 1
    try:
        cache = TTLCache(max_size=10, default_ttl=0)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        stats = cache.get_stats()
        if stats['size'] != 0:
            all_validation_failures.append(f"Test 6: Cache not empty after clear, size={stats['size']}")
        else:
            logger.success("Test 6 PASSED: Clear operation works")
    except Exception as e:
        all_validation_failures.append(f"Test 6: Exception raised: {e}")

    # Test 7: OHLCVCache operations
    total_tests += 1
    try:
        ohlcv_cache = OHLCVCache(max_size=10, default_ttl=60)

        # Create test DataFrame
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
        test_df = pd.DataFrame({
            'open': np.random.uniform(40000, 42000, 10),
            'high': np.random.uniform(42000, 43000, 10),
            'low': np.random.uniform(39000, 40000, 10),
            'close': np.random.uniform(40000, 42000, 10),
            'volume': np.random.uniform(100, 1000, 10)
        }, index=dates)

        # Cache and retrieve
        ohlcv_cache.set_ohlcv("BTC/USDT", "1h", test_df)
        cached_df = ohlcv_cache.get_ohlcv("BTC/USDT", "1h")

        if cached_df is None:
            all_validation_failures.append("Test 7: Cached DataFrame is None")
        elif len(cached_df) != len(test_df):
            all_validation_failures.append(f"Test 7: Expected {len(test_df)} rows, got {len(cached_df)}")
        else:
            logger.success("Test 7 PASSED: OHLCVCache operations work")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Test 8: Cache indicator data
    total_tests += 1
    try:
        ohlcv_cache = OHLCVCache(max_size=10, default_ttl=60)
        indicator_data = {"rsi": [45.2, 52.1, 48.9], "macd": [0.5, -0.3, 0.2]}

        ohlcv_cache.set_indicator("BTC/USDT", "1h", "rsi_macd", indicator_data)
        cached_indicator = ohlcv_cache.get_indicator("BTC/USDT", "1h", "rsi_macd")

        if cached_indicator is None:
            all_validation_failures.append("Test 8: Cached indicator is None")
        elif cached_indicator != indicator_data:
            all_validation_failures.append(f"Test 8: Indicator data mismatch")
        else:
            logger.success("Test 8 PASSED: Indicator caching works")
    except Exception as e:
        all_validation_failures.append(f"Test 8: Exception raised: {e}")

    # Test 9: Cached decorator
    total_tests += 1
    try:
        # Use a mutable container to track call count
        call_tracker = {'count': 0}

        @cached(ttl=60)
        def expensive_function(x: int, y: int) -> int:
            call_tracker['count'] += 1
            return x + y

        # First call - should execute function
        result1 = expensive_function(1, 2)
        # Second call - should use cache
        result2 = expensive_function(1, 2)

        if result1 != 3 or result2 != 3:
            all_validation_failures.append(f"Test 9: Expected result 3, got {result1} and {result2}")
        elif call_tracker['count'] != 1:
            all_validation_failures.append(f"Test 9: Expected 1 function call, got {call_tracker['count']}")
        else:
            logger.success("Test 9 PASSED: Cached decorator works")
    except Exception as e:
        all_validation_failures.append(f"Test 9: Exception raised: {e}")

    # Test 10: Cleanup expired entries
    total_tests += 1
    try:
        cache = TTLCache(max_size=10, default_ttl=1)
        cache.set("key1", "value1", ttl=1)
        cache.set("key2", "value2", ttl=1)
        cache.set("key3", "value3", ttl=0)  # No expiry

        time.sleep(1.5)
        removed = cache.cleanup_expired()

        if removed != 2:
            all_validation_failures.append(f"Test 10: Expected 2 expired items, removed {removed}")
        elif cache.get("key3") is None:
            all_validation_failures.append("Test 10: Non-expired key was removed")
        else:
            logger.success("Test 10 PASSED: Cleanup expired entries works")
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
        print("Cache module is validated and ready for production use")
        sys.exit(0)
