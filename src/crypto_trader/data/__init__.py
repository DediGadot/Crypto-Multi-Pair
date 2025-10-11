"""
Data Layer for Crypto Trading System

This package provides comprehensive data management for cryptocurrency trading,
including fetching, storage, and caching of OHLCV data from exchanges.

Components:
- providers: Abstract interface for data providers
- fetchers: CCXT-based exchange data fetchers
- storage: File-based CSV storage for historical data
- cache: In-memory caching with TTL support

Usage Example:
    from crypto_trader.data import BinanceDataFetcher

    # Initialize fetcher with storage and caching
    fetcher = BinanceDataFetcher()

    # Fetch OHLCV data
    df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)

    # Update with latest data
    fetcher.update_data("BTC/USDT", "1h")

    # Batch fetch multiple symbols
    data = fetcher.fetch_batch(["BTC/USDT", "ETH/USDT"], "1h")
"""

from crypto_trader.data.cache import OHLCVCache, TTLCache, cached
from crypto_trader.data.fetchers import BinanceDataFetcher, RateLimiter
from crypto_trader.data.providers import DataProvider, MockDataProvider
from crypto_trader.data.storage import OHLCVStorage

__all__ = [
    # Core classes
    "DataProvider",
    "BinanceDataFetcher",
    "OHLCVStorage",
    "OHLCVCache",
    # Utilities
    "RateLimiter",
    "TTLCache",
    "cached",
    # Testing
    "MockDataProvider",
]

__version__ = "0.1.0"
