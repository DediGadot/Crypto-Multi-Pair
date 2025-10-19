"""
Data Pipeline Abstraction

Unified interface for data fetching, feature augmentation, and caching.
Eliminates duplicated data pipeline code scattered across master.py and strategies.

**Purpose**: Provide a single, well-tested entry point for all data operations
with proper caching, feature enrichment, and horizon management.

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Usage**:
```python
from crypto_trader.data.pipeline import DataPipeline

# Create pipeline
pipeline = DataPipeline(
    fetcher=binance_fetcher,
    feature_factory=feature_factory
)

# Fetch data with features
data = pipeline.fetch(
    symbol='BTC/USDT',
    timeframe='1h',
    horizon_days=90,
    include_features=['onchain', 'sentiment']
)

# Multi-pair data for portfolio strategies
multi_data = pipeline.fetch_multi(
    symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
    timeframe='1h',
    horizon_days=180
)
```

**Expected Output**:
- DataFrame with OHLCV + features
- Automatic caching for repeated requests
- Consistent data quality across strategies
"""

from typing import Optional, Dict, List
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.features import factory as feature_factory
from crypto_trader.features.factory import DEFAULT_JOIN_CONFIG, FeatureJoinConfig


class DataPipeline:
    """
    Unified data pipeline for fetching and enriching market data.

    Handles all data operations with proper caching, validation,
    and feature augmentation.
    """

    def __init__(
        self,
        fetcher: BinanceDataFetcher,
        enable_cache: bool = True
    ):
        """
        Initialize data pipeline.

        Args:
            fetcher: Data fetcher instance (BinanceDataFetcher)
            enable_cache: Whether to use caching (default: True)
        """
        self.fetcher = fetcher
        self.enable_cache = enable_cache

        # In-memory cache for pipeline operations
        self._cache: Dict[str, pd.DataFrame] = {}

        logger.debug("DataPipeline initialized")

    def fetch(
        self,
        symbol: str,
        timeframe: str = '1h',
        horizon_days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_features: Optional[List[str]] = None,
        warmup_multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Fetch market data with optional feature augmentation.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '1d')
            horizon_days: Number of days to fetch (alternative to dates)
            start_date: Start date for data range
            end_date: End date for data range
            include_features: List of feature pillars to include
                             (e.g., ['onchain', 'sentiment', 'orderflow'])
            warmup_multiplier: Extra data multiplier for indicator warmup

        Returns:
            DataFrame with OHLCV data and optional features

        Example:
            >>> data = pipeline.fetch(
            ...     symbol='BTC/USDT',
            ...     horizon_days=90,
            ...     include_features=['onchain', 'sentiment']
            ... )
        """
        # Generate cache key
        cache_key = self._make_cache_key(
            symbol, timeframe, horizon_days, start_date, end_date, include_features
        )

        # Check cache
        if self.enable_cache and cache_key in self._cache:
            logger.debug(f"Serving cached data for {symbol}")
            return self._cache[cache_key].copy()

        # Calculate date range
        if horizon_days is not None:
            end_date = end_date or datetime.now()
            # Add warmup period
            total_days = int(horizon_days * warmup_multiplier)
            start_date = end_date - timedelta(days=total_days)

        # Fetch OHLCV data
        logger.debug(
            f"Fetching {symbol} {timeframe} data: "
            f"{start_date} to {end_date}"
        )

        try:
            data = self.fetcher.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise

        if data.empty:
            logger.warning(f"No data returned for {symbol}")
            return data

        logger.debug(f"Fetched {len(data)} candles for {symbol}")

        # Augment with features if requested
        if include_features:
            data = self._augment_features(data, symbol, timeframe, include_features)

        # Slice to exact horizon (remove warmup period)
        if horizon_days is not None and warmup_multiplier > 1.0:
            target_end = end_date or datetime.now()
            target_start = target_end - timedelta(days=horizon_days)
            data = data.loc[target_start:target_end]
            logger.debug(
                f"Sliced to exact horizon: {len(data)} candles "
                f"({horizon_days} days)"
            )

        # Cache result
        if self.enable_cache:
            self._cache[cache_key] = data.copy()

        return data

    def fetch_multi(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        horizon_days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_features: Optional[List[str]] = None,
        warmup_multiplier: float = 1.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Optimized for multi-pair strategies by fetching once and
        sharing across all strategies.

        Args:
            symbols: List of trading pair symbols
            timeframe: Candle timeframe
            horizon_days: Number of days to fetch
            start_date: Start date for data range
            end_date: End date for data range
            include_features: List of feature pillars to include
            warmup_multiplier: Extra data multiplier for warmup

        Returns:
            Dictionary mapping symbol to DataFrame

        Example:
            >>> multi_data = pipeline.fetch_multi(
            ...     symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            ...     horizon_days=180,
            ...     include_features=['onchain']
            ... )
        """
        result = {}

        logger.info(f"Fetching data for {len(symbols)} symbols")

        for symbol in symbols:
            try:
                data = self.fetch(
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon_days=horizon_days,
                    start_date=start_date,
                    end_date=end_date,
                    include_features=include_features,
                    warmup_multiplier=warmup_multiplier
                )
                result[symbol] = data
                logger.debug(f"  ✓ {symbol}: {len(data)} candles")
            except Exception as e:
                logger.warning(f"  ✗ {symbol}: {e}")

        logger.success(f"Fetched {len(result)}/{len(symbols)} symbols successfully")
        return result

    def clear_cache(self) -> None:
        """Clear the data pipeline cache."""
        self._cache.clear()
        logger.debug("DataPipeline cache cleared")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics

        Example:
            >>> stats = pipeline.get_cache_stats()
            >>> print(f"Cached items: {stats['item_count']}")
        """
        total_size = sum(len(df) for df in self._cache.values())
        return {
            'item_count': len(self._cache),
            'total_rows': total_size,
            'keys': list(self._cache.keys())
        }

    def _augment_features(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        pillars: List[str]
    ) -> pd.DataFrame:
        """
        Augment data with features from specified pillars.

        Args:
            data: OHLCV DataFrame
            symbol: Trading pair symbol
            pillars: List of feature pillars (onchain, sentiment, etc.)

        Returns:
            DataFrame with features joined
        """
        if not pillars:
            pillars = list(DEFAULT_JOIN_CONFIG.pillars)

        # Build a join config that respects the requested pillars
        staleness = {
            pillar: DEFAULT_JOIN_CONFIG.max_staleness.get(pillar, pd.Timedelta.max)
            for pillar in pillars
        }
        join_config = FeatureJoinConfig(
            pillars=list(pillars),
            max_staleness=staleness
        )

        try:
            augmented = feature_factory.augment_with_features(
                market_df=data,
                symbol=symbol,
                timeframe=timeframe,
                config=join_config
            )
            logger.debug(f"Augmented {symbol} with pillars: {', '.join(pillars)}")
            return augmented
        except Exception as e:
            logger.warning(f"Feature augmentation failed for {symbol}: {e}")
            return data

    def _make_cache_key(
        self,
        symbol: str,
        timeframe: str,
        horizon_days: Optional[int],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        features: Optional[List[str]]
    ) -> str:
        """Generate cache key from parameters."""
        parts = [
            symbol,
            timeframe,
            str(horizon_days) if horizon_days else '',
            start_date.isoformat() if start_date else '',
            end_date.isoformat() if end_date else '',
            ','.join(sorted(features)) if features else ''
        ]
        return '|'.join(parts)


if __name__ == "__main__":
    """
    Validation block for DataPipeline.
    Tests pipeline with mock fetcher.
    """
    import sys
    from unittest.mock import Mock
    import numpy as np

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating DataPipeline...\n")

    # Create mock fetcher
    mock_fetcher = Mock(spec=BinanceDataFetcher)

    def mock_get_ohlcv(symbol, timeframe, start=None, end=None):
        """Generate mock OHLCV data."""
        periods = 100
        dates = pd.date_range(end='2025-01-10', periods=periods, freq='1h')
        return pd.DataFrame({
            'open': 100 + np.random.randn(periods).cumsum(),
            'high': 102 + np.random.randn(periods).cumsum(),
            'low': 98 + np.random.randn(periods).cumsum(),
            'close': 100 + np.random.randn(periods).cumsum(),
            'volume': np.random.randint(1000, 5000, periods)
        }, index=dates)

    mock_fetcher.get_ohlcv.side_effect = mock_get_ohlcv

    # Mock the feature_factory.augment_with_features function
    original_augment = feature_factory.augment_with_features

    def mock_augment(data, pillar, symbol):
        """Add mock feature column."""
        result = data.copy()
        result[f'{pillar}_feature'] = np.random.randn(len(data))
        return result

    feature_factory.augment_with_features = mock_augment

    # Test 1: Basic fetch (without horizon slicing to keep data)
    total_tests += 1
    print("Test 1: Basic data fetch")
    try:
        pipeline = DataPipeline(
            fetcher=mock_fetcher
        )

        # Fetch without horizon_days to avoid slicing away all data
        data = pipeline.fetch(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 10)
        )

        if not isinstance(data, pd.DataFrame):
            all_validation_failures.append(
                f"Expected DataFrame, got {type(data)}"
            )
        elif data.empty:
            all_validation_failures.append("Data is empty")
        elif 'close' not in data.columns:
            all_validation_failures.append("Missing 'close' column")
        else:
            print(f"  ✓ Fetched {len(data)} candles")
            print(f"  ✓ Columns: {list(data.columns)}")
    except Exception as e:
        all_validation_failures.append(f"Test 1 failed: {e}")

    # Test 2: Fetch with features
    total_tests += 1
    print("\nTest 2: Fetch with features")
    try:
        data = pipeline.fetch(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 10),
            include_features=['onchain', 'sentiment']
        )

        if 'onchain_feature' not in data.columns:
            all_validation_failures.append("onchain_feature not added")
        elif 'sentiment_feature' not in data.columns:
            all_validation_failures.append("sentiment_feature not added")
        else:
            print(f"  ✓ Features added successfully")
            print(f"  ✓ Total columns: {len(data.columns)}")
    except Exception as e:
        all_validation_failures.append(f"Test 2 failed: {e}")

    # Test 3: Caching
    total_tests += 1
    print("\nTest 3: Caching functionality")
    try:
        # First fetch
        data1 = pipeline.fetch(
            symbol='ETH/USDT',
            timeframe='1h',
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 10)
        )

        # Second fetch (should be cached)
        data2 = pipeline.fetch(
            symbol='ETH/USDT',
            timeframe='1h',
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 10)
        )

        stats = pipeline.get_cache_stats()

        if stats['item_count'] == 0:
            all_validation_failures.append("Cache is empty")
        elif not data1.equals(data2):
            all_validation_failures.append("Cached data differs from original")
        else:
            print(f"  ✓ Caching working")
            print(f"  ✓ Cached items: {stats['item_count']}")
    except Exception as e:
        all_validation_failures.append(f"Test 3 failed: {e}")

    # Test 4: Multi-symbol fetch
    total_tests += 1
    print("\nTest 4: Multi-symbol fetch")
    try:
        multi_data = pipeline.fetch_multi(
            symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            timeframe='1h',
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 10)
        )

        if not isinstance(multi_data, dict):
            all_validation_failures.append(
                f"Expected dict, got {type(multi_data)}"
            )
        elif len(multi_data) != 3:
            all_validation_failures.append(
                f"Expected 3 symbols, got {len(multi_data)}"
            )
        elif not all(isinstance(df, pd.DataFrame) for df in multi_data.values()):
            all_validation_failures.append("Not all values are DataFrames")
        else:
            print(f"  ✓ Fetched {len(multi_data)} symbols")
            for symbol, df in multi_data.items():
                print(f"    - {symbol}: {len(df)} candles")
    except Exception as e:
        all_validation_failures.append(f"Test 4 failed: {e}")

    # Test 5: Cache clearing
    total_tests += 1
    print("\nTest 5: Cache clearing")
    try:
        pipeline.clear_cache()
        stats = pipeline.get_cache_stats()

        if stats['item_count'] != 0:
            all_validation_failures.append(
                f"Cache not cleared, still has {stats['item_count']} items"
            )
        else:
            print(f"  ✓ Cache cleared successfully")
    except Exception as e:
        all_validation_failures.append(f"Test 5 failed: {e}")

    # Restore original function
    feature_factory.augment_with_features = original_augment

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("DataPipeline is validated and ready for use")
        sys.exit(0)
