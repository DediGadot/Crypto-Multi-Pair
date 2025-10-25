"""
Windowed Results Cache

This module provides efficient caching of backtest results for windowed analysis.
Avoids recomputation of already-tested windows.

**Purpose**: Cache and retrieve windowed backtest results

**Key Classes**:
- WindowedResultsCache: Manages cache storage and retrieval

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/

**Sample Input**:
```python
cache = WindowedResultsCache()
cache.store_result(
    strategy='SMA_Crossover',
    symbol='BTC/USDT',
    horizon='30d',
    window_id=0,
    dataset_type='train',
    result={'sharpe_ratio': 1.5, 'total_return': 0.12, ...}
)

cached = cache.get_result('SMA_Crossover', 'BTC/USDT', '30d', 0, 'train')
```

**Expected Output**:
Cached results dictionary or None if not found.

**Cache Key Format**:
strategy|symbol|timeframe|horizon|window_id|dataset_type|start_date|end_date
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
from loguru import logger


class WindowedResultsCache:
    """
    Persistent cache for windowed backtest results.

    Uses CSV storage for simplicity and human-readability.
    Cache key includes strategy, symbol, horizon, window dates, and dataset type.
    """

    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize cache.

        Args:
            cache_file: Path to cache CSV file. If None, uses default location.
        """
        if cache_file is None:
            cache_file = Path("data/performance/windowed_results_cache.csv")

        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing cache or create new
        if self.cache_file.exists():
            try:
                self.cache_df = pd.read_csv(self.cache_file)
                logger.info(f"📦 Loaded cache with {len(self.cache_df)} entries from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_file}: {e}")
                logger.info("Creating new cache")
                self.cache_df = self._create_empty_cache()
        else:
            logger.info(f"📦 Creating new cache at {self.cache_file}")
            self.cache_df = self._create_empty_cache()

    def _create_empty_cache(self) -> pd.DataFrame:
        """Create empty cache DataFrame with proper schema."""
        return pd.DataFrame(columns=[
            'strategy', 'symbol', 'timeframe', 'horizon', 'window_id',
            'dataset_type', 'start_date', 'end_date',
            'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
            'total_trades', 'profit_factor', 'final_capital',
            'cached_at'
        ])

    def _make_cache_key(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        horizon: str,
        window_id: int,
        dataset_type: str,
        start_date: str,
        end_date: str
    ) -> str:
        """Generate cache key from parameters."""
        return f"{strategy}|{symbol}|{timeframe}|{horizon}|{window_id}|{dataset_type}|{start_date}|{end_date}"

    def get_result(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        horizon: str,
        window_id: int,
        dataset_type: str,
        start_date: str,
        end_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result if it exists.

        Args:
            strategy: Strategy name
            symbol: Trading symbol
            timeframe: Timeframe string
            horizon: Horizon name (e.g., '30d')
            window_id: Window identifier
            dataset_type: 'train' or 'test'
            start_date: Window start date (ISO format)
            end_date: Window end date (ISO format)

        Returns:
            Cached result dictionary or None if not found
        """
        # BUGFIX: Normalize datetime strings to avoid format mismatch
        # ISO8601 can have variations (microseconds, timezone, etc.)
        # Parse and re-format to ensure consistent comparison
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            # Normalize to consistent format (no microseconds, UTC)
            start_normalized = start_dt.strftime('%Y-%m-%d %H:%M:%S')
            end_normalized = end_dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.warning(f"Failed to normalize dates: {e}, using original strings")
            start_normalized = start_date
            end_normalized = end_date

        # Query cache
        mask = (
            (self.cache_df['strategy'] == strategy) &
            (self.cache_df['symbol'] == symbol) &
            (self.cache_df['timeframe'] == timeframe) &
            (self.cache_df['horizon'] == horizon) &
            (self.cache_df['window_id'] == window_id) &
            (self.cache_df['dataset_type'] == dataset_type)
        )

        # For dates, normalize cached values and compare
        if len(self.cache_df) > 0:
            try:
                cached_start = pd.to_datetime(self.cache_df['start_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                cached_end = pd.to_datetime(self.cache_df['end_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                mask = mask & (cached_start == start_normalized) & (cached_end == end_normalized)
            except Exception:
                # Fallback to string comparison if normalization fails
                mask = mask & (self.cache_df['start_date'] == start_date) & (self.cache_df['end_date'] == end_date)

        matches = self.cache_df[mask]

        if len(matches) > 0:
            # Return first match as dictionary
            row = matches.iloc[0]
            result = {
                'strategy_name': row['strategy'],
                'symbol': row['symbol'],
                'horizon': row['horizon'],
                'window_id': row['window_id'],
                'dataset_type': row['dataset_type'],
                'total_return': row['total_return'],
                'sharpe_ratio': row['sharpe_ratio'],
                'max_drawdown': row['max_drawdown'],
                'win_rate': row['win_rate'],
                'total_trades': row['total_trades'],
                'profit_factor': row['profit_factor'],
                'final_capital': row['final_capital'],
            }
            logger.debug(
                f"✓ Cache hit: {strategy}/{horizon}/win{window_id}/{dataset_type}"
            )
            return result

        logger.debug(
            f"✗ Cache miss: {strategy}/{horizon}/win{window_id}/{dataset_type}"
        )
        return None

    def store_result(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        horizon: str,
        window_id: int,
        dataset_type: str,
        start_date: str,
        end_date: str,
        result: Dict[str, Any]
    ):
        """
        Store result in cache.

        Args:
            strategy: Strategy name
            symbol: Trading symbol
            timeframe: Timeframe string
            horizon: Horizon name
            window_id: Window identifier
            dataset_type: 'train' or 'test'
            start_date: Window start date (ISO format)
            end_date: Window end date (ISO format)
            result: Backtest result dictionary
        """
        # Check if already cached (avoid duplicates)
        existing = self.get_result(
            strategy, symbol, timeframe, horizon, window_id,
            dataset_type, start_date, end_date
        )

        if existing is not None:
            logger.debug(f"Result already cached, skipping: {strategy}/{horizon}/win{window_id}")
            return

        # BUGFIX: Normalize dates before storing to ensure consistent format
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            start_normalized = start_dt.strftime('%Y-%m-%d %H:%M:%S')
            end_normalized = end_dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.warning(f"Failed to normalize dates for storage: {e}")
            start_normalized = start_date
            end_normalized = end_date

        # Create new row
        new_row = {
            'strategy': strategy,
            'symbol': symbol,
            'timeframe': timeframe,
            'horizon': horizon,
            'window_id': window_id,
            'dataset_type': dataset_type,
            'start_date': start_normalized,
            'end_date': end_normalized,
            'total_return': result.get('total_return', 0.0),
            'sharpe_ratio': result.get('sharpe_ratio', 0.0),
            'max_drawdown': result.get('max_drawdown', 0.0),
            'win_rate': result.get('win_rate', 0.0),
            'total_trades': result.get('total_trades', 0),
            'profit_factor': result.get('profit_factor', 0.0),
            'final_capital': result.get('final_capital', 0.0),
            'cached_at': datetime.now().isoformat()
        }

        # Append to cache
        self.cache_df = pd.concat([
            self.cache_df,
            pd.DataFrame([new_row])
        ], ignore_index=True)

        logger.debug(
            f"💾 Stored: {strategy}/{horizon}/win{window_id}/{dataset_type}"
        )

    def save(self):
        """Persist cache to disk."""
        try:
            self.cache_df.to_csv(self.cache_file, index=False)
            logger.info(f"💾 Saved cache ({len(self.cache_df)} entries) to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_cached_windows(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        horizon: str,
        dataset_type: str
    ) -> List[int]:
        """
        Get list of cached window IDs for a strategy/horizon/dataset combination.

        Args:
            strategy: Strategy name
            symbol: Trading symbol
            timeframe: Timeframe string
            horizon: Horizon name
            dataset_type: 'train' or 'test'

        Returns:
            List of cached window IDs
        """
        mask = (
            (self.cache_df['strategy'] == strategy) &
            (self.cache_df['symbol'] == symbol) &
            (self.cache_df['timeframe'] == timeframe) &
            (self.cache_df['horizon'] == horizon) &
            (self.cache_df['dataset_type'] == dataset_type)
        )

        cached_ids = self.cache_df[mask]['window_id'].unique().tolist()
        return sorted(cached_ids)

    def clear(self):
        """Clear all cached results."""
        self.cache_df = self._create_empty_cache()
        logger.info("🗑️  Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if len(self.cache_df) == 0:
            return {
                'total_entries': 0,
                'strategies': 0,
                'horizons': 0,
                'train_windows': 0,
                'test_windows': 0
            }

        return {
            'total_entries': len(self.cache_df),
            'strategies': self.cache_df['strategy'].nunique(),
            'horizons': self.cache_df['horizon'].nunique(),
            'train_windows': len(self.cache_df[self.cache_df['dataset_type'] == 'train']),
            'test_windows': len(self.cache_df[self.cache_df['dataset_type'] == 'test']),
        }


if __name__ == "__main__":
    """
    Validation block for windowed results cache.

    Tests cache storage, retrieval, and persistence.
    """
    import sys
    import tempfile

    all_validation_failures = []
    total_tests = 0

    # Create temporary cache file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_cache_file = Path(tmp.name)

    try:
        # Test 1: Cache creation
        total_tests += 1
        print("Test 1: Cache Creation")
        try:
            cache = WindowedResultsCache(cache_file=tmp_cache_file)

            if not tmp_cache_file.exists():
                all_validation_failures.append("Cache file was not created")

            stats = cache.get_cache_stats()
            if stats['total_entries'] != 0:
                all_validation_failures.append(
                    f"New cache should have 0 entries, got {stats['total_entries']}"
                )

            print(f"  ✓ Cache created at {tmp_cache_file}")
            print(f"  ✓ Initial entries: {stats['total_entries']}")

        except Exception as e:
            all_validation_failures.append(f"Cache creation failed: {e}")

        # Test 2: Store and retrieve result
        total_tests += 1
        print("\nTest 2: Store and Retrieve Result")
        try:
            test_result = {
                'total_return': 0.15,
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.12,
                'win_rate': 0.58,
                'total_trades': 25,
                'profit_factor': 1.5,
                'final_capital': 11500.0
            }

            cache.store_result(
                strategy='TestStrategy',
                symbol='BTC/USDT',
                timeframe='1h',
                horizon='30d',
                window_id=0,
                dataset_type='train',
                start_date='2023-01-01',
                end_date='2023-01-31',
                result=test_result
            )

            # Retrieve
            retrieved = cache.get_result(
                'TestStrategy', 'BTC/USDT', '1h', '30d', 0, 'train',
                '2023-01-01', '2023-01-31'
            )

            if retrieved is None:
                all_validation_failures.append("Stored result was not retrieved")
            elif abs(retrieved['total_return'] - test_result['total_return']) > 0.001:
                all_validation_failures.append(
                    f"Retrieved return {retrieved['total_return']} != stored {test_result['total_return']}"
                )

            print(f"  ✓ Result stored and retrieved")
            print(f"    Return: {retrieved['total_return']:.4f}")
            print(f"    Sharpe: {retrieved['sharpe_ratio']:.2f}")

        except Exception as e:
            all_validation_failures.append(f"Store/retrieve failed: {e}")

        # Test 3: Cache miss
        total_tests += 1
        print("\nTest 3: Cache Miss")
        try:
            missing = cache.get_result(
                'NonExistentStrategy', 'BTC/USDT', '1h', '30d', 0, 'train',
                '2023-01-01', '2023-01-31'
            )

            if missing is not None:
                all_validation_failures.append("Cache miss should return None")

            print(f"  ✓ Cache miss correctly returns None")

        except Exception as e:
            all_validation_failures.append(f"Cache miss test failed: {e}")

        # Test 4: Get cached windows list
        total_tests += 1
        print("\nTest 4: Get Cached Windows List")
        try:
            # Store multiple windows
            for win_id in [1, 2, 3]:
                cache.store_result(
                    strategy='TestStrategy',
                    symbol='BTC/USDT',
                    timeframe='1h',
                    horizon='30d',
                    window_id=win_id,
                    dataset_type='train',
                    start_date=f'2023-0{win_id+1}-01',
                    end_date=f'2023-0{win_id+1}-28',
                    result=test_result
                )

            cached_windows = cache.get_cached_windows(
                'TestStrategy', 'BTC/USDT', '1h', '30d', 'train'
            )

            expected_windows = [0, 1, 2, 3]
            if cached_windows != expected_windows:
                all_validation_failures.append(
                    f"Cached windows {cached_windows} != expected {expected_windows}"
                )

            print(f"  ✓ Cached windows: {cached_windows}")

        except Exception as e:
            all_validation_failures.append(f"Get cached windows failed: {e}")

        # Test 5: Cache persistence
        total_tests += 1
        print("\nTest 5: Cache Persistence")
        try:
            # Save cache
            cache.save()

            # Create new cache instance loading from same file
            cache2 = WindowedResultsCache(cache_file=tmp_cache_file)

            # Verify loaded cache has same entries
            stats2 = cache2.get_cache_stats()
            if stats2['total_entries'] != 4:  # 4 windows stored
                all_validation_failures.append(
                    f"Loaded cache has {stats2['total_entries']} entries, expected 4"
                )

            # Verify specific result can still be retrieved
            retrieved2 = cache2.get_result(
                'TestStrategy', 'BTC/USDT', '1h', '30d', 0, 'train',
                '2023-01-01', '2023-01-31'
            )

            if retrieved2 is None:
                all_validation_failures.append("Persisted result not found after reload")

            print(f"  ✓ Cache persisted and reloaded")
            print(f"    Entries after reload: {stats2['total_entries']}")

        except Exception as e:
            all_validation_failures.append(f"Cache persistence failed: {e}")

        # Test 6: Cache statistics
        total_tests += 1
        print("\nTest 6: Cache Statistics")
        try:
            stats = cache.get_cache_stats()

            required_keys = {'total_entries', 'strategies', 'horizons', 'train_windows', 'test_windows'}
            missing_keys = required_keys - set(stats.keys())

            if missing_keys:
                all_validation_failures.append(f"Missing stat keys: {missing_keys}")

            print(f"  ✓ Cache stats:")
            for key, value in stats.items():
                print(f"    {key}: {value}")

        except Exception as e:
            all_validation_failures.append(f"Cache statistics failed: {e}")

    finally:
        # Cleanup
        if tmp_cache_file.exists():
            tmp_cache_file.unlink()

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Windowed results cache validated: efficient storage and retrieval")
        print("\n💾 Cache Features:")
        print("  - Persistent CSV storage")
        print("  - Unique cache keys per window")
        print("  - Fast lookups for cached results")
        print("  - Avoids recomputation of tested windows")
        sys.exit(0)
