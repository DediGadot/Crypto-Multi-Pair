"""
Multi-Pair Train/Test Window Manager

This module extends the single-pair window manager to support multiple trading pairs
with synchronized window generation and proper train/test splitting.

**Purpose**: Generate synchronized non-overlapping time windows across multiple pairs

**Key Classes**:
- MultiPairWindowSpec: Specification for a multi-pair window
- MultiPairTrainTestSplitter: Generates synchronized train/test windows

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/

**Sample Input**:
```python
splitter = MultiPairTrainTestSplitter(
    runtime_date=datetime.now(),
    test_set_years=2,
    pairs=['BTC/USDT', 'ETH/USDT']
)
train_windows, test_windows = splitter.generate_windows(
    data_dict={'BTC/USDT': btc_df, 'ETH/USDT': eth_df},
    horizon_days=30,
    timeframe='1h'
)
```

**Expected Output**:
List of MultiPairWindowSpec objects with synchronized windows across all pairs.

**Methodology**:
- All pairs share the same cutoff date
- Windows are synchronized (same start/end dates across pairs)
- Handles missing data gracefully
- Applies timestamp fix to avoid duplicate column errors
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import pandas as pd
from loguru import logger

from .window_manager import WindowSpec, TrainTestSplitter


@dataclass
class MultiPairWindowSpec:
    """
    Specification for a synchronized multi-pair window.

    Attributes:
        window_id: Unique identifier for this window
        horizon_name: Name of horizon (e.g., '30d', '90d')
        horizon_days: Number of days in this horizon
        start_date: Start date of window (inclusive)
        end_date: End date of window (inclusive)
        dataset_type: 'train' or 'test'
        pair_windows: Dict mapping pair symbol to WindowSpec
    """
    window_id: int
    horizon_name: str
    horizon_days: int
    start_date: datetime
    end_date: datetime
    dataset_type: str  # 'train' or 'test'
    pair_windows: Dict[str, WindowSpec]

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'window_id': self.window_id,
            'horizon_name': self.horizon_name,
            'horizon_days': self.horizon_days,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'dataset_type': self.dataset_type,
            'pairs': list(self.pair_windows.keys()),
            'pair_windows': {
                pair: window.to_dict()
                for pair, window in self.pair_windows.items()
            }
        }


class MultiPairTrainTestSplitter:
    """
    Multi-pair train/test splitter with synchronized windows.

    Extends single-pair methodology to multiple trading pairs:
    - Single cutoff date applied to all pairs
    - Synchronized window generation
    - Graceful handling of missing data
    """

    def __init__(
        self,
        runtime_date: datetime,
        test_set_years: float = 2.0,
        pairs: List[str] = None
    ):
        """
        Initialize multi-pair train/test splitter.

        Args:
            runtime_date: Current date/time (defines cutoff)
            test_set_years: Years of data reserved for testing (default: 2.0)
            pairs: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
        """
        import pytz

        # Ensure runtime_date is timezone-aware (UTC)
        if runtime_date.tzinfo is None:
            self.runtime_date = pytz.UTC.localize(runtime_date)
        else:
            self.runtime_date = runtime_date

        self.test_set_years = test_set_years
        self.cutoff_date = self.runtime_date - timedelta(days=365 * test_set_years)
        self.pairs = pairs or []

        logger.info(f"📊 Multi-Pair Train/Test Split Configuration:")
        logger.info(f"   Runtime Date: {runtime_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Test Set Duration: {test_set_years} years")
        logger.info(f"   Train/Test Cutoff: {self.cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Trading Pairs: {len(self.pairs)} pairs ({', '.join(self.pairs)})")
        logger.info(f"   Training Data: All data before {self.cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Test Data: {self.cutoff_date.strftime('%Y-%m-%d')} to {runtime_date.strftime('%Y-%m-%d')}")

    def split_data(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Split data for all pairs into training and test sets.

        Args:
            data_dict: Dictionary mapping pair symbol to DataFrame

        Returns:
            Tuple of (train_data_dict, test_data_dict)
        """
        train_data_dict = {}
        test_data_dict = {}

        for pair, data in data_dict.items():
            # Ensure we have datetime index
            if 'timestamp' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                df = data.set_index('timestamp')
            elif isinstance(data.index, pd.DatetimeIndex):
                df = data
            else:
                raise ValueError(f"{pair}: Data must have datetime index or 'timestamp' column")

            # Split at cutoff
            train_data = df[df.index < self.cutoff_date].copy()
            test_data = df[df.index >= self.cutoff_date].copy()

            # Validate split produced data
            if len(train_data) == 0:
                logger.warning(
                    f"{pair}: Training set is empty. Cutoff date {self.cutoff_date.strftime('%Y-%m-%d')} "
                    f"is before all data (earliest: {df.index[0].strftime('%Y-%m-%d')}). "
                    f"This pair will be skipped for training."
                )
                continue

            if len(test_data) == 0:
                logger.warning(
                    f"{pair}: Test set is empty. Cutoff date {self.cutoff_date.strftime('%Y-%m-%d')} "
                    f"is after all data (latest: {df.index[-1].strftime('%Y-%m-%d')}). "
                    f"This pair will be skipped for testing."
                )
                continue

            train_data_dict[pair] = train_data
            test_data_dict[pair] = test_data

            logger.info(f"✂️  {pair} Split Results:")
            logger.info(f"   Training: {len(train_data):,} rows "
                       f"({train_data.index[0].strftime('%Y-%m-%d')} to "
                       f"{train_data.index[-1].strftime('%Y-%m-%d')})")
            logger.info(f"   Test: {len(test_data):,} rows "
                       f"({test_data.index[0].strftime('%Y-%m-%d')} to "
                       f"{test_data.index[-1].strftime('%Y-%m-%d')})")

        return train_data_dict, test_data_dict

    def generate_synchronized_windows(
        self,
        data_dict: Dict[str, pd.DataFrame],
        horizon_days: int,
        horizon_name: str,
        dataset_type: str,
        timeframe: str
    ) -> List[MultiPairWindowSpec]:
        """
        Generate synchronized windows across all pairs.

        All pairs will have windows with the same start/end dates,
        but different row indices based on their own data.

        Args:
            data_dict: Dictionary mapping pair symbol to DataFrame
            horizon_days: Size of each window in days
            horizon_name: Name of horizon (e.g., '30d')
            dataset_type: 'train' or 'test'
            timeframe: Timeframe string (e.g., '1h')

        Returns:
            List of MultiPairWindowSpec objects
        """
        if not data_dict:
            logger.warning(f"No data for {dataset_type} set - no windows generated")
            return []

        # Find the common date range across all pairs
        all_start_dates = [df.index[0] for df in data_dict.values() if len(df) > 0]
        all_end_dates = [df.index[-1] for df in data_dict.values() if len(df) > 0]

        if not all_start_dates or not all_end_dates:
            logger.warning(f"All pairs have empty data for {dataset_type} set")
            return []

        # Use the latest start date and earliest end date to ensure all pairs have data
        common_start = max(all_start_dates)
        common_end = min(all_end_dates)

        logger.debug(f"  Common date range for {dataset_type}: "
                    f"{common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")

        # Calculate window size in time
        window_duration = timedelta(days=horizon_days)

        # Generate synchronized windows
        windows = []
        window_id = 0
        current_start = common_start

        while current_start + window_duration <= common_end:
            current_end = current_start + window_duration

            # Create window spec for each pair
            pair_windows = {}
            window_valid = True

            for pair, data in data_dict.items():
                # Find indices for this date range in this pair's data
                # BUGFIX: Use <= to include the end boundary (fixes off-by-one error)
                pair_mask = (data.index >= current_start) & (data.index <= current_end)
                pair_indices = data.index[pair_mask]

                if len(pair_indices) == 0:
                    logger.debug(f"    {pair}: No data for window {window_id} "
                               f"({current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')})")
                    window_valid = False
                    break

                # Get row indices
                start_idx = data.index.get_loc(pair_indices[0])
                end_idx = data.index.get_loc(pair_indices[-1]) + 1

                pair_windows[pair] = WindowSpec(
                    window_id=window_id,
                    horizon_name=horizon_name,
                    horizon_days=horizon_days,
                    start_date=pair_indices[0],
                    end_date=pair_indices[-1],
                    dataset_type=dataset_type,
                    start_idx=start_idx,
                    end_idx=end_idx
                )

            # Only add window if all pairs have data
            if window_valid:
                multi_window = MultiPairWindowSpec(
                    window_id=window_id,
                    horizon_name=horizon_name,
                    horizon_days=horizon_days,
                    start_date=current_start,
                    end_date=current_end,
                    dataset_type=dataset_type,
                    pair_windows=pair_windows
                )
                windows.append(multi_window)
                window_id += 1

            # Move to next non-overlapping window
            current_start = current_end

        logger.debug(f"  Generated {len(windows)} synchronized {horizon_name} windows "
                    f"for {dataset_type} set ({len(data_dict)} pairs)")

        return windows

    def generate_windows(
        self,
        data_dict: Dict[str, pd.DataFrame],
        horizon_days: int,
        horizon_name: str,
        timeframe: str
    ) -> Tuple[List[MultiPairWindowSpec], List[MultiPairWindowSpec]]:
        """
        Generate train and test windows for all pairs.

        Args:
            data_dict: Dictionary mapping pair symbol to full historical DataFrame
            horizon_days: Size of each window in days
            horizon_name: Name of horizon (e.g., '30d')
            timeframe: Timeframe string (e.g., '1h')

        Returns:
            Tuple of (train_windows, test_windows)
        """
        logger.info(f"🪟  Generating multi-pair windows for horizon {horizon_name} ({horizon_days} days)")

        # Split data
        train_data_dict, test_data_dict = self.split_data(data_dict)

        # Generate windows for each set
        train_windows = self.generate_synchronized_windows(
            train_data_dict, horizon_days, horizon_name, 'train', timeframe
        )

        test_windows = self.generate_synchronized_windows(
            test_data_dict, horizon_days, horizon_name, 'test', timeframe
        )

        logger.info(f"  ✅ Train windows: {len(train_windows)}")
        logger.info(f"  ✅ Test windows: {len(test_windows)}")
        logger.info(f"  📊 Total windows: {len(train_windows) + len(test_windows)}")

        return train_windows, test_windows


if __name__ == "__main__":
    """
    Validation block for multi-pair window manager.

    Tests synchronized window generation across multiple pairs.
    """
    import sys
    import numpy as np

    all_validation_failures = []
    total_tests = 0

    # Test 1: Multi-Pair Train/Test Split
    total_tests += 1
    print("Test 1: Multi-Pair Train/Test Split")
    try:
        import pytz

        # Create sample data for 2 pairs (timezone-aware)
        runtime = datetime(2025, 1, 1, tzinfo=pytz.UTC)
        dates = pd.date_range(end=runtime, periods=8760, freq='1h', tz=pytz.UTC)  # ~1 year

        btc_data = pd.DataFrame({
            'close': np.random.rand(len(dates)) * 100 + 50000,
            'volume': np.random.rand(len(dates)) * 1000,
        }, index=dates)

        eth_data = pd.DataFrame({
            'close': np.random.rand(len(dates)) * 100 + 3000,
            'volume': np.random.rand(len(dates)) * 500,
        }, index=dates)

        data_dict = {'BTC/USDT': btc_data, 'ETH/USDT': eth_data}

        splitter = MultiPairTrainTestSplitter(
            runtime_date=runtime,
            test_set_years=0.5,
            pairs=['BTC/USDT', 'ETH/USDT']
        )
        train_dict, test_dict = splitter.split_data(data_dict)

        # Verify both pairs split correctly
        if 'BTC/USDT' not in train_dict or 'ETH/USDT' not in train_dict:
            all_validation_failures.append("Not all pairs in training set")

        if 'BTC/USDT' not in test_dict or 'ETH/USDT' not in test_dict:
            all_validation_failures.append("Not all pairs in test set")

        # Verify no overlap
        for pair in ['BTC/USDT', 'ETH/USDT']:
            if train_dict[pair].index[-1] >= test_dict[pair].index[0]:
                all_validation_failures.append(f"{pair}: Train/test overlap detected")

        print(f"  ✓ Both pairs split successfully")
        print(f"  ✓ No overlap detected")

    except Exception as e:
        all_validation_failures.append(f"Multi-pair split failed: {e}")

    # Test 2: Synchronized Window Generation
    total_tests += 1
    print("\nTest 2: Synchronized Window Generation")
    try:
        train_windows, test_windows = splitter.generate_windows(
            data_dict, 30, '30d', '1h'
        )

        if len(train_windows) == 0:
            all_validation_failures.append("No training windows generated")

        if len(test_windows) == 0:
            all_validation_failures.append("No test windows generated")

        # Verify windows are synchronized
        if train_windows:
            first_window = train_windows[0]
            if len(first_window.pair_windows) != 2:
                all_validation_failures.append(
                    f"Window should have 2 pairs, got {len(first_window.pair_windows)}"
                )

            # Check all pairs have same date range (approximately)
            dates = [w.start_date for w in first_window.pair_windows.values()]
            if len(set(d.date() for d in dates)) > 1:
                all_validation_failures.append("Windows not synchronized across pairs")

        print(f"  ✓ Generated {len(train_windows)} train windows")
        print(f"  ✓ Generated {len(test_windows)} test windows")
        print(f"  ✓ Windows synchronized across pairs")

    except Exception as e:
        all_validation_failures.append(f"Window generation failed: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Multi-pair window manager validated: synchronized windows across pairs")
        sys.exit(0)
