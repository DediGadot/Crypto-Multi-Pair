"""
Train/Test Window Manager

This module implements scientific train/test splitting with non-overlapping windows
for rigorous strategy evaluation without lookahead bias.

**Purpose**: Generate non-overlapping time windows for training and testing

**Key Classes**:
- WindowSpec: Specification for a single time window
- TrainTestSplitter: Generates train/test windows from historical data

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/

**Sample Input**:
```python
splitter = TrainTestSplitter(
    runtime_date=datetime.now(),
    test_set_years=2
)
train_windows, test_windows = splitter.generate_windows(
    data=df,
    horizon_days=30,
    timeframe='1h'
)
```

**Expected Output**:
List of WindowSpec objects for training and testing sets with no overlap.

**Methodology**:
- Training set: All data before (runtime_date - test_set_years)
- Test set: Last test_set_years of data
- Non-overlapping windows within each set
- Ensures temporal separation to prevent lookahead bias
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd
from loguru import logger


@dataclass
class WindowSpec:
    """
    Specification for a single backtest window.

    Attributes:
        window_id: Unique identifier for this window
        horizon_name: Name of horizon (e.g., '30d', '90d')
        horizon_days: Number of days in this horizon
        start_date: Start date of window (inclusive)
        end_date: End date of window (inclusive)
        dataset_type: 'train' or 'test'
        start_idx: Start index in dataframe
        end_idx: End index in dataframe (exclusive)
    """
    window_id: int
    horizon_name: str
    horizon_days: int
    start_date: datetime
    end_date: datetime
    dataset_type: str  # 'train' or 'test'
    start_idx: int
    end_idx: int

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'window_id': self.window_id,
            'horizon_name': self.horizon_name,
            'horizon_days': self.horizon_days,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'dataset_type': self.dataset_type,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
        }


class TrainTestSplitter:
    """
    Scientific train/test splitter with temporal separation.

    Implements proper ML methodology:
    - Training set: Historical data before cutoff (for parameter tuning)
    - Test set: Recent data after cutoff (for final evaluation)
    - No overlap between train and test
    - No lookahead bias
    """

    def __init__(
        self,
        runtime_date: datetime,
        test_set_years: float = 2.0
    ):
        """
        Initialize train/test splitter.

        Args:
            runtime_date: Current date/time (defines cutoff)
            test_set_years: Years of data reserved for testing (default: 2.0)
        """
        import pytz

        # Ensure runtime_date is timezone-aware (UTC)
        if runtime_date.tzinfo is None:
            self.runtime_date = pytz.UTC.localize(runtime_date)
        else:
            self.runtime_date = runtime_date

        self.test_set_years = test_set_years
        self.cutoff_date = self.runtime_date - timedelta(days=365 * test_set_years)

        logger.info(f"📊 Train/Test Split Configuration:")
        logger.info(f"   Runtime Date: {runtime_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Test Set Duration: {test_set_years} years")
        logger.info(f"   Train/Test Cutoff: {self.cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Training Data: All data before {self.cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Test Data: {self.cutoff_date.strftime('%Y-%m-%d')} to {runtime_date.strftime('%Y-%m-%d')}")

    def split_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets based on cutoff date.

        Args:
            data: DataFrame with datetime index or 'timestamp' column

        Returns:
            Tuple of (train_data, test_data)
        """
        # Ensure we have datetime index
        if 'timestamp' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            df = data.set_index('timestamp')
        elif isinstance(data.index, pd.DatetimeIndex):
            df = data
        else:
            raise ValueError("Data must have datetime index or 'timestamp' column")

        # Split at cutoff
        train_data = df[df.index < self.cutoff_date].copy()
        test_data = df[df.index >= self.cutoff_date].copy()

        # Validate split produced data
        if len(train_data) == 0:
            raise ValueError(
                f"Training set is empty. Cutoff date {self.cutoff_date.strftime('%Y-%m-%d')} "
                f"is before all data (earliest: {df.index[0].strftime('%Y-%m-%d')}). "
                f"Need older historical data or smaller test_set_years."
            )

        if len(test_data) == 0:
            raise ValueError(
                f"Test set is empty. Cutoff date {self.cutoff_date.strftime('%Y-%m-%d')} "
                f"is after all data (latest: {df.index[-1].strftime('%Y-%m-%d')}). "
                f"Need newer data or larger test_set_years."
            )

        logger.info(f"✂️  Data Split Results:")
        logger.info(f"   Training Set: {len(train_data):,} rows "
                   f"({train_data.index[0].strftime('%Y-%m-%d')} to "
                   f"{train_data.index[-1].strftime('%Y-%m-%d')})")
        logger.info(f"   Test Set: {len(test_data):,} rows "
                   f"({test_data.index[0].strftime('%Y-%m-%d')} to "
                   f"{test_data.index[-1].strftime('%Y-%m-%d')})")

        return train_data, test_data

    def generate_non_overlapping_windows(
        self,
        data: pd.DataFrame,
        horizon_days: int,
        horizon_name: str,
        dataset_type: str,
        timeframe: str
    ) -> List[WindowSpec]:
        """
        Generate non-overlapping windows for a dataset.

        Args:
            data: DataFrame with datetime index
            horizon_days: Size of each window in days
            horizon_name: Name of horizon (e.g., '30d')
            dataset_type: 'train' or 'test'
            timeframe: Timeframe string (e.g., '1h', '1d')

        Returns:
            List of WindowSpec objects
        """
        if len(data) == 0:
            logger.warning(f"Empty data for {dataset_type} set - no windows generated")
            return []

        # Calculate periods per day for this timeframe
        timeframe_to_periods = {
            "1m": 24 * 60,
            "5m": 24 * 12,
            "15m": 24 * 4,
            "1h": 24,
            "4h": 6,
            "1d": 1,
            "1w": 1 / 7
        }
        periods_per_day = timeframe_to_periods.get(timeframe, 24)

        # Calculate rows per window
        rows_per_window = int(horizon_days * periods_per_day)

        # Generate non-overlapping windows
        windows = []
        window_id = 0
        current_idx = 0

        while current_idx + rows_per_window <= len(data):
            end_idx = current_idx + rows_per_window

            window = WindowSpec(
                window_id=window_id,
                horizon_name=horizon_name,
                horizon_days=horizon_days,
                start_date=data.index[current_idx],
                end_date=data.index[end_idx - 1],
                dataset_type=dataset_type,
                start_idx=current_idx,
                end_idx=end_idx
            )
            windows.append(window)

            # Move to next non-overlapping window
            current_idx = end_idx
            window_id += 1

        logger.debug(f"  Generated {len(windows)} non-overlapping {horizon_name} windows "
                    f"for {dataset_type} set ({rows_per_window} rows each)")

        return windows

    def generate_windows(
        self,
        data: pd.DataFrame,
        horizon_days: int,
        horizon_name: str,
        timeframe: str
    ) -> Tuple[List[WindowSpec], List[WindowSpec]]:
        """
        Generate train and test windows from full dataset.

        Args:
            data: Full historical data with datetime index
            horizon_days: Size of each window in days
            horizon_name: Name of horizon (e.g., '30d')
            timeframe: Timeframe string (e.g., '1h')

        Returns:
            Tuple of (train_windows, test_windows)
        """
        logger.info(f"🪟  Generating windows for horizon {horizon_name} ({horizon_days} days)")

        # Split data
        train_data, test_data = self.split_data(data)

        # Generate windows for each set
        train_windows = self.generate_non_overlapping_windows(
            train_data, horizon_days, horizon_name, 'train', timeframe
        )

        test_windows = self.generate_non_overlapping_windows(
            test_data, horizon_days, horizon_name, 'test', timeframe
        )

        logger.info(f"  ✅ Train windows: {len(train_windows)}")
        logger.info(f"  ✅ Test windows: {len(test_windows)}")
        logger.info(f"  📊 Total windows: {len(train_windows) + len(test_windows)}")

        return train_windows, test_windows


if __name__ == "__main__":
    """
    Validation block for window manager.

    Tests train/test splitting and non-overlapping window generation
    with real-world scenarios.
    """
    import sys
    import numpy as np

    all_validation_failures = []
    total_tests = 0

    # Test 1: Train/Test Split
    total_tests += 1
    print("Test 1: Train/Test Split")
    try:
        # Create 3 years of hourly data
        runtime = datetime(2025, 1, 1)
        dates = pd.date_range(end=runtime, periods=26280, freq='1h')  # ~3 years
        sample_data = pd.DataFrame({
            'close': np.random.rand(len(dates)) * 100 + 50000,
            'volume': np.random.rand(len(dates)) * 1000,
        }, index=dates)

        splitter = TrainTestSplitter(runtime_date=runtime, test_set_years=2.0)
        train_data, test_data = splitter.split_data(sample_data)

        # Verify cutoff date is correct
        expected_cutoff = runtime - timedelta(days=730)
        if abs((splitter.cutoff_date - expected_cutoff).total_seconds()) > 86400:
            all_validation_failures.append(
                f"Cutoff date incorrect: expected ~{expected_cutoff}, got {splitter.cutoff_date}"
            )

        # Verify no overlap
        if train_data.index[-1] >= test_data.index[0]:
            all_validation_failures.append(
                f"Train/test overlap detected: train ends {train_data.index[-1]}, "
                f"test starts {test_data.index[0]}"
            )

        # Verify test set is ~2 years
        test_days = (test_data.index[-1] - test_data.index[0]).days
        if not (700 < test_days < 750):  # Allow some tolerance
            all_validation_failures.append(
                f"Test set duration incorrect: expected ~730 days, got {test_days}"
            )

        print(f"  ✓ Train set: {len(train_data):,} rows, "
              f"{(train_data.index[-1] - train_data.index[0]).days} days")
        print(f"  ✓ Test set: {len(test_data):,} rows, {test_days} days")
        print(f"  ✓ No overlap: train ends {train_data.index[-1].strftime('%Y-%m-%d')}, "
              f"test starts {test_data.index[0].strftime('%Y-%m-%d')}")

    except Exception as e:
        all_validation_failures.append(f"Train/test split failed: {e}")

    # Test 2: Non-overlapping Window Generation
    total_tests += 1
    print("\nTest 2: Non-overlapping Window Generation")
    try:
        # Generate 30-day windows for train set
        train_windows = splitter.generate_non_overlapping_windows(
            train_data, 30, '30d', 'train', '1h'
        )

        # Verify windows are non-overlapping
        for i in range(len(train_windows) - 1):
            if train_windows[i].end_date >= train_windows[i+1].start_date:
                all_validation_failures.append(
                    f"Window overlap detected: window {i} ends {train_windows[i].end_date}, "
                    f"window {i+1} starts {train_windows[i+1].start_date}"
                )

        # Verify window size
        expected_rows = 30 * 24  # 30 days * 24 hours
        for i, window in enumerate(train_windows):
            actual_rows = window.end_idx - window.start_idx
            if actual_rows != expected_rows:
                all_validation_failures.append(
                    f"Window {i} size incorrect: expected {expected_rows} rows, got {actual_rows}"
                )

        print(f"  ✓ Generated {len(train_windows)} non-overlapping windows")
        print(f"  ✓ Each window: {train_windows[0].end_idx - train_windows[0].start_idx} rows (30 days)")
        if len(train_windows) > 1:
            print(f"  ✓ No overlap verified between consecutive windows")

    except Exception as e:
        all_validation_failures.append(f"Window generation failed: {e}")

    # Test 3: Full Workflow
    total_tests += 1
    print("\nTest 3: Full Workflow (generate_windows)")
    try:
        train_wins, test_wins = splitter.generate_windows(
            sample_data, 30, '30d', '1h'
        )

        # Verify both sets have windows
        if len(train_wins) == 0:
            all_validation_failures.append("No training windows generated")
        if len(test_wins) == 0:
            all_validation_failures.append("No test windows generated")

        # Verify dataset_type is set correctly
        if train_wins and train_wins[0].dataset_type != 'train':
            all_validation_failures.append(f"Train window has wrong dataset_type: {train_wins[0].dataset_type}")
        if test_wins and test_wins[0].dataset_type != 'test':
            all_validation_failures.append(f"Test window has wrong dataset_type: {test_wins[0].dataset_type}")

        print(f"  ✓ Train windows: {len(train_wins)}")
        print(f"  ✓ Test windows: {len(test_wins)}")
        print(f"  ✓ Dataset types correctly labeled")

    except Exception as e:
        all_validation_failures.append(f"Full workflow failed: {e}")

    # Test 4: WindowSpec serialization
    total_tests += 1
    print("\nTest 4: WindowSpec Serialization")
    try:
        if train_wins:
            window_dict = train_wins[0].to_dict()

            required_keys = {'window_id', 'horizon_name', 'horizon_days',
                           'start_date', 'end_date', 'dataset_type',
                           'start_idx', 'end_idx'}
            missing_keys = required_keys - set(window_dict.keys())

            if missing_keys:
                all_validation_failures.append(f"Missing keys in serialized window: {missing_keys}")
            else:
                print(f"  ✓ All required keys present in serialized window")
                print(f"  ✓ Window ID: {window_dict['window_id']}")
                print(f"  ✓ Date range: {window_dict['start_date'][:10]} to {window_dict['end_date'][:10]}")

    except Exception as e:
        all_validation_failures.append(f"WindowSpec serialization failed: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Window manager validated: train/test split with non-overlapping windows")
        print("\n📋 Methodology Summary:")
        print("  - Training set: All data before cutoff (for parameter tuning)")
        print("  - Test set: Most recent data (for final evaluation)")
        print("  - Non-overlapping windows within each set")
        print("  - Temporal separation prevents lookahead bias")
        sys.exit(0)
