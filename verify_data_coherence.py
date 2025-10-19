#!/usr/bin/env python3
"""
Data Coherence Verification Script

Verifies that multi-pair workers correctly slice data to the appropriate horizon.

Run this to ensure each horizon tests on the correct time period.
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
from master import _calculate_data_limit, _slice_data_to_horizon


def test_data_slicing():
    """Test that data slicing works correctly for different horizons."""
    print("=" * 80)
    print("DATA COHERENCE VERIFICATION")
    print("=" * 80)
    print()

    # Create mock data: 730 days at 1h timeframe = 17,520 candles
    full_days = 730
    timeframe = "1h"
    full_candles = full_days * 24

    print(f"Creating mock dataset: {full_days} days at {timeframe} = {full_candles} candles")

    timestamps = pd.date_range(start='2023-01-01', periods=full_candles, freq='1H')
    full_data = pd.DataFrame({
        'timestamp': timestamps,
        'close': range(full_candles)  # Simple sequence for verification
    })
    full_data = full_data.set_index('timestamp')

    print(f"✓ Created dataset from {full_data.index[0]} to {full_data.index[-1]}")
    print()

    # Test different horizons
    test_cases = [
        (30, 1.5),   # 30 days with 50% warmup
        (90, 1.5),   # 90 days with 50% warmup
        (180, 1.5),  # 180 days with 50% warmup
        (365, 1.5),  # 365 days with 50% warmup
        (730, 1.5),  # 730 days with 50% warmup
    ]

    all_passed = True

    for horizon_days, warmup in test_cases:
        print(f"Testing horizon: {horizon_days} days (warmup={warmup}x)")

        # Calculate expected candles
        expected_candles = _calculate_data_limit(timeframe, horizon_days, warmup)
        print(f"  Expected candles: {expected_candles}")

        # Slice data
        sliced = _slice_data_to_horizon(full_data, timeframe, horizon_days, warmup)

        print(f"  Sliced to: {len(sliced)} candles")
        print(f"  Date range: {sliced.index[0]} to {sliced.index[-1]}")

        # Verify
        if len(sliced) == min(expected_candles, len(full_data)):
            print(f"  ✅ PASS: Got expected number of candles")
        else:
            print(f"  ❌ FAIL: Expected {expected_candles}, got {len(sliced)}")
            all_passed = False

        # Verify it's the LAST N candles (most recent data)
        expected_last_close = full_data['close'].iloc[-1]
        actual_last_close = sliced['close'].iloc[-1]

        if expected_last_close == actual_last_close:
            print(f"  ✅ PASS: Using most recent data (last close = {actual_last_close})")
        else:
            print(f"  ❌ FAIL: Not using most recent data (expected {expected_last_close}, got {actual_last_close})")
            all_passed = False

        print()

    print("=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Data coherence is correct!")
        print()
        print("Each horizon will test on the appropriate time period:")
        print("  • 30d horizon → last 45 days of data")
        print("  • 90d horizon → last 135 days of data")
        print("  • 180d horizon → last 270 days of data")
        print("  • 365d horizon → last 547 days of data")
        print("  • 730d horizon → all available data")
        print()
        return 0
    else:
        print("❌ SOME TESTS FAILED - Data coherence issue detected!")
        print()
        print("This means different horizons may be testing on incorrect time periods.")
        print("Please review the _slice_data_to_horizon() implementation.")
        print()
        return 1


def verify_multi_horizon_independence():
    """Verify that different horizons get different data windows."""
    print("=" * 80)
    print("MULTI-HORIZON INDEPENDENCE VERIFICATION")
    print("=" * 80)
    print()
    print("Verifying that 30d, 90d, and 180d horizons get DIFFERENT data windows...")
    print()

    # Create full dataset
    full_days = 270  # Max needed for 180d × 1.5
    timeframe = "1h"
    full_candles = full_days * 24

    timestamps = pd.date_range(start='2023-01-01', periods=full_candles, freq='1H')
    full_data = pd.DataFrame({
        'timestamp': timestamps,
        'close': range(full_candles)
    })
    full_data = full_data.set_index('timestamp')

    # Slice for different horizons
    data_30d = _slice_data_to_horizon(full_data, timeframe, 30, 1.5)
    data_90d = _slice_data_to_horizon(full_data, timeframe, 90, 1.5)
    data_180d = _slice_data_to_horizon(full_data, timeframe, 180, 1.5)

    print(f"Full dataset: {len(full_data)} candles")
    print(f"30d horizon: {len(data_30d)} candles (start: {data_30d.index[0]})")
    print(f"90d horizon: {len(data_90d)} candles (start: {data_90d.index[0]})")
    print(f"180d horizon: {len(data_180d)} candles (start: {data_180d.index[0]})")
    print()

    # Verify they're different
    if len(data_30d) != len(data_90d) and len(data_90d) != len(data_180d):
        print("✅ PASS: Each horizon has different data window size")
    else:
        print("❌ FAIL: Some horizons have the same data window size!")
        return 1

    # Verify they all end at the same point (most recent data)
    if (data_30d.index[-1] == data_90d.index[-1] == data_180d.index[-1]):
        print("✅ PASS: All horizons end at the same point (using most recent data)")
    else:
        print("❌ FAIL: Horizons don't all end at the same point!")
        return 1

    # Verify they start at different points
    start_30d = data_30d.index[0]
    start_90d = data_90d.index[0]
    start_180d = data_180d.index[0]

    if start_30d > start_90d > start_180d:
        print("✅ PASS: Each horizon starts at a different point (30d < 90d < 180d)")
        print()
        print("Data windows are correctly independent:")
        print(f"  • 30d: {start_30d} to {data_30d.index[-1]}")
        print(f"  • 90d: {start_90d} to {data_90d.index[-1]}")
        print(f"  • 180d: {start_180d} to {data_180d.index[-1]}")
        print()
        return 0
    else:
        print("❌ FAIL: Start points are not in expected order!")
        return 1


if __name__ == "__main__":
    print()
    result1 = test_data_slicing()
    print()
    result2 = verify_multi_horizon_independence()
    print()

    if result1 == 0 and result2 == 0:
        print("=" * 80)
        print("✅ ALL VERIFICATION PASSED")
        print("=" * 80)
        print()
        print("Data coherence is working correctly!")
        print("Each horizon in --multi-pair mode will test on the appropriate time period.")
        print()
        sys.exit(0)
    else:
        print("=" * 80)
        print("❌ VERIFICATION FAILED")
        print("=" * 80)
        print()
        print("Data coherence issue detected.")
        print("Please review the implementation before running --multi-pair mode.")
        print()
        sys.exit(1)
