#!/usr/bin/env python3
"""
Quick diagnostic script to test timestamp handling in the backtest pipeline.
"""

import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

# Test what happens when we convert DataFrame to dict and back
print("=" * 80)
print("TEST 1: DataFrame with DatetimeIndex -> dict -> DataFrame")
print("=" * 80)

# Create test data similar to what we have
dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
df_original = pd.DataFrame({
    'open': np.random.random(100) * 1000,
    'high': np.random.random(100) * 1000,
    'low': np.random.random(100) * 1000,
    'close': np.random.random(100) * 1000,
    'volume': np.random.random(100) * 1000000,
}, index=dates)

print(f"\nOriginal DataFrame:")
print(f"  Shape: {df_original.shape}")
print(f"  Columns: {df_original.columns.tolist()}")
print(f"  Index type: {type(df_original.index).__name__}")
print(f"  Index is DatetimeIndex: {isinstance(df_original.index, pd.DatetimeIndex)}")

# Simulate what master_windowed_multipair.py does
print(f"\n--- Simulating master_windowed_multipair.py lines 115-136 ---")

window_df = df_original.copy()

# Lines 115-124: Check if we have DatetimeIndex
if isinstance(window_df.index, pd.DatetimeIndex):
    print("  ✓ Index is DatetimeIndex")
    if 'timestamp' in window_df.columns:
        print("  ⚠ 'timestamp' column exists - dropping it")
        df_for_worker = window_df.drop(columns=['timestamp']).reset_index()
        df_for_worker = df_for_worker.rename(columns={'index': 'timestamp'})
    else:
        print("  ✓ No 'timestamp' column - calling reset_index()")
        df_for_worker = window_df.reset_index()
        if 'index' in df_for_worker.columns:
            df_for_worker = df_for_worker.rename(columns={'index': 'timestamp'})

print(f"\nAfter reset_index + rename:")
print(f"  Shape: {df_for_worker.shape}")
print(f"  Columns: {df_for_worker.columns.tolist()}")
print(f"  Index type: {type(df_for_worker.index).__name__}")
print(f"  Has 'timestamp' column: {'timestamp' in df_for_worker.columns}")

# Lines 133-136: Convert timestamps to ISO strings
df_for_worker_copy = df_for_worker.copy()
if 'timestamp' in df_for_worker_copy.columns:
    print(f"\n  Converting 'timestamp' column to strings...")
    print(f"    Before: {df_for_worker_copy['timestamp'].dtype}")
    df_for_worker_copy['timestamp'] = df_for_worker_copy['timestamp'].astype(str)
    print(f"    After: {df_for_worker_copy['timestamp'].dtype}")

data_dict_for_worker = df_for_worker_copy.to_dict('list')

print(f"\nAfter to_dict('list'):")
print(f"  Dict keys: {list(data_dict_for_worker.keys())}")
print(f"  'timestamp' sample: {data_dict_for_worker['timestamp'][:3]}")

# Simulate what workers.py does
print(f"\n--- Simulating workers.py lines 90-93 ---")

data = pd.DataFrame(data_dict_for_worker)
print(f"\nAfter pd.DataFrame(data_dict):")
print(f"  Shape: {data.shape}")
print(f"  Columns: {data.columns.tolist()}")
print(f"  Index type: {type(data.index).__name__}")
print(f"  'timestamp' dtype: {data['timestamp'].dtype}")

# Line 93: Convert timestamp column from string to datetime
if 'timestamp' in data.columns:
    print(f"\n  Converting 'timestamp' from string to datetime...")
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    print(f"    After: {data['timestamp'].dtype}")

print(f"\nFinal data state:")
print(f"  Shape: {data.shape}")
print(f"  Columns: {data.columns.tolist()}")
print(f"  Index type: {type(data.index).__name__}")
print(f"  Index is DatetimeIndex: {isinstance(data.index, pd.DatetimeIndex)}")
print(f"  Has 'timestamp' column: {'timestamp' in data.columns}")
print(f"  'timestamp' dtype: {data['timestamp'].dtype if 'timestamp' in data.columns else 'N/A'}")

print("\n" + "=" * 80)
print("TEST 2: What does _get_datetime_index do with this?")
print("=" * 80)

# Simulate what engine.py's _get_datetime_index does
if 'timestamp' in data.columns:
    print("  Found 'timestamp' column")
    idx = pd.to_datetime(data['timestamp'])
    print(f"  Converted to DatetimeIndex: {isinstance(idx, pd.DatetimeIndex)}")
    print(f"  Length: {len(idx)}")
elif isinstance(data.index, pd.DatetimeIndex):
    print("  Using existing DatetimeIndex from data.index")
    idx = data.index
else:
    print("  ❌ ERROR: No timestamp column or DatetimeIndex!")

print("\n" + "=" * 80)
print("TEST 3: Try creating VectorBT portfolio")
print("=" * 80)

try:
    import vectorbt as vbt

    # Create close series with datetime index
    close_series = pd.Series(data['close'].values, index=idx, name='close')
    print(f"  Created close_series:")
    print(f"    Length: {len(close_series)}")
    print(f"    Index type: {type(close_series.index).__name__}")

    # Create simple buy-and-hold signals
    entries = pd.Series([True] + [False] * (len(close_series) - 1), index=close_series.index)
    exits = pd.Series([False] * (len(close_series) - 1) + [True], index=close_series.index)

    print(f"\n  Creating VectorBT portfolio...")
    portfolio = vbt.Portfolio.from_signals(
        close=close_series,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001,
        freq='1H'
    )
    print(f"  ✅ SUCCESS! Portfolio created")
    print(f"    Final value: ${portfolio.value()[-1]:,.2f}")
    print(f"    Total return: {portfolio.total_return() * 100:.2f}%")
except Exception as e:
    print(f"  ❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
