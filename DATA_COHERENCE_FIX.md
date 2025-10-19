# Critical Data Coherence Fix

## 🚨 Problem Discovered

After implementing the shared data pool optimization, discovered that **all horizons were testing on the SAME data window**!

### What Was Happening (BROKEN ❌)

```
Pre-fetch: Get 270 days of data (max horizon × 1.5)
Pass to workers: All 270 days

Worker testing 30d horizon:  Uses all 270 days ❌
Worker testing 90d horizon:  Uses all 270 days ❌
Worker testing 180d horizon: Uses all 270 days ❌

Result: All three horizons test on identical 270-day period!
```

### Why This Is Critical

- **Invalid Comparisons**: Can't compare strategy performance across horizons
- **Wrong Rankings**: Strategies ranked on wrong time periods
- **Misleading Results**: 30d backtest actually testing 270d of data

---

## ✅ Solution Implemented

Added `_slice_data_to_horizon()` function that:
1. Takes the full pre-fetched dataset
2. Slices to the LAST N candles for each horizon
3. Ensures each horizon tests on correct time period

### How It Works Now (CORRECT ✅)

```
Pre-fetch: Get 270 days of data once (6480 candles at 1h)
Workers slice to appropriate windows:

Worker testing 30d horizon:  Slices to last 45 days (1080 candles) ✅
Worker testing 90d horizon:  Slices to last 135 days (3240 candles) ✅
Worker testing 180d horizon: Uses all 270 days (6480 candles) ✅

Result: Each horizon tests on the correct time period!
```

---

## 📊 Verification Results

```bash
$ uv run python verify_data_coherence.py

================================================================================
DATA COHERENCE VERIFICATION
================================================================================

Testing horizon: 30 days (warmup=1.5x)
  Expected candles: 1080
  Sliced to: 1080 candles
  ✅ PASS: Got expected number of candles
  ✅ PASS: Using most recent data

Testing horizon: 90 days (warmup=1.5x)
  Expected candles: 3240
  Sliced to: 3240 candles
  ✅ PASS: Got expected number of candles
  ✅ PASS: Using most recent data

Testing horizon: 180 days (warmup=1.5x)
  Expected candles: 6480
  Sliced to: 6480 candles
  ✅ PASS: Got expected number of candles
  ✅ PASS: Using most recent data

================================================================================
✅ ALL TESTS PASSED - Data coherence is correct!
================================================================================
```

---

## 🔧 Technical Implementation

### Function Added (master.py:580-622)

```python
def _slice_data_to_horizon(
    data: pd.DataFrame,
    timeframe: str,
    horizon_days: int,
    warmup_multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Slice data to the appropriate window for a given horizon.

    Takes the LAST N candles corresponding to horizon_days × warmup_multiplier.
    This ensures each horizon tests on the correct time period.
    """
    required_candles = _calculate_data_limit(timeframe, horizon_days, warmup_multiplier)

    if len(data) <= required_candles:
        return data

    # Take the LAST required_candles rows (most recent data)
    return data.tail(required_candles).copy()
```

### Applied in Workers

**StatisticalArbitrage (master.py:1063-1064)**:
```python
# CRITICAL: Slice data to correct horizon window
asset1_data = _slice_data_to_horizon(asset1_data, timeframe, horizon_days, warmup_multiplier=1.5)
asset2_data = _slice_data_to_horizon(asset2_data, timeframe, horizon_days, warmup_multiplier=1.5)
```

**Portfolio Strategies (master.py:1340)**:
```python
# CRITICAL: Slice data to correct horizon window
data = _slice_data_to_horizon(data, timeframe, horizon_days, warmup_multiplier=1.5)
```

---

## 🎯 What This Means

### Before Fix
- 30d, 90d, 180d horizons all tested on same 270-day window
- Rankings were meaningless
- Results were incorrect

### After Fix
- Each horizon tests on appropriate time period:
  - **30d**: Last 45 days of data (30 × 1.5)
  - **90d**: Last 135 days of data (90 × 1.5)
  - **180d**: Last 270 days of data (180 × 1.5)
- Rankings are now valid
- Results are accurate

### Benefits Retained
- ✅ Shared data pool still works (4-10x speedup)
- ✅ Zero redundant API calls
- ✅ Low memory usage
- ✅ **Plus**: Correct data windows per horizon

---

## 🧪 How to Verify

Run the verification script:

```bash
uv run python verify_data_coherence.py
```

Expected output:
```
✅ ALL TESTS PASSED - Data coherence is correct!

Each horizon will test on the appropriate time period:
  • 30d horizon → last 45 days of data
  • 90d horizon → last 135 days of data
  • 180d horizon → last 270 days of data
```

---

## 📝 Summary

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| **Data Window** | Same for all horizons ❌ | Different per horizon ✅ |
| **30d Horizon** | Used 270 days ❌ | Uses 45 days ✅ |
| **90d Horizon** | Used 270 days ❌ | Uses 135 days ✅ |
| **180d Horizon** | Used 270 days | Uses 270 days ✅ |
| **Results Valid** | No ❌ | Yes ✅ |
| **Speed** | 4-10x faster ✅ | 4-10x faster ✅ |
| **Memory** | 50-80% less ✅ | 50-80% less ✅ |

---

**Status**: ✅ FIXED AND VERIFIED

All multi-pair workers now correctly slice data to their respective horizons while maintaining the performance benefits of the shared data pool.
