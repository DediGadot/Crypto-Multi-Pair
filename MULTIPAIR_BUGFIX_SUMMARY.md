# Multi-Pair Windowed Analysis - Bug Fix Summary

**Date**: 2025-10-20
**Issue**: Multi-pair windowed analysis report showed `inf` Sharpe ratios and `nan` overfitting metrics

## Root Cause Analysis

### Primary Issue: Non-Finite Sharpe Ratios
- **Source**: VectorBT's `portfolio.sharpe_ratio()` method can return `inf` when a strategy has constant positive returns (std dev ≈ 0)
- **Location**: `/home/fiod/crypto/src/crypto_trader/backtesting/engine.py` line 204
- **Original Code**:
  ```python
  sharpe_ratio=float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0,
  ```
- **Problem**: Only checked for `nan`, not `inf`

### Secondary Issue: Propagation Through Aggregation
- **Impact**: `inf` values propagated through aggregation layers unchanged
- **Location**: `/home/fiod/crypto/src/crypto_trader/analysis/aggregator.py` `_calculate_statistics` method
- **Problem**: NumPy statistics functions (`mean`, `std`, etc.) preserve `inf` values

### Tertiary Issue: NaN in Overfitting Calculation
- **Symptom**: Overfitting gap calculation produced `nan` when both train and test Sharpe were `inf`
- **Cause**: `inf - inf = nan` in floating point arithmetic

### Cache Issue: Results Not Persisted
- **Problem**: Cache was created and populated but never saved to disk
- **Location**: `/home/fiod/crypto/master_windowed_multipair.py`
- **Missing**: Call to `cache.save()`

## Fixes Applied

### Fix 1: Filter Non-Finite Values in Backtesting Engine
**File**: `/home/fiod/crypto/src/crypto_trader/backtesting/engine.py`

**Changes**:
- Line 204: Changed `not np.isnan(sharpe_ratio)` to `np.isfinite(sharpe_ratio)`
- Line 216: Changed `not np.isnan(sortino_ratio)` to `np.isfinite(sortino_ratio)`
- Line 217: Changed `not np.isnan(calmar_ratio)` to `np.isfinite(calmar_ratio)`

**Impact**: All non-finite Sharpe ratios (including `inf` and `-inf`) now converted to `0.0`

### Fix 2: Filter Non-Finite Values in Aggregation
**File**: `/home/fiod/crypto/src/crypto_trader/analysis/aggregator.py`

**Changes**: Added filtering logic in `_calculate_statistics` method (lines 160-178):
```python
# Filter out inf and nan values before computing statistics
arr = np.array(values)
finite_mask = np.isfinite(arr)
if not finite_mask.any():
    # All values are inf/nan - return zeros
    logger.warning(f"All values are non-finite (inf/nan), returning zero statistics")
    return {
        'mean': 0.0,
        'median': 0.0,
        'std': 0.0,
        'p25': 0.0,
        'p75': 0.0,
        'weighted': 0.0
    }

# Use only finite values
arr = arr[finite_mask]
if len(arr) != len(values):
    logger.warning(f"Filtered {len(values) - len(arr)} non-finite values from statistics")
```

**Impact**: Aggregation now filters out non-finite values before computing statistics, producing valid numerical results

### Fix 3: Persist Cache to Disk
**File**: `/home/fiod/crypto/master_windowed_multipair.py`

**Changes**: Added line 600: `cache.save()`

**Impact**: Cache is now persisted after analysis completes, enabling faster re-runs

## Verification

### Test Results
Created test script `test_multipair_fixes.py` with three test cases:

1. **Test 1**: Aggregator handles inf/nan values
   - Input: 4 results with 2 inf/nan Sharpe ratios
   - Result: ✅ Mean Sharpe correctly calculated from 2 valid values

2. **Test 2**: MultiPair aggregator handles inf/nan values
   - Input: BTC results with 1 inf Sharpe, ETH results with valid Sharpes
   - Result: ✅ Portfolio Sharpe is finite and correct

3. **Test 3**: Overfitting gap calculation
   - Input: Both train and test with inf Sharpe
   - Result: ✅ Gap is 0.0 (not nan)

**All tests passed successfully!**

### Log Output
The fixes generate appropriate warnings:
```
WARNING | Filtered 2 non-finite values from statistics
WARNING | All values are non-finite (inf/nan), returning zero statistics
```

## Expected Behavior After Fixes

### Before Fixes
```
1. SMA_Crossover: inf (Avg Test Sharpe)
2. VWAP_MeanReversion: inf (Avg Test Sharpe)
Overfitting Risk: Low (nan)
```

### After Fixes
```
1. Strategy_A: 1.52 (Avg Test Sharpe)
2. Strategy_B: 1.35 (Avg Test Sharpe)
Overfitting Risk: Low (0.15)
```

## Recommendations

1. **Re-run Analysis**: Execute `master_windowed_multipair.py` again to generate clean reports
2. **Monitor Warnings**: Check logs for "non-finite values" warnings to identify problematic strategies
3. **Strategy Review**: Strategies that produce constant returns (leading to inf Sharpe) should be reviewed for correctness
4. **Cache Benefits**: Subsequent runs will be faster due to caching

## Files Modified

1. `/home/fiod/crypto/src/crypto_trader/backtesting/engine.py` - Line 204, 216, 217
2. `/home/fiod/crypto/src/crypto_trader/analysis/aggregator.py` - Lines 160-206
3. `/home/fiod/crypto/master_windowed_multipair.py` - Line 600

## Testing

To verify fixes work correctly:
```bash
python test_multipair_fixes.py
```

Expected output: "✅ ALL TESTS PASSED"

## Notes

- VectorBT behavior is not a bug - it's mathematically correct that Sharpe ratio is infinite when returns are constant and positive
- Our fix treats infinite Sharpe as 0.0 which is conservative but prevents downstream errors
- Alternative approach could cap Sharpe at a large finite value (e.g., 10.0), but 0.0 is safer
- The filtering happens at multiple levels for defense-in-depth

## Performance Impact

- Minimal: Only adds one `np.isfinite()` check per metric
- Cache persistence adds negligible overhead at end of analysis
- No impact on backtest execution speed
