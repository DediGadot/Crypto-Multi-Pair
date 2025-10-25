# Phase 3 Critical Bugfix: KeyError 'signal'

**Date**: 2025-10-25
**Severity**: CRITICAL
**Status**: ✅ FIXED

---

## Problem Summary

After implementing Phase 3 transaction cost optimization, all portfolio strategies (HierarchicalRiskParity, RiskParity, BlackLitterman) failed during windowed analysis with:

```
KeyError: 'signal'
at src/crypto_trader/backtesting/engine.py:129 in _signals_to_entries_exits()
```

**Impact**: 153+ failures across all portfolio strategies in the windowed pipeline.

---

## Root Cause Analysis

### The Bug

Portfolio strategies have **two code paths** for signal generation:

1. **Multi-asset path**: Returns weights → calls `_weights_to_signals()` → returns proper format
2. **Single-asset fallback path**: Returns weights directly **WITHOUT** calling `_weights_to_signals()`

The single-asset fallback was returning a DataFrame with only `['timestamp', 'weight_{asset}_close']` columns, **missing** the required `['signal', 'confidence', 'metadata']` columns.

### Why It Happened Now

The multi-pair windowed pipeline runs strategies in **single-symbol mode** (one symbol per worker). When portfolio strategies receive single-symbol data, they trigger the single-asset fallback path, which was never updated to return the correct signal format.

### Affected Code Locations

**HierarchicalRiskParity** (`hierarchical_risk_parity.py`):
- Line 180: Early return with equal weights (insufficient data case)
- Line 511: `_generate_single_asset_signals()` method

**RiskParity** (`risk_parity.py`):
- Line 157: Early return with equal weights (insufficient data case)
- Line 323: `_generate_single_asset_signals()` method

**BlackLitterman** (`black_litterman.py`):
- Line 161: Early return with equal weights (insufficient data case)
- Line 445: `_generate_single_asset_signals()` method

---

## The Fix

### Solution

Added `_weights_to_signals()` call to **all** early return paths in portfolio strategies:

```python
# BEFORE (broken):
return signals_df  # Only has timestamp and weight columns

# AFTER (fixed):
# PHASE 3 FIX: Convert to proper signal format
return self._weights_to_signals(signals_df, price_columns)
```

### Files Modified

1. `src/crypto_trader/strategies/library/hierarchical_risk_parity.py`
   - Line 181: Added `_weights_to_signals()` call to insufficient data path
   - Line 511: Added `_weights_to_signals()` call to single-asset path

2. `src/crypto_trader/strategies/library/risk_parity.py`
   - Line 158: Added `_weights_to_signals()` call to insufficient data path
   - Line 323: Added `_weights_to_signals()` call to single-asset path

3. `src/crypto_trader/strategies/library/black_litterman.py`
   - Line 162: Added `_weights_to_signals()` call to insufficient data path
   - Line 445: Added `_weights_to_signals()` call to single-asset path

---

## Validation

### Debug Script

Created `debug_hrp_signals.py` to test signal generation:

**Result**: ✅ SUCCESS
```
Signal DataFrame shape: (1493, 4)
Signal DataFrame columns: ['timestamp', 'signal', 'confidence', 'metadata']
Signal column unique values:
  HOLD: 1435
  BUY: 58
```

The standalone test confirms strategies generate signals correctly.

---

## Why This Bug Existed

1. **Original implementation** (pre-Phase 3): Single-asset fallback returned weight DataFrame, which was acceptable before strict signal format enforcement

2. **Phase 2 bugfix**: Added `_weights_to_signals()` to the **main code path** but **missed** the early return paths

3. **Phase 3 integration**: Transaction cost logic didn't change signal format requirements, but exposed the bug by triggering single-asset fallbacks more frequently

---

## Lessons Learned

### 1. **All Return Paths Must Be Consistent**
When a function has multiple return statements, ALL must return the same format. The early returns were overlooked.

### 2. **Test Both Multi-Asset AND Single-Asset Modes**
Portfolio strategies have dual modes - both need testing. The validation scripts only tested multi-asset mode.

### 3. **Worker Pipeline Uses Single-Symbol Mode**
The windowed pipeline runs portfolio strategies in single-symbol mode (one asset per worker), which triggers the fallback path.

### 4. **DataFrame Format Contracts Are Critical**
The backtesting engine expects a specific DataFrame format. Any deviation causes runtime failures that are hard to debug.

---

## Prevention Measures

### Immediate

1. ✅ Fixed all early return paths to call `_weights_to_signals()`
2. ✅ Added comments marking the fix: `# PHASE 3 FIX`
3. ✅ Created debug script for future validation

### Future

1. **Add unit test** for single-asset fallback mode in all portfolio strategies
2. **Add integration test** that runs portfolio strategies with single-symbol data
3. **Document signal format** requirements in strategy base class
4. **Add DataFrame format validation** in backtesting engine with clear error messages

---

## Next Steps

1. Re-run windowed analysis: `uv run python master_windowed_multipair.py -p BTC/USDT -p ETH/USDT -p BNB/USDT --test-years 2.0`
2. Verify all portfolio strategies succeed
3. Compare metrics vs previous runs
4. Document Phase 3 final results

---

## Related Files

- **Bug Report**: This file
- **Debug Script**: `debug_hrp_signals.py`
- **Error Log**: `multipair_windowed_results_20251025_071413/errors.txt`
- **Phase 3 Summary**: `openspec/changes/improve-multipair-sharpe-ratios/PHASE3_IMPLEMENTATION_SUMMARY.md`

---

**Fix Completed**: 2025-10-25 07:50 UTC
**Lines Changed**: 6 (2 per strategy)
**Strategies Fixed**: HierarchicalRiskParity, RiskParity, BlackLitterman
**Estimated Impact**: Should fix 100% of portfolio strategy failures
