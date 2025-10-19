# Bug Analysis Report - debug.log

**Analysis Date:** 2025-10-19
**Log File:** debug.log (57.4MB)
**Analysis Scope:** Complete error analysis from master.py execution

---

## Executive Summary

Identified **3 distinct issues** affecting 31 backtest failures:

1. **🔴 CRITICAL**: DynamicEnsemble TypeError - Data type mismatch (6 failures)
2. **🟡 EXPECTED**: StatisticalArbitrage cointegration failures (24 failures)
3. **🟡 MINOR**: Portfolio strategy configuration issue (1 failure)

---

## Bug #1: DynamicEnsemble TypeError ⚠️ CRITICAL

### Symptom
```
TypeError: '>=' not supported between instances of 'str' and 'Timestamp'
```

### Affected Strategy
- **DynamicEnsemble** (all time horizons: 30d, 90d, 180d, 365d, 730d, 1050d)
- **Failure Count:** 6 out of 6 attempts (100% failure rate)

### Error Location
```
File: /home/fiod/crypto/src/crypto_trader/analysis/performance_store.py
Line: 109
Code: df = df[df["timestamp"] >= cutoff]
```

### Full Stack Trace
```
Traceback (most recent call last):
  File "/home/fiod/crypto/master.py", line 1129, in run_backtest_worker
    result = engine.run_backtest(...)
  File "/home/fiod/crypto/src/crypto_trader/backtesting/engine.py", line 258, in run_backtest
    signals = strategy.generate_signals(data)
  File "/home/fiod/crypto/src/crypto_trader/strategies/library/dynamic_ensemble.py", line 103, in generate_signals
    weights = self._load_weights()
  File "/home/fiod/crypto/src/crypto_trader/strategies/library/dynamic_ensemble.py", line 73, in _load_weights
    metrics = self.store.recent(self.child_names, days=self.lookback_days)
  File "/home/fiod/crypto/src/crypto_trader/analysis/performance_store.py", line 109, in recent
    df = df[df["timestamp"] >= cutoff]
```

### Root Cause Analysis

**PRIMARY CAUSE:** Corrupted CSV file preventing proper timestamp parsing

**Evidence:**
1. Performance metrics CSV has a malformed row:
   ```
   Line 1: timestamp,strategy,symbol,timeframe,sharpe,total_return,max_drawdown,win_rate
   Line 2: 2025-10-18 22:16:40.204423+00:00,RSI_MeanReversion,BTC/USDT,1h,-2.4003...
   Line 3: 9318  <-- CORRUPTED: Only one value, missing all other columns
   Line 4: 2025-10-17 20:34:16.708359+00:00,RSI_MeanReversion,BTC/USDT,1h,0.20188...
   ```

2. When pandas reads this CSV with `parse_dates=["timestamp"]`, the malformed row causes:
   - Inconsistent column parsing
   - Mixed data types in the "timestamp" column (some strings, some timestamps)
   - Comparison failures when filtering by date

3. The code attempts to parse dates on load:
   ```python
   # performance_store.py:60
   df = pd.read_csv(self.path, parse_dates=["timestamp"])
   ```
   But with the corrupted row, parsing fails silently or produces mixed types.

### Impact

**Severity:** CRITICAL
- **DynamicEnsemble strategy completely non-functional**
- Affects meta-strategy performance evaluation
- Prevents ensemble-based trading

**Affected Backtests:**
- DynamicEnsemble_30d: FAILED (0.06s)
- DynamicEnsemble_90d: FAILED (0.06s)
- DynamicEnsemble_180d: FAILED (0.14s)
- DynamicEnsemble_365d: FAILED (0.17s)
- DynamicEnsemble_730d: FAILED (0.29s)
- DynamicEnsemble_1050d: FAILED (1.66s)

### Recommended Fix

**Solution 1: Clean Corrupted CSV (Immediate Fix)**
```bash
# Backup current file
cp data/performance/performance_metrics.csv data/performance/performance_metrics.csv.backup

# Remove corrupted row (line 3)
sed -i '3d' data/performance/performance_metrics.csv
```

**Solution 2: Add Robust CSV Validation (Permanent Fix)**

Modify `performance_store.py` `_load()` method:

```python
def _load(self) -> pd.DataFrame:
    if not self.path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "strategy",
                "symbol",
                "timeframe",
                "sharpe",
                "total_return",
                "max_drawdown",
                "win_rate",
            ]
        )
    try:
        # Add error_bad_lines parameter to skip malformed rows
        df = pd.read_csv(
            self.path,
            parse_dates=["timestamp"],
            on_bad_lines='warn'  # Pandas 1.4+ (or use error_bad_lines=False for older)
        )

        # Ensure timestamp column is datetime type
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Drop rows where timestamp couldn't be parsed
            df = df.dropna(subset=['timestamp'])

        logger.debug(f"Loaded {len(df)} performance records from {self.path}")
        return df
    except Exception as exc:  # pragma: no cover - unexpected IO failure
        logger.warning(f"Failed to load performance store {self.path}: {exc}")
        return pd.DataFrame()
```

**Solution 3: Add CSV Validation on Write**

Prevent corrupted data from being written in the first place by validating records before appending.

---

## Bug #2: StatisticalArbitrage Cointegration Failures ⚠️ EXPECTED BEHAVIOR

### Symptom
```
Pairs BTC/USDT/ETH/USDT not cointegrated - no trading opportunity
```

### Affected Strategy
- **StatisticalArbitrage** (multi-pair)
- **Failure Count:** 24 out of 24 attempts (100% failure rate)

### Error Analysis

**This is EXPECTED BEHAVIOR, not a bug.**

**Explanation:**
1. StatisticalArbitrage strategy requires pairs to be **cointegrated** (statistically bound)
2. BTC/USDT and ETH/USDT are NOT cointegrated in the tested time periods
3. The strategy correctly identifies this and refuses to trade
4. This is proper risk management - trading non-cointegrated pairs would be dangerous

**Affected Horizons:**
- 30d: 4 failures
- 90d: 4 failures
- 180d: 4 failures
- 365d: 4 failures
- 730d: 4 failures
- 1050d: 4 failures

### Recommendation

**No fix needed** - This is correct behavior.

**Alternative Actions:**
1. Test with different asset pairs (e.g., similar coins like BTC/ETH direct pair)
2. Use different time periods where cointegration might exist
3. Accept that StatisticalArbitrage won't always find opportunities (this is normal)
4. Consider implementing cointegration detection in pre-flight checks to avoid wasted backtests

### Suggested Enhancement

Add early cointegration check to skip unnecessary backtests:

```python
# In master.py or strategy configuration
def should_run_statistical_arbitrage(symbols, data):
    """Check if pairs are cointegrated before running backtest."""
    from statsmodels.tsa.stattools import coint

    if len(symbols) < 2:
        return False

    # Quick cointegration test
    p_value = coint(data[symbols[0]]['close'], data[symbols[1]]['close'])[1]

    if p_value > 0.05:  # Not cointegrated at 95% confidence
        logger.info(f"Skipping StatisticalArbitrage - pairs not cointegrated (p={p_value:.4f})")
        return False

    return True
```

---

## Bug #3: Portfolio Strategy Configuration Issue ⚠️ MINOR

### Symptom
```
Deep dive analysis failed: Portfolio strategy requires 'assets' configuration
```

### Affected Component
- **Deep dive analysis generation** (reporting phase)
- **Failure Count:** 1 occurrence

### Error Location
```
File: /home/fiod/crypto/master.py
Line: 3143
Function: _generate_deep_dive_analysis
```

### Root Cause

Portfolio strategies require an 'assets' configuration parameter that wasn't provided during deep dive analysis.

### Impact

**Severity:** MINOR
- Only affects report generation, not core backtesting
- Deep dive analysis is supplementary to main results
- Main backtest results are still valid

### Recommended Fix

Add configuration validation before generating deep dive:

```python
def _generate_deep_dive_analysis(self, strategy_name, results):
    """Generate deep dive analysis with proper validation."""
    try:
        # Check if strategy requires special configuration
        if self._is_portfolio_strategy(strategy_name):
            if not hasattr(self, 'assets_config'):
                logger.warning(
                    f"Skipping deep dive for {strategy_name} - "
                    "portfolio strategies require assets configuration"
                )
                return None

        # ... rest of deep dive logic
    except Exception as exc:
        logger.error(f"Deep dive analysis failed: {exc}")
        logger.debug(f"Traceback:", exc_info=True)
```

---

## Summary Statistics

### Total Errors Found
| Error Type | Count | Severity | Status |
|-----------|-------|----------|--------|
| DynamicEnsemble TypeError | 6 | CRITICAL | Needs Fix |
| StatisticalArbitrage cointegration | 24 | INFO | Expected Behavior |
| Portfolio config missing | 1 | MINOR | Low Priority |
| **TOTAL** | **31** | - | - |

### Failure Rate by Strategy
| Strategy | Attempts | Failures | Success Rate |
|----------|----------|----------|--------------|
| DynamicEnsemble | 6 | 6 | 0% ⚠️ |
| StatisticalArbitrage | 24 | 24 | 0% ✓ (expected) |
| Others | ~230 | 0 | 100% ✓ |

### Critical Bugs Requiring Action
1. **Fix corrupted performance_metrics.csv** (Line 3: "9318")
2. **Add robust CSV parsing to performance_store.py**

---

## Immediate Action Items

### Priority 1: Fix DynamicEnsemble (CRITICAL)

1. **Clean corrupted CSV:**
   ```bash
   cd /home/fiod/crypto
   cp data/performance/performance_metrics.csv data/performance/performance_metrics.csv.backup
   sed -i '3d' data/performance/performance_metrics.csv
   ```

2. **Verify fix:**
   ```bash
   python3 -c "
   import pandas as pd
   df = pd.read_csv('data/performance/performance_metrics.csv', parse_dates=['timestamp'])
   print(f'Loaded {len(df)} rows')
   print(f'Timestamp type: {df[\"timestamp\"].dtype}')
   print(f'All timestamps valid: {df[\"timestamp\"].notna().all()}')
   "
   ```

3. **Test DynamicEnsemble:**
   ```bash
   uv run python master.py -h 30 --quick --workers 1 2>&1 | grep -A 5 "DynamicEnsemble"
   ```

### Priority 2: Enhance PerformanceStore (IMPORTANT)

Apply the robust CSV loading fix shown in Bug #1 Solution 2.

### Priority 3: Document Cointegration Behavior (NICE TO HAVE)

Add note to StatisticalArbitrage documentation explaining when it will fail (non-cointegrated pairs).

---

## Testing Recommendations

After fixes, test with:

```bash
# Test DynamicEnsemble specifically
uv run python -c "
from crypto_trader.strategies.library.dynamic_ensemble import DynamicEnsemble
from crypto_trader.analysis.performance_store import PerformanceStore
import pandas as pd

# Test performance store loading
store = PerformanceStore()
df = store.recent(['RSI_MeanReversion', 'SMA_Crossover'], days=90)
print(f'Loaded {len(df)} records')
print(f'Timestamp dtype: {df[\"timestamp\"].dtype}')

# Test DynamicEnsemble initialization
strategy = DynamicEnsemble()
print('✅ DynamicEnsemble initialized successfully')
"

# Run full backtest
uv run python master.py -h 30 --quick --workers 2
```

---

## Prevention Measures

1. **Add CSV integrity validation** in PerformanceStore
2. **Add unit tests** for PerformanceStore with malformed data
3. **Add data validation** before writing to CSV
4. **Add logging** for CSV parse warnings
5. **Monitor performance_metrics.csv** for corruption

---

## Files to Modify

1. `src/crypto_trader/analysis/performance_store.py` - Add robust CSV parsing
2. `data/performance/performance_metrics.csv` - Remove corrupted line 3
3. `master.py` (optional) - Add deep dive analysis validation

---

**Report Generated:** 2025-10-19
**Analyzed By:** Claude Code Bug Analysis Tool
**Status:** Ready for Implementation
