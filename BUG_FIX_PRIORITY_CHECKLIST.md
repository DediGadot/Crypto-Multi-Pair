# Bug Fix Priority Checklist

**Quick Reference**: Use this checklist to track bug fixes across all reports
**Total Bugs**: 23 (8 from previous reports + 15 from new audit)

---

## CRITICAL PRIORITY (Must Fix Before Production)

### From Previous Reports
- [ ] **ALGO-1**: Window slicing data leakage
  - File: `master_windowed_multipair.py:109-116`
  - Status: ✅ FIXED (uses correct dataset based on window type)
  - Verification: Check line 102-106 has dataset_type conditional

- [ ] **ALGO-3**: Portfolio Sharpe calculation wrong
  - File: `src/crypto_trader/analysis/multipair_aggregator.py:249-344`
  - Status: ⚠️ PARTIALLY FIXED (computes from return series, but see below)
  - Fix needed: Verify portfolio_std_return calculation includes correlations

- [ ] **ALGO-5**: Sharpe ratio annualization incorrect
  - File: `src/crypto_trader/backtesting/engine.py:126-128`
  - Status: ❌ UNFIXED
  - Impact: Sharpe ratios inflated ~3.5× for 30-day windows

### From New Audit
- [ ] **M1**: Memory leak - passing full datasets to workers
  - File: `master_windowed_multipair.py:536-544`
  - Impact: 2.6GB waste, OOM risk
  - Fix: Pre-slice window data before executor.submit()

- [ ] **CC1**: Cache key string comparison fails
  - File: `src/crypto_trader/analysis/windowed_cache.py:137-138`
  - Impact: 50-100× slower (cache misses)
  - Fix: Use epoch timestamps instead of ISO strings

- [ ] **RC1**: Cache race condition (TOCTOU)
  - File: `src/crypto_trader/analysis/windowed_cache.py:196-230`
  - Impact: Cache corruption, lost results
  - Fix: Add threading.Lock around check-and-insert

- [ ] **TZ1**: Missing timezone conversion to UTC
  - File: `src/crypto_trader/orchestration/multipair_window_manager.py:117`
  - Impact: Wrong train/test split if non-UTC timezone
  - Fix: Use `runtime_date.astimezone(pytz.UTC)`

---

## HIGH PRIORITY (Fix Before Trusting Results)

### From Previous Reports
- [ ] **ALGO-4**: Window boundary off-by-one
  - File: `src/crypto_trader/orchestration/multipair_window_manager.py:252`
  - Impact: All windows 1 period short
  - Fix: Change `< current_end` to `<= current_end`

- [ ] **ALGO-C1**: No temporal validation (train/test separation)
  - File: `master_windowed_multipair.py` (needs new code)
  - Impact: Could miss overlap bugs
  - Fix: Add validation after window generation

### From New Audit
- [ ] **DT1**: Empty pair results crash
  - File: `src/crypto_trader/analysis/multipair_aggregator.py:278-311`
  - Impact: Analysis crashes when one pair fails
  - Fix: Filter out pairs with no valid results

- [ ] **CC2**: No cache versioning
  - File: `src/crypto_trader/analysis/windowed_cache.py:66-76`
  - Impact: Cache breaks after schema changes
  - Fix: Add CACHE_VERSION and _validate_cache_schema()

---

## MEDIUM PRIORITY (Quality & Reliability)

### From Previous Reports
- [ ] **ALGO-C2**: No data alignment verification
  - File: `src/crypto_trader/execution/workers.py` (needs new code)
  - Impact: Silent data corruption possible
  - Fix: Validate timestamp ranges in worker

- [ ] **ALGO-C3**: Inf/NaN masking hides issues
  - File: `src/crypto_trader/analysis/aggregator.py:164-174`
  - Status: ✅ IMPROVED (now logs warnings)
  - Enhancement: Could raise exceptions instead of returning zeros

- [ ] **MULTI-1**: Non-finite Sharpe handling
  - File: `src/crypto_trader/backtesting/engine.py:204`
  - Status: ✅ FIXED (uses np.isfinite())

### From New Audit
- [ ] **M2**: Cache DataFrame O(n²) append
  - File: `src/crypto_trader/analysis/windowed_cache.py:226-230`
  - Impact: Memory fragmentation, slow inserts
  - Fix: Batch insert with _pending_rows buffer

- [ ] **PF1**: Redundant DataFrame copy
  - File: `master_windowed_multipair.py:109-110`
  - Impact: 2× memory, slower processing
  - Fix: Remove .copy() or combine with reset_index()

- [ ] **RC2**: Futures list keeps memory alive
  - File: `master_windowed_multipair.py:555-594`
  - Impact: Memory not freed until all tasks done
  - Fix: Use as_completed() instead of futures list

---

## LOW PRIORITY (Nice to Have)

### From New Audit
- [ ] **PF2**: Inefficient correlation matrix
  - File: `src/crypto_trader/analysis/multipair_aggregator.py:215-230`
  - Impact: 20-30× slower correlation calc
  - Fix: Compute full N×N matrix once with np.corrcoef()

- [ ] **EC1**: No warning for single window
  - File: `src/crypto_trader/analysis/aggregator.py:217-331`
  - Impact: Misleading statistics
  - Fix: Add warning when num_windows < 5

- [ ] **EC2**: Division by zero edge case
  - File: `src/crypto_trader/analysis/multipair_aggregator.py:332-335`
  - Impact: Wrong diversification ratio
  - Fix: Handle avg_individual_sharpe = 0, portfolio_sharpe > 0

- [ ] **TZ2**: DST handling in window duration
  - File: `src/crypto_trader/orchestration/multipair_window_manager.py:236`
  - Impact: None for crypto (no DST)
  - Fix: Use relativedelta instead of timedelta

- [ ] **CQ1**: Inconsistent error handling
  - File: `master_windowed_multipair.py:120-136`
  - Impact: Harder debugging
  - Fix: Standardize error representation

---

## Quick Progress Tracker

**Total Bugs**: 23
**Fixed**: 3 ✅
**Partially Fixed**: 2 ⚠️
**Unfixed**: 18 ❌

**By Priority**:
- Critical: 7 total (1 fixed, 1 partial, 5 unfixed)
- High: 4 total (0 fixed, 4 unfixed)
- Medium: 7 total (2 fixed, 1 partial, 4 unfixed)
- Low: 5 total (0 fixed, 5 unfixed)

---

## Sprint 1 Checklist (Critical Fixes - 40 hours)

Week 1:
- [ ] Day 1-2: Fix **M1** (memory leak) + tests (12h)
- [ ] Day 3: Fix **CC1** (cache keys) + tests (8h)
- [ ] Day 4: Fix **RC1** (race condition) + tests (6h)
- [ ] Day 5: Fix **TZ1** (timezone) + Fix **ALGO-5** (Sharpe annualization) (8h)
- [ ] Weekend: Integration testing (6h)

**Sprint 1 Exit Criteria**:
- [ ] Memory usage < 500MB for 3 pairs analysis
- [ ] Cache hit rate > 90% on re-runs
- [ ] No crashes with mixed UTC/non-UTC timezones
- [ ] All critical tests pass

---

## Sprint 2 Checklist (High Priority - 24 hours)

Week 2:
- [ ] Day 1: Fix **DT1** (empty results) + Fix **ALGO-4** (boundary) (8h)
- [ ] Day 2: Fix **CC2** (versioning) + Fix **ALGO-C1** (validation) (8h)
- [ ] Day 3-4: Fix **M2** + **PF1** + **RC2** (8h)

**Sprint 2 Exit Criteria**:
- [ ] Handles all edge cases gracefully (no crashes)
- [ ] Cache survives code updates
- [ ] Window boundaries validated automatically
- [ ] Memory profile is optimal

---

## Sprint 3 Checklist (Polish - 8 hours)

Week 3:
- [ ] Day 1: Fix **PF2** + **EC1** + **EC2** (4h)
- [ ] Day 2: Fix **CQ1** + Final testing (4h)

**Sprint 3 Exit Criteria**:
- [ ] All performance optimizations applied
- [ ] Edge cases have appropriate warnings
- [ ] Error handling is consistent
- [ ] Documentation updated

---

## Verification Commands

After each fix, run these to verify:

```bash
# Memory profiling
python -m memory_profiler master_windowed_multipair.py -p BTC/USDT -p ETH/USDT --quick

# Cache hit rate check
grep "Cache hit" multipair_windowed_results_*/cache/*.log | wc -l
grep "Cache miss" multipair_windowed_results_*/cache/*.log | wc -l

# Timezone test
python -c "
from datetime import datetime
import pytz
from src.crypto_trader.orchestration.multipair_window_manager import MultiPairTrainTestSplitter

runtime_est = datetime(2025, 1, 1, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
splitter = MultiPairTrainTestSplitter(runtime_date=runtime_est)
assert splitter.runtime_date.tzinfo == pytz.UTC
print('✅ Timezone conversion working')
"

# Edge case test (empty results)
python verify_empty_results_handling.py

# Full test suite
pytest tests/test_memory_profiling.py
pytest tests/test_cache_consistency.py
pytest tests/test_timezone_handling.py
pytest tests/test_edge_cases.py
```

---

## Notes for Developers

### Before Starting Fixes
1. Read full bug reports:
   - `COMPREHENSIVE_BUG_HUNT_REPORT.md` (detailed analysis)
   - `BUG_HUNT_EXECUTIVE_SUMMARY.md` (overview)
   - `ALGORITHMIC_BUGS_REPORT.md` (previous critical bugs)
   - `MULTIPAIR_BUGFIX_SUMMARY.md` (inf/nan fixes)

2. Set up testing environment:
   ```bash
   # Create test data
   python -c "from src.crypto_trader.data.fetchers import BinanceDataFetcher; ..."

   # Install profiling tools
   pip install memory_profiler pytest-benchmark
   ```

3. Create branch for fixes:
   ```bash
   git checkout -b bugfix/sprint1-critical-fixes
   ```

### During Fixes
- **One bug at a time**: Fix, test, commit
- **Write tests first**: Ensure bug is reproducible
- **Verify fix works**: Run verification commands
- **Update this checklist**: Mark items as complete

### After Fixes
- Update documentation
- Run full test suite
- Create PR with:
  - Summary of fixes
  - Before/after metrics
  - Test coverage report

---

## Contact

**Questions about bugs?** Check full reports first:
- Detailed analysis → `COMPREHENSIVE_BUG_HUNT_REPORT.md`
- Quick summary → `BUG_HUNT_EXECUTIVE_SUMMARY.md`

**Found new bugs?** Add to this checklist and update total count.
