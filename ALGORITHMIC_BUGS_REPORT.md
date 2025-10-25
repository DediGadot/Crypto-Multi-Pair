# Algorithmic Bugs Report - Multi-Pair Windowed Analysis Pipeline

**Date**: 2025-10-20
**Pipeline Version**: Post-8-bugfix
**Analysis Depth**: Complete algorithmic audit

---

## Executive Summary

After comprehensive review of the entire multi-pair windowed analysis pipeline, I've identified **5 CRITICAL ALGORITHMIC BUGS** and **3 DESIGN CONCERNS** that affect the validity of train/test results.

**Status**: ⚠️ **PIPELINE HAS MAJOR ALGORITHMIC ISSUES**

**Impact**: Results are currently unreliable due to data leakage, incorrect window slicing, and statistical aggregation errors.

---

## CRITICAL BUG #1: Window Slicing Uses Wrong Index Range (DATA LEAKAGE)

### Location
`master_windowed_multipair.py`, line 116

### Issue
```python
window_data = pair_data.iloc[pair_window.start_idx:pair_window.end_idx].copy()
```

**This slices the FULL dataset, not the split train/test data!**

### Root Cause
The window generation in `MultiPairTrainTestSplitter.generate_synchronized_windows()` (lines 192-296) creates windows with indices that reference positions in the **split** data (train_data_dict or test_data_dict).

However, `master_windowed_multipair.py` line 115 uses:
```python
pair_data = data_dict[pair]  # This is the FULL dataset!
```

Then at line 116, it slices using indices meant for the split data:
```python
window_data = pair_data.iloc[pair_window.start_idx:pair_window.end_idx].copy()
```

### Consequence
**SEVERE DATA LEAKAGE**: Windows can access data from both train AND test sets because indices calculated for split data are applied to full data.

### Example
If train set has 10,000 rows and test set has 5,000 rows:
- A test window with `start_idx=100, end_idx=200` (meant for test_data[100:200])
- Gets applied to `full_data[100:200]` instead
- This accesses **training set data** when it should only access test data!

### Fix Required
```python
# BEFORE (WRONG):
pair_data = data_dict[pair]
window_data = pair_data.iloc[pair_window.start_idx:pair_window.end_idx].copy()

# AFTER (CORRECT):
# Need to maintain separate train_data_dict and test_data_dict
if window.dataset_type == 'train':
    pair_data = train_data_dict[pair]
else:
    pair_data = test_data_dict[pair]
window_data = pair_data.iloc[pair_window.start_idx:pair_window.end_idx].copy()
```

### Verification Test
1. Check if test window indices ever produce data from before the cutoff date
2. Verify window timestamps fall within expected train/test ranges
3. Compare actual sliced data dates vs expected dates from WindowSpec

---

## CRITICAL BUG #2: Incorrect Return Calculation in Aggregator

### Location
`src/crypto_trader/analysis/aggregator.py`, line 260

### Issue
```python
returns = [r.get('total_return', 0.0) for r in results if 'error' not in r]
```

**This extracts raw decimals but treats them as percentages later!**

### Root Cause
VectorBT returns `total_return` as a decimal (e.g., 0.15 = 15% return).

The aggregator stores this as `mean_return = 0.15`, but later code interprets it as a percentage value in reports.

### Evidence
In `multipair_aggregator.py` line 274:
```python
portfolio_mean_return = float(np.mean(mean_returns))
```

This averages decimal values (e.g., `mean([0.10, 0.12]) = 0.11`), which is **11%**, not **0.11%**.

But in HTML report generation (`master_windowed_multipair.py` line 333):
```python
html_parts.append(f"<td>{formatter.format_percentage(pair_metric.mean_return)}</td>")
```

The `format_percentage()` function expects a decimal and multiplies by 100, so:
- If `mean_return = 0.11` (11%)
- `format_percentage(0.11)` → "11.00%"  ✅ CORRECT

Actually, upon further inspection, this is **NOT a bug** - the system correctly handles decimals throughout. The HTML formatter handles the conversion.

**STATUS: FALSE ALARM - NOT A BUG**

---

## CRITICAL BUG #3: Portfolio Sharpe Calculation is Wrong

### Location
`src/crypto_trader/analysis/multipair_aggregator.py`, lines 267-283

### Issue
```python
# Equal-weight portfolio: average metrics across pairs
sharpes = [m.mean_sharpe for m in pair_metrics.values()]
portfolio_sharpe = float(np.mean(sharpes))  # WRONG!
```

### Root Cause
**Portfolio Sharpe ≠ Average of Individual Sharpes**

The correct formula for portfolio Sharpe is:
```
Portfolio Sharpe = Portfolio Return / Portfolio Volatility
```

Where portfolio volatility depends on:
- Individual asset volatilities
- **Correlations between assets** (the whole point of diversification!)

Simply averaging Sharpe ratios ignores correlation structure and produces **mathematically incorrect** risk-adjusted returns.

### Mathematical Example
Two assets with correlation = 0:
- Asset A: Sharpe = 1.0, Return = 10%, Vol = 10%
- Asset B: Sharpe = 1.0, Return = 10%, Vol = 10%

**Naive average**: Sharpe = (1.0 + 1.0) / 2 = 1.0

**Actual portfolio** (50/50 weights, 0 correlation):
- Portfolio Return = 10%
- Portfolio Vol = sqrt(0.5² × 10² + 0.5² × 10²) = 7.07%
- **Portfolio Sharpe = 10% / 7.07% = 1.41**  ⬅️ MUCH HIGHER!

### Current Code Impact
The current implementation **systematically underestimates** diversification benefits. Strategies appear worse than they actually are when combining uncorrelated assets.

### Fix Required
```python
# Correct implementation:
def compute_portfolio_metrics(self, pair_metrics, pair_results):
    """Compute TRUE portfolio metrics using return series."""

    # Extract return series for each pair across all windows
    pair_return_series = {}
    for pair, results in pair_results.items():
        returns = [r.get('total_return', 0.0) for r in results if r]
        pair_return_series[pair] = returns

    # Calculate equal-weight portfolio returns
    num_pairs = len(pair_return_series)
    num_windows = min(len(r) for r in pair_return_series.values())

    portfolio_returns = []
    for i in range(num_windows):
        # Equal-weight portfolio return
        window_return = sum(pair_return_series[p][i] for p in pair_return_series.keys()) / num_pairs
        portfolio_returns.append(window_return)

    # Calculate portfolio metrics from return series
    portfolio_mean_return = np.mean(portfolio_returns)
    portfolio_std_return = np.std(portfolio_returns)

    # TRUE Portfolio Sharpe = Mean Return / Std Return
    if portfolio_std_return > 0:
        portfolio_sharpe = portfolio_mean_return / portfolio_std_return
    else:
        portfolio_sharpe = 0.0

    # Drawdown calculation (use actual portfolio equity curve)
    # ... (implementation needed)

    return {
        'portfolio_mean_return': portfolio_mean_return,
        'portfolio_std_return': portfolio_std_return,
        'portfolio_sharpe': portfolio_sharpe,
        # ...
    }
```

---

## CRITICAL BUG #4: Window Boundary Off-By-One Error

### Location
`src/crypto_trader/orchestration/multipair_window_manager.py`, line 252

### Issue
```python
pair_mask = (data.index >= current_start) & (data.index < current_end)
```

This uses `< current_end` (exclusive end), but line 244 sets:
```python
current_end = current_start + window_duration
```

### Root Cause
If window_duration is exactly 30 days, `current_end` is exactly 30 days after start.

But using `< current_end` means the last day is **excluded** from the window.

So a "30-day window" actually contains **less than 30 days of data**.

### Example
- `current_start = 2024-01-01 00:00`
- `window_duration = timedelta(days=30)`
- `current_end = 2024-01-31 00:00`  ⬅️ Exactly 30 days later

Mask: `(data.index >= 2024-01-01) & (data.index < 2024-01-31)`

**Result**: Excludes all data from 2024-01-31! Window is only 29 full days.

### Fix Required
```python
# Use <= current_end OR adjust current_end to be inclusive
pair_mask = (data.index >= current_start) & (data.index <= current_end)
```

Or:
```python
# Add one period to make end inclusive
current_end = current_start + window_duration + timedelta(hours=1)  # For 1h timeframe
pair_mask = (data.index >= current_start) & (data.index < current_end)
```

### Impact
- All windows are **1 period shorter** than specified
- Horizon calculations are incorrect
- Window statistics are computed on less data than intended

---

## CRITICAL BUG #5: Sharpe Ratio Not Properly Annualized

### Location
`src/crypto_trader/backtesting/engine.py`, lines 127-128

### Issue
```python
sharpe_ratio = portfolio.sharpe_ratio()
```

VectorBT's `sharpe_ratio()` uses the `freq` parameter passed to `Portfolio.from_signals()` (line 311).

However, this `freq` value comes from:
```python
freq_map: Dict[Timeframe, str] = {
    Timeframe.HOUR_1: "1H",
    # ...
}
freq_value = freq_map.get(timeframe, "1H")
```

### Root Cause
VectorBT annualizes Sharpe ratio based on `freq`, but the **window length varies** by horizon!

For a 30-day window with 1h data:
- VectorBT assumes 365 days of data (8760 hours)
- Actually only has 720 hours (30 days)
- **Annualization factor is WRONG**

### Mathematical Impact
Sharpe ratio formula: `(Mean Return - Risk Free Rate) / Std Dev × sqrt(periods per year)`

If VectorBT uses `sqrt(8760)` but window only has `720 hours`:
- Actual annualization: `sqrt(8760 / 720) = sqrt(12.17) = 3.49x`
- **Sharpe ratios are inflated by ~3.5x for 30-day windows!**

### Verification
Check if 30-day Sharpe > 90-day Sharpe **systematically** even when raw returns are similar.

### Fix Required
Calculate Sharpe manually using actual window length:
```python
# After backtest
returns = portfolio.returns()
window_days = (end_date - start_date).days
periods_per_year = (365 / window_days) * len(returns)

mean_return = returns.mean()
std_return = returns.std()
sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year) if std_return > 0 else 0.0
```

---

## DESIGN CONCERN #1: No Validation of Train/Test Temporal Separation

### Issue
No code validates that test windows occur **after** all train windows.

### Risk
If window generation has bugs, test windows could overlap with or precede training windows, invalidating the entire methodology.

### Recommendation
Add validation in `master_windowed_multipair.py`:
```python
# After window generation
for horizon_name, windows in all_windows.items():
    train_windows = windows['train']
    test_windows = windows['test']

    if train_windows and test_windows:
        last_train_date = max(w.end_date for w in train_windows)
        first_test_date = min(w.start_date for w in test_windows)

        if first_test_date <= last_train_date:
            raise ValueError(
                f"TEMPORAL VIOLATION: Test data ({first_test_date}) overlaps "
                f"with training data (ends {last_train_date})"
            )
```

---

## DESIGN CONCERN #2: Missing Data Alignment Verification

### Issue
Workers receive data as dictionaries (`data_dict`) reconstructed from DataFrames.

No verification that:
1. Timestamps are preserved correctly
2. Index alignment is maintained
3. Window slicing produces expected date ranges

### Risk
Silent data corruption or misalignment during dict conversion.

### Recommendation
Add timestamp validation in worker:
```python
# In run_backtest_worker, after data reconstruction
data = pd.DataFrame(data_dict)
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # VALIDATE: Check date range matches expected window
    actual_start = data['timestamp'].min()
    actual_end = data['timestamp'].max()

    # Expected dates should be passed in metadata
    # Log warning if mismatch detected
```

---

## DESIGN CONCERN #3: Inf/NaN Handling May Hide Real Issues

### Location
`src/crypto_trader/analysis/aggregator.py`, lines 160-179

### Issue
```python
finite_mask = np.isfinite(arr)
if not finite_mask.any():
    logger.warning(f"All values are non-finite (inf/nan), returning zero statistics")
    return {'mean': 0.0, 'median': 0.0, ...}
```

### Problem
Silently replacing inf/NaN with zeros **hides underlying issues**:
- Why are metrics infinite/NaN?
- Division by zero in calculations?
- Numerical instability?

### Recommendation
Replace silent fallback with **exception raising**:
```python
if not finite_mask.any():
    raise ValueError(
        f"Cannot compute statistics: all values are inf/NaN. "
        f"This indicates a serious problem in metric calculation. "
        f"Original values: {values}"
    )
```

This forces investigation of root causes rather than masking problems.

---

## Summary of Bugs by Severity

| Bug | Severity | Impact | Status |
|-----|----------|--------|--------|
| #1: Window Slicing Data Leakage | 🔴 CRITICAL | Train/test contamination | **UNFIXED** |
| #2: Return Aggregation | 🟢 NONE | False alarm - not a bug | **N/A** |
| #3: Portfolio Sharpe Formula | 🔴 CRITICAL | Underestimates diversification | **UNFIXED** |
| #4: Window Boundary Off-By-One | 🟡 HIGH | All windows 1 period short | **UNFIXED** |
| #5: Sharpe Annualization | 🔴 CRITICAL | 3.5x inflation of Sharpe ratios | **UNFIXED** |
| Concern #1: No Temporal Validation | 🟡 MEDIUM | Could miss overlap bugs | **UNFIXED** |
| Concern #2: No Data Alignment Check | 🟡 MEDIUM | Silent corruption possible | **UNFIXED** |
| Concern #3: Inf/NaN Masking | 🟠 MEDIUM | Hides real problems | **UNFIXED** |

---

## Recommended Fix Priority

### Phase 1: Critical Fixes (Must Fix Before ANY Production Use)
1. **BUG #1**: Fix window slicing data leakage
2. **BUG #3**: Implement correct portfolio Sharpe calculation
3. **BUG #5**: Fix Sharpe ratio annualization

### Phase 2: High Priority (Fix Before Trusting Results)
4. **BUG #4**: Fix window boundary off-by-one
5. **Concern #1**: Add temporal separation validation

### Phase 3: Quality Improvements
6. **Concern #2**: Add data alignment verification
7. **Concern #3**: Replace silent NaN handling with exceptions

---

## Test Cases Needed

### Test 1: Window Slicing Validation
```python
# Generate windows for known data
# Check that:
# - Train windows only access data before cutoff
# - Test windows only access data after cutoff
# - No window spans the cutoff date
```

### Test 2: Portfolio Sharpe Verification
```python
# Create two uncorrelated synthetic assets
# Known: correlation = 0, individual Sharpe = 1.0
# Expected: portfolio Sharpe = 1.41 (with proper calculation)
# Current: portfolio Sharpe = 1.0 (with buggy averaging)
```

### Test 3: Sharpe Annualization
```python
# Same strategy, same data
# Test with 30-day window vs 90-day window
# Verify: Sharpe ratio converges (not 3x different)
```

### Test 4: Boundary Conditions
```python
# 30-day window starting 2024-01-01
# Verify: Last data point is from 2024-01-30 23:00 (for 1h data)
# Current bug: Last point is 2024-01-29 23:00 (missing full day)
```

---

## Conclusion

The pipeline has **4 critical algorithmic bugs** that make current results **unreliable**:

1. **Data leakage** from using wrong index ranges
2. **Mathematically incorrect** portfolio Sharpe calculation
3. **Incorrectly annualized** Sharpe ratios (inflated 3.5x)
4. **Off-by-one** window boundaries (missing last period)

**ALL CRITICAL BUGS MUST BE FIXED before results can be trusted.**

Current report showing SMA_Crossover with Sharpe=0.75 is **not reliable** due to these issues.

---

**Report Compiled By**: Claude Code Debugger
**Analysis Duration**: Comprehensive code review
**Files Analyzed**: 6 core modules, 2000+ lines of code
**Bugs Found**: 4 critical, 3 design concerns
