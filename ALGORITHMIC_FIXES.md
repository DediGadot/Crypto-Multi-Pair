# Algorithmic Bug Fixes

**Date**: 2025-10-20
**Status**: ✅ **2 CRITICAL BUGS CONFIRMED** by automated tests
**Action Required**: Apply all fixes before trusting results

---

## CONFIRMED BUG #1: Window Slicing Data Leakage ⚠️ CRITICAL

### Verification Result
```
❌ DATA LEAKAGE CONFIRMED
Test window 0 accesses training data from 2023-01-01 (before cutoff 2024-01-01)
```

### Root Cause
`master_windowed_multipair.py` lines 115-116:
```python
pair_data = data_dict[pair]  # ← FULL dataset (both train + test)
window_data = pair_data.iloc[pair_window.start_idx:pair_window.end_idx].copy()
```

Indices from `pair_window` reference positions in **split** data (train_data_dict or test_data_dict), but are applied to the **full** dataset.

### Fix

Replace lines 82-154 in `master_windowed_multipair.py`:

```python
def run_multipair_window_backtest(
    strategy_name: str,
    window: MultiPairWindowSpec,
    train_data_dict: Dict[str, pd.DataFrame],  # ← ADD THIS
    test_data_dict: Dict[str, pd.DataFrame],   # ← ADD THIS
    timeframe: str,
    cache: WindowedResultsCache
) -> Dict[str, Any]:
    """
    Run backtest for a multi-pair window.

    Returns dict mapping pair -> result
    """
    results = {}

    for pair, pair_window in window.pair_windows.items():
        # Check cache
        cached = cache.get_result(
            strategy=strategy_name,
            symbol=pair,
            timeframe=timeframe,
            horizon=window.horizon_name,
            window_id=window.window_id,
            dataset_type=window.dataset_type,
            start_date=pair_window.start_date.isoformat(),
            end_date=pair_window.end_date.isoformat()
        )

        if cached is not None:
            results[pair] = cached
            continue

        # FIX: Use correct dataset based on window type
        if window.dataset_type == 'train':
            pair_data = train_data_dict[pair]
        else:
            pair_data = test_data_dict[pair]

        # Now indices are correct for the split dataset
        window_data = pair_data.iloc[pair_window.start_idx:pair_window.end_idx].copy()

        # Apply timestamp fix
        if 'timestamp' in window_data.columns:
            window_data = window_data.drop(columns=['timestamp'])

        data_dict_for_worker = window_data.reset_index().to_dict('list')

        try:
            # Worker will use default params from strategy class
            result = run_backtest_worker(
                strategy_name=strategy_name,
                data_dict=data_dict_for_worker,
                horizon_name=window.horizon_name,
                horizon_days=int(window.horizon_name.replace('d', '')),
                symbol=pair,
                timeframe=timeframe,
                default_params={}  # Worker uses strategy defaults
            )

            if result and 'error' not in result:
                # Cache it
                cache.store_result(
                    result=result,
                    strategy=strategy_name,
                    symbol=pair,
                    timeframe=timeframe,
                    horizon=window.horizon_name,
                    window_id=window.window_id,
                    dataset_type=window.dataset_type,
                    start_date=pair_window.start_date.isoformat(),
                    end_date=pair_window.end_date.isoformat()
                )
                results[pair] = result
        except Exception as e:
            logger.debug(f"Backtest failed for {strategy_name}/{pair}: {e}")
            results[pair] = None

    return results
```

Then update the call sites (around line 520-527):

```python
# OLD (BUGGY):
future = executor.submit(
    run_multipair_window_backtest,
    strategy_name,
    window,
    data_dict,  # ← WRONG: Full dataset
    timeframe,
    cache
)

# NEW (FIXED):
# First split the data
train_data_dict, test_data_dict = splitter.split_data(data_dict)

# Then pass both splits to worker
future = executor.submit(
    run_multipair_window_backtest,
    strategy_name,
    window,
    train_data_dict,  # ← Split training data
    test_data_dict,   # ← Split test data
    timeframe,
    cache
)
```

### Testing the Fix

After applying fix, run:
```bash
python verify_algorithmic_bugs.py
```

Expected output should change from:
```
❌ Contains training data! (starts before cutoff)
```

To:
```
✅ Correctly isolates test data
```

---

## CONFIRMED BUG #3: Incorrect Portfolio Sharpe Calculation ⚠️ CRITICAL

### Verification Result
```
❌ BUG CONFIRMED
TRUE portfolio Sharpe: 3.9093
BUGGY averaging: 2.6284
Error: 32.8% underestimation
```

### Root Cause
`src/crypto_trader/analysis/multipair_aggregator.py` lines 281-283:
```python
sharpes = [m.mean_sharpe for m in pair_metrics.values()]
portfolio_sharpe = float(np.mean(sharpes))  # ← WRONG FORMULA
```

**Mathematical Error**: Portfolio Sharpe ≠ Average of Individual Sharpes

Correct formula:
```
Portfolio Sharpe = (Portfolio Mean Return) / (Portfolio Std Dev)
```

### Fix

Replace the entire `compute_portfolio_metrics()` function (lines 242-302):

```python
def compute_portfolio_metrics(
    self,
    pair_metrics: Dict[str, WindowedMetrics],
    pair_results: Dict[str, List[Dict[str, Any]]]  # ← ADD THIS PARAMETER
) -> Dict[str, float]:
    """
    Compute portfolio-level metrics from per-pair metrics.

    Uses equal-weight portfolio assumption with CORRECT Sharpe calculation.

    Args:
        pair_metrics: Dict mapping pair symbol to WindowedMetrics
        pair_results: Dict mapping pair symbol to list of window results
                     (REQUIRED for proper Sharpe calculation)

    Returns:
        Dict with portfolio metrics
    """
    if not pair_metrics:
        return {
            'portfolio_mean_return': 0.0,
            'portfolio_median_return': 0.0,
            'portfolio_std_return': 0.0,
            'portfolio_sharpe': 0.0,
            'portfolio_drawdown': 0.0,
            'diversification_ratio': 1.0
        }

    # Extract return series for each pair across all windows
    pair_return_series = {}
    for pair, results in pair_results.items():
        # Extract returns, handling both 'total_return' and 'total_return_pct'
        returns = []
        for r in results:
            if r and 'error' not in r:
                ret = r.get('total_return', r.get('total_return_pct', 0.0))
                returns.append(ret)
        pair_return_series[pair] = returns

    # Calculate number of windows (use minimum across all pairs)
    num_windows = min(len(r) for r in pair_return_series.values()) if pair_return_series else 0

    if num_windows == 0:
        return {
            'portfolio_mean_return': 0.0,
            'portfolio_median_return': 0.0,
            'portfolio_std_return': 0.0,
            'portfolio_sharpe': 0.0,
            'portfolio_drawdown': 0.0,
            'diversification_ratio': 1.0
        }

    # Compute equal-weight portfolio returns for each window
    num_pairs = len(pair_return_series)
    portfolio_returns = []

    for i in range(num_windows):
        # Equal-weight portfolio return for this window
        window_return = sum(
            pair_return_series[pair][i]
            for pair in pair_return_series.keys()
        ) / num_pairs
        portfolio_returns.append(window_return)

    # Calculate portfolio statistics from return series
    portfolio_mean_return = float(np.mean(portfolio_returns))
    portfolio_median_return = float(np.median(portfolio_returns))
    portfolio_std_return = float(np.std(portfolio_returns))

    # TRUE Portfolio Sharpe = Mean / Std
    if portfolio_std_return > 0:
        portfolio_sharpe = portfolio_mean_return / portfolio_std_return
    else:
        portfolio_sharpe = 0.0

    # Portfolio drawdown: worst drawdown across all pairs (conservative)
    drawdowns = [m.mean_drawdown for m in pair_metrics.values()]
    portfolio_drawdown = float(np.max(drawdowns))

    # Diversification ratio: Portfolio Sharpe / Average individual Sharpe
    individual_sharpes = [m.mean_sharpe for m in pair_metrics.values()]
    avg_individual_sharpe = float(np.mean(individual_sharpes))

    if avg_individual_sharpe != 0:
        diversification_ratio = portfolio_sharpe / avg_individual_sharpe
    else:
        diversification_ratio = 1.0

    return {
        'portfolio_mean_return': portfolio_mean_return,
        'portfolio_median_return': portfolio_median_return,
        'portfolio_std_return': portfolio_std_return,
        'portfolio_sharpe': portfolio_sharpe,
        'portfolio_drawdown': portfolio_drawdown,
        'diversification_ratio': diversification_ratio
    }
```

Then update the call site in `aggregate_multipair_windows()` (line 343):

```python
# OLD:
portfolio_metrics = self.compute_portfolio_metrics(pair_metrics)

# NEW:
portfolio_metrics = self.compute_portfolio_metrics(pair_metrics, pair_results)
```

### Testing the Fix

After applying fix, run:
```bash
python verify_algorithmic_bugs.py
```

The error percentage should drop from ~33% to <5%.

---

## BUG #4: Window Boundary (VERIFIED AS NOT A BUG)

### Verification Result
```
✅ Window contains expected number of periods
```

**Analysis**: The test showed that despite using `< current_end`, the window still contains the correct number of periods (720 hours = 30 days).

This is because:
- `current_end = current_start + timedelta(days=30)` = `2024-01-31 00:00`
- Hourly data from `2024-01-01 00:00` to `2024-01-30 23:00` = 30 full days
- The `< 2024-01-31 00:00` correctly excludes data from Jan 31

**Conclusion**: NOT A BUG - The boundary logic is correct as-is.

---

## BUG #5: Sharpe Annualization (MANUAL VERIFICATION REQUIRED)

### Current Status
```
⚠️  MANUAL VERIFICATION REQUIRED
```

### Issue
VectorBT annualizes Sharpe using `freq` parameter, but this may not correctly account for window length differences.

### Verification Procedure

Run actual backtests with different horizons:

```bash
# Test with 30d horizon
python master_windowed_multipair.py -p BTC/USDT -p ETH/USDT --horizons 30

# Test with 90d horizon
python master_windowed_multipair.py -p BTC/USDT -p ETH/USDT --horizons 90

# Compare Sharpe ratios for same strategy
```

**Expected**: Sharpe ratios should be comparable (within 20% variance)
**Bug exists if**: 30d Sharpe is >2x the 90d Sharpe

### Fix (if bug confirmed)

Replace Sharpe calculation in `src/crypto_trader/backtesting/engine.py` (line 127):

```python
# OLD:
sharpe_ratio = portfolio.sharpe_ratio()

# NEW:
# Calculate Sharpe manually with correct annualization
returns = portfolio.returns()
window_days = (end_date - start_date).days

# Periods per year based on actual window length
periods_in_window = len(returns)
days_per_period = window_days / periods_in_window if periods_in_window > 0 else 1
periods_per_year = 365.25 / days_per_period

mean_return = returns.mean()
std_return = returns.std()

if std_return > 0:
    sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year)
else:
    sharpe_ratio = 0.0
```

---

## Summary of Required Actions

### Immediate (Critical Bugs - CONFIRMED)

1. ✅ **Apply Fix for Bug #1** (Window Slicing Data Leakage)
   - File: `master_windowed_multipair.py`
   - Impact: Eliminates train/test contamination
   - Verification: Run `verify_algorithmic_bugs.py`

2. ✅ **Apply Fix for Bug #3** (Portfolio Sharpe Calculation)
   - File: `src/crypto_trader/analysis/multipair_aggregator.py`
   - Impact: Correct diversification benefits
   - Verification: Run `verify_algorithmic_bugs.py`

### Optional (Verification Needed)

3. ⚠️ **Verify Bug #5** (Sharpe Annualization)
   - Method: Compare results across different horizons
   - Apply fix only if confirmed

### Not Needed

4. ✅ **Bug #4** - Verified as not a bug, no action required

---

## Post-Fix Validation

After applying fixes #1 and #3:

1. Run verification script:
   ```bash
   python verify_algorithmic_bugs.py
   ```
   Expected: All automated tests pass

2. Re-run full analysis:
   ```bash
   python master_windowed_multipair.py -p BTC/USDT -p ETH/USDT --quick
   ```

3. Compare results to previous run:
   - Train/test Sharpe ratios should change
   - Portfolio Sharpe should increase (diversification benefit)
   - Overfitting patterns should become clearer

4. Document changes in results summary

---

## Expected Impact on Results

### Bug #1 Fix
- Test set Sharpe ratios will **decrease** (removing leaked training data)
- Overfitting gap (train - test) will **increase**
- Some strategies may drop from positive to negative test Sharpe

### Bug #3 Fix
- Portfolio Sharpe will **increase** (correct diversification benefit)
- Diversification ratio will change from ~1.0 to >1.0 for uncorrelated pairs
- Multi-asset strategies will appear more attractive

**Overall**: Results will be **more conservative** and **mathematically correct**.

---

**Next Steps**:
1. Apply Bug #1 fix
2. Apply Bug #3 fix
3. Run verification tests
4. Re-run full pipeline
5. Compare new results to old results
6. Update documentation with corrected methodology

