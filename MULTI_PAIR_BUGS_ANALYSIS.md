# Multi-Pair Master.py Bug Analysis

## 📊 Report Analysis Summary

**Report**: `/home/fiod/crypto/master_results_20251014_091331/MASTER_REPORT.txt`
**Command**: `python master.py --multi-pair`
**Total Jobs**: 180 (40 single-pair + 140 multi-pair)
**Successful**: 76 (42% success rate)
**Failed**: 104 (58% failure rate)

---

## 🐛 Bug #11: StatisticalArbitrage Fails on 30d Horizon

### **Symptom**:
StatisticalArbitrage shows 0.00% returns on ALL horizons (90d, 180d, 365d, 730d) and 0 trades executed.

### **Root Cause**:
The strategy has a validation that requires `lookback_period >= 50` (line 153 of `statistical_arbitrage_pairs.py`), but the master.py worker is setting lookback_period based on the horizon_days parameter without ensuring the minimum threshold.

### **Evidence from Log**:
```
2025-10-14 09:14:40.350 | ERROR    | __main__:run_parallel_analysis:1352 - Backtest failed for StatisticalArbitrage (multi) on 30d: Statistical Arbitrage execution error: lookback_period must be at least 50
    raise ValueError("lookback_period must be at least 50")
ValueError: lookback_period must be at least 50
```

### **Fix Location**: `master.py` line ~545

### **Proposed Fix**:
```python
# In run_statistical_arbitrage_worker function
lookback_period = max(50, horizon_days // 2)  # Ensure minimum 50

params = {
    'pair1_symbol': pair[0],
    'pair2_symbol': pair[1],
    'lookback_period': lookback_period,  # Already validated minimum
    'entry_threshold': 2.0,
    'exit_threshold': 0.5
}
```

---

## 🐛 Bug #12: SOTA 2025 Strategies Fail with Constructor Mismatch

### **Symptom**:
All SOTA 2025 portfolio strategies fail with:
```
TypeError: HierarchicalRiskParityStrategy.__init__() got an unexpected keyword argument 'name'
```

Affected strategies:
- HierarchicalRiskParity
- BlackLitterman
- RiskParity
- CopulaPairsTrading
- DeepRLPortfolio

### **Root Cause**:
Line 817 of `master.py`:
```python
strategy = strategy_class(name=strategy_name, config=config_params)
```

These strategies have `__init__()` methods that take NO parameters:
```python
def __init__(self):  # No parameters!
    super().__init__(name="HierarchicalRiskParity")
```

They expect to be instantiated without arguments, then configured via `initialize(params)`.

### **Evidence from Log**:
```
2025-10-14 09:15:14.777 | ERROR    | __main__:run_parallel_analysis:1352 - Backtest failed for HierarchicalRiskParity (multi) on 30d: HierarchicalRiskParity execution error: HierarchicalRiskParityStrategy.__init__() got an unexpected keyword argument 'name'
TypeError: HierarchicalRiskParityStrategy.__init__() got an unexpected keyword argument 'name'
```

### **Fix Location**: `master.py` line 817

### **Proposed Fix**:
```python
# Check strategy signature and instantiate appropriately
import inspect

strategy_class = registry.get_strategy(strategy_name)
config_params = default_params or {}

# Inspect __init__ signature
init_signature = inspect.signature(strategy_class.__init__)
params = list(init_signature.parameters.keys())

# If __init__ accepts name/config, pass them
if 'name' in params and 'config' in params:
    strategy = strategy_class(name=strategy_name, config=config_params)
else:
    # SOTA 2025 strategies: instantiate without args, then initialize
    strategy = strategy_class()
    strategy.initialize({
        'asset_symbols': asset_symbols,
        **config_params
    })
```

---

## 🐛 Bug #13: Composite Scoring Tie-Breaking Incorrect

### **Symptom**:
Ichimoku_Cloud and PortfolioRebalancer both show score 0.569, but:
- **PortfolioRebalancer**: +103.2% returns, won 5/5 horizons, Sharpe 1.10
- **Ichimoku_Cloud**: +71.6% returns, won 0/5 horizons, Sharpe 0.48

PortfolioRebalancer should clearly rank #1, not tied at #1.

### **Root Cause**:
The composite scoring formula may be producing identical scores due to:
1. Rounding/precision issues in the normalization
2. Missing tie-breaker logic (should use total_return or horizons_won as secondary sort)
3. Possible bug in min-max normalization when values are at extremes

### **Fix Location**: `master.py` lines ~1400-1500 (ranking/scoring section)

### **Proposed Fix**:
```python
# After calculating composite scores, sort with tie-breakers
results_df = results_df.sort_values(
    by=['composite_score', 'total_return', 'horizons_won', 'sharpe_ratio'],
    ascending=[False, False, False, False]
)

# Add explicit rank with tie-breaker
results_df['rank'] = range(1, len(results_df) + 1)
```

---

## 🐛 Bug #14: StatisticalArbitrage Shows 0% When Cointegration Fails

### **Symptom**:
When StatisticalArbitrage finds that assets are NOT cointegrated, it returns 0.00% returns, 0 trades, 0 Sharpe, 0 drawdown, making it look like the strategy "ran" but had no opportunity.

### **Root Cause**:
The strategy correctly returns HOLD signals when pairs are not cointegrated, but the backtest results show confusing 0.00% metrics rather than indicating "N/A - pairs not cointegrated".

### **Current Behavior**:
```python
if not self.is_cointegrated:
    logger.warning("Assets not cointegrated. No trading signals generated.")
    return self._create_hold_signals(data)  # All HOLD → 0% return
```

### **Proposed Fix**:
Master.py should detect when StatisticalArbitrage returns all HOLD signals and report it as an error with a clear message:

```python
# In worker function after getting signals
if strategy_name == "StatisticalArbitrage":
    if (signals['signal'] == SignalType.HOLD.value).all():
        return {
            'strategy_name': strategy_name,
            'horizon': horizon_name,
            'error': 'Pairs not cointegrated - no trading opportunity'
        }
```

---

## 📈 Success Rate Impact

### Current State (with bugs):
- **76/180 successful** (42%)
- 5 out of 7 multi-pair strategies completely failing
- StatisticalArbitrage producing misleading 0% results

### Expected After Fixes:
- **~140/180 successful** (78%)
- All 7 multi-pair strategies working
- Clear error messages for cointegration failures
- Proper ranking with tie-breakers

---

## 🔧 Priority Order for Fixes

1. **Bug #12** (HIGHEST): Fix SOTA 2025 strategy instantiation
   - Impact: Enables 5 strategies (HRP, BlackLitterman, RiskParity, Copula, DeepRL)
   - Expected to fix ~100 failed jobs

2. **Bug #11** (HIGH): Fix StatisticalArbitrage lookback validation
   - Impact: Fixes StatisticalArbitrage on shorter horizons
   - Expected to fix ~4 failed jobs on 30d

3. **Bug #13** (MEDIUM): Fix composite scoring ties
   - Impact: Correct ranking in reports
   - User experience/accuracy issue

4. **Bug #14** (LOW): Improve cointegration failure messaging
   - Impact: Better error reporting
   - Nice-to-have for clarity

---

## 🧪 Testing Plan

After fixes, run:
```bash
uv run python master.py --multi-pair
```

Expected results:
- 140+ successful backtests (up from 76)
- All 7 multi-pair strategies producing results
- PortfolioRebalancer ranked #1 (not tied)
- Clear error messages for any remaining failures
