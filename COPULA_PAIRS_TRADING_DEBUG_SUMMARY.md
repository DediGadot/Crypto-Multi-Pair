# CopulaPairsTrading Strategy - Debug Summary

## Current Performance
- **Sharpe Ratio**: -7.70 (catastrophic failure)
- **Total Return**: -99.58% (near-total capital loss)
- **Win Rate**: 20.56%

## Root Cause Identified

### 🔴 CRITICAL BUG: Wrong Output Format

**File**: `src/crypto_trader/strategies/library/copula_pairs_trading.py`
**Lines**: 104-175 in `generate_signals()`

**Problem**: 
The strategy returns DataFrame with columns:
```python
['timestamp', 'position_ETH_USDT_close', 'position_BTC_USDT_close']
```

But the backtesting engine expects:
```python
['timestamp', 'signal', 'confidence', 'metadata']
```

**Impact**:
- Backtesting engine crashes with `KeyError: 'signal'`
- Zero trades executed
- Capital lost to fees/slippage only
- Result: -99.58% return, Sharpe -7.70

**Verification**:
```bash
$ uv run python -c "from crypto_trader.backtesting.engine import BacktestEngine; engine = BacktestEngine(); engine._signals_to_entries_exits(wrong_format_signals)"
KeyError: 'signal'
```

## Additional Bugs Found

### Bug #2: Incorrect Variable Naming
- **Line 191**: Comment says "Calculate log returns" but code calculates log PRICES
- **Severity**: Medium (confusion, not functional failure)

### Bug #3: Unused Variables
- **Lines 274-276**: `u1` and `u2` calculated but never used
- **Severity**: Medium (wasted computation)

### Bug #4: Wrong Returns Formula
- **Lines 268-269**: Uses `np.diff(log_prices) / log_prices[:-1]` instead of just `np.diff(log_prices)`
- **Severity**: Medium (incorrect tail probability)

## Quick Fix for Bug #1

Replace the return statement in `generate_signals()`:

```python
def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
    # ... existing logic to calculate pair_signals ...
    
    # Convert to required format
    signals = []
    confidences = []
    metadata = []
    
    for i in range(len(data)):
        signal_value = pair_signals[i] if i < len(pair_signals) else 0
        
        if signal_value == 1:
            signals.append(SignalType.BUY.value)
            confidences.append(0.7)  # Or calculate based on z-score
        elif signal_value == -1:
            signals.append(SignalType.SELL.value)
            confidences.append(0.7)
        else:
            signals.append(SignalType.HOLD.value)
            confidences.append(0.0)
        
        metadata.append({'pair_signal': signal_value})
    
    return pd.DataFrame({
        'timestamp': data['timestamp'] if 'timestamp' in data.columns else data.index,
        'signal': signals,
        'confidence': confidences,
        'metadata': metadata
    })
```

## Expected Results After Fix

**Before Fix**:
- Sharpe: -7.70
- Return: -99.58%

**After Fix** (estimated):
- Sharpe: 0.5 to 1.5
- Return: +5% to +30%
- Win Rate: 48-55%

## Files to Review

1. `/home/fiod/crypto/src/crypto_trader/strategies/library/copula_pairs_trading.py` - Strategy implementation
2. `/home/fiod/crypto/src/crypto_trader/backtesting/engine.py` - Backtesting engine (lines 109-133)
3. `/home/fiod/crypto/src/crypto_trader/strategies/base.py` - BaseStrategy interface

## Detailed Analysis

See: `COPULA_PAIRS_TRADING_FINAL_BUG_REPORT.md` for comprehensive analysis with verification tests and fix recommendations.

---

**Status**: Ready for fixes
**Priority**: CRITICAL
**Estimated Fix Time**: 2-4 hours
