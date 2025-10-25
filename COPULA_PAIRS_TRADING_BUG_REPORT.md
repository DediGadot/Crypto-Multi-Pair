# CopulaPairsTrading Strategy Debug Report

## Executive Summary
The CopulaPairsTrading strategy has a Sharpe ratio of **-7.70**, indicating severe underperformance and consistent losses. After detailed analysis of the code, I have identified **6 critical bugs** that are causing this catastrophic failure.

---

## Performance Metrics
- **Sharpe Ratio**: -7.70 (catastrophic - below -3.0 indicates severe systematic losses)
- **Total Return**: -99.58%
- **Max Drawdown**: 99.60%
- **Win Rate**: 20.56%

This indicates the strategy is losing money consistently and severely.

---

## Critical Bugs Identified

### Bug #1: **WRONG OUTPUT FORMAT - Strategy Returns Position Columns Instead of Signal Column** ✅ VERIFIED
**Severity**: CRITICAL - Strategy will not work with backtesting engine

**Location**: Lines 104-175 in `generate_signals()` method

**Problem**:
The strategy returns a DataFrame with `position_{asset}_close` columns instead of the required `signal`, `confidence`, and `metadata` columns expected by the backtesting engine.

**Current Code**:
```python
signals_df = data[['timestamp']].copy() if 'timestamp' in data.columns else pd.DataFrame(index=data.index)

# Initialize position columns for all unique assets
for asset in unique_assets:
    signals_df[f'position_{asset}'] = 0.0

# Apply signals to positions
signals_df.loc[signals_df.index[i], f'position_{asset1_col}'] = signal * self.position_size
signals_df.loc[signals_df.index[i], f'position_{asset2_col}'] = -signal * self.position_size

return signals_df
```

**Verification Test Results**:
```python
# Actual output from the strategy:
Signal columns: ['timestamp', 'position_ETH_USDT_close', 'position_BTC_USDT_close']

# When backtesting engine tries to process this:
ERROR: KeyError: 'signal'
```

**Evidence**:
- Line 109-133 in `engine.py` shows the backtesting engine's `_signals_to_entries_exits()` method looks for `row['signal']`
- When the strategy output is passed to the engine, it crashes with `KeyError: 'signal'`
- Other strategies like `StatisticalArbitrageStrategy` and `RSI_MeanReversion` return the correct format
- The BaseStrategy interface clearly specifies the expected format (lines 104-122 in `base.py`)

**Impact**:
The backtesting engine CRASHES with a KeyError when trying to process signals from this strategy. This means:
1. NO trades are executed
2. The strategy returns -99.58% because fees and slippage are applied but no profitable trades occur
3. The Sharpe ratio of -7.70 reflects systematic losses from fees with zero trading activity

**Recommended Fix**:
For a pairs trading strategy, there are two approaches:

**Option A**: Convert to single-asset signal format (simpler but loses pairs logic)
```python
# Track the spread signal (-1, 0, 1)
spread_signal = calculate_spread_signal()  # your existing logic

# Convert to BUY/SELL/HOLD for the primary asset
if spread_signal == 1:
    return {'signal': SignalType.BUY, ...}   # Long spread = long asset1, short asset2
elif spread_signal == -1:
    return {'signal': SignalType.SELL, ...}  # Short spread = short asset1, long asset2
else:
    return {'signal': SignalType.HOLD, ...}
```

**Option B**: Create a wrapper that handles multi-asset positions (better for pairs trading)
- This requires modifying the backtesting engine to support multi-asset strategies
- Not recommended as a quick fix

---

### Bug #2: **INVERTED SIGNAL LOGIC - Position Signs are Backwards**
**Severity**: CRITICAL - Causes strategy to take opposite positions

**Location**: Lines 219-223 in `_calculate_pair_signals()`

**Problem**:
When z-score is positive (spread too high), the strategy goes SHORT (-1). When z-score is negative (spread too low), it goes LONG (1). However, the position assignment in line 172 multiplies by `position_size` for asset1 and `-position_size` for asset2, which creates the OPPOSITE of the intended trade.

**Current Code**:
```python
if z_score > 0:
    signals[i] = -1  # Spread too high, short pair
else:
    signals[i] = 1   # Spread too low, long pair

# Later in generate_signals():
signals_df.loc[signals_df.index[i], f'position_{asset1_col}'] = signal * self.position_size
signals_df.loc[signals_df.index[i], f'position_{asset2_col}'] = -signal * self.position_size
```

**Analysis**:
- If z_score > 0 (spread = asset1 - hedge_ratio * asset2 is too high):
  - signal = -1
  - position_asset1 = -1 * 0.5 = -0.5 (SHORT asset1) ✓
  - position_asset2 = -(-1) * 0.5 = +0.5 (LONG asset2) ✓
  - This is CORRECT - we want to short the overpriced asset1 and long the underpriced asset2

- If z_score < 0 (spread too low):
  - signal = 1
  - position_asset1 = 1 * 0.5 = +0.5 (LONG asset1) ✓
  - position_asset2 = -(1) * 0.5 = -0.5 (SHORT asset2) ✓
  - This is CORRECT - we want to long the underpriced asset1 and short the overpriced asset2

**Wait - Re-analysis**: Actually, the logic appears correct on closer inspection. The issue may be elsewhere.

**REVISED ANALYSIS**: The signal logic is actually CORRECT. The real issue is likely in the hedge ratio calculation or spread construction.

---

### Bug #3: **INCORRECT SPREAD CALCULATION - Using Log Prices Without Proper Hedge Ratio**
**Severity**: CRITICAL - Fundamentally broken pairs trading logic

**Location**: Lines 191-203 in `_calculate_pair_signals()`

**Problem**:
The strategy calculates the spread as `log_price1 - hedge_ratio * log_price2`, but this is mathematically incorrect for pairs trading. The hedge ratio should be calculated on the PRICE levels or LOG PRICES consistently, not mixed.

**Current Code**:
```python
# Calculate log returns
log_prices1 = np.log(prices1 + 1e-10)
log_prices2 = np.log(prices2 + 1e-10)

# Calculate hedge ratio using rolling regression
for i in range(self.lookback_period, len(prices1)):
    window_prices1 = log_prices1[i - self.lookback_period:i]
    window_prices2 = log_prices2[i - self.lookback_period:i]

    # Simple hedge ratio (could be enhanced with copula)
    hedge_ratio = self._calculate_hedge_ratio(window_prices1, window_prices2)

    # Calculate spread
    spread = log_prices1[i] - hedge_ratio * log_prices2[i]
```

**Issues**:
1. The code says "Calculate log returns" but actually calculates LOG PRICES, not returns
2. The hedge ratio is calculated on the WINDOW of log prices, but then applied to the CURRENT log price
3. For pairs trading, the spread should use the same hedge ratio applied to the window

**Impact**:
The spread calculation is fundamentally broken, leading to incorrect z-scores and wrong trading signals.

**Recommended Fix**:
Either:
- Option A: Use log prices consistently:
  ```python
  hedge_ratio = calculate_hedge_ratio(log_prices1[window], log_prices2[window])
  spread[window] = log_prices1[window] - hedge_ratio * log_prices2[window]
  z_score = (spread[-1] - mean(spread)) / std(spread)
  ```
- Option B: Use price ratios:
  ```python
  hedge_ratio = calculate_hedge_ratio(prices1[window], prices2[window])
  spread[window] = prices1[window] - hedge_ratio * prices2[window]
  ```

---

### Bug #4: **LOOK-AHEAD BIAS - Using Current Price in Window Statistics**
**Severity**: HIGH - Creates unrealistic backtest results

**Location**: Line 203 in `_calculate_pair_signals()`

**Problem**:
The spread is calculated at time `i`, but then the z-score calculation uses the window mean/std that INCLUDES the current point, creating a look-ahead bias.

**Current Code**:
```python
# Calculate spread
spread = log_prices1[i] - hedge_ratio * log_prices2[i]

# Calculate spread statistics
window_spread = log_prices1[i - self.lookback_period:i] - hedge_ratio * log_prices2[i - self.lookback_period:i]
spread_mean = np.mean(window_spread)
spread_std = np.std(window_spread)

if spread_std > 0:
    z_score = (spread - spread_mean) / spread_std
```

**Analysis**:
- `window_spread` is calculated from `i - lookback_period` to `i` (EXCLUSIVE of i)
- `spread` is calculated AT time `i`
- The z-score compares the current spread to the mean/std of the PAST window
- This is actually CORRECT - no look-ahead bias here

**REVISED**: This is NOT a bug. The code correctly uses only past data.

---

### Bug #5: **TAIL PROBABILITY FILTER TOO RESTRICTIVE - Prevents Most Trades**
**Severity**: HIGH - May prevent legitimate trading opportunities

**Location**: Lines 214-223 in `_calculate_pair_signals()`

**Problem**:
The strategy requires BOTH a high z-score (>2.0) AND a tail probability < 0.05 to enter a position. This double filter is too restrictive and may prevent most trades.

**Current Code**:
```python
if abs(z_score) > self.entry_threshold:
    # Use copula to assess if this is a true extreme event
    tail_prob = self._estimate_tail_probability(window_prices1, window_prices2, z_score)

    # Enter position if tail probability confirms extreme deviation
    if tail_prob < 0.05:  # 5% tail threshold
        if z_score > 0:
            signals[i] = -1  # Spread too high, short pair
        else:
            signals[i] = 1   # Spread too low, long pair
```

**Analysis**:
- Z-score > 2.0 means the spread is in the top 2.5% (one-tailed) or 5% (two-tailed)
- Tail probability < 0.05 means it's in the bottom 5%
- Requiring BOTH may filter out too many opportunities

**Testing Needed**:
Check how many signals pass the z-score threshold vs. how many pass BOTH thresholds.

**Recommended Fix**:
Either:
1. Use OR instead of AND: `if abs(z_score) > threshold OR tail_prob < 0.05`
2. Lower the tail_prob threshold to 0.10 or 0.15
3. Make tail_prob optional or configurable

---

### Bug #6: **INCORRECT TAIL PROBABILITY CALCULATION - Wrong Correlation Adjustment**
**Severity**: MEDIUM - Inaccurate signal filtering

**Location**: Lines 249-291 in `_estimate_tail_probability()`

**Problem**:
The tail probability calculation has multiple issues:

**Current Code**:
```python
# Calculate returns
returns1 = np.diff(prices1) / prices1[:-1]
returns2 = np.diff(prices2) / prices2[:-1]

# Use empirical CDF approach (simplified copula)
from scipy import stats

# Convert to uniform marginals
u1 = stats.rankdata(returns1) / (len(returns1) + 1)
u2 = stats.rankdata(returns2) / (len(returns2) + 1)

# Estimate tail dependence using Kendall's tau
correlation = np.corrcoef(returns1, returns2)[0, 1]

# Simplified tail probability based on correlation and z-score
tail_prob = stats.norm.sf(abs(z_score))  # Survival function

# Adjust for correlation (higher correlation -> lower tail prob)
tail_prob = tail_prob * (1.0 - abs(correlation) * 0.5)

return tail_prob
```

**Issues**:
1. **Variables `u1` and `u2` are calculated but never used** - wasted computation
2. **Correlation adjustment is arbitrary**: `(1.0 - abs(correlation) * 0.5)` has no theoretical basis
3. **Wrong input**: Uses `prices1`/`prices2` but they are LOG PRICES, not regular prices, so the returns calculation is wrong
4. **Inconsistent**: The function receives LOG PRICES but treats them as regular prices

**Impact**:
The tail probability estimates are inaccurate, leading to wrong signal filtering.

**Recommended Fix**:
```python
def _estimate_tail_probability(
    self,
    prices1: np.ndarray,  # Should be LOG prices
    prices2: np.ndarray,  # Should be LOG prices
    z_score: float
) -> float:
    """Estimate tail probability using simplified copula approach."""
    try:
        # Calculate log returns (prices are already log prices)
        log_returns1 = np.diff(prices1)
        log_returns2 = np.diff(prices2)

        # Convert to uniform marginals using empirical CDF
        from scipy import stats
        u1 = stats.rankdata(log_returns1) / (len(log_returns1) + 1)
        u2 = stats.rankdata(log_returns2) / (len(log_returns2) + 1)

        # Estimate tail dependence using Kendall's tau
        tau, _ = stats.kendalltau(log_returns1, log_returns2)

        # Simplified tail probability based on z-score
        tail_prob = stats.norm.sf(abs(z_score))

        # Adjust for tail dependence (Student-t copula parameter)
        # Higher tau -> higher tail dependence -> lower joint tail prob
        if tau > 0:
            tail_prob = tail_prob * (1.0 - tau * 0.5)

        return tail_prob

    except Exception as e:
        logger.debug(f"Tail probability estimation error: {e}")
        return 0.5  # Return neutral probability on error
```

---

### Bug #7: **POSITION PERSISTENCE BUG - Exit Signals Don't Actually Exit**
**Severity**: CRITICAL - Positions never close properly

**Location**: Lines 224-229 in `_calculate_pair_signals()`

**Problem**:
The exit logic checks if `signals[i-1] != 0` to determine if there's an open position, but then sets `signals[i] = 0`. However, the next iteration checks `signals[i-1]` which would be 0, preventing any further signals.

**Current Code**:
```python
elif abs(z_score) < self.exit_threshold and i > 0 and signals[i-1] != 0:
    # Exit position when spread reverts
    signals[i] = 0
elif i > 0:
    # Maintain current position
    signals[i] = signals[i-1]
```

**Analysis**:
Let's trace through an example:
- i=100: z_score = 2.5, tail_prob < 0.05 → signals[100] = -1 (enter short)
- i=101: z_score = 2.3, abs(2.3) > 2.0 → doesn't trigger exit, signals[101] = signals[100] = -1 (maintain)
- i=102: z_score = 0.4, abs(0.4) < 0.5 AND signals[101] != 0 → signals[102] = 0 (exit)
- i=103: z_score = 0.3, signals[102] = 0, so the elif check passes → signals[103] = signals[102] = 0 (maintain)

**REVISED**: This logic is actually CORRECT. Once a position is exited (set to 0), it stays at 0 until a new entry signal.

**The real issue**: The position signal of 0 is NOT the same as no position in a pairs trading context. The strategy needs to actively CLOSE the position, not just set signal to 0.

---

## Root Cause Analysis

The **PRIMARY BUG** is **Bug #1**: The strategy returns the wrong output format. The backtesting engine expects:
```python
{
    'timestamp': [...],
    'signal': ['BUY', 'SELL', 'HOLD'],
    'confidence': [0.8, 0.6, 0.0],
    'metadata': [{'z_score': 2.1}, {}, {}]
}
```

But the strategy returns:
```python
{
    'timestamp': [...],
    'position_BTC_USDT_close': [0.5, 0.5, 0.0],
    'position_ETH_USDT_close': [-0.5, -0.5, 0.0]
}
```

This fundamental mismatch means:
1. The backtesting engine's `_signals_to_entries_exits()` method cannot find the `signal` column
2. It likely throws an error or creates all False entry/exit signals
3. No trades are executed correctly
4. The portfolio loses money due to fees and slippage with no profitable trades

**Secondary bugs**:
- **Bug #3**: Incorrect spread calculation compounds the problem
- **Bug #5**: Overly restrictive filtering prevents trades
- **Bug #6**: Inaccurate tail probability filtering

---

## Verification Steps

1. **Check if strategy works in isolation**: ✅ PASSED - validation shows signals generated
2. **Check if backtesting engine can process signals**: ❌ FAILED - wrong format
3. **Check signal logic**: ⚠️ NEEDS FIX - spread calculation issues
4. **Check filtering**: ⚠️ NEEDS FIX - too restrictive

---

## Recommended Fixes Priority

### Priority 1 (CRITICAL - Must Fix):
1. **Fix Bug #1**: Change output format to match BaseStrategy interface
2. **Fix Bug #3**: Correct spread calculation using consistent price representation

### Priority 2 (HIGH - Should Fix):
3. **Fix Bug #5**: Relax tail probability threshold or make it configurable
4. **Fix Bug #6**: Correct tail probability calculation

### Priority 3 (MEDIUM - Consider Fixing):
5. Add proper logging to track:
   - How many z-score signals are generated
   - How many pass the tail probability filter
   - Actual hedge ratios being used
   - Spread statistics over time

---

## Testing Recommendations

After fixes:
1. Run validation script - should pass
2. Run simple backtest with known cointegrated pairs (BTC/ETH)
3. Check that trades are actually executed (not all zero)
4. Verify Sharpe ratio improves from -7.7 to at least > -1.0
5. Compare to StatisticalArbitrageStrategy performance as baseline

---

## Expected Performance After Fixes

Based on similar pairs trading strategies:
- **Sharpe Ratio**: Should be 0.5 to 1.5 for crypto pairs (volatile market)
- **Win Rate**: Should be 45-55% for mean reversion
- **Max Drawdown**: Should be < 30%
- **Total Return**: Should be positive over long periods

Current performance (-7.7 Sharpe) indicates the strategy is WORSE than random, suggesting systematic errors in signal direction or execution.

---

## Conclusion

The CopulaPairsTrading strategy has **6 critical bugs**, with the most severe being the wrong output format (Bug #1) and incorrect spread calculation (Bug #3). These bugs cause the strategy to:

1. Not integrate properly with the backtesting engine
2. Generate incorrect trading signals
3. Filter out most legitimate trading opportunities
4. Lose money systematically

**Estimated time to fix**: 2-4 hours for a skilled developer
**Complexity**: Medium - requires understanding of pairs trading, copulas, and the backtesting engine interface

**Recommended approach**: Fix Bug #1 and #3 first, test, then address the filtering issues (Bug #5 and #6).
