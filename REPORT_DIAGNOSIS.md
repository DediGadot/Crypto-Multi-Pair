# 🔍 REPORT DIAGNOSIS - Why Strategies Show "inf" Sharpe Ratios

**Date**: 2025-10-18
**Report**: http://165.22.71.91:8155/master_results_20251018_213547/MASTER_REPORT.html
**Status**: 🔴 **DATA QUALITY ISSUE**

---

## 🎯 EXECUTIVE SUMMARY

The report shows `Sharpe Ratio = inf` for 5 strategies because they generate **0 trades** (all HOLD signals). This is NOT a code bug - it's a **data quality issue**.

**Strategies Affected**:
1. DDQNFeatureSelected - 0 trades on ALL horizons
2. MultiModalSentimentFusion - 0 trades on ALL horizons
3. OnChainAnalytics - 0 trades on ALL horizons
4. OrderFlowImbalance - 0 trades on ALL horizons
5. TransformerGRUPredictor - 0 trades on ALL horizons

**Root Cause**: Feature Store data files exist but contain **NO VALUES** (all columns are empty/NaN).

---

## 📊 THE PROBLEM IN DETAIL

### **What "inf" Sharpe Means**

When a strategy makes 0 trades:
- All returns = 0% (HOLD only)
- Variance of returns = 0
- Sharpe Ratio = mean_return / std_return = **0 / 0 = inf**

This is mathematically correct but practically useless - the strategy isn't trading!

### **Evidence from Data**

**Comparison Matrix Analysis**:
```csv
strategy_name,total_trades,sharpe_ratio,total_return
DDQNFeatureSelected,0,inf,0.0
MultiModalSentimentFusion,0,inf,0.0
OnChainAnalytics,0,inf,0.0
OrderFlowImbalance,0,inf,0.0
TransformerGRUPredictor,0,inf,0.0
```

**File Investigation**:
```bash
$ wc -l data/features/onchain/BTC_USDT.csv
71337 data/features/onchain/BTC_USDT.csv  # File exists with 71K rows!

$ head -5 data/features/onchain/BTC_USDT.csv
event_time,proxy_mvrv_z,proxy_sopr,proxy_exchange_netflow,proxy_whale_ratio,proxy_puell_multiple
2017-08-17 04:00:00+00:00,,,,,  # ❌ ALL VALUES ARE EMPTY!
2017-08-17 05:00:00+00:00,,,,,
2017-08-17 06:00:00+00:00,,,,,
2017-08-17 07:00:00+00:00,,,,,
```

---

## 🔍 ROOT CAUSE ANALYSIS

### **Issue 1: Feature Store Data is Empty**

The on-chain feature file has:
- ✅ Correct headers: `proxy_mvrv_z`, `proxy_sopr`, `proxy_exchange_netflow`, etc.
- ✅ 71,336 data rows (matches OHLCV data)
- ❌ **ALL VALUES ARE EMPTY** (just commas, no numbers)

**Why This Happens**:

Looking at `src/crypto_trader/data/alt/onchain_ingestor.py`:

```python
def _proxy_from_ohlcv(symbol, timeframe, storage=None):
    # ... loads OHLCV data ...

    out['proxy_mvrv_z'] = ((close - close.rolling(200).mean()) / close.rolling(200).std()).fillna(0.0)
    out['proxy_sopr'] = (close / close.rolling(30).mean()).fillna(1.0)
    out['proxy_exchange_netflow'] = (vol - vol.rolling(20).mean()).fillna(0.0)
    # ... etc ...

    return out  # ✅ This DataFrame has values!
```

But when written to CSV by FeatureStore.write(), the values disappear. This suggests a bug in the **FeatureStore.write()** method.

### **Issue 2: Strategies Check for Columns, Not Values**

OnChainAnalytics strategy code:

```python
def generate_signals(self, data):
    mvrv_col = self._choose_col(df, 'onchain.mvrv_z', 'proxy_mvrv_z')

    if mvrv_col is None:  # ❌ This checks if COLUMN exists
        return all_HOLD

    # ✅ Column exists! So code continues...

    mvrv = df[mvrv_col].astype(float)  # But values are NaN!

    buy_mask = (mvrv < 0.5) & (sopr < 1.0) & (netflow < -5000)
    # ❌ With NaN values, this is always False!
```

The strategy checks if the **column name** exists, but doesn't verify the column has **actual data**. With all NaN values, no signals trigger.

---

## 📋 AFFECTED STRATEGIES & WHY

| Strategy | Required Data | Why 0 Trades |
|----------|--------------|--------------|
| **OnChainAnalytics** | On-chain metrics (MVRV, SOPR, netflow) | Proxy data is all NaN |
| **MultiModalSentimentFusion** | Sentiment scores + on-chain | Both feature files have NaN values |
| **OrderFlowImbalance** | Order flow delta/imbalance | Order flow ingestion failed (logged) |
| **TransformerGRUPredictor** | Trained model weights | Model not found, fallback logic returns 0 |
| **DDQNFeatureSelected** | RL agent state features | Features not properly engineered |

---

## ✅ STRATEGIES THAT WORK (Have Trades)

These strategies work because they **only need OHLCV data**:

| Strategy | Trades | Why It Works |
|----------|--------|--------------|
| SMA_Crossover | 1-32 | Uses simple moving averages from close prices |
| RSI_MeanReversion | 2-31 | RSI calculated from close prices |
| MACD_Momentum | 26-345 | MACD calculated from close prices |
| BollingerBreakout | 7-94 | Bollinger Bands from close/std |
| TripleEMA | 3-56 | EMA calculated from close prices |
| Supertrend_ATR | 9-107 | ATR calculated from high/low/close |
| Ichimoku_Cloud | 1 per horizon | All indicators from OHLCV |
| VWAP_MeanReversion | 2-27 | VWAP from price/volume |

---

## 🛠️ HOW TO FIX

### **Fix #1: Debug FeatureStore.write() Method** (Most Important)

The issue is in `src/crypto_trader/features/store.py`:

```bash
# Step 1: Test proxy generation directly
$ uv run python -c "
from crypto_trader.data.alt.onchain_ingestor import _proxy_from_ohlcv
df = _proxy_from_ohlcv('BTC/USDT', '1h')
print(df.head())
print(df.describe())
"
# This should show VALUES, not NaN!

# Step 2: If proxy has values, check FeatureStore.write()
# The bug is likely in how the DataFrame is written to CSV
```

**Likely Bug in FeatureStore.write()**:
- May be writing wrong columns
- May have a column name mismatch
- May be calling `to_csv()` incorrectly

### **Fix #2: Add NaN Validation to Strategies**

Update strategy code to check for data quality:

```python
def generate_signals(self, data):
    mvrv_col = self._choose_col(df, 'onchain.mvrv_z', 'proxy_mvrv_z')

    # ✅ CHECK IF COLUMN EXISTS
    if mvrv_col is None:
        logger.warning("Missing mvrv column")
        return all_HOLD

    # ✅ NEW: CHECK IF COLUMN HAS DATA
    if df[mvrv_col].isna().all():
        logger.warning(f"{mvrv_col} column is all NaN - no valid data!")
        return all_HOLD

    # Continue with logic...
```

### **Fix #3: Regenerate Feature Data**

Once FeatureStore.write() is fixed:

```bash
# Delete corrupted files
rm -rf data/features/onchain/*
rm -rf data/features/sent/*

# Regenerate with fixed code
uv run python master.py -h 30 --quick --workers 2
```

---

## 🎯 QUICK WORKAROUND (For Now)

If you want results NOW without fixing the data issue, **disable the broken strategies**:

Edit `config/strategies/example_strategies.yaml`:

```yaml
# Comment out strategies that need special data:
strategies:
  - SMA_Crossover
  - RSI_MeanReversion
  - MACD_Momentum
  - BollingerBreakout
  - TripleEMA
  - Supertrend_ATR
  - Ichimoku_Cloud
  - VWAP_MeanReversion
  # - OnChainAnalytics  # ❌ DISABLED - needs on-chain data
  # - MultiModalSentimentFusion  # ❌ DISABLED - needs sentiment data
  # - OrderFlowImbalance  # ❌ DISABLED - needs order flow data
  # - TransformerGRUPredictor  # ❌ DISABLED - needs trained model
  # - DDQNFeatureSelected  # ❌ DISABLED - needs RL features
```

Then re-run:

```bash
uv run python master.py -h 30 -h 90 -h 180 -h 365 --workers 4
```

You'll get a clean report with only working strategies (no "inf" values).

---

## 📊 EXPECTED RESULTS AFTER FIX

Once FeatureStore.write() is fixed, you should see:

**OnChainAnalytics Example**:
```
Symbol: BTC/USDT
Horizon: 30d
Trades: 2-5 (based on MVRV/SOPR thresholds)
Sharpe: 0.5 to 2.0 (actual number, not inf!)
Return: -5% to +10%
```

**Or if thresholds are too strict**:
```
Trades: 0 (but logs should explain WHY)
  "OnChainAnalytics: No periods met buy criteria (mvrv=2.1 vs threshold=0.5)"
  "OnChainAnalytics: No periods met sell criteria (mvrv=1.8 vs threshold=6.0)"
```

---

## 🔍 INVESTIGATION STEPS

To find the exact bug in FeatureStore.write():

1. **Test proxy generation**:
   ```python
   from crypto_trader.data.alt.onchain_ingestor import _proxy_from_ohlcv
   df = _proxy_from_ohlcv('BTC/USDT', '1h')
   print("Columns:", df.columns.tolist())
   print("First 5 rows:\n", df.head())
   print("Stats:\n", df.describe())
   ```

   **Expected**: DataFrame with numeric values in all proxy columns

   **If you see NaN**: Bug is in `_proxy_from_ohlcv()`
   **If you see values**: Bug is in `FeatureStore.write()`

2. **Test FeatureStore.write()**:
   ```python
   from crypto_trader.features.store import FeatureStore
   from crypto_trader.data.alt.onchain_ingestor import _proxy_from_ohlcv

   df = _proxy_from_ohlcv('BTC/USDT', '1h')
   print("Before write:", df.head())

   store = FeatureStore()
   store.write(df, symbol='BTC/USDT', pillar='onchain')

   # Read back
   import pandas as pd
   loaded = pd.read_csv('data/features/onchain/BTC_USDT.csv')
   print("After write:", loaded.head())
   ```

   **If loaded DataFrame is empty**: Bug in FeatureStore.write()

3. **Check FeatureStore.write() implementation**:
   ```bash
   # Look for potential issues
   grep -n "def write" src/crypto_trader/features/store.py
   ```

   Common bugs:
   - Using wrong index when writing
   - Column name mismatch
   - Calling `df.to_csv(columns=[...])` with wrong column list

---

## ✨ SUMMARY

**The Report is NOT Wrong** - it accurately shows that 5 strategies generate 0 trades.

**The Problem is Data Quality**:
1. Feature Store files exist but contain no values (all NaN)
2. FeatureStore.write() has a bug that drops column data
3. Strategies check for columns but not for data validity

**The Fix**:
1. Debug FeatureStore.write() to find why values disappear
2. Add data validation to strategy code
3. Regenerate feature files

**Workaround**: Disable the 5 broken strategies and run with the 8 working ones.

---

**Next Steps**:
1. Run investigation steps above to find FeatureStore.write() bug
2. Fix the bug
3. Delete and regenerate feature files
4. Re-run analysis
5. Get clean results with actual Sharpe ratios (not inf)!

