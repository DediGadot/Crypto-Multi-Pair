# Bug Fix Line Numbers Reference

Quick reference for all bug fixes applied to Black-Litterman and CopulaPairsTrading strategies.

---

## Black-Litterman (`black_litterman.py`)

### Lines Modified

| Line Range | Description | Change Type |
|------------|-------------|-------------|
| 121-127 | Single-asset detection | Modified from error+empty to warning+fallback |
| 192-197 | Ledoit-Wolf shrinkage | Enhanced comments and variable naming |
| 283-313 | Single-asset fallback method | **NEW METHOD** - Returns 100% weight |

### Detailed Changes

**Lines 121-127**: Single-asset graceful handling
```python
# OLD (lines 121-123):
if len(price_columns) < 2:
    logger.error(f"Need at least 2 assets, found {len(price_columns)}")
    return pd.DataFrame()

# NEW (lines 121-127):
# BUGFIX: Gracefully handle single-asset case
if len(price_columns) < 2:
    logger.warning(
        f"Black-Litterman requires ≥2 assets, found {len(price_columns)}. "
        f"Falling back to single-asset allocation."
    )
    return self._generate_single_asset_signals(data, price_columns)
```

**Lines 192-197**: Ledoit-Wolf clarity
```python
# OLD (lines 188-191):
S = risk_models.CovarianceShrinkage(
    pd.DataFrame({col: (1 + returns[col]).cumprod() for col in returns.columns})
).ledoit_wolf()

# NEW (lines 192-197):
# CRITICAL FIX: Use Ledoit-Wolf shrinkage instead of sample covariance
# This is essential for crypto (high noise, low sample size)
prices_df = pd.DataFrame({
    col: (1 + returns[col]).cumprod() for col in returns.columns
})
S = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()
```

**Lines 283-313**: New fallback method
```python
def _generate_single_asset_signals(
    self,
    data: pd.DataFrame,
    price_columns: list
) -> pd.DataFrame:
    """Generate signals for single-asset case (graceful degradation)."""
    signals_df = pd.DataFrame({
        'timestamp': data['timestamp'] if 'timestamp' in data.columns else data.index
    })

    if len(price_columns) == 1:
        # 100% allocation to single asset
        signals_df[f'weight_{price_columns[0]}'] = 1.0
        logger.info(f"Generated single-asset signals: 100% to {price_columns[0]}")
    else:
        # No assets - return empty weights (edge case)
        logger.warning("No price columns found, returning empty signals")

    return signals_df
```

---

## CopulaPairsTrading (`copula_pairs_trading.py`)

### Lines Modified

| Line Range | Description | Change Type |
|------------|-------------|-------------|
| 33-36 | Remove conditional import | Modified from try/except to direct import |
| 132-138 | Single-asset detection | Modified from error+hold_frame to warning+fallback |
| 645-675 | Single-asset fallback method | **NEW METHOD** - Returns HOLD signals |

### Detailed Changes

**Lines 33-36**: Direct import (no conditional)
```python
# OLD (lines 34-39):
try:
    from statsmodels.tsa.stattools import adfuller, coint
    STATSMODELS_AVAILABLE = True
except ImportError:
    logger.warning("statsmodels not available - cointegration testing disabled")
    STATSMODELS_AVAILABLE = False

# NEW (lines 33-36):
from statsmodels.tsa.stattools import adfuller, coint

# statsmodels is a required dependency in pyproject.toml
STATSMODELS_AVAILABLE = True
```

**Lines 132-138**: Single-asset graceful handling
```python
# OLD (lines 132-134):
if len(price_columns) < 2:
    logger.error(f"Need at least 2 assets, found {len(price_columns)}")
    return self._hold_frame(data)

# NEW (lines 132-138):
# BUGFIX: Gracefully handle single-asset case
if len(price_columns) < 2:
    logger.warning(
        f"CopulaPairsTrading requires ≥2 assets for pairs, found {len(price_columns)}. "
        f"Falling back to single-asset HOLD signals."
    )
    return self._generate_single_asset_signals(data, price_columns)
```

**Lines 645-675**: New fallback method
```python
def _generate_single_asset_signals(
    self,
    data: pd.DataFrame,
    price_columns: list
) -> pd.DataFrame:
    """
    Generate signals for single-asset case (graceful degradation).

    Returns HOLD signals since pairs trading requires at least 2 assets.
    """
    signals_df = pd.DataFrame({
        'timestamp': data['timestamp'] if 'timestamp' in data.columns else data.index,
        'signal': [SignalType.HOLD.value] * len(data),
        'confidence': [0.0] * len(data),
        'metadata': [{}] * len(data)
    })

    if len(price_columns) == 1:
        logger.info(f"Generated single-asset HOLD signals for {price_columns[0]} (pairs trading N/A)")
    else:
        logger.warning("No price columns found, returning HOLD signals")

    return signals_df
```

---

## Pattern Summary

All fixes follow the same pattern from HRP and RiskParity:

1. **Detect single-asset case** early in `generate_signals()`
2. **Log descriptive warning** instead of error
3. **Call dedicated fallback method** instead of returning empty/None
4. **Return valid DataFrame** with appropriate format

### Portfolio Strategies (BL, HRP, RiskParity)
- Return weight columns with 100% allocation to single asset

### Pairs Trading Strategies (Copula)
- Return signal/confidence/metadata with all HOLD signals

---

## Quick Search Commands

Find these changes in your codebase:

```bash
# Black-Litterman changes
grep -n "_generate_single_asset_signals" src/crypto_trader/strategies/library/black_litterman.py

# CopulaPairsTrading changes
grep -n "_generate_single_asset_signals" src/crypto_trader/strategies/library/copula_pairs_trading.py

# All single-asset fallback methods
grep -rn "def _generate_single_asset_signals" src/crypto_trader/strategies/library/
```

---

## Verification Checklist

- [x] Black-Litterman lines 121-127 modified (single-asset detection)
- [x] Black-Litterman lines 192-197 modified (Ledoit-Wolf clarity)
- [x] Black-Litterman lines 283-313 added (fallback method)
- [x] CopulaPairsTrading lines 33-36 modified (direct import)
- [x] CopulaPairsTrading lines 132-138 modified (single-asset detection)
- [x] CopulaPairsTrading lines 645-675 added (fallback method)
- [x] All validation tests pass
- [x] Single-asset test passes
- [x] Edge cases handled

---

## Cross-Reference

For complete context, see:
- **BL_COPULA_BUGFIX_SUMMARY.md** - Detailed explanation of all changes
- **BUGFIX_APPLICATION_COMPLETE.md** - Validation results and impact analysis
- **test_single_asset_fallback.py** - Automated test suite

All changes are consistent with the pattern established in:
- `hierarchical_risk_parity.py` (lines 142-147, 441-444, 389-419)
- `risk_parity.py` (similar pattern)
