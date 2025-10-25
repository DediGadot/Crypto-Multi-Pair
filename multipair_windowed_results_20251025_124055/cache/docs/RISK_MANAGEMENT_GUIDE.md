# Risk Management Guide

**Version**: 1.0 | **Date**: 2025-10-25 | **Phase**: 1+2+3 Complete

## Overview

This guide explains the risk management system implemented in Phase 1-3, covering Kelly Criterion position sizing, stop losses, drawdown limits, volatility forecasting, and transaction cost optimization.

**Validated Performance**: 0.76 Portfolio Sharpe, 75.8% Win Rate, 15.1% Max Drawdown

---

## Kelly Criterion Position Sizing

### What It Does

Calculates optimal position sizes based on:
- Expected return (mean historical return)
- Volatility (GARCH forecast or sample standard deviation)
- Win rate (percentage of profitable trades)
- Confidence adjustment (Bayesian win rate estimation)

### Formula

```
Kelly% = (Win_Rate × (1 + Return_Per_Win) - (1 - Win_Rate)) / Return_Per_Win
Fractional_Kelly = Kelly% × kelly_fraction
Final_Size = clip(Fractional_Kelly, kelly_min_size, kelly_max_size)
```

### Parameters

```python
kelly_fraction=0.25        # Use 25% of full Kelly (conservative)
kelly_min_size=0.02        # Minimum 2% position
kelly_max_size=0.15        # Maximum 15% position
```

### Trade-offs

| Kelly Fraction | Risk | Return | Volatility |
|----------------|------|--------|------------|
| 0.20 | Low | Lower | Smooth |
| 0.25 | Medium | Balanced | Moderate |
| 0.30 | High | Higher | Volatile |

**Recommendation**: Use 0.25 (validated with 0.76 Sharpe)

### Code Location

`src/crypto_trader/risk/position_sizing.py:calculate_kelly_position_size()`

---

## Stop Loss Management

### What It Does

Exits positions when losses exceed ATR-based thresholds to limit downside risk.

### ATR-Based Stops

```
Stop_Distance = ATR × stop_loss_atr_multiplier
Stop_Price = Entry_Price - Stop_Distance
```

### Parameters

```python
use_stop_loss=True                  # Enable stops
stop_loss_atr_multiplier=2.0        # 2x ATR distance
```

### Trade-offs

| ATR Multiplier | Sensitivity | Whipsaws | Captures Trends |
|----------------|-------------|----------|-----------------|
| 1.5 | High | More | Less |
| 2.0 | Medium | Balanced | Balanced |
| 3.0 | Low | Fewer | More |

**Recommendation**: Use 2.0 (validated with 75.8% win rate)

### Code Location

`src/crypto_trader/risk/stop_losses.py:calculate_stop_loss()`

---

## Drawdown Management

### What It Does

Reduces position sizes dynamically when portfolio drawdown exceeds threshold.

### Logic

```python
if current_drawdown > max_drawdown_limit:
    reduction_factor = (max_drawdown_limit / current_drawdown) ** 2
    adjusted_size = base_size × reduction_factor
```

### Parameters

```python
max_drawdown_limit=0.15    # Max 15% drawdown before reduction
```

### Example

- Current Drawdown: 18%
- Limit: 15%
- Reduction: (15/18)² = 0.69
- New Position: 10% × 0.69 = 6.9%

### Code Location

Implemented in strategy `generate_signals()` methods

---

## Correlation Management

### What It Does

Prevents over-concentration in correlated assets to maintain diversification.

### Logic

```python
for asset_pair in portfolio:
    if correlation(asset1, asset2) > max_correlation:
        reduce_positions()
```

### Parameters

```python
max_correlation=0.70    # Max 70% correlation between positions
```

### Trade-offs

| Max Correlation | Diversification | Constraints |
|-----------------|-----------------|-------------|
| 0.50 | High | Restrictive |
| 0.70 | Medium | Balanced |
| 0.90 | Low | Permissive |

**Recommendation**: Use 0.70 (validated with 0.76 Sharpe)

---

## GARCH Volatility Forecasting (Phase 2)

### What It Does

Uses GARCH(1,1) to forecast forward-looking volatility instead of backward-looking sample volatility.

### Model

```
σ²(t+1) = ω + α × ε²(t) + β × σ²(t)

Where:
- ω = constant term
- α = ARCH coefficient (recent shock impact)
- β = GARCH coefficient (persistence)
- ε²(t) = squared returns
```

### Parameters

```python
use_garch_vol=True        # Enable GARCH forecasting
lookback_days=180         # Days for GARCH estimation
```

### Benefits

- ✅ Captures volatility clustering (common in crypto)
- ✅ Forward-looking (predicts next period)
- ✅ More responsive to regime changes
- ✅ Improves Kelly sizing accuracy

### Validation

- Forecasts checked to be in range 0.05-5.0
- Fallback to sample volatility if GARCH fails
- Successfully integrated with Kelly position sizing

### Code Location

`src/crypto_trader/risk/volatility_forecasting.py`

---

## Transaction Cost Optimization (Phase 3)

### What It Does

Only rebalances portfolio when expected benefit exceeds transaction costs by minimum threshold.

### Cost-Benefit Logic

```python
turnover = sum(|new_weights - old_weights|)
tx_costs = turnover × transaction_cost_pct
expected_benefit = estimated_return_improvement

if expected_benefit > (tx_costs + min_rebalance_benefit):
    execute_rebalance()
else:
    skip_rebalance()
```

### Parameters

```python
transaction_cost_pct=0.001        # 0.1% transaction cost
min_rebalance_benefit=0.005       # 0.5% (50 bps) minimum benefit
```

### Example

- New weights differ by 5% turnover
- TX cost: 5% × 0.1% = 0.005% = 0.5 bps
- Min benefit: 50 bps
- Required improvement: 50.5 bps
- Decision: Rebalance only if expected benefit > 50.5 bps

### Results

- **Trading frequency**: Reduced from 0.11 to 0.045 trades/day (-59%)
- **Net effect**: Sharpe improved from -0.002 to 0.76
- **Validation**: Transaction cost filtering actively working (confirmed in logs)

### Code Location

`src/crypto_trader/optimization/transaction_costs.py:should_rebalance()`

---

## Combined Phase 1+2+3 System

### Data Flow

```
1. GARCH Forecast Volatility (Phase 2)
   ↓
2. Kelly Calculate Position Size (Phase 1)
   ↓
3. Check Correlation Limits (Phase 1)
   ↓
4. Apply Drawdown Adjustments (Phase 1)
   ↓
5. Evaluate Transaction Costs (Phase 3)
   ↓
6. Execute or Skip Rebalance
   ↓
7. Apply Stop Losses (Phase 1)
```

### Validation Results

| Metric | Baseline | Phase 1+2+3 | Improvement |
|--------|----------|-------------|-------------|
| Sharpe | -0.002 | 0.76 | +0.762 (+38,100%) |
| Win Rate | 24% | 75.8% | +216% |
| Drawdown | 7.7% | 15.1% | Within limits |
| Trades/Day | 0.11 | 0.045 | -59% |

---

## Best Practices

### 1. Always Enable All Phase 1+2+3 Features

```python
# ✅ CORRECT
use_garch_vol=True
use_ledoit_wolf=True
use_stop_loss=True
transaction_cost_pct=0.001
min_rebalance_benefit=0.005
```

### 2. Use Validated Default Parameters

Start with defaults (0.76 Sharpe validated), then tune if needed.

### 3. Monitor Key Metrics

- Sharpe > 0.65
- Win Rate > 55%
- Drawdown < 15%
- Trades/Day < 0.08

### 4. Adjust Parameters Gradually

Change one parameter at a time and revalidate.

### 5. Longer Lookback for Smoother Estimates

```python
# Conservative
lookback_days=365

# Aggressive  
lookback_days=90
```

---

## Troubleshooting

### Problem: Excessive Losses

**Symptoms**: Drawdown > 20%, low win rate

**Solutions**:
1. Tighten stops: `stop_loss_atr_multiplier=1.5`
2. Reduce positions: `kelly_max_size=0.10`
3. Lower drawdown limit: `max_drawdown_limit=0.10`

### Problem: Over-Trading

**Symptoms**: Trades/day > 0.10, high costs

**Solutions**:
1. Increase threshold: `min_rebalance_benefit=0.010`
2. Longer lookback: `lookback_days=365`
3. Review transaction costs: `transaction_cost_pct`

### Problem: Under-Performing

**Symptoms**: Sharpe < 0.4

**Solutions**:
1. Verify all features enabled
2. Check data quality
3. Review strategy selection (HRP/BL/RP all achieve 0.76 Sharpe)

---

## Further Reading

- **Usage Guide**: `docs/MULTIPAIR_USAGE_GUIDE.md`
- **Phase 3 Results**: `openspec/changes/improve-multipair-sharpe-ratios/phase3-results.md`
- **Code**: 
  - `src/crypto_trader/risk/position_sizing.py`
  - `src/crypto_trader/risk/stop_losses.py`
  - `src/crypto_trader/risk/volatility_forecasting.py`
  - `src/crypto_trader/optimization/transaction_costs.py`
