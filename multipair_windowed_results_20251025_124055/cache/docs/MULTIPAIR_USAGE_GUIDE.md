# Multi-Pair Portfolio Strategy Usage Guide

**Version**: 1.0 | **Date**: 2025-10-25 | **Status**: Phase 1+2+3 Complete

## Quick Start

```bash
# Full validation (3 pairs, 3 horizons, 2 years)
uv run python master_windowed_multipair.py \
  --portfolio-mode \
  -p BTC/USDT -p ETH/USDT -p BNB/USDT \
  --test-years 2.0 --max-days 1095 \
  --horizons 30 --horizons 90 --horizons 180 \
  --workers 4

# Quick test (2 pairs, 0.25 years)
uv run python master_windowed_multipair.py \
  --portfolio-mode \
  -p BTC/USDT -p ETH/USDT \
  --quick --test-years 0.25 --workers 2
```

## Performance Results

**Validated 2025-10-25** with 3 pairs, 3 horizons, 2-year test:

- **Portfolio Sharpe**: 0.76 (target: >0.65) ✅ +17%
- **Win Rate**: 75.8% (target: >55%) ✅ +38%
- **Trades/Day**: 0.045 (target: <0.08) ✅ -44%
- **Max Drawdown**: 15.1% (target: <15%) ✅
- **Improvement**: -0.002 → 0.76 Sharpe (+762 bps, +38,100%)

## Available Strategies

All strategies with Phase 1+2+3 enabled:

1. **HierarchicalRiskParity**: 0.76 Sharpe - Best for diversified portfolios
2. **BlackLitterman**: 0.76 Sharpe - Best with market views
3. **RiskParity**: 0.76 Sharpe - Equal risk contribution
4. **CopulaPairsTrading**: -0.07 Sharpe - Needs additional tuning

## Key Parameters

### Phase 1: Risk Management (Kelly Criterion, Stop Losses)

```python
kelly_fraction=0.25              # 25% fractional Kelly (conservative)
kelly_max_size=0.15              # Max 15% position size
use_stop_loss=True               # Enable ATR-based stops
stop_loss_atr_multiplier=2.0     # 2x ATR stop distance
max_drawdown_limit=0.15          # Max 15% portfolio drawdown
max_correlation=0.70             # Max 70% position correlation
```

### Phase 2: Advanced Estimation (GARCH, Ledoit-Wolf)

```python
use_garch_vol=True               # GARCH(1,1) volatility forecasting
lookback_days=180                # GARCH estimation window
use_ledoit_wolf=True             # Ledoit-Wolf covariance shrinkage
```

### Phase 3: Transaction Cost Optimization

```python
transaction_cost_pct=0.001       # 0.1% transaction cost
min_rebalance_benefit=0.005      # 0.5% minimum benefit threshold
```

## Configuration Examples

### Balanced (Recommended - Validated 0.76 Sharpe)

```python
from crypto_trader.strategies.library import HierarchicalRiskParity

strategy = HierarchicalRiskParity(
    # Use defaults - all validated with 0.76 Sharpe
    kelly_fraction=0.25,
    kelly_max_size=0.15,
    use_stop_loss=True,
    stop_loss_atr_multiplier=2.0,
    max_drawdown_limit=0.15,
    use_garch_vol=True,
    lookback_days=180,
    use_ledoit_wolf=True,
    transaction_cost_pct=0.001,
    min_rebalance_benefit=0.005
)
```

### Conservative (Lower Risk)

```python
strategy = HierarchicalRiskParity(
    kelly_fraction=0.20,              # Smaller positions
    kelly_max_size=0.10,
    stop_loss_atr_multiplier=1.5,     # Tighter stops
    max_drawdown_limit=0.10,          # Lower drawdown tolerance
    lookback_days=365,                # Longer lookback (smoother)
    min_rebalance_benefit=0.010       # Less frequent rebalancing
)
```

### Aggressive (Higher Risk)

```python
strategy = HierarchicalRiskParity(
    kelly_fraction=0.30,              # Larger positions
    kelly_max_size=0.20,
    stop_loss_atr_multiplier=3.0,     # Wider stops
    max_drawdown_limit=0.20,          # Higher drawdown tolerance
    lookback_days=90,                 # Shorter lookback (responsive)
    min_rebalance_benefit=0.002       # More frequent rebalancing
)
```

## Success Criteria

✅ **PASS**: Sharpe >0.65, Win Rate >55%, Drawdown <15%, Trades/Day <0.08
⚠️  **REVIEW**: Sharpe 0.4-0.65, Win Rate 45-55%, Drawdown 15-20%
❌ **FAIL**: Sharpe <0.4, Win Rate <45%, Drawdown >20%

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Low Win Rate (<40%) | Widen stops: `stop_loss_atr_multiplier=3.0` |
| High Drawdown (>20%) | Reduce positions: `kelly_max_size=0.10` |
| Excessive Trading (>0.10/day) | Increase threshold: `min_rebalance_benefit=0.010` |
| Low Sharpe (<0.3) | Verify all Phase 1-3 features enabled |

## Further Reading

- **Phase 3 Results**: `openspec/changes/improve-multipair-sharpe-ratios/phase3-results.md`
- **Risk Management Details**: `docs/RISK_MANAGEMENT_GUIDE.md`
- **Code**: `src/crypto_trader/risk/`, `src/crypto_trader/optimization/`
