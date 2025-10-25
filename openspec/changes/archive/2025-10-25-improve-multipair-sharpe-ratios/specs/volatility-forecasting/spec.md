# Spec: Volatility Forecasting

**Capability**: NEW
**Purpose**: Provide forward-looking volatility forecasts for risk management and position sizing

## Overview

This capability enables portfolio strategies to use GARCH(1,1) volatility forecasting instead of backward-looking historical volatility. This improves position sizing accuracy and risk-adjusted returns by capturing volatility clustering common in cryptocurrency markets.

---

## ADDED Requirements

### Requirement: GARCH(1,1) Volatility Forecasting

System SHALL provide GARCH(1,1) volatility forecasts for cryptocurrency return series.

**Acceptance Criteria**:
- GARCH(1,1) model fitted using `arch` library
- One-step-ahead volatility forecast generated
- Forecast annualized for compatibility with Sharpe calculations
- Minimum 60 data points required for stable estimation
- Comprehensive validation prevents unreasonable forecasts

**Parameters**:
- `horizon`: Forecast horizon in periods (default: 1)
- `min_data_points`: Minimum data required (default: 60)
- `min_vol`: Minimum reasonable volatility (default: 0.05 = 5% annual)
- `max_vol`: Maximum reasonable volatility (default: 5.0 = 500% annual)

**Model Specification**:
```
r_t = μ + ε_t
ε_t = σ_t × z_t,  z_t ~ N(0,1)
σ_t² = ω + α × ε_{t-1}² + β × σ_{t-1}²
```

Where:
- `r_t`: Return at time t
- `σ_t`: Conditional volatility at time t
- `ω, α, β`: GARCH parameters (estimated)
- `α + β < 1` for stationarity

#### Scenario: Forecast volatility with sufficient data

**Given** a return series with 90 daily observations

**And** returns have volatility clustering pattern

**And** GARCH(1,1) model parameters can be estimated

**When** forecasting next-period volatility

**Then** model SHALL fit successfully with α + β < 1

**And** one-step forecast SHALL be generated

**And** forecast SHALL be annualized: vol_annual = vol_daily × √252

**And** forecast SHALL pass validation (0.05 ≤ vol ≤ 5.0)

**And** forecast SHALL be returned as float

#### Scenario: Handle insufficient data gracefully

**Given** a return series with only 40 observations

**And** minimum required data is 60 points

**When** attempting GARCH forecasting

**Then** system SHALL log warning about insufficient data

**And** system SHALL fall back to sample volatility

**And** sample volatility SHALL be: returns.std() × √252

**And** no exception SHALL be raised

#### Scenario: Validate and reject unreasonable forecasts

**Given** a return series with extreme values

**And** GARCH model fits but produces forecast of 8.0 (800% annual vol)

**When** validating forecast

**Then** forecast SHALL be rejected (8.0 > 5.0 max)

**And** warning SHALL be logged: "GARCH forecast out of bounds: 8.0, using sample vol"

**And** system SHALL fall back to sample volatility

**And** strategy SHALL continue without crashing

#### Scenario: Handle GARCH fitting failures

**Given** a return series with numerical issues (NaN, inf)

**And** GARCH model fitting raises exception

**When** forecasting volatility

**Then** exception SHALL be caught and logged

**And** system SHALL fall back to sample volatility

**And** warning SHALL indicate failure reason

**And** fallback SHALL allow strategy to continue

---

### Requirement: Volatility Forecast Caching

System SHALL cache GARCH forecasts to avoid redundant computation.

**Acceptance Criteria**:
- Forecasts cached with key (asset, timestamp, horizon)
- Cache invalidated when new data arrives
- Maximum cache size enforced (default: 100 entries)
- Cache hit rate tracked and logged

**Parameters**:
- `cache_enabled`: Enable forecast caching (default: True)
- `cache_max_size`: Maximum cache entries (default: 100)

#### Scenario: Cache hit avoids recomputation

**Given** GARCH forecast computed for BTC/USDT at timestamp T

**And** forecast cached with key (BTC/USDT, T, horizon=1)

**When** requesting same forecast within same bar

**Then** cached forecast SHALL be returned

**And** GARCH model SHALL NOT be refitted

**And** cache hit SHALL be logged

**And** computation time SHALL be < 1ms (vs ~50ms for fitting)

#### Scenario: Cache miss triggers computation

**Given** no cached forecast exists for ETH/USDT at timestamp T

**When** requesting forecast for ETH/USDT

**Then** GARCH model SHALL be fitted

**And** forecast SHALL be computed and cached

**And** cache miss SHALL be logged

**And** subsequent requests SHALL hit cache

#### Scenario: Cache invalidation on new data

**Given** cached forecast for BTC/USDT at timestamp T

**When** new data arrives at timestamp T+1

**Then** cache entry SHALL be invalidated

**And** next forecast request SHALL trigger recomputation

**And** new forecast SHALL be cached with updated timestamp

---

### Requirement: Integration with Position Sizing

Volatility forecasts SHALL integrate with Kelly position sizing.

**Acceptance Criteria**:
- Position sizing accepts volatility forecaster callable
- GARCH forecast used when `use_garch_vol=True`
- Sample volatility used when `use_garch_vol=False`
- Forecast failures gracefully degrade to sample volatility

#### Scenario: Kelly sizing with GARCH volatility

**Given** a portfolio strategy with `use_garch_vol=True`

**And** GARCH forecaster returns vol = 0.35 (35% annual)

**And** expected return = 0.13 (13% annual)

**And** win rate = 0.55

**When** calculating Kelly position size

**Then** volatility input SHALL be 0.35 (from GARCH)

**And** Kelly size SHALL be 0.13 / (0.35²) ≈ 1.06

**And** fractional Kelly (25%) SHALL be ~0.265

**And** position size SHALL be capped at 15% maximum

#### Scenario: Fallback to sample volatility

**Given** a portfolio strategy with `use_garch_vol=True`

**And** GARCH forecasting fails (insufficient data)

**When** calculating position size

**Then** sample volatility SHALL be used as fallback

**And** position size calculation SHALL complete successfully

**And** warning SHALL be logged about fallback

---

### Requirement: Regime-Adaptive Volatility

System SHALL support regime-based volatility forecasting (OPTIONAL enhancement).

**Acceptance Criteria**:
- Volatility regime detected (low/medium/high)
- GARCH parameters adapted based on regime
- Regime detection uses rolling window statistics

**Parameters**:
- `regime_detection_window`: Window for regime detection (default: 20)
- `use_regime_adaptation`: Enable regime adaptation (default: False)

**Status**: OPTIONAL (not required for initial implementation)

#### Scenario: Detect high volatility regime

**Given** recent 20-day volatility = 0.60 (60% annual)

**And** long-term 90-day volatility = 0.35 (35% annual)

**And** volatility ratio = 0.60 / 0.35 = 1.71

**When** detecting volatility regime

**Then** regime SHALL be classified as HIGH (ratio > 1.5)

**And** GARCH model SHALL use shorter lookback

**And** forecast SHALL adapt more quickly to changes

---

## Integration Points

**Consumers**:
- `portfolio-optimization`: Kelly position sizing
- `risk-management`: ATR calculation for stop losses

**Dependencies**:
- `arch` library (GARCH implementation)
- `numpy` (numerical operations)
- `pandas` (time series handling)

**Module Location**:
- `src/crypto_trader/risk/volatility_forecasting.py`

**Tests**:
- `tests/crypto_trader/risk/test_volatility_forecasting.py`

---

## Performance Considerations

**Computational Cost**:
- GARCH fitting: ~50ms per asset per rebalance
- Forecast generation: <5ms
- Cache lookup: <1ms

**Optimization Strategies**:
- Cache forecasts within rebalancing period
- Parallelize forecasting across multiple assets
- Use pre-fitted models when possible

**Expected Overhead**:
- 4 assets × 50ms = 200ms per rebalance
- Weekly rebalancing = negligible daily overhead

---

## Validation Strategy

**Unit Tests**:
- Test GARCH fitting with synthetic data
- Test validation rejects extreme forecasts
- Test fallback to sample volatility
- Test caching behavior

**Integration Tests**:
- Test with real crypto return data
- Verify forecasts improve out-of-sample
- Compare GARCH vs sample volatility performance

**Validation Function**:
```python
if __name__ == "__main__":
    # Test with BTC/USDT historical data
    btc_returns = fetch_btc_returns(days=180)

    # GARCH forecast
    garch_vol = forecast_volatility_garch(btc_returns)

    # Sample volatility
    sample_vol = btc_returns.std() * np.sqrt(252)

    # Compare
    print(f"GARCH: {garch_vol:.4f}")
    print(f"Sample: {sample_vol:.4f}")

    # Validate in range
    assert 0.05 <= garch_vol <= 5.0
    print("✅ GARCH forecast passes validation")
```

---

## References

### Research Papers
- Engle, R. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation"
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
- Hansen, P. & Lunde, A. (2005). "A forecast comparison of volatility models"

### Documentation
- arch library: https://arch.readthedocs.io/
- GARCH models: https://arch.readthedocs.io/en/latest/univariate/introduction.html

---

**Status**: PROPOSED
**Impact**: MODERATE (new module, integrated with existing risk management)
**Breaking Changes**: None (new capability, no modifications to existing APIs)
