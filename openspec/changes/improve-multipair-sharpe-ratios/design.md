# Design: Improve Multi-Pair Strategy Sharpe Ratios

## Architecture Overview

This change introduces four complementary subsystems to improve multi-pair portfolio strategy performance:

```
┌─────────────────────────────────────────────────────────────┐
│                   Portfolio Strategy                         │
│  (HRP, Risk Parity, Black-Litterman, Copula Pairs)         │
└──────┬──────────────────────────────────────────────────────┘
       │
       │ Uses
       │
       ├──────────────────────────────────────────────────────┐
       │                                                       │
       ▼                                                       ▼
┌──────────────────────┐                        ┌──────────────────────┐
│  Position Sizing     │                        │  Covariance          │
│  (Kelly Criterion)   │                        │  Estimation          │
│                      │                        │  (Ledoit-Wolf)       │
│  • Fractional Kelly  │                        │                      │
│  • Position limits   │                        │  • Shrinkage         │
│  • Confidence scale  │                        │  • Validation        │
└──────┬───────────────┘                        └──────┬───────────────┘
       │                                               │
       │ Requires                                      │ Requires
       │                                               │
       ▼                                               ▼
┌──────────────────────┐                        ┌──────────────────────┐
│  Volatility          │                        │  Transaction Cost    │
│  Forecasting         │                        │  Optimization        │
│  (GARCH)             │                        │                      │
│                      │                        │  • Rebalance thresh  │
│  • GARCH(1,1)        │◄───────────────────────│  • Cost penalty      │
│  • Fallback to sample│                        │  • Turnover tracking │
│  • Validation        │                        │                      │
└──────────────────────┘                        └──────────────────────┘
       │
       │ Informs
       │
       ▼
┌──────────────────────┐
│  Stop Losses         │
│  (Trailing)          │
│                      │
│  • 8% trail          │
│  • ATR-adjusted      │
│  • Profit locking    │
└──────────────────────┘
```

## Design Decisions

### 1. Risk Management Framework

#### Position Sizing: Kelly Criterion

**Decision**: Use fractional Kelly (25%) with hard limits

**Rationale**:
- Full Kelly is too aggressive for crypto volatility
- 25% fraction provides good risk-reward balance
- Hard limits (2%-15%) prevent extreme positions
- Signal confidence scaling adapts to certainty

**Implementation**:
```python
# src/crypto_trader/risk/position_sizing.py

def calculate_kelly_position_size(
    expected_return: float,
    volatility: float,
    win_rate: float,
    signal_confidence: float = 1.0,
    kelly_fraction: float = 0.25
) -> float:
    """
    Calculate position size using fractional Kelly Criterion.

    Args:
        expected_return: Expected annual return (e.g., 0.13 = 13%)
        volatility: Annual volatility (e.g., 0.40 = 40%)
        win_rate: Historical win rate (0-1)
        signal_confidence: Confidence in signal (0-1)
        kelly_fraction: Fraction of full Kelly (default 0.25)

    Returns:
        Position size as fraction of capital (0-1)
    """
    if volatility == 0 or win_rate == 0:
        return 0.02  # Minimum position

    # Simplified Kelly for continuous returns
    # f* = (expected_return) / (volatility^2)
    kelly_size = expected_return / (volatility ** 2)

    # Apply fraction and confidence
    position_size = kelly_size * kelly_fraction * signal_confidence

    # Hard limits
    MIN_POSITION = 0.02  # 2%
    MAX_POSITION = 0.15  # 15%

    return np.clip(position_size, MIN_POSITION, MAX_POSITION)
```

**Trade-offs**:
- ✅ Mathematically optimal leverage
- ✅ Adapts to signal confidence
- ⚠️ Requires accurate expected return estimation
- ⚠️ May be conservative for high-confidence signals

#### Stop Losses: Trailing 8%

**Decision**: Implement 8% trailing stop with ATR adjustment

**Rationale**:
- 8% matches typical crypto intraday volatility
- Trailing allows profits to run
- ATR adjustment prevents premature stops in volatile periods
- Lock-in prevents giving back profits

**Implementation**:
```python
# src/crypto_trader/risk/stop_losses.py

def calculate_stop_loss_level(
    entry_price: float,
    current_price: float,
    highest_price_since_entry: float,
    atr: float,
    stop_pct: float = 0.08,
    atr_multiplier: float = 2.5
) -> float:
    """
    Calculate trailing stop loss level with ATR adjustment.

    Args:
        entry_price: Entry price
        current_price: Current price
        highest_price_since_entry: Peak price since entry
        atr: Average True Range (volatility measure)
        stop_pct: Stop percentage (default 8%)
        atr_multiplier: ATR multiplier for volatility adjustment

    Returns:
        Stop loss price level
    """
    # Three stop types
    fixed_stop = entry_price * (1 - stop_pct)
    trailing_stop = highest_price_since_entry * (1 - stop_pct)
    atr_stop = current_price - (atr_multiplier * atr)

    # Use the tightest stop (most protective)
    return max(fixed_stop, trailing_stop, atr_stop)
```

**Trade-offs**:
- ✅ Limits downside risk
- ✅ Locks in profits
- ✅ Adapts to volatility
- ⚠️ May exit too early in mean-reversion scenarios
- ⚠️ Requires tracking per-position state

#### Portfolio Limits

**Decision**: Implement correlation and drawdown limits

**Rationale**:
- 0.70 correlation limit ensures diversification
- 15% drawdown limit prevents catastrophic losses
- Dynamic position reduction after 10% drawdown

**Integration Point**: Backtesting engine validates limits before each trade

### 2. Covariance Estimation: Ledoit-Wolf Shrinkage

**Decision**: Replace all sample covariance with Ledoit-Wolf shrinkage

**Rationale**:
- Sample covariance is unstable with limited crypto data
- Ledoit-Wolf shrinks toward structured estimator
- Improves out-of-sample performance
- Already available in PyPortfolioOpt

**Implementation**:
```python
# Modification to portfolio strategies

from pypfopt import risk_models

# Instead of:
# cov_matrix = returns.cov()

# Use:
# Convert returns to prices for PyPortfolioOpt
prices_df = pd.DataFrame({
    col: (1 + returns[col]).cumprod()
    for col in returns.columns
})

# Apply Ledoit-Wolf shrinkage
cov_matrix = risk_models.CovarianceShrinkage(
    prices_df
).ledoit_wolf()
```

**Trade-offs**:
- ✅ More stable covariance estimates
- ✅ Better out-of-sample performance
- ✅ Minimal code change (drop-in replacement)
- ⚠️ Slight computational overhead
- ⚠️ Requires price series (not returns)

**Affected Strategies**:
- HierarchicalRiskParity
- RiskParity
- BlackLitterman
- CopulaPairsTrading (for correlation estimation)

### 3. Volatility Forecasting: GARCH(1,1)

**Decision**: Implement GARCH(1,1) with comprehensive validation

**Rationale**:
- GARCH captures volatility clustering (common in crypto)
- Forward-looking (unlike historical volatility)
- GARCH(1,1) is sufficient for most financial time series
- Validation prevents unstable forecasts

**Implementation**:
```python
# src/crypto_trader/risk/volatility_forecasting.py

from arch import arch_model

def forecast_volatility_garch(
    returns: pd.Series,
    horizon: int = 1,
    min_data_points: int = 60
) -> float:
    """
    Forecast volatility using GARCH(1,1) with validation.

    Args:
        returns: Historical return series
        horizon: Forecast horizon (1 = next period)
        min_data_points: Minimum required data points

    Returns:
        Annualized volatility forecast

    Raises:
        ValueError: If returns are invalid
    """
    # Validation
    if len(returns) < min_data_points:
        logger.warning(
            f"Insufficient data for GARCH: {len(returns)} < {min_data_points}"
        )
        return returns.std() * np.sqrt(252)  # Fallback

    try:
        # Fit GARCH(1,1)
        model = arch_model(
            returns * 100,  # Scale to percentage
            vol='GARCH',
            p=1,  # GARCH order
            q=1,  # ARCH order
            dist='normal',
            rescale=True
        )

        results = model.fit(disp='off', show_warning=False)

        # Forecast
        forecast = results.forecast(horizon=horizon)
        variance_forecast = forecast.variance.iloc[-1, 0]

        # Convert to annualized volatility
        vol_forecast = np.sqrt(variance_forecast) / 100 * np.sqrt(252)

        # Validation: check for reasonable values
        MIN_VOL = 0.05  # 5% annual
        MAX_VOL = 5.00  # 500% annual

        if not (MIN_VOL <= vol_forecast <= MAX_VOL):
            logger.warning(
                f"GARCH forecast out of bounds: {vol_forecast:.4f}, "
                f"using sample vol"
            )
            return returns.std() * np.sqrt(252)

        return vol_forecast

    except Exception as e:
        logger.warning(f"GARCH forecasting failed: {e}, using sample vol")
        return returns.std() * np.sqrt(252)
```

**Trade-offs**:
- ✅ Forward-looking volatility
- ✅ Captures volatility clustering
- ✅ Comprehensive validation
- ⚠️ Requires 60+ data points
- ⚠️ Can be unstable with extreme data
- ⚠️ Additional computational cost

**Integration**: Used by position sizing and stop loss calculations

### 4. Transaction Cost Optimization

**Decision**: Add transaction cost penalty to optimization and rebalancing thresholds

**Rationale**:
- Crypto exchanges charge ~10 bps per trade
- Excessive rebalancing destroys returns
- PyPortfolioOpt supports transaction cost objectives
- Threshold logic prevents tiny rebalances

**Implementation**:

```python
# src/crypto_trader/optimization/transaction_costs.py

def should_rebalance(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    transaction_cost_pct: float = 0.001,  # 10 bps
    min_benefit_pct: float = 0.005  # 50 bps
) -> tuple[bool, float]:
    """
    Determine if rebalancing is worthwhile.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        transaction_cost_pct: Cost per trade (default 0.1%)
        min_benefit_pct: Minimum benefit to justify rebalance

    Returns:
        (should_rebalance, turnover)
    """
    # Calculate turnover
    turnover = sum(
        abs(target_weights.get(asset, 0) - current_weights.get(asset, 0))
        for asset in set(current_weights.keys()) | set(target_weights.keys())
    )

    # Estimate transaction cost
    tx_cost = turnover * transaction_cost_pct

    # Only rebalance if benefit exceeds threshold
    should_rebalance = tx_cost < min_benefit_pct

    return should_rebalance, turnover
```

**PyPortfolioOpt Integration**:
```python
# In portfolio strategy optimization

from pypfopt.objective_functions import transaction_cost
from pypfopt import EfficientFrontier

# Create efficient frontier
ef = EfficientFrontier(expected_returns, cov_matrix)

# Add transaction cost penalty if we have previous weights
if self.last_weights is not None:
    ef.add_objective(
        transaction_cost,
        w_prev=self.last_weights,
        k=0.001  # 10 basis points
    )

# Optimize
weights = ef.max_sharpe()

# Check if rebalancing is worthwhile
should_rebalance, turnover = should_rebalance(
    current_weights=self.last_weights,
    target_weights=weights,
    transaction_cost_pct=0.001,
    min_benefit_pct=0.005
)

if not should_rebalance:
    return self.last_weights  # Skip rebalance
```

**Trade-offs**:
- ✅ Reduces excessive trading
- ✅ Improves net returns
- ✅ Simple threshold logic
- ⚠️ May miss profitable rebalances
- ⚠️ Requires tracking previous weights

## Integration with Existing System

### Backtesting Engine Integration

**Current Flow**:
```
Strategy.generate_signals() → Backtest Engine → Metrics Calculator
```

**New Flow**:
```
Strategy.generate_signals()
  ├─> Position Sizing (Kelly)
  ├─> Volatility Forecasting (GARCH)
  ├─> Covariance Estimation (Ledoit-Wolf)
  └─> Transaction Cost Check
      │
      ▼
  Backtest Engine
      ├─> Risk Limit Validation (correlation, drawdown)
      ├─> Stop Loss Check (per position)
      └─> Trade Execution
          │
          ▼
  Metrics Calculator
      └─> Enhanced Metrics (risk-adjusted)
```

**Modified Files**:
- `src/crypto_trader/backtesting/engine.py`: Add risk limit checks
- `src/crypto_trader/strategies/base.py`: Add risk management hooks
- `src/crypto_trader/analysis/metrics.py`: Add new risk metrics

### Strategy Parameter Schema

**New Parameters** (added to all portfolio strategies):

```python
{
    # Position Sizing
    'kelly_fraction': 0.25,
    'min_position_pct': 0.02,
    'max_position_pct': 0.15,

    # Stop Losses
    'trailing_stop_pct': 0.08,
    'atr_multiplier': 2.5,

    # Portfolio Limits
    'max_correlation': 0.70,
    'max_drawdown_pct': 0.15,
    'drawdown_reduction_factor': 0.5,

    # Volatility Forecasting
    'use_garch_vol': True,
    'garch_min_data': 60,

    # Transaction Costs
    'transaction_cost_pct': 0.001,
    'min_rebalance_benefit': 0.005,
    'use_ledoit_wolf': True
}
```

**Backward Compatibility**: All parameters have sensible defaults

## Testing Strategy

### Unit Tests

**Position Sizing** (`tests/crypto_trader/risk/test_position_sizing.py`):
```python
def test_kelly_position_size_basic():
    # Test basic Kelly calculation
    size = calculate_kelly_position_size(
        expected_return=0.13,
        volatility=0.40,
        win_rate=0.55,
        signal_confidence=1.0
    )
    assert 0.02 <= size <= 0.15

def test_kelly_position_size_limits():
    # Test hard limits are enforced
    size = calculate_kelly_position_size(
        expected_return=10.0,  # Unrealistic
        volatility=0.10,
        win_rate=0.99
    )
    assert size == 0.15  # Capped at max

def test_kelly_position_size_confidence_scaling():
    # Test confidence scaling
    size_high = calculate_kelly_position_size(
        expected_return=0.13,
        volatility=0.40,
        win_rate=0.55,
        signal_confidence=1.0
    )
    size_low = calculate_kelly_position_size(
        expected_return=0.13,
        volatility=0.40,
        win_rate=0.55,
        signal_confidence=0.5
    )
    assert size_low < size_high
```

**Stop Losses** (`tests/crypto_trader/risk/test_stop_losses.py`):
```python
def test_trailing_stop_follows_price():
    # Test trailing stop follows price up
    stop1 = calculate_stop_loss_level(
        entry_price=100,
        current_price=110,
        highest_price_since_entry=110,
        atr=5.0
    )
    stop2 = calculate_stop_loss_level(
        entry_price=100,
        current_price=120,
        highest_price_since_entry=120,
        atr=5.0
    )
    assert stop2 > stop1

def test_stop_loss_locks_in_profit():
    # Test stop never goes below entry after profit
    stop = calculate_stop_loss_level(
        entry_price=100,
        current_price=150,
        highest_price_since_entry=150,
        atr=5.0,
        stop_pct=0.08
    )
    assert stop >= 100  # At least entry price
```

**GARCH Forecasting** (`tests/crypto_trader/risk/test_volatility_forecasting.py`):
```python
def test_garch_forecast_validation():
    # Test validation rejects unreasonable forecasts
    returns = pd.Series(np.random.normal(0, 0.02, 100))
    vol = forecast_volatility_garch(returns)
    assert 0.05 <= vol <= 5.0

def test_garch_insufficient_data_fallback():
    # Test fallback to sample vol with insufficient data
    returns = pd.Series(np.random.normal(0, 0.02, 30))
    vol = forecast_volatility_garch(returns, min_data_points=60)
    expected_vol = returns.std() * np.sqrt(252)
    assert abs(vol - expected_vol) < 0.01
```

### Integration Tests

**Full Strategy Backtest** (`tests/integration/test_improved_strategies.py`):
```python
def test_hrp_with_risk_management():
    # Test HRP with all improvements
    strategy = HierarchicalRiskParityStrategy()
    strategy.initialize({
        'asset_symbols': ['BTC/USDT', 'ETH/USDT'],
        'kelly_fraction': 0.25,
        'use_garch_vol': True,
        'use_ledoit_wolf': True,
        'trailing_stop_pct': 0.08
    })

    # Run backtest
    results = run_backtest(strategy, data, initial_capital=10000)

    # Validate improvements
    assert results.sharpe_ratio > 0.5
    assert results.max_drawdown < 0.15
    assert results.profit_factor > 1.2
```

**Multi-Pair Windowed Analysis** (`tests/integration/test_multipair_windowed.py`):
```python
def test_multipair_windowed_with_improvements():
    # Test full windowed analysis
    results = run_multipair_windowed_analysis(
        pairs=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        strategies=['HierarchicalRiskParity', 'RiskParity'],
        test_years=2.0
    )

    # Validate all strategies improve
    for strategy_name, metrics in results.items():
        assert metrics['avg_sharpe'] > 0.5
        assert metrics['win_rate'] > 0.45
```

## Performance Considerations

### Computational Cost

**GARCH Forecasting**: ~50ms per forecast (60+ data points)
- **Mitigation**: Cache forecasts, only recompute periodically

**Ledoit-Wolf Shrinkage**: ~10ms per covariance estimation
- **Impact**: Minimal, already faster than sample covariance

**Position Sizing**: <1ms per calculation
- **Impact**: Negligible

**Total Overhead**: ~100ms per rebalance decision
- **Acceptable**: Rebalancing happens weekly, not intraday

### Memory Usage

**Additional State Tracking**:
- Previous weights: ~1KB per strategy
- Stop loss levels: ~100 bytes per position
- GARCH models: ~10KB per asset

**Total Additional Memory**: ~50KB per strategy instance
- **Impact**: Negligible for modern systems

## Rollout Plan

### Phase 1: Risk Management (Week 1)
1. Implement position sizing module
2. Implement stop losses module
3. Add portfolio limits to backtesting engine
4. Update all portfolio strategies with new parameters
5. Run validation tests
6. Generate comparison report

### Phase 2: Estimation Improvements (Week 2)
1. Integrate Ledoit-Wolf covariance
2. Implement GARCH forecasting module
3. Update strategies to use new estimators
4. Run validation tests
5. Generate comparison report

### Phase 3: Transaction Cost Optimization (Week 3)
1. Implement transaction cost module
2. Add rebalancing thresholds to strategies
3. Integrate PyPortfolioOpt transaction cost objective
4. Run validation tests
5. Generate final comparison report

### Rollback Strategy

Each phase can be independently disabled via feature flags:

```python
RISK_MANAGEMENT_ENABLED = True
GARCH_FORECASTING_ENABLED = True
LEDOIT_WOLF_ENABLED = True
TRANSACTION_COST_OPTIMIZATION_ENABLED = True
```

If any phase shows regressions, disable via flag and investigate.

## Open Technical Questions

1. **Dynamic Lookback Period**: Should we implement volatility-based lookback adjustment?
   - **Recommendation**: Yes, add to Phase 2 (regime adaptation)
   - **Complexity**: Medium (requires volatility regime detection)

2. **Stop Loss State Persistence**: How to track stop levels across backtest runs?
   - **Recommendation**: Add to backtesting engine state
   - **Implementation**: Store in `Position` objects

3. **Multi-Asset GARCH**: Should we use multivariate GARCH for covariance forecasting?
   - **Recommendation**: No (Phase 1), too complex for initial implementation
   - **Future Enhancement**: Consider for Phase 4

4. **Correlation Limit Enforcement**: What if no uncorrelated assets available?
   - **Recommendation**: Log warning, use best available pair
   - **Fallback**: Allow single-asset allocation if necessary

---

**Status**: DRAFT
**Last Updated**: 2025-10-24
**Review Status**: Pending technical review
