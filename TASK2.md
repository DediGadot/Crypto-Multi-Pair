# Multi-Pair Strategy Optimization Plan for Better Sharpe Ratios

> **Objective**: Systematically optimize multi-pair and portfolio trading strategies to achieve consistent Sharpe ratios > 2.0, focusing on correlation dynamics, portfolio effects, and cross-asset relationships.

## Executive Summary

Based on comprehensive analysis of the codebase and current performance metrics, multi-pair strategies show significant potential but need targeted optimization. Current Sharpe ratios range from **-7.7** (CopulaPairsTrading) to **1.31** (PortfolioRebalancer). This plan focuses exclusively on enhancing multi-pair and portfolio strategies.

### Current Multi-Pair Strategy Performance

| Strategy | Current Sharpe Range | Status | Priority |
|----------|---------------------|---------|----------|
| CopulaPairsTrading | -7.7 | BROKEN | Critical |
| StatisticalArbitrage | Unknown | Needs Testing | High |
| PortfolioRebalancer | 1.16 - 1.31 | Good | Optimize |
| HierarchicalRiskParity | 0.52 - 0.76 | Moderate | Optimize |
| BlackLitterman | 0.52 - 0.76 | Moderate | Optimize |
| RiskParity | 0.50 - 0.73 | Moderate | Optimize |
| DeepRLPortfolio | 0.15 - 0.32 | Poor | Enhance |

---

## Phase 1: Fix Critical Multi-Pair Strategy Issues

**Timeline**: Day 1-2
**Expected Sharpe Improvement**: +1.0 to +2.0

### Task 1.1: Fix CopulaPairsTrading Strategy ⚠️ CRITICAL

**Current Issue**: Sharpe ratio of -7.7 indicates fundamental bugs

**File**: `src/crypto_trader/strategies/library/copula_pairs_trading.py`

**Changes Required**:

1. **Fix spread calculation and z-score normalization**
   - Review `_calculate_pair_signals_detailed()` method
   - Ensure spread = log(price1) - hedge_ratio * log(price2)
   - Verify z-score: (spread - mean) / std
   - Add numerical stability checks for extreme values

2. **Add cointegration pre-filter**
   - Implement Johansen test before trading any pair
   - Only trade pairs with test statistic > critical value
   - Add cointegration strength to metadata

3. **Implement proper hedge ratio calculation**
   - Use Johansen test eigenvector for hedge ratio
   - Add rolling window recalibration (every 30 days)
   - Validate hedge ratio stability (reject if std > 0.3)

4. **Fix tail probability calculations**
   - Review Student-t copula fitting
   - Ensure degrees of freedom > 2
   - Add fallback to Gaussian copula if fitting fails

5. **Add position sizing based on spread volatility**
   - Scale position by 1/spread_volatility
   - Add maximum position limits

**Validation**:
```python
# Test with BTC/ETH pair over last 365 days
# Expected: Sharpe > 0.5, Max Drawdown < 20%
# Verify cointegration test fires correctly
```

### Task 1.2: Enhance StatisticalArbitrage Strategy

**File**: `src/crypto_trader/strategies/library/statistical_arbitrage_pairs.py`

**Changes Required**:

1. **Add dynamic half-life calculation**
   - Use Ornstein-Uhlenbeck process: dS = θ(μ - S)dt + σdW
   - Half-life = -log(2)/θ
   - Recalculate every week

2. **Implement Kalman filter for hedge ratio updates**
   - State vector: [hedge_ratio, drift]
   - Observation: spread
   - Update hedge ratio in real-time

3. **Add regime-aware entry/exit thresholds**
   - Use HMM regime detection (already in codebase)
   - Volatile regime: wider thresholds (±2.5σ entry, ±0.3σ exit)
   - Stable regime: tighter thresholds (±2.0σ entry, ±0.5σ exit)

4. **Fix position sizing based on spread volatility**
   - Kelly criterion with 25% reduction for safety
   - Target 2% portfolio risk per trade

**Validation**:
```python
# Test regime detection accuracy
# Verify half-life calculations
# Check Kalman filter convergence
```

---

## Phase 2: Multi-Pair Correlation & Risk Management

**Timeline**: Day 3-5
**Expected Sharpe Improvement**: +0.5 to +1.0

### Task 2.1: Create Cross-Pair Correlation Framework

**New File**: `src/crypto_trader/analysis/correlation_manager.py`

**Implementation**:

```python
"""
Multi-Pair Correlation Management

Tracks real-time correlation matrices, detects regime changes,
and provides correlation-based risk limits.
"""

class CorrelationManager:
    def __init__(self, lookback_period: int = 90):
        self.lookback_period = lookback_period
        self.correlation_matrix = None
        self.rolling_correlations = None

    def update_correlations(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling correlation matrix"""
        # Use exponential weighting for recency

    def detect_correlation_regime(self) -> str:
        """Detect crisis/normal/decorrelated regimes"""
        # Crisis: avg correlation > 0.7
        # Normal: 0.3 < avg correlation < 0.7
        # Decorrelated: avg correlation < 0.3

    def get_correlation_adjusted_weights(self,
                                        base_weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust portfolio weights based on correlation"""
        # Reduce exposure to highly correlated pairs

    def get_marginal_var_contributions(self,
                                      positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate each pair's contribution to portfolio VaR"""
```

**Key Features**:
- Exponential weighting (λ=0.94 for daily data)
- DCC-GARCH for dynamic correlations
- Regime detection using correlation clustering
- Position limits: max 50% weight when correlation > 0.8

### Task 2.2: Multi-Pair Risk Aggregation

**File**: `src/crypto_trader/orchestration/multipair_window_manager.py`

**Enhancements**:

1. **Add portfolio VaR calculation**
   ```python
   def calculate_portfolio_var(self,
                              positions: Dict[str, float],
                              confidence: float = 0.95) -> float:
       """Calculate portfolio VaR using correlation matrix"""
       # Use variance-covariance method
       # VaR = sqrt(w' * Σ * w) * z_score
   ```

2. **Add correlation-adjusted Sharpe ratio**
   ```python
   def calculate_diversification_ratio(self) -> float:
       """(Weighted avg volatility) / (Portfolio volatility)"""
       # Should be > 1.0 for true diversification
   ```

3. **Dynamic pair weight optimization**
   - Minimize portfolio variance subject to return constraints
   - Use CVXPY for convex optimization
   - Rebalance when weights drift > 15%

---

## Phase 3: Advanced Portfolio Strategies Enhancement

**Timeline**: Week 1
**Expected Sharpe Improvement**: +0.5 to +1.0

### Task 3.1: Optimize HierarchicalRiskParity

**File**: `src/crypto_trader/strategies/library/hierarchical_risk_parity.py`

**Current Sharpe**: 0.52 - 0.76 → **Target**: > 1.5

**Enhancements**:

1. **Add GARCH volatility forecasting**
   ```python
   from arch import arch_model

   def forecast_volatility(self, returns: pd.Series) -> float:
       """Use GARCH(1,1) for better vol estimates"""
       model = arch_model(returns, vol='Garch', p=1, q=1)
       results = model.fit(disp='off')
       forecast = results.forecast(horizon=1)
       return forecast.variance.values[-1, :][0] ** 0.5
   ```

2. **Implement transaction cost-aware rebalancing**
   - Only rebalance if expected benefit > transaction costs
   - Use quadratic penalty for turnover
   - Target < 10% monthly turnover

3. **Add regime-specific clustering**
   - Crisis: use correlation-based clustering
   - Normal: use covariance-based clustering
   - Bull market: add momentum tilt

4. **Optimize lookback window**
   - Dynamic lookback based on regime
   - High volatility: shorter window (60 days)
   - Low volatility: longer window (180 days)

**Validation**:
```python
# Test on 2-year rolling windows
# Target: Sharpe > 1.5, Max DD < 15%
# Verify clustering stability
```

### Task 3.2: Enhance PortfolioRebalancer

**File**: `src/crypto_trader/strategies/library/portfolio_rebalancer.py`

**Current Sharpe**: 1.16 - 1.31 → **Target**: > 2.0

**Enhancements**:

1. **Implement mean-variance optimization with Sharpe maximization**
   ```python
   def optimize_weights_sharpe(self,
                              expected_returns: pd.Series,
                              cov_matrix: pd.DataFrame) -> Dict[str, float]:
       """Maximize Sharpe ratio subject to constraints"""
       from cvxpy import Variable, Maximize, Problem, norm

       w = Variable(len(expected_returns))
       ret = expected_returns.values @ w
       risk = norm(cov_matrix.values @ w, 2)

       # Maximize Sharpe = return / risk
       objective = Maximize(ret / risk)
       constraints = [sum(w) == 1, w >= 0]  # Long-only

       problem = Problem(objective, constraints)
       problem.solve()
       return dict(zip(expected_returns.index, w.value))
   ```

2. **Add momentum overlay for tactical allocation**
   - 12-month momentum score for each asset
   - Overweight top momentum quintile by 20%
   - Underweight bottom quintile by 20%

3. **Dynamic rebalancing thresholds**
   - High volatility regime: 20% threshold
   - Low volatility regime: 10% threshold
   - Use realized volatility to determine regime

4. **Multi-timeframe rebalancing signals**
   - Daily: tactical adjustments (max 5% weight change)
   - Weekly: threshold-based rebalancing
   - Monthly: full reoptimization

**Expected Impact**: +0.5 to +0.7 Sharpe improvement

### Task 3.3: Improve BlackLitterman Strategy

**File**: `src/crypto_trader/strategies/library/black_litterman.py`

**Current Sharpe**: 0.52 - 0.76 → **Target**: > 1.5

**Enhancements**:

1. **Add ML-based views generation**
   ```python
   def generate_ml_views(self, features: pd.DataFrame) -> Dict[str, float]:
       """Use gradient boosting for return predictions"""
       from lightgbm import LGBMRegressor

       # Features: momentum, volatility, on-chain metrics
       # Target: forward 30-day returns
       # Output: expected return per asset
   ```

2. **Implement confidence decay**
   - Recent views: 100% confidence
   - 1-month old: 70% confidence
   - 3-month old: 30% confidence
   - Exponential decay: conf = exp(-age/30)

3. **Dynamic tau parameter**
   - Uncertainty measure: VIX equivalent for crypto
   - High uncertainty: tau = 0.1 (trust views more)
   - Low uncertainty: tau = 0.025 (trust market more)

4. **Cross-asset momentum views**
   - If BTC momentum > 80th percentile → positive view on all crypto
   - If correlation spike → negative view on low-cap alts
   - If funding rate > 0.1% → mean reversion view

**Expected Impact**: +0.6 to +1.0 Sharpe improvement

---

## Phase 4: New Multi-Pair High-Sharpe Strategies

**Timeline**: Week 1-2
**Expected Sharpe Improvement**: +1.0 to +1.5

### Task 4.1: Create Cross-Sectional Momentum Strategy

**New File**: `src/crypto_trader/strategies/library/cross_sectional_momentum.py`

**Strategy Logic**:

1. **Rank assets by multiple momentum factors**
   ```python
   def calculate_momentum_score(self, data: pd.DataFrame) -> pd.Series:
       """Composite momentum score"""
       # 1-month momentum: 25% weight
       # 3-month momentum: 25% weight
       # 6-month momentum: 25% weight
       # 12-month momentum: 25% weight

       # Z-score normalization for each factor
       # Combine into composite score
   ```

2. **Long top quartile, short bottom quartile**
   - Rebalance monthly
   - Equal weight within each bucket
   - Market-neutral construction

3. **Dynamic leverage based on momentum dispersion**
   - High dispersion → higher leverage (max 2x)
   - Low dispersion → lower leverage (min 1x)
   - Dispersion = std(momentum_scores)

4. **Risk parity position sizing**
   - Target equal risk contribution
   - Scale by inverse volatility

**Expected Sharpe**: > 2.0
**Expected Max Drawdown**: < 15%

### Task 4.2: Create Multi-Pair Mean Reversion Basket

**New File**: `src/crypto_trader/strategies/library/basket_mean_reversion.py`

**Strategy Logic**:

1. **PCA-based factor extraction**
   ```python
   def extract_factors(self, returns: pd.DataFrame, n_components: int = 3):
       """Extract principal components"""
       from sklearn.decomposition import PCA

       pca = PCA(n_components=n_components)
       factors = pca.fit_transform(returns)
       loadings = pca.components_

       # Factor 1: market factor (usually explains 60-70% variance)
       # Factor 2: size factor
       # Factor 3: momentum factor
   ```

2. **Trade factor deviations**
   - Calculate factor z-scores
   - Entry: |z-score| > 2.0
   - Exit: |z-score| < 0.5
   - Mean reversion on factors, not individual assets

3. **Optimal basket construction using CVaR**
   - Minimize CVaR (tail risk) for given return
   - Diversify across factors
   - Constraint: max 30% weight per asset

4. **Dynamic entry/exit based on factor half-life**
   - Estimate Ornstein-Uhlenbeck parameters
   - Only trade factors with half-life: 5-30 days
   - Reject factors with half-life > 60 days (trending)

**Expected Sharpe**: > 1.8
**Expected Max Drawdown**: < 12%

### Task 4.3: Create Correlation Arbitrage Strategy

**New File**: `src/crypto_trader/strategies/library/correlation_arbitrage.py`

**Strategy Logic**:

1. **Trade correlation mean reversion between pairs**
   ```python
   def calculate_realized_correlation(self,
                                      returns1: pd.Series,
                                      returns2: pd.Series,
                                      window: int = 30) -> pd.Series:
       """Rolling correlation"""
       return returns1.rolling(window).corr(returns2)
   ```

2. **Identify correlation regime changes**
   - Normal correlation for BTC/ETH: 0.7-0.9
   - Entry: correlation > 0.95 or < 0.5 (extreme)
   - Exit: correlation returns to 0.7-0.9 range

3. **Position sizing based on correlation stability**
   - Stable correlation (low std): larger positions
   - Unstable correlation (high std): smaller positions

4. **Synthetic correlation swap**
   - Long-short portfolio to isolate correlation exposure
   - Delta-hedge individual asset exposure
   - Pure correlation bet

**Expected Sharpe**: > 1.5
**Target**: 65% win rate on correlation mean reversion

---

## Phase 5: Multi-Pair Execution Optimization

**Timeline**: Week 2
**Expected Sharpe Improvement**: +0.3 to +0.5

### Task 5.1: Synchronized Multi-Pair Execution

**New File**: `src/crypto_trader/execution/multipair_executor.py`

**Implementation**:

```python
class MultiPairExecutor:
    """Execute trades across multiple pairs atomically"""

    def execute_portfolio_rebalance(self,
                                   target_weights: Dict[str, float],
                                   current_weights: Dict[str, float],
                                   total_capital: float) -> List[Trade]:
        """Atomic multi-pair rebalancing"""
        # 1. Calculate required trades
        # 2. Optimize execution order (sell first to raise capital)
        # 3. Execute with TWAP over 15-minute window
        # 4. Monitor slippage and adjust

    def minimize_market_impact(self,
                              trades: List[Dict],
                              urgency: str = 'medium') -> List[Trade]:
        """Stagger orders to minimize impact"""
        # High urgency: 5-minute TWAP
        # Medium urgency: 15-minute TWAP
        # Low urgency: 1-hour VWAP
```

**Features**:
- Atomic execution (all-or-nothing for portfolio rebalancing)
- Intelligent order routing across exchanges
- Cross-pair netting to reduce fees
- Real-time slippage monitoring

### Task 5.2: Portfolio Rebalancing Optimization

**New File**: `src/crypto_trader/execution/rebalance_optimizer.py`

**Implementation**:

```python
class RebalanceOptimizer:
    """Optimize portfolio rebalancing for costs and tracking error"""

    def optimize_rebalance_trades(self,
                                 current_weights: Dict[str, float],
                                 target_weights: Dict[str, float],
                                 transaction_costs: Dict[str, float]) -> Dict[str, float]:
        """Find optimal trades minimizing cost + tracking error"""
        from cvxpy import Variable, Minimize, Problem, quad_form, norm

        # Decision variable: trade sizes
        # Objective: minimize tracking_error + lambda * transaction_costs
        # Constraints: budget constraint, no short sales (if long-only)

    def calculate_turnover(self, trades: Dict[str, float]) -> float:
        """Calculate portfolio turnover percentage"""
        return sum(abs(trade) for trade in trades.values()) / 2
```

**Features**:
- Tax-loss harvesting for crypto pairs
- Minimize tracking error during rebalancing
- Transaction cost optimization using quadratic programming
- Partial rebalancing for large portfolios (only move weights > 10% off target)

---

## Phase 6: Multi-Pair Specific Enhancements

**Timeline**: Week 2-3
**Expected Sharpe Improvement**: +0.3 to +0.5

### Task 6.1: Enhanced Window Management

**File**: `src/crypto_trader/orchestration/multipair_window_manager.py`

**Enhancements**:

1. **Adaptive window sizing based on regime**
   ```python
   def determine_optimal_window_size(self, volatility: float) -> int:
       """Larger windows in stable regimes, smaller in volatile"""
       if volatility < 0.02:  # Low vol
           return 180
       elif volatility < 0.04:  # Medium vol
           return 90
       else:  # High vol
           return 60
   ```

2. **Overlapping window analysis for stability**
   - Test strategy on overlapping windows
   - Require Sharpe > threshold in 80% of windows
   - Reject strategies with high window-to-window variance

3. **Cross-validation across multiple pairs**
   - Train on 70% of pairs, test on 30%
   - Rotate testing pairs
   - Ensure strategy works across different pair types

4. **Synchronized train/test splits with embargo gap**
   - Add 7-day gap between train and test
   - Prevent look-ahead bias
   - Synchronized windows across all pairs

### Task 6.2: Multi-Pair Feature Engineering

**New File**: `src/crypto_trader/features/multipair_features.py`

**Implementation**:

```python
class MultiPairFeatureGenerator:
    """Generate cross-pair features"""

    def calculate_cross_pair_order_flow(self,
                                       pairs: List[str]) -> pd.DataFrame:
        """Aggregate order flow imbalance across pairs"""
        # Net buying pressure across all pairs
        # Leading indicator for market moves

    def detect_lead_lag_relationships(self,
                                     data: Dict[str, pd.DataFrame]) -> Dict[Tuple[str, str], int]:
        """Find which pairs lead others"""
        # Use Granger causality test
        # BTC often leads altcoins by 1-4 hours

    def calculate_market_microstructure_aggregate(self,
                                                 pairs: List[str]) -> pd.DataFrame:
        """Aggregate bid-ask spreads, depth, etc."""
        # Market-wide liquidity measure

    def detect_cross_exchange_arbitrage(self,
                                       pair: str,
                                       exchanges: List[str]) -> pd.DataFrame:
        """Identify arbitrage opportunities"""
        # Price differences across exchanges
        # Proxy for market efficiency
```

**Key Features**:
- Cross-pair order flow imbalance
- Lead-lag relationship detection
- Aggregate market microstructure metrics
- Cross-exchange arbitrage signals

### Task 6.3: Portfolio-Level Signal Aggregation

**New File**: `src/crypto_trader/strategies/aggregators/signal_combiner.py`

**Implementation**:

```python
class SignalCombiner:
    """Combine signals across multiple pairs and strategies"""

    def weighted_signal_combination(self,
                                   signals: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Weight signals by recent Sharpe ratio"""
        # signals = {pair: {strategy: signal}}
        # weights based on rolling 30-day Sharpe

    def correlation_aware_filtering(self,
                                   signals: Dict[str, float]) -> Dict[str, float]:
        """Filter out redundant signals from correlated pairs"""
        # If BTC and ETH both signal BUY, reduce weight on ETH
        # Avoid over-concentration

    def portfolio_entry_exit_timing(self,
                                   individual_signals: Dict[str, float]) -> str:
        """Determine portfolio-level entry/exit"""
        # Enter when 60% of pairs signal BUY
        # Exit when 60% signal SELL
        # Ensemble approach

    def risk_budgeted_allocation(self,
                                signals: Dict[str, float],
                                risk_budgets: Dict[str, float]) -> Dict[str, float]:
        """Allocate risk budget across signals"""
        # Each pair gets fixed risk allocation
        # Scale position size accordingly
```

---

## Phase 7: Multi-Pair Performance Analytics

**Timeline**: Week 3
**Expected Sharpe Improvement**: +0.2 to +0.3

### Task 7.1: Enhanced Multi-Pair Metrics

**File**: `src/crypto_trader/analysis/multipair_aggregator.py`

**New Metrics**:

1. **Information Ratio per pair**
   ```python
   def calculate_information_ratio(self,
                                  pair_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> float:
       """IR = (return - benchmark) / tracking_error"""
       active_returns = pair_returns - benchmark_returns
       return active_returns.mean() / active_returns.std()
   ```

2. **Contribution to portfolio Sharpe**
   ```python
   def calculate_sharpe_contribution(self,
                                    pair: str,
                                    portfolio_returns: pd.Series) -> float:
       """Marginal contribution of pair to portfolio Sharpe"""
       # Remove pair and recalculate Sharpe
       # Contribution = Sharpe_with - Sharpe_without
   ```

3. **Rolling correlation windows**
   - 30-day, 90-day, 180-day rolling correlations
   - Track correlation stability
   - Flag regime changes

4. **Diversification ratio tracking**
   - Target: DR > 1.5
   - Monitor over time
   - Alert if DR drops below 1.2

### Task 7.2: Multi-Pair Backtesting Improvements

**File**: `src/crypto_trader/backtesting/engine.py`

**Enhancements**:

1. **Synchronized multi-pair execution**
   ```python
   def execute_portfolio_trades(self,
                               trades: Dict[str, Trade],
                               timestamp: datetime) -> Dict[str, Trade]:
       """Execute all trades at same timestamp"""
       # Ensure atomic execution
       # Calculate portfolio-level metrics
   ```

2. **Cross-pair margin requirements**
   - Calculate portfolio margin (not sum of individual margins)
   - Use correlation for margin offsets
   - Enforce portfolio-level leverage limits

3. **Portfolio-level stop losses**
   - Trigger if portfolio drawdown > 15%
   - Close all positions
   - Re-enter when drawdown recovers to < 10%

4. **Correlation-aware position limits**
   - Max 30% in any single pair
   - Max 60% in highly correlated pairs (correlation > 0.8)
   - Dynamically adjust based on correlation matrix

---

## Implementation Priority Summary

### Week 1 Priority Tasks

**Critical (Must Complete)**:
1. ✅ Task 1.1: Fix CopulaPairsTrading (-7.7 Sharpe → positive)
2. ✅ Task 1.2: Enhance StatisticalArbitrage
3. ✅ Task 2.1: Create CorrelationManager

**High Priority**:
4. ✅ Task 3.1: Optimize HierarchicalRiskParity
5. ✅ Task 3.2: Enhance PortfolioRebalancer
6. ✅ Task 4.1: Create CrossSectionalMomentum

### Week 2 Priority Tasks

**High Priority**:
7. ✅ Task 3.3: Improve BlackLitterman
8. ✅ Task 4.2: Create BasketMeanReversion
9. ✅ Task 5.1: MultiPairExecutor

**Medium Priority**:
10. ✅ Task 4.3: Create CorrelationArbitrage
11. ✅ Task 5.2: RebalanceOptimizer
12. ✅ Task 6.1: Enhanced WindowManager

### Week 3 Priority Tasks

**Medium Priority**:
13. ✅ Task 6.2: MultiPairFeatureGenerator
14. ✅ Task 6.3: SignalCombiner
15. ✅ Task 7.1: Enhanced Metrics

**Low Priority (Optional)**:
16. ✅ Task 7.2: Backtesting improvements
17. ✅ Task 2.2: Risk aggregation enhancements

---

## Success Metrics & Validation

### Portfolio-Level Targets

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Average Sharpe | 0.8 | > 2.0 | 90-day rolling |
| Portfolio Sharpe | 1.2 | > 2.5 | Full portfolio |
| Max Drawdown | 25% | < 15% | Historical max |
| Diversification Ratio | 1.1 | > 1.5 | Daily calculation |
| Win Rate | 48% | > 55% | Per strategy |
| Correlation Stability | 0.4 | < 0.3 | Std of rolling corr |

### Strategy-Specific Targets

| Strategy | Current Sharpe | Target Sharpe | Key Improvement |
|----------|---------------|---------------|-----------------|
| CopulaPairsTrading | -7.7 | > 0.5 | Bug fixes |
| StatisticalArbitrage | Unknown | > 1.5 | Kalman filter |
| PortfolioRebalancer | 1.3 | > 2.0 | Sharpe optimization |
| HierarchicalRiskParity | 0.7 | > 1.5 | GARCH forecasting |
| BlackLitterman | 0.7 | > 1.5 | ML views |
| CrossSectionalMomentum | N/A | > 2.0 | New strategy |
| BasketMeanReversion | N/A | > 1.8 | New strategy |
| CorrelationArbitrage | N/A | > 1.5 | New strategy |

### Testing & Validation Checklist

**Before Deployment**:
- [ ] All strategies tested on 2+ years of data
- [ ] Out-of-sample testing shows Sharpe > 1.0
- [ ] Walk-forward analysis shows stability
- [ ] Correlation matrix stability verified
- [ ] Cointegration tests validate pair selection
- [ ] Transaction costs included in all backtests
- [ ] Slippage model calibrated to real trades
- [ ] Portfolio VaR < 5% at 95% confidence
- [ ] Max drawdown < 20% in worst window
- [ ] Strategy survives stress test scenarios

**Stress Test Scenarios**:
1. **2022 Crypto Winter**: BTC -60%, high correlation
2. **Flash Crash**: -20% in 1 hour
3. **Low Liquidity**: 10x normal slippage
4. **Correlation Spike**: All pairs correlation > 0.95
5. **Funding Rate Shock**: Funding > 1% per 8 hours

---

## Technical Dependencies

### New Python Packages Required

```toml
[project.dependencies]
# Optimization
cvxpy = "^1.4.0"  # Convex optimization for portfolio
optuna = "^3.5.0"  # Hyperparameter optimization

# Time Series / Econometrics
arch = "^6.3.0"  # GARCH volatility models
statsmodels = "^0.14.0"  # Already present, ensure version

# Machine Learning
lightgbm = "^4.2.0"  # Gradient boosting for views
shap = "^0.44.0"  # Feature importance

# Portfolio Optimization
pypfopt = "^1.5.5"  # Portfolio optimization (already present)

# Correlation & Clustering
scipy = "^1.11.0"  # Already present
scikit-learn = "^1.3.0"  # Already present

# Execution
ccxt = "^4.2.0"  # Already present for exchange API
```

### Existing Infrastructure to Leverage

- ✅ VectorBT for backtesting
- ✅ PyPortfolioOpt for HRP/Black-Litterman
- ✅ HMMLearn for regime detection
- ✅ Statsmodels for cointegration tests
- ✅ Pandas/NumPy for data processing
- ✅ Loguru for logging

---

## Risk Management & Monitoring

### Real-Time Monitoring Dashboard

**Key Metrics to Display**:
1. Current portfolio Sharpe (30-day rolling)
2. Per-pair Sharpe contributions
3. Correlation matrix heatmap
4. Portfolio VaR (95%, 99%)
5. Current drawdown from peak
6. Diversification ratio
7. Transaction cost ratio
8. Strategy allocation weights

### Alerts & Circuit Breakers

**Trigger Immediate Alert If**:
- Portfolio drawdown > 10%
- Any pair Sharpe < -2.0 over 7 days
- Correlation spike: average > 0.9
- VaR breach: loss exceeds VaR 3 days in row
- Execution slippage > 1% on any trade
- Cointegration breaks (test statistic below critical)

**Automatic Position Closing If**:
- Portfolio drawdown > 15%
- Single pair loss > 5% of portfolio
- Leverage exceeds 1.5x (if using margin)
- Correlation > 0.95 across all pairs (crisis mode)

### Weekly Review Checklist

**Every Monday**:
- [ ] Review all strategy Sharpe ratios
- [ ] Check correlation matrix for changes
- [ ] Validate cointegration for pairs trading
- [ ] Review transaction costs vs. expectations
- [ ] Analyze worst-performing pair
- [ ] Update volatility forecasts
- [ ] Rebalance if needed

---

## Documentation & Code Standards

### Code Review Requirements

**Before Merging**:
1. All functions have type hints
2. Docstrings follow Google style
3. Unit tests cover 80%+ of code
4. Integration tests pass on 2 year backtest
5. No pylint warnings (score > 9.0)
6. Performance validated on real data
7. Sharpe improvement demonstrated

### Documentation Updates

**For Each New Strategy**:
1. Strategy description in docstring
2. Mathematical formulation
3. Parameter descriptions with ranges
4. Expected Sharpe and drawdown
5. Validation results
6. Example usage in `if __name__ == "__main__"`
7. Integration with `master_windowed_multipair.py`

### Version Control

**Git Commit Standards**:
- Prefix: `[TASK-X.Y]` for each task
- Example: `[TASK-1.1] Fix CopulaPairsTrading spread calculation`
- One commit per logical change
- Include validation results in commit message

**Branching Strategy**:
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/task-X.Y`: Individual task branches
- PR required for merge to `develop`
- 2 approvals required for merge to `main`

---

## Next Steps

### Immediate Actions (Today)

1. **Read and understand** this task plan thoroughly
2. **Set up development environment** with new dependencies
3. **Run baseline backtests** to establish current metrics
4. **Begin Task 1.1**: Fix CopulaPairsTrading (highest priority)

### First Week Goals

- ✅ Complete all Phase 1 tasks (critical fixes)
- ✅ Complete Phase 2 correlation framework
- ✅ Start Phase 3 portfolio enhancements
- 🎯 **Achieve**: Average multi-pair Sharpe > 1.0

### By End of Month

- ✅ Complete all 7 phases
- ✅ All new strategies deployed and tested
- ✅ Monitoring dashboard operational
- 🎯 **Achieve**: Average multi-pair Sharpe > 2.0

---

## Appendix: Mathematical Formulations

### A1: Sharpe Ratio Calculation

```
Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev of Returns

For crypto (annualized):
- Mean Return: Average of daily returns × 365
- Std Dev: Std of daily returns × √365
- Risk-Free Rate: 2% (typical for 2024-2025)
```

### A2: Portfolio Variance with Correlations

```
Portfolio Variance = w' Σ w

Where:
- w = vector of portfolio weights
- Σ = covariance matrix
- w' = transpose of w

Portfolio Std Dev = √(Portfolio Variance)
```

### A3: Diversification Ratio

```
DR = (Σ wᵢ σᵢ) / σₚ

Where:
- wᵢ = weight of asset i
- σᵢ = volatility of asset i
- σₚ = portfolio volatility

DR > 1.0 indicates diversification benefit
DR = 1.0 indicates no diversification
```

### A4: Information Ratio

```
IR = (Rₚ - Rᵦ) / TE

Where:
- Rₚ = portfolio return
- Rᵦ = benchmark return
- TE = tracking error (std dev of active returns)
```

### A5: Cointegration Test (Johansen)

```
Δy_t = Π y_{t-1} + Σ Γᵢ Δy_{t-i} + ε_t

Where:
- Π = α β' (error correction term)
- α = adjustment coefficients
- β = cointegrating vectors
- Rank(Π) = number of cointegrating relationships

Test statistic > critical value → cointegration exists
```

### A6: Half-Life Calculation (Ornstein-Uhlenbeck)

```
dS_t = θ(μ - S_t)dt + σ dW_t

Half-life = -ln(2) / θ

Where:
- θ = mean reversion speed
- μ = long-term mean
- σ = volatility

Estimate θ via regression: ΔS_t = α + β S_{t-1} + ε_t
Then: θ = -β
```

---

**Document Version**: 1.0
**Last Updated**: 2025-01-23
**Next Review**: After Phase 1 completion
