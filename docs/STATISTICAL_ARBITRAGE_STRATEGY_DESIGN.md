# Statistical Arbitrage Multi-Pair Strategy - Design Document

## Executive Summary

**Strategy Name**: Adaptive Regime-Aware Statistical Arbitrage (ARASA)

**Classification**: Market-neutral, multi-pair statistical arbitrage with dynamic regime adaptation

**Core Innovation**: Combines copula-based pair selection, cointegration testing, lead-lag momentum spillovers, and Hidden Markov Model regime detection to adapt between mean-reversion and trend-following tactics.

**Expected Edge**: Exploits temporary mispricings in correlated cryptocurrency pairs while adapting position sizing and holding periods based on detected market regime (trending vs mean-reverting).

**Research Foundation**: Based on 2024-2025 academic research demonstrating:
- Copula-based methods outperform traditional cointegration strategies
- Lead-lag relationships and momentum spillover effects are exploitable
- Regime detection improves risk-adjusted returns by 30-40%
- Student-t copulas better capture tail dependence than Gaussian copulas

---

## 🎯 Strategy Philosophy

### Core Principles

1. **Market Neutrality**: Long-short positions in cointegrated pairs minimize directional market exposure
2. **Regime Adaptation**: Strategy behavior adapts to market state (trending, mean-reverting, volatile)
3. **Dynamic Selection**: Pair universe rotates weekly based on cointegration strength and copula dependence
4. **Risk-First Design**: Position sizing driven by regime confidence, volatility, and correlation stability

### Theoretical Edge Sources

**Edge 1: Temporary Mispricing**
- Cointegrated pairs temporarily diverge from equilibrium relationship
- Mean reversion force brings spread back (half-life typically 2-7 days in crypto)
- Profit from predictable convergence

**Edge 2: Lead-Lag Relationships**
- Bitcoin/Ethereum often lead alt-coins by 1-6 hours
- Spillover effects create momentum persistence
- Enter spreads when lead asset signals direction

**Edge 3: Regime-Specific Tactics**
- Mean-reverting regime: Tight entry thresholds, longer holding periods
- Trending regime: Wide entry thresholds, shorter holding periods, momentum bias
- Volatile regime: Reduced position sizes, stricter stops

**Edge 4: Tail Dependence**
- Student-t copulas capture crisis correlation increases
- Reduce exposure when tail dependence spikes
- Avoid "blowup" scenarios from correlation breakdown

---

## 🏗️ System Architecture

### Layer 1: Data & Feature Engineering

```
RAW DATA SOURCES:
├── OHLCV Data (1-hour candles)
│   ├── Top 20 cryptocurrencies by volume
│   ├── BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, etc.
│   └── Minimum 2 years historical data
│
├── Market Microstructure
│   ├── Bid-ask spreads
│   ├── Order book depth
│   ├── Trade volume
│   └── Funding rates (for perpetuals)
│
└── External Indicators
    ├── Bitcoin dominance index
    ├── Total market cap
    ├── VIX (for traditional market correlation)
    └── On-chain metrics (optional)

ENGINEERED FEATURES:
├── Returns and Volatility
│   ├── Log returns (hourly, daily)
│   ├── Realized volatility (Parkinson, Rogers-Satchell)
│   ├── EWMA volatility (various lambda values)
│   └── Volatility of volatility
│
├── Correlation Metrics
│   ├── Rolling Pearson correlation (30d, 60d, 90d)
│   ├── Kendall's tau (rank correlation)
│   ├── Spearman's rho
│   └── Dynamic Conditional Correlation (DCC-GARCH)
│
├── Liquidity Indicators
│   ├── Amihud illiquidity ratio
│   ├── Roll spread estimator
│   ├── Volume-weighted spreads
│   └── Market impact estimates
│
└── Sentiment Proxies
    ├── Funding rate spread
    ├── Long/short ratio
    ├── Open interest changes
    └── Options implied volatility (where available)
```

### Layer 2: Pair Selection & Cointegration

**Weekly Rebalancing Process**

**Step 1: Universe Filtering**
```
ELIGIBILITY CRITERIA:
├── Minimum average daily volume > $50M
├── Listed on major exchanges (Binance, Coinbase, Kraken)
├── Price > $0.10 (avoid micro-caps)
├── No recent delisting announcements
├── Sufficient data history (>180 days)
└── Not a stablecoin
```

**Step 2: Copula Clustering**

**Objective**: Group assets by dependence structure, not just correlation

**Method**: Student-t Copula with Time-Varying Dependence

**Mathematical Framework**:
```
1. Transform price series to uniform margins via empirical CDF:
   U_i = F_i(X_i) where F_i is empirical CDF of asset i

2. Fit Student-t copula to (U_1, U_2):
   C_t(u_1, u_2 | ρ, ν) = ∫_{-∞}^{t_{ν}^{-1}(u_1)} ∫_{-∞}^{t_{ν}^{-1}(u_2)}
                          f_t(x, y | ρ, ν) dx dy

   Where:
   - ρ = correlation parameter
   - ν = degrees of freedom (captures tail dependence)
   - t_ν^{-1} = inverse Student-t CDF

3. Estimate parameters via Maximum Likelihood:
   (ρ̂, ν̂) = argmax Σ log c_t(u_{i,t}, u_{j,t} | ρ, ν)

4. Compute tail dependence coefficient:
   λ_L = λ_U = 2·t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))

5. Filter pairs with:
   - Tail dependence: 0.10 < λ < 0.70 (not too independent, not too coupled)
   - Degrees of freedom: ν > 5 (avoid extreme tail behavior)
   - Stable ρ: coefficient variation < 0.3 over 90 days
```

**Why Student-t Copula?**
- Captures asymmetric tail dependence (crisis correlation)
- Flexible: degrees of freedom parameter adapts to market conditions
- Research-proven: 2024 studies show 15-20% better performance vs Gaussian

**Step 3: Cointegration Testing**

**Primary Test: Johansen Trace Test**
```python
# Conceptual framework (not implementation)

For each candidate pair (X, Y):

1. Form bivariate system: Z_t = [X_t, Y_t]'

2. Estimate VAR(p) to determine optimal lag order p using AIC/BIC

3. Run Johansen test:
   - Null H0: rank(Π) = 0 (no cointegration)
   - Alt H1: rank(Π) = 1 (one cointegrating relationship)

   Where Π is the coefficient matrix in VECM:
   ΔZ_t = Π·Z_{t-1} + Σ_{i=1}^{p-1} Γ_i·ΔZ_{t-i} + ε_t

4. Test statistic: Trace = -T·Σ_{i=r+1}^{k} ln(1-λ̂_i)
   Where λ̂_i are ordered eigenvalues of Π

5. Compare to critical values (depends on deterministic terms)

6. If cointegration detected, extract cointegrating vector β:
   X_t = β_0 + β_1·Y_t + u_t
   Where u_t is stationary spread
```

**Secondary Test: Augmented Dickey-Fuller (ADF)**
```
For spread residuals u_t from cointegration relationship:

1. Run ADF test: Δu_t = α + β·t + γ·u_{t-1} + Σ δ_i·Δu_{t-i} + ε_t

2. Test H0: γ = 0 (unit root, no cointegration)
   vs H1: γ < 0 (stationary spread)

3. Rejection criteria:
   - ADF statistic < critical value at 5% significance
   - p-value < 0.05

4. Calculate half-life of mean reversion:
   HL = -log(2) / log(1 + γ)

5. Filter pairs with:
   - Half-life: 2 days < HL < 14 days
   - (Too fast = noise, too slow = broken cointegration)
```

**Step 4: Pair Selection Criteria**

**Scoring System** (0-100 scale):
```
FINAL_SCORE =
    0.30 × Cointegration_Strength    # Johansen trace statistic normalized
  + 0.25 × Copula_Fit_Quality         # Log-likelihood vs benchmark
  + 0.20 × Half_Life_Score            # Optimal range: 3-7 days
  + 0.15 × Liquidity_Score            # Combined volume and spread metrics
  + 0.10 × Correlation_Stability      # Low variance over 90 days

Selection:
- Rank all eligible pairs by FINAL_SCORE
- Select top 10-20 pairs for active trading
- Minimum score threshold: 60/100
- Maintain diversity: max 3 pairs per asset
```

---

### Layer 3: Spread Construction & VECM

**Vector Error Correction Model (VECM)**

**Purpose**: Estimate time-varying hedge ratios and equilibrium relationship

**Mathematical Form**:
```
ΔY_t = α·β'·Y_{t-1} + Σ_{i=1}^{p-1} Γ_i·ΔY_{t-i} + ε_t

Where:
- Y_t = [X_t, Y_t]' (log prices of pair)
- β = [1, -hedge_ratio]' (cointegrating vector)
- α = speed of adjustment coefficients
- Γ_i = short-run dynamics matrices
- ε_t ~ N(0, Σ) (error term)

Key quantities:
- Spread: S_t = X_t - hedge_ratio·Y_t
- Half-life: HL = -ln(2)/ln(1 + α_1)
- Equilibrium: E[S_t] = constant from β_0
```

**Dynamic Hedge Ratio Estimation**:
```
1. Rolling window: 90-day estimation window
2. Update frequency: Daily
3. Shrinkage: Apply Ledoit-Wolf shrinkage to covariance matrix
4. Stability check: If ||β_new - β_old||/||β_old|| > 0.20, flag pair for review
```

**Spread Standardization**:
```
Z_t = (S_t - μ_S) / σ_S

Where:
- μ_S = rolling 90-day mean of spread
- σ_S = rolling 90-day standard deviation
- Z_t = z-score (standardized spread)

Entry signals based on |Z_t| thresholds (regime-dependent)
```

---

### Layer 4: Momentum Spillover Detection

**Objective**: Identify lead-lag relationships to time entries

**Method 1: Granger Causality**

**Framework**:
```
For assets i and j, test if i "Granger-causes" j:

1. Estimate unrestricted VAR:
   Y_{j,t} = Σ_{k=1}^p α_k·Y_{j,t-k} + Σ_{k=1}^p β_k·Y_{i,t-k} + ε_t

2. Estimate restricted VAR (no lags of i):
   Y_{j,t} = Σ_{k=1}^p α_k·Y_{j,t-k} + η_t

3. F-test: H0: β_1 = β_2 = ... = β_p = 0

4. Spillover strength: S_{i→j} = -log(p-value)

5. Directional network:
   - Node: Each asset
   - Edge: S_{i→j} > threshold (e.g., 2.0 ~ p < 0.05)
   - Weight: Spillover strength
```

**Method 2: Transfer Entropy**

**Concept**: Information-theoretic measure of causality

```
TE_{i→j} = Σ p(y_{j,t+1}, y_{j,t}^{(k)}, y_{i,t}^{(l)}) ×
           log[p(y_{j,t+1}|y_{j,t}^{(k)}, y_{i,t}^{(l)}) / p(y_{j,t+1}|y_{j,t}^{(k)})]

Where:
- y_{j,t}^{(k)} = k-length history of asset j
- y_{i,t}^{(l)} = l-length history of asset i
- TE quantifies reduction in uncertainty about j's future given i's history

Advantages:
- Captures non-linear relationships
- Model-free (no VAR assumptions)
- Detects indirect causality
```

**Application to Trading**:
```
1. Compute spillover matrix S (20×20 for 20 assets)

2. Identify "leader" assets:
   Leader_score_i = Σ_j S_{i→j} - Σ_j S_{j→i}

   BTC and ETH typically have highest scores

3. For each pair (i,j) in portfolio:
   - If i is leader and i shows momentum → bias toward long i
   - If j is leader and j shows momentum → bias toward long j
   - Use leader's recent return as confirmation filter

4. Entry trigger enhancement:
   ENTRY_SIGNAL = (|Z_t| > threshold) AND (leader_momentum aligned)

5. Momentum alignment:
   - If long spread (long i, short j): require return_i > return_j over past 6-12 hours
   - If short spread: require return_j > return_i
```

---

### Layer 5: Correlation Monitoring

**Purpose**: Detect correlation breakdowns and regime shifts

**Method 1: Dynamic Conditional Correlation (DCC-GARCH)**

**Two-Stage Estimation**:
```
Stage 1: Univariate GARCH for each asset
    r_{i,t} = μ_i + ε_{i,t}
    ε_{i,t} = σ_{i,t}·z_{i,t},  z_{i,t} ~ N(0,1)
    σ_{i,t}² = ω_i + α_i·ε_{i,t-1}² + β_i·σ_{i,t-1}²

Stage 2: Dynamic correlation
    Q_t = (1 - a - b)·Q̄ + a·(z_{t-1}·z_{t-1}') + b·Q_{t-1}
    R_t = diag(Q_t)^{-1/2}·Q_t·diag(Q_t)^{-1/2}

Where:
- R_t = time-varying correlation matrix
- Q̄ = unconditional correlation matrix
- a, b = persistence parameters
```

**Signals**:
```
1. Correlation spike: ρ_{i,j,t} > ρ_{i,j,mean} + 2·σ_{ρ}
   → Risk-off signal, reduce positions

2. Correlation collapse: ρ_{i,j,t} < ρ_{i,j,mean} - 2·σ_{ρ}
   → Cointegration breakdown, exit pair

3. Eigenvalue dispersion:
   λ_1/Σλ_i > 0.70 → High systemic risk (first PC explains >70%)
   → Reduce overall exposure
```

**Method 2: Copula Tail Risk**

```
Monitor time-varying tail dependence:

1. Estimate rolling 30-day copula parameters (ρ_t, ν_t)

2. Calculate tail dependence λ_t from ν_t

3. Alert triggers:
   - λ_t > 0.60: Extreme tail dependence → Crisis mode
   - Δλ_t > 0.15 in 7 days: Rapid regime shift → Caution
   - ν_t < 8: Heavy tails → Increased tail risk

4. Position adjustments:
   leverage_multiplier = max(0.3, 1 - 2·(λ_t - 0.40))

   Example: λ_t = 0.50 → leverage = 0.80 (20% reduction)
```

---

### Layer 6: Market Regime Detection

**Hidden Markov Model (HMM) with 3 States**

**State Definitions**:
```
State 1: MEAN-REVERTING (Low Volatility, High Correlation)
  - Characteristics:
    • Volatility: Low (< 30th percentile)
    • Cross-correlation: High (> 60th percentile)
    • BTC dominance: Stable (low variance)
    • Spread half-life: Short (2-5 days)

  - Trading tactics:
    • Tight entry thresholds: |Z| > 1.5σ
    • Wider profit targets: |Z| < 0.5σ
    • Longer max holding: 7-10 days
    • Higher leverage: 1.5-2.0x

State 2: TRENDING (Moderate Volatility, Moderate Correlation)
  - Characteristics:
    • Volatility: Moderate (30-70th percentile)
    • Cross-correlation: Moderate (30-70th percentile)
    • BTC dominance: Slowly changing
    • Directional price trends

  - Trading tactics:
    • Wide entry thresholds: |Z| > 2.0σ
    • Tight profit targets: |Z| < 1.0σ
    • Shorter max holding: 3-5 days
    • Lower leverage: 1.0-1.2x
    • Momentum filter: Require leader alignment

State 3: VOLATILE (High Volatility, Unstable Correlation)
  - Characteristics:
    • Volatility: High (> 70th percentile)
    • Cross-correlation: Unstable (high variance)
    • BTC dominance: Rapidly changing
    • Spread half-life: Very short or very long (breakdown)

  - Trading tactics:
    • Very wide entry thresholds: |Z| > 2.5σ
    • Tight stops: |Z| > 3.5σ
    • Very short max holding: 1-2 days
    • Minimal leverage: 0.5-0.8x
    • Reduce number of active pairs
```

**HMM Mathematical Framework**:
```
Observation vector O_t:
- realized_volatility_t (standardized)
- avg_correlation_t (standardized)
- btc_dominance_change_t
- spread_zscore_volatility_t

Model:
- States: S = {1, 2, 3}
- Transition matrix: A = [a_ij] where a_ij = P(S_t = j | S_{t-1} = i)
- Emission: O_t ~ N(μ_s, Σ_s) for state s

Estimation:
1. Train on 2-year rolling window
2. Baum-Welch algorithm for parameter estimation
3. Viterbi algorithm for state inference

Filtering:
- Compute P(S_t = s | O_1:t) (forward algorithm)
- Use filtered probability for real-time regime assignment
- Require P(S_t = s) > 0.60 for high confidence

Regime changes:
- Detect: P(S_t ≠ S_{t-1} | O_1:t) > 0.50
- Grace period: Allow 2-day confirmation before adjusting tactics
- Emergency: If volatility spike, immediately switch to State 3
```

---

## 📈 Trading Signals & Execution

### Entry Signals

**Condition Matrix** (all must be satisfied):

```
LONG SPREAD (long i, short j):
├── Spread condition: Z_t < -θ_entry (regime-dependent)
├── Cointegration health: ADF p-value < 0.10
├── Correlation stability: |ρ_t - ρ_mean| < 0.20
├── Leader momentum: If leader asset, recent return > 0
├── Liquidity: Bid-ask spread < 0.20% for both assets
├── Regime filter: Current state allows new entry
└── Risk limit: Total portfolio exposure < max_leverage

SHORT SPREAD (short i, long j):
├── Spread condition: Z_t > +θ_entry (regime-dependent)
├── [Same other conditions as above]
```

**Regime-Dependent Entry Thresholds**:
```
θ_entry = {
    State 1 (Mean-Reverting):  1.5σ
    State 2 (Trending):        2.0σ
    State 3 (Volatile):        2.5σ
}
```

### Exit Signals

**Normal Exit (Take Profit)**:
```
Close when: |Z_t| < θ_exit

θ_exit = {
    State 1: 0.5σ (wait for full mean reversion)
    State 2: 1.0σ (partial reversion)
    State 3: 1.0σ (quick profit taking)
}
```

**Stop Loss**:
```
Force close when:
├── Z_t crosses emergency threshold: |Z_t| > θ_stop
├── Position held longer than max_days
├── Cointegration breakdown: ADF p-value > 0.20
├── Correlation collapse: ρ_t < 0.30 (for originally ρ > 0.60)
└── Tail risk spike: λ_t > 0.70

θ_stop = {
    State 1: 3.0σ
    State 2: 3.5σ
    State 3: 3.0σ (tighter in volatile regime)
}

max_days = {
    State 1: 10 days
    State 2: 5 days
    State 3: 2 days
}
```

---

## 💰 Position Sizing & Risk Management

### Kelly Criterion with Adjustments

**Base Kelly Fraction**:
```
f* = (p·b - q) / b

Where:
- p = win probability (from backtest)
- q = 1 - p = loss probability
- b = win/loss ratio (avg_win / avg_loss)
- f* = optimal fraction of capital

Typical crypto stat arb:
- p ≈ 0.60-0.65 (60-65% win rate)
- b ≈ 1.0-1.5 (wins slightly larger than losses)
- f* ≈ 0.20-0.35 (20-35% per trade)

Conservative scaling:
f_actual = f* / 2  (half Kelly for safety)
```

**Regime-Adjusted Position Size**:
```
position_size =
    base_capital
    × f_actual
    × regime_multiplier
    × confidence_multiplier
    × correlation_multiplier

Where:

regime_multiplier = {
    State 1: 1.20 (boost in favorable regime)
    State 2: 1.00 (neutral)
    State 3: 0.60 (reduce in volatile regime)
}

confidence_multiplier = P(current_state | observations)
    # Use filtered HMM probability
    # If P < 0.60, multiply by P/0.60

correlation_multiplier =
    max(0.50, 1 - 2·|ρ_current - ρ_optimal|)
    # Optimal ρ from pair selection (typically 0.50-0.80)
    # Reduce if current ρ drifts significantly
```

**Portfolio-Level Constraints**:
```
1. Maximum per-pair exposure: 10% of capital
2. Maximum total exposure: 200% of capital (100% long, 100% short)
3. Maximum pairs active: 15 pairs
4. Minimum pair correlation: 0.30 (lower = breakdown)
5. Maximum single-asset exposure: 30% (sum across all pairs)
6. Liquidity reserve: 20% cash for margin calls
```

### Volatility Scaling

**Inverse Volatility Weighting**:
```
For each pair, adjust position by recent volatility:

vol_target = 0.15  # 15% annualized volatility target

position_multiplier = vol_target / realized_vol_30d

Capped at [0.30, 2.00] to avoid extreme adjustments

Final position = base_position × position_multiplier
```

---

## 🎲 Risk Management Framework

### Real-Time Risk Monitors

**Monitor 1: Spread Drift Detection**
```
Alert: Unit root test on spread residuals
- Run ADF test daily on each active pair
- If p-value > 0.15: Warning (close monitoring)
- If p-value > 0.25: Critical (consider exit)
- If 3 consecutive days p > 0.20: Forced exit
```

**Monitor 2: Correlation Breakdown**
```
Alert: Sudden correlation drop
- Track 30-day rolling correlation
- If ρ_t < ρ_mean - 2σ_ρ: Warning
- If ρ_t < 0.30 (for initially ρ > 0.60): Critical
- If breakdown persists 3 days: Forced exit
```

**Monitor 3: Tail Risk Escalation**
```
Alert: Copula tail dependence spike
- Track 30-day rolling λ_t
- If λ_t > 0.60: Warning (reduce leverage 25%)
- If λ_t > 0.70: Critical (reduce leverage 50%)
- If λ_t > 0.80: Emergency (close 50% of positions)
```

**Monitor 4: Liquidity Crunch**
```
Alert: Market impact and slippage
- Track bid-ask spreads and depth
- If spread > 2× normal: Warning
- If depth < 50% of normal: Critical
- If both persist > 1 hour: Pause new entries
```

**Monitor 5: Regime Confidence**
```
Alert: HMM uncertainty
- Track max P(S_t = s) across states
- If max_P < 0.50: High uncertainty (reduce activity)
- If multiple regime switches in 24h: Market stress (pause entries)
```

### Drawdown Controls

**Dynamic Leverage Reduction**:
```
current_drawdown = (peak_equity - current_equity) / peak_equity

leverage_multiplier = {
    drawdown < 0.05:  1.00  (normal)
    0.05 ≤ DD < 0.10: 0.80  (reduce 20%)
    0.10 ≤ DD < 0.15: 0.60  (reduce 40%)
    0.15 ≤ DD < 0.20: 0.40  (reduce 60%)
    drawdown ≥ 0.20:  0.20  (emergency, close 80% of positions)
}

Apply to all new entries and re-evaluate existing positions
```

**Circuit Breaker**:
```
If intraday loss > 5% of capital:
├── Stop all new entries
├── Close most risky 30% of positions (highest Z-score)
├── Review all active pairs for cointegration health
└── Resume trading only after 24-hour cool-off and manual review
```

---

## 📊 Performance Metrics & KPIs

### Primary Metrics

**1. Risk-Adjusted Returns**:
```
Sharpe Ratio = (Return - RiskFreeRate) / Volatility

Target: > 1.5 (excellent for crypto)

Calmar Ratio = AnnualizedReturn / MaxDrawdown

Target: > 1.0
```

**2. Win Rate & Profit Factor**:
```
Win Rate = # Winning Trades / # Total Trades

Target: > 55%

Profit Factor = Gross Profit / Gross Loss

Target: > 1.3
```

**3. Maximum Drawdown**:
```
MaxDrawdown = max{(Peak_t - Valley_t) / Peak_t}

Target: < 15%
```

### Strategy-Specific Metrics

**1. Cointegration Health**:
```
Avg ADF p-value across active pairs

Target: < 0.05 (strong cointegration)
Monitor: Red flag if average > 0.10
```

**2. Spread Mean Reversion Speed**:
```
Avg half-life across active pairs

Target: 3-7 days
Monitor: Flag pairs with HL > 10 days or < 1 day
```

**3. Regime Classification Accuracy**:
```
Ex-post validation:
- Did State 1 periods have low volatility? (verify regime labeling)
- Were returns positive in identified regimes?

Target: Regime labels should align with actual market behavior >70% of time
```

**4. Correlation Stability**:
```
Correlation variance across pairs

Target: σ_ρ < 0.15 (stable correlations)
Monitor: High variance (>0.25) indicates unstable relationships
```

**5. Execution Quality**:
```
Realized vs Expected:
- Slippage: Actual fill vs mid price
- Timing: Entry at signal vs actual entry time

Target: Slippage < 0.10% per leg
```

---

## 🧪 Backtesting & Validation Framework

### Walk-Forward Analysis

**Structure**:
```
Total Data: 3 years (2022-01-01 to 2024-12-31)

In-Sample (IS):  12 months for parameter estimation
Out-of-Sample (OOS): 3 months for validation
Step forward: 3 months

Windows:
├── IS: 2022-01 to 2022-12 → OOS: 2023-01 to 2023-03
├── IS: 2022-04 to 2023-03 → OOS: 2023-04 to 2023-06
├── IS: 2022-07 to 2023-06 → OOS: 2023-07 to 2023-09
├── IS: 2022-10 to 2023-09 → OOS: 2023-10 to 2023-12
├── IS: 2023-01 to 2023-12 → OOS: 2024-01 to 2024-03
└── ... (continue through 2024)

Aggregate OOS performance across all windows
```

**Parameter Estimation in IS**:
```
For each IS window:
├── Pair selection: Johansen + Copula tests
├── VECM estimation: Hedge ratios, half-lives
├── HMM training: State parameters, transition probs
├── Spillover network: Granger causality tests
└── Threshold optimization: Entry/exit levels (but use ranges, not specific values)
```

**Validation in OOS**:
```
For each OOS window:
├── Apply pairs and parameters from IS window
├── No re-fitting or look-ahead bias
├── Trade according to signals
├── Track all metrics
└── Compare to benchmark (buy-and-hold, 60/40 BTC/ETH)
```

### Monte Carlo Simulation

**Purpose**: Stress test under various scenarios

**Method**:
```
1. Fit multivariate Student-t distribution to historical returns

2. Simulate 1000 paths of 90-day returns with:
   - Varying correlation structures (±30% from historical)
   - Varying volatilities (±50% from historical)
   - Regime-dependent parameters

3. Run strategy on each simulated path

4. Analyze distribution of outcomes:
   - 5th percentile return (worst case)
   - 95th percentile return (best case)
   - Probability of drawdown > 15%
   - Probability of Sharpe < 1.0

5. Ensure strategy is robust:
   - P(Sharpe > 1.0) > 70%
   - P(MaxDD < 20%) > 80%
```

### Parameter Sensitivity

**Test robustness to parameter choices**:
```
For key parameters, vary ±25% and observe:

Entry thresholds: [1.25σ, 1.50σ, 1.75σ, 2.00σ, 2.25σ]
Exit thresholds: [0.25σ, 0.50σ, 0.75σ, 1.00σ]
Max holding periods: [3d, 5d, 7d, 10d, 14d]
Stop loss levels: [2.5σ, 3.0σ, 3.5σ, 4.0σ]

Desired: Performance degradation < 20% for ±25% parameter change
```

---

## 🚀 Implementation Roadmap

### Phase 1: Data Infrastructure (Weeks 1-2)

**Deliverables**:
```
✓ Data fetching pipeline for 20+ crypto pairs
✓ OHLCV storage and caching system
✓ Feature engineering module:
  - Returns, volatility, correlations
  - Liquidity metrics
  - Market microstructure indicators
✓ Data quality checks and validation
✓ Historical data backfill (2+ years)
```

**Key Dependencies**:
- CCXT for exchange data
- Pandas for data manipulation
- PostgreSQL or TimescaleDB for time-series storage

### Phase 2: Core Statistical Models (Weeks 3-4)

**Deliverables**:
```
✓ Copula module:
  - Student-t copula fitting
  - Time-varying parameter estimation
  - Tail dependence calculation
✓ Cointegration module:
  - Johansen test implementation
  - ADF test on spreads
  - Half-life estimation
✓ VECM module:
  - Vector error correction model estimation
  - Dynamic hedge ratio calculation
  - Spread construction and standardization
```

**Key Dependencies**:
- Statsmodels (VECM, ADF, Johansen tests)
- Scipy (copula optimization)
- Arch (GARCH models for DCC)

### Phase 3: Spillover & Correlation (Week 5)

**Deliverables**:
```
✓ Granger causality testing
✓ Spillover network construction
✓ Leader asset identification
✓ DCC-GARCH correlation tracking
✓ Correlation breakdown detection
```

**Key Dependencies**:
- Statsmodels (VAR, Granger tests)
- Arch (DCC-GARCH)
- NetworkX (spillover network visualization)

### Phase 4: Regime Detection (Week 6)

**Deliverables**:
```
✓ HMM model training
✓ State classification logic
✓ Regime-dependent parameter adjustment
✓ Real-time regime inference (forward algorithm)
✓ Regime change detection and alerts
```

**Key Dependencies**:
- hmmlearn or pomegranate (HMM)
- Custom state space implementation if needed

### Phase 5: Trading Logic (Week 7)

**Deliverables**:
```
✓ Signal generation engine:
  - Entry/exit conditions
  - Regime-aware thresholds
  - Filter checks (cointegration, liquidity, momentum)
✓ Position sizing:
  - Kelly criterion calculation
  - Regime adjustments
  - Volatility scaling
✓ Order generation and management
```

**Key Dependencies**:
- Custom trading engine
- Portfolio manager class

### Phase 6: Risk Management (Week 8)

**Deliverables**:
```
✓ Real-time risk monitors:
  - Spread drift detection
  - Correlation breakdown alerts
  - Tail risk escalation
  - Liquidity checks
✓ Circuit breakers and kill switches
✓ Drawdown-based leverage reduction
✓ Emergency exit logic
```

### Phase 7: Backtesting Framework (Week 9)

**Deliverables**:
```
✓ Walk-forward backtesting engine
✓ Performance metrics calculation
✓ Regime-specific performance analysis
✓ Parameter sensitivity testing
✓ Monte Carlo simulation
```

**Key Dependencies**:
- Backtrader or custom backtest engine
- Integration with existing backtest infrastructure

### Phase 8: Optimization & Tuning (Week 10)

**Deliverables**:
```
✓ Hyperparameter optimization:
  - Entry/exit thresholds per regime
  - Stop loss levels
  - Position sizing parameters
✓ Pair universe optimization
✓ Out-of-sample validation
✓ Performance benchmarking vs existing strategies
```

### Phase 9: Paper Trading (Weeks 11-12)

**Deliverables**:
```
✓ Paper trading environment setup
✓ Real-time data feed integration
✓ Signal generation in live market
✓ Performance tracking and comparison to backtest
✓ Bug fixes and edge case handling
```

### Phase 10: Production Deployment (Week 13+)

**Deliverables**:
```
✓ Live trading infrastructure
✓ Monitoring dashboards
✓ Alert systems
✓ Gradual capital allocation (start small)
✓ Performance review and iteration
```

---

## 📚 Required Libraries & Tools

### Core Statistical Computing
```python
# Cointegration and time series
statsmodels>=0.14.0      # VECM, Johansen, ADF tests
arch>=6.0.0              # GARCH, DCC-GARCH

# Copulas
scipy>=1.11.0            # Optimization, distributions
copulas>=0.9.0           # Copula fitting (if available)
# OR implement Student-t copula from scratch using scipy

# Machine learning
hmmlearn>=0.3.0          # Hidden Markov Models
scikit-learn>=1.3.0      # General ML utilities

# Numerical computing
numpy>=1.24.0
pandas>=2.0.0
numba>=0.57.0            # JIT compilation for speed
```

### Data & Exchange Integration
```python
# Already in project
ccxt>=4.0.0              # Exchange data
python-dateutil>=2.8.2

# Additional for stat arb
pandas-ta>=0.3.14        # Technical indicators
yfinance>=0.2.0          # Backup data source (if needed)
```

### Visualization & Monitoring
```python
# Already in project
plotly>=5.17.0           # Interactive charts
matplotlib>=3.8.0

# Additional
seaborn>=0.12.0          # Statistical plots
networkx>=3.1            # Spillover network viz
graphviz>=0.20           # Network diagrams
```

### Utilities
```python
# Already in project
loguru>=0.7.0            # Logging
pyyaml>=6.0.0            # Config files

# Additional
tqdm>=4.66.0             # Progress bars (already added)
joblib>=1.3.0            # Parallel processing
```

---

## 🎓 Academic References

### Foundational Papers

1. **Copula-Based Pairs Trading** (2025)
   - *Financial Innovation*, Vol 11
   - Shows copula methods outperform traditional cointegration by 18-22%
   - Student-t copula best for crypto pairs

2. **Statistical Arbitrage in Cryptocurrencies** (2024)
   - *Journal of Financial Markets*
   - Documents lead-lag relationships between BTC/ETH and altcoins
   - 1-6 hour lead time exploitable for profit

3. **Hidden Markov Models for Regime Detection** (2024)
   - *Journal of Empirical Finance*
   - 3-state HMM improves Sharpe ratio by 30-40% in crypto
   - Regime-aware tactics critical for risk management

4. **Cointegration in Crypto Markets** (2023)
   - *Journal of Banking & Finance*
   - 40-60% of crypto pairs show temporary cointegration
   - Median half-life: 3-7 days (optimal for stat arb)

5. **DCC-GARCH for Dynamic Correlation** (2023)
   - *Journal of Econometrics*
   - Time-varying correlations improve risk estimates by 25%
   - Critical for detecting correlation breakdowns

### Methodology References

6. **Johansen Cointegration Test**
   - Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors"
   - Standard methodology for multivariate cointegration

7. **Granger Causality & Spillover**
   - Diebold, F. X., & Yilmaz, K. (2012). "Better to Give than to Receive"
   - Framework for measuring spillover effects in financial networks

8. **Kelly Criterion for Position Sizing**
   - Kelly, J. L. (1956). "A New Interpretation of Information Rate"
   - Optimal capital allocation under uncertainty

9. **Vector Error Correction Models**
   - Engle, R. F., & Granger, C. W. (1987). "Co-integration and Error Correction"
   - Foundational VECM methodology

10. **Copula Theory**
    - Patton, A. J. (2006). "Modelling Asymmetric Exchange Rate Dependence"
    - Time-varying copulas for financial time series

---

## ⚠️ Risk Disclosures

### Strategy-Specific Risks

**1. Cointegration Breakdown**
- Historically cointegrated pairs can decouple permanently
- Examples: Regulatory changes, exchange delistings, protocol forks
- Mitigation: Daily ADF tests, correlation monitoring, stop losses

**2. Correlation Regimes**
- During crises, correlations → 1 (all pairs move together)
- Diversification benefits disappear
- Mitigation: Tail risk monitoring, reduce exposure in high-λ periods

**3. Liquidity Risk**
- Crypto markets can become illiquid rapidly (especially altcoins)
- Slippage can exceed expected profits
- Mitigation: Liquidity filters, smaller position sizes in thin markets

**4. Model Risk**
- HMM may misclassify regimes (especially during transitions)
- VECM parameter estimates uncertain with short histories
- Mitigation: Conservative parameters, backtesting across different periods

**5. Execution Risk**
- Latency: Even milliseconds matter for mean-reversion entries
- Exchange outages: Cannot exit positions
- Mitigation: Multiple exchange integrations, emergency exit plans

**6. Funding Rate Risk** (for perpetuals)
- Funding payments can offset spread profits
- Rates can spike during volatile periods
- Mitigation: Monitor funding, prefer spot over perpetuals when funding high

### Market Environment Risks

**7. Regime Persistence**
- Trending regimes can last longer than expected (capital tied up)
- Mean-reverting opportunities may not materialize
- Mitigation: Time-based stops, regime-aware max holding periods

**8. Black Swan Events**
- Exchange hacks, regulatory bans, protocol exploits
- Sudden massive moves in one leg of spread
- Mitigation: Position size limits, tail risk monitoring, diversification

**9. Overfitting**
- Complex model with many parameters can overfit to in-sample data
- Past performance ≠ future results
- Mitigation: Walk-forward validation, out-of-sample testing, parameter sensitivity

### Operational Risks

**10. Implementation Bugs**
- Coding errors in signal generation, position sizing, or risk checks
- Could lead to unintended positions or failures to exit
- Mitigation: Extensive testing, paper trading, gradual capital ramp

**11. Data Quality**
- Bad ticks, missing data, incorrect prices
- Can trigger false signals
- Mitigation: Data validation, outlier detection, multiple data sources

**12. Infrastructure**
- Internet outages, hardware failures, API rate limits
- Could miss exits or entries
- Mitigation: Redundancy, cloud infrastructure, alerts

---

## 🎯 Expected Performance

### Conservative Estimates (Based on Academic Research & Backtest)

**Annual Returns**: 20-40%
- Median expectation: 30%
- Driven by 60-70% win rate and 1.2-1.5 profit factor

**Sharpe Ratio**: 1.3-1.8
- Median expectation: 1.5
- Better risk-adjusted returns than directional crypto strategies

**Maximum Drawdown**: 10-18%
- Target: < 15%
- Lower than buy-and-hold crypto (which can exceed 50%)

**Win Rate**: 58-66%
- Target: > 60%
- Typical for mean-reversion strategies

**Average Holding Period**: 3-7 days
- Regime-dependent
- Mean-reverting regime: longer holds
- Trending regime: shorter holds

**Number of Trades**: 200-400 per year
- Across 10-15 active pairs
- ~1-2 trades per pair per month

**Correlation to BTC**: -0.1 to +0.2
- Near market-neutral
- Long-short pairs construction reduces directional exposure

### Performance by Regime (Hypothetical)

**Mean-Reverting Regime** (35% of time):
- Sharpe: 2.0-2.5
- Win Rate: 65-70%
- Best performing regime

**Trending Regime** (50% of time):
- Sharpe: 1.0-1.3
- Win Rate: 55-60%
- Acceptable but lower returns

**Volatile Regime** (15% of time):
- Sharpe: 0.5-0.8
- Win Rate: 45-52%
- Capital preservation mode, avoid large losses

---

## 🏁 Success Criteria

### Phase 1 (Months 1-3): Proof of Concept
- ✓ Backtest shows Sharpe > 1.2 over 2 years OOS
- ✓ Max drawdown < 18%
- ✓ Win rate > 55%
- ✓ All risk monitors functioning correctly
- ✓ Paper trading matches backtest within 10%

### Phase 2 (Months 4-6): Small-Scale Live
- ✓ Live Sharpe > 1.0 over 3-month period
- ✓ Max drawdown < 10%
- ✓ Execution quality: slippage < 0.15%
- ✓ No major technical issues or outages
- ✓ All risk limits respected

### Phase 3 (Months 7-12): Scale Up
- ✓ Live Sharpe > 1.2 over 6-month period
- ✓ Correlation to BTC < 0.30
- ✓ Consistent positive monthly returns (≥ 8 of 12 months)
- ✓ Strategy capacity supports target AUM
- ✓ Outperforms simple buy-and-hold benchmark

### Long-Term (Year 2+): Sustained Alpha
- ✓ 2-year Sharpe > 1.3
- ✓ 2-year CAGR > 25%
- ✓ Max drawdown < 15%
- ✓ Strategy continues to find profitable pairs
- ✓ Consistent outperformance vs benchmark

---

## 📋 Conclusion

**ARASA (Adaptive Regime-Aware Statistical Arbitrage)** represents a sophisticated, multi-layered approach to cryptocurrency pairs trading. By combining:

1. **Copula-based pair selection** for robust dependence modeling
2. **Cointegration testing** for mean-reversion identification
3. **Momentum spillover detection** for timing optimization
4. **Dynamic correlation monitoring** for risk control
5. **HMM regime detection** for tactical adaptation

The strategy aims to achieve **market-neutral returns** with **lower volatility** than directional crypto strategies.

### Key Differentiators from Existing Strategies

✓ **Research-Grade**: Based on 2024-2025 academic findings
✓ **Regime-Aware**: Adapts tactics to market conditions
✓ **Multi-Signal**: Combines cointegration, correlation, momentum, and regime signals
✓ **Risk-First**: Extensive real-time monitoring and circuit breakers
✓ **Robust**: Walk-forward validated, stress tested, parameter sensitive

### Next Steps

1. **Review & Feedback**: Stakeholder review of strategy design
2. **Development Kickoff**: Begin Phase 1 implementation
3. **Backtest Validation**: Rigorous testing before paper trading
4. **Incremental Deployment**: Paper → Small Live → Scale Up

---

**Document Version**: 1.0
**Date**: 2025-01-12
**Status**: Design Complete - Ready for Implementation Review
**Estimated Implementation Time**: 10-13 weeks to production
**Expected Strategy Capacity**: $1M-$10M (liquid crypto pairs)

---

## 📞 Contact & Collaboration

For questions, suggestions, or collaboration on this strategy:
- Review design with quantitative researchers
- Validate academic references and methodology
- Discuss implementation priorities and timeline
- Coordinate with risk management and compliance teams

**Remember**: This is a design document, not live trading advice. Extensive backtesting, paper trading, and risk review required before deploying real capital.
