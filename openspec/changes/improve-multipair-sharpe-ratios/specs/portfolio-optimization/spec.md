# Spec Delta: Portfolio Optimization

## ADDED Requirements

### Requirement: Kelly Criterion Position Sizing

Portfolio strategies SHALL implement Kelly Criterion position sizing to optimize capital allocation.

**Acceptance Criteria**:
- Position sizes calculated using fractional Kelly (25% of full Kelly)
- Hard limits enforced (2% minimum, 15% maximum)
- Signal confidence scaling applied
- Expected return and volatility inputs validated

**Parameters**:
- `kelly_fraction`: Fraction of full Kelly to use (default: 0.25)
- `min_position_pct`: Minimum position size (default: 0.02)
- `max_position_pct`: Maximum position size (default: 0.15)

#### Scenario: Calculate position size for high-confidence signal

**Given** a portfolio strategy with:
- Expected annual return: 13%
- Annual volatility: 40%
- Win rate: 55%
- Signal confidence: 1.0
- Kelly fraction: 0.25

**When** calculating position size

**Then** position size SHALL be between 2% and 15% of capital

**And** position size SHALL scale with signal confidence

**And** position size SHALL respect hard limits even with extreme inputs

---

### Requirement: Ledoit-Wolf Covariance Shrinkage

Portfolio strategies SHALL use Ledoit-Wolf shrinkage estimator for covariance matrix calculation instead of sample covariance.

**Acceptance Criteria**:
- Covariance matrices calculated using PyPortfolioOpt's `risk_models.CovarianceShrinkage().ledoit_wolf()`
- Returns converted to prices before estimation
- Resulting matrices are positive semi-definite
- Fallback to sample covariance if Ledoit-Wolf fails

**Parameters**:
- `use_ledoit_wolf`: Enable Ledoit-Wolf shrinkage (default: True)

**Rationale**: Sample covariance is unstable with limited crypto data. Ledoit-Wolf shrinkage improves out-of-sample performance by shrinking toward a structured estimator.

#### Scenario: Estimate covariance for 3-asset portfolio

**Given** return series for BTC/USDT, ETH/USDT, BNB/USDT with 90 days of data

**When** calculating covariance matrix using Ledoit-Wolf shrinkage

**Then** covariance matrix SHALL be positive semi-definite

**And** eigenvalues SHALL all be non-negative

**And** estimation SHALL complete without numerical errors

**And** out-of-sample forecast error SHALL be lower than sample covariance

#### Scenario: Fallback when Ledoit-Wolf fails

**Given** insufficient data (< 30 points) or numerical instability

**When** Ledoit-Wolf shrinkage fails

**Then** system SHALL fall back to sample covariance

**And** warning SHALL be logged

**And** strategy SHALL continue without crashing

---

### Requirement: Transaction Cost Penalty in Optimization

Portfolio optimization SHALL include transaction cost penalty to reduce excessive rebalancing.

**Acceptance Criteria**:
- Transaction cost penalty added to optimization objective
- Previous weights tracked for turnover calculation
- PyPortfolioOpt's `objective_functions.transaction_cost` used
- Rebalancing skipped if cost exceeds benefit threshold

**Parameters**:
- `transaction_cost_pct`: Transaction cost per trade (default: 0.001 = 10 bps)
- `min_rebalance_benefit`: Minimum benefit to justify rebalance (default: 0.005 = 50 bps)

#### Scenario: Skip rebalancing when cost exceeds benefit

**Given** a portfolio with current weights {BTC: 0.50, ETH: 0.50}

**And** target weights {BTC: 0.52, ETH: 0.48}

**And** transaction cost of 10 bps per trade

**When** calculating rebalancing decision

**Then** turnover SHALL be calculated as sum of absolute weight changes = 0.04

**And** estimated transaction cost SHALL be turnover × cost_pct = 0.04 × 0.001 = 0.0004 = 4 bps

**And** rebalancing SHALL be skipped because 4 bps < 50 bps threshold

**And** strategy SHALL return previous weights unchanged

#### Scenario: Execute rebalancing when benefit exceeds cost

**Given** a portfolio with current weights {BTC: 0.60, ETH: 0.40}

**And** target weights {BTC: 0.40, ETH: 0.60}

**And** transaction cost of 10 bps per trade

**When** calculating rebalancing decision

**Then** turnover SHALL be 0.40

**And** estimated transaction cost SHALL be 40 bps

**And** rebalancing SHALL be executed because benefit threshold is only 50 bps

**And** strategy SHALL return target weights

---

## MODIFIED Requirements

### Requirement: Hierarchical Risk Parity Optimization

HRP strategy SHALL incorporate Kelly position sizing, Ledoit-Wolf covariance, and transaction cost optimization.

**Changes**:
- ADDED: Kelly position sizing for weight allocation
- ADDED: Ledoit-Wolf shrinkage for covariance estimation
- ADDED: Transaction cost penalty in recursive bisection
- MODIFIED: Previous weights tracked for turnover calculation

#### Scenario: HRP with all improvements

**Given** an HRP strategy with:
- Assets: BTC/USDT, ETH/USDT, BNB/USDT
- 90 days of return data
- Kelly fraction: 0.25
- Ledoit-Wolf enabled
- Transaction cost: 10 bps

**When** generating portfolio weights

**Then** covariance matrix SHALL be estimated using Ledoit-Wolf

**And** hierarchical clustering SHALL group correlated assets

**And** weights SHALL be allocated using Kelly-scaled recursive bisection

**And** rebalancing SHALL only occur if benefit > 50 bps

**And** final position sizes SHALL be between 2% and 15%

---

### Requirement: Risk Parity Optimization

Risk Parity strategy SHALL use Ledoit-Wolf covariance and Kelly position sizing.

**Changes**:
- ADDED: Ledoit-Wolf shrinkage for risk contribution calculation
- ADDED: Kelly position sizing for equal risk contribution
- ADDED: Transaction cost awareness in rebalancing

#### Scenario: Risk Parity with improved covariance

**Given** a Risk Parity strategy with 3 assets and 60 days of data

**When** calculating equal risk contribution weights

**Then** risk contributions SHALL be calculated using Ledoit-Wolf covariance

**And** weights SHALL satisfy equal risk contribution constraint

**And** position sizes SHALL be Kelly-scaled

**And** total portfolio volatility SHALL be controlled

---

### Requirement: Black-Litterman Optimization

Black-Litterman strategy SHALL incorporate all portfolio optimization improvements.

**Changes**:
- ADDED: Ledoit-Wolf shrinkage for posterior covariance
- ADDED: Kelly position sizing for final weights
- ADDED: Transaction cost penalty in Efficient Frontier optimization

#### Scenario: Black-Litterman with transaction costs

**Given** a Black-Litterman strategy with:
- Market equilibrium weights
- 2 investor views
- Confidence in views: [0.8, 0.6]

**When** calculating posterior weights

**Then** prior covariance SHALL use Ledoit-Wolf shrinkage

**And** posterior distribution SHALL incorporate views

**And** Efficient Frontier optimization SHALL include transaction cost penalty

**And** final weights SHALL be Kelly-scaled

**And** rebalancing decision SHALL consider transaction costs

---

## Cross-References

**Related Capabilities**:
- `risk-management`: Position sizing and stop losses
- `volatility-forecasting`: GARCH volatility for Kelly sizing
- `transaction-cost-optimization`: Rebalancing thresholds

**Affected Strategies**:
- `HierarchicalRiskParity`
- `RiskParity`
- `BlackLitterman`
- `CopulaPairsTrading`

**Dependencies**:
- PyPortfolioOpt >= 1.5.0 (Ledoit-Wolf, transaction costs)
- NumPy >= 1.20.0 (matrix operations)
- Pandas >= 1.3.0 (data handling)

---

**Status**: PROPOSED
**Impact**: MODERATE (modifies 4 strategies, adds new parameters)
**Breaking Changes**: None (all parameters have defaults)
