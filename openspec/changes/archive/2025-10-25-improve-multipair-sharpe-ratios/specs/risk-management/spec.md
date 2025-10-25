# Spec Delta: Risk Management

## ADDED Requirements

### Requirement: Trailing Stop Loss Protection

Portfolio strategies SHALL implement trailing stop losses to limit downside risk and lock in profits.

**Acceptance Criteria**:
- Stop loss trails 8% below peak price by default
- Stop loss adjusted using ATR (Average True Range) for volatility
- Stop loss never goes below entry price once in profit
- Stop loss triggered automatically by backtesting engine

**Parameters**:
- `trailing_stop_pct`: Percentage trail from peak (default: 0.08 = 8%)
- `atr_multiplier`: ATR multiplier for volatility adjustment (default: 2.5)

**Rationale**: Trailing stops limit downside while allowing profits to run. ATR adjustment prevents premature stops in volatile markets.

#### Scenario: Stop loss follows price upward

**Given** a position with:
- Entry price: $100
- Current price: $120
- Peak price since entry: $120
- ATR: $5
- Trailing stop: 8%

**When** calculating stop loss level

**Then** fixed stop SHALL be $100 × (1 - 0.08) = $92

**And** trailing stop SHALL be $120 × (1 - 0.08) = $110.40

**And** ATR stop SHALL be $120 - (2.5 × $5) = $107.50

**And** active stop SHALL be max($92, $110.40, $107.50) = $110.40 (trailing)

**And** if price drops to $110.40, position SHALL be closed

#### Scenario: Stop loss locks in profit

**Given** a position with:
- Entry price: $100
- Current price: $130
- Peak price since entry: $150
- Trailing stop: 8%

**When** calculating stop loss level

**Then** trailing stop SHALL be $150 × (1 - 0.08) = $138

**And** active stop SHALL be $138 (above entry price)

**And** profit of at least 38% SHALL be locked in

**And** stop SHALL never go below entry price of $100

#### Scenario: ATR adjustment in high volatility

**Given** a position with:
- Current price: $100
- Peak price: $100
- ATR: $15 (high volatility)
- Trailing stop: 8%
- ATR multiplier: 2.5

**When** calculating stop loss level

**Then** trailing stop SHALL be $100 × (1 - 0.08) = $92

**And** ATR stop SHALL be $100 - (2.5 × $15) = $62.50

**And** active stop SHALL be max($92, $62.50) = $92

**And** wider stop SHALL prevent premature exit in volatile market

---

### Requirement: Correlation-Based Position Limits

Portfolio strategies SHALL enforce maximum correlation between positions to ensure diversification.

**Acceptance Criteria**:
- Correlation calculated using rolling window (default: 30 days)
- Maximum correlation threshold enforced (default: 0.70)
- New positions rejected if correlation exceeds threshold
- Warning logged when limit prevents position

**Parameters**:
- `max_correlation`: Maximum pairwise correlation (default: 0.70)
- `correlation_window`: Rolling window for correlation (default: 30)

#### Scenario: Reject highly correlated position

**Given** a portfolio holding BTC/USDT (40% weight)

**And** ETH/USDT has 30-day correlation of 0.85 with BTC/USDT

**And** maximum correlation limit is 0.70

**When** attempting to add ETH/USDT position

**Then** position SHALL be rejected

**And** warning SHALL be logged: "Correlation 0.85 exceeds limit 0.70"

**And** portfolio SHALL maintain only BTC/USDT position

#### Scenario: Accept uncorrelated position

**Given** a portfolio holding BTC/USDT and ETH/USDT

**And** BNB/USDT has correlation < 0.70 with both existing positions

**When** attempting to add BNB/USDT position

**Then** position SHALL be accepted

**And** portfolio SHALL hold all three assets

#### Scenario: Fallback when no uncorrelated assets available

**Given** all available assets have correlation > 0.70 with existing positions

**When** selecting new positions

**Then** system SHALL log warning

**And** system SHALL select asset with lowest correlation

**And** strategy SHALL continue with best available diversification

---

### Requirement: Portfolio Drawdown Control

Portfolio strategies SHALL implement drawdown controls to reduce risk after losses.

**Acceptance Criteria**:
- Portfolio-level drawdown tracked continuously
- Maximum drawdown threshold enforced (default: 15%)
- Position sizes halved after 10% drawdown
- Recovery period enforced before returning to full size (default: 10 days)

**Parameters**:
- `max_drawdown_pct`: Maximum portfolio drawdown (default: 0.15 = 15%)
- `drawdown_reduction_trigger`: Drawdown that triggers reduction (default: 0.10 = 10%)
- `drawdown_reduction_factor`: Position size reduction factor (default: 0.5 = 50%)
- `recovery_period_days`: Days before returning to full size (default: 10)

#### Scenario: Reduce position sizes after 10% drawdown

**Given** a portfolio with:
- Peak equity: $100,000
- Current equity: $90,000 (10% drawdown)
- Normal position size: 10%

**When** calculating new position sizes

**Then** drawdown SHALL be calculated as (100000 - 90000) / 100000 = 0.10 = 10%

**And** reduction trigger SHALL be activated (10% ≥ 10%)

**And** position size SHALL be reduced to 10% × 0.5 = 5%

**And** recovery period timer SHALL start

#### Scenario: Enforce maximum 15% drawdown

**Given** a portfolio with:
- Peak equity: $100,000
- Current equity: $85,000 (15% drawdown)
- Current position value: $8,500

**When** checking risk limits

**Then** drawdown SHALL be 15%

**And** maximum drawdown threshold SHALL be reached

**And** all new positions SHALL be rejected

**And** system SHALL log critical warning

**And** existing positions SHALL be monitored for stop loss triggers

#### Scenario: Recovery after drawdown

**Given** a portfolio that had 10% drawdown 8 days ago

**And** equity has recovered to only -2% from peak

**And** recovery period is 10 days

**When** calculating position sizes

**Then** position sizes SHALL still be reduced (8 days < 10 days)

**And** after 10 days, position sizes SHALL return to normal

**And** recovery period SHALL reset

---

### Requirement: Per-Position State Tracking

Backtesting engine SHALL track per-position state for risk management.

**Acceptance Criteria**:
- Entry price tracked for each position
- Peak price since entry tracked and updated
- Stop loss level tracked and updated
- Position age tracked (for time-based exits)

**Implementation**:
- Extend `Position` class with risk management fields
- Update state on every price update
- Validate stop loss on every bar

#### Scenario: Track position lifecycle

**Given** a new position opened at $100

**When** price moves to $110, then $105

**Then** entry price SHALL remain $100

**And** peak price SHALL be updated to $110

**And** peak price SHALL NOT decrease when price drops to $105

**And** stop loss level SHALL trail below $110

**And** position age SHALL increment each bar

---

## Cross-References

**Related Capabilities**:
- `portfolio-optimization`: Kelly position sizing uses stop loss protection
- `volatility-forecasting`: ATR calculation for stop loss adjustment
- `transaction-cost-optimization`: Reduced trading frequency benefits from wider stops

**Integration Points**:
- `src/crypto_trader/backtesting/engine.py`: Risk limit validation
- `src/crypto_trader/core/types.py`: Position state tracking
- `src/crypto_trader/strategies/base.py`: Risk management hooks

**Dependencies**:
- NumPy >= 1.20.0 (correlation calculations)
- Pandas >= 1.3.0 (rolling windows)

---

**Status**: PROPOSED
**Impact**: MODERATE (modifies backtesting engine, adds Position fields)
**Breaking Changes**: None (Position class extensions are backward-compatible)
