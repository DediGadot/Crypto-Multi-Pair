# Spec: Transaction Cost Optimization

**Capability**: NEW
**Purpose**: Reduce excessive rebalancing and transaction costs to improve net returns

## Overview

This capability implements transaction cost awareness in portfolio rebalancing decisions. Portfolio strategies will only rebalance when the expected benefit exceeds transaction costs, significantly reducing trading frequency and improving net returns.

---

## ADDED Requirements

### Requirement: Rebalancing Threshold Logic

Portfolio strategies SHALL implement rebalancing thresholds to prevent costly unnecessary trades.

**Acceptance Criteria**:
- Turnover calculated as sum of absolute weight changes
- Transaction cost estimated based on turnover
- Rebalancing executed only if benefit exceeds threshold
- Previous weights tracked for comparison

**Parameters**:
- `transaction_cost_pct`: Cost per trade as percentage (default: 0.001 = 10 bps)
- `min_rebalance_benefit`: Minimum benefit to justify rebalance (default: 0.005 = 50 bps)

**Formula**:
```
turnover = Σ|w_target(i) - w_current(i)| for all assets i
tx_cost = turnover × transaction_cost_pct
should_rebalance = tx_cost < min_rebalance_benefit
```

#### Scenario: Calculate turnover for portfolio rebalance

**Given** a portfolio with current weights:
- BTC/USDT: 0.50
- ETH/USDT: 0.30
- BNB/USDT: 0.20

**And** target weights:
- BTC/USDT: 0.45
- ETH/USDT: 0.35
- BNB/USDT: 0.20

**When** calculating turnover

**Then** weight changes SHALL be:
- BTC/USDT: |0.45 - 0.50| = 0.05
- ETH/USDT: |0.35 - 0.30| = 0.05
- BNB/USDT: |0.20 - 0.20| = 0.00

**And** turnover SHALL be 0.05 + 0.05 + 0.00 = 0.10 (10% of portfolio)

#### Scenario: Skip rebalancing when cost exceeds benefit

**Given** current weights: {BTC: 0.60, ETH: 0.40}

**And** target weights: {BTC: 0.62, ETH: 0.38}

**And** transaction cost: 0.001 (10 bps)

**And** minimum benefit: 0.005 (50 bps)

**When** deciding whether to rebalance

**Then** turnover SHALL be |0.62-0.60| + |0.38-0.40| = 0.04

**And** estimated cost SHALL be 0.04 × 0.001 = 0.00004 = 0.4 bps

**And** cost < benefit threshold (0.4 < 50 bps)

**And** rebalancing SHALL be skipped

**And** previous weights SHALL be returned

**And** decision SHALL be logged

#### Scenario: Execute rebalancing when benefit exceeds cost

**Given** current weights: {BTC: 0.70, ETH: 0.30}

**And** target weights: {BTC: 0.40, ETH: 0.60}

**And** transaction cost: 0.001 (10 bps)

**When** deciding whether to rebalance

**Then** turnover SHALL be |0.40-0.70| + |0.60-0.30| = 0.60

**And** estimated cost SHALL be 0.60 × 0.001 = 0.0006 = 60 bps

**And** cost > benefit threshold (60 > 50 bps) - significant rebalance

**And** rebalancing SHALL be executed

**And** target weights SHALL be returned

---

### Requirement: Transaction Cost Penalty in Optimization

Portfolio optimization SHALL include transaction cost as penalty in objective function.

**Acceptance Criteria**:
- PyPortfolioOpt's `objective_functions.transaction_cost` integrated
- Previous weights passed to optimization
- Cost penalty scaled by transaction cost parameter
- Optimization finds balance between return and turnover

**Implementation**:
```python
from pypfopt.objective_functions import transaction_cost
from pypfopt import EfficientFrontier

ef = EfficientFrontier(expected_returns, cov_matrix)

if previous_weights is not None:
    ef.add_objective(
        transaction_cost,
        w_prev=previous_weights,
        k=transaction_cost_pct  # 0.001 = 10 bps
    )

weights = ef.max_sharpe()
```

#### Scenario: Optimization considers transaction costs

**Given** a portfolio with previous weights: {BTC: 0.50, ETH: 0.50}

**And** optimization without cost penalty suggests: {BTC: 0.55, ETH: 0.45}

**And** transaction cost: 10 bps per trade

**When** optimizing with transaction cost penalty

**Then** objective function SHALL include turnover penalty

**And** optimizer SHALL balance return improvement vs cost

**And** final weights MAY be closer to previous weights

**And** net return (after costs) SHALL be maximized

---

### Requirement: Previous Weights Tracking

Portfolio strategies SHALL track previous weights for turnover calculation.

**Acceptance Criteria**:
- Previous weights stored in strategy instance
- Weights updated after each rebalancing decision
- Initial weights set to None (no previous portfolio)
- Weights persisted across backtest periods

**Implementation**:
```python
class PortfolioStrategy(BaseStrategy):
    def __init__(self):
        self.last_weights: Optional[Dict[str, float]] = None

    def generate_signals(self, data):
        # Calculate target weights
        target_weights = self.calculate_weights(data)

        # Check if should rebalance
        if self.last_weights is not None:
            should_rebalance, turnover = self.should_rebalance(
                current=self.last_weights,
                target=target_weights
            )
            if not should_rebalance:
                return self.last_weights

        # Update last weights
        self.last_weights = target_weights
        return target_weights
```

#### Scenario: Track weights across rebalancing periods

**Given** a portfolio strategy with no previous weights

**When** generating first signals

**Then** `self.last_weights` SHALL be None

**And** rebalancing SHALL execute (no previous portfolio)

**And** `self.last_weights` SHALL be updated to current weights

**When** generating second signals

**Then** `self.last_weights` SHALL contain previous weights

**And** turnover calculation SHALL use previous weights

**And** rebalancing decision SHALL consider transaction costs

---

### Requirement: Transaction Cost Tracking and Reporting

System SHALL track and report total transaction costs in performance metrics.

**Acceptance Criteria**:
- Total transaction costs accumulated per strategy
- Costs reported in performance metrics
- Net return calculated (return - costs)
- Trades per day metric included

**Metrics Added**:
- `total_transaction_costs`: Sum of all trading costs
- `net_return`: Gross return minus transaction costs
- `average_turnover`: Average portfolio turnover per rebalance
- `trades_per_day`: Average daily trading frequency

#### Scenario: Track transaction costs during backtest

**Given** a strategy that rebalances 10 times in 100 days

**And** average turnover per rebalance: 0.20 (20% of portfolio)

**And** transaction cost: 10 bps per trade

**When** calculating total costs

**Then** cost per rebalance SHALL be 0.20 × 0.001 = 0.0002 = 20 bps

**And** total cost SHALL be 10 × 20 bps = 200 bps = 2.0%

**And** trades per day SHALL be 10 / 100 = 0.10

**And** net return SHALL be gross_return - 2.0%

#### Scenario: Report transaction costs in metrics

**Given** a backtest with:
- Gross return: 15%
- Total transaction costs: 2.5%
- 12 rebalances in 180 days

**When** generating performance metrics

**Then** metrics SHALL include:
- `total_return`: 15%
- `net_return`: 12.5% (15% - 2.5%)
- `total_transaction_costs`: 2.5%
- `trades_per_day`: 12 / 180 = 0.067

**And** net Sharpe ratio SHALL be calculated using net returns

---

### Requirement: Smart Rebalancing Frequency

Strategies SHALL adapt rebalancing frequency based on market conditions (OPTIONAL enhancement).

**Acceptance Criteria**:
- Volatility-based rebalancing frequency
- Higher volatility → less frequent rebalancing
- Lower volatility → potentially more frequent rebalancing
- Minimum rebalancing interval enforced (default: 7 days)

**Parameters**:
- `min_rebalance_days`: Minimum days between rebalances (default: 7)
- `adaptive_rebalancing`: Enable adaptive frequency (default: False)

**Status**: OPTIONAL (not required for initial implementation)

#### Scenario: Skip rebalancing within minimum interval

**Given** last rebalance occurred 5 days ago

**And** minimum rebalancing interval is 7 days

**When** evaluating rebalancing decision

**Then** rebalancing SHALL be skipped (5 < 7 days)

**And** previous weights SHALL be returned

**And** interval check SHALL occur before turnover calculation

---

## Integration Points

**Consumers**:
- All portfolio strategies (HRP, Risk Parity, Black-Litterman, Copula Pairs)

**Dependencies**:
- PyPortfolioOpt >= 1.5.0 (transaction_cost objective)
- NumPy (array operations)

**Affected Modules**:
- `src/crypto_trader/strategies/library/hierarchical_risk_parity.py`
- `src/crypto_trader/strategies/library/risk_parity.py`
- `src/crypto_trader/strategies/library/black_litterman.py`
- `src/crypto_trader/strategies/library/copula_pairs_trading.py`
- `src/crypto_trader/backtesting/engine.py` (cost tracking)
- `src/crypto_trader/analysis/metrics.py` (new metrics)

**Module Location**:
- `src/crypto_trader/optimization/transaction_costs.py`

**Tests**:
- `tests/crypto_trader/optimization/test_transaction_costs.py`

---

## Expected Impact

**Trading Frequency Reduction**:
- Current: 0.11 trades/day
- Target: 0.07 trades/day
- Reduction: 36%

**Cost Savings**:
- Annual cost (current): ~4% of returns
- Annual cost (target): ~2.4% of returns
- Savings: ~1.6% per year

**Net Return Improvement**:
- Additional 1.6% annual return from cost savings
- Sharpe improvement: +0.10 (from reduced denominator)

---

## Performance Considerations

**Computational Cost**:
- Turnover calculation: <1ms per rebalance check
- PyPortfolioOpt optimization with cost penalty: +10ms per optimization
- Weight tracking: negligible memory overhead

**Total Overhead**: ~10ms per rebalancing decision
- Negligible compared to overall backtest runtime

---

## Validation Strategy

**Unit Tests**:
- Test turnover calculation accuracy
- Test rebalancing threshold logic
- Test previous weights tracking
- Test edge cases (no previous weights, zero turnover)

**Integration Tests**:
- Test with real portfolio strategies
- Verify trading frequency reduction
- Compare net returns with/without optimization

**Validation Function**:
```python
if __name__ == "__main__":
    # Test rebalancing decision
    current = {"BTC": 0.5, "ETH": 0.5}
    target = {"BTC": 0.52, "ETH": 0.48}

    should_rebalance, turnover = should_rebalance(
        current_weights=current,
        target_weights=target,
        transaction_cost_pct=0.001,
        min_benefit_pct=0.005
    )

    print(f"Turnover: {turnover:.4f}")
    print(f"Should rebalance: {should_rebalance}")

    # Verify logic
    assert turnover == 0.04
    assert not should_rebalance  # 4 bps < 50 bps threshold

    print("✅ Transaction cost optimization passes validation")
```

---

## References

### Research Papers
- French, K. (2008). "Presidential Address: The Cost of Active Investing"
- Balduzzi, P. & Lynch, A. (1999). "Transaction Costs and Predictability"

### PyPortfolioOpt Documentation
- Transaction costs: https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html#transaction-costs
- Objectives: https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimizers.html

---

**Status**: PROPOSED
**Impact**: MODERATE (modifies all portfolio strategies, adds metrics)
**Breaking Changes**: None (backward-compatible parameters)
