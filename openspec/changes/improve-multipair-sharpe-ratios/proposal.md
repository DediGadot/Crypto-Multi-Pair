# Proposal: Improve Multi-Pair Strategy Sharpe Ratios

## Why

Analysis of 2,754 windowed backtest results reveals multi-pair portfolio strategies (HRP, Risk Parity, Black-Litterman, Copula Pairs) are failing with average Sharpe of -0.002, 24% win rate, and 0% success rate above 0.5 Sharpe. Root causes: missing risk management (position sizing, stop losses), poor covariance estimation with noisy crypto data, backward-looking volatility, and excessive rebalancing costs. Top performing simple strategy (BuyAndHold) achieves only 0.58 Sharpe, indicating significant room for improvement through systematic risk management and better estimation techniques.

## What Changes

- **Phase 1: Risk Management** - Implement Kelly Criterion position sizing (25% fractional, 2-15% limits), 8% trailing stop losses with ATR adjustment, portfolio correlation limits (0.70 max), and 15% max drawdown with dynamic position adjustment
- **Phase 2: Estimation Improvements** - Replace sample covariance with Ledoit-Wolf shrinkage in all portfolio strategies; implement GARCH(1,1) volatility forecasting with validation and fallback to sample volatility
- **Phase 3: Transaction Cost Optimization** - Add rebalancing thresholds (minimum 50 bps benefit), transaction cost penalties in optimization objectives, target 40% reduction in trading frequency (0.11 → 0.07 trades/day)
- **New Modules**: `risk/position_sizing.py`, `risk/stop_losses.py`, `risk/volatility_forecasting.py`, `optimization/transaction_costs.py`
- **Strategy Updates**: Modify `hierarchical_risk_parity.py`, `risk_parity.py`, `black_litterman.py`, `copula_pairs_trading.py` to integrate new risk management and estimation components

## Impact

### Expected Performance
Target improvements: Average Sharpe 0.65+ (from -0.002), Top Strategy Sharpe 1.20+ (from 0.58), Win Rate 55%+ (from 24%), Profit Factor 1.50+ (from 0.88), Max Drawdown <15% (from 7.7%), Trades/Day 0.07 (from 0.11). Phased rollout: Phase 1 (+0.38 Sharpe), Phase 2 (+0.25 Sharpe), Phase 3 (+0.10 Sharpe) = +0.73 total over 3 weeks.

**Affected Specs**:
- `portfolio-optimization` - MODIFIED covariance estimation, transaction costs
- `risk-management` - ADDED position sizing, stop losses, portfolio limits
- `volatility-forecasting` - ADDED GARCH forecasting module
- `transaction-cost-optimization` - ADDED rebalancing thresholds

**Affected Code**:
- Strategy files: `hierarchical_risk_parity.py`, `risk_parity.py`, `black_litterman.py`, `copula_pairs_trading.py` (4 files)
- New modules: `risk/position_sizing.py`, `risk/stop_losses.py`, `risk/volatility_forecasting.py`, `optimization/transaction_costs.py` (4 files)
- Integration: Backtesting engine (risk checks), metrics calculator (new metrics), HTML reporter (visualization)

**Dependencies**: All required packages already available (`arch`, `pypfopt`, `scipy`)

**Validation**: Train/test split methodology via `master_windowed_multipair.py` with 2.0 year test sets. Per-phase validation: Phase 1 (Sharpe >0.3, drawdown <15%, win rate >40%), Phase 2 (Sharpe >0.5, covariance PSD), Phase 3 (Sharpe >0.65, trades/day <0.08, profit factor >1.2).

**References**: Ledoit & Wolf (2004) covariance shrinkage, Lopez de Prado (2016) portfolio optimization, Thorp (2006) Kelly criterion, Engle (1982) GARCH. Implementation uses `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf()` and `pypfopt.objective_functions.transaction_cost()`.
