# Quick Start Guide: Advanced Risk Metrics

## 5-Minute Overview

The crypto trading system now includes five advanced risk metrics that help you understand your strategy's risk profile beyond basic returns and drawdowns.

## The Five Metrics

### 1. Value at Risk (VaR) - "What's my worst-case loss?"
**What it tells you**: Maximum expected loss at 95% confidence
**Example**: VaR of 2% means there's a 5% chance you'll lose more than 2%
**Good value**: < 2% (low risk), 2-5% (moderate), > 5% (high risk)

### 2. Conditional VaR (CVaR) - "How bad can it get?"
**What it tells you**: Average loss when things go really wrong
**Example**: CVaR of 3% means when you're in the worst 5% of outcomes, you lose 3% on average
**Good value**: Close to VaR (contained risk), much higher than VaR (dangerous tails)

### 3. Skewness - "Do I have more big wins or big losses?"
**What it tells you**: Whether your returns are symmetric or lopsided
**Example**: Skewness of +0.8 means you have occasional big wins (good!)
**Good value**: Positive (more big wins), Zero (symmetric), Negative (more big losses - bad!)

### 4. Kurtosis - "How often do extreme events happen?"
**What it tells you**: Probability of extreme returns
**Example**: Kurtosis of 3.0 means fat tails with more extreme events than normal
**Good value**: Depends on risk tolerance. High = more extreme events (good and bad)

### 5. Information Ratio - "Am I beating the benchmark efficiently?"
**What it tells you**: How much excess return per unit of risk vs benchmark
**Example**: IR of 1.2 means excellent risk-adjusted outperformance
**Good value**: > 1.0 (excellent), > 0.5 (good), > 0 (moderate)

## Quick Usage

```python
from crypto_trader.analysis.metrics import MetricsCalculator

# Create calculator
calculator = MetricsCalculator(risk_free_rate=0.02)

# Calculate all metrics (including advanced)
metrics = calculator.calculate_all_metrics(
    returns=your_returns,
    trades=your_trades,
    equity_curve=your_equity_curve,
    initial_capital=10000.0
)

# Access advanced metrics
print(f"VaR 95%: {metrics.value_at_risk_95:.2%}")
print(f"CVaR 95%: {metrics.conditional_var_95:.2%}")
print(f"Skewness: {metrics.skewness:.4f}")
print(f"Kurtosis: {metrics.kurtosis:.4f}")
print(f"Information Ratio: {metrics.information_ratio:.4f}")
```

## Real-World Examples

### Example 1: Conservative Strategy
```
VaR: 1.5%      ← Low risk
CVaR: 1.8%     ← Contained tail risk (CVaR/VaR = 1.2)
Skewness: 0.3  ← Slight positive skew (a few big wins)
Kurtosis: 0.5  ← Normal tails
IR: 1.2        ← Excellent risk-adjusted returns
```
**Interpretation**: Great for risk-averse investors. Low risk, occasional big wins, good returns.

### Example 2: Aggressive Strategy
```
VaR: 4.5%      ← High risk
CVaR: 6.2%     ← Dangerous tails (CVaR/VaR = 1.4)
Skewness: -0.8 ← More big losses than wins (bad!)
Kurtosis: 3.2  ← Fat tails, many extreme events
IR: 0.8        ← Moderate risk-adjusted returns
```
**Interpretation**: High risk with unfavorable distribution. Only for very risk-tolerant investors.

### Example 3: Ideal Strategy
```
VaR: 2.0%      ← Moderate risk
CVaR: 2.3%     ← Controlled tails (CVaR/VaR = 1.15)
Skewness: 0.8  ← More big wins (great!)
Kurtosis: -0.5 ← Thin tails, fewer extreme events
IR: 1.5        ← Excellent outperformance
```
**Interpretation**: Perfect risk profile. Moderate risk, positive skew, excellent returns.

## Decision Matrix

| If you see... | It means... | Action |
|---------------|-------------|---------|
| VaR > 5% | High risk strategy | Reduce position size or use stops |
| CVaR >> VaR (ratio > 1.5) | Dangerous tail risk | Add tail risk hedges |
| Negative skewness | More big losses | Avoid or reduce exposure |
| High kurtosis (> 3) | Many extreme events | Expect volatility, size accordingly |
| IR < 0.5 | Poor risk-adjusted returns | Compare to other strategies |
| Positive skew + Low VaR | Ideal profile | Increase allocation |

## Common Patterns

### Pattern 1: "Volatility Seller"
- Negative skewness (big losses)
- High kurtosis (extreme events)
- Low VaR most of the time, but dangerous CVaR
- **Action**: Use strict position limits

### Pattern 2: "Trend Follower"
- Positive skewness (big wins)
- Moderate kurtosis
- Higher VaR but controlled CVaR
- **Action**: Let winners run, cut losers quickly

### Pattern 3: "Mean Reversion"
- Near-zero skewness (symmetric)
- Low kurtosis (normal distribution)
- Low VaR and CVaR
- **Action**: Stable strategy, good for consistent returns

## Position Sizing with VaR

```python
# Example: Target 2% max portfolio loss at 95% confidence
target_portfolio_var = 0.02

# Your strategy has 5% VaR
strategy_var = metrics.value_at_risk_95  # 0.05

# Position size to meet target
position_fraction = target_portfolio_var / strategy_var
# = 0.02 / 0.05 = 0.4 (40% of capital)

print(f"Allocate {position_fraction:.1%} of capital to this strategy")
```

## Red Flags 🚩

Watch out for these warning signs:

1. **CVaR > 2 × VaR**: Extreme tail risk
2. **Negative skewness < -1.0**: Very unfavorable distribution
3. **Kurtosis > 5.0**: Expect many extreme events
4. **IR < 0**: Underperforming benchmark
5. **VaR increasing over time**: Risk is growing

## Quick Checks

Before deploying a strategy, verify:

- ✅ VaR is acceptable for your risk tolerance
- ✅ CVaR/VaR ratio < 1.5 (controlled tails)
- ✅ Skewness is not highly negative
- ✅ Information Ratio > 0.5
- ✅ Overall risk profile matches your goals

## Further Reading

- **Full Documentation**: `/home/fiod/crypto/docs/ADVANCED_RISK_METRICS.md`
- **Demo Script**: `/home/fiod/crypto/demo_advanced_risk_metrics.py`
- **Integration Test**: `/home/fiod/crypto/test_advanced_metrics_integration.py`

## Support

All metrics are automatically calculated when you use `calculate_all_metrics()`. No extra configuration needed!

The metrics handle edge cases automatically:
- Empty returns → 0.0
- Insufficient data → 0.0
- Invalid values → 0.0

Just run your backtest and the metrics will be there.
