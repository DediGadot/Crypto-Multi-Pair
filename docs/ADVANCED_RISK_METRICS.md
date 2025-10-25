# Advanced Risk Metrics Documentation

This document describes the advanced risk metrics implemented in the crypto trading system for comprehensive risk assessment and portfolio analysis.

## Overview

The system now includes five advanced risk metrics that provide deeper insights into strategy performance beyond traditional metrics like Sharpe ratio and maximum drawdown:

1. **Value at Risk (VaR)** - Maximum expected loss at 95% confidence
2. **Conditional Value at Risk (CVaR)** - Expected loss beyond VaR threshold
3. **Skewness** - Return distribution asymmetry
4. **Kurtosis** - Return distribution tail risk
5. **Information Ratio** - Risk-adjusted excess return vs benchmark

## Metrics Details

### Value at Risk (VaR)

**Definition**: The maximum expected loss over a given time period at a specified confidence level (95%).

**Interpretation**:
- VaR of 2% at 95% confidence means there's a 5% chance of losing more than 2% in a period
- Lower VaR indicates lower risk
- VaR < 2%: Low risk
- VaR 2-5%: Moderate risk
- VaR > 5%: High risk

**Calculation**:
```python
var_95 = calculator.value_at_risk(returns, confidence=0.95)
```

**Use Case**:
- Position sizing based on risk tolerance
- Regulatory compliance (Basel III)
- Setting stop-loss levels

### Conditional Value at Risk (CVaR)

**Definition**: The expected loss given that the loss exceeds VaR. Also known as Expected Shortfall.

**Interpretation**:
- CVaR provides the average magnitude of losses in the worst 5% of cases
- Always >= VaR (measures tail severity)
- CVaR/VaR ratio indicates tail risk:
  - < 1.2: Contained tail risk
  - 1.2-1.5: Moderate tail risk
  - > 1.5: Significant tail risk

**Calculation**:
```python
cvar_95 = calculator.conditional_var(returns, confidence=0.95)
```

**Use Case**:
- Better tail risk assessment than VaR alone
- Stress testing
- Risk-adjusted position sizing

### Skewness

**Definition**: Measures the asymmetry of the return distribution.

**Interpretation**:
- Positive skew (> 0.5): More large gains than losses - **FAVORABLE**
  - Strategy has occasional big wins
  - Desirable for trend-following strategies
- Negative skew (< -0.5): More large losses than gains - **UNFAVORABLE**
  - Strategy has occasional big losses
  - Common in volatility selling strategies
- Near zero (-0.5 to 0.5): Symmetric distribution

**Calculation**:
```python
skew = calculator.skewness(returns)
```

**Use Case**:
- Strategy selection and comparison
- Understanding return distribution characteristics
- Adjusting position sizes based on distribution shape

### Kurtosis

**Definition**: Measures the "tailedness" or extreme event probability of the return distribution.

**Interpretation**:
- Positive excess kurtosis (> 1.0): Fat tails - **HIGH TAIL RISK**
  - Higher probability of extreme returns (both gains and losses)
  - Requires more conservative position sizing
- Near zero (-1.0 to 1.0): Normal distribution tails
- Negative excess kurtosis (< -1.0): Thin tails - **LOW TAIL RISK**
  - Lower probability of extreme events

**Calculation**:
```python
kurt = calculator.kurtosis(returns)
```

**Use Case**:
- Tail risk assessment
- Stress testing scenarios
- Determining appropriate risk limits

### Information Ratio

**Definition**: Measures the risk-adjusted returns of a strategy relative to a benchmark.

**Formula**: `(Portfolio Return - Benchmark Return) / Tracking Error`

**Interpretation**:
- IR > 1.0: Excellent risk-adjusted outperformance
- IR > 0.5: Good risk-adjusted outperformance
- IR > 0: Moderate outperformance
- IR < 0: Underperformance

**Calculation**:
```python
# Compare to specific benchmark
ir = calculator.information_ratio(strategy_returns, benchmark_returns)

# Compare to cash (zero returns)
ir = calculator.information_ratio(strategy_returns, None)
```

**Use Case**:
- Strategy comparison against benchmarks
- Manager skill assessment
- Portfolio optimization

## Integration with Existing Metrics

All advanced metrics are automatically calculated when using `calculate_all_metrics()`:

```python
from crypto_trader.analysis.metrics import MetricsCalculator

calculator = MetricsCalculator(risk_free_rate=0.02)

metrics = calculator.calculate_all_metrics(
    returns=returns,
    trades=trades,
    equity_curve=equity_curve,
    initial_capital=initial_capital,
)

# Access advanced metrics
print(f"VaR 95%: {metrics.value_at_risk_95:.2%}")
print(f"CVaR 95%: {metrics.conditional_var_95:.2%}")
print(f"Skewness: {metrics.skewness:.4f}")
print(f"Kurtosis: {metrics.kurtosis:.4f}")
print(f"Information Ratio: {metrics.information_ratio:.4f}")
```

## Risk Profile Assessment

Combine metrics for comprehensive risk assessment:

### Low-Risk Profile
- VaR < 2%
- Positive skewness
- Low to moderate kurtosis
- High Information Ratio (> 1.0)
- Sharpe Ratio > 2.0

### Moderate-Risk Profile
- VaR 2-5%
- Near-zero skewness
- Moderate kurtosis
- Moderate Information Ratio (0.5-1.0)
- Sharpe Ratio 1.0-2.0

### High-Risk Profile
- VaR > 5%
- Negative skewness
- High kurtosis (fat tails)
- Low Information Ratio (< 0.5)
- Sharpe Ratio < 1.0

## Practical Examples

### Example 1: Comparing Two Strategies

```python
# Strategy A: Conservative
strategy_a_metrics = calculate_all_metrics(...)
# VaR: 1.5%, CVaR: 1.8%, Skewness: 0.3, Kurtosis: 0.5, IR: 1.2

# Strategy B: Aggressive
strategy_b_metrics = calculate_all_metrics(...)
# VaR: 4.5%, CVaR: 6.2%, Skewness: -0.8, Kurtosis: 3.2, IR: 0.8

# Analysis:
# Strategy A: Lower tail risk, positive skew, normal tails, better IR
# Strategy B: Higher tail risk, negative skew, fat tails, lower IR
# Choose A for risk-averse portfolio, B for higher return potential with risk tolerance
```

### Example 2: Position Sizing Based on VaR

```python
# Target maximum portfolio loss: 2% at 95% confidence
target_var = 0.02

# Strategy has VaR of 5%
strategy_var = 0.05

# Position size to meet target
position_fraction = target_var / strategy_var  # 0.4 or 40% of capital
```

### Example 3: Detecting Strategy Regime Changes

```python
# Monitor skewness and kurtosis over rolling windows
# Sudden changes indicate regime shifts:
# - Skewness turning negative: Strategy entering drawdown period
# - Kurtosis increasing: Entering volatile period with more extreme events
```

## Edge Cases and Limitations

### Minimum Data Requirements
- **VaR/CVaR**: At least 20 returns for meaningful calculation
- **Skewness**: At least 3 returns (preferably 30+)
- **Kurtosis**: At least 4 returns (preferably 50+)
- **Information Ratio**: Returns and benchmark must have same length

### Edge Case Handling
All metrics return 0.0 for:
- Empty return series
- Insufficient data points
- Invalid inputs (NaN, Inf)

### Limitations
1. **VaR/CVaR**: Based on historical data, may not predict future tail events
2. **Skewness**: Sensitive to outliers, requires sufficient data
3. **Kurtosis**: Requires large sample size for stability
4. **Information Ratio**: Sensitive to benchmark choice

## Best Practices

1. **Use Multiple Metrics**: Don't rely on a single risk metric
2. **Regular Monitoring**: Recalculate metrics periodically to detect regime changes
3. **Stress Testing**: Simulate extreme scenarios beyond historical data
4. **Combine with Qualitative Analysis**: Metrics don't capture all risks
5. **Benchmark Selection**: Choose appropriate benchmark for Information Ratio
6. **Rolling Windows**: Calculate metrics over different time windows

## References

- **VaR/CVaR**: Basel Committee on Banking Supervision - Market Risk Framework
- **Skewness/Kurtosis**: Cont, R. (2001). "Empirical properties of asset returns"
- **Information Ratio**: Grinold & Kahn (1999). "Active Portfolio Management"
- **SciPy Documentation**: https://docs.scipy.org/doc/scipy/reference/stats.html
- **NumPy Documentation**: https://numpy.org/doc/stable/

## Code References

- **Types Definition**: `/home/fiod/crypto/src/crypto_trader/core/types.py`
- **Metrics Implementation**: `/home/fiod/crypto/src/crypto_trader/analysis/metrics.py`
- **Demo Script**: `/home/fiod/crypto/demo_advanced_risk_metrics.py`
