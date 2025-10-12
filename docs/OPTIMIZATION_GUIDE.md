# Portfolio Optimization Guide

## Overview

The `optimize_portfolio_comprehensive.py` script performs **research-grade parameter optimization** using **walk-forward analysis** to find the best portfolio configuration that generalizes across different market conditions.

## What It Optimizes

### 1. **Asset Selection**
- Which cryptocurrencies to include in your portfolio
- Tests combinations of 2-5 assets from 8-coin universe
- Examples: BTC+ETH, BTC+ETH+SOL, BTC+ETH+SOL+BNB+ADA

### 2. **Weight Allocation**
- How to distribute capital across selected assets
- Tests multiple schemes: Equal weight, descending, concentrated
- Examples: 40/30/15/15, 25/25/25/25, 50/30/20

### 3. **Rebalancing Parameters**
- **Threshold**: When to trigger rebalancing (5%-20% deviation)
- **Method**: Threshold-based, calendar-based, or hybrid
- **Calendar Period**: How often to rebalance (7-90 days)
- **Min Interval**: Minimum hours between rebalances (12-72h)
- **Momentum Filter**: Whether to skip rebalancing during strong trends

## Walk-Forward Analysis Explained

### Why Walk-Forward?

Traditional backtesting can **overfit** to historical data, producing configs that look great in-sample but fail in real trading. Walk-forward analysis prevents this by:

1. **Training** on historical data (in-sample)
2. **Testing** on unseen future data (out-of-sample)
3. **Repeating** with expanding training window

### How It Works

```
Data Timeline: |------Window 1------|------Window 2------|------Window 3------|

Split 1:
  Train: Window 1         →  Test: Window 2

Split 2:
  Train: Windows 1-2      →  Test: Window 3

Split 3:
  Train: Windows 1-2-3    →  Test: Window 4
```

**Key Principle**: You NEVER train on data you're testing on. This simulates real-world forward testing.

## Usage

### Quick Start (Recommended for First Run)

```bash
# Fast test with reduced parameter grid (~15 minutes)
uv run python optimize_portfolio_comprehensive.py --quick
```

### Full Optimization

```bash
# Complete grid search (~2-4 hours)
uv run python optimize_portfolio_comprehensive.py \
  --window-days 365 \
  --test-windows 5 \
  --timeframe 1h
```

### Custom Parameters

```bash
# Shorter windows for faster iteration
uv run python optimize_portfolio_comprehensive.py \
  --window-days 180 \
  --test-windows 4 \
  --timeframe 4h

# Daily timeframe (faster but less data)
uv run python optimize_portfolio_comprehensive.py \
  --timeframe 1d \
  --window-days 365
```

## Output Files

After optimization completes, you'll get:

### 1. `OPTIMIZATION_REPORT.txt` ⭐ **Start Here**

Research-grade report with:

- **TL;DR Section**: Executive summary with recommended config and expected performance
- **Detailed Analysis**: Top 5 configurations, parameter sensitivity, statistical tests
- **Recommendations**: Deployment readiness, risk management, monitoring guidelines

### 2. `optimized_config.yaml`

Ready-to-use configuration file for the best-performing strategy. Use it directly:

```bash
uv run python run_full_pipeline.py \
  --portfolio \
  --config optimization_results/optimized_config.yaml \
  --report
```

### 3. `optimization_results_YYYYMMDD_HHMM.csv`

Detailed results for all tested configurations. Columns include:

- `test_outperformance`: Out-of-sample performance vs buy-and-hold ⭐
- `test_return`: Average return across test periods
- `test_sharpe`: Risk-adjusted return
- `test_win_rate`: Percentage of test periods that beat buy-and-hold
- `generalization_gap`: Difference between train and test performance

## Understanding the Results

### Key Metrics

#### 1. **Test Outperformance** (Primary Metric)
- How much the strategy beats buy-and-hold in unseen data
- Example: `+12.5%` means you earn 12.5% more than just holding
- **Higher is better**, but watch for overfitting

#### 2. **Test Win Rate**
- Percentage of test periods where strategy beat buy-and-hold
- Example: `80%` means won in 4 out of 5 test periods
- **Aim for >60%** for robust strategies

#### 3. **Generalization Gap**
- Difference between training and testing performance
- Example: `+2.1%` means strategy performs 2.1% worse in real test vs training
- **Smaller absolute value is better** (ideally <5%)

#### 4. **Consistency (σ)**
- Standard deviation of outperformance across test periods
- Example: `±3.2%` means performance varies by ±3.2% between periods
- **Lower is better** for predictable results

### Quality Indicators

#### ✅ Excellent Strategy
- Test Win Rate: ≥80%
- Generalization Gap: <5%
- Test Outperformance: >10%
- **Action**: Deploy with full position sizing

#### ✓ Good Strategy
- Test Win Rate: 60-80%
- Generalization Gap: 5-10%
- Test Outperformance: >5%
- **Action**: Deploy with 50-75% position sizing

#### ⚠ Marginal Strategy
- Test Win Rate: 40-60%
- Generalization Gap: 10-15%
- **Action**: Paper trade first, use 25-50% sizing

#### ❌ Poor Strategy
- Test Win Rate: <40%
- Generalization Gap: >15%
- **Action**: Do not deploy, re-optimize

## Example Workflow

### Step 1: Run Optimization

```bash
# Start with quick mode
uv run python optimize_portfolio_comprehensive.py --quick

# If results look promising, run full optimization
uv run python optimize_portfolio_comprehensive.py
```

### Step 2: Review TL;DR

Open `optimization_results/OPTIMIZATION_REPORT.txt` and read the **TL;DR section**:

```
TL;DR - EXECUTIVE SUMMARY
====================================

🎯 RECOMMENDED CONFIGURATION:
   Assets: BTC/USDT + ETH/USDT + SOL/USDT + BNB/USDT
   Allocation: BTC/USDT=40%, ETH/USDT=30%, SOL/USDT=15%, BNB/USDT=15%
   Rebalance: Threshold method, 10% threshold

📈 EXPECTED PERFORMANCE (Out-of-Sample):
   Outperforms Buy-and-Hold by: 8.11% per year
   Average Return: 74.93%
   Risk-Adjusted (Sharpe): 2.15
   Win Rate: 80% (won in 4/5 test periods)

⚠️  KEY RISKS:
   Average Drawdown: -15.3%
   Performance Consistency (σ): ±3.2%
   Generalization Gap: +2.1% (✓ Low - good generalization)

🔬 ROBUSTNESS ASSESSMENT:
   Status: ✅ HIGHLY ROBUST - Consistent out-of-sample performance
```

### Step 3: Validate with Full Backtest

```bash
# Run complete backtest with optimized config
uv run python run_full_pipeline.py \
  --portfolio \
  --config optimization_results/optimized_config.yaml \
  --report
```

### Step 4: Compare Results

Check if the full backtest results match optimization expectations:

- **Total Return**: Should be similar to test_avg_return
- **Outperformance**: Should match test_avg_outperformance (±5%)
- **Rebalance Frequency**: Should match expected behavior

### Step 5: Deploy or Iterate

**If results match**:
- Deploy with appropriate position sizing based on robustness rating
- Monitor actual performance vs expected

**If results don't match**:
- Check for data issues or configuration errors
- Consider re-optimizing with different parameters
- Review the detailed analysis section of the report

## Advanced Usage

### Customizing the Search Space

Edit the script to modify what's being tested:

#### Add More Assets

```python
def get_asset_universe(self) -> List[str]:
    return [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "AVAX/USDT",  # Add new assets
        "LINK/USDT",
    ]
```

#### Test Different Thresholds

```python
def get_rebalancing_parameters(self) -> Dict[str, List]:
    return {
        'threshold': [0.03, 0.05, 0.08, 0.10, 0.15],  # Custom thresholds
        ...
    }
```

### Optimization Strategies

#### Conservative (Risk-Averse)
- Optimize for **Sharpe Ratio** instead of outperformance
- Focus on configurations with low drawdown
- Prefer higher test win rates (>75%)

#### Aggressive (Return-Focused)
- Optimize for **Total Return**
- Accept higher drawdowns and volatility
- Focus on absolute performance

#### Balanced (Recommended)
- Optimize for **Outperformance** (default)
- Filter for reasonable drawdown (<25%)
- Require decent win rate (>60%)

## Performance Expectations

### Realistic Goals

Based on research and backtesting:

- **Good Portfolio**: 5-15% outperformance vs buy-and-hold
- **Great Portfolio**: 15-30% outperformance
- **Exceptional Portfolio**: >30% outperformance

### Warning Signs

- **Too good to be true**: >50% outperformance → likely overfit
- **High gap**: Train 40%, Test 10% → severe overfitting
- **Low win rate**: <40% → unreliable strategy

## Troubleshooting

### Issue: Optimization Takes Too Long

**Solutions**:
1. Use `--quick` mode
2. Reduce `--test-windows` to 3
3. Use `--timeframe 4h` or `1d` instead of `1h`
4. Edit script to reduce asset combinations

### Issue: Poor Out-of-Sample Results

**Causes**:
- Market regime changed (bull → bear)
- Parameters overfit to specific conditions
- Insufficient test windows

**Solutions**:
1. Increase `--test-windows` to 7-10
2. Use longer windows (365 days minimum)
3. Test across bull/bear/sideways markets

### Issue: All Configs Perform Similarly

**Meaning**: Parameters don't matter much, market dominates

**Action**:
- Consider simpler strategies
- Focus on asset selection over rebalancing
- Check if buy-and-hold beats most configs

## Best Practices

### 1. **Always Use Walk-Forward**
Never trust results from simple backtesting. Walk-forward is the gold standard.

### 2. **Check Generalization Gap**
If train performance >> test performance, you're overfitting.

### 3. **Require Multiple Wins**
A strategy that wins in 1/5 test periods isn't robust. Aim for 3/5 or better.

### 4. **Start Conservative**
Use 50% position sizing initially, even with excellent results.

### 5. **Re-Optimize Regularly**
Markets change. Re-run optimization every 3-6 months.

### 6. **Document Everything**
Save all reports. Compare new optimizations to old ones.

## Theory: Why This Works

### Mean Reversion at Portfolio Level

Rebalancing exploits mean reversion:
1. Asset A outperforms → becomes overweight → SELL
2. Asset B underperforms → becomes underweight → BUY
3. When assets revert to mean → you profit

### Systematic "Buy Low, Sell High"

- **Threshold Method**: Rebalance when weights drift
- **Calendar Method**: Rebalance on fixed schedule
- **Hybrid Method**: Best of both worlds

### Why Walk-Forward Prevents Overfitting

- **In-sample**: Easy to find patterns (even random noise)
- **Out-of-sample**: True test of strategy's edge
- **Walk-forward**: Simulates real forward testing

## References

### Academic Research
- "Portfolio Rebalancing" by Tsai (2001)
- "The Rebalancing Bonus" by Willenbrock (2011)
- Walk-forward analysis: Pardo (2008)

### Related Documentation
- [README.md](../README.md): Overview of full system
- [HOW_TO_RUN_PORTFOLIO_STRATEGY.md](HOW_TO_RUN_PORTFOLIO_STRATEGY.md): Running portfolios
- [PORTFOLIO_REBALANCING_ANALYSIS.md](PORTFOLIO_REBALANCING_ANALYSIS.md): Theory deep-dive

## Next Steps

1. **Run quick optimization**: `uv run python optimize_portfolio_comprehensive.py --quick`
2. **Read TL;DR**: Review `optimization_results/OPTIMIZATION_REPORT.txt`
3. **Validate**: Backtest with `run_full_pipeline.py --portfolio`
4. **Deploy carefully**: Start with small position sizes
5. **Monitor**: Track actual vs expected performance

## Support

For issues or questions:
- Check the research report's recommendations section
- Review parameter sensitivity analysis
- Compare your results to the example in this guide
- Consult academic papers on walk-forward analysis

---

**Remember**: Past performance doesn't guarantee future results. Always use proper risk management and never invest more than you can afford to lose.
