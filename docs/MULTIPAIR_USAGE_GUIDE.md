# Multi-Pair Trading Analysis - Usage Guide

## Overview

This guide explains how to run multi-pair trading strategy analysis with the crypto trading system.

## Two Approaches Available

### Approach 1: Existing Multi-Pair System (master.py)
**Status:** Working now, generates HTML/text reports
**Methodology:** Single recent window per horizon (traditional approach)
**Advantages:** Ready to use immediately, includes HTML visualizations
**Limitations:** No train/test split, no windowed analysis, no timestamp fix

### Approach 2: Windowed Multi-Pair System (In Development)
**Status:** Core components built (multipair_window_manager.py validated)
**Methodology:** Train/test split with non-overlapping windows
**Advantages:** Proper ML methodology, overfitting detection, statistical confidence
**Status:** Requires integration work to complete

---

## Using Approach 1: Existing Multi-Pair System

### Quick Test (Recommended First Step)
```bash
uv run python master.py --multi-pair --quick --workers 2
```

This will:
- Test multi-pair strategies (Portfolio, StatArb, Risk Parity, etc.)
- Use 3 horizons: 30d, 90d, 180d
- Fetch data for BTC/USDT, ETH/USDT, BNB/USDT
- Generate HTML and text reports

### Full Analysis
```bash
uv run python master.py --multi-pair \
  --symbol BTC/USDT \
  --timeframe 1h \
  --horizons 30 90 180 365 \
  --workers 4 \
  --output multi_pair_full
```

### Output Files
```
multi_pair_full_YYYYMMDD_HHMMSS/
├── master_analysis.log          # Detailed execution logs
├── performance_metrics.csv       # All strategy results
├── REPORT.txt                    # Human-readable text report
├── REPORT.html                   # Interactive HTML report
└── detailed_results/             # Per-strategy detailed results
```

### Multi-Pair Strategies Tested

The system automatically tests these multi-pair strategies:

1. **PortfolioRebalancer** - Threshold-based rebalancing
2. **StatisticalArbitrage** - Cointegration-based pairs trading
3. **HierarchicalRiskParity** - HRP portfolio optimization
4. **BlackLitterman** - Bayesian portfolio with views
5. **RiskParity** - Equal risk contribution
6. **CopulaPairsTrading** - Tail dependency modeling
7. **DeepRLPortfolio** - Deep RL with PPO agent

### Asset Combinations

The system tests multiple asset combinations:
- 2-asset: BTC/USDT + ETH/USDT
- 3-asset: BTC/USDT + ETH/USDT + BNB/USDT
- Plus additional pairs based on correlation analysis

---

## Using Approach 2: Windowed Multi-Pair (When Ready)

### Command Structure (Future)
```bash
uv run python master_windowed_multipair.py \
  --pairs BTC/USDT ETH/USDT \
  --quick \
  --test-years 2.0 \
  --workers 2
```

### What Makes It Better

1. **Train/Test Split**
   - Training: All data before (runtime_date - 2 years)
   - Test: Last 2 years
   - Prevents lookahead bias

2. **Non-Overlapping Windows**
   - Multiple independent samples per horizon
   - Example: 30d horizon → 12 train windows + 24 test windows

3. **Statistical Aggregation**
   - Mean, median, std dev across windows
   - Percentiles (25th, 75th)
   - Consistency score
   - Weighted average (recent windows weighted more)

4. **Overfitting Detection**
   - Compare train vs test performance
   - Gap = Test Sharpe - Train Sharpe
   - Gap > 0 = Good generalization
   - Gap < 0 = Overf

itting

### Current Status

**✅ Completed:**
- `multipair_window_manager.py` - Synchronized window generation (validated)
- Core windowed analysis system (`master_windowed.py`) with timestamp fix

**🚧 In Progress:**
- `multipair_aggregator.py` - Cross-pair statistics
- `master_windowed_multipair.py` - Main entry point
- HTML report generation with visualizations

---

## Debugging HTML Reports with Chrome DevTools

### Step 1: Generate HTML Report
```bash
# Using existing system
uv run python master.py --multi-pair --quick
```

### Step 2: Open in Chrome
```bash
# Find the latest results directory
LATEST=$(ls -td multi_pair_*/ | head -1)
google-chrome "${LATEST}REPORT.html"
```

Or use the Chrome DevTools MCP:
```bash
# This will be used to verify the HTML report renders correctly
# Check for JavaScript errors, broken visualizations, etc.
```

### Step 3: Common Issues to Check

1. **JavaScript Errors**
   - Open DevTools (F12)
   - Check Console tab for errors
   - Common: Plotly not loaded, data format issues

2. **Missing Visualizations**
   - Check if chart divs are empty
   - Verify data is being passed correctly
   - Look for CSS display: none issues

3. **Performance Issues**
   - Too much data causing slow rendering
   - Large tables not paginated
   - Heavy computations in browser

### Step 4: Iterate Until Perfect
- Fix errors one by one
- Rerun analysis
- Verify in Chrome
- Repeat until all visualizations work

---

## Comparison: Single-Pair vs Multi-Pair

| Aspect | Single-Pair | Multi-Pair |
|--------|-------------|------------|
| **Strategies** | 15+ trend/momentum/ML | 7 portfolio/arbitrage |
| **Data Fetching** | 1 symbol | 3+ symbols |
| **Execution Time** | ~5-10 minutes (quick) | ~15-30 minutes (quick) |
| **Output** | Per-strategy metrics | Portfolio metrics + correlations |
| **Use Case** | Individual asset trading | Portfolio management |

---

## Best Practices

### 1. Start with Quick Mode
Always test with `--quick` first:
```bash
uv run python master.py --multi-pair --quick
```

### 2. Check Logs for Errors
```bash
tail -100 multi_pair_*/master_analysis.log | grep ERROR
```

### 3. Verify Data Quality
- Check that all symbols have sufficient data
- Verify date ranges align across pairs
- Look for missing data warnings

### 4. Resource Management
- Use fewer workers for multi-pair (2-4 recommended)
- Multi-pair uses more memory (3+ DataFrames)
- Monitor with `htop` during execution

### 5. Interpret Results
- Portfolio strategies need minimum data across ALL pairs
- Cointegration tests require sufficient history
- Sharpe ratios may be lower (diversification effect)

---

## Troubleshooting

### "Broken pipe" Error
**Cause:** Output was piped to `head` or similar
**Fix:** Run without output truncation

### "No data for symbol X"
**Cause:** Symbol not available or insufficient history
**Fix:** Check symbol exists on Binance, increase `--max-days`

### "Multi-pair strategies returned empty results"
**Cause:** Insufficient overlap in date ranges across pairs
**Fix:** Ensure all pairs have data for the same time period

### Slow Execution
**Cause:** Too many workers or complex strategies
**Fix:** Reduce `--workers` to 2, use `--quick` mode

### HTML Report Not Rendering
**Cause:** JavaScript errors or missing dependencies
**Fix:** Open Chrome DevTools, check Console for errors

---

## Migration Path: Old → New System

When the windowed multi-pair system is ready:

1. **Run Both Systems Side-by-Side**
   ```bash
   # Old system (for immediate results)
   uv run python master.py --multi-pair --quick

   # New system (for proper methodology)
   uv run python master_windowed_multipair.py --quick
   ```

2. **Compare Results**
   - Old system: Single window, quick feedback
   - New system: Statistical confidence, train/test validation

3. **Gradual Migration**
   - Use old system for rapid prototyping
   - Use new system for final analysis before live trading
   - Trust new system's overfitting detection

---

## Example Workflows

### Workflow 1: Quick Multi-Pair Test
```bash
# 1. Run quick analysis
uv run python master.py --multi-pair --quick --workers 2

# 2. Check results
LATEST=$(ls -td multi_pair_*/ | head -1)
cat "${LATEST}REPORT.txt" | head -100

# 3. Open HTML report
google-chrome "${LATEST}REPORT.html"
```

### Workflow 2: Full Multi-Pair Analysis
```bash
# 1. Run full analysis with custom horizons
uv run python master.py --multi-pair \
  --horizons 30 60 90 180 365 \
  --workers 4 \
  --output portfolio_analysis

# 2. Wait for completion (15-30 minutes)

# 3. Review results
cat portfolio_analysis_*/REPORT.txt | less

# 4. Extract top strategies
grep "Sharpe:" portfolio_analysis_*/REPORT.txt | sort -rn | head -10
```

### Workflow 3: Debug HTML Report
```bash
# 1. Generate report
uv run python master.py --multi-pair --quick

# 2. Find and open report
LATEST=$(ls -td multi_pair_*/ | head -1)
google-chrome "${LATEST}REPORT.html"

# 3. Use Chrome DevTools
# - Press F12
# - Check Console for errors
# - Check Network for failed requests
# - Use Elements to inspect visualizations

# 4. Fix issues and regenerate
# (Iterate until perfect)
```

---

## Future Enhancements

Planned features for the windowed multi-pair system:

- [ ] Interactive correlation heatmaps
- [ ] Portfolio rebalancing simulation
- [ ] Risk decomposition analysis
- [ ] Cross-pair arbitrage opportunities
- [ ] Regime-based pair selection
- [ ] Real-time portfolio monitoring

---

## Summary

**For immediate use:** `master.py --multi-pair`
- Works now
- Generates HTML reports
- Traditional single-window methodology

**For proper ML methodology:** Windowed multi-pair system (in development)
- Train/test split
- Statistical confidence
- Overfitting detection
- Requires completion of aggregator and main script

Choose based on your needs:
- **Speed:** Use existing system
- **Rigor:** Wait for windowed system or help complete it
- **Both:** Run existing now, migrate to windowed later
