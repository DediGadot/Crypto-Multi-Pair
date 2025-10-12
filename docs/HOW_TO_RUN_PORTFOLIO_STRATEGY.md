# How to Run Multi-Crypto Portfolio Rebalancing Strategy

**Quick Start Guide for Portfolio Backtesting**

---

## Quick Start (Recommended)

### Run with Optimized 10% Threshold (Best Performance)

```bash
# Navigate to project directory
cd /home/fiod/crypto

# Run the optimized strategy (10% threshold - outperforms buy-and-hold by 601%)
uv run python run_portfolio_backtest.py --config config_improved_10pct.yaml
```

**Expected Results**:
- Final Value: ~$297,509 (from $10,000)
- Total Return: ~2,875%
- Outperformance: +601.79% vs buy-and-hold
- Rebalance Events: ~15 over 8 years
- Runtime: ~30-60 seconds

---

## All Available Configurations

### 1. Optimized 10% Threshold (RECOMMENDED) 🏆

```bash
uv run python run_portfolio_backtest.py --config config_improved_10pct.yaml
```

**Best For**: Maximum performance with optimal rebalancing frequency
- Threshold: 10%
- Expected Rebalances: ~15 over 8 years
- Performance: +601.79% vs buy-and-hold

### 2. Aggressive 5% Threshold

```bash
uv run python run_portfolio_backtest.py --config config_improved_5pct.yaml
```

**Best For**: More frequent rebalancing, capturing smaller divergences
- Threshold: 5%
- Expected Rebalances: ~56 over 8 years
- Performance: +554.10% vs buy-and-hold
- Note: More trading, potentially higher costs

### 3. Original 15% Threshold (Not Recommended)

```bash
uv run python run_portfolio_backtest.py --config config.yaml
```

**Best For**: Historical baseline comparison
- Threshold: 15%
- Expected Rebalances: ~5 over 8 years
- Performance: -32.59% vs buy-and-hold ❌
- Note: Too conservative, underperforms

### 4. Calendar-Based Monthly Rebalancing

```bash
uv run python run_portfolio_backtest.py --config config_improved_calendar.yaml
```

**Best For**: Fixed schedule, predictable trading
- Method: Calendar (monthly)
- Expected Rebalances: ~97 over 8 years (monthly)
- Performance: Not yet tested

### 5. Hybrid Approach

```bash
uv run python run_portfolio_backtest.py --config config_improved_hybrid.yaml
```

**Best For**: Combining calendar + threshold + momentum filter
- Method: Hybrid (calendar + 10% threshold + momentum)
- Momentum Filter: Skips rebalance if >20% gain in 30 days
- Performance: Not yet tested

---

## Understanding the Output

### Console Output

```
======================================================================
PORTFOLIO BACKTESTING - STARTING
======================================================================

STEP 1: Fetching Portfolio Data
======================================================================
Fetching BTC/USDT...
  ✓ Fetched 71320 candles
  Date range: 2017-08-17 to 2025-10-11

[... fetching other assets ...]

STEP 2: Running Portfolio Rebalancing Simulation
======================================================================
Common timespan: 45276 periods
Simulation complete:
  Rebalance events: 15
  Initial value: $10,000.00
  Final value: $297,509.04
  Total return: 2875.09%

STEP 3: Generating Reports
======================================================================
  ✓ Equity curve: results_10pct/data/portfolio_equity_curve.csv
  ✓ Buy-hold benchmark: results_10pct/data/buy_hold_benchmark.csv
  ✓ Rebalance events: results_10pct/data/rebalance_events.csv
  ✓ Summary report: results_10pct/PORTFOLIO_SUMMARY.txt

PORTFOLIO BACKTEST COMPLETED SUCCESSFULLY!
Duration: 28.5 seconds
Results saved to: results_10pct
```

### Output Files

After running, you'll find results in the output directory:

```
results_10pct/
├── PORTFOLIO_SUMMARY.txt          # Human-readable summary
├── portfolio_20251012_114103.log  # Execution log
└── data/
    ├── portfolio_equity_curve.csv # Full equity curve (45,276 rows)
    ├── buy_hold_benchmark.csv     # Buy-and-hold comparison
    └── rebalance_events.csv       # Detailed rebalance history (15 events)
```

### Summary Report

View the summary:

```bash
cat results_10pct/PORTFOLIO_SUMMARY.txt
```

**Key Sections**:
1. **Portfolio Configuration**: Assets, weights, threshold
2. **Performance Comparison**: Rebalanced vs Buy-and-hold
3. **Result**: Outperformance or underperformance
4. **Output Files**: List of generated CSV files

---

## Customizing the Strategy

### Create Your Own Configuration

1. **Copy a template**:
```bash
cp config_improved_10pct.yaml my_custom_config.yaml
```

2. **Edit the configuration**:
```yaml
run:
  name: "my_custom_portfolio"
  description: "My custom portfolio configuration"

portfolio:
  assets:
    # Customize asset allocation
    - symbol: "BTC/USDT"
      weight: 0.50    # 50% BTC
    - symbol: "ETH/USDT"
      weight: 0.30    # 30% ETH
    - symbol: "SOL/USDT"
      weight: 0.20    # 20% SOL
    # Remove or add assets as needed

  rebalancing:
    threshold: 0.10   # Change threshold (0.05, 0.10, 0.15)
    rebalance_method: "threshold"  # or "calendar" or "hybrid"
    calendar_period_days: 30       # For calendar/hybrid methods
    min_rebalance_interval_hours: 24
    use_momentum_filter: false     # Set to true to avoid rebalancing in trends
    momentum_lookback_days: 30

data:
  timeframe: "1h"     # Keep as 1h for best results
  days: 2977          # ~8 years, or customize

capital:
  initial_capital: 10000.0  # Starting capital

output:
  directory: "results_custom"  # Custom output directory
```

3. **Run your custom config**:
```bash
uv run python run_portfolio_backtest.py --config my_custom_config.yaml
```

### Asset Allocation Examples

**Bitcoin Dominant (70/30)**:
```yaml
portfolio:
  assets:
    - symbol: "BTC/USDT"
      weight: 0.70
    - symbol: "ETH/USDT"
      weight: 0.30
```

**Equal Weight 3-Asset**:
```yaml
portfolio:
  assets:
    - symbol: "BTC/USDT"
      weight: 0.33
    - symbol: "ETH/USDT"
      weight: 0.33
    - symbol: "SOL/USDT"
      weight: 0.34
```

**Conservative with Stablecoin** (if you add stablecoin support):
```yaml
portfolio:
  assets:
    - symbol: "BTC/USDT"
      weight: 0.30
    - symbol: "ETH/USDT"
      weight: 0.20
    - symbol: "USDT"        # Stablecoin (requires implementation)
      weight: 0.50
```

---

## Command-Line Options

### Basic Usage

```bash
uv run python run_portfolio_backtest.py [OPTIONS]
```

### Available Options

```bash
--config CONFIG_FILE    # Path to YAML config file (default: config.yaml)
--output OUTPUT_DIR     # Override output directory from config
```

### Examples

**Use custom config**:
```bash
uv run python run_portfolio_backtest.py --config my_portfolio.yaml
```

**Custom output directory**:
```bash
uv run python run_portfolio_backtest.py \
    --config config_improved_10pct.yaml \
    --output my_results
```

**Quick test with 30 days**:
```bash
# Edit config first to set days: 30
uv run python run_portfolio_backtest.py --config config_test.yaml
```

---

## Comparing Multiple Strategies

### Run All Configurations

```bash
# Run all optimized configurations
uv run python run_portfolio_backtest.py --config config_improved_5pct.yaml
uv run python run_portfolio_backtest.py --config config_improved_10pct.yaml
uv run python run_portfolio_backtest.py --config config_improved_calendar.yaml
uv run python run_portfolio_backtest.py --config config_improved_hybrid.yaml
```

### Compare Results

```bash
# Run comparison script
uv run python compare_results.py
```

**Output**:
```
====================================================================================================
PORTFOLIO REBALANCING STRATEGY - PERFORMANCE COMPARISON
====================================================================================================

                Strategy Final Value   Return  Rebalances Buy&Hold Return Outperformance Relative
           10% Threshold $297,509.04 2875.09%          15        2273.30%        601.79%   26.47%
            5% Threshold $292,739.94 2827.40%          56        2273.30%        554.10%   24.37%
Original (15% threshold) $234,070.46 2240.70%           5        2273.30%        -32.60%   -1.43%

====================================================================================================

🏆 BEST PERFORMER: 10% Threshold
   Return: 2875.09%
   Outperformance vs Buy&Hold: 601.79%
   Rebalance Events: 15
```

---

## Analyzing Results

### View Summary

```bash
cat results_10pct/PORTFOLIO_SUMMARY.txt
```

### Examine Equity Curve

```bash
# View first 10 rows
head -n 10 results_10pct/data/portfolio_equity_curve.csv

# Count data points
wc -l results_10pct/data/portfolio_equity_curve.csv
```

### Analyze Rebalance Events

```bash
# View all rebalance events
cat results_10pct/data/rebalance_events.csv

# Count rebalances
tail -n +2 results_10pct/data/rebalance_events.csv | wc -l
```

### Python Analysis (Optional)

```python
import pandas as pd

# Load equity curve
equity = pd.read_csv('results_10pct/data/portfolio_equity_curve.csv')
print(equity.head())

# Load rebalance events
rebalances = pd.read_csv('results_10pct/data/rebalance_events.csv')
print(f"Total rebalances: {len(rebalances)}")
print(rebalances)

# Plot equity curve
import matplotlib.pyplot as plt
equity['total_value'].plot(title='Portfolio Equity Curve')
plt.ylabel('Portfolio Value ($)')
plt.show()
```

---

## Performance Benchmarks

### Expected Runtime

| Configuration | Data Fetching | Simulation | Total |
|--------------|---------------|------------|-------|
| 10% Threshold | 15-30s | 3-5s | 20-35s |
| 5% Threshold | 15-30s | 4-6s | 20-36s |
| 15% Threshold | 15-30s | 2-4s | 18-34s |

**Note**: First run is slower (data fetching). Subsequent runs use cached data.

### Memory Usage

- **Typical**: 200-400 MB RAM
- **Peak**: 500-800 MB RAM (during data processing)

### Disk Space

- **Data Cache**: ~50-100 MB (OHLCV data for 4 assets × 8 years)
- **Results Per Run**: ~5-10 MB (equity curves, reports)

---

## Troubleshooting

### Issue: "Config file not found"

**Solution**: Ensure you're in the project directory
```bash
cd /home/fiod/crypto
ls config*.yaml  # Should list configuration files
```

### Issue: "No module named 'crypto_trader'"

**Solution**: Use `uv run` instead of `python`
```bash
# ❌ Wrong
python run_portfolio_backtest.py

# ✅ Correct
uv run python run_portfolio_backtest.py
```

### Issue: Data fetching fails

**Solution**: Check internet connection and Binance API
```bash
# Test Binance connectivity
curl https://api.binance.com/api/v3/ping
```

### Issue: "Asset weights must sum to 1.0"

**Solution**: Fix weights in config file
```yaml
# ❌ Wrong (sums to 0.9)
assets:
  - symbol: "BTC/USDT"
    weight: 0.50
  - symbol: "ETH/USDT"
    weight: 0.40

# ✅ Correct (sums to 1.0)
assets:
  - symbol: "BTC/USDT"
    weight: 0.50
  - symbol: "ETH/USDT"
    weight: 0.50
```

### Issue: Slow execution

**Solutions**:
1. **Reduce timeframe**:
   ```yaml
   data:
     days: 365  # Test with 1 year instead of 8
   ```

2. **Use cached data**: Second run is much faster

3. **Check disk space**: Ensure enough space for data cache

---

## Best Practices

### 1. Start with Short Backtests

```yaml
# Test configuration first
data:
  days: 30  # 30 days for quick validation
```

Then run full 8-year backtest:
```yaml
data:
  days: 2977  # Full 8+ years
```

### 2. Use Version Control

```bash
# Save your configuration
git add my_custom_config.yaml
git commit -m "Add custom portfolio configuration"
```

### 3. Document Your Runs

```bash
# Add metadata to config
run:
  name: "my_portfolio_test_v1"
  description: "Testing 10% threshold with equal weight BTC/ETH"
```

### 4. Back Up Results

```bash
# Copy results before next run
cp -r results_10pct results_10pct_backup_$(date +%Y%m%d)
```

### 5. Track Performance

Keep a spreadsheet of your runs:

| Date | Config | Return | Rebalances | vs Buy&Hold | Notes |
|------|--------|--------|------------|-------------|-------|
| 2025-10-12 | 10% threshold | 2875% | 15 | +601.79% | Best performer |
| 2025-10-12 | 5% threshold | 2827% | 56 | +554.10% | More frequent |

---

## Next Steps

### 1. Run the Recommended Configuration

```bash
uv run python run_portfolio_backtest.py --config config_improved_10pct.yaml
```

### 2. Review the Results

```bash
cat results_10pct/PORTFOLIO_SUMMARY.txt
```

### 3. Experiment with Customization

- Try different asset allocations
- Test different thresholds
- Explore calendar-based rebalancing

### 4. Analyze Rebalance Events

```bash
cat results_10pct/data/rebalance_events.csv
```

### 5. Compare Multiple Strategies

```bash
# Run several configs, then:
uv run python compare_results.py
```

---

## Advanced Usage

### Parallel Runs

Run multiple backtests in parallel:

```bash
# Terminal 1
uv run python run_portfolio_backtest.py --config config_improved_5pct.yaml &

# Terminal 2
uv run python run_portfolio_backtest.py --config config_improved_10pct.yaml &

# Wait for both
wait
```

### Automated Testing

```bash
# Create a test script
cat > test_all_configs.sh << 'EOF'
#!/bin/bash
for config in config_improved_*.yaml; do
    echo "Testing $config..."
    uv run python run_portfolio_backtest.py --config "$config"
done
uv run python compare_results.py
EOF

chmod +x test_all_configs.sh
./test_all_configs.sh
```

---

## Summary of Commands

### Essential Commands

```bash
# Quick start (recommended)
uv run python run_portfolio_backtest.py --config config_improved_10pct.yaml

# View results
cat results_10pct/PORTFOLIO_SUMMARY.txt

# Compare all strategies
uv run python compare_results.py

# Run tests
uv run pytest tests/test_portfolio*.py
```

### Configuration Files

| File | Threshold | Method | Best For |
|------|-----------|--------|----------|
| `config_improved_10pct.yaml` | 10% | Threshold | **Recommended - Best Performance** 🏆 |
| `config_improved_5pct.yaml` | 5% | Threshold | More frequent rebalancing |
| `config_improved_calendar.yaml` | N/A | Calendar | Predictable monthly schedule |
| `config_improved_hybrid.yaml` | 10% | Hybrid | Calendar + Threshold + Momentum |
| `config.yaml` | 15% | Threshold | Original (not recommended) |

---

## Quick Reference

**Run optimized strategy**:
```bash
uv run python run_portfolio_backtest.py --config config_improved_10pct.yaml
```

**Expected output directory**: `results_10pct/`

**Expected performance**: ~2,875% return, +601.79% vs buy-and-hold

**Runtime**: 20-35 seconds

**Results to check**:
1. `results_10pct/PORTFOLIO_SUMMARY.txt` - Main report
2. `results_10pct/data/rebalance_events.csv` - Rebalance history
3. `results_10pct/data/portfolio_equity_curve.csv` - Full equity curve

---

**Need Help?**
- Check logs in `results_10pct/*.log`
- Review `/docs/ALGORITHM_IMPROVEMENTS_RESULTS.md` for detailed analysis
- Run tests: `uv run pytest tests/test_portfolio*.py -v`
