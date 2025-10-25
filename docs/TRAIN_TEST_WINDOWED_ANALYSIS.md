# Train/Test Windowed Analysis

## Overview

This document describes the new train/test windowed analysis system that implements proper machine learning methodology for strategy evaluation.

## Problem with Original Approach

The original `master.py` system tests each time horizon (30d, 90d, 180d, etc.) using **only the most recent window** of that size. For example:
- 30d horizon: Tests on days 951-980 (most recent 30 days)
- 90d horizon: Tests on days 891-980 (most recent 90 days)
- 180d horizon: Tests on days 801-980 (most recent 180 days)

### Issues:
1. **Single sample per horizon**: Only one 30-day period tested, not representative
2. **No lookahead protection**: No proper train/test split
3. **Overfitting risk**: Can't distinguish between robust strategies and lucky ones
4. **No statistical confidence**: No std dev, percentiles, or consistency metrics

## New Approach: Train/Test Windowed Analysis

The new system (`master_windowed.py`) implements scientific methodology:

### 1. Train/Test Split

```
Full Historical Data (e.g., 3 years)
├── Training Set: All data before cutoff (e.g., first 1 year)
│   └── Used for parameter tuning and strategy selection
└── Test Set: Recent data (e.g., last 2 years)
    └── Used for final evaluation (unseen data)
```

**Key Dates:**
- Runtime Date: When you run the analysis (e.g., 2025-10-20)
- Cutoff Date: Runtime Date - test_set_years (e.g., 2023-10-20 with 2-year test set)
- Training Data: All data before cutoff
- Test Data: Cutoff to runtime

### 2. Non-Overlapping Windows

Instead of testing one window per horizon, we test **all non-overlapping windows**:

```
Example for 30d horizon with 365 days of training data:

Day 1-30    | Window 0 (train)
Day 31-60   | Window 1 (train)
Day 61-90   | Window 2 (train)
...
Day 331-360 | Window 11 (train)
[unused 5 days]

Then for 730 days of test data:
Day 1-30    | Window 0 (test)
Day 31-60   | Window 1 (test)
...
Day 701-730 | Window 23 (test)
```

**Benefits:**
- 12 training windows + 24 test windows = 36 independent samples
- Statistical significance
- Measure consistency across different market conditions

### 3. Comprehensive Statistics

For each strategy × horizon × dataset (train/test), compute:

| Metric | Description |
|--------|-------------|
| Mean | Average performance |
| Median | Robust central tendency |
| Std Dev | Consistency measure (lower = more consistent) |
| 25th Percentile | Worst-case quartile |
| 75th Percentile | Best-case quartile |
| Weighted Mean | Time-weighted (gives more weight to recent windows) |
| Consistency Score | 1/(1 + coefficient_of_variation) |

### 4. Result Caching

Results for each window are cached:
- Cache key: `strategy|symbol|timeframe|horizon|window_id|dataset|start_date|end_date`
- Storage: CSV file (`data/performance/windowed_results_cache.csv`)
- Benefit: Rerunning analysis doesn't recompute existing windows

### 5. Generalization Analysis

Compare train vs test performance:

```
Gap = Test Sharpe - Train Sharpe

Gap > 0  → ✓ Generalizes well (test better than train)
Gap ≈ 0  → ○ Good generalization
Gap < 0  → ✗ Overfitting (worse on unseen data)
```

## Architecture

### New Modules

1. **`src/crypto_trader/orchestration/window_manager.py`**
   - `TrainTestSplitter`: Splits data at cutoff date
   - `WindowSpec`: Specification for a single window
   - `generate_non_overlapping_windows()`: Creates windows

2. **`src/crypto_trader/analysis/aggregator.py`**
   - `ResultsAggregator`: Computes statistics across windows
   - `WindowedMetrics`: Holds aggregated metrics
   - `compute_composite_score()`: Multi-factor scoring

3. **`src/crypto_trader/analysis/windowed_cache.py`**
   - `WindowedResultsCache`: Persistent result storage
   - `get_result()`: Retrieve cached result
   - `store_result()`: Cache new result

4. **`master_windowed.py`**
   - Main entry point for windowed analysis
   - Orchestrates: fetch → split → window → backtest → aggregate → report

### Workflow

```
1. Fetch historical data (e.g., 3 years)
   ↓
2. Split into train/test sets
   ↓
3. Generate non-overlapping windows for each horizon
   ├── 30d: 12 train windows + 24 test windows
   ├── 90d: 4 train windows + 8 test windows
   └── 180d: 2 train windows + 4 test windows
   ↓
4. Run backtests for all strategy × window combinations
   - Check cache first
   - Execute in parallel (ProcessPoolExecutor)
   - Store results in cache
   ↓
5. Aggregate results
   - Group by strategy × horizon × dataset
   - Compute mean, median, std, percentiles
   ↓
6. Generate reports
   - aggregated_results.csv: All statistics
   - REPORT.txt: Rankings and analysis
```

## Usage

### Basic Usage

```bash
# Quick mode (3 horizons, 180 days of data)
python master_windowed.py --quick

# Standard mode (4 horizons, 3 years of data)
python master_windowed.py

# Custom configuration
python master_windowed.py \
  --symbol BTC/USDT \
  --timeframe 1h \
  --test-years 2 \
  --horizons 30 60 90 180 \
  --max-days 1095 \
  --workers 8
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--symbol` | `BTC/USDT` | Trading pair |
| `--timeframe` | `1h` | Candle timeframe |
| `--test-years` | `2.0` | Years reserved for test set |
| `--horizons` | - | Custom horizons (days) |
| `--workers` | `4` | Parallel workers |
| `--quick` | `False` | Fast mode (fewer horizons) |
| `--max-days` | `1095` | Max days to fetch |
| `--output` | `windowed_results` | Output directory base name |

## Output Files

```
windowed_results_YYYYMMDD_HHMMSS/
├── windowed_analysis.log              # Detailed logs
├── aggregated_results.csv             # All statistics
└── REPORT.txt                         # Human-readable report
    ├── Configuration
    ├── Methodology
    ├── Top Strategies by Test Performance
    └── Generalization Analysis
```

### aggregated_results.csv Columns

- `strategy`: Strategy name
- `horizon_name`: Horizon (e.g., '30d')
- `dataset_type`: 'train' or 'test'
- `num_windows`: Number of windows tested
- `mean_return`, `median_return`, `std_return`, `p25_return`, `p75_return`, `weighted_return`
- `mean_sharpe`, `median_sharpe`, `std_sharpe`, `p25_sharpe`, `p75_sharpe`, `weighted_sharpe`
- `mean_drawdown`, `median_drawdown`, `std_drawdown`, `p25_drawdown`, `p75_drawdown`, `weighted_drawdown`
- `mean_win_rate`, `median_win_rate`, `std_win_rate`, `p25_win_rate`, `p75_win_rate`, `weighted_win_rate`
- `mean_trades`, `total_trades`
- `consistency_score`

## Interpretation Guide

### Reading Results

**Example Output:**

```
SMA_Crossover (30d, train):
  Windows: 12
  Return: 8.50% ± 3.20% (median: 8.10%)
  Sharpe: 1.45 ± 0.35 (median: 1.40)
  Drawdown: 12.30% ± 4.10% (median: 11.80%)
  Consistency: 0.805

SMA_Crossover (30d, test):
  Windows: 24
  Return: 7.80% ± 2.90% (median: 7.50%)
  Sharpe: 1.38 ± 0.31 (median: 1.35)
  Drawdown: 13.10% ± 3.80% (median: 12.90%)
  Consistency: 0.816
```

**Analysis:**
- Train return: 8.50% ± 3.20% → Average 8.5%, variability of ±3.2%
- Test return: 7.80% ± 2.90% → Slightly lower but similar (good generalization)
- Consistency: High (>0.8) → Performs consistently across windows
- Gap: 1.45 - 1.38 = +0.07 Sharpe → Minimal overfitting

### Red Flags

🚩 **High Overfitting** (Train Sharpe: 2.5, Test Sharpe: 1.0)
- Strategy learned noise in training data
- Not recommended for live trading

🚩 **High Variance** (Sharpe: 1.5 ± 1.2)
- Inconsistent performance
- High risk, unpredictable

🚩 **Negative Test Performance** (Test Sharpe: -0.5)
- Strategy doesn't work on unseen data
- Avoid

### Green Flags

✅ **Good Generalization** (Train Sharpe: 1.5, Test Sharpe: 1.6)
- Better or similar on test set
- Robust strategy

✅ **High Consistency** (Sharpe: 1.5 ± 0.2, Consistency: 0.95)
- Low variance
- Reliable performance

✅ **Positive Across All Windows** (p25_sharpe > 0)
- Even worst quartile is profitable
- Strong strategy

## Comparison: Old vs New System

| Aspect | Old System (master.py) | New System (master_windowed.py) |
|--------|------------------------|----------------------------------|
| **Windows per horizon** | 1 (most recent) | N (all non-overlapping) |
| **Example for 30d** | 1 window | 12 train + 24 test = 36 windows |
| **Train/Test split** | No | Yes (temporal separation) |
| **Lookahead bias** | Possible | Prevented |
| **Statistics** | Single value | Mean, median, std, percentiles |
| **Consistency measure** | No | Yes (coefficient of variation) |
| **Generalization check** | No | Yes (train vs test comparison) |
| **Caching** | No | Yes (avoids recomputation) |
| **Overfitting detection** | No | Yes (train/test gap) |

## Best Practices

### 1. Parameter Tuning
- **Only tune on training set**
- Test set is for final evaluation only
- Rerun analysis after tuning to verify on new test set

### 2. Horizon Selection
- Start with quick mode (`--quick`) to iterate fast
- Use multiple horizons: [30, 60, 90, 180, 365]
- Strategies should perform well across multiple horizons

### 3. Data Amount
- Minimum: 2 years (1 year train, 1 year test)
- Recommended: 3 years (1 year train, 2 years test)
- More data = more windows = better statistics

### 4. Interpreting Results
- Prioritize: Test set Sharpe > Train set Sharpe (generalization)
- Prioritize: High consistency score (low variance)
- Prioritize: Positive p25 metrics (robust even in worst case)

### 5. Strategy Selection
1. Filter by test set performance (Sharpe > threshold)
2. Filter by consistency (consistency_score > 0.7)
3. Filter by generalization (train/test gap < 0.5)
4. Rank by weighted test Sharpe (recent performance)

## Troubleshooting

### "Empty data for test set"
- Your data doesn't cover the test period
- Increase `--max-days` or decrease `--test-years`

### "Not enough windows generated"
- Horizon too large for available data
- Use smaller horizons or fetch more data

### "Cache keeps growing"
- Expected behavior (cache persists across runs)
- Clear manually if needed: `rm data/performance/windowed_results_cache.csv`

## Future Enhancements

Planned improvements:
- [ ] HTML report with interactive plots
- [ ] Distribution visualizations (box plots, histograms)
- [ ] Walk-forward optimization
- [ ] Cross-validation (rolling windows)
- [ ] Monte Carlo simulation for confidence intervals
- [ ] Strategy ensemble recommendations based on consistency

## References

- Linus Torvalds coding principles: Write clean new code over hacking legacy code
- Proper ML methodology: Train/validation/test split
- Statistical rigor: Multiple samples, confidence intervals
- Cache optimization: Avoid redundant computation

## Migration Guide

### From master.py to master_windowed.py

**Before (master.py):**
```bash
python master.py --quick --symbol BTC/USDT
```

**After (master_windowed.py):**
```bash
python master_windowed.py --quick --symbol BTC/USDT
```

**Key Differences:**
1. Results now include train/test split
2. Multiple windows per horizon instead of one
3. Statistics include std dev, percentiles
4. New output format (aggregated_results.csv)
5. Caching enabled (faster reruns)

### Interpreting Old vs New Reports

**Old Report:** "SMA_Crossover 30d: Sharpe 1.5"
- Based on single most recent 30-day window
- No confidence interval
- No train/test validation

**New Report:** "SMA_Crossover 30d test: Sharpe 1.45 ± 0.35 (24 windows)"
- Based on 24 independent test windows
- Confidence interval: ±0.35
- Validated on unseen data

## Conclusion

The train/test windowed analysis system provides:
1. **Scientific rigor**: Proper train/test split prevents lookahead bias
2. **Statistical confidence**: Multiple windows provide confidence intervals
3. **Overfitting detection**: Compare train vs test to identify overfitting
4. **Consistency measurement**: Std dev and percentiles show reliability
5. **Efficient caching**: Avoid recomputing identical windows

This methodology aligns with machine learning best practices and provides robust strategy evaluation.
