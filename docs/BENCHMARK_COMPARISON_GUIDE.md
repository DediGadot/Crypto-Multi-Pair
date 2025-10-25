# Benchmark Comparison Module Guide

## Overview

The `benchmark_comparator.py` module provides comprehensive tools for comparing trading strategy performance against buy-and-hold benchmarks.

## Location

```
src/crypto_trader/analysis/benchmark_comparator.py
```

## Key Components

### 1. BenchmarkComparison (Dataclass)

Stores all comparative metrics between a strategy and benchmark:

**Core Metrics:**
- `alpha`: Absolute return difference (strategy - benchmark) in percentage points
- `relative_alpha`: Alpha as percentage of benchmark return
- `sharpe_alpha`: Sharpe ratio difference (strategy - benchmark)

**Win Rate Statistics:**
- `windows_beat_benchmark`: Number of windows where strategy outperformed
- `total_windows`: Total windows analyzed
- `win_rate_vs_benchmark`: Percentage of windows where strategy beat benchmark

**Per-Window Data:**
- `window_alphas`: List of per-window alpha values
- `window_returns_strategy`: Strategy returns for each window
- `window_returns_benchmark`: Benchmark returns for each window

### 2. BenchmarkComparator (Class)

Performs the comparison calculations.

**Main Method:**
```python
def compare_to_benchmark(
    strategy_metrics: MultiPairWindowedMetrics,
    benchmark_metrics: MultiPairWindowedMetrics,
    strategy_window_returns: Optional[List[float]] = None,
    benchmark_window_returns: Optional[List[float]] = None
) -> BenchmarkComparison
```

## Usage Examples

### Basic Usage (Portfolio-Level Comparison Only)

```python
from src.crypto_trader.analysis.benchmark_comparator import BenchmarkComparator
from src.crypto_trader.analysis.multipair_aggregator import MultiPairAggregator

# Create aggregator and get metrics
aggregator = MultiPairAggregator()

strategy_metrics = aggregator.aggregate_multipair_windows(
    pair_results=strategy_results,
    strategy_name="Copula Pairs Trading",
    horizon_name="30d",
    dataset_type="test"
)

benchmark_metrics = aggregator.aggregate_multipair_windows(
    pair_results=benchmark_results,
    strategy_name="Buy and Hold",
    horizon_name="30d",
    dataset_type="test"
)

# Compare
comparator = BenchmarkComparator()
comparison = comparator.compare_to_benchmark(
    strategy_metrics,
    benchmark_metrics
)

print(f"Alpha: {comparison.alpha:.2f}%")
print(f"Sharpe Alpha: {comparison.sharpe_alpha:.2f}")
```

### Advanced Usage (With Per-Window Analysis)

```python
# Provide per-window returns for win rate calculation
strategy_window_returns = [15.0, 22.0, 18.0, 25.0, 14.0, 20.0, 17.0, 19.0, 11.0, 24.0]
benchmark_window_returns = [10.0, 14.0, 11.0, 18.0, 12.0, 13.0, 12.0, 11.0, 13.0, 16.0]

comparison = comparator.compare_to_benchmark(
    strategy_metrics,
    benchmark_metrics,
    strategy_window_returns,
    benchmark_window_returns
)

# Access detailed metrics
print(f"Win Rate: {comparison.win_rate_vs_benchmark:.1f}%")
print(f"Windows Beat Benchmark: {comparison.windows_beat_benchmark}/{comparison.total_windows}")
print(f"Alpha Distribution: μ={np.mean(comparison.window_alphas):.2f}%, σ={np.std(comparison.window_alphas):.2f}%")
```

### Export and Serialization

```python
# Export to dictionary for storage or further analysis
comparison_dict = comparison.to_dict()

# Save to file
import json
with open('benchmark_comparison.json', 'w') as f:
    json.dump(comparison_dict, f, indent=2)

# Print human-readable summary
print(comparison.summary_string())
```

## Interpretation Guidelines

### Alpha Metrics

**Positive Alpha (α > 0)**
- Strategy outperforms benchmark
- Example: α = +5.0% means strategy returns 5% more than benchmark

**Negative Alpha (α < 0)**
- Strategy underperforms benchmark
- Example: α = -3.0% means strategy returns 3% less than benchmark

**Relative Alpha**
- Shows alpha as percentage of benchmark return
- Example: If benchmark = 10% and α = 5%, relative alpha = 50%

### Sharpe Alpha

**Positive Sharpe Alpha**
- Strategy has better risk-adjusted returns
- Example: Sharpe α = +0.5 means strategy Sharpe is 0.5 higher

**Negative Sharpe Alpha**
- Benchmark has better risk-adjusted returns
- Consider whether higher returns justify lower Sharpe

### Win Rate

**High Win Rate (>60%)**
- Consistent outperformance across windows
- Indicates robust strategy performance

**Low Win Rate (<40%)**
- Inconsistent performance
- May indicate strategy works only in specific market conditions

**Moderate Win Rate (40-60%)**
- Mixed performance
- Examine window_alphas distribution for patterns

## Mathematical Definitions

### Alpha (Absolute)
```
α = R_strategy - R_benchmark
```
Where:
- R_strategy = Mean portfolio return of strategy
- R_benchmark = Mean portfolio return of benchmark

### Relative Alpha
```
α_rel = (α / |R_benchmark|) × 100%
```

### Sharpe Alpha
```
α_sharpe = Sharpe_strategy - Sharpe_benchmark
```

### Win Rate
```
Win_rate = (# windows where R_strategy > R_benchmark) / total_windows × 100%
```

Note: Ties (R_strategy = R_benchmark) are counted as losses.

## Key Features

1. **Flexible Input**: Works with or without per-window data
2. **Robust Handling**: Gracefully handles edge cases (zero benchmark, missing data)
3. **Comprehensive Metrics**: Returns, Sharpe, and win rate comparisons
4. **Distribution Analysis**: Per-window alphas for advanced analysis
5. **Serialization Ready**: Easy export to dictionaries/JSON

## Validation

Run the built-in validation:

```bash
uv run python src/crypto_trader/analysis/benchmark_comparator.py
```

Expected output:
```
✅ VALIDATION PASSED - All 6 tests produced expected results
```

## Demo

Run the comprehensive demo:

```bash
uv run python demo_benchmark_comparison.py
```

This demonstrates:
- Outperforming strategy comparison
- Underperforming strategy comparison
- Data export and serialization

## Integration Points

### Input Requirements

**Required:**
- `MultiPairWindowedMetrics` for strategy
- `MultiPairWindowedMetrics` for benchmark
- Both must have same `horizon_name` and `dataset_type`

**Optional (for full analysis):**
- `strategy_window_returns`: List[float] - per-window portfolio returns
- `benchmark_window_returns`: List[float] - per-window portfolio returns

### Output Usage

The `BenchmarkComparison` object can be:
1. Logged for monitoring
2. Stored in databases
3. Visualized in reports
4. Used for strategy selection
5. Exported for external analysis

## Performance Notes

- **Time Complexity**: O(n) where n = number of windows
- **Space Complexity**: O(n) for per-window data storage
- **Recommended**: Use per-window data for comprehensive analysis
- **Warning**: Large window counts (>1000) may impact memory

## Future Enhancements

Potential additions:
- Statistical significance testing (t-test for alpha)
- Rolling alpha calculations
- Conditional alpha (bull/bear markets)
- Multi-benchmark comparisons
- Alpha persistence analysis

## References

- Alpha: https://en.wikipedia.org/wiki/Alpha_(finance)
- Sharpe Ratio: https://en.wikipedia.org/wiki/Sharpe_ratio
- Win Rate: https://www.investopedia.com/terms/w/win-loss-ratio.asp
