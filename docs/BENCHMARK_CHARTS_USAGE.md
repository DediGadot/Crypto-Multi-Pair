# Benchmark Charts Usage Guide

Complete guide for using the `plotly_benchmark_charts` module to create interactive visualizations comparing trading strategies to buy-and-hold benchmarks.

## Overview

The `plotly_benchmark_charts` module provides four production-quality interactive chart types:

1. **Alpha Comparison Chart**: Bar chart showing excess returns by strategy
2. **Win Rate Heatmap**: Strategy × horizon grid showing win rates
3. **Cumulative Returns Chart**: Line chart tracking returns over windows
4. **Return Distribution Violin**: Distribution comparison between strategy and benchmark

All charts feature:
- Professional styling with consistent color schemes
- Responsive design for different screen sizes
- Interactive hover tooltips with detailed information
- Export to HTML for reports and presentations

## Installation

The module is part of the `crypto_trader` package:

```python
from crypto_trader.reports.formatters.plotly_benchmark_charts import (
    create_alpha_comparison_chart,
    create_win_rate_heatmap,
    create_cumulative_returns_chart,
    create_return_distribution_violin
)
```

## Dependencies

- `plotly` (https://plotly.com/python/)
- `numpy`
- `crypto_trader.analysis.benchmark_comparator.BenchmarkComparison`

## Chart Functions

### 1. Alpha Comparison Chart

Shows excess returns (alpha) for each strategy compared to benchmark.

**Function Signature:**
```python
def create_alpha_comparison_chart(
    comparisons: Dict[str, BenchmarkComparison]
) -> go.Figure
```

**Parameters:**
- `comparisons`: Dict mapping strategy keys (e.g., 'MACD_30d') to BenchmarkComparison objects

**Returns:**
- Plotly Figure object

**Features:**
- Green bars for positive alpha (outperformance)
- Red bars for negative alpha (underperformance)
- Sorted by alpha value (best performers first)
- Hover tooltips show alpha, relative alpha, and win rate

**Example:**
```python
from crypto_trader.analysis.benchmark_comparator import BenchmarkComparator

# Assume you have strategy_metrics and benchmark_metrics
comparator = BenchmarkComparator()
comparison = comparator.compare_to_benchmark(
    strategy_metrics,
    benchmark_metrics,
    strategy_window_returns,
    benchmark_window_returns
)

# Create chart
comparisons = {'MACD_30d': comparison}
fig = create_alpha_comparison_chart(comparisons)

# Save to HTML
fig.write_html('alpha_comparison.html')

# Or display in Jupyter
fig.show()
```

**Visual Interpretation:**
- Taller bars = Greater outperformance or underperformance
- Green above 0 = Strategy beats benchmark
- Red below 0 = Strategy underperforms benchmark
- Text labels show exact alpha percentage

---

### 2. Win Rate Heatmap

Displays a matrix showing win rate percentages across strategies and time horizons.

**Function Signature:**
```python
def create_win_rate_heatmap(
    comparisons: Dict[str, Dict[str, BenchmarkComparison]]
) -> go.Figure
```

**Parameters:**
- `comparisons`: Nested dict structure:
  ```python
  {
      'StrategyName': {
          '30d': BenchmarkComparison,
          '90d': BenchmarkComparison,
          '180d': BenchmarkComparison
      }
  }
  ```

**Returns:**
- Plotly Figure object

**Features:**
- RdYlGn color scale (red=low, yellow=mid, green=high)
- Cell text showing exact win rate percentages
- Sorted by average win rate (best strategies on top)
- Hover tooltips with alpha, win rate, and window counts

**Example:**
```python
# Organize comparisons by strategy and horizon
heatmap_data = {
    'MACD Strategy': {
        '30d': comparison_macd_30d,
        '90d': comparison_macd_90d,
        '180d': comparison_macd_180d
    },
    'RSI Strategy': {
        '30d': comparison_rsi_30d,
        '90d': comparison_rsi_90d,
        '180d': comparison_rsi_180d
    }
}

fig = create_win_rate_heatmap(heatmap_data)
fig.write_html('win_rate_heatmap.html')
```

**Visual Interpretation:**
- Darker green = Higher win rate (strategy consistently beats benchmark)
- Yellow/Orange = ~50% win rate (mixed performance)
- Red = Low win rate (strategy frequently underperforms)
- Numbers in cells show exact win rate percentage

---

### 3. Cumulative Returns Chart

Line chart showing how strategy and benchmark returns accumulate over sliding windows.

**Function Signature:**
```python
def create_cumulative_returns_chart(
    comparisons: Dict[str, BenchmarkComparison]
) -> go.Figure
```

**Parameters:**
- `comparisons`: Dict mapping strategy keys to BenchmarkComparison objects
  - All comparisons should ideally have the same horizon for meaningful comparison
  - Requires `window_returns_strategy` and `window_returns_benchmark` to be populated

**Returns:**
- Plotly Figure object

**Features:**
- Solid lines for strategy cumulative returns
- Dashed gray line for benchmark cumulative returns
- Multiple strategies can be compared on same chart
- Interactive legend for toggling strategies on/off

**Example:**
```python
# Use comparisons from same horizon for meaningful comparison
cumulative_data = {
    'MACD_30d': comparison_macd,
    'RSI_30d': comparison_rsi,
    'BB_30d': comparison_bb
}

fig = create_cumulative_returns_chart(cumulative_data)
fig.write_html('cumulative_returns.html')
```

**Visual Interpretation:**
- Steeper upward slope = Better cumulative performance
- Distance from benchmark line = Magnitude of outperformance
- Crossing benchmark = Periods of underperformance
- Consistency of slope = Stability of returns

---

### 4. Return Distribution Violin Plot

Shows the full statistical distribution of per-window returns for strategies vs benchmark.

**Function Signature:**
```python
def create_return_distribution_violin(
    comparisons: Dict[str, BenchmarkComparison]
) -> go.Figure
```

**Parameters:**
- `comparisons`: Dict mapping strategy keys to BenchmarkComparison objects
  - Requires `window_returns_strategy` and `window_returns_benchmark` to be populated

**Returns:**
- Plotly Figure object

**Features:**
- Violin plots showing full return distribution
- Box plot overlay with quartiles and median
- Strategy violins colored green (positive alpha) or red (negative alpha)
- Benchmark violin in gray for reference

**Example:**
```python
violin_data = {
    'MACD_30d': comparison_macd,
    'RSI_30d': comparison_rsi,
    'BB_30d': comparison_bb
}

fig = create_return_distribution_violin(violin_data)
fig.write_html('return_distribution.html')
```

**Visual Interpretation:**
- Width of violin = Frequency of returns at that level
- Wider sections = More common return values
- Box inside = Quartiles (25th, 50th, 75th percentile)
- Line inside box = Median return
- Height of violin = Range of returns (min to max)
- Symmetric violin = Balanced distribution
- Skewed violin = Asymmetric returns

---

## Complete Workflow Example

Here's a complete example integrating with the benchmark comparison workflow:

```python
from pathlib import Path
from crypto_trader.analysis.benchmark_comparator import BenchmarkComparator
from crypto_trader.reports.formatters.plotly_benchmark_charts import (
    create_alpha_comparison_chart,
    create_win_rate_heatmap,
    create_cumulative_returns_chart,
    create_return_distribution_violin
)

# Step 1: Run backtests and compute metrics
# (Assume you have strategy_metrics and benchmark_metrics for each strategy/horizon)

# Step 2: Create benchmark comparisons
comparator = BenchmarkComparator()

comparisons = {}
heatmap_data = {}

for strategy_name in ['MACD', 'RSI', 'Bollinger']:
    heatmap_data[strategy_name] = {}

    for horizon in ['30d', '90d', '180d']:
        # Get metrics for this strategy/horizon
        strat_metrics = get_strategy_metrics(strategy_name, horizon)
        bench_metrics = get_benchmark_metrics(horizon)

        # Get per-window returns
        strat_window_returns = get_window_returns(strategy_name, horizon)
        bench_window_returns = get_benchmark_window_returns(horizon)

        # Create comparison
        comparison = comparator.compare_to_benchmark(
            strat_metrics,
            bench_metrics,
            strat_window_returns,
            bench_window_returns
        )

        # Store for charts
        key = f"{strategy_name}_{horizon}"
        comparisons[key] = comparison
        heatmap_data[strategy_name][horizon] = comparison

# Step 3: Generate all charts
output_dir = Path("benchmark_report")
output_dir.mkdir(exist_ok=True)

# Alpha comparison (all strategies/horizons)
fig_alpha = create_alpha_comparison_chart(comparisons)
fig_alpha.write_html(output_dir / "alpha_comparison.html")

# Win rate heatmap
fig_heatmap = create_win_rate_heatmap(heatmap_data)
fig_heatmap.write_html(output_dir / "win_rate_heatmap.html")

# Cumulative returns (30d horizon only)
cumulative_30d = {k: v for k, v in comparisons.items() if '30d' in k}
fig_cumulative = create_cumulative_returns_chart(cumulative_30d)
fig_cumulative.write_html(output_dir / "cumulative_returns_30d.html")

# Return distribution (30d horizon)
fig_violin = create_return_distribution_violin(cumulative_30d)
fig_violin.write_html(output_dir / "return_distribution_30d.html")

print(f"✅ Charts saved to {output_dir}")
```

## Color Scheme Reference

The module uses consistent, professional color schemes:

### Alpha Chart
- **Green (#27AE60)**: Positive alpha (strategy outperforms)
- **Red (#E74C3C)**: Negative alpha (strategy underperforms)
- **Dark borders (#34495E)**: Professional framing

### Win Rate Heatmap
- **RdYlGn** colorscale:
  - Red (0%): Poor win rate
  - Yellow (50%): Neutral
  - Green (100%): Excellent win rate
- Center point at 50% for balanced interpretation

### Cumulative Returns
- **Strategy lines**: Colorful solid lines from qualitative Set2 palette
- **Benchmark line**: Gray (#95A5A6) dashed line
- Clear visual distinction between strategies and baseline

### Violin Plots
- **Positive alpha strategies**: Green (#27AE60)
- **Negative alpha strategies**: Red (#E74C3C)
- **Benchmark**: Gray (#95A5A6)
- 60% opacity for better overlap visibility

## Advanced Usage

### Combining Charts into Reports

Create an HTML report combining all charts:

```python
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Analysis Report</title>
</head>
<body>
    <h1>Trading Strategy Benchmark Analysis</h1>

    <h2>1. Alpha Comparison</h2>
    <iframe src="alpha_comparison.html" width="100%" height="600"></iframe>

    <h2>2. Win Rate Heatmap</h2>
    <iframe src="win_rate_heatmap.html" width="100%" height="600"></iframe>

    <h2>3. Cumulative Returns</h2>
    <iframe src="cumulative_returns_30d.html" width="100%" height="600"></iframe>

    <h2>4. Return Distribution</h2>
    <iframe src="return_distribution_30d.html" width="100%" height="650"></iframe>
</body>
</html>
"""

Path("benchmark_report/index.html").write_text(html_template)
```

### Filtering Data

Filter comparisons before visualization:

```python
# Show only strategies with positive alpha
positive_alpha = {
    k: v for k, v in comparisons.items()
    if v.alpha > 0
}
fig = create_alpha_comparison_chart(positive_alpha)

# Show only long-term horizons (90d+)
long_term = {
    k: v for k, v in comparisons.items()
    if v.horizon_name in ['90d', '180d']
}
fig = create_cumulative_returns_chart(long_term)
```

### Customizing Figures

Modify figures after creation:

```python
fig = create_alpha_comparison_chart(comparisons)

# Update title
fig.update_layout(
    title='Custom Title: Alpha Analysis Q4 2024'
)

# Adjust height
fig.update_layout(height=700)

# Change color scheme (advanced)
fig.data[0].marker.color = ['blue', 'green', 'red', 'yellow']

# Save with custom config
fig.write_html(
    'custom_alpha.html',
    config={'displayModeBar': False}  # Hide toolbar
)
```

## Troubleshooting

### Empty Charts

**Problem**: Chart shows "No data available"

**Solution**: Ensure comparisons dict is not empty:
```python
if not comparisons:
    print("Warning: No comparison data")
else:
    fig = create_alpha_comparison_chart(comparisons)
```

### Missing Window Returns

**Problem**: Cumulative returns or violin plots show warning about missing data

**Solution**: Ensure BenchmarkComparison objects have window-level data:
```python
comparison = comparator.compare_to_benchmark(
    strategy_metrics,
    benchmark_metrics,
    strategy_window_returns,  # ← Must provide these
    benchmark_window_returns  # ← Must provide these
)
```

### Heatmap Data Structure

**Problem**: Heatmap shows incorrect structure

**Solution**: Verify nested dict structure:
```python
# Correct structure
heatmap_data = {
    'Strategy1': {
        '30d': comparison_obj,
        '90d': comparison_obj
    },
    'Strategy2': {
        '30d': comparison_obj,
        '90d': comparison_obj
    }
}

# NOT flat like alpha chart
# comparisons = {'Strategy1_30d': comparison_obj}  # ← Wrong for heatmap
```

## Performance Considerations

- Each chart generates ~4-5MB HTML file (includes Plotly.js library)
- For large datasets (100+ windows), consider downsampling for violin plots
- Heatmaps scale well up to ~20 strategies × 10 horizons
- Use `include_plotlyjs='cdn'` in `write_html()` to reduce file size if embedding multiple charts

## Demo

Run the included demo to see all charts:

```bash
uv run python demo_benchmark_charts.py
```

This generates:
- `benchmark_charts_demo/alpha_comparison.html`
- `benchmark_charts_demo/win_rate_heatmap.html`
- `benchmark_charts_demo/cumulative_returns_30d.html`
- `benchmark_charts_demo/return_distribution_30d.html`
- `benchmark_charts_demo/index.html` (combined report)

## Related Documentation

- [Benchmark Comparator](../src/crypto_trader/analysis/benchmark_comparator.py) - Creating BenchmarkComparison objects
- [Multi-Pair Windowed Analysis](./TRAIN_TEST_WINDOWED_ANALYSIS.md) - Generating windowed metrics
- [Plotly Interactive Charts](../src/crypto_trader/reports/formatters/plotly_interactive.py) - Related visualization module

## Support

For issues or questions:
1. Check the validation block in `plotly_benchmark_charts.py` for usage examples
2. Run the demo script to see expected output
3. Review the docstrings in each function for parameter details
