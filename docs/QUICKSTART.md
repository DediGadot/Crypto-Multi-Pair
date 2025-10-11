# Quick Start Guide

Get up and running with the Crypto Strategy Comparison Dashboard in minutes.

## Installation

### Prerequisites

- Python 3.11 or higher
- `uv` package manager (recommended) or `pip`

### Step 1: Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Step 2: Run the Dashboard

```bash
# Using uv
uv run streamlit run src/crypto_strategy_comparison/app.py

# Or directly
streamlit run src/crypto_strategy_comparison/app.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`.

## Basic Usage

### 1. Select Strategies

In the sidebar, check the strategies you want to compare (minimum 2, maximum 10):

```
☑ Momentum ETH
☑ Mean Reversion BTC
☐ Grid Trading Multi
☐ DCA Bitcoin
```

### 2. Choose Time Horizon

Select a time period for comparison:

- **1W**: 1 Week
- **1M**: 1 Month
- **3M**: 3 Months
- **6M**: 6 Months (recommended)
- **1Y**: 1 Year
- **ALL**: All available time

### 3. Run Analysis

Click the **"🔄 Run Analysis"** button in the main area.

### 4. View Results

The dashboard will display:

- **Equity Curves**: Visual comparison of portfolio values over time
- **Metrics Table**: Key performance metrics side-by-side
- **Summary Cards**: Best performers highlighted

### 5. Explore Details

Navigate through the tabs for deeper analysis:

- **📊 Charts**: Returns distribution, risk-return profile, correlations
- **📉 Drawdowns**: Underwater plots showing drawdowns
- **📋 Trades**: Trade-level details with filtering
- **⚙️ Parameters**: Strategy parameter comparison
- **📤 Export**: Download reports in various formats

## Example Workflow

### Compare Top 3 Strategies

```python
# In Python shell or notebook
from crypto_strategy_comparison.strategy_loader import StrategyLoader
from crypto_strategy_comparison.comparison_engine import ComparisonEngine

# Initialize
loader = StrategyLoader()
engine = ComparisonEngine()

# Load strategies
strategies = loader.load_strategies([
    "Momentum ETH",
    "Mean Reversion BTC",
    "Grid Trading Multi"
])

# Run comparison
results = engine.compare(strategies, time_horizon="6M")

# View metrics
for strategy, metrics in results["metrics"].items():
    print(f"{strategy}: {metrics['total_return']:.2f}% return")
```

### Run Example Script

```bash
uv run python examples/basic_comparison.py
```

This will:
1. Load the first 3 available strategies
2. Run a 6-month comparison
3. Display results in the terminal

Expected output:
```
📊 Available Strategies:
  1. Momentum ETH
  2. Mean Reversion BTC
  3. Grid Trading Multi
  ...

🎯 Comparing: Momentum ETH, Mean Reversion BTC, Grid Trading Multi

📥 Loading strategy data...
⚡ Running comparison analysis...

============================================================
📊 COMPARISON RESULTS
============================================================

Time Horizon: 6M
Strategies Compared: 3

📈 Performance Metrics:
------------------------------------------------------------
Strategy                  Return       Sharpe     Max DD
------------------------------------------------------------
Momentum ETH               145.23%      2.34    -18.50%
Mean Reversion BTC          98.67%      1.89    -12.30%
Grid Trading Multi          76.34%      1.56    -22.10%
------------------------------------------------------------

🏆 Best Performers:
  Highest Return: Momentum ETH (145.23%)
  Best Sharpe: Momentum ETH (2.34)

✅ Example completed successfully!
```

## Common Tasks

### Filter by Risk Tolerance

In the sidebar, adjust risk filters:

1. **Max Acceptable Drawdown**: Set to your risk tolerance (e.g., 20%)
2. **Minimum Sharpe Ratio**: Set minimum risk-adjusted return (e.g., 1.5)
3. Click **"Apply Filters"**

Only strategies meeting these criteria will be shown.

### Adjust Strategy Parameters

1. In the sidebar, click **"Parameter Explorer"**
2. Select a strategy from the dropdown
3. Adjust parameters using sliders/inputs
4. Click **"Apply Parameters"**
5. Re-run analysis to see the effect

### Export Reports

1. Navigate to the **"Export"** tab
2. Select desired formats:
   - ☑ PDF Report
   - ☑ HTML
   - ☐ CSV Data
   - ☐ JSON
3. Choose what to include:
   - ☑ Include Charts
   - ☑ Include Detailed Metrics
   - ☐ Include Trade Details
4. Click **"📥 Download"**

## Keyboard Shortcuts

When the dashboard is active:

- **Tab**: Navigate between controls
- **Space/Enter**: Activate buttons
- **Arrow Keys**: Adjust sliders
- **Esc**: Close modals/dialogs

## Tips & Best Practices

### Performance Tips

1. **Start with fewer strategies** (2-3) for faster initial analysis
2. **Use shorter time horizons** (1M or 3M) for quick comparisons
3. **Add more strategies** gradually after initial results
4. **Cache results** by not changing selections unnecessarily

### Analysis Tips

1. **Compare similar strategy types** first (e.g., all momentum strategies)
2. **Look at risk-adjusted returns** (Sharpe/Sortino) not just absolute returns
3. **Check correlation matrix** to find uncorrelated strategies for diversification
4. **Review drawdown charts** to understand risk characteristics
5. **Examine trade-level data** to spot patterns or issues

### Interpretation Guide

#### Good Strategy Characteristics

- **Total Return**: > 20% annually
- **Sharpe Ratio**: > 1.5 (excellent if > 2.0)
- **Max Drawdown**: < -20%
- **Win Rate**: > 60%
- **Profit Factor**: > 2.0

#### Warning Signs

- **High volatility** with low returns
- **Large drawdowns** (> -30%)
- **Low win rate** (< 50%) unless profit factor is very high
- **High correlation** with market (no edge)

## Troubleshooting

### Dashboard Won't Start

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
uv sync  # Reinstall dependencies
```

### No Data Displayed

**Error:** "No strategies selected"

**Solution:** Select at least 2 strategies from the sidebar before running analysis.

### Charts Not Loading

**Error:** Blank chart area

**Solution:**
1. Check browser console for JavaScript errors
2. Try refreshing the page (Ctrl+R or Cmd+R)
3. Clear browser cache
4. Try a different browser

### Performance Issues

**Problem:** Dashboard is slow

**Solutions:**
1. Reduce number of strategies (< 5)
2. Use shorter time horizon
3. Close other browser tabs
4. Check system resources (RAM, CPU)

## Next Steps

### Learn More

- Read the [UI Design Specification](UI_DESIGN_SPEC.md) for detailed UI/UX information
- Explore the [API Documentation](API.md) for programmatic usage
- Check the [Examples](../examples/) directory for more code samples

### Customize

1. **Add Custom Strategies**: See `strategy_loader.py`
2. **Modify Metrics**: Edit `metrics_calculator.py`
3. **Customize Charts**: Update `ui/charts.py`
4. **Add New Visualizations**: Create new chart functions in `ui/charts.py`

### Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

- **Documentation**: See `/docs` directory
- **Issues**: Open an issue on GitHub
- **Examples**: Check `/examples` directory

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

---

**Happy Analyzing! 🚀**
