# Crypto Trading Strategy Dashboard

A comprehensive Streamlit web dashboard for comparing, analyzing, and backtesting cryptocurrency trading strategies with interactive visualizations and advanced analytics.

## Features

### Main Dashboard (`app.py`)
- **Multi-strategy comparison** - Compare 2-10 strategies simultaneously
- **Interactive charts** - Plotly-powered equity curves, drawdowns, and distributions
- **Performance metrics** - Comprehensive table with sortable columns
- **Time horizon analysis** - Filter by different time periods
- **Correlation analysis** - Understand strategy relationships
- **Risk-return analysis** - Scatter plots and radar charts
- **Export functionality** - HTML reports, CSV data, and JSON exports

### Backtest Page
- **Individual strategy testing** - Run backtests with custom parameters
- **Parameter configuration** - Adjust strategy-specific settings
- **Real-time execution** - Watch backtest progress
- **Detailed results** - Equity curves, drawdowns, and trade analysis
- **Quality ratings** - Automated strategy quality assessment
- **Multiple export formats** - HTML, CSV, and JSON

### Comparison Page
- **Advanced analytics** - Statistical significance tests
- **Pairwise comparisons** - Compare any two strategies
- **Performance rankings** - Sort by any metric
- **Correlation matrix** - Heatmap visualization
- **Performance radar** - Multi-metric visualization
- **Filter strategies** - By Sharpe ratio, drawdown, or trade count

### Data Management Page
- **Data inventory** - View all available market data
- **Fetch new data** - Download historical data from exchanges
- **Update datasets** - Keep data current
- **Quality checks** - Identify data issues
- **Coverage visualization** - Timeline and completeness charts
- **Batch operations** - Update multiple datasets at once

## Installation

### Prerequisites
- Python 3.12+
- UV package manager (or pip)

### Install Dependencies

```bash
# From project root
cd /home/fiod/crypto

# Install with UV
uv sync

# Or with pip
pip install -e .
```

## Running the Dashboard

### Start the Application

```bash
# Navigate to project root
cd /home/fiod/crypto

# Run with UV
uv run streamlit run src/crypto_trader/web/app.py

# Or activate venv and run with streamlit
source .venv/bin/activate
streamlit run src/crypto_trader/web/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Configuration

Configure Streamlit settings in `.streamlit/config.toml`:

```toml
[server]
port = 8501
headless = false
enableCORS = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## Usage Guide

### Main Dashboard

1. **Select Strategies**
   - Use the sidebar to select 2-10 strategies
   - Filter by symbol and timeframe
   - Choose time horizon (1 month to all time)

2. **View Comparisons**
   - Navigate through tabs for different visualizations
   - Equity curves show normalized returns
   - Drawdown charts reveal risk profiles
   - Correlation matrix shows strategy relationships

3. **Export Results**
   - Click "Export HTML" for comprehensive reports
   - Click "Export CSV" for data analysis
   - Reports saved to `exports/` directory

### Running Backtests

1. **Select Strategy**
   - Choose from available strategies
   - View strategy description

2. **Configure Parameters**
   - Set symbol and timeframe
   - Adjust initial capital
   - Configure strategy-specific parameters
   - Set risk management (stop loss, take profit)
   - Define trading costs (fees, slippage)

3. **Run Backtest**
   - Click "Run Backtest" button
   - Wait for execution (progress shown)
   - Review comprehensive results

4. **Analyze Results**
   - Performance metrics overview
   - Interactive charts
   - Trade-by-trade analysis
   - Export results in multiple formats

### Comparing Strategies

1. **Select Strategies**
   - Choose 2-10 strategies for comparison
   - Select primary comparison metric

2. **View Rankings**
   - Strategies ranked by selected metric
   - Side-by-side bar charts
   - Performance radar chart

3. **Correlation Analysis**
   - View correlation heatmap
   - Understand strategy diversification
   - Identify independent strategies

4. **Statistical Tests**
   - Pairwise significance tests
   - T-statistics and p-values
   - Determine if differences are meaningful

5. **Filter Results**
   - Set minimum Sharpe ratio
   - Set maximum drawdown
   - Set minimum trade count
   - View filtered strategies

### Managing Data

1. **View Data Inventory**
   - See all available datasets
   - Check completeness and coverage
   - Filter by symbol and timeframe

2. **Fetch New Data**
   - Select symbol and timeframe
   - Choose date range
   - Select exchange
   - Click "Fetch Data"

3. **Update Datasets**
   - View datasets needing update
   - Select datasets to update
   - Run bulk updates

4. **Quality Checks**
   - Run checks on specific dataset
   - View missing data analysis
   - Check for price anomalies
   - Identify time gaps
   - Run batch quality checks

## Charts and Visualizations

### Equity Curves
- **Normalized returns** - All strategies start at 0%
- **Hover information** - Date, return, strategy name
- **Legend** - Toggle strategies on/off
- **Zoom and pan** - Interactive controls

### Drawdown Charts
- **Peak-to-trough declines** - Show risk over time
- **Multiple strategies** - Overlaid for comparison
- **Color-coded** - Easy visual identification
- **Max drawdown markers** - Highlight worst periods

### Distribution Charts
- **Violin plots** - Show return distributions
- **Box plots** - Median, quartiles, outliers
- **Comparison** - Side-by-side distributions

### Correlation Heatmap
- **Color-coded** - Red (negative) to blue (positive)
- **Values shown** - Correlation coefficients
- **Interactive** - Hover for details

### Risk-Return Scatter
- **X-axis** - Annualized volatility
- **Y-axis** - Annualized return
- **Color** - Sharpe ratio
- **Labels** - Strategy names

### Performance Radar
- **Multiple metrics** - Normalized to 0-1 scale
- **Overlaid strategies** - Easy comparison
- **5 key metrics** - Sharpe, return, win rate, profit factor, Calmar

## Performance Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio** - Return per unit of risk
- **Sortino Ratio** - Return per unit of downside risk
- **Calmar Ratio** - Return per unit of maximum drawdown

### Risk Metrics
- **Maximum Drawdown** - Largest peak-to-trough decline
- **Recovery Factor** - Net profit / max drawdown
- **Volatility** - Standard deviation of returns

### Trade Statistics
- **Win Rate** - Percentage of winning trades
- **Profit Factor** - Gross profit / gross loss
- **Expectancy** - Expected profit per trade
- **Average Win/Loss** - Mean profit/loss per trade

### Quality Rating
- **EXCELLENT** - Sharpe > 2.5, Drawdown < 15%
- **GOOD** - Sharpe > 1.5, Drawdown < 20%
- **FAIR** - Sharpe > 1.0, Drawdown < 30%
- **POOR** - Below fair thresholds

## Export Formats

### HTML Reports
- Comprehensive strategy report
- Embedded interactive charts
- Metrics tables
- Trade details
- Styled and professional

### CSV Data
- Comparison metrics
- Trade-by-trade data
- Easy Excel/Python import
- Comma-separated values

### JSON Exports
- Complete backtest results
- Machine-readable format
- All metrics and trades
- Timestamp data

## Troubleshooting

### Dashboard Won't Start

```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
uv pip install streamlit --force-reinstall

# Check for port conflicts
lsof -i :8501
```

### Charts Not Displaying

```bash
# Check Plotly installation
uv pip install plotly --force-reinstall

# Clear Streamlit cache
streamlit cache clear
```

### Import Errors

```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:/home/fiod/crypto/src"

# Or reinstall package
uv pip install -e .
```

### Performance Issues

- Reduce number of strategies compared (max 10)
- Shorten time horizon
- Clear browser cache
- Restart Streamlit server

## Architecture

### File Structure

```
src/crypto_trader/web/
├── __init__.py           # Package initialization
├── app.py                # Main dashboard application
├── utils.py              # Shared utility functions
├── README.md             # This file
└── pages/
    ├── 1_Backtest.py     # Individual strategy backtesting
    ├── 2_Comparison.py   # Multi-strategy comparison
    └── 3_Data_Management.py  # Data viewing and management
```

### Session State
- `backtest_results` - Cached backtest results
- `selected_strategies` - Currently selected strategies
- `comparison_data` - Comparison DataFrame
- `time_horizon` - Selected time period

### Data Flow
1. User selects strategies → Load backtest results
2. Results processed → Generate metrics and charts
3. Interactive charts → User exploration
4. Export actions → Generate reports

## Development

### Adding New Charts

```python
# In app.py or utils.py
import plotly.graph_objects as go

def create_custom_chart(results: List[BacktestResult]) -> go.Figure:
    fig = go.Figure()

    # Add traces
    for result in results:
        fig.add_trace(go.Scatter(...))

    # Update layout
    fig.update_layout(
        title="Custom Chart",
        template="plotly_white",
    )

    return fig

# Use in Streamlit
st.plotly_chart(create_custom_chart(results), use_container_width=True)
```

### Adding New Metrics

```python
# In comparison.py
def calculate_custom_metric(result: BacktestResult) -> float:
    # Custom calculation
    return value

# Add to comparison table
df["custom_metric"] = df.apply(lambda row: calculate_custom_metric(row), axis=1)
```

### Customizing Styling

```python
# In app.py
st.markdown(
    """
    <style>
    /* Your custom CSS */
    .custom-class {
        color: #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
```

## Best Practices

### Performance
- Limit strategies to 10 maximum
- Use session state for caching
- Implement lazy loading for charts
- Use Streamlit's @st.cache_data decorator

### User Experience
- Show loading spinners for long operations
- Provide clear error messages
- Include help text and tooltips
- Make all charts interactive

### Data Validation
- Validate date ranges
- Check for minimum data requirements
- Handle missing data gracefully
- Show warnings for data quality issues

## Future Enhancements

- Real-time data updates
- Live trading integration
- Portfolio optimization
- Machine learning insights
- Custom strategy builder
- Alert system
- Mobile-responsive design
- Multi-user support
- Cloud deployment

## Support

For issues and questions:
1. Check this README
2. Review inline documentation
3. Check Streamlit docs: https://docs.streamlit.io
4. Review Plotly docs: https://plotly.com/python/

## License

Part of the Crypto Trader project. See main project LICENSE file.
