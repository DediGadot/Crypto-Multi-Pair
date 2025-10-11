# Dashboard Quick Start Guide

Get the Crypto Trading Strategy Dashboard up and running in 5 minutes.

## Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip
- Terminal/command line access

## Step 1: Install Dependencies

```bash
# Navigate to project root
cd /home/fiod/crypto

# Install with UV (recommended)
uv sync

# Or with pip
pip install -e .
```

## Step 2: Launch Dashboard

### Option A: Using Launcher Script (Easiest)

```bash
# Make script executable (first time only)
chmod +x run_dashboard.sh

# Run the dashboard
./run_dashboard.sh
```

### Option B: Direct Command

```bash
# Using UV
uv run streamlit run src/crypto_trader/web/app.py

# Or using activated venv
source .venv/bin/activate
streamlit run src/crypto_trader/web/app.py
```

## Step 3: Access Dashboard

Open your browser and go to:
```
http://localhost:8501
```

The dashboard should automatically open. If not, copy the URL from the terminal.

## Quick Tour

### Main Dashboard
1. **Sidebar** - Select 2-10 strategies to compare
2. **Time Horizon** - Choose your analysis period
3. **Charts Tab** - View equity curves and performance
4. **Metrics Tab** - See detailed performance table

### Backtest Page
1. Click "Backtest" in the left sidebar
2. Select a strategy from the dropdown
3. Configure parameters (symbol, timeframe, capital)
4. Click "Run Backtest"
5. View results and export reports

### Comparison Page
1. Click "Comparison" in the left sidebar
2. Select multiple strategies
3. Choose primary comparison metric
4. Explore rankings, charts, and statistics

### Data Management Page
1. Click "Data Management" in the left sidebar
2. View available data inventory
3. Fetch new data or update existing datasets
4. Run quality checks

## Common Tasks

### Compare Multiple Strategies

1. Open main dashboard
2. In sidebar, select 2-10 strategies
3. Click through tabs to see different visualizations
4. Export results using sidebar buttons

### Run a Backtest

1. Go to Backtest page (sidebar)
2. Select strategy: "MA Crossover"
3. Set parameters:
   - Symbol: BTCUSDT
   - Timeframe: 4h
   - Initial Capital: $10,000
4. Click "Run Backtest"
5. Export HTML report when done

### Check Data Quality

1. Go to Data Management page
2. Click "Quality Checks" tab
3. Select symbol and timeframe
4. Click "Run Quality Check"
5. Review results

## Troubleshooting

### Port Already in Use

```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Or use different port
streamlit run src/crypto_trader/web/app.py --server.port 8502
```

### Import Errors

```bash
# Reinstall dependencies
uv sync --force

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/fiod/crypto/src"
```

### Charts Not Showing

```bash
# Clear Streamlit cache
streamlit cache clear

# Restart dashboard
```

### Slow Performance

- Reduce number of strategies (max 10)
- Shorten time horizon
- Clear browser cache
- Restart dashboard

## Next Steps

1. **Explore Charts** - Try all visualization tabs
2. **Run Backtests** - Test different strategies
3. **Compare Strategies** - Use statistical tests
4. **Export Reports** - Generate HTML/CSV exports
5. **Check Data** - Ensure quality is good

## Tips

- **Hover over charts** - See detailed information
- **Use filters** - Narrow down results
- **Export often** - Save interesting findings
- **Try different metrics** - Sort by Sharpe, return, etc.
- **Check correlations** - Find diversified strategies

## Configuration

Create `.streamlit/config.toml` to customize:

```toml
[server]
port = 8501
headless = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"

[browser]
gatherUsageStats = false
```

## Demo Mode

The dashboard includes sample data for demonstration. To use real data:

1. Go to Data Management page
2. Fetch historical data for your symbols
3. Run backtests with real data
4. Compare results

## Help

- **Inline Help** - Hover over ℹ️ icons
- **README** - See `src/crypto_trader/web/README.md`
- **Docs** - Check `docs/` directory
- **Streamlit Docs** - https://docs.streamlit.io

## Stopping the Dashboard

Press `Ctrl+C` in the terminal where the dashboard is running.

---

**Ready to analyze strategies?** 🚀

Run `./run_dashboard.sh` and start exploring!
