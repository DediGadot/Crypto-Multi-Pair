# Full Pipeline Runner - Quick Start Guide

The `run_full_pipeline.py` script runs the complete trading analysis pipeline for any symbol pair from Binance.

## What It Does

1. **Fetches Data**: Downloads all available historical data from Binance
2. **Runs All 5 Strategies**: Backtests every strategy with optimized parameters
3. **Generates Reports**: Creates comprehensive HTML, JSON, and CSV reports
4. **Compares Performance**: Analyzes and ranks all strategies
5. **Saves Everything**: Organized output with charts, data, and summaries

## Quick Start

### Basic Usage (Defaults)
```bash
# Run with all defaults: 1 year of 1h data, $10k capital
uv run python run_full_pipeline.py BTC/USDT
```

This creates a `results/` directory with:
- Individual strategy HTML reports
- Comparison charts
- Trade logs (CSV)
- Performance data (JSON)
- Summary report (TXT)

## Common Use Cases

### 1. Quick Analysis (Recent Data)
```bash
# Last 30 days, 1h timeframe
uv run python run_full_pipeline.py BTC/USDT --days 30
```

### 2. Long-Term Backtesting
```bash
# 2 years of daily data
uv run python run_full_pipeline.py BTC/USDT --timeframe 1d --days 730
```

### 3. Different Trading Pair
```bash
# Ethereum with 6 months of 4h data
uv run python run_full_pipeline.py ETH/USDT --timeframe 4h --days 180
```

### 4. High Capital Scenario
```bash
# $100k initial capital
uv run python run_full_pipeline.py BTC/USDT --capital 100000
```

### 5. Custom Output Location
```bash
# Save results to specific directory
uv run python run_full_pipeline.py SOL/USDT --output-dir sol_analysis
```

### 6. Lower Trading Costs
```bash
# VIP trading tier: 0.04% commission
uv run python run_full_pipeline.py BTC/USDT --commission 0.0004
```

## All Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | *required* | Trading pair (e.g., BTC/USDT, ETH/USDT) |
| `--timeframe` | `1h` | Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d) |
| `--days` | `365` | Number of days of historical data |
| `--capital` | `10000` | Initial capital for backtests |
| `--commission` | `0.001` | Trading commission rate (0.1%) |
| `--slippage` | `0.0005` | Slippage rate (0.05%) |
| `--output-dir` | `results` | Directory for saving results |
| `--max-position-risk` | `0.02` | Max risk per position (2%) |
| `--max-portfolio-risk` | `0.10` | Max total portfolio risk (10%) |

## Output Structure

After running the pipeline, you'll get:

```
results/
├── SUMMARY.txt                      # Quick overview of results
├── pipeline_YYYYMMDD_HHMMSS.log    # Detailed execution log
├── reports/                         # HTML reports
│   ├── SMA_Crossover_report.html
│   ├── RSI_MeanReversion_report.html
│   ├── MACD_Momentum_report.html
│   ├── BollingerBreakout_report.html
│   ├── TripleEMA_report.html
│   ├── comparison_total_return.html
│   ├── comparison_sharpe_ratio.html
│   ├── comparison_max_drawdown.html
│   └── comparison_win_rate.html
└── data/                            # Raw data exports
    ├── BTC_USDT_1h.csv              # OHLCV data
    ├── strategy_comparison.csv       # Performance comparison
    ├── SMA_Crossover_result.json    # Detailed results (JSON)
    ├── SMA_Crossover_trades.csv     # Trade log
    ├── RSI_MeanReversion_result.json
    ├── RSI_MeanReversion_trades.csv
    └── ... (more strategy results)
```

## Example Workflows

### Workflow 1: Compare Trading Pairs
```bash
# Run pipeline for multiple pairs
uv run python run_full_pipeline.py BTC/USDT --output-dir btc_results
uv run python run_full_pipeline.py ETH/USDT --output-dir eth_results
uv run python run_full_pipeline.py SOL/USDT --output-dir sol_results

# Compare SUMMARY.txt files to find best pair
```

### Workflow 2: Test Different Timeframes
```bash
# Same pair, different timeframes
uv run python run_full_pipeline.py BTC/USDT --timeframe 1h --output-dir btc_1h
uv run python run_full_pipeline.py BTC/USDT --timeframe 4h --output-dir btc_4h
uv run python run_full_pipeline.py BTC/USDT --timeframe 1d --output-dir btc_1d
```

### Workflow 3: Parameter Sensitivity
```bash
# Test different capital levels
uv run python run_full_pipeline.py BTC/USDT --capital 10000 --output-dir capital_10k
uv run python run_full_pipeline.py BTC/USDT --capital 50000 --output-dir capital_50k
uv run python run_full_pipeline.py BTC/USDT --capital 100000 --output-dir capital_100k
```

## Reading the Results

### 1. Quick Overview
Start with `SUMMARY.txt` for:
- Best performing strategy by Sharpe ratio
- Best total return
- Best risk-adjusted performance

### 2. Detailed Analysis
Open HTML reports in browser:
- `reports/STRATEGY_NAME_report.html` - Individual strategy performance
- `reports/comparison_*.html` - Visual comparisons

### 3. Raw Data
Use CSV/JSON files for:
- Custom analysis in Excel/Python
- Trade-by-trade review
- Further statistical analysis

## Strategies Tested

The pipeline automatically tests these 5 strategies:

1. **SMA Crossover** (50/200) - Classic trend following
2. **RSI Mean Reversion** (14, 30/70) - Counter-trend trading
3. **MACD Momentum** (12/26/9) - Momentum signals
4. **Bollinger Breakout** (20, 2.0) - Volatility breakouts
5. **Triple EMA** (8/21/55) - Advanced trend filter

## Performance Tips

### Faster Execution
- Use shorter time periods: `--days 90`
- Use higher timeframes: `--timeframe 4h` or `--timeframe 1d`

### More Comprehensive Analysis
- Use longer periods: `--days 730` (2 years)
- Use lower timeframes: `--timeframe 1h` or `--timeframe 15m`

### Memory Considerations
- 1m timeframe with 365 days = ~500k candles (high memory)
- 1h timeframe with 365 days = ~8.7k candles (moderate)
- 1d timeframe with 365 days = ~365 candles (low memory)

## Troubleshooting

### "No data fetched"
- Check symbol format: must be `BASE/QUOTE` (e.g., `BTC/USDT`)
- Verify symbol exists on Binance
- Try reducing `--days` (some pairs have limited history)

### "Strategy failed"
- Check log file in output directory
- Verify sufficient data for strategy parameters
- Some strategies need minimum data (e.g., SMA 200 needs 200+ candles)

### Slow execution
- Reduce `--days` parameter
- Use higher timeframe (1d instead of 1h)
- Data fetching is cached after first run

## Advanced Usage

### Combining with Other Tools
```bash
# Run pipeline and generate custom analysis
uv run python run_full_pipeline.py BTC/USDT --output-dir btc_analysis

# Then analyze with custom script
uv run python my_custom_analysis.py --input btc_analysis/data/
```

### Automated Daily Runs
```bash
# Create a daily cron job (Linux/Mac)
# Add to crontab: 0 0 * * * /path/to/run_daily.sh

#!/bin/bash
cd /home/fiod/crypto
source venv/bin/activate
uv run python run_full_pipeline.py BTC/USDT --days 365 \\
    --output-dir daily_$(date +%Y%m%d)
```

## Notes

- First run downloads data and may take 1-5 minutes
- Subsequent runs use cached data (much faster)
- All strategies use conservative default parameters
- Commission and slippage are included in results
- Results are deterministic (same data = same results)

## Support

For issues or questions:
1. Check the log file in output directory
2. Review `SUMMARY.txt` for high-level errors
3. Consult main README.md for system documentation
4. Run validation: `uv run python tests/test_end_to_end.py`
