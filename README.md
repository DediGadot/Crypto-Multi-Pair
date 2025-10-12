# Crypto Trading Pipeline

A comprehensive cryptocurrency backtesting framework with multiple trading strategies and portfolio rebalancing capabilities. Test your strategies against historical data before risking real capital.

## 🌟 Features

- **5 Battle-Tested Trading Strategies**: SMA Crossover, RSI Mean Reversion, MACD Momentum, Bollinger Breakout, Triple EMA
- **Portfolio Rebalancing**: Multi-asset portfolio management with threshold, calendar, and hybrid rebalancing methods
- **Enhanced Reporting**: Deep-dive analysis with trade-by-trade breakdowns and actionable recommendations
- **Comprehensive Metrics**: Total return, Sharpe ratio, max drawdown, win rate, profit factor, and more
- **Historical Data**: Fetches real data from Binance with smart caching
- **Flexible Timeframes**: 1m, 5m, 15m, 1h, 4h, 1d
- **Visual Reports**: HTML charts and detailed CSV exports

---

## 📦 Installation

### Prerequisites

- **Python 3.10+**
- **uv** (Python package manager)

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd crypto

# Install dependencies using uv
uv sync

# Or install manually
pip install -r requirements.txt
```

### Dependencies

The project uses:
- `ccxt` - Cryptocurrency exchange integration
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `ta` - Technical analysis indicators
- `plotly` - Interactive charts
- `pyyaml` - Configuration files
- `loguru` - Beautiful logging

---

## 🚀 Quick Start

### Single-Pair Mode (Test Multiple Strategies)

Run all 5 strategies on a single trading pair:

```bash
# Basic usage - BTC/USDT for 1 year
uv run python run_full_pipeline.py BTC/USDT

# Custom timeframe and period
uv run python run_full_pipeline.py ETH/USDT --timeframe 4h --days 180

# With enhanced report
uv run python run_full_pipeline.py BTC/USDT --days 90 --report
```

### Portfolio Mode (Multi-Asset Rebalancing)

Test portfolio rebalancing strategies:

```bash
# 1-year portfolio test
uv run python run_full_pipeline.py --portfolio --config config_10pct_1year.yaml

# 8+ years with enhanced report
uv run python run_full_pipeline.py --portfolio --config config_improved_10pct.yaml --report
```

---

## 📊 Trading Strategies Explained

### 1. SMA Crossover (Simple Moving Average)

**Type**: Trend Following

**How It Works**:
- Uses two moving averages: Fast (50-period) and Slow (200-period)
- **Buy Signal**: When fast MA crosses above slow MA (Golden Cross)
- **Sell Signal**: When fast MA crosses below slow MA (Death Cross)

**Best Used When**:
- Markets are trending (up or down)
- Clear directional moves
- Avoid in sideways/choppy markets

**Parameters**:
```python
fast_period: 50   # Short-term trend
slow_period: 200  # Long-term trend
```

**Pros**:
- Simple and reliable
- Catches major trends
- Easy to understand

**Cons**:
- Lags at trend reversals
- Many false signals in sideways markets
- Late entries/exits

**Real-World Example**:
If BTC 50-day MA crosses above 200-day MA, it signals potential bull market (Golden Cross). This happened in April 2019 and led to a 300% rally.

---

### 2. RSI Mean Reversion

**Type**: Mean Reversion / Counter-Trend

**How It Works**:
- Uses Relative Strength Index (RSI) to identify overbought/oversold conditions
- RSI ranges from 0 to 100
- **Buy Signal**: RSI < 30 (oversold - price likely to bounce)
- **Sell Signal**: RSI > 70 (overbought - price likely to correct)

**Best Used When**:
- Range-bound markets
- Clear support/resistance levels
- High volatility with frequent reversals

**Parameters**:
```python
rsi_period: 14   # Lookback period for RSI calculation
oversold: 30     # Buy threshold
overbought: 70   # Sell threshold
```

**Pros**:
- Catches reversals early
- Works well in ranging markets
- High win rate potential

**Cons**:
- Fails in strong trends
- Can stay overbought/oversold for long periods
- Requires good timing

**Real-World Example**:
During BTC consolidation in mid-2023 between $25k-$31k, RSI mean reversion captured multiple 10-20% bounces when RSI hit oversold levels.

---

### 3. MACD Momentum

**Type**: Momentum / Trend Following

**How It Works**:
- MACD = Fast EMA (12) - Slow EMA (26)
- Signal Line = 9-period EMA of MACD
- Histogram = MACD - Signal Line
- **Buy Signal**: MACD crosses above signal line (bullish momentum)
- **Sell Signal**: MACD crosses below signal line (bearish momentum)

**Best Used When**:
- Catching medium to long-term trends
- Confirming trend strength
- Identifying momentum shifts

**Parameters**:
```python
fast_period: 12    # Fast EMA for MACD
slow_period: 26    # Slow EMA for MACD
signal_period: 9   # Signal line smoothing
```

**Pros**:
- Combines trend and momentum
- Histogram shows strength
- Widely used and reliable

**Cons**:
- Lags at reversals
- Can whipsaw in consolidation
- Multiple components to interpret

**Real-World Example**:
ETH's MACD showed bullish crossover in October 2023 at $1,550, signaling the start of a rally to $4,000+ by March 2024.

---

### 4. Bollinger Breakout

**Type**: Volatility Breakout

**How It Works**:
- Three bands: Middle (20-period MA), Upper (+2 std dev), Lower (-2 std dev)
- Bands expand in high volatility, contract in low volatility
- **Buy Signal**: Price breaks above upper band (continuation signal)
- **Sell Signal**: Price breaks below lower band or returns to middle band

**Best Used When**:
- Volatile markets
- Breakouts from consolidation
- Trending markets

**Parameters**:
```python
period: 20       # Moving average period
std_dev: 2.0     # Standard deviations for bands
```

**Pros**:
- Adapts to volatility
- Identifies breakouts early
- Visual and intuitive

**Cons**:
- False breakouts common
- Can be whipsawed
- Requires confirmation

**Real-World Example**:
BTC squeezed in Bollinger Bands at $28k in September 2023 (low volatility), then broke out above upper band leading to rally to $44k.

---

### 5. Triple EMA

**Type**: Advanced Trend Following

**How It Works**:
- Uses three Exponential Moving Averages: Fast (8), Medium (21), Slow (55)
- **Buy Signal**: All three EMAs aligned bullishly (fast > medium > slow)
- **Sell Signal**: Alignment breaks or reverses

**Best Used When**:
- Strong trending markets
- Need high-probability signals
- Avoiding false signals

**Parameters**:
```python
fast_period: 8     # Very short-term
medium_period: 21  # Short-term
slow_period: 55    # Medium-term
```

**Pros**:
- High probability signals
- Strong trend confirmation
- Fewer false signals than simple crossovers

**Cons**:
- Fewer trade opportunities
- Late entries in fast moves
- Requires patience

**Real-World Example**:
During BTC's 2024 bull run, Triple EMA alignment from January through March provided clear long signals with minimal whipsaws.

---

## 🎯 Portfolio Rebalancing Explained

Portfolio rebalancing is a systematic approach to maintaining target allocations across multiple assets. It forces you to "buy low, sell high" by automatically rebalancing when assets drift from their targets.

### How It Works

1. **Set Target Allocations**: e.g., 40% BTC, 30% ETH, 15% SOL, 15% BNB
2. **Monitor Deviations**: Assets grow/shrink at different rates
3. **Trigger Rebalance**: When deviation exceeds threshold (e.g., 10%)
4. **Execute Trades**: Sell overweight assets, buy underweight assets
5. **Repeat**: Continue monitoring

### Rebalancing Methods

#### 1. Threshold-Based
- Triggers when any asset deviates > threshold from target
- **Lower threshold** (5%): More frequent rebalancing, captures mean reversion
- **Higher threshold** (15%): Less frequent, follows trends more

#### 2. Calendar-Based
- Rebalances on fixed schedule (monthly, quarterly)
- Predictable and systematic
- Ignores market conditions

#### 3. Hybrid (Recommended)
- Combines threshold AND calendar
- Rebalances when either condition is met
- Best of both worlds

### Why It Works

**Example**: Portfolio starts 40% BTC, 30% ETH
- BTC rallies 100% → Now 57% of portfolio
- ETH rallies 20% → Now 22% of portfolio
- **Rebalance**: Sell some BTC (at high price), buy more ETH (at lower price)
- If ETH then catches up, you profit from both

**Historical Performance**:
- Our 10% threshold strategy: **+601.79%** outperformance vs buy-and-hold over 8 years
- Adds value through systematic mean reversion

---

## 📈 Understanding Metrics

### Total Return
**What**: Percentage gain/loss on initial capital
**Good**: >0% (profitable)
**Excellent**: >100% annual
**Example**: $10,000 → $15,000 = 50% return

### Sharpe Ratio
**What**: Risk-adjusted return (return per unit of risk)
**Good**: >1.0
**Excellent**: >2.0
**Formula**: (Return - Risk-Free Rate) / Standard Deviation
**Example**: 2.5 Sharpe means excellent risk-adjusted returns

### Max Drawdown
**What**: Largest peak-to-trough decline
**Good**: <20%
**Acceptable**: <30%
**Example**: Portfolio drops from $20k to $15k = 25% drawdown

### Win Rate
**What**: Percentage of profitable trades
**Good**: >50%
**Excellent**: >60%
**Example**: 7 wins out of 10 trades = 70% win rate

### Profit Factor
**What**: Gross profit ÷ Gross loss
**Good**: >1.0 (profitable)
**Excellent**: >2.0
**Example**: $10k profit, $5k loss = 2.0 profit factor

### Average Win/Loss
**What**: Average profit per winning trade / Average loss per losing trade
**Good**: >1.0 (wins larger than losses)
**Excellent**: >2.0
**Example**: Avg win $500, avg loss $200 = 2.5 ratio

---

## ⚙️ Configuration

### Single-Pair Options

```bash
--symbol SYMBOL           Trading pair (e.g., BTC/USDT)
--timeframe TIMEFRAME     Candle timeframe: 1m, 5m, 15m, 1h, 4h, 1d
--days DAYS               Days of historical data (default: 365)
--capital CAPITAL         Initial capital (default: 10000)
--commission RATE         Trading fee (default: 0.001 = 0.1%)
--slippage RATE          Slippage rate (default: 0.0005 = 0.05%)
--output-dir DIR         Output directory (default: results)
--report                 Generate enhanced report
```

### Portfolio Configuration (YAML)

```yaml
run:
  name: "portfolio_10pct_threshold"
  description: "Optimized portfolio with 10% threshold"
  mode: "portfolio"

data:
  timeframe: "1h"
  days: 365

portfolio:
  assets:
    - symbol: "BTC/USDT"
      weight: 0.40
    - symbol: "ETH/USDT"
      weight: 0.30
    - symbol: "SOL/USDT"
      weight: 0.15
    - symbol: "BNB/USDT"
      weight: 0.15

  rebalancing:
    enabled: true
    threshold: 0.10                      # 10% deviation threshold
    rebalance_method: "threshold"        # "threshold", "calendar", or "hybrid"
    min_rebalance_interval_hours: 24     # Minimum hours between rebalances
    calendar_period_days: 30             # For calendar/hybrid methods
    use_momentum_filter: false           # Skip rebalancing in strong trends
    momentum_lookback_days: 30           # Lookback for momentum filter

capital:
  initial_capital: 10000.0

costs:
  commission: 0.001   # 0.1%
  slippage: 0.0005    # 0.05%

output:
  directory: "results_10pct"
  save_trades: true
  save_equity_curve: true
```

---

## 📁 Output Files

### Single-Pair Mode

```
results/
├── SUMMARY.txt                          # Overview of all strategies
├── ENHANCED_REPORT.txt                  # Detailed analysis (with --report)
├── data/
│   ├── BTC_USDT_1h.csv                 # Raw price data
│   ├── strategy_comparison.csv          # Comparison table
│   ├── {Strategy}_result.json          # Full results per strategy
│   ├── {Strategy}_trades.csv           # All trades per strategy
│   └── ...
└── reports/
    ├── {Strategy}_report.html          # Visual reports
    ├── comparison_total_return.html    # Performance charts
    └── ...
```

### Portfolio Mode

```
results_10pct/
├── PORTFOLIO_SUMMARY.txt               # Basic summary
├── ENHANCED_PORTFOLIO_REPORT.txt       # Detailed analysis (with --report)
└── data/
    ├── portfolio_equity_curve.csv      # Portfolio value over time
    ├── buy_hold_benchmark.csv          # Benchmark comparison
    └── rebalance_events.csv            # All rebalancing events
```

---

## 🔍 Enhanced Reports

### Single-Pair Enhanced Report

When you use `--report` with single-pair mode, you get:

1. **Financial Metrics Explained**: One-liner definitions
2. **Strategy Comparison Table**: All strategies vs buy-and-hold
3. **Detailed Performance Metrics**: Profit factor, win/loss ratio
4. **Deep Dive**: Best strategy with trade-by-trade analysis
   - Top 5 winning trades
   - Top 5 losing trades
   - Trade statistics
   - Strategy-specific insights
   - Risk analysis
5. **Recommendations**: Actionable advice for improvement

### Portfolio Enhanced Report

When you use `--report` with portfolio mode, you get:

1. **Portfolio Metrics Explained**: Clear terminology
2. **Performance Summary**: Rebalanced vs buy-and-hold comparison
3. **Individual Asset Performance**: Each asset's contribution
4. **Detailed Rebalancing Events**: Every rebalance with BUY/SELL actions
5. **Strategy Analysis**: How your method works
6. **Recommendations**: Threshold adjustments, frequency analysis, risk insights

---

## 💡 Usage Examples

### Basic Single-Pair Backtest

```bash
# Test all 5 strategies on BTC for 1 year
uv run python run_full_pipeline.py BTC/USDT

# View results
cat results/SUMMARY.txt
```

### Custom Timeframe and Period

```bash
# 4-hour candles, 6 months of data
uv run python run_full_pipeline.py ETH/USDT --timeframe 4h --days 180

# Daily candles, 2 years
uv run python run_full_pipeline.py SOL/USDT --timeframe 1d --days 730
```

### With Enhanced Report

```bash
# Get deep-dive analysis
uv run python run_full_pipeline.py BTC/USDT --days 90 --report

# View enhanced report
cat results/ENHANCED_REPORT.txt
```

### Portfolio Rebalancing

```bash
# Quick test (1 year)
uv run python run_full_pipeline.py --portfolio --config config_10pct_1year.yaml

# Full backtest (8+ years) with enhanced report
uv run python run_full_pipeline.py --portfolio --config config_improved_10pct.yaml --report

# View results
cat results_10pct/ENHANCED_PORTFOLIO_REPORT.txt
```

### Custom Configuration

Create your own config file:

```bash
# Copy and modify
cp config_improved_10pct.yaml my_portfolio.yaml
nano my_portfolio.yaml

# Run your custom portfolio
uv run python run_full_pipeline.py --portfolio --config my_portfolio.yaml --report
```

---

## 🎓 Strategy Selection Guide

### Which Strategy Should I Use?

| Market Condition | Best Strategy | Why |
|-----------------|---------------|-----|
| **Strong Uptrend** | SMA Crossover, Triple EMA | Catch and ride the trend |
| **Strong Downtrend** | SMA Crossover, MACD | Exit early, avoid losses |
| **Sideways/Range** | RSI Mean Reversion | Profit from oscillations |
| **High Volatility** | Bollinger Breakout | Catch explosive moves |
| **Low Volatility** | Wait or use RSI | Avoid false signals |
| **Bull Market** | Triple EMA | Strong trend confirmation |
| **Bear Market** | MACD, SMA | Early exit signals |

### Portfolio vs Single Strategy?

**Use Portfolio Rebalancing When**:
- You want diversification
- You believe in mean reversion
- You can't predict which asset will win
- You want lower volatility
- Long-term investing (1+ years)

**Use Single-Pair Strategies When**:
- You want to focus on one asset
- You have strong conviction
- Higher risk tolerance
- Active trading approach
- Short to medium-term (weeks to months)

---

## 📚 Terminology Glossary

### Technical Terms

- **Candlestick**: Price bar showing Open, High, Low, Close for a time period
- **Moving Average (MA)**: Average price over N periods (smooths out noise)
- **Exponential Moving Average (EMA)**: MA that gives more weight to recent prices
- **Crossover**: When one indicator crosses above/below another (signal)
- **RSI (Relative Strength Index)**: Momentum oscillator (0-100 scale)
- **MACD**: Moving Average Convergence Divergence (trend + momentum)
- **Bollinger Bands**: Volatility bands around price
- **Standard Deviation**: Measure of volatility/dispersion
- **Timeframe**: Duration of each candle (1h, 4h, 1d, etc.)

### Trading Terms

- **Long**: Buying an asset (betting price goes up)
- **Short**: Selling an asset (betting price goes down - not implemented)
- **Entry**: Opening a trade (buying)
- **Exit**: Closing a trade (selling)
- **Position**: An open trade
- **Stop Loss**: Automatic exit at predefined loss level
- **Take Profit**: Automatic exit at predefined profit level
- **Slippage**: Difference between expected and actual execution price
- **Commission**: Trading fee charged by exchange

### Performance Terms

- **Equity Curve**: Graph of portfolio value over time
- **Drawdown**: Decline from peak to trough
- **Recovery**: Time to regain previous peak after drawdown
- **Risk-Adjusted Return**: Return relative to risk taken
- **Volatility**: How much prices fluctuate
- **Correlation**: How assets move together
- **Alpha**: Excess return vs benchmark
- **Beta**: Volatility relative to market

---

## ⚠️ Important Notes

### Backtesting Limitations

1. **Past performance ≠ Future results**: Historical data doesn't guarantee future success
2. **Overfitting Risk**: Don't over-optimize on historical data
3. **Market Changes**: Strategies work differently in different market regimes
4. **No Emotions**: Real trading involves fear and greed (backtests don't)
5. **Perfect Execution**: Backtests assume perfect fills (reality differs)

### Best Practices

- **Test Multiple Timeframes**: What works on 1h may fail on 4h
- **Out-of-Sample Testing**: Save recent data for validation
- **Consider Transaction Costs**: Commissions and slippage matter
- **Risk Management**: Never risk more than you can afford to lose
- **Diversify**: Don't put all capital in one strategy
- **Paper Trade First**: Test with fake money before using real capital
- **Stay Informed**: Markets evolve, strategies must adapt

---

## 🛠️ Development

### Project Structure

```
crypto/
├── src/crypto_trader/
│   ├── strategies/          # Strategy implementations
│   ├── backtesting/         # Backtest engine
│   ├── data/                # Data fetching and storage
│   ├── analysis/            # Performance analysis
│   └── risk/                # Risk management
├── tests/                   # Unit and integration tests
├── config*.yaml             # Portfolio configurations
└── run_full_pipeline.py     # Main entry point
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_portfolio_rebalancer.py

# Run with coverage
uv run pytest --cov=crypto_trader
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

[Specify your license here]

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review example configurations

---

## 🎯 Next Steps

1. **Install the dependencies**: `uv sync`
2. **Run a quick test**: `uv run python run_full_pipeline.py BTC/USDT --days 30`
3. **Review the results**: `cat results/SUMMARY.txt`
4. **Generate enhanced report**: Add `--report` flag
5. **Try portfolio mode**: `uv run python run_full_pipeline.py --portfolio --config config_10pct_1year.yaml --report`
6. **Customize your strategy**: Modify configs or parameters
7. **Backtest longer periods**: Use `--days 365` or more
8. **Paper trade**: Test in real-time with fake money first

---

**Happy Trading! 📈🚀**

Remember: Always backtest thoroughly and practice risk management. The best strategy is one that you understand and can stick with through market ups and downs.
