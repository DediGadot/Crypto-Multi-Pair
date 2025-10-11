# Crypto Trader - Modular Algorithmic Trading System

A comprehensive, modular cryptocurrency trading and analysis platform built for backtesting, strategy development, and systematic trading.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Modular Architecture**: Clean separation of concerns across 7 layers (UI, Orchestration, Strategy, Backtesting, Analysis, Data, Storage)
- **5 Starter Strategies**: SMA Crossover, RSI Mean Reversion, MACD Momentum, Bollinger Breakout, Triple EMA
- **Plugin System**: Easy strategy registration with decorators
- **Comprehensive Backtesting**: VectorBT-powered vectorized backtesting engine
- **20+ Performance Metrics**: Sharpe, Sortino, Calmar, Max DD, Win Rate, Profit Factor, and more
- **Risk Management**: Position sizing (Fixed Fraction, Kelly, Volatility-based, Risk Parity), risk limits, drawdown tracking
- **Data Integration**: Binance integration via CCXT, CSV storage, caching layer
- **Interactive Dashboard**: Streamlit-based web interface for strategy comparison
- **CLI Tools**: Typer-based command-line interface
- **Comprehensive Analysis**: Strategy comparison, correlation analysis, statistical significance testing
- **Rich Reporting**: HTML reports, JSON/CSV exports, interactive Plotly charts

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd crypto

# Install dependencies using uv
uv sync

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### Quick Analysis - Full Pipeline Runner

The fastest way to analyze any trading pair is using the full pipeline runner:

```bash
# Run complete analysis with one command (all 5 strategies, full reports)
uv run python run_full_pipeline.py BTC/USDT

# This automatically:
# 1. Fetches 1 year of hourly data from Binance
# 2. Runs all 5 strategies with optimized parameters
# 3. Generates HTML reports, comparison charts, and CSV exports
# 4. Creates a summary report with best performers
# Results saved to: results/ directory
```

**Customize the analysis:**
```bash
# Different timeframe and period
uv run python run_full_pipeline.py ETH/USDT --timeframe 4h --days 180

# Higher capital
uv run python run_full_pipeline.py BTC/USDT --capital 100000

# Save to custom directory
uv run python run_full_pipeline.py SOL/USDT --output-dir sol_analysis
```

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for complete documentation.

### Running Individual Backtests (CLI)

For more control, use the CLI commands:

```bash
# Fetch BTC/USDT data
uv run crypto-trader data fetch BTC/USDT --timeframe 1h --days 30

# Run a backtest
uv run crypto-trader backtest run SMA_Crossover BTC/USDT 1h \
  --fast-period 10 --slow-period 20 --capital 10000

# Compare multiple strategies
uv run crypto-trader backtest compare BTC/USDT 1h \
  --strategies SMA_Crossover RSI_MeanReversion MACD_Momentum \
  --capital 10000
```

### Launching the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/crypto_trader/web/app.py
```

Navigate to `http://localhost:8501` to access the interactive dashboard.

## Project Structure

```
crypto/
├── data/                      # OHLCV data storage (CSV files)
│   └── ohlcv/
│       ├── BTC_USDT/
│       └── ETH_USDT/
├── docs/                      # Documentation
├── src/
│   └── crypto_trader/
│       ├── core/              # Core types, config, exceptions
│       ├── data/              # Data fetchers, storage, cache
│       ├── strategies/        # Strategy framework & library
│       │   ├── base.py        # BaseStrategy abstract class
│       │   ├── registry.py    # Strategy registry (singleton)
│       │   └── library/       # 5 pre-built strategies
│       ├── backtesting/       # Backtesting engine (VectorBT)
│       ├── analysis/          # Metrics, comparison, reporting
│       ├── risk/              # Position sizing, limits, manager
│       ├── cli/               # CLI commands (Typer)
│       └── web/               # Streamlit dashboard
├── tests/                     # Test suite
│   └── test_end_to_end.py    # Comprehensive integration test
├── run_full_pipeline.py       # 🚀 Full pipeline runner script
├── PIPELINE_GUIDE.md          # Pipeline runner documentation
├── requirements.txt           # pip dependencies
├── requirements-dev.txt       # Development dependencies
├── pyproject.toml             # Project configuration
└── README.md
```

## Available Strategies

### 1. SMA Crossover
Classic moving average crossover (Golden/Death Cross)
- **Tags**: `trend_following`, `moving_average`, `crossover`
- **Parameters**: `fast_period` (default: 50), `slow_period` (default: 200)
- **Signals**: BUY on golden cross, SELL on death cross

### 2. RSI Mean Reversion
RSI-based oversold/overbought strategy
- **Tags**: `mean_reversion`, `rsi`, `oscillator`
- **Parameters**: `rsi_period` (14), `oversold` (30), `overbought` (70)
- **Signals**: BUY when RSI < oversold, SELL when RSI > overbought

### 3. MACD Momentum
MACD signal line crossover
- **Tags**: `momentum`, `macd`, `crossover`
- **Parameters**: `fast_period` (12), `slow_period` (26), `signal_period` (9)
- **Signals**: BUY on bullish crossover, SELL on bearish crossover

### 4. Bollinger Breakout
Bollinger Bands volatility breakout
- **Tags**: `volatility`, `bollinger_bands`, `breakout`
- **Parameters**: `period` (20), `std_dev` (2.0)
- **Signals**: BUY on upper band breakout, SELL on lower band breakout

### 5. Triple EMA
Triple EMA trend filter with reduced lag
- **Tags**: `trend_following`, `ema`, `crossover`, `trend_filter`
- **Parameters**: `fast_period` (8), `medium_period` (21), `slow_period` (55)
- **Signals**: BUY when all EMAs align bullish, SELL when bearish

## Creating Custom Strategies

```python
from crypto_trader.strategies import register_strategy, BaseStrategy, SignalType

@register_strategy(
    name="MyStrategy",
    description="Custom strategy description",
    tags=["custom", "tag"]
)
class MyStrategy(BaseStrategy):
    """Custom strategy implementation."""

    def __init__(self):
        super().__init__(name="MyStrategy")

    def initialize(self, config: dict) -> None:
        """Initialize strategy parameters."""
        self.param1 = config.get("param1", 10)
        self.param2 = config.get("param2", 20)

    def generate_signals(self, data):
        """Generate trading signals."""
        signals = data.copy()
        signals['signal'] = SignalType.HOLD.value

        # Your strategy logic here
        # Set signals['signal'] to 'BUY', 'SELL', or 'HOLD'

        return signals
```

## CLI Commands

### Data Management
```bash
# Fetch data
crypto-trader data fetch SYMBOL --timeframe TIMEFRAME --days DAYS

# Update existing data
crypto-trader data update SYMBOL TIMEFRAME

# List available data
crypto-trader data list

# Validate data integrity
crypto-trader data validate SYMBOL TIMEFRAME
```

### Strategy Management
```bash
# List all strategies
crypto-trader strategy list

# Filter by tags
crypto-trader strategy list --tags trend_following

# Get strategy info
crypto-trader strategy info SMA_Crossover

# Test strategy on data
crypto-trader strategy test SMA_Crossover BTC/USDT 1h
```

### Backtesting
```bash
# Run single strategy
crypto-trader backtest run STRATEGY SYMBOL TIMEFRAME [OPTIONS]

# Compare multiple strategies
crypto-trader backtest compare SYMBOL TIMEFRAME \
  --strategies STRATEGY1 STRATEGY2 [OPTIONS]

# Optimize parameters
crypto-trader backtest optimize STRATEGY SYMBOL TIMEFRAME [OPTIONS]

# Generate HTML report
crypto-trader backtest report RESULT_FILE --output report.html
```

## Configuration

### Risk Management
Configure risk parameters in your backtest:

```python
from crypto_trader.core.config import RiskConfig

risk_config = RiskConfig(
    max_position_risk=0.02,        # 2% max risk per position
    max_portfolio_risk=0.10,       # 10% max total portfolio risk
    max_daily_loss=0.05,           # 5% max daily loss
    max_drawdown=0.15,             # 15% max drawdown
    position_sizing_method="fixed_percent",  # or kelly, volatility, risk_parity
)
```

### Backtesting Parameters
```python
from crypto_trader.core.config import BacktestConfig

backtest_config = BacktestConfig(
    initial_capital=10000.0,
    commission=0.001,              # 0.1% commission
    slippage=0.0005,               # 0.05% slippage
)
```

## Performance Metrics

The system calculates 20+ performance metrics:

- **Returns**: Total Return, Annual Return, Monthly Return
- **Risk-Adjusted**: Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Risk Metrics**: Max Drawdown, Volatility, Downside Deviation
- **Trade Statistics**: Win Rate, Profit Factor, Total Trades, Avg Win/Loss
- **Quality Metrics**: Risk-Adjusted Quality Score

## Data Sources

- **Exchange**: Binance (via CCXT)
- **Supported Symbols**: All Binance spot pairs
- **Timeframes**: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
- **Storage**: CSV files in `data/ohlcv/{symbol}/{timeframe}.csv`
- **Caching**: In-memory LRU cache with TTL (1700x faster than API)

## Testing

### Run End-to-End Test
```bash
uv run python tests/test_end_to_end.py
```

This comprehensive test validates:
1. Configuration system
2. Data layer (fetching, storage, caching)
3. Strategy framework
4. Backtesting engine
5. Analysis and comparison
6. Risk management
7. Report generation

### Validation Tests
All modules include validation functions:
```bash
# Validate individual modules
uv run python src/crypto_trader/data/storage.py
uv run python src/crypto_trader/strategies/library/__init__.py
uv run python src/crypto_trader/analysis/metrics.py
```

## Architecture

The system follows a 7-layer architecture:

1. **UI Layer**: Streamlit dashboard, CLI commands
2. **Orchestration Layer**: Workflow coordination
3. **Strategy Layer**: Strategy framework, registry, library
4. **Backtesting Layer**: VectorBT integration, portfolio simulation
5. **Analysis Layer**: Metrics calculation, comparison, reporting
6. **Data Layer**: CCXT integration, caching
7. **Storage Layer**: CSV storage, data validation

## Dependencies

Key dependencies:
- **Python**: 3.12+
- **Data**: `ccxt`, `pandas`, `numpy`
- **Analysis**: `pandas-ta` (150+ indicators), `vectorbt`
- **Backtesting**: `vectorbt`
- **Web**: `streamlit`, `fastapi`
- **CLI**: `typer`, `rich`
- **Visualization**: `plotly`, `matplotlib`
- **Validation**: `pydantic`
- **Logging**: `loguru`

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all validation tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [VectorBT](https://vectorbt.dev/) for high-performance backtesting
- Exchange data via [CCXT](https://github.com/ccxt/ccxt)
- Technical indicators from [pandas-ta](https://github.com/twopirllc/pandas-ta)
- Dashboard powered by [Streamlit](https://streamlit.io/)

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Run validation tests to verify your setup

---

**Disclaimer**: This software is for educational and research purposes only. Cryptocurrency trading carries significant risk. Always do your own research and never invest more than you can afford to lose.
