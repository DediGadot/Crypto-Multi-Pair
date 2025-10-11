# Backtesting Engine Guide

## Overview

The backtesting engine uses VectorBT for high-performance vectorized backtesting of trading strategies. It provides comprehensive performance metrics, realistic order execution simulation, and portfolio tracking.

## Architecture

### Components

1. **BacktestEngine** (`engine.py`):
   - Main orchestrator for backtesting
   - Integrates with VectorBT for vectorized operations
   - Calculates comprehensive performance metrics
   - Supports multiple strategies and parameter optimization

2. **OrderExecutor** (`executor.py`):
   - Simulates realistic order execution
   - Applies fees and slippage
   - Tracks trade history
   - Supports maker/taker fee structures

3. **PortfolioManager** (`portfolio.py`):
   - Tracks portfolio state through time
   - Manages positions and cash
   - Calculates equity curves
   - Supports long-only and long-short strategies

## Quick Start

### Basic Usage

```python
from crypto_trader.backtesting import BacktestEngine
from crypto_trader.core.config import BacktestConfig
from crypto_trader.core.types import Timeframe
from crypto_trader.strategies.library.moving_average_crossover import MovingAverageCrossover
import pandas as pd

# Prepare data with indicators
data = pd.DataFrame({
    'timestamp': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
    'SMA_10': [...],  # Required by strategy
    'SMA_20': [...]   # Required by strategy
})

# Initialize strategy
strategy = MovingAverageCrossover(
    name="MA_Cross",
    config={
        "fast_period": 10,
        "slow_period": 20,
        "signal_threshold": 0.5
    }
)
strategy.initialize(strategy.config)

# Configure backtest
config = BacktestConfig(
    initial_capital=10000.0,
    trading_fee_percent=0.001,   # 0.1%
    slippage_percent=0.0005,      # 0.05%
    max_position_size=0.95        # 95% of capital
)

# Run backtest
engine = BacktestEngine()
result = engine.run_backtest(
    strategy=strategy,
    data=data,
    config=config,
    symbol="BTCUSDT",
    timeframe=Timeframe.HOUR_1
)

# Analyze results
print(f"Total Return: {result.metrics.total_return:.2%}")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")
print(f"Total Trades: {result.metrics.total_trades}")
print(f"Win Rate: {result.metrics.win_rate:.2%}")
```

### Compare Multiple Strategies

```python
from crypto_trader.backtesting import BacktestEngine
from crypto_trader.strategies.library.moving_average_crossover import MovingAverageCrossover

# Create multiple strategy configurations
strategies = [
    MovingAverageCrossover(
        name="MA_Fast",
        config={"fast_period": 5, "slow_period": 15}
    ),
    MovingAverageCrossover(
        name="MA_Medium",
        config={"fast_period": 10, "slow_period": 20}
    ),
    MovingAverageCrossover(
        name="MA_Slow",
        config={"fast_period": 20, "slow_period": 50}
    )
]

# Initialize all strategies
for strategy in strategies:
    strategy.initialize(strategy.config)

# Run backtests
engine = BacktestEngine()
results = engine.run_multiple(
    strategies=strategies,
    data=data,
    config=config
)

# Results are sorted by total return
for result in results:
    print(f"{result.strategy_name}:")
    print(f"  Return: {result.metrics.total_return:.2%}")
    print(f"  Sharpe: {result.metrics.sharpe_ratio:.2f}")
    print()
```

### Parameter Optimization

```python
from crypto_trader.backtesting import BacktestEngine

# Define parameter grid
param_grid = {
    "fast_period": [5, 10, 15],
    "slow_period": [20, 30, 40],
    "signal_threshold": [0.5, 0.6, 0.7]
}

# Run optimization (uses Sharpe ratio as metric)
engine = BacktestEngine()
best_result, best_params = engine.optimize_parameters(
    strategy=strategy,
    data=data,
    config=config,
    param_grid=param_grid
)

print(f"Best parameters: {best_params}")
print(f"Best Sharpe: {best_result.metrics.sharpe_ratio:.2f}")
print(f"Return: {best_result.metrics.total_return:.2%}")
```

## Performance Metrics

The backtesting engine calculates comprehensive metrics:

### Return Metrics
- **Total Return**: Overall percentage return
- **Final Capital**: Ending portfolio value
- **Realized P&L**: Total profit/loss from closed trades

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Maximum peak-to-trough decline
- **Calmar Ratio**: Return / Max Drawdown
- **Recovery Factor**: Net profit / Max Drawdown

### Trading Metrics
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Expectancy**: Average expected profit per trade
- **Average Win/Loss**: Mean profit per winning/losing trade
- **Max Consecutive Wins/Losses**: Longest streaks
- **Average Trade Duration**: Mean time in position

### Quality Assessment

The system automatically rates strategy quality based on risk-adjusted metrics:

- **Excellent**: Sharpe ≥ 2.0 and Max Drawdown < 15%
- **Good**: Sharpe ≥ 1.5 and Max Drawdown < 25%
- **Fair**: Sharpe ≥ 1.0 and Max Drawdown < 35%
- **Poor**: Below fair thresholds

## Configuration Options

### BacktestConfig

```python
from crypto_trader.core.config import BacktestConfig

config = BacktestConfig(
    initial_capital=10000.0,        # Starting capital
    trading_fee_percent=0.001,       # Trading fee (0.1%)
    slippage_percent=0.0005,         # Slippage (0.05%)
    max_position_size=0.95,          # Max position as % of equity
    enable_short_selling=False,      # Allow short positions
    compound_returns=True,           # Reinvest profits
    commission_type="percentage"     # "percentage" or "fixed"
)
```

### OrderExecutor Options

```python
from crypto_trader.backtesting import OrderExecutor

# Basic configuration
executor = OrderExecutor(
    fee_percent=0.001,
    slippage_percent=0.0005
)

# With maker/taker fees
executor = OrderExecutor(
    fee_percent=0.001,
    slippage_percent=0.0005,
    use_maker_taker_fees=True,
    maker_fee=0.0,      # 0% maker fee
    taker_fee=0.002     # 0.2% taker fee
)
```

## Working with Results

### BacktestResult Object

```python
# Access metrics
result.metrics.total_return
result.metrics.sharpe_ratio
result.metrics.max_drawdown
result.metrics.total_trades

# Get trade history
for trade in result.trades:
    print(f"Entry: {trade.entry_time} @ ${trade.entry_price}")
    print(f"Exit: {trade.exit_time} @ ${trade.exit_price}")
    print(f"P&L: ${trade.pnl} ({trade.pnl_percent}%)")

# Get equity curve
for timestamp, equity in result.equity_curve:
    print(f"{timestamp}: ${equity:,.2f}")

# Get summary dictionary
summary = result.summary()
# Returns: {
#     'strategy': 'MA Crossover',
#     'symbol': 'BTCUSDT',
#     'duration_days': 365,
#     'total_return': '15.00%',
#     'sharpe_ratio': '1.80',
#     ...
# }
```

### Quality Checks

```python
# Check if strategy is profitable
if result.metrics.is_profitable():
    print("Strategy is profitable!")

# Get quality rating
rating = result.metrics.risk_adjusted_quality()
print(f"Quality: {rating}")  # Excellent, Good, Fair, or Poor
```

## Best Practices

### 1. Data Preparation

Ensure your data includes all required indicators before backtesting:

```python
# Check strategy requirements
required = strategy.get_required_indicators()
print(f"Required indicators: {required}")

# Verify data has all columns
assert all(col in data.columns for col in required)
```

### 2. Realistic Parameters

Use realistic fees and slippage based on your exchange:

| Exchange | Maker Fee | Taker Fee | Typical Slippage |
|----------|-----------|-----------|------------------|
| Binance  | 0.10%     | 0.10%     | 0.05%            |
| Coinbase | 0.00%     | 0.40%     | 0.10%            |
| Kraken   | 0.16%     | 0.26%     | 0.05%            |

### 3. Walk-Forward Analysis

Split your data into training and testing periods:

```python
# Train on first 70% of data
train_data = data.iloc[:int(len(data) * 0.7)]
test_data = data.iloc[int(len(data) * 0.7):]

# Optimize on training data
best_result, best_params = engine.optimize_parameters(
    strategy=strategy,
    data=train_data,
    config=config,
    param_grid=param_grid
)

# Test on unseen data
strategy.set_parameters(best_params)
final_result = engine.run_backtest(
    strategy=strategy,
    data=test_data,
    config=config
)
```

### 4. Multiple Timeframes

Test strategies across different timeframes:

```python
timeframes = [
    Timeframe.HOUR_1,
    Timeframe.HOUR_4,
    Timeframe.DAY_1
]

for tf in timeframes:
    # Prepare data for this timeframe
    data_tf = prepare_data_for_timeframe(raw_data, tf)

    result = engine.run_backtest(
        strategy=strategy,
        data=data_tf,
        config=config,
        timeframe=tf
    )

    print(f"{tf.value}: {result.metrics.total_return:.2%}")
```

## Examples

See the `examples/` directory for complete examples:

- `backtest_ma_crossover.py`: Basic MA crossover backtest
- Additional examples coming soon...

## Limitations

1. **Market Impact**: Does not model market impact for large orders
2. **Liquidity**: Assumes sufficient liquidity at all times
3. **Order Book**: Uses simple slippage model instead of order book simulation
4. **Latency**: Does not model network latency or order delays
5. **Partial Fills**: Assumes all orders fill completely

For production trading, consider these factors and use paper trading to validate results.

## Performance Tips

1. **VectorBT**: Leverages vectorized operations for fast backtesting
2. **Caching**: Cache indicator calculations to avoid recomputation
3. **Batch Testing**: Use `run_multiple()` for comparing strategies
4. **Data Size**: For very large datasets, consider sampling or chunking

## Troubleshooting

### No Trades Executed

If your backtest shows 0 trades:

1. Check signal generation: `signals = strategy.generate_signals(data)`
2. Verify signal threshold isn't too high
3. Ensure data has sufficient crossovers/patterns
4. Check that required indicators are present

### Poor Performance

If results are worse than expected:

1. Verify data quality (no gaps, correct timeframe)
2. Check fees and slippage settings
3. Compare with buy-and-hold baseline
4. Ensure no look-ahead bias in indicators
5. Validate strategy logic

### Memory Issues

For large datasets:

1. Use chunked processing
2. Reduce equity curve granularity
3. Sample data at lower frequency
4. Use iterator-based processing

## API Reference

See inline documentation in:
- `crypto_trader/backtesting/engine.py`
- `crypto_trader/backtesting/executor.py`
- `crypto_trader/backtesting/portfolio.py`

Each module includes comprehensive docstrings and usage examples.
