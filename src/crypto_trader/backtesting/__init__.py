"""
Backtesting Module for Crypto Trading System

This module provides comprehensive backtesting capabilities using VectorBT for
vectorized performance analysis. It includes order execution simulation, portfolio
management, and the main backtesting engine.

**Public API**:
- BacktestEngine: Main backtesting orchestrator
- OrderExecutor: Order execution simulation with fees and slippage
- PortfolioManager: Portfolio state and equity tracking
- ExecutionResult: Order execution result details
- Position: Position tracking dataclass
- PortfolioState: Portfolio state snapshot

**Usage Example**:
```python
from crypto_trader.backtesting import BacktestEngine
from crypto_trader.core.config import BacktestConfig
from crypto_trader.strategies.library.moving_average_crossover import MovingAverageCrossover

# Create strategy
strategy = MovingAverageCrossover(
    name="MA_Cross",
    config={"fast_period": 10, "slow_period": 20}
)
strategy.initialize(strategy.config)

# Configure backtest
config = BacktestConfig(
    initial_capital=10000.0,
    trading_fee_percent=0.001,
    slippage_percent=0.0005
)

# Run backtest
engine = BacktestEngine()
result = engine.run_backtest(strategy, data, config)

# Analyze results
print(f"Return: {result.metrics.total_return:.2%}")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
print(f"Trades: {result.metrics.total_trades}")
```

**Key Features**:
- Vectorized backtesting with VectorBT
- Realistic order execution simulation
- Comprehensive performance metrics
- Multiple strategy comparison
- Parameter optimization
- Portfolio tracking and equity curves
"""

from crypto_trader.backtesting.engine import BacktestEngine
from crypto_trader.backtesting.executor import ExecutionResult, OrderExecutor
from crypto_trader.backtesting.portfolio import (
    PortfolioManager,
    PortfolioState,
    Position,
)

__all__ = [
    # Main engine
    "BacktestEngine",
    # Execution
    "OrderExecutor",
    "ExecutionResult",
    # Portfolio management
    "PortfolioManager",
    "Position",
    "PortfolioState",
]
