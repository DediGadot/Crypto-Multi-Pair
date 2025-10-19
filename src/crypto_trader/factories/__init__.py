"""
Factory patterns for strategy and data pipeline creation.

This module provides factory classes for creating trading strategies
and data pipelines with consistent configuration and lifecycle management.

**Purpose**: Centralize object creation logic to ensure proper initialization,
logging, and validation across the system.

**Third-party packages**: None (pure Python patterns)

**Sample Usage**:
```python
from crypto_trader.factories import StrategyFactory

# Create a strategy with validation
strategy = StrategyFactory.create(
    name="SMA_Crossover",
    config={"fast_period": 10, "slow_period": 20}
)

# Strategy is initialized, validated, and ready to use
signals = strategy.generate_signals(data)
```
"""

from crypto_trader.factories.strategy_factory import StrategyFactory

__all__ = ["StrategyFactory"]
