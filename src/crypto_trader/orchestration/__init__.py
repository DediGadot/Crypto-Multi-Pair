"""
Orchestration Layer for Crypto Trading Analysis

This module provides high-level orchestration classes for running
comprehensive strategy backtests across multiple timeframes and symbols.

**Purpose**: Orchestrate backtest execution and strategy analysis

**Key Classes**:
- HorizonConfig: Configuration for time horizon tests
- StrategyScore: Aggregated scoring for strategies
- MasterStrategyAnalyzer: Main orchestration engine

**Third-party packages**: None (orchestration layer)

**Sample Input**:
```python
from crypto_trader.orchestration import MasterStrategyAnalyzer, HorizonConfig

horizons = [
    HorizonConfig("30d", 30, "30 days"),
    HorizonConfig("90d", 90, "90 days"),
]

analyzer = MasterStrategyAnalyzer(
    symbol="BTC/USD",
    timeframe="1h",
    horizons=horizons,
    workers=4,
    quick_mode=False,
    multi_pair=False,
    output_dir="results"
)

analyzer.run()
```

**Expected Output**:
Comprehensive backtest results and strategy rankings.

Extracted from master.py during Phase 3 refactoring.
"""

from crypto_trader.orchestration.analyzer import (
    HorizonConfig,
    StrategyScore,
    MasterStrategyAnalyzer,
)

__all__ = [
    'HorizonConfig',
    'StrategyScore',
    'MasterStrategyAnalyzer',
]
