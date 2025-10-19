"""
Execution Module - Backtest Execution Layer

This module provides the execution layer for running backtests across
single-pair and multi-pair strategies with proper worker pool management.

**Purpose**: Separate execution concerns from orchestration logic

**Architecture**:
- workers.py: Worker functions for parallel backtest execution

**Third-party packages**:
- concurrent.futures: https://docs.python.org/3/library/concurrent.futures.html

**Usage**:
```python
from crypto_trader.execution.workers import run_backtest_worker

result = run_backtest_worker(strategy_name, data_dict, ...)
```

Created during Phase 2 refactoring to modularize execution layer.

**Phase 2 Status**: Module structure created, import compatibility layer established.
Full worker extraction (1000+ lines) deferred to future iteration due to complex
dependencies on master.py helper functions.
"""

from crypto_trader.execution.workers import (
    run_backtest_worker,
    run_multipair_backtest_worker,
)

__all__ = [
    'run_backtest_worker',
    'run_multipair_backtest_worker',
]
