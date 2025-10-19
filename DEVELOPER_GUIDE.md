# CRYPTO TRADER - DEVELOPER GUIDE

**Last Updated**: 2025-10-19 (Post Phase 4 Refactoring)
**Target Audience**: Developers contributing to or extending the codebase

---

## Quick Start

### Prerequisites
- Python 3.12+
- uv (package manager)
- Git

### Installation

```bash
# Clone repository
git clone <repo-url>
cd crypto-trader

# Install dependencies
uv sync

# Verify installation
uv run python master.py --help
```

### First Run

```bash
# Quick analysis (2-3 minutes)
uv run python master.py --quick

# Full analysis (custom horizons)
uv run python master.py --horizons 30 90 180 365

# View results
open master_results_*/MASTER_REPORT.html
```

---

## Project Structure

```
crypto-trader/
├── master.py                 # Entry point (64 lines)
├── src/crypto_trader/
│   ├── cli/                  # CLI layer (229 lines)
│   │   └── commands/
│   │       └── analyze.py    # Master analysis command
│   ├── orchestration/        # Orchestration (2,714 lines)
│   │   └── analyzer.py       # MasterStrategyAnalyzer
│   ├── execution/            # Execution (2,204 lines)
│   │   ├── workers.py        # Parallel workers
│   │   ├── data_utils.py     # Data manipulation
│   │   ├── metric_utils.py   # Performance metrics
│   │   ├── error_utils.py    # Error handling
│   │   └── logging_utils.py  # Enhanced logging
│   ├── reports/              # Reports (777 lines)
│   │   └── formatters/html.py
│   ├── strategies/           # Trading strategies
│   ├── backtesting/          # Backtest engine
│   ├── data/                 # Data fetchers
│   └── core/                 # Core types & config
├── tests/                    # Test suite
└── data/                     # Cached data

Total modular code: ~6,000 lines across 4 specialized modules
```

---

## Development Workflow

### 1. Adding a New Strategy

Strategies are **auto-discovered** - just create and register!

```python
# File: src/crypto_trader/strategies/my_strategy.py

from crypto_trader.strategies.base import BaseStrategy
from crypto_trader.strategies.registry import register_strategy

@register_strategy
class MyStrategy(BaseStrategy):
    """My custom strategy."""

    name = "MyStrategy"
    description = "Description of what it does"

    def __init__(self, **params):
        super().__init__(**params)
        self.my_param = params.get('my_param', 10)

    def generate_signals(self, data):
        """
        Generate buy/sell signals.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
        """
        signals = data.copy()
        signals['signal'] = 0

        # Your logic here
        # Example: Simple moving average crossover
        signals['sma_fast'] = signals['close'].rolling(5).mean()
        signals['sma_slow'] = signals['close'].rolling(20).mean()

        signals.loc[signals['sma_fast'] > signals['sma_slow'], 'signal'] = 1
        signals.loc[signals['sma_fast'] < signals['sma_slow'], 'signal'] = -1

        return signals
```

**Test it**:
```bash
# Strategy is auto-discovered and tested!
uv run python master.py --quick
```

### 2. Adding a New CLI Command

```python
# File: src/crypto_trader/cli/commands/my_command.py

import typer
from loguru import logger

app = typer.Typer()

@app.command()
def my_command(
    param1: str = typer.Option("default", "--param1", "-p"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Description of what this command does.

    Examples:
        python master.py my-command --param1 value
    """
    logger.info(f"Running my_command with param1={param1}")

    # Your logic here
    # ...

    logger.success("Command completed!")
```

**Register it**:
```python
# File: src/crypto_trader/cli/commands/__init__.py

from crypto_trader.cli.commands import backtest, data, strategy
from crypto_trader.cli.commands.analyze import app, analyze
from crypto_trader.cli.commands.my_command import my_command  # Add this

__all__ = ["data", "strategy", "backtest", "analyze", "app", "my_command"]
```

**Test it**:
```bash
python master.py my-command --help
python master.py my-command --param1 test
```

### 3. Modifying the Orchestrator

```python
# File: src/crypto_trader/orchestration/analyzer.py

class MasterStrategyAnalyzer:

    def my_new_feature(self):
        """Add a new feature to the analyzer."""
        logger.info("Running new feature...")

        # Your code here
        # Can access:
        # - self.symbol
        # - self.timeframe
        # - self.fetcher
        # - self.engine
        # - self.all_results

        return result

    def run(self):
        """Main entry point."""
        # ... existing code ...

        # Add your feature to the workflow
        self.my_new_feature()

        # ... existing code ...
```

### 4. Adding Utilities

```python
# File: src/crypto_trader/execution/my_utils.py

"""
My Utilities

**Purpose**: Description of what these utilities do

**Functions**:
- my_util_function: Does X

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/

**Sample Input**:
```python
result = my_util_function(data, param=10)
```

**Expected Output**:
Transformed data with additional columns.
"""

import pandas as pd
from loguru import logger

def my_util_function(data: pd.DataFrame, param: int = 10) -> pd.DataFrame:
    """
    Transform data in some way.

    Args:
        data: Input DataFrame
        param: Configuration parameter

    Returns:
        Transformed DataFrame
    """
    result = data.copy()

    # Your logic
    # ...

    return result


if __name__ == "__main__":
    """Validation block"""
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: Basic functionality
    total_tests += 1
    print("Test 1: Basic functionality")
    try:
        test_data = pd.DataFrame({'a': [1, 2, 3]})
        result = my_util_function(test_data, param=5)

        if result is None:
            all_validation_failures.append("Function returned None")
        else:
            print(f"  ✓ Function executed successfully")

    except Exception as e:
        all_validation_failures.append(f"Test 1 failed: {e}")

    # Final validation
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests passed")
        sys.exit(0)
```

**Test it**:
```bash
uv run python src/crypto_trader/execution/my_utils.py
```

---

## Testing

### Module-Level Tests

Every module MUST have a validation block:

```python
if __name__ == "__main__":
    """Validation block"""
    # Test with real data
    # Track ALL failures
    # Exit with proper code (0=success, 1=failure)
```

**Run tests**:
```bash
# Test specific module
uv run python src/crypto_trader/execution/data_utils.py

# Test all modules
for f in src/crypto_trader/**/*.py; do
    if grep -q "if __name__" $f; then
        echo "Testing $f"
        uv run python $f || echo "FAILED: $f"
    fi
done
```

### End-to-End Tests

```bash
# Quick validation
uv run python master.py --quick

# Full validation
uv run python master.py --horizons 30 90 180

# Verify results
ls -lh master_results_*/
cat master_results_*/MASTER_REPORT.txt
```

---

## Debugging

### Enable Debug Logging

```python
# In any module
from loguru import logger

# Add debug logs
logger.debug(f"Variable value: {my_var}")
logger.debug(f"DataFrame shape: {df.shape}")
```

### Check Worker Logs

```python
# In orchestration/analyzer.py

def run_parallel_analysis(self):
    # Workers log to main log file
    # Check: master_results_*/master_analysis.log

    # Add detailed logging
    from crypto_trader.execution.logging_utils import log_worker_lifecycle

    log_worker_lifecycle(worker_id, "STARTED", strategy=strategy_name)
```

### Profile Performance

```python
import time

start = time.perf_counter()
# ... your code ...
duration = time.perf_counter() - start
logger.info(f"Operation took {duration:.3f}s")
```

---

## Common Tasks

### Change Default Horizons

```python
# File: src/crypto_trader/orchestration/analyzer.py

def __init__(self, ..., horizons=None, quick_mode=False, ...):
    if quick_mode:
        self.horizons = [
            HorizonConfig("30d", 30, "30 days"),
            HorizonConfig("90d", 90, "90 days"),  # Add more
        ]
```

### Add Custom Metrics

```python
# File: src/crypto_trader/execution/metric_utils.py

def calculate_my_metric(returns: pd.Series) -> float:
    """
    Calculate custom performance metric.

    Args:
        returns: Series of returns

    Returns:
        Metric value
    """
    # Your calculation
    return metric_value

# Add to __all__ for export
__all__ = ["...", "calculate_my_metric"]
```

### Customize Report Format

```python
# File: src/crypto_trader/reports/formatters/html.py

class HTMLFormatter:

    @staticmethod
    def format_my_metric(value: float) -> str:
        """Format custom metric for display."""
        return f"{value:.2f}x"
```

---

## Code Style

### Follow Existing Patterns

1. **Docstrings**: Use NumPy style
2. **Type Hints**: Use typing module
3. **Logging**: Use loguru
4. **Validation**: Include `if __name__ == "__main__"` block
5. **Module Header**: Include purpose, packages, sample usage

### Example Module Template

```python
"""
Module Name - Brief Description

**Purpose**: What this module does

**Key Components**:
- Component1: Does X
- Component2: Does Y

**Third-party packages**:
- package: URL

**Sample Usage**:
```python
from module import function
result = function(data)
```

**Expected Output**:
Description of output.

Created during [phase/date].
"""

import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


def my_function(param: str) -> Dict[str, Any]:
    """
    Function description.

    Args:
        param: Parameter description

    Returns:
        Result description
    """
    # Implementation
    return result


if __name__ == "__main__":
    """Validation block"""
    all_validation_failures = []
    total_tests = 0

    # Tests...

    # Final result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED")
        sys.exit(0)
```

---

## Performance Optimization

### Parallel Execution

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_item(item):
    # CPU-bound work
    return result

# Use process pool for CPU-bound tasks
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_item, item) for item in items]

    for future in as_completed(futures):
        result = future.result()
```

### Data Caching

```python
# Pre-fetch data once
data_cache = {}
for horizon in horizons:
    data_cache[horizon.name] = fetch_data(horizon.days)

# Share across workers via dict
def worker(horizon_name):
    data = data_cache[horizon_name]  # No redundant fetching!
```

### Memory Management

```python
import gc

# Clear large DataFrames when done
del large_dataframe
gc.collect()
```

---

## Troubleshooting

### Import Errors

```bash
# Ensure src/ is in PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Or use absolute imports
from crypto_trader.module import function
```

### Worker Failures

```python
# Check worker logs in master_analysis.log
# Workers log:
# - STARTED
# - PROGRESS
# - COMPLETED
# - FAILED (with error details)
```

### Missing Dependencies

```bash
# Reinstall all dependencies
uv sync --reinstall
```

---

## Best Practices

### DO:
- ✅ Write validation tests for every module
- ✅ Use type hints
- ✅ Log important operations
- ✅ Handle errors gracefully
- ✅ Document your code
- ✅ Test with real data
- ✅ Keep functions focused (single responsibility)

### DON'T:
- ❌ Modify core modules without tests
- ❌ Skip validation blocks
- ❌ Use print() (use logger instead)
- ❌ Ignore type hints
- ❌ Create circular imports
- ❌ Test with fake data
- ❌ Write functions >100 lines

---

## Getting Help

### Documentation
- Architecture: See ARCHITECTURE.md
- Evidence: See PHASE*_EVIDENCE.md files
- Code: Read module headers

### Debugging Steps
1. Check module validation: `uv run python src/crypto_trader/module/file.py`
2. Check end-to-end: `uv run python master.py --quick`
3. Check logs: `cat master_results_*/master_analysis.log`
4. Add debug logging: `logger.debug(...)`

---

## Contributing Guidelines

### Before Submitting Changes

1. **Run all validations**:
```bash
# Validate all modules
find src/crypto_trader -name "*.py" -exec grep -l "if __name__" {} \; | while read f; do uv run python "$f" || echo "FAILED: $f"; done

# End-to-end test
uv run python master.py --quick
```

2. **Update documentation**:
- Module headers
- ARCHITECTURE.md (if structure changed)
- This DEVELOPER_GUIDE.md (if workflow changed)

3. **Verify zero regressions**:
- All existing tests still pass
- master.py still runs successfully
- Exit code: 0

### Commit Messages

```
<type>: <short description>

<detailed description if needed>

- Changes made
- Tests added
- Documentation updated
```

Types: feat, fix, docs, refactor, test, perf

---

## Conclusion

This codebase follows **pragmatic engineering principles**:
- Clean architecture with clear separation of concerns
- Comprehensive testing at every level
- Thorough documentation for maintainability
- Production-ready code that actually works

**No bullshit. Write code that works. Test it. Document it.**

---

**Document Version**: 1.0
**Last Updated**: 2025-10-19
**Maintainer**: Linus Torvalds Mode
