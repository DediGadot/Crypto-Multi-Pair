"""
Logging Utilities for Backtest Execution

This module provides specialized logging functions for debugging
backtest execution, worker lifecycle, and data flow.

**Purpose**: Enhanced logging for parallel execution debugging

**Key Functions**:
- log_dataframe_info: Log DataFrame structure and statistics
- log_worker_lifecycle: Log worker status events
- log_error_with_context: Log errors with contextual information
- log_memory_usage: Log memory and CPU usage (optional, requires psutil)

**Third-party packages**:
- loguru: https://github.com/Delgan/loguru
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/
- psutil (optional): https://github.com/giampaolo/psutil

**Sample Input**:
```python
log_worker_lifecycle("worker-1", "STARTED", strategy="momentum")
log_dataframe_info(df, "Input Data", detailed=True)
log_error_with_context(exception, {"strategy": "momentum", "symbol": "BTC/USD"})
```

**Expected Output**:
Structured log messages for debugging execution flow.

Extracted from master.py (lines 154-291) during Phase 2.5 refactoring.
"""

import traceback
from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger


def log_dataframe_info(df: pd.DataFrame, label: str, detailed: bool = True, sample_rows: int = 0):
    """
    Log comprehensive DataFrame information for debugging data flow.

    Args:
        df: DataFrame to analyze
        label: Label for this DataFrame in logs
        detailed: If True, log detailed statistics
        sample_rows: Number of sample rows to log (0 = none)
    """
    logger.debug(f"[DATA] 📊 {label}:")
    logger.debug(f"  Shape: {df.shape} ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    logger.debug(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    logger.debug(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    if detailed and len(df) > 0:
        # Index information
        if hasattr(df.index, 'min'):
            logger.debug(f"  Index: {df.index.min()} to {df.index.max()}")

        # Null value analysis
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) > 0:
            logger.warning(f"  ⚠️  Null values detected:")
            for col, count in null_cols.items():
                pct = (count / len(df)) * 100
                logger.warning(f"    - {col}: {count:,} ({pct:.1f}%)")
        else:
            logger.debug(f"  ✓ No null values")

        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and len(numeric_cols) <= 20:  # Limit output
            logger.debug(f"  Numeric columns ({len(numeric_cols)}):")
            for col in numeric_cols[:10]:  # Show first 10
                try:
                    logger.debug(f"    - {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}, std={df[col].std():.4f}")
                except Exception:
                    pass

    # Sample rows
    if sample_rows > 0 and len(df) > 0:
        logger.debug(f"  Sample rows (first {min(sample_rows, len(df))}):")
        logger.debug(f"\n{df.head(sample_rows).to_string()}")


def log_memory_usage(label: str, detailed: bool = False):
    """
    Log current memory and CPU usage.

    Args:
        label: Label for this measurement
        detailed: If True, log additional system metrics
    """
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        logger.info(f"[MEMORY] 💾 {label}:")
        logger.info(f"  RSS: {mem_info.rss / 1024 / 1024:.1f} MB (resident set size)")
        logger.info(f"  VMS: {mem_info.vms / 1024 / 1024:.1f} MB (virtual memory size)")

        try:
            cpu_percent = process.cpu_percent(interval=0.1)
            logger.info(f"  CPU: {cpu_percent:.1f}%")
        except Exception:
            pass

        if detailed:
            try:
                mem_percent = process.memory_percent()
                logger.info(f"  Memory %: {mem_percent:.2f}%")

                # System-wide memory
                sys_mem = psutil.virtual_memory()
                logger.info(f"  System Memory: {sys_mem.used / 1024 / 1024 / 1024:.1f}GB / {sys_mem.total / 1024 / 1024 / 1024:.1f}GB ({sys_mem.percent:.1f}%)")
            except Exception:
                pass

    except ImportError:
        logger.debug(f"[MEMORY] {label}: psutil not available (install with: pip install psutil)")
    except Exception as e:
        logger.debug(f"[MEMORY] {label}: Could not get memory info: {e}")


def log_worker_lifecycle(worker_id: str, status: str, **kwargs):
    """
    Log worker lifecycle events for parallel execution debugging.

    Args:
        worker_id: Unique identifier for this worker
        status: Status (STARTED, PROGRESS, COMPLETED, FAILED)
        **kwargs: Additional context to log
    """
    status_emoji = {
        'STARTED': '🚀',
        'PROGRESS': '⏳',
        'COMPLETED': '✅',
        'FAILED': '❌'
    }
    emoji = status_emoji.get(status, '📌')

    log_msg = f"[WORKER-{worker_id}] {emoji} {status}"
    if kwargs:
        log_msg += f" | {', '.join(f'{k}={v}' for k, v in kwargs.items())}"

    if status == 'FAILED':
        logger.error(log_msg)
    elif status == 'COMPLETED':
        logger.success(log_msg)
    else:
        logger.info(log_msg)


def log_error_with_context(error: Exception, context: Dict[str, Any], include_traceback: bool = True):
    """
    Log errors with full context for debugging.

    Args:
        error: The exception that occurred
        context: Dictionary of contextual information
        include_traceback: Whether to include full traceback
    """
    logger.error(f"[ERROR] 🔥 {type(error).__name__}: {str(error)}")
    logger.error(f"[ERROR] Context:")
    for key, value in context.items():
        # Truncate long values
        str_value = str(value)
        if len(str_value) > 200:
            str_value = str_value[:200] + "..."
        logger.error(f"  - {key}: {str_value}")

    if include_traceback:
        logger.error(f"[ERROR] Traceback:")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    """
    Validation block for logging utilities.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: log_dataframe_info
    total_tests += 1
    print("Test 1: log_dataframe_info")
    try:
        # Create sample DataFrame
        test_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10.5, 20.3, 30.1, 40.9, 50.2],
            'c': ['x', 'y', 'z', 'w', 'v']
        })

        # This should log without errors
        log_dataframe_info(test_df, "Test DataFrame", detailed=True, sample_rows=2)
        print(f"  ✓ log_dataframe_info executed without errors")

    except Exception as e:
        all_validation_failures.append(f"log_dataframe_info failed: {e}")

    # Test 2: log_worker_lifecycle
    total_tests += 1
    print("\nTest 2: log_worker_lifecycle")
    try:
        # Test different statuses
        log_worker_lifecycle("test-1", "STARTED", strategy="momentum", symbol="BTC/USD")
        log_worker_lifecycle("test-1", "PROGRESS", completed=50)
        log_worker_lifecycle("test-1", "COMPLETED", duration=1.5)
        log_worker_lifecycle("test-2", "FAILED", error="Timeout")

        print(f"  ✓ log_worker_lifecycle executed for all statuses")

    except Exception as e:
        all_validation_failures.append(f"log_worker_lifecycle failed: {e}")

    # Test 3: log_error_with_context
    total_tests += 1
    print("\nTest 3: log_error_with_context")
    try:
        # Create test error
        test_error = ValueError("Test validation error")
        test_context = {
            "strategy": "momentum",
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "long_value": "A" * 500  # Test truncation
        }

        log_error_with_context(test_error, test_context, include_traceback=False)
        print(f"  ✓ log_error_with_context executed without errors")

    except Exception as e:
        all_validation_failures.append(f"log_error_with_context failed: {e}")

    # Test 4: log_memory_usage
    total_tests += 1
    print("\nTest 4: log_memory_usage")
    try:
        # This might fail if psutil not available, but should not raise
        log_memory_usage("Test Memory Check", detailed=False)
        print(f"  ✓ log_memory_usage executed (psutil may or may not be available)")

    except Exception as e:
        all_validation_failures.append(f"log_memory_usage failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Logging utilities are validated and ready for use")
        print("\nNOTE: Check loguru output above for actual log messages")
        sys.exit(0)
