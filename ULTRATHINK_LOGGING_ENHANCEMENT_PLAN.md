# ULTRATHINK: Master.py Comprehensive Logging Enhancement Plan

## Analysis Summary

**Current State:**
- File: master.py (4,206 lines)
- Existing logger statements: ~93
- Main components:
  - `MasterStrategyAnalyzer` class (orchestrator)
  - `run_backtest_worker()` (single-pair parallel worker)
  - `run_multipair_backtest_worker()` (multi-pair parallel worker)
  - Multiple helper functions
  - HTML report generation

**Identified Gaps:**
1. **Limited timing information** - No comprehensive execution time tracking
2. **Minimal data flow visibility** - Hard to trace data transformations
3. **Sparse worker logging** - Parallel workers have limited debug output
4. **Missing state snapshots** - Can't see full state at critical points
5. **No resource metrics** - Limited memory/CPU tracking
6. **Incomplete error context** - Errors lack full state information
7. **Missing validation checkpoints** - Data integrity not logged throughout

## Enhancement Strategy

### 1. **Execution Timing & Performance**
- Add timestamp decorator for all major functions
- Track elapsed time for each operation
- Log performance bottlenecks
- Add cumulative timing summaries

### 2. **Data Flow Tracing**
- Log data shape/size at every transformation
- Track column additions/removals
- Log data quality metrics (nulls, ranges, etc.)
- Record data alignment operations

### 3. **Worker-Level Detailed Logging**
- Unique worker ID for each parallel task
- Log worker lifecycle (start, progress, completion)
- Track worker resource usage
- Capture worker-specific errors with full context

### 4. **State Snapshots**
- Log full configuration at startup
- Snapshot intermediate results
- Track object state transitions
- Log critical variable values

### 5. **Enhanced Error Reporting**
- Full traceback with context
- State dump on errors
- Data sample on failures
- Recovery attempts logged

### 6. **Resource Monitoring**
- Memory usage at key checkpoints
- CPU utilization tracking
- Disk I/O monitoring
- Network call tracking

### 7. **Progress & Visibility**
- Detailed progress indicators
- ETA calculations
- Throughput metrics
- Completion percentages

### 8. **Validation Checkpoints**
- Data integrity checks logged
- Schema validation results
- Numerical sanity checks
- Cross-validation of results

## Implementation Sections

### Section A: Timing Decorator & Context Manager
```python
# Add timing utilities
import functools
import time
from contextlib import contextmanager

def log_execution_time(func_name: str = None):
    """Decorator to log function execution time"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            logger.debug(f"[TIMING] Starting: {name}")
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.info(f"[TIMING] ✓ {name}: {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(f"[TIMING] ✗ {name}: {duration:.3f}s (FAILED: {e})")
                raise
        return wrapper
    return decorator

@contextmanager
def log_operation(operation_name: str, log_level: str = "INFO"):
    """Context manager for logging operations with timing"""
    logger.log(log_level, f"[START] {operation_name}")
    start = time.perf_counter()
    try:
        yield
        duration = time.perf_counter() - start
        logger.log(log_level, f"[COMPLETE] {operation_name} ({duration:.3f}s)")
    except Exception as e:
        duration = time.perf_counter() - start
        logger.error(f"[FAILED] {operation_name} after {duration:.3f}s: {e}")
        raise
```

### Section B: Data Logging Utilities
```python
def log_dataframe_info(df: pd.DataFrame, label: str, detailed: bool = True):
    """Log comprehensive DataFrame information"""
    logger.debug(f"[DATA] {label}:")
    logger.debug(f"  Shape: {df.shape} ({df.shape[0]} rows × {df.shape[1]} cols)")
    logger.debug(f"  Columns: {list(df.columns)}")
    logger.debug(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    if detailed:
        logger.debug(f"  Index: {df.index.min()} to {df.index.max()}")
        null_cols = df.isnull().sum()
        if null_cols.any():
            logger.warning(f"  Null values: {dict(null_cols[null_cols > 0])}")

        # Log numeric column ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            logger.debug(f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")
```

### Section C: Worker Logging Enhancement
```python
def enhanced_worker_logging(worker_func):
    """Decorator to add comprehensive logging to worker functions"""
    @functools.wraps(worker_func)
    def wrapper(*args, **kwargs):
        worker_id = f"{worker_func.__name__}_{id(threading.current_thread())}"
        logger.info(f"[WORKER-{worker_id}] STARTED")
        start_time = time.perf_counter()

        try:
            # Log input parameters
            logger.debug(f"[WORKER-{worker_id}] Input args: {len(args)}, kwargs: {list(kwargs.keys())}")

            result = worker_func(*args, **kwargs)

            duration = time.perf_counter() - start_time
            logger.success(f"[WORKER-{worker_id}] COMPLETED in {duration:.3f}s")
            return result

        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"[WORKER-{worker_id}] FAILED after {duration:.3f}s")
            logger.error(f"[WORKER-{worker_id}] Error: {type(e).__name__}: {str(e)}")
            logger.error(f"[WORKER-{worker_id}] Traceback: {traceback.format_exc()}")
            raise

    return wrapper
```

### Section D: Memory & Resource Tracking
```python
def log_memory_usage(label: str):
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.info(f"[MEMORY] {label}:")
        logger.info(f"  RSS: {mem_info.rss / 1024 / 1024:.1f} MB")
        logger.info(f"  VMS: {mem_info.vms / 1024 / 1024:.1f} MB")
        logger.info(f"  CPU: {process.cpu_percent()}%")
    except ImportError:
        logger.debug(f"[MEMORY] {label}: psutil not available")
```

## Logging Levels Strategy

- **DEBUG**: Data shapes, transformations, validation checks, detailed state
- **INFO**: Major operations, timing, progress, milestones
- **SUCCESS**: Completed operations, successful validations
- **WARNING**: Non-critical issues, fallbacks, missing optional data
- **ERROR**: Failures, exceptions, critical problems

## Key Logging Points

1. **Initialization** (line ~1673)
   - Log all configuration parameters
   - Log environment info
   - Log available resources

2. **Data Fetching** (line ~1804)
   - Log fetch parameters
   - Log data shape before/after augmentation
   - Log feature additions
   - Log data quality metrics

3. **Strategy Discovery** (line ~1747)
   - Log all discovered strategies
   - Log strategy categorization
   - Log strategy initialization

4. **Parallel Execution** (line ~1985)
   - Log worker pool setup
   - Log job distribution
   - Log worker progress
   - Log completion stats

5. **Worker Functions** (lines 728, 868)
   - Log worker start/end
   - Log data processing steps
   - Log backtest execution
   - Log result extraction

6. **Report Generation** (line ~3008)
   - Log report sections
   - Log data aggregation
   - Log file writes

## Expected Benefits

1. **Debugging**: Complete visibility into execution flow
2. **Performance**: Identify bottlenecks precisely
3. **Data Quality**: Track data issues immediately
4. **Error Resolution**: Full context for any failure
5. **Monitoring**: Real-time progress tracking
6. **Optimization**: Data-driven improvement decisions

## Implementation Priority

1. ✅ Add timing utilities (HIGH)
2. ✅ Add data logging utilities (HIGH)
3. ✅ Enhance worker logging (HIGH)
4. ✅ Add memory tracking (MEDIUM)
5. ✅ Add validation logging (MEDIUM)
6. ✅ Enhance error reporting (HIGH)
7. ✅ Add progress visibility (MEDIUM)

---

**Total Enhancement**: Will add approximately 150-200 new logger statements with rich context, bringing total from ~93 to ~250-300 comprehensive log points.
