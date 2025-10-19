"""
Metric Utilities for Backtest Execution

This module provides metric calculation utilities for backtesting,
including risk-adjusted performance metrics.

**Purpose**: Calculate performance metrics with proper edge case handling

**Key Functions**:
- periods_per_year_from_timeframe: Get annualization factor for timeframe
- calculate_sharpe_ratio_safe: Calculate Sharpe ratio with edge case handling

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/

**Sample Input**:
```python
returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
periods = periods_per_year_from_timeframe("1h")
sharpe = calculate_sharpe_ratio_safe(returns, periods)
```

**Expected Output**:
Float representing Sharpe ratio (annualized risk-adjusted return).

Extracted from master.py (lines 365-421) during Phase 2.5 refactoring.
"""

import numpy as np
import pandas as pd


def periods_per_year_from_timeframe(timeframe: str) -> float:
    """
    Return annualization factor for a given timeframe string.
    Defaults to hourly spacing if unknown.

    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d', '1w')

    Returns:
        Number of periods per year for annualization
    """
    mapping = {
        "1m": 60 * 24 * 365,
        "5m": 12 * 24 * 365,
        "15m": 4 * 24 * 365,
        "1h": 24 * 365,
        "4h": 6 * 365,
        "1d": 365,
        "1w": 52,
    }
    return float(mapping.get(timeframe, 24 * 365))


def calculate_sharpe_ratio_safe(returns: pd.Series, periods_per_year: float) -> float:
    """
    Calculate Sharpe ratio with proper edge case handling.

    Args:
        returns: Series of returns
        periods_per_year: Annualization factor

    Returns:
        Sharpe ratio (0.0 if undefined)

    Raises:
        ValueError: If returns have non-zero but constant values (broken strategy)
                   or if Sharpe ratio is non-finite
    """
    if len(returns) == 0:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    # CRITICAL: Zero variance - distinguish no trades from broken strategy
    if std_return <= 1e-8:
        # If all returns are exactly 0, strategy made no trades - OK
        if (returns == 0).all():
            return 0.0  # No trades = Sharpe of 0
        # Otherwise: trades made but constant returns = BROKEN
        raise ValueError(
            f"Cannot calculate Sharpe ratio: non-zero but constant returns (std={std_return:.2e}). "
            f"This indicates a broken strategy (all trades same P&L). "
            f"Returns: mean={mean_return:.6f}, std={std_return:.2e}"
        )

    # Normal Sharpe calculation
    sharpe = (mean_return * periods_per_year) / (std_return * np.sqrt(periods_per_year))

    # Sanity check for extreme values (but don't cap - let them through for debugging)
    if not np.isfinite(sharpe):
        raise ValueError(
            f"Sharpe ratio is non-finite ({sharpe}). "
            f"Returns: mean={mean_return}, std={std_return}, periods={periods_per_year}"
        )

    return float(sharpe)


if __name__ == "__main__":
    """
    Validation block for metric utilities.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: periods_per_year_from_timeframe
    total_tests += 1
    print("Test 1: periods_per_year_from_timeframe")
    try:
        # Test known timeframes
        test_cases = [
            ("1h", 24 * 365),
            ("1d", 365),
            ("1w", 52),
            ("4h", 6 * 365),
            ("unknown", 24 * 365),  # Default to hourly
        ]

        for timeframe, expected in test_cases:
            result = periods_per_year_from_timeframe(timeframe)
            if result != expected:
                all_validation_failures.append(f"periods_per_year({timeframe}): Expected {expected}, got {result}")
            else:
                print(f"  ✓ {timeframe}: {result} periods/year")

    except Exception as e:
        all_validation_failures.append(f"periods_per_year_from_timeframe failed: {e}")

    # Test 2: calculate_sharpe_ratio_safe - normal case
    total_tests += 1
    print("\nTest 2: calculate_sharpe_ratio_safe - normal returns")
    try:
        # Simulated returns with positive mean and some variance
        normal_returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.008, -0.002, 0.012])
        periods = 252  # Daily returns, 252 trading days

        sharpe = calculate_sharpe_ratio_safe(normal_returns, periods)

        if not np.isfinite(sharpe):
            all_validation_failures.append("Normal returns: Sharpe should be finite")
        else:
            print(f"  ✓ Normal returns Sharpe: {sharpe:.2f}")

    except Exception as e:
        all_validation_failures.append(f"calculate_sharpe_ratio_safe (normal) failed: {e}")

    # Test 3: calculate_sharpe_ratio_safe - zero returns (no trades)
    total_tests += 1
    print("\nTest 3: calculate_sharpe_ratio_safe - zero returns")
    try:
        zero_returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
        periods = 252

        sharpe = calculate_sharpe_ratio_safe(zero_returns, periods)

        if sharpe != 0.0:
            all_validation_failures.append(f"Zero returns: Expected Sharpe=0.0, got {sharpe}")
        else:
            print(f"  ✓ Zero returns (no trades) Sharpe: {sharpe}")

    except Exception as e:
        all_validation_failures.append(f"calculate_sharpe_ratio_safe (zero) failed: {e}")

    # Test 4: calculate_sharpe_ratio_safe - constant non-zero returns (broken strategy)
    total_tests += 1
    print("\nTest 4: calculate_sharpe_ratio_safe - constant non-zero returns")
    try:
        constant_returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
        periods = 252

        try:
            sharpe = calculate_sharpe_ratio_safe(constant_returns, periods)
            all_validation_failures.append("Constant non-zero returns: Should raise ValueError")
        except ValueError as e:
            # This is expected
            print(f"  ✓ Constant non-zero returns correctly raises ValueError")
            print(f"    Error message: {str(e)[:80]}...")

    except Exception as e:
        if not isinstance(e, ValueError):
            all_validation_failures.append(f"calculate_sharpe_ratio_safe (constant) unexpected error: {e}")

    # Test 5: calculate_sharpe_ratio_safe - empty returns
    total_tests += 1
    print("\nTest 5: calculate_sharpe_ratio_safe - empty returns")
    try:
        empty_returns = pd.Series([])
        periods = 252

        sharpe = calculate_sharpe_ratio_safe(empty_returns, periods)

        if sharpe != 0.0:
            all_validation_failures.append(f"Empty returns: Expected Sharpe=0.0, got {sharpe}")
        else:
            print(f"  ✓ Empty returns Sharpe: {sharpe}")

    except Exception as e:
        all_validation_failures.append(f"calculate_sharpe_ratio_safe (empty) failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Metric utilities are validated and ready for use")
        sys.exit(0)
