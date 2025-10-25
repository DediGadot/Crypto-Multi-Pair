#!/usr/bin/env python3
"""
Quick validation test for benchmark integration in master_windowed_multipair.py

Tests that all required imports and components are available.
"""

import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

all_validation_failures = []
total_tests = 0

print("🔍 Validating benchmark integration components...\n")

# Test 1: Import BenchmarkComparator
total_tests += 1
print("Test 1: Import BenchmarkComparator")
try:
    from crypto_trader.analysis.benchmark_comparator import BenchmarkComparator
    print("  ✓ BenchmarkComparator imported successfully")
except ImportError as e:
    all_validation_failures.append(f"Failed to import BenchmarkComparator: {e}")

# Test 2: Import Plotly chart functions
total_tests += 1
print("\nTest 2: Import Plotly chart functions")
try:
    from crypto_trader.reports.formatters.plotly_benchmark_charts import (
        create_alpha_comparison_chart,
        create_win_rate_heatmap,
        create_cumulative_returns_chart,
        create_return_distribution_violin
    )
    print("  ✓ All 4 chart functions imported successfully")
except ImportError as e:
    all_validation_failures.append(f"Failed to import chart functions: {e}")

# Test 3: Check BuyAndHold strategy exists
total_tests += 1
print("\nTest 3: Check BuyAndHold strategy registration")
try:
    from crypto_trader.strategies import get_registry
    import crypto_trader.strategies.library  # Trigger registration

    registry = get_registry()
    strategy_names = registry.get_strategy_names()

    if "BuyAndHold" in strategy_names:
        print("  ✓ BuyAndHold strategy is registered")
    else:
        all_validation_failures.append(
            f"BuyAndHold not in registry. Available: {', '.join(strategy_names[:10])}"
        )
except Exception as e:
    all_validation_failures.append(f"Failed to check strategy registry: {e}")

# Test 4: Verify master_windowed_multipair.py syntax
total_tests += 1
print("\nTest 4: Check master_windowed_multipair.py syntax")
try:
    import py_compile
    master_file = script_dir / "master_windowed_multipair.py"
    py_compile.compile(str(master_file), doraise=True)
    print("  ✓ master_windowed_multipair.py has valid syntax")
except py_compile.PyCompileError as e:
    all_validation_failures.append(f"Syntax error in master_windowed_multipair.py: {e}")

# Test 5: Test BenchmarkComparator instantiation
total_tests += 1
print("\nTest 5: Test BenchmarkComparator instantiation")
try:
    from crypto_trader.analysis.benchmark_comparator import BenchmarkComparator
    comparator = BenchmarkComparator()
    print("  ✓ BenchmarkComparator instantiated successfully")
except Exception as e:
    all_validation_failures.append(f"Failed to instantiate BenchmarkComparator: {e}")

# Test 6: Check that plotly is available
total_tests += 1
print("\nTest 6: Check Plotly availability")
try:
    import plotly.graph_objects as go
    print("  ✓ Plotly is available")
except ImportError as e:
    all_validation_failures.append(f"Plotly not available: {e}")

# Final validation result
print("\n" + "="*70)
if all_validation_failures:
    print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
    for failure in all_validation_failures:
        print(f"  - {failure}")
    sys.exit(1)
else:
    print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
    print("\nBenchmark integration is ready to use!")
    print("\n📝 Next steps:")
    print("  1. Run master_windowed_multipair.py with --quick flag")
    print("  2. Check HTML report for benchmark comparison sections")
    print("  3. Verify interactive charts render correctly")
    sys.exit(0)
