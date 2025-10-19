#!/usr/bin/env python3
"""
Error Diagnostic Tool
=====================

Quickly diagnoses critical errors in the crypto trading codebase.

Checks:
1. Process pool functionality
2. Strategy initialization
3. Data coherence
4. Alternative data sources
5. Recent error logs

Usage:
    uv run python diagnose_errors.py
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Any
import pandas as pd
from loguru import logger


def test_process_pool() -> Dict[str, Any]:
    """Test if ProcessPoolExecutor works."""
    result = {
        "name": "Process Pool Functionality",
        "status": "unknown",
        "details": []
    }

    def dummy_task(x):
        return x * 2

    try:
        with ProcessPoolExecutor(max_workers=2) as executor:
            future = executor.submit(dummy_task, 5)
            value = future.result(timeout=5)

            if value == 10:
                result["status"] = "pass"
                result["details"].append("✅ ProcessPoolExecutor works correctly")
            else:
                result["status"] = "fail"
                result["details"].append(f"❌ ProcessPoolExecutor returned wrong value: {value}")

    except PermissionError as e:
        result["status"] = "fail"
        result["details"].append(f"❌ PermissionError: {e}")
        result["details"].append("   This is the critical issue preventing 7 strategies from working")
        result["details"].append("   Fix: Run fix_process_pool_issue.py")

    except Exception as e:
        result["status"] = "fail"
        result["details"].append(f"❌ Unexpected error: {type(e).__name__}: {e}")

    return result


def test_thread_pool() -> Dict[str, Any]:
    """Test if ThreadPoolExecutor works as fallback."""
    result = {
        "name": "Thread Pool Fallback",
        "status": "unknown",
        "details": []
    }

    def dummy_task(x):
        return x * 3

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(dummy_task, 7)
            value = future.result(timeout=5)

            if value == 21:
                result["status"] = "pass"
                result["details"].append("✅ ThreadPoolExecutor works as fallback")
            else:
                result["status"] = "fail"
                result["details"].append(f"❌ ThreadPoolExecutor returned wrong value: {value}")

    except Exception as e:
        result["status"] = "fail"
        result["details"].append(f"❌ ThreadPoolExecutor also fails: {type(e).__name__}: {e}")

    return result


def test_strategy_initialization() -> Dict[str, Any]:
    """Test if all strategies can initialize."""
    result = {
        "name": "Strategy Initialization",
        "status": "unknown",
        "details": []
    }

    try:
        from crypto_trader.strategies import get_registry

        registry = get_registry()
        strategy_names = registry.get_strategy_names()

        result["details"].append(f"Found {len(strategy_names)} strategies in registry")

        failed = []
        passed = []

        for name in strategy_names:
            try:
                StrategyClass = registry.get_strategy(name)
                strategy = StrategyClass()
                strategy.initialize({})

                if hasattr(strategy, '_initialized') and strategy._initialized:
                    passed.append(name)
                else:
                    failed.append(f"{name} (initialized=False)")

            except Exception as e:
                failed.append(f"{name} ({type(e).__name__}: {str(e)[:50]})")

        if failed:
            result["status"] = "fail"
            result["details"].append(f"❌ {len(failed)} strategies failed to initialize:")
            for f in failed:
                result["details"].append(f"   - {f}")
        else:
            result["status"] = "pass"
            result["details"].append(f"✅ All {len(passed)} strategies initialized successfully")

    except Exception as e:
        result["status"] = "fail"
        result["details"].append(f"❌ Could not test strategies: {e}")

    return result


def test_data_coherence() -> Dict[str, Any]:
    """Test if data slicing for horizons is correct."""
    result = {
        "name": "Data Coherence (Horizon Slicing)",
        "status": "unknown",
        "details": []
    }

    try:
        # Check if _slice_data_to_horizon function exists in master.py
        master_py = Path("master.py")
        if not master_py.exists():
            result["status"] = "skip"
            result["details"].append("⚠️ master.py not found")
            return result

        content = master_py.read_text()

        if "_slice_data_to_horizon" in content:
            result["status"] = "pass"
            result["details"].append("✅ Data coherence fix is present (_slice_data_to_horizon found)")
        else:
            result["status"] = "fail"
            result["details"].append("❌ Data coherence fix NOT found")
            result["details"].append("   All horizons may be testing on same data window")
            result["details"].append("   See: DATA_COHERENCE_FIX.md")

        # Check if verification script exists
        verify_script = Path("verify_data_coherence.py")
        if verify_script.exists():
            result["details"].append("ℹ️ Verification script available: verify_data_coherence.py")
        else:
            result["details"].append("⚠️ Verification script not found")

    except Exception as e:
        result["status"] = "fail"
        result["details"].append(f"❌ Error checking data coherence: {e}")

    return result


def test_alternative_data() -> Dict[str, Any]:
    """Test if alternative data sources are available."""
    result = {
        "name": "Alternative Data Sources",
        "status": "unknown",
        "details": []
    }

    sources = {
        "Order Flow": "src/crypto_trader/data/alt/orderflow_stream.py",
        "Sentiment": "src/crypto_trader/data/alt/sentiment_ingestor.py",
        "On-Chain": "src/crypto_trader/data/alt/onchain_ingestor.py"
    }

    available = []
    missing = []

    for name, path in sources.items():
        if Path(path).exists():
            available.append(name)
        else:
            missing.append(name)

    if missing:
        result["status"] = "warn"
        result["details"].append(f"⚠️ {len(missing)} alternative data sources missing:")
        for m in missing:
            result["details"].append(f"   - {m}")
    else:
        result["status"] = "pass"
        result["details"].append(f"✅ All {len(available)} alternative data sources present")

    if available:
        result["details"].append(f"ℹ️ Available: {', '.join(available)}")

    return result


def check_recent_errors() -> Dict[str, Any]:
    """Check recent error logs."""
    result = {
        "name": "Recent Error Logs",
        "status": "unknown",
        "details": []
    }

    try:
        # Find most recent master_results directory
        results_dirs = sorted(Path(".").glob("master_results_*"), reverse=True)

        if not results_dirs:
            result["status"] = "skip"
            result["details"].append("⚠️ No master_results directories found")
            return result

        latest = results_dirs[0]
        log_file = latest / "master_analysis.log"

        if not log_file.exists():
            result["status"] = "skip"
            result["details"].append(f"⚠️ No log file in {latest}")
            return result

        result["details"].append(f"Analyzing: {log_file}")

        # Count errors
        content = log_file.read_text()
        lines = content.split('\n')

        error_count = sum(1 for line in lines if '| ERROR' in line)
        warning_count = sum(1 for line in lines if '| WARNING' in line)
        permission_errors = sum(1 for line in lines if 'PermissionError' in line)
        initialization_errors = sum(1 for line in lines if 'not initialized' in line)

        if error_count > 0 or permission_errors > 0:
            result["status"] = "fail"
            result["details"].append(f"❌ Found {error_count} ERROR entries")
            if permission_errors > 0:
                result["details"].append(f"❌ Found {permission_errors} PermissionError occurrences (CRITICAL)")
            if initialization_errors > 0:
                result["details"].append(f"❌ Found {initialization_errors} 'not initialized' errors")
        else:
            result["status"] = "pass"
            result["details"].append(f"✅ No errors in latest run")

        if warning_count > 0:
            result["details"].append(f"⚠️ Found {warning_count} WARNING entries")

        # Get last few errors
        error_lines = [line for line in lines if '| ERROR' in line]
        if error_lines:
            result["details"].append("\nRecent errors:")
            for line in error_lines[-5:]:  # Last 5 errors
                # Extract just the error message
                if '|' in line:
                    msg = line.split('|')[-1].strip()
                    result["details"].append(f"   {msg[:100]}")

    except Exception as e:
        result["status"] = "fail"
        result["details"].append(f"❌ Error reading logs: {e}")

    return result


def print_result(result: Dict[str, Any]) -> None:
    """Pretty print a test result."""
    status_symbols = {
        "pass": "✅",
        "fail": "❌",
        "warn": "⚠️",
        "skip": "⏭️",
        "unknown": "❓"
    }

    symbol = status_symbols.get(result["status"], "❓")
    print(f"\n{symbol} {result['name']}")
    print("─" * 70)

    for detail in result["details"]:
        print(f"  {detail}")


def main():
    """Run all diagnostic tests."""
    print("="*70)
    print("CRYPTO TRADING SYSTEM - ERROR DIAGNOSTIC TOOL")
    print("="*70)

    tests = [
        test_process_pool,
        test_thread_pool,
        test_strategy_initialization,
        test_data_coherence,
        test_alternative_data,
        check_recent_errors
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"\n❌ Test crashed: {test.__name__}")
            print(f"  Error: {e}")

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    status_counts = {
        "pass": sum(1 for r in results if r["status"] == "pass"),
        "fail": sum(1 for r in results if r["status"] == "fail"),
        "warn": sum(1 for r in results if r["status"] == "warn"),
        "skip": sum(1 for r in results if r["status"] == "skip")
    }

    print(f"\n✅ Passed: {status_counts['pass']}")
    print(f"❌ Failed: {status_counts['fail']}")
    print(f"⚠️ Warnings: {status_counts['warn']}")
    print(f"⏭️ Skipped: {status_counts['skip']}")

    if status_counts['fail'] > 0:
        print("\n🔴 CRITICAL ISSUES DETECTED")
        print("\nRecommended actions:")
        print("  1. Review ERROR_ANALYSIS_REPORT.md for detailed analysis")
        print("  2. Run fix_process_pool_issue.py to fix PermissionError")
        print("  3. Test fixes with: uv run python master.py --quick")
        return False
    elif status_counts['warn'] > 0:
        print("\n🟡 WARNINGS DETECTED")
        print("  Review warnings above and consider fixes")
        return True
    else:
        print("\n🟢 ALL CHECKS PASSED")
        print("  System appears healthy")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
