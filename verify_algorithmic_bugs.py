#!/usr/bin/env python3
"""
Verification Script for Algorithmic Bugs

Tests each identified bug with concrete examples to prove they exist.

Usage:
    python verify_algorithmic_bugs.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from crypto_trader.orchestration.multipair_window_manager import MultiPairTrainTestSplitter
from crypto_trader.analysis.multipair_aggregator import MultiPairAggregator


def test_bug1_window_slicing_data_leakage():
    """
    BUG #1: Window slicing uses indices from split data on full dataset.

    Expected: Test windows only access data after cutoff
    Actual: Test windows may access training data due to index mismatch
    """
    print("=" * 70)
    print("TEST: Bug #1 - Window Slicing Data Leakage")
    print("=" * 70)

    # Create sample data
    runtime = datetime(2024, 12, 31, tzinfo=pytz.UTC)
    dates = pd.date_range(end=runtime, periods=17520, freq='1h', tz=pytz.UTC)  # 2 years

    btc_data = pd.DataFrame({
        'close': np.random.rand(len(dates)) * 100 + 50000,
        'volume': np.random.rand(len(dates)) * 1000,
    }, index=dates)

    eth_data = pd.DataFrame({
        'close': np.random.rand(len(dates)) * 100 + 3000,
        'volume': np.random.rand(len(dates)) * 500,
    }, index=dates)

    data_dict = {'BTC/USDT': btc_data, 'ETH/USDT': eth_data}

    # Create splitter with 1 year test set
    splitter = MultiPairTrainTestSplitter(
        runtime_date=runtime,
        test_set_years=1.0,
        pairs=['BTC/USDT', 'ETH/USDT']
    )

    cutoff_date = splitter.cutoff_date
    print(f"\nCutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"Train Set: Before {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"Test Set: After {cutoff_date.strftime('%Y-%m-%d')}")

    # Generate windows
    train_windows, test_windows = splitter.generate_windows(
        data_dict, 30, '30d', '1h'
    )

    print(f"\nGenerated {len(train_windows)} train windows, {len(test_windows)} test windows")

    # Check: Do window indices match the intended data?
    print("\n🔍 Checking TEST windows for data leakage...")

    issues_found = []

    if test_windows:
        # Take first test window
        test_window = test_windows[0]
        pair = 'BTC/USDT'
        pair_window = test_window.pair_windows[pair]

        print(f"\nTest Window 0 for {pair}:")
        print(f"  WindowSpec dates: {pair_window.start_date.strftime('%Y-%m-%d')} to {pair_window.end_date.strftime('%Y-%m-%d')}")
        print(f"  Indices: start_idx={pair_window.start_idx}, end_idx={pair_window.end_idx}")

        # Simulate the BUGGY behavior in master_windowed_multipair.py line 115-116
        # Using FULL dataset instead of split dataset
        full_data = data_dict[pair]
        buggy_slice = full_data.iloc[pair_window.start_idx:pair_window.end_idx]

        print(f"\n  BUGGY behavior (using full dataset):")
        print(f"    Actual sliced dates: {buggy_slice.index[0].strftime('%Y-%m-%d')} to {buggy_slice.index[-1].strftime('%Y-%m-%d')}")

        # Check if buggy slice contains data from BEFORE cutoff
        if buggy_slice.index[0] < cutoff_date:
            issues_found.append(
                f"❌ DATA LEAKAGE: Test window {test_window.window_id} accesses "
                f"training data from {buggy_slice.index[0].strftime('%Y-%m-%d')} "
                f"(before cutoff {cutoff_date.strftime('%Y-%m-%d')})"
            )
            print(f"    ❌ Contains training data! (starts before cutoff)")
        else:
            print(f"    ✓ No leakage in this window")

        # Now show CORRECT behavior
        train_data_dict, test_data_dict = splitter.split_data(data_dict)
        correct_data = test_data_dict[pair]
        correct_slice = correct_data.iloc[pair_window.start_idx:pair_window.end_idx]

        print(f"\n  CORRECT behavior (using test dataset):")
        print(f"    Actual sliced dates: {correct_slice.index[0].strftime('%Y-%m-%d')} to {correct_slice.index[-1].strftime('%Y-%m-%d')}")

        if correct_slice.index[0] >= cutoff_date:
            print(f"    ✅ Correctly isolates test data")
        else:
            print(f"    ❌ Still has issues!")

    print("\n" + "=" * 70)
    if issues_found:
        print("BUG CONFIRMED:")
        for issue in issues_found:
            print(f"  {issue}")
        return False
    else:
        print("✓ No data leakage detected in sampled windows")
        print("  (Note: Bug may still exist with different data patterns)")
        return True


def test_bug3_portfolio_sharpe_formula():
    """
    BUG #3: Portfolio Sharpe = Average of Individual Sharpes (WRONG!)

    Expected: Portfolio Sharpe considers correlations
    Actual: Naive average ignores diversification benefits
    """
    print("\n" + "=" * 70)
    print("TEST: Bug #3 - Incorrect Portfolio Sharpe Calculation")
    print("=" * 70)

    # Create synthetic results for two UNCORRELATED assets
    # Both have same performance: 10% return, Sharpe = 1.0

    # Simulate 10 windows with returns that are UNCORRELATED
    np.random.seed(42)

    btc_returns = np.random.randn(10) * 0.05 + 0.10  # Mean 10%, std 5%
    eth_returns = np.random.randn(10) * 0.05 + 0.10  # Mean 10%, std 5%

    # Calculate individual Sharpes
    btc_sharpe = btc_returns.mean() / btc_returns.std()
    eth_sharpe = eth_returns.mean() / eth_returns.std()

    print(f"\nAsset Performance:")
    print(f"  BTC: Mean Return = {btc_returns.mean():.4f}, Sharpe = {btc_sharpe:.4f}")
    print(f"  ETH: Mean Return = {eth_returns.mean():.4f}, Sharpe = {eth_sharpe:.4f}")

    # Correlation between assets
    correlation = np.corrcoef(btc_returns, eth_returns)[0, 1]
    print(f"  Correlation: {correlation:.4f}")

    # Portfolio returns (50/50 allocation)
    portfolio_returns = (btc_returns + eth_returns) / 2

    # TRUE portfolio Sharpe
    true_portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std()

    # BUGGY calculation (what current code does)
    buggy_portfolio_sharpe = (btc_sharpe + eth_sharpe) / 2

    print(f"\nPortfolio Sharpe Ratio:")
    print(f"  TRUE (using portfolio returns):   {true_portfolio_sharpe:.4f}")
    print(f"  BUGGY (averaging Sharpes):        {buggy_portfolio_sharpe:.4f}")
    print(f"  Error: {abs(true_portfolio_sharpe - buggy_portfolio_sharpe):.4f}")
    print(f"  Percent Error: {abs(true_portfolio_sharpe - buggy_portfolio_sharpe) / true_portfolio_sharpe * 100:.1f}%")

    # For uncorrelated assets with equal Sharpe, diversification should INCREASE Sharpe
    # Theoretical: Portfolio vol = individual_vol / sqrt(2) for uncorrelated 50/50
    # So Portfolio Sharpe = Individual Sharpe * sqrt(2)

    theoretical_improvement = np.sqrt(2)
    expected_portfolio_sharpe = btc_sharpe * theoretical_improvement

    print(f"\nDiversification Benefit:")
    print(f"  Theoretical improvement factor (uncorrelated): {theoretical_improvement:.4f}")
    print(f"  Expected portfolio Sharpe: {expected_portfolio_sharpe:.4f}")
    print(f"  Actual portfolio Sharpe: {true_portfolio_sharpe:.4f}")
    print(f"  Buggy code misses: {true_portfolio_sharpe - buggy_portfolio_sharpe:.4f} Sharpe points")

    print("\n" + "=" * 70)
    if abs(buggy_portfolio_sharpe - true_portfolio_sharpe) > 0.1:
        print("BUG CONFIRMED:")
        print(f"  ❌ Buggy calculation differs by {abs(true_portfolio_sharpe - buggy_portfolio_sharpe):.4f}")
        print(f"  ❌ Underestimates diversification benefit")
        return False
    else:
        print("✓ Sharpe calculations match (or assets are highly correlated)")
        return True


def test_bug4_window_boundary_off_by_one():
    """
    BUG #4: Window boundary uses < current_end, excluding last period.

    Expected: 30-day window contains 30 full days
    Actual: Window excludes last period, contains <30 days
    """
    print("\n" + "=" * 70)
    print("TEST: Bug #4 - Window Boundary Off-By-One")
    print("=" * 70)

    # Create data with known start date
    start_date = datetime(2024, 1, 1, 0, 0, tzinfo=pytz.UTC)
    dates = pd.date_range(start=start_date, periods=720, freq='1h', tz=pytz.UTC)  # 30 days

    data = pd.DataFrame({
        'close': np.random.rand(len(dates)) * 100 + 50000,
    }, index=dates)

    print(f"\nTest Data:")
    print(f"  Start: {dates[0].strftime('%Y-%m-%d %H:%M')}")
    print(f"  End:   {dates[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Total hours: {len(dates)}")
    print(f"  Total days: {len(dates) / 24:.1f}")

    # Simulate window generation logic from multipair_window_manager.py
    current_start = dates[0]
    window_duration = timedelta(days=30)
    current_end = current_start + window_duration

    print(f"\nWindow Specification:")
    print(f"  current_start: {current_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  window_duration: 30 days")
    print(f"  current_end: {current_end.strftime('%Y-%m-%d %H:%M')}")

    # BUGGY mask (from line 252)
    buggy_mask = (data.index >= current_start) & (data.index < current_end)
    buggy_data = data[buggy_mask]

    print(f"\nBUGGY behavior (using < current_end):")
    print(f"  First timestamp: {buggy_data.index[0].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Last timestamp:  {buggy_data.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Total hours: {len(buggy_data)}")
    print(f"  Total days: {len(buggy_data) / 24:.1f}")

    # CORRECT mask (using <= current_end)
    correct_mask = (data.index >= current_start) & (data.index <= current_end)
    correct_data = data[correct_mask]

    print(f"\nCORRECT behavior (using <= current_end):")
    print(f"  First timestamp: {correct_data.index[0].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Last timestamp:  {correct_data.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Total hours: {len(correct_data)}")
    print(f"  Total days: {len(correct_data) / 24:.1f}")

    expected_hours = 30 * 24  # 720 hours
    missing_hours = expected_hours - len(buggy_data)

    print("\n" + "=" * 70)
    if missing_hours > 0:
        print("BUG CONFIRMED:")
        print(f"  ❌ Window missing {missing_hours} hours ({missing_hours/24:.1f} days)")
        print(f"  ❌ Expected {expected_hours} hours, got {len(buggy_data)}")
        return False
    else:
        print("✓ Window contains expected number of periods")
        return True


def test_bug5_sharpe_annualization():
    """
    BUG #5: Sharpe ratio incorrectly annualized for short windows.

    Expected: Sharpe adjusts for window length
    Actual: VectorBT uses full-year annualization regardless of window size
    """
    print("\n" + "=" * 70)
    print("TEST: Bug #5 - Sharpe Ratio Annualization")
    print("=" * 70)

    print("\nThis bug requires running actual VectorBT backtests.")
    print("Manual verification needed:")
    print("\n  1. Run same strategy on 30-day window")
    print("  2. Run same strategy on 90-day window")
    print("  3. Compare Sharpe ratios")
    print("\nExpected: Similar Sharpe ratios (slight variance due to sample size)")
    print("Actual (if bug exists): 30-day Sharpe ~3x higher than 90-day Sharpe")

    print("\nMathematical explanation:")
    print("  VectorBT annualization: sqrt(periods_per_year)")
    print("  For 1h data: sqrt(8760) = 93.6")
    print("\n  30-day window (720 hours):")
    print("    Correct factor: sqrt(8760 / 720) = 3.49")
    print("    VectorBT factor: 93.6")
    print("    Over-annualization: 93.6 / 3.49 = 26.8x")
    print("\n  90-day window (2160 hours):")
    print("    Correct factor: sqrt(8760 / 2160) = 2.02")
    print("    VectorBT factor: 93.6")
    print("    Over-annualization: 93.6 / 2.02 = 46.3x")

    print("\n  NOTE: If both are over-annualized by same factor, they remain comparable")
    print("        Bug only matters if comparing across different window sizes")

    print("\n" + "=" * 70)
    print("⚠️  MANUAL VERIFICATION REQUIRED")
    print("    Run the full pipeline with different horizons and compare results")
    return None  # Cannot auto-verify without running full backtests


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 20 + "ALGORITHMIC BUG VERIFICATION")
    print("=" * 80)

    results = {}

    # Test each bug
    try:
        results['Bug #1'] = test_bug1_window_slicing_data_leakage()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['Bug #1'] = False

    try:
        results['Bug #3'] = test_bug3_portfolio_sharpe_formula()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['Bug #3'] = False

    try:
        results['Bug #4'] = test_bug4_window_boundary_off_by_one()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['Bug #4'] = False

    try:
        results['Bug #5'] = test_bug5_sharpe_annualization()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['Bug #5'] = None

    # Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "SUMMARY")
    print("=" * 80)

    for bug, result in results.items():
        if result is True:
            print(f"  ✅ {bug}: No issues detected")
        elif result is False:
            print(f"  ❌ {bug}: BUG CONFIRMED")
        else:
            print(f"  ⚠️  {bug}: Manual verification required")

    confirmed_bugs = sum(1 for r in results.values() if r is False)
    print(f"\n  Total bugs confirmed: {confirmed_bugs}")

    if confirmed_bugs > 0:
        print("\n  ⚠️  CRITICAL: Pipeline has confirmed algorithmic bugs!")
        print("     Results are NOT reliable until bugs are fixed.")
        print("\n     See ALGORITHMIC_BUGS_REPORT.md for details and fixes.")
        sys.exit(1)
    else:
        print("\n  ✓ No bugs confirmed in automated tests")
        print("    (Manual verification still recommended)")
        sys.exit(0)
