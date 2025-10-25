"""
Test single-asset fallback behavior for all portfolio strategies.

Validates that strategies gracefully handle single-asset data instead of crashing.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from crypto_trader.strategies.library.black_litterman import BlackLittermanStrategy
from crypto_trader.strategies.library.copula_pairs_trading import CopulaPairsTradingStrategy
from crypto_trader.strategies.library.hierarchical_risk_parity import HierarchicalRiskParityStrategy
from crypto_trader.strategies.base import SignalType

# Track validation failures
all_validation_failures = []
total_tests = 0

print("🔍 Testing Single-Asset Fallback Behavior...\n")

# Create synthetic single-asset data
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
prices = 100 * (1 + np.random.randn(100) * 0.01).cumprod()

single_asset_data = pd.DataFrame({
    'timestamp': dates,
    'BTC_USDT_close': prices
})

print(f"Created single-asset test data: {len(single_asset_data)} periods\n")

# Test 1: Black-Litterman single-asset fallback
total_tests += 1
print("Test 1: Black-Litterman single-asset fallback")
try:
    strategy = BlackLittermanStrategy()
    strategy.initialize({
        'asset_symbols': ['BTC/USDT'],
        'lookback_period': 30,
        'rebalance_freq': 7
    })

    signals = strategy.generate_signals(single_asset_data)

    if signals.empty:
        all_validation_failures.append("Black-Litterman returned empty DataFrame for single asset")
    elif 'weight_BTC_USDT_close' not in signals.columns:
        all_validation_failures.append("Black-Litterman missing weight column for single asset")
    else:
        weight = signals['weight_BTC_USDT_close'].iloc[-1]
        if abs(weight - 1.0) > 0.001:
            all_validation_failures.append(f"Black-Litterman weight != 1.0: {weight}")
        else:
            print(f"  ✓ Black-Litterman correctly allocated 100% to single asset")
            print(f"  ✓ Generated {len(signals)} signal periods")
except Exception as e:
    all_validation_failures.append(f"Black-Litterman single-asset test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: CopulaPairsTrading single-asset fallback
total_tests += 1
print("\nTest 2: CopulaPairsTrading single-asset fallback")
try:
    strategy = CopulaPairsTradingStrategy()
    strategy.initialize({
        'asset_pairs': [('BTC/USDT', 'ETH/USDT')],  # Will be ignored
        'lookback_period': 30,
        'entry_threshold': 2.0,
        'exit_threshold': 0.5
    })

    signals = strategy.generate_signals(single_asset_data)

    if signals.empty:
        all_validation_failures.append("CopulaPairsTrading returned empty DataFrame for single asset")
    elif 'signal' not in signals.columns:
        all_validation_failures.append("CopulaPairsTrading missing signal column")
    else:
        # Should be all HOLD since pairs trading requires 2 assets
        non_hold_count = (signals['signal'] != SignalType.HOLD.value).sum()
        if non_hold_count > 0:
            all_validation_failures.append(
                f"CopulaPairsTrading generated non-HOLD signals for single asset: {non_hold_count}"
            )
        else:
            print(f"  ✓ CopulaPairsTrading correctly returned all HOLD signals")
            print(f"  ✓ Generated {len(signals)} signal periods")
except Exception as e:
    all_validation_failures.append(f"CopulaPairsTrading single-asset test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: HierarchicalRiskParity single-asset fallback
total_tests += 1
print("\nTest 3: HierarchicalRiskParity single-asset fallback")
try:
    strategy = HierarchicalRiskParityStrategy()
    strategy.initialize({
        'asset_symbols': ['BTC/USDT'],
        'lookback_period': 30,
        'rebalance_freq': 7
    })

    signals = strategy.generate_signals(single_asset_data)

    if signals.empty:
        all_validation_failures.append("HierarchicalRiskParity returned empty DataFrame for single asset")
    elif 'weight_BTC_USDT_close' not in signals.columns:
        all_validation_failures.append("HierarchicalRiskParity missing weight column for single asset")
    else:
        weight = signals['weight_BTC_USDT_close'].iloc[-1]
        if abs(weight - 1.0) > 0.001:
            all_validation_failures.append(f"HierarchicalRiskParity weight != 1.0: {weight}")
        else:
            print(f"  ✓ HierarchicalRiskParity correctly allocated 100% to single asset")
            print(f"  ✓ Generated {len(signals)} signal periods")
except Exception as e:
    all_validation_failures.append(f"HierarchicalRiskParity single-asset test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Verify no crashes with zero assets (edge case)
total_tests += 1
print("\nTest 4: Zero-asset edge case handling")
try:
    empty_data = pd.DataFrame({
        'timestamp': dates,
        'volume': np.random.rand(100)  # No price columns
    })

    bl_strategy = BlackLittermanStrategy()
    bl_strategy.initialize({'asset_symbols': [], 'lookback_period': 30})
    bl_signals = bl_strategy.generate_signals(empty_data)

    if not isinstance(bl_signals, pd.DataFrame):
        all_validation_failures.append("Black-Litterman didn't return DataFrame for zero assets")
    else:
        print(f"  ✓ Black-Litterman handled zero assets gracefully")

    hrp_strategy = HierarchicalRiskParityStrategy()
    hrp_strategy.initialize({'asset_symbols': [], 'lookback_period': 30})
    hrp_signals = hrp_strategy.generate_signals(empty_data)

    if not isinstance(hrp_signals, pd.DataFrame):
        all_validation_failures.append("HierarchicalRiskParity didn't return DataFrame for zero assets")
    else:
        print(f"  ✓ HierarchicalRiskParity handled zero assets gracefully")

    copula_strategy = CopulaPairsTradingStrategy()
    copula_strategy.initialize({'asset_pairs': [], 'lookback_period': 30})
    copula_signals = copula_strategy.generate_signals(empty_data)

    if not isinstance(copula_signals, pd.DataFrame):
        all_validation_failures.append("CopulaPairsTrading didn't return DataFrame for zero assets")
    else:
        print(f"  ✓ CopulaPairsTrading handled zero assets gracefully")

except Exception as e:
    all_validation_failures.append(f"Zero-asset edge case test failed: {e}")
    import traceback
    traceback.print_exc()

# Final validation result
print("\n" + "="*60)
if all_validation_failures:
    print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
    for failure in all_validation_failures:
        print(f"  - {failure}")
    sys.exit(1)
else:
    print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
    print("Single-asset fallback behavior is working correctly across all strategies")
    sys.exit(0)
