"""
Quick Validation: Portfolio Strategy Signal Format Fix

Validates that portfolio strategies (HRP, RiskParity, BlackLitterman) now
return proper signal/confidence/metadata format instead of weight columns.

This fixes the KeyError: 'signal' error encountered in the full validation run.
"""

import sys
import pandas as pd
import numpy as np
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def validate_strategy_output():
    """Test that portfolio strategies return correct signal format."""

    all_validation_failures = []
    total_tests = 0

    logger.info("=" * 80)
    logger.info("PORTFOLIO STRATEGY SIGNAL FORMAT VALIDATION")
    logger.info("=" * 80)

    # Create sample multi-asset data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    data = pd.DataFrame({
        'timestamp': dates,
        'BTC/USDT_close': 40000 + np.cumsum(np.random.normal(0, 500, 200)),
        'ETH/USDT_close': 2000 + np.cumsum(np.random.normal(0, 50, 200)),
        'BNB/USDT_close': 300 + np.cumsum(np.random.normal(0, 10, 200))
    })

    # Test 1: HierarchicalRiskParity
    logger.info("\n[Test 1] HierarchicalRiskParity signal format...")
    total_tests += 1
    try:
        from crypto_trader.strategies.library.hierarchical_risk_parity import HierarchicalRiskParityStrategy

        hrp = HierarchicalRiskParityStrategy()
        hrp.initialize({
            'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'lookback_period': 60,
            'use_garch_vol': False,  # Disable for quick test
            'use_kelly_sizing': False
        })

        signals = hrp.generate_signals(data)

        # Verify required columns
        required_cols = ['timestamp', 'signal', 'confidence', 'metadata']
        missing_cols = [col for col in required_cols if col not in signals.columns]

        if missing_cols:
            all_validation_failures.append(
                f"Test 1: HRP missing columns: {missing_cols}"
            )
        else:
            # Verify no weight_ columns remain
            weight_cols = [col for col in signals.columns if col.startswith('weight_')]
            if weight_cols:
                all_validation_failures.append(
                    f"Test 1: HRP still has weight columns: {weight_cols}"
                )
            else:
                # Check signal values are valid
                unique_signals = signals['signal'].unique()
                logger.info(f"  ✓ HRP has correct columns: {required_cols}")
                logger.info(f"  ✓ Unique signals: {unique_signals}")
                logger.info(f"  ✓ Signal counts: BUY={sum(signals['signal'] == 'BUY')}, "
                           f"SELL={sum(signals['signal'] == 'SELL')}, "
                           f"HOLD={sum(signals['signal'] == 'HOLD')}")

    except Exception as e:
        all_validation_failures.append(f"Test 1: HRP exception: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: RiskParity
    logger.info("\n[Test 2] RiskParity signal format...")
    total_tests += 1
    try:
        from crypto_trader.strategies.library.risk_parity import RiskParityStrategy

        rp = RiskParityStrategy()
        rp.initialize({
            'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'lookback_period': 60
        })

        signals = rp.generate_signals(data)

        # Verify required columns
        required_cols = ['timestamp', 'signal', 'confidence', 'metadata']
        missing_cols = [col for col in required_cols if col not in signals.columns]

        if missing_cols:
            all_validation_failures.append(
                f"Test 2: RiskParity missing columns: {missing_cols}"
            )
        else:
            weight_cols = [col for col in signals.columns if col.startswith('weight_')]
            if weight_cols:
                all_validation_failures.append(
                    f"Test 2: RiskParity still has weight columns: {weight_cols}"
                )
            else:
                unique_signals = signals['signal'].unique()
                logger.info(f"  ✓ RiskParity has correct columns: {required_cols}")
                logger.info(f"  ✓ Unique signals: {unique_signals}")

    except Exception as e:
        all_validation_failures.append(f"Test 2: RiskParity exception: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: BlackLitterman
    logger.info("\n[Test 3] BlackLitterman signal format...")
    total_tests += 1
    try:
        from crypto_trader.strategies.library.black_litterman import BlackLittermanStrategy

        bl = BlackLittermanStrategy()
        bl.initialize({
            'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'lookback_period': 60
        })

        signals = bl.generate_signals(data)

        # Verify required columns
        required_cols = ['timestamp', 'signal', 'confidence', 'metadata']
        missing_cols = [col for col in required_cols if col not in signals.columns]

        if missing_cols:
            all_validation_failures.append(
                f"Test 3: BlackLitterman missing columns: {missing_cols}"
            )
        else:
            weight_cols = [col for col in signals.columns if col.startswith('weight_')]
            if weight_cols:
                all_validation_failures.append(
                    f"Test 3: BlackLitterman still has weight columns: {weight_cols}"
                )
            else:
                unique_signals = signals['signal'].unique()
                logger.info(f"  ✓ BlackLitterman has correct columns: {required_cols}")
                logger.info(f"  ✓ Unique signals: {unique_signals}")

    except Exception as e:
        all_validation_failures.append(f"Test 3: BlackLitterman exception: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Metadata contains weights
    logger.info("\n[Test 4] Metadata contains weight information...")
    total_tests += 1
    try:
        # Use HRP signals from Test 1
        from crypto_trader.strategies.library.hierarchical_risk_parity import HierarchicalRiskParityStrategy

        hrp = HierarchicalRiskParityStrategy()
        hrp.initialize({
            'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'lookback_period': 60
        })

        signals = hrp.generate_signals(data)

        # Check first non-empty metadata
        for meta in signals['metadata']:
            if meta and 'weights' in meta:
                logger.info(f"  ✓ Metadata contains weights: {list(meta['weights'].keys())[:2]}...")
                logger.info(f"  ✓ Sample weight values: {list(meta['weights'].values())[:2]}")
                break
        else:
            all_validation_failures.append(
                "Test 4: No metadata found with 'weights' key"
            )

    except Exception as e:
        all_validation_failures.append(f"Test 4: Metadata exception: {e}")

    # Final validation result
    logger.info("\n" + "=" * 80)
    if all_validation_failures:
        logger.error(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            logger.error(f"  - {failure}")
        sys.exit(1)
    else:
        logger.success(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        logger.info("\nPortfolio strategies now return correct signal format!")
        logger.info("Ready to run full windowed validation.")
        sys.exit(0)


if __name__ == "__main__":
    validate_strategy_output()
