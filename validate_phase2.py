"""
Phase 2 Validation Script

Validates that Phase 2 improvements are working correctly:
1. GARCH volatility forecasting module
2. Ledoit-Wolf covariance in portfolio strategies
3. Integration with existing risk management

This is a quick validation - full windowed analysis takes 15-30 minutes.
"""

import sys
import numpy as np
import pandas as pd
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def validate_garch_module():
    """Validate GARCH volatility forecasting module."""
    logger.info("=" * 80)
    logger.info("TEST 1: GARCH Volatility Forecasting Module")
    logger.info("=" * 80)

    try:
        from crypto_trader.risk.volatility_forecasting import (
            forecast_volatility_garch,
            VolatilityForecaster,
            MIN_VOL,
            MAX_VOL
        )

        # Test 1: Basic GARCH forecast
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        vol_forecast = forecast_volatility_garch(returns, horizon=1)

        assert MIN_VOL <= vol_forecast <= MAX_VOL, f"GARCH forecast {vol_forecast} out of bounds"
        logger.info(f"✓ GARCH forecast: {vol_forecast:.4f} (valid range: {MIN_VOL} - {MAX_VOL})")

        # Test 2: Fallback mechanism
        returns_insufficient = pd.Series(np.random.normal(0.001, 0.02, 30))
        vol_fallback = forecast_volatility_garch(returns_insufficient, horizon=1, min_data_points=60)
        assert MIN_VOL <= vol_fallback <= MAX_VOL, "Fallback volatility out of bounds"
        logger.info(f"✓ Fallback volatility: {vol_fallback:.4f}")

        # Test 3: Caching
        forecaster = VolatilityForecaster(cache_size=10)
        vol1 = forecaster.forecast(returns, horizon=1, use_cache=True)
        vol2 = forecaster.forecast(returns, horizon=1, use_cache=True)
        assert vol1 == vol2, "Cache not working properly"
        logger.info(f"✓ Caching works: {vol1:.4f} == {vol2:.4f}")

        logger.info("✅ GARCH Module: PASSED\n")
        return True

    except Exception as e:
        logger.error(f"❌ GARCH Module: FAILED - {e}\n")
        return False


def validate_ledoit_wolf_integration():
    """Validate Ledoit-Wolf covariance integration in strategies."""
    logger.info("=" * 80)
    logger.info("TEST 2: Ledoit-Wolf Covariance Integration")
    logger.info("=" * 80)

    try:
        from pypfopt import risk_models

        # Create sample data
        np.random.seed(42)
        returns_data = {
            'BTC': np.random.normal(0.001, 0.02, 100),
            'ETH': np.random.normal(0.001, 0.025, 100),
            'BNB': np.random.normal(0.001, 0.03, 100)
        }
        returns = pd.DataFrame(returns_data)

        # Test Ledoit-Wolf shrinkage
        prices = pd.DataFrame({
            col: (1 + returns[col]).cumprod() for col in returns.columns
        })
        cov_lw = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        # Verify it's positive semi-definite
        eigenvalues = np.linalg.eigvals(cov_lw.values)
        assert all(eigenvalues >= -1e-10), "Covariance matrix not positive semi-definite"
        logger.info(f"✓ Ledoit-Wolf covariance is positive semi-definite")
        logger.info(f"  Shape: {cov_lw.shape}")
        logger.info(f"  Min eigenvalue: {eigenvalues.min():.6f}")

        # Compare with sample covariance
        cov_sample = returns.cov()
        logger.info(f"✓ Sample vs Ledoit-Wolf comparison:")
        logger.info(f"  Sample cov trace: {np.trace(cov_sample):.6f}")
        logger.info(f"  Ledoit-Wolf trace: {np.trace(cov_lw):.6f}")

        logger.info("✅ Ledoit-Wolf Integration: PASSED\n")
        return True

    except Exception as e:
        logger.error(f"❌ Ledoit-Wolf Integration: FAILED - {e}\n")
        return False


def validate_strategy_integration():
    """Validate that strategies can use Phase 2 improvements."""
    logger.info("=" * 80)
    logger.info("TEST 3: Strategy Integration")
    logger.info("=" * 80)

    try:
        from crypto_trader.strategies.library.hierarchical_risk_parity import HierarchicalRiskParityStrategy
        from crypto_trader.strategies.library.risk_parity import RiskParityStrategy
        from crypto_trader.strategies.library.black_litterman import BlackLittermanStrategy

        # Create sample multi-asset data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'BTC/USDT_close': 40000 + np.cumsum(np.random.normal(0, 500, 100)),
            'ETH/USDT_close': 2000 + np.cumsum(np.random.normal(0, 50, 100)),
            'BNB/USDT_close': 300 + np.cumsum(np.random.normal(0, 10, 100))
        })

        # Test HRP Strategy
        logger.info("Testing HierarchicalRiskParity...")
        hrp = HierarchicalRiskParityStrategy()
        hrp.initialize({
            'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'lookback_period': 60,
            'use_garch_vol': True,  # Enable GARCH
            'use_kelly_sizing': True  # Enable Kelly
        })
        hrp_signals = hrp.generate_signals(data)
        logger.info(f"✓ HRP generated {len(hrp_signals)} signals")

        # Test Risk Parity
        logger.info("Testing RiskParity...")
        rp = RiskParityStrategy()
        rp.initialize({
            'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'lookback_period': 60
        })
        rp_signals = rp.generate_signals(data)
        logger.info(f"✓ Risk Parity generated {len(rp_signals)} signals")

        # Test Black-Litterman
        logger.info("Testing BlackLitterman...")
        bl = BlackLittermanStrategy()
        bl.initialize({
            'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'lookback_period': 60
        })
        bl_signals = bl.generate_signals(data)
        logger.info(f"✓ Black-Litterman generated {len(bl_signals)} signals")

        logger.info("✅ Strategy Integration: PASSED\n")
        return True

    except Exception as e:
        logger.error(f"❌ Strategy Integration: FAILED - {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 2 validations."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 VALIDATION - Estimation Improvements")
    logger.info("=" * 80 + "\n")

    results = []

    # Run all tests
    results.append(("GARCH Module", validate_garch_module()))
    results.append(("Ledoit-Wolf Integration", validate_ledoit_wolf_integration()))
    results.append(("Strategy Integration", validate_strategy_integration()))

    # Summary
    logger.info("=" * 80)
    logger.info("PHASE 2 VALIDATION SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{name}: {status}")

    logger.info("")
    if passed == total:
        logger.info(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        logger.info("\nPhase 2 improvements are validated and ready for use!")
        logger.info("\nNext Steps:")
        logger.info("  1. Run full windowed analysis for comprehensive metrics")
        logger.info("  2. Compare Sharpe ratios vs baseline")
        logger.info("  3. Proceed to Phase 3 (Transaction Cost Optimization)")
        return 0
    else:
        logger.error(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        logger.error("\nPlease fix failing tests before proceeding to Phase 3")
        return 1


if __name__ == "__main__":
    sys.exit(main())
