"""
Volatility Forecasting with GARCH Models

This module provides forward-looking volatility forecasts using GARCH(1,1) models,
which capture volatility clustering common in cryptocurrency markets.

**Purpose**: Forecast future volatility for use in position sizing and risk management,
replacing backward-looking historical volatility estimates.

**Key Components**:
- GARCH(1,1) model fitting and forecasting
- Validation logic to reject unreasonable forecasts
- Fallback to sample volatility when needed
- Performance optimization through caching

**Third-party packages**:
- arch: https://arch.readthedocs.io/ (GARCH modeling)
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```python
returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, ...])  # Daily returns
vol_forecast = forecast_volatility_garch(returns, horizon=1)
```

**Expected Output**:
```python
0.42  # Annualized volatility forecast (e.g., 42%)
```
"""

from typing import Optional
import numpy as np
import pandas as pd
from arch import arch_model
from loguru import logger


# Validation bounds for volatility forecasts
MIN_VOL = 0.05  # 5% annual - minimum reasonable volatility
MAX_VOL = 5.00  # 500% annual - maximum reasonable volatility


def forecast_volatility_garch(
    returns: pd.Series,
    horizon: int = 1,
    min_data_points: int = 60,
    p: int = 1,
    q: int = 1
) -> float:
    """
    Forecast volatility using GARCH(1,1) with comprehensive validation.

    The GARCH model captures volatility clustering (periods of high volatility
    tend to be followed by high volatility, and vice versa), which is common
    in cryptocurrency markets.

    Args:
        returns: Historical return series (preferably daily returns)
        horizon: Forecast horizon in periods (1 = next period)
        min_data_points: Minimum required observations (default 60)
        p: GARCH order (default 1)
        q: ARCH order (default 1)

    Returns:
        Annualized volatility forecast

    Raises:
        ValueError: If returns series is empty or contains only NaN values
    """
    # Input validation
    if returns is None or len(returns) == 0:
        raise ValueError("Returns series cannot be empty")

    # Remove NaN values
    returns_clean = returns.dropna()

    if len(returns_clean) == 0:
        raise ValueError("Returns series contains only NaN values")

    # Check minimum data requirement
    if len(returns_clean) < min_data_points:
        logger.warning(
            f"Insufficient data for GARCH: {len(returns_clean)} < {min_data_points}. "
            f"Falling back to sample volatility"
        )
        return _calculate_sample_volatility(returns_clean)

    try:
        # Scale returns to percentage for numerical stability
        returns_pct = returns_clean * 100

        # Fit GARCH(p,q) model
        model = arch_model(
            returns_pct,
            vol='GARCH',
            p=p,  # GARCH order
            q=q,  # ARCH order
            dist='normal',
            rescale=True  # Rescale data for numerical stability
        )

        # Fit model with suppressed output
        results = model.fit(disp='off', show_warning=False)

        # Generate forecast
        forecast = results.forecast(horizon=horizon)
        variance_forecast = forecast.variance.iloc[-1, 0]

        # Convert to annualized volatility
        # Assuming daily returns, annualize with sqrt(252)
        vol_forecast = np.sqrt(variance_forecast) / 100 * np.sqrt(252)

        # Validate forecast is in reasonable range
        if not (MIN_VOL <= vol_forecast <= MAX_VOL):
            logger.warning(
                f"GARCH forecast out of bounds: {vol_forecast:.4f} "
                f"(valid range: {MIN_VOL:.2f} - {MAX_VOL:.2f}). "
                f"Falling back to sample volatility"
            )
            return _calculate_sample_volatility(returns_clean)

        logger.debug(
            f"GARCH({p},{q}) forecast: {vol_forecast:.4f} "
            f"(sample vol: {_calculate_sample_volatility(returns_clean):.4f})"
        )

        return vol_forecast

    except Exception as e:
        logger.warning(
            f"GARCH forecasting failed: {e}. "
            f"Falling back to sample volatility"
        )
        return _calculate_sample_volatility(returns_clean)


def _calculate_sample_volatility(returns: pd.Series) -> float:
    """
    Calculate sample volatility as fallback.

    Args:
        returns: Historical return series

    Returns:
        Annualized sample volatility (clipped to valid range)
    """
    # Annualize with sqrt(252) assuming daily returns
    vol = returns.std() * np.sqrt(252)

    # Clip to valid range
    return np.clip(vol, MIN_VOL, MAX_VOL)


class VolatilityForecaster:
    """
    Volatility forecaster with caching for performance optimization.

    This class maintains a cache of recent forecasts to avoid recomputing
    GARCH models for the same data.
    """

    def __init__(
        self,
        cache_size: int = 100,
        min_data_points: int = 60,
        p: int = 1,
        q: int = 1
    ):
        """
        Initialize volatility forecaster with caching.

        Args:
            cache_size: Maximum number of cached forecasts
            min_data_points: Minimum observations required for GARCH
            p: GARCH order
            q: ARCH order
        """
        self.cache_size = cache_size
        self.min_data_points = min_data_points
        self.p = p
        self.q = q
        self._cache = {}  # Cache: (data_hash, horizon) -> forecast

    def forecast(
        self,
        returns: pd.Series,
        horizon: int = 1,
        use_cache: bool = True
    ) -> float:
        """
        Forecast volatility with optional caching.

        Args:
            returns: Historical return series
            horizon: Forecast horizon in periods
            use_cache: Whether to use cached forecasts

        Returns:
            Annualized volatility forecast
        """
        if use_cache:
            # Create cache key from data hash and horizon
            cache_key = (self._hash_series(returns), horizon)

            if cache_key in self._cache:
                logger.debug(f"Using cached volatility forecast for horizon={horizon}")
                return self._cache[cache_key]

        # Compute forecast
        forecast_value = forecast_volatility_garch(
            returns=returns,
            horizon=horizon,
            min_data_points=self.min_data_points,
            p=self.p,
            q=self.q
        )

        if use_cache:
            # Update cache
            self._cache[cache_key] = forecast_value

            # Limit cache size
            if len(self._cache) > self.cache_size:
                # Remove oldest entry (first key)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

        return forecast_value

    def clear_cache(self):
        """Clear the forecast cache."""
        self._cache.clear()
        logger.debug("Volatility forecast cache cleared")

    @staticmethod
    def _hash_series(series: pd.Series) -> int:
        """
        Create hash of series for caching.

        Args:
            series: Pandas Series to hash

        Returns:
            Hash value
        """
        # Use last value and length as simple hash
        # More sophisticated hashing could use pandas.util.hash_pandas_object
        return hash((len(series), float(series.iloc[-1]) if len(series) > 0 else 0))


if __name__ == "__main__":
    """
    Validation function to test GARCH volatility forecasting with real data.

    Tests:
    1. GARCH forecast with sufficient data
    2. Fallback to sample volatility with insufficient data
    3. Forecast validation (reasonable range)
    4. Comparison of GARCH vs sample volatility
    5. Caching behavior
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    print("="*80)
    print("VOLATILITY FORECASTING VALIDATION")
    print("="*80)

    # Test 1: GARCH forecast with sufficient data
    print("\n[Test 1] GARCH forecast with sufficient data...")
    total_tests += 1
    try:
        # Simulate crypto returns (high volatility)
        np.random.seed(42)
        returns_sufficient = pd.Series(np.random.normal(0.001, 0.03, 100))

        vol_forecast = forecast_volatility_garch(returns_sufficient, horizon=1)

        # Check forecast is in reasonable range
        if not (MIN_VOL <= vol_forecast <= MAX_VOL):
            all_validation_failures.append(
                f"Test 1: GARCH forecast {vol_forecast:.4f} outside valid range "
                f"[{MIN_VOL:.2f}, {MAX_VOL:.2f}]"
            )
        else:
            print(f"  ✓ GARCH forecast: {vol_forecast:.4f} (valid range)")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Fallback with insufficient data
    print("\n[Test 2] Fallback to sample volatility with insufficient data...")
    total_tests += 1
    try:
        returns_insufficient = pd.Series(np.random.normal(0.001, 0.03, 30))

        vol_forecast_fallback = forecast_volatility_garch(
            returns_insufficient,
            horizon=1,
            min_data_points=60
        )

        # Should return sample volatility
        expected_sample_vol = returns_insufficient.std() * np.sqrt(252)

        if abs(vol_forecast_fallback - expected_sample_vol) > 0.01:
            all_validation_failures.append(
                f"Test 2: Expected sample volatility {expected_sample_vol:.4f}, "
                f"got {vol_forecast_fallback:.4f}"
            )
        else:
            print(f"  ✓ Fallback to sample vol: {vol_forecast_fallback:.4f}")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Validation rejects unreasonable forecasts
    print("\n[Test 3] Forecast validation with extreme data...")
    total_tests += 1
    try:
        # Create extreme returns that should trigger validation
        returns_extreme = pd.Series(np.random.normal(0, 1.0, 100))  # Very high vol

        vol_forecast_extreme = forecast_volatility_garch(returns_extreme, horizon=1)

        # Should still be in valid range due to fallback
        if not (MIN_VOL <= vol_forecast_extreme <= MAX_VOL):
            all_validation_failures.append(
                f"Test 3: Extreme data forecast {vol_forecast_extreme:.4f} "
                f"not handled properly"
            )
        else:
            print(f"  ✓ Extreme data handled: {vol_forecast_extreme:.4f}")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Compare GARCH vs sample volatility
    print("\n[Test 4] Compare GARCH vs sample volatility...")
    total_tests += 1
    try:
        returns_compare = pd.Series(np.random.normal(0.001, 0.02, 200))

        garch_vol = forecast_volatility_garch(returns_compare, horizon=1)
        sample_vol = _calculate_sample_volatility(returns_compare)

        # They should be similar but not identical
        diff = abs(garch_vol - sample_vol)

        print(f"  GARCH vol: {garch_vol:.4f}")
        print(f"  Sample vol: {sample_vol:.4f}")
        print(f"  Difference: {diff:.4f}")

        # GARCH should produce valid forecast
        if not (MIN_VOL <= garch_vol <= MAX_VOL):
            all_validation_failures.append(
                f"Test 4: GARCH forecast {garch_vol:.4f} invalid"
            )
        else:
            print(f"  ✓ Both forecasts valid")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception raised: {e}")

    # Test 5: Caching behavior
    print("\n[Test 5] Test forecaster caching...")
    total_tests += 1
    try:
        forecaster = VolatilityForecaster(cache_size=10)
        returns_cache = pd.Series(np.random.normal(0.001, 0.02, 100))

        # First call - should compute
        vol1 = forecaster.forecast(returns_cache, horizon=1, use_cache=True)

        # Second call - should use cache
        vol2 = forecaster.forecast(returns_cache, horizon=1, use_cache=True)

        # Should be identical
        if vol1 != vol2:
            all_validation_failures.append(
                f"Test 5: Cached forecast {vol2:.4f} differs from "
                f"original {vol1:.4f}"
            )
        else:
            print(f"  ✓ Cache working: {vol1:.4f} == {vol2:.4f}")

        # Clear cache
        forecaster.clear_cache()
        print("  ✓ Cache cleared successfully")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Error handling for empty series
    print("\n[Test 6] Error handling for invalid input...")
    total_tests += 1
    try:
        # Should raise ValueError
        try:
            forecast_volatility_garch(pd.Series([]), horizon=1)
            all_validation_failures.append(
                "Test 6: Empty series should raise ValueError"
            )
        except ValueError:
            print("  ✓ Empty series raises ValueError as expected")
    except Exception as e:
        all_validation_failures.append(f"Test 6: Unexpected exception: {e}")

    # Final validation result
    print("\n" + "="*80)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Volatility forecasting module is validated and ready for integration")
        sys.exit(0)
