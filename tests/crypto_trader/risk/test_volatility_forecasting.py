"""
Unit tests for volatility forecasting module.

Tests the GARCH(1,1) volatility forecasting functionality including:
- Forecast computation with sufficient data
- Fallback behavior with insufficient data
- Validation of unreasonable forecasts
- Caching behavior
- Error handling
"""

import pytest
import numpy as np
import pandas as pd
from crypto_trader.risk.volatility_forecasting import (
    forecast_volatility_garch,
    VolatilityForecaster,
    _calculate_sample_volatility,
    MIN_VOL,
    MAX_VOL
)


class TestGARCHForecasting:
    """Test GARCH volatility forecasting functionality."""

    def test_garch_forecast_with_sufficient_data(self):
        """Test GARCH forecast computes successfully with enough data."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        vol_forecast = forecast_volatility_garch(returns, horizon=1)

        # Should return a valid volatility forecast
        assert isinstance(vol_forecast, float)
        assert MIN_VOL <= vol_forecast <= MAX_VOL
        assert vol_forecast > 0

    def test_fallback_with_insufficient_data(self):
        """Test fallback to sample volatility when data is insufficient."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 30))

        vol_forecast = forecast_volatility_garch(
            returns,
            horizon=1,
            min_data_points=60
        )

        # Should fallback to sample volatility
        expected_sample_vol = _calculate_sample_volatility(returns)
        assert abs(vol_forecast - expected_sample_vol) < 0.01

    def test_validation_rejects_unreasonable_forecasts(self):
        """Test that validation rejects forecasts outside reasonable bounds."""
        np.random.seed(42)
        # Create extreme returns that might produce unreasonable forecasts
        returns = pd.Series(np.random.normal(0, 1.0, 100))

        vol_forecast = forecast_volatility_garch(returns, horizon=1)

        # Should still be in valid range (due to fallback)
        assert MIN_VOL <= vol_forecast <= MAX_VOL

    def test_garch_vs_sample_volatility(self):
        """Test GARCH forecast differs from sample volatility appropriately."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 200))

        garch_vol = forecast_volatility_garch(returns, horizon=1)
        sample_vol = _calculate_sample_volatility(returns)

        # Both should be valid
        assert MIN_VOL <= garch_vol <= MAX_VOL
        assert MIN_VOL <= sample_vol <= MAX_VOL

        # They should be similar but not necessarily identical
        assert abs(garch_vol - sample_vol) < 1.0  # Within 100pp

    def test_different_horizons(self):
        """Test forecasting with different time horizons."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        vol_h1 = forecast_volatility_garch(returns, horizon=1)
        vol_h5 = forecast_volatility_garch(returns, horizon=5)

        # Both should be valid forecasts
        assert MIN_VOL <= vol_h1 <= MAX_VOL
        assert MIN_VOL <= vol_h5 <= MAX_VOL

    def test_custom_garch_orders(self):
        """Test GARCH with custom p and q orders."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        # Test GARCH(2,1)
        vol_forecast = forecast_volatility_garch(returns, horizon=1, p=2, q=1)

        assert isinstance(vol_forecast, float)
        assert MIN_VOL <= vol_forecast <= MAX_VOL

    def test_empty_series_raises_error(self):
        """Test that empty series raises ValueError."""
        with pytest.raises(ValueError, match="Returns series cannot be empty"):
            forecast_volatility_garch(pd.Series([]), horizon=1)

    def test_nan_only_series_raises_error(self):
        """Test that series with only NaN values raises ValueError."""
        returns = pd.Series([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="only NaN values"):
            forecast_volatility_garch(returns, horizon=1)

    def test_series_with_some_nans(self):
        """Test that series with some NaN values still works."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        returns.iloc[10:15] = np.nan  # Insert some NaNs

        vol_forecast = forecast_volatility_garch(returns, horizon=1)

        # Should still compute successfully
        assert isinstance(vol_forecast, float)
        assert MIN_VOL <= vol_forecast <= MAX_VOL


class TestSampleVolatility:
    """Test sample volatility calculation."""

    def test_sample_volatility_calculation(self):
        """Test basic sample volatility calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        sample_vol = _calculate_sample_volatility(returns)

        # Should be annualized and in valid range
        assert isinstance(sample_vol, float)
        assert sample_vol > 0
        assert MIN_VOL <= sample_vol <= MAX_VOL

    def test_sample_volatility_clipping(self):
        """Test that sample volatility is clipped to valid range."""
        # Create extreme returns
        returns_low = pd.Series(np.random.normal(0, 0.0001, 100))
        returns_high = pd.Series(np.random.normal(0, 10.0, 100))

        vol_low = _calculate_sample_volatility(returns_low)
        vol_high = _calculate_sample_volatility(returns_high)

        # Both should be clipped to valid range
        assert vol_low >= MIN_VOL
        assert vol_high <= MAX_VOL


class TestVolatilityForecaster:
    """Test VolatilityForecaster class with caching."""

    def test_forecaster_initialization(self):
        """Test forecaster initialization."""
        forecaster = VolatilityForecaster(
            cache_size=50,
            min_data_points=60,
            p=1,
            q=1
        )

        assert forecaster.cache_size == 50
        assert forecaster.min_data_points == 60
        assert forecaster.p == 1
        assert forecaster.q == 1

    def test_caching_behavior(self):
        """Test that caching returns same results for same input."""
        np.random.seed(42)
        forecaster = VolatilityForecaster(cache_size=10)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        # First call - computes
        vol1 = forecaster.forecast(returns, horizon=1, use_cache=True)

        # Second call - uses cache
        vol2 = forecaster.forecast(returns, horizon=1, use_cache=True)

        # Should be identical
        assert vol1 == vol2

    def test_cache_disabled(self):
        """Test forecasting with cache disabled."""
        np.random.seed(42)
        forecaster = VolatilityForecaster(cache_size=10)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        vol1 = forecaster.forecast(returns, horizon=1, use_cache=False)
        vol2 = forecaster.forecast(returns, horizon=1, use_cache=False)

        # Should both be valid (may differ slightly due to stochastic fitting)
        assert MIN_VOL <= vol1 <= MAX_VOL
        assert MIN_VOL <= vol2 <= MAX_VOL

    def test_cache_size_limit(self):
        """Test that cache respects size limit."""
        forecaster = VolatilityForecaster(cache_size=2)
        np.random.seed(42)

        # Add 3 different forecasts
        for i in range(3):
            returns = pd.Series(np.random.normal(0.001, 0.02, 100) + i * 0.01)
            forecaster.forecast(returns, horizon=1, use_cache=True)

        # Cache should only have 2 entries (oldest removed)
        assert len(forecaster._cache) <= 2

    def test_clear_cache(self):
        """Test cache clearing."""
        np.random.seed(42)
        forecaster = VolatilityForecaster(cache_size=10)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        # Add to cache
        forecaster.forecast(returns, horizon=1, use_cache=True)
        assert len(forecaster._cache) > 0

        # Clear cache
        forecaster.clear_cache()
        assert len(forecaster._cache) == 0

    def test_different_horizons_cached_separately(self):
        """Test that different horizons are cached separately."""
        np.random.seed(42)
        forecaster = VolatilityForecaster(cache_size=10)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        vol_h1 = forecaster.forecast(returns, horizon=1, use_cache=True)
        vol_h5 = forecaster.forecast(returns, horizon=5, use_cache=True)

        # Should have 2 cache entries
        assert len(forecaster._cache) == 2

        # Forecasts should be valid
        assert MIN_VOL <= vol_h1 <= MAX_VOL
        assert MIN_VOL <= vol_h5 <= MAX_VOL


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_value_series(self):
        """Test with series containing single value."""
        returns = pd.Series([0.01])

        # Should fallback to sample vol (which will be 0, then clipped to MIN_VOL)
        vol_forecast = forecast_volatility_garch(
            returns,
            horizon=1,
            min_data_points=60
        )

        assert vol_forecast == MIN_VOL

    def test_zero_volatility_series(self):
        """Test with series having zero volatility."""
        returns = pd.Series([0.01] * 100)

        vol_forecast = forecast_volatility_garch(returns, horizon=1)

        # Should return MIN_VOL due to clipping
        assert vol_forecast == MIN_VOL

    def test_negative_returns(self):
        """Test with all negative returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(-0.01, 0.02, 100))

        vol_forecast = forecast_volatility_garch(returns, horizon=1)

        # Should still compute valid volatility
        assert MIN_VOL <= vol_forecast <= MAX_VOL

    def test_mixed_positive_negative_returns(self):
        """Test with mixed positive and negative returns."""
        np.random.seed(42)
        returns = pd.Series(
            [0.02, -0.03, 0.01, -0.015, 0.025, -0.02] * 20
        )

        vol_forecast = forecast_volatility_garch(returns, horizon=1)

        # Should compute valid volatility
        assert MIN_VOL <= vol_forecast <= MAX_VOL
