"""
Unit Tests for Portfolio Rebalancer Strategy

Tests the PortfolioRebalancerStrategy implementation including:
- Configuration validation
- Signal generation
- Rebalancing logic
- Weight calculations
- Edge cases

**Purpose**: Verify portfolio rebalancing strategy works correctly with various scenarios.

**Third-party packages**:
- pytest: https://docs.pytest.org/
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
"""

from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
import numpy as np
import pytest

from crypto_trader.strategies.library.portfolio_rebalancer import PortfolioRebalancerStrategy
from crypto_trader.strategies.base import SignalType


class TestPortfolioRebalancerInitialization:
    """Test strategy initialization and configuration validation."""

    def test_basic_initialization(self):
        """Test strategy can be initialized with default parameters."""
        strategy = PortfolioRebalancerStrategy()
        assert strategy.name == "PortfolioRebalancer"
        assert strategy.rebalance_threshold == 0.15
        assert strategy.min_rebalance_interval_hours == 24

    def test_initialization_with_valid_config(self):
        """Test initialization with valid configuration."""
        assets = [
            ("BTC/USDT", 0.50),
            ("ETH/USDT", 0.30),
            ("SOL/USDT", 0.20)
        ]

        config = {
            "assets": assets,
            "rebalance_threshold": 0.10,
            "min_rebalance_interval_hours": 48
        }

        strategy = PortfolioRebalancerStrategy()
        strategy.initialize(config)

        params = strategy.get_parameters()
        assert params['assets'] == assets
        assert params['rebalance_threshold'] == 0.10
        assert params['min_rebalance_interval_hours'] == 48

    def test_initialization_without_assets_raises_error(self):
        """Test that initialization without assets raises ValueError."""
        strategy = PortfolioRebalancerStrategy()

        with pytest.raises(ValueError, match="Portfolio strategy requires 'assets' configuration"):
            strategy.initialize({})

    def test_initialization_with_single_asset_raises_error(self):
        """Test that portfolio with less than 2 assets raises ValueError."""
        strategy = PortfolioRebalancerStrategy()

        config = {
            "assets": [("BTC/USDT", 1.0)]
        }

        with pytest.raises(ValueError, match="Portfolio must have at least 2 assets"):
            strategy.initialize(config)

    def test_initialization_with_invalid_weights_raises_error(self):
        """Test that weights not summing to 1.0 raises ValueError."""
        strategy = PortfolioRebalancerStrategy()

        # Weights sum to 0.8 instead of 1.0
        config = {
            "assets": [
                ("BTC/USDT", 0.50),
                ("ETH/USDT", 0.30)
            ]
        }

        with pytest.raises(ValueError, match="Asset weights must sum to 1.0"):
            strategy.initialize(config)

    def test_initialization_with_invalid_threshold_raises_error(self):
        """Test that invalid threshold values raise ValueError."""
        strategy = PortfolioRebalancerStrategy()

        # Threshold of 0 is invalid
        config = {
            "assets": [("BTC/USDT", 0.5), ("ETH/USDT", 0.5)],
            "rebalance_threshold": 0.0
        }

        with pytest.raises(ValueError, match="Rebalance threshold must be between 0 and 1"):
            strategy.initialize(config)

        # Threshold >= 1.0 is invalid
        config["rebalance_threshold"] = 1.0
        with pytest.raises(ValueError, match="Rebalance threshold must be between 0 and 1"):
            strategy.initialize(config)


class TestPortfolioSignalGeneration:
    """Test signal generation logic."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.assets = [
            ("BTC/USDT", 0.50),
            ("ETH/USDT", 0.30),
            ("SOL/USDT", 0.20)
        ]

        self.strategy = PortfolioRebalancerStrategy()
        self.strategy.initialize({
            "assets": self.assets,
            "rebalance_threshold": 0.15,
            "min_rebalance_interval_hours": 24
        })

    def test_signal_generation_with_valid_data(self):
        """Test signal generation with valid multi-asset data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')

        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': np.linspace(40000, 45000, 100),
                'open': np.linspace(39900, 44900, 100),
                'high': np.linspace(40100, 45100, 100),
                'low': np.linspace(39800, 44800, 100),
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': np.linspace(2000, 2100, 100),
                'open': np.linspace(1990, 2090, 100),
                'high': np.linspace(2010, 2110, 100),
                'low': np.linspace(1980, 2080, 100),
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates),
            "SOL/USDT": pd.DataFrame({
                'close': np.linspace(100, 105, 100),
                'open': np.linspace(99, 104, 100),
                'high': np.linspace(101, 106, 100),
                'low': np.linspace(98, 103, 100),
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates)
        }

        signals = self.strategy.generate_signals(portfolio_data)

        assert signals is not None
        assert not signals.empty
        assert len(signals) == 100
        assert 'timestamp' in signals.columns
        assert 'BTC/USDT_signal' in signals.columns
        assert 'ETH/USDT_signal' in signals.columns
        assert 'SOL/USDT_signal' in signals.columns
        assert 'rebalance_event' in signals.columns
        assert 'metadata' in signals.columns

    def test_signal_generation_with_missing_asset_raises_error(self):
        """Test that missing asset data raises ValueError."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')

        # Only provide data for 2 out of 3 assets
        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': [40000] * 10,
                'open': [39900] * 10,
                'high': [40100] * 10,
                'low': [39800] * 10,
                'volume': [1000] * 10
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': [2000] * 10,
                'open': [1990] * 10,
                'high': [2010] * 10,
                'low': [1980] * 10,
                'volume': [1000] * 10
            }, index=dates)
        }

        with pytest.raises(ValueError, match="Missing data for asset: SOL/USDT"):
            self.strategy.generate_signals(portfolio_data)

    def test_signal_generation_with_insufficient_data_raises_error(self):
        """Test that insufficient overlapping data raises ValueError."""
        # Create data with only 1 overlapping timestamp
        dates1 = pd.date_range('2024-01-01', periods=1, freq='1h')
        dates2 = pd.date_range('2024-01-01', periods=1, freq='1h')

        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': [40000],
                'open': [39900],
                'high': [40100],
                'low': [39800],
                'volume': [1000]
            }, index=dates1),
            "ETH/USDT": pd.DataFrame({
                'close': [2000],
                'open': [1990],
                'high': [2010],
                'low': [1980],
                'volume': [1000]
            }, index=dates2),
            "SOL/USDT": pd.DataFrame({
                'close': [100],
                'open': [99],
                'high': [101],
                'low': [98],
                'volume': [1000]
            }, index=dates1)
        }

        with pytest.raises(ValueError, match="Insufficient overlapping data across assets"):
            self.strategy.generate_signals(portfolio_data)


class TestRebalancingLogic:
    """Test rebalancing trigger logic and weight calculations."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.assets = [
            ("BTC/USDT", 0.50),
            ("ETH/USDT", 0.50)
        ]

        self.strategy = PortfolioRebalancerStrategy()
        self.strategy.initialize({
            "assets": self.assets,
            "rebalance_threshold": 0.15,
            "min_rebalance_interval_hours": 24
        })

    def test_no_rebalance_when_weights_within_threshold(self):
        """Test that no rebalance occurs when weights are within threshold."""
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')

        # Create prices that keep weights balanced (both increase equally)
        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': np.linspace(40000, 42000, 50),
                'open': np.linspace(39900, 41900, 50),
                'high': np.linspace(40100, 42100, 50),
                'low': np.linspace(39800, 41800, 50),
                'volume': [1000] * 50
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': np.linspace(2000, 2100, 50),  # 5% increase, same as BTC
                'open': np.linspace(1990, 2090, 50),
                'high': np.linspace(2010, 2110, 50),
                'low': np.linspace(1980, 2080, 50),
                'volume': [1000] * 50
            }, index=dates)
        }

        signals = self.strategy.generate_signals(portfolio_data)

        # Should have no rebalance events (or very few)
        rebalance_count = signals['rebalance_event'].sum()
        assert rebalance_count <= 2  # Allow minimal rebalancing due to numerical drift

    def test_rebalance_triggered_when_threshold_exceeded(self):
        """Test that rebalance is triggered when deviation exceeds threshold."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')

        # BTC goes up 100%, ETH stays flat -> will exceed 15% threshold
        # With 50/50 target: BTC doubles means weight goes to 66.7%, deviation = 16.7%
        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': np.linspace(40000, 80000, 100),  # 100% increase
                'open': np.linspace(39900, 79900, 100),
                'high': np.linspace(40100, 80100, 100),
                'low': np.linspace(39800, 79800, 100),
                'volume': [1000] * 100
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': [2000] * 100,  # No change
                'open': [1990] * 100,
                'high': [2010] * 100,
                'low': [1980] * 100,
                'volume': [1000] * 100
            }, index=dates)
        }

        signals = self.strategy.generate_signals(portfolio_data)

        # Should have at least 1 rebalance event
        rebalance_count = signals['rebalance_event'].sum()
        assert rebalance_count >= 1

    def test_rebalance_signals_correctness(self):
        """Test that rebalance generates correct buy/sell signals."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')

        # BTC goes up significantly -> should sell BTC, buy ETH
        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': np.linspace(40000, 80000, 100),  # 100% increase
                'open': np.linspace(39900, 79900, 100),
                'high': np.linspace(40100, 80100, 100),
                'low': np.linspace(39800, 79800, 100),
                'volume': [1000] * 100
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': [2000] * 100,  # No change
                'open': [1990] * 100,
                'high': [2010] * 100,
                'low': [1980] * 100,
                'volume': [1000] * 100
            }, index=dates)
        }

        signals = self.strategy.generate_signals(portfolio_data)

        # Find first rebalance event
        rebalance_rows = signals[signals['rebalance_event'] == True]

        if len(rebalance_rows) > 0:
            first_rebalance = rebalance_rows.iloc[0]

            # BTC should be overweight -> SELL signal
            assert first_rebalance['BTC/USDT_signal'] == SignalType.SELL.value

            # ETH should be underweight -> BUY signal
            assert first_rebalance['ETH/USDT_signal'] == SignalType.BUY.value

    def test_minimum_rebalance_interval_enforced(self):
        """Test that minimum interval between rebalances is enforced."""
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')

        # Create volatile prices that would trigger many rebalances
        btc_prices = [40000]
        eth_prices = [2000]

        for i in range(1, 50):
            if i % 5 == 0:  # Every 5 hours, create big swing
                btc_prices.append(btc_prices[-1] * 1.3)
                eth_prices.append(eth_prices[-1] * 0.8)
            else:
                btc_prices.append(btc_prices[-1])
                eth_prices.append(eth_prices[-1])

        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': btc_prices,
                'open': [p * 0.999 for p in btc_prices],
                'high': [p * 1.001 for p in btc_prices],
                'low': [p * 0.998 for p in btc_prices],
                'volume': [1000] * 50
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': eth_prices,
                'open': [p * 0.999 for p in eth_prices],
                'high': [p * 1.001 for p in eth_prices],
                'low': [p * 0.998 for p in eth_prices],
                'volume': [1000] * 50
            }, index=dates)
        }

        signals = self.strategy.generate_signals(portfolio_data)

        # Check that rebalances are at least 24 hours apart
        rebalance_rows = signals[signals['rebalance_event'] == True]

        if len(rebalance_rows) > 1:
            timestamps = rebalance_rows['timestamp'].values
            for i in range(1, len(timestamps)):
                time_diff = pd.Timestamp(timestamps[i]) - pd.Timestamp(timestamps[i-1])
                hours_diff = time_diff.total_seconds() / 3600
                assert hours_diff >= 24


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_three_asset_portfolio(self):
        """Test portfolio with 3 assets."""
        assets = [
            ("BTC/USDT", 0.40),
            ("ETH/USDT", 0.35),
            ("SOL/USDT", 0.25)
        ]

        strategy = PortfolioRebalancerStrategy()
        strategy.initialize({
            "assets": assets,
            "rebalance_threshold": 0.10
        })

        dates = pd.date_range('2024-01-01', periods=50, freq='1h')

        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': np.linspace(40000, 45000, 50),
                'open': np.linspace(39900, 44900, 50),
                'high': np.linspace(40100, 45100, 50),
                'low': np.linspace(39800, 44800, 50),
                'volume': [1000] * 50
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': np.linspace(2000, 2200, 50),
                'open': np.linspace(1990, 2190, 50),
                'high': np.linspace(2010, 2210, 50),
                'low': np.linspace(1980, 2180, 50),
                'volume': [1000] * 50
            }, index=dates),
            "SOL/USDT": pd.DataFrame({
                'close': np.linspace(100, 110, 50),
                'open': np.linspace(99, 109, 50),
                'high': np.linspace(101, 111, 50),
                'low': np.linspace(98, 108, 50),
                'volume': [1000] * 50
            }, index=dates)
        }

        signals = strategy.generate_signals(portfolio_data)

        assert len(signals) == 50
        assert 'BTC/USDT_signal' in signals.columns
        assert 'ETH/USDT_signal' in signals.columns
        assert 'SOL/USDT_signal' in signals.columns

    def test_extreme_price_movements(self):
        """Test portfolio with extreme price movements."""
        assets = [
            ("BTC/USDT", 0.50),
            ("ETH/USDT", 0.50)
        ]

        strategy = PortfolioRebalancerStrategy()
        strategy.initialize({
            "assets": assets,
            "rebalance_threshold": 0.10  # Lower threshold to ensure trigger
        })

        dates = pd.date_range('2024-01-01', periods=10, freq='1h')

        # BTC crashes 60%, ETH stays stable
        # With 50/50 target: BTC drops to 40% of value, weight becomes 28.6%, deviation = 21.4%
        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': [40000, 16000, 16000, 16000, 16000, 16000, 16000, 16000, 16000, 16000],
                'open': [40000, 39000, 16000, 16000, 16000, 16000, 16000, 16000, 16000, 16000],
                'high': [40000, 40000, 16100, 16100, 16100, 16100, 16100, 16100, 16100, 16100],
                'low': [39000, 15000, 15900, 15900, 15900, 15900, 15900, 15900, 15900, 15900],
                'volume': [1000] * 10
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': [2000] * 10,
                'open': [1990] * 10,
                'high': [2010] * 10,
                'low': [1980] * 10,
                'volume': [1000] * 10
            }, index=dates)
        }

        signals = strategy.generate_signals(portfolio_data)

        # Should trigger rebalance
        assert signals['rebalance_event'].sum() >= 1

    def test_partial_timestamp_overlap(self):
        """Test handling of partial timestamp overlap between assets."""
        assets = [
            ("BTC/USDT", 0.50),
            ("ETH/USDT", 0.50)
        ]

        strategy = PortfolioRebalancerStrategy()
        strategy.initialize({
            "assets": assets,
            "rebalance_threshold": 0.15
        })

        # BTC has 100 hours of data starting from Jan 1
        dates_btc = pd.date_range('2024-01-01 00:00:00', periods=100, freq='1h')
        # ETH only has 50 hours, starting from Jan 2 (25 hours after BTC starts)
        dates_eth = pd.date_range('2024-01-02 01:00:00', periods=50, freq='1h')

        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': [40000] * 100,
                'open': [39900] * 100,
                'high': [40100] * 100,
                'low': [39800] * 100,
                'volume': [1000] * 100
            }, index=dates_btc),
            "ETH/USDT": pd.DataFrame({
                'close': [2000] * 50,
                'open': [1990] * 50,
                'high': [2010] * 50,
                'low': [1980] * 50,
                'volume': [1000] * 50
            }, index=dates_eth)
        }

        signals = strategy.generate_signals(portfolio_data)

        # Should only process the overlapping 50 hours
        assert len(signals) == 50


class TestMetadataTracking:
    """Test metadata tracking in signals."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.assets = [
            ("BTC/USDT", 0.60),
            ("ETH/USDT", 0.40)
        ]

        self.strategy = PortfolioRebalancerStrategy()
        self.strategy.initialize({
            "assets": self.assets,
            "rebalance_threshold": 0.10
        })

    def test_metadata_contains_current_weights(self):
        """Test that metadata includes current portfolio weights."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')

        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': [40000] * 20,
                'open': [39900] * 20,
                'high': [40100] * 20,
                'low': [39800] * 20,
                'volume': [1000] * 20
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': [2000] * 20,
                'open': [1990] * 20,
                'high': [2010] * 20,
                'low': [1980] * 20,
                'volume': [1000] * 20
            }, index=dates)
        }

        signals = self.strategy.generate_signals(portfolio_data)

        # Check first row metadata
        first_metadata = signals.iloc[0]['metadata']
        assert 'current_weights' in first_metadata
        assert 'BTC/USDT' in first_metadata['current_weights']
        assert 'ETH/USDT' in first_metadata['current_weights']

    def test_rebalance_metadata_includes_deviation(self):
        """Test that rebalance event metadata includes max deviation."""
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')

        # Create scenario that triggers rebalance
        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': np.linspace(40000, 60000, 50),  # 50% increase
                'open': np.linspace(39900, 59900, 50),
                'high': np.linspace(40100, 60100, 50),
                'low': np.linspace(39800, 59800, 50),
                'volume': [1000] * 50
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': [2000] * 50,  # No change
                'open': [1990] * 50,
                'high': [2010] * 50,
                'low': [1980] * 50,
                'volume': [1000] * 50
            }, index=dates)
        }

        signals = self.strategy.generate_signals(portfolio_data)

        # Find rebalance events
        rebalance_rows = signals[signals['rebalance_event'] == True]

        if len(rebalance_rows) > 0:
            first_rebalance = rebalance_rows.iloc[0]
            metadata = first_rebalance['metadata']

            assert 'max_deviation' in metadata
            assert 'reason' in metadata
            assert metadata['reason'] == 'threshold_rebalance'
            assert metadata['max_deviation'] > 0.10  # Above threshold


if __name__ == "__main__":
    """
    Validation block for Portfolio Rebalancer Unit Tests.
    Runs all tests and reports results.
    """
    import sys

    # Run pytest with verbose output
    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    if exit_code == 0:
        print("\n" + "="*70)
        print("✅ ALL PORTFOLIO REBALANCER TESTS PASSED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ SOME PORTFOLIO REBALANCER TESTS FAILED")
        print("="*70)

    sys.exit(exit_code)
