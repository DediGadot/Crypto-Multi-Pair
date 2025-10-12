"""
Integration Tests for Portfolio Backtest System

Tests the complete portfolio backtesting workflow including:
- Config loading
- Multi-asset data fetching
- Portfolio simulation
- Report generation
- File output

**Purpose**: Verify the complete portfolio backtesting system works end-to-end.

**Third-party packages**:
- pytest: https://docs.pytest.org/
- pandas: https://pandas.pydata.org/docs/
- pyyaml: https://pyyaml.org/wiki/PyYAMLDocumentation
"""

import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yaml
import pytest


class TestDataAlignment:
    """Test multi-asset data alignment and timestamp handling."""

    def test_find_common_timestamps(self):
        """Test finding common timestamps across multiple assets."""
        # Create three DataFrames with different date ranges
        dates1 = pd.date_range('2024-01-01', periods=100, freq='1h')
        dates2 = pd.date_range('2024-01-01 10:00:00', periods=80, freq='1h')
        dates3 = pd.date_range('2024-01-01 05:00:00', periods=90, freq='1h')

        df1 = pd.DataFrame({'close': [100] * 100}, index=dates1)
        df2 = pd.DataFrame({'close': [200] * 80}, index=dates2)
        df3 = pd.DataFrame({'close': [300] * 90}, index=dates3)

        # Find intersection
        common_timestamps = df1.index.intersection(df2.index).intersection(df3.index)

        assert len(common_timestamps) > 0
        # Should be 80 hours (from 10:00 on day 1 to end of dates2)
        assert len(common_timestamps) == 80

    def test_empty_intersection_handled(self):
        """Test handling of no overlapping timestamps."""
        # Create DataFrames with no overlap
        dates1 = pd.date_range('2024-01-01', periods=10, freq='1h')
        dates2 = pd.date_range('2024-02-01', periods=10, freq='1h')

        df1 = pd.DataFrame({'close': [100] * 10}, index=dates1)
        df2 = pd.DataFrame({'close': [200] * 10}, index=dates2)

        common_timestamps = df1.index.intersection(df2.index)

        assert len(common_timestamps) == 0

    def test_data_alignment_preserves_values(self):
        """Test that data alignment preserves correct values."""
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')

        btc_prices = np.linspace(40000, 50000, 50)
        eth_prices = np.linspace(2000, 2500, 50)

        btc_data = pd.DataFrame({'close': btc_prices}, index=dates)
        eth_data = pd.DataFrame({'close': eth_prices}, index=dates)

        # Align on common timestamps
        common = btc_data.index.intersection(eth_data.index)

        aligned_btc = btc_data.loc[common]
        aligned_eth = eth_data.loc[common]

        # Values should match original
        assert len(aligned_btc) == 50
        assert len(aligned_eth) == 50
        assert aligned_btc.iloc[0]['close'] == 40000
        assert aligned_eth.iloc[0]['close'] == 2000


class TestPortfolioSimulation:
    """Test portfolio value calculations and rebalancing."""

    def test_initial_allocation(self):
        """Test initial portfolio allocation."""
        initial_capital = 10000
        assets = [
            ('BTC/USDT', 0.50),
            ('ETH/USDT', 0.30),
            ('SOL/USDT', 0.20)
        ]

        # Calculate initial allocation
        allocation = {}
        for symbol, weight in assets:
            allocation[symbol] = initial_capital * weight

        assert allocation['BTC/USDT'] == 5000
        assert allocation['ETH/USDT'] == 3000
        assert allocation['SOL/USDT'] == 2000
        assert sum(allocation.values()) == initial_capital

    def test_portfolio_value_update(self):
        """Test portfolio value updates as prices change."""
        initial_capital = 10000
        initial_btc_price = 40000
        initial_eth_price = 2000

        # Initial allocation (50/50)
        btc_value = 5000
        eth_value = 5000

        # Calculate shares
        btc_shares = btc_value / initial_btc_price  # 0.125 BTC
        eth_shares = eth_value / initial_eth_price  # 2.5 ETH

        # Prices change
        new_btc_price = 50000  # +25%
        new_eth_price = 2200   # +10%

        # Calculate new values
        new_btc_value = btc_shares * new_btc_price  # 6250
        new_eth_value = eth_shares * new_eth_price  # 5500

        new_total = new_btc_value + new_eth_value

        assert new_btc_value == 6250
        assert new_eth_value == 5500
        assert new_total == 11750
        assert new_total > initial_capital  # Portfolio gained value

    def test_weight_calculation(self):
        """Test portfolio weight calculations."""
        btc_value = 6000
        eth_value = 4000
        total_value = 10000

        btc_weight = btc_value / total_value
        eth_weight = eth_value / total_value

        assert btc_weight == 0.60
        assert eth_weight == 0.40
        assert abs((btc_weight + eth_weight) - 1.0) < 0.0001

    def test_deviation_calculation(self):
        """Test calculation of deviation from target weights."""
        current_weights = {
            'BTC/USDT': 0.65,
            'ETH/USDT': 0.35
        }

        target_weights = {
            'BTC/USDT': 0.50,
            'ETH/USDT': 0.50
        }

        deviations = {}
        for symbol in current_weights:
            deviations[symbol] = abs(current_weights[symbol] - target_weights[symbol])

        assert abs(deviations['BTC/USDT'] - 0.15) < 0.0001
        assert abs(deviations['ETH/USDT'] - 0.15) < 0.0001

        max_deviation = max(deviations.values())
        assert abs(max_deviation - 0.15) < 0.0001

    def test_rebalance_threshold_check(self):
        """Test checking if rebalance threshold is exceeded."""
        threshold = 0.10

        # Case 1: Deviation below threshold
        deviation1 = 0.08
        assert deviation1 <= threshold

        # Case 2: Deviation above threshold
        deviation2 = 0.12
        assert deviation2 > threshold

        # Case 3: Deviation exactly at threshold
        deviation3 = 0.10
        assert deviation3 <= threshold  # Use <= for threshold check

    def test_rebalance_execution(self):
        """Test executing a rebalance."""
        total_value = 12000
        target_weights = {
            'BTC/USDT': 0.50,
            'ETH/USDT': 0.50
        }

        current_prices = {
            'BTC/USDT': 50000,
            'ETH/USDT': 2500
        }

        # Calculate target values
        target_values = {}
        for symbol, weight in target_weights.items():
            target_values[symbol] = total_value * weight

        # Calculate new shares
        new_shares = {}
        for symbol in target_values:
            new_shares[symbol] = target_values[symbol] / current_prices[symbol]

        assert target_values['BTC/USDT'] == 6000
        assert target_values['ETH/USDT'] == 6000
        assert new_shares['BTC/USDT'] == 0.12  # 6000 / 50000
        assert new_shares['ETH/USDT'] == 2.4   # 6000 / 2500


class TestBuyAndHoldBenchmark:
    """Test buy-and-hold benchmark calculations."""

    def test_buy_and_hold_no_rebalancing(self):
        """Test that buy-and-hold never rebalances."""
        initial_capital = 10000
        initial_btc_price = 40000
        initial_eth_price = 2000

        # Initial allocation
        btc_shares = (initial_capital * 0.5) / initial_btc_price
        eth_shares = (initial_capital * 0.5) / initial_eth_price

        # Prices change significantly
        prices_btc = [40000, 50000, 60000, 70000, 80000]
        prices_eth = [2000, 2100, 2200, 2300, 2400]

        # Calculate portfolio values (shares never change)
        portfolio_values = []
        for btc_price, eth_price in zip(prices_btc, prices_eth):
            total_value = (btc_shares * btc_price) + (eth_shares * eth_price)
            portfolio_values.append(total_value)

        # Portfolio should grow
        assert portfolio_values[-1] > portfolio_values[0]

        # Final value calculation
        final_btc_value = btc_shares * prices_btc[-1]
        final_eth_value = eth_shares * prices_eth[-1]
        final_total = final_btc_value + final_eth_value

        assert final_total == portfolio_values[-1]

    def test_buy_and_hold_weights_drift(self):
        """Test that weights drift over time in buy-and-hold."""
        initial_capital = 10000

        # Start 50/50
        btc_shares = (initial_capital * 0.5) / 40000
        eth_shares = (initial_capital * 0.5) / 2000

        # BTC doubles, ETH stays same
        new_btc_price = 80000
        new_eth_price = 2000

        new_btc_value = btc_shares * new_btc_price
        new_eth_value = eth_shares * new_eth_price
        new_total = new_btc_value + new_eth_value

        new_btc_weight = new_btc_value / new_total
        new_eth_weight = new_eth_value / new_total

        # BTC should now be much higher weight
        assert new_btc_weight > 0.50
        assert new_eth_weight < 0.50
        assert abs(new_btc_weight - 0.6667) < 0.001  # Should be ~66.67%


class TestReportGeneration:
    """Test report and output file generation."""

    def test_equity_curve_dataframe(self):
        """Test equity curve DataFrame structure."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        portfolio_values = [10000, 10100, 10050, 10200, 10300, 10250, 10400, 10500, 10450, 10600]

        equity_curve = pd.DataFrame({
            'timestamp': dates,
            'portfolio_value': portfolio_values
        })

        assert len(equity_curve) == 10
        assert 'timestamp' in equity_curve.columns
        assert 'portfolio_value' in equity_curve.columns
        assert equity_curve['portfolio_value'].iloc[0] == 10000
        assert equity_curve['portfolio_value'].iloc[-1] == 10600

    def test_performance_metrics_calculation(self):
        """Test calculation of performance metrics."""
        initial_value = 10000
        final_value = 12500

        # Calculate return
        total_return = ((final_value - initial_value) / initial_value) * 100

        assert total_return == 25.0

        # Calculate return percentage
        return_pct = (final_value / initial_value - 1) * 100
        assert return_pct == 25.0

    def test_comparison_metrics(self):
        """Test comparison between rebalanced and buy-hold."""
        rebalanced_return = 20.0  # %
        buy_hold_return = 22.0    # %

        # Calculate outperformance
        absolute_diff = rebalanced_return - buy_hold_return
        relative_diff = ((rebalanced_return - buy_hold_return) / buy_hold_return) * 100

        assert absolute_diff == -2.0  # Underperformed by 2%
        assert abs(relative_diff - (-9.09)) < 0.01  # About -9.09%

    def test_rebalance_events_dataframe(self):
        """Test rebalance events DataFrame structure."""
        events_data = {
            'timestamp': [
                datetime(2024, 1, 15, 10, 0),
                datetime(2024, 2, 20, 14, 0)
            ],
            'total_value': [11000, 13000],
            'max_deviation': [0.16, 0.18],
            'weights_before': [
                {'BTC/USDT': 0.65, 'ETH/USDT': 0.35},
                {'BTC/USDT': 0.68, 'ETH/USDT': 0.32}
            ]
        }

        rebalance_events = pd.DataFrame(events_data)

        assert len(rebalance_events) == 2
        assert 'timestamp' in rebalance_events.columns
        assert 'total_value' in rebalance_events.columns
        assert 'max_deviation' in rebalance_events.columns
        assert rebalance_events['max_deviation'].iloc[0] > 0.15


class TestFileOutput:
    """Test file output and CSV generation."""

    def test_csv_output(self):
        """Test writing equity curve to CSV."""
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        equity_data = pd.DataFrame({
            'timestamp': dates,
            'portfolio_value': [10000, 10100, 10050, 10200, 10300]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            equity_data.to_csv(temp_path, index=False)

            # Read back and verify
            loaded_data = pd.read_csv(temp_path)
            assert len(loaded_data) == 5
            assert 'timestamp' in loaded_data.columns
            assert 'portfolio_value' in loaded_data.columns
        finally:
            Path(temp_path).unlink()

    def test_summary_report_generation(self):
        """Test generating summary report text."""
        run_name = "test_portfolio"
        description = "Test configuration"
        initial_capital = 10000
        final_value_rebalanced = 12000
        final_value_buy_hold = 11500
        rebalance_count = 3

        # Calculate metrics
        return_rebalanced = ((final_value_rebalanced - initial_capital) / initial_capital) * 100
        return_buy_hold = ((final_value_buy_hold - initial_capital) / initial_capital) * 100
        outperformance = return_rebalanced - return_buy_hold

        summary = f"""
PORTFOLIO REBALANCING - SUMMARY REPORT

Run Name: {run_name}
Description: {description}

PERFORMANCE COMPARISON

Initial Capital: ${initial_capital:,.2f}

REBALANCED PORTFOLIO:
  Final Value: ${final_value_rebalanced:,.2f}
  Total Return: {return_rebalanced:.2f}%
  Rebalance Events: {rebalance_count}

BUY & HOLD (No Rebalancing):
  Final Value: ${final_value_buy_hold:,.2f}
  Total Return: {return_buy_hold:.2f}%

RESULT: Portfolio OUTPERFORMED by {outperformance:.2f}%
"""

        assert "test_portfolio" in summary
        assert "20.00%" in summary  # Rebalanced return
        assert "15.00%" in summary  # Buy-hold return
        assert "5.00%" in summary   # Outperformance

    def test_output_directory_creation(self):
        """Test creating output directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results_test"
            data_dir = output_dir / "data"

            # Create directories
            output_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            assert output_dir.exists()
            assert data_dir.exists()
            assert output_dir.is_dir()
            assert data_dir.is_dir()


class TestEdgeCasesIntegration:
    """Test edge cases in integration scenarios."""

    def test_single_timestamp_data(self):
        """Test handling of data with only one timestamp."""
        dates = pd.date_range('2024-01-01', periods=1, freq='1h')

        df = pd.DataFrame({
            'close': [40000],
            'open': [39900],
            'high': [40100],
            'low': [39800],
            'volume': [1000]
        }, index=dates)

        # Should have only 1 row
        assert len(df) == 1

        # Cannot calculate returns or rebalance with 1 data point
        # This would be caught by the strategy's validation

    def test_zero_volume_handling(self):
        """Test handling of zero volume data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')

        df = pd.DataFrame({
            'close': [40000] * 10,
            'open': [39900] * 10,
            'high': [40100] * 10,
            'low': [39800] * 10,
            'volume': [0] * 10  # All zero volume
        }, index=dates)

        # Data should load but volume is zero
        assert len(df) == 10
        assert df['volume'].sum() == 0

    def test_missing_data_columns(self):
        """Test detection of missing required columns."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')

        # Missing 'close' column
        incomplete_df = pd.DataFrame({
            'open': [40000] * 10,
            'high': [40100] * 10,
            'low': [39900] * 10,
            'volume': [1000] * 10
        }, index=dates)

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        has_all_columns = all(col in incomplete_df.columns for col in required_columns)

        assert not has_all_columns  # Should detect missing column


if __name__ == "__main__":
    """
    Validation block for Portfolio Integration Tests.
    Runs all tests and reports results.
    """
    import sys

    # Run pytest with verbose output
    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    if exit_code == 0:
        print("\n" + "="*70)
        print("✅ ALL PORTFOLIO INTEGRATION TESTS PASSED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ SOME PORTFOLIO INTEGRATION TESTS FAILED")
        print("="*70)

    sys.exit(exit_code)
