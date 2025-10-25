"""
Unit tests for transaction cost optimization module.
"""

import pytest
from crypto_trader.optimization.transaction_costs import (
    calculate_turnover,
    estimate_transaction_cost,
    should_rebalance,
)


class TestCalculateTurnover:
    """Tests for calculate_turnover function."""

    def test_basic_turnover(self):
        """Test basic turnover calculation."""
        current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        target = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        turnover = calculate_turnover(current, target)
        assert abs(turnover - 0.2) < 0.001

    def test_zero_turnover(self):
        """Test zero turnover when weights are identical."""
        weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        turnover = calculate_turnover(weights, weights)
        assert turnover == 0.0

    def test_complete_rebalance(self):
        """Test complete portfolio rebalance."""
        current = {"BTC/USDT": 1.0}
        target = {"ETH/USDT": 1.0}
        turnover = calculate_turnover(current, target)
        assert abs(turnover - 2.0) < 0.001  # 100% out + 100% in

    def test_new_asset_addition(self):
        """Test turnover when adding new asset."""
        current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        target = {"BTC/USDT": 0.33, "ETH/USDT": 0.33, "BNB/USDT": 0.34}
        turnover = calculate_turnover(current, target)
        # BTC: -0.17, ETH: -0.17, BNB: +0.34 = 0.68 total
        assert abs(turnover - 0.68) < 0.001

    def test_asset_removal(self):
        """Test turnover when removing asset."""
        current = {"BTC/USDT": 0.33, "ETH/USDT": 0.33, "BNB/USDT": 0.34}
        target = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        turnover = calculate_turnover(current, target)
        # BTC: +0.17, ETH: +0.17, BNB: -0.34 = 0.68 total
        assert abs(turnover - 0.68) < 0.001

    def test_empty_current_weights(self):
        """Test turnover with empty current weights (initial allocation)."""
        current = {}
        target = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        turnover = calculate_turnover(current, target)
        assert abs(turnover - 1.0) < 0.001  # 100% allocation


class TestEstimateTransactionCost:
    """Tests for estimate_transaction_cost function."""

    def test_basic_cost_estimation(self):
        """Test basic transaction cost estimation."""
        current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        target = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
        cost = estimate_transaction_cost(current, target, transaction_cost_pct=0.001)
        # Turnover = 0.4, cost = (0.4/2) * 0.001 = 0.0002
        assert abs(cost - 0.0002) < 0.000001

    def test_zero_cost_for_no_rebalance(self):
        """Test zero cost when no rebalancing needed."""
        weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        cost = estimate_transaction_cost(weights, weights, transaction_cost_pct=0.001)
        assert cost == 0.0

    def test_cost_scales_with_transaction_fee(self):
        """Test that cost scales linearly with transaction fee."""
        current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        target = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}

        cost_1bps = estimate_transaction_cost(current, target, transaction_cost_pct=0.0001)
        cost_10bps = estimate_transaction_cost(current, target, transaction_cost_pct=0.001)

        assert abs(cost_10bps - cost_1bps * 10) < 0.000001

    def test_high_turnover_high_cost(self):
        """Test that high turnover leads to high transaction costs."""
        current = {"BTC/USDT": 1.0}
        target = {"ETH/USDT": 1.0}
        cost = estimate_transaction_cost(current, target, transaction_cost_pct=0.001)
        # Turnover = 2.0, cost = (2.0/2) * 0.001 = 0.001
        assert abs(cost - 0.001) < 0.000001


class TestShouldRebalance:
    """Tests for should_rebalance function."""

    def test_tiny_rebalance_skipped(self):
        """Test that tiny rebalances are skipped."""
        current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        target = {"BTC/USDT": 0.501, "ETH/USDT": 0.499}
        should_rebal, turnover = should_rebalance(
            current, target,
            transaction_cost_pct=0.001,
            min_benefit_pct=0.005
        )
        assert not should_rebal
        assert turnover < 0.005

    def test_large_rebalance_executed(self):
        """Test that large rebalances are executed."""
        current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        target = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
        should_rebal, turnover = should_rebalance(
            current, target,
            transaction_cost_pct=0.001,
            min_benefit_pct=0.005
        )
        assert should_rebal
        assert turnover > 0.005

    def test_threshold_boundary(self):
        """Test behavior at threshold boundary."""
        current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        # Exactly at threshold
        target = {"BTC/USDT": 0.5025, "ETH/USDT": 0.4975}
        should_rebal, turnover = should_rebalance(
            current, target,
            transaction_cost_pct=0.001,
            min_benefit_pct=0.005
        )
        # At exactly 0.005 turnover, should rebalance (>=)
        assert should_rebal
        assert abs(turnover - 0.005) < 0.0001

    def test_returns_turnover_value(self):
        """Test that function returns turnover value."""
        current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        target = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        should_rebal, turnover = should_rebalance(current, target)
        assert isinstance(turnover, float)
        assert abs(turnover - 0.2) < 0.001

    def test_custom_thresholds(self):
        """Test with custom cost and benefit thresholds."""
        current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        target = {"BTC/USDT": 0.55, "ETH/USDT": 0.45}

        # With low threshold, should rebalance
        should_rebal_low, _ = should_rebalance(
            current, target,
            transaction_cost_pct=0.001,
            min_benefit_pct=0.001
        )
        assert should_rebal_low

        # With high threshold, should skip
        should_rebal_high, _ = should_rebalance(
            current, target,
            transaction_cost_pct=0.001,
            min_benefit_pct=0.20
        )
        assert not should_rebal_high

    def test_identical_weights_no_rebalance(self):
        """Test that identical weights result in no rebalance."""
        weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        should_rebal, turnover = should_rebalance(weights, weights)
        assert not should_rebal
        assert turnover == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
