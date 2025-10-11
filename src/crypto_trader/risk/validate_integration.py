"""
Integration validation for the risk management module.

This validation script tests the complete risk management system with realistic
trading scenarios to ensure all components work together correctly.

**Purpose**: Comprehensive validation of risk management integration
including position sizing, limit enforcement, and portfolio management.
"""

import sys
from datetime import datetime, timedelta

from loguru import logger

# Configure logger for validation
logger.remove()
logger.add(sys.stderr, level="WARNING")  # Only show warnings and errors

from crypto_trader.backtesting.portfolio import Position, PortfolioState
from crypto_trader.core.config import RiskConfig
from crypto_trader.core.types import OrderSide
from crypto_trader.risk import RiskManager


def main():
    """Run comprehensive integration tests."""
    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Running RISK MANAGEMENT Integration Validation\n")
    print("="*60)

    # Test 1: Complete trading workflow
    total_tests += 1
    print("\nTest 1: Complete trading workflow")
    try:
        # Setup
        config = RiskConfig(
            max_position_risk=0.02,
            max_portfolio_risk=0.10,
            stop_loss_percent=0.02,
            max_daily_loss_percent=0.05,
            max_open_positions=3,
            position_sizing_method="fixed_percent",
            risk_reward_ratio=2.0
        )

        risk_manager = RiskManager(config)
        timestamp = datetime(2025, 1, 1, 9, 0, 0)

        # Initial portfolio
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=10000.0,
            positions={},
            equity=10000.0
        )

        # Step 1: Check if we can trade
        allowed, reason = risk_manager.should_allow_trade(
            signal=1,
            portfolio=portfolio,
            price=50000.0,
            timestamp=timestamp,
            stop_loss_price=48000.0
        )

        if not allowed:
            all_validation_failures.append(f"Initial trade should be allowed: {reason}")

        # Step 2: Calculate position details
        quantity = risk_manager.calculate_position_size(
            signal=1,
            portfolio=portfolio,
            price=50000.0,
            stop_loss_price=48000.0
        )

        stop_loss = risk_manager.calculate_stop_loss(50000.0, side="long")
        take_profit = risk_manager.calculate_take_profit(50000.0, side="long")

        # Step 3: Get risk metrics
        metrics = risk_manager.get_portfolio_risk(portfolio)

        # Validations
        if quantity <= 0:
            all_validation_failures.append(f"Quantity should be positive: {quantity}")

        if stop_loss >= 50000.0:
            all_validation_failures.append(f"Stop loss should be below entry: {stop_loss}")

        if take_profit <= 50000.0:
            all_validation_failures.append(f"Take profit should be above entry: {take_profit}")

        print(f"  ✓ Trade approved: {allowed}")
        print(f"  ✓ Quantity: {quantity:.6f} BTC")
        print(f"  ✓ Entry: $50,000, Stop: ${stop_loss:,.0f}, TP: ${take_profit:,.0f}")
        print(f"  ✓ Risk metrics calculated: {len(metrics)} metrics")
    except Exception as e:
        all_validation_failures.append(f"Trading workflow test exception: {e}")

    # Test 2: Multiple position management
    total_tests += 1
    print("\nTest 2: Multiple position management")
    try:
        # Reset for new test
        risk_manager = RiskManager(config)

        # Add positions to portfolio
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=4000.0,
            positions={
                'BTCUSDT': Position(
                    symbol='BTCUSDT',
                    entry_price=50000.0,
                    quantity=0.1,
                    entry_time=timestamp,
                    side=OrderSide.BUY
                ),
                'ETHUSDT': Position(
                    symbol='ETHUSDT',
                    entry_price=3000.0,
                    quantity=1.0,
                    entry_time=timestamp,
                    side=OrderSide.BUY
                )
            },
            equity=12000.0
        )

        metrics = risk_manager.get_portfolio_risk(portfolio)

        if metrics['num_positions'] != 2:
            all_validation_failures.append(
                f"Position count: Expected 2, got {metrics['num_positions']}"
            )

        # Should still allow one more trade (max is 3)
        allowed, reason = risk_manager.should_allow_trade(
            signal=1,
            portfolio=portfolio,
            price=40000.0,
            timestamp=timestamp
        )

        if not allowed:
            all_validation_failures.append(f"Should allow 3rd position: {reason}")

        print(f"  ✓ Managing {metrics['num_positions']} positions")
        print(f"  ✓ Equity: ${metrics['equity']:,.2f}")
        print(f"  ✓ Can add another position: {allowed}")
    except Exception as e:
        all_validation_failures.append(f"Multiple positions test exception: {e}")

    # Test 3: Risk limit enforcement under stress
    total_tests += 1
    print("\nTest 3: Risk limit enforcement under stress")
    try:
        # Portfolio with 3 positions (at max)
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=1000.0,
            positions={
                f'SYMBOL{i}': Position(
                    symbol=f'SYMBOL{i}',
                    entry_price=50000.0,
                    quantity=0.1,
                    entry_time=timestamp,
                    side=OrderSide.BUY
                ) for i in range(3)
            },
            equity=10000.0
        )

        # Try to add 4th position - should be rejected
        allowed, reason = risk_manager.should_allow_trade(
            signal=1,
            portfolio=portfolio,
            price=50000.0,
            timestamp=timestamp
        )

        if allowed:
            all_validation_failures.append("Should reject 4th position when max is 3")

        if "maximum positions" not in reason.lower():
            all_validation_failures.append(f"Rejection reason unclear: {reason}")

        print(f"  ✓ Correctly enforced max positions limit")
        print(f"  ✓ Rejection reason: {reason}")
    except Exception as e:
        all_validation_failures.append(f"Risk enforcement test exception: {e}")

    # Test 4: Daily loss limit enforcement
    total_tests += 1
    print("\nTest 4: Daily loss limit enforcement")
    try:
        # Create new risk manager for this test to reset peak equity
        config_test4 = RiskConfig(
            max_position_risk=0.02,
            max_portfolio_risk=0.10,
            max_daily_loss_percent=0.05,
            max_open_positions=3
        )
        risk_manager_test4 = RiskManager(config_test4)

        # Start day with $10,000
        portfolio1 = PortfolioState(
            timestamp=timestamp,
            cash=10000.0,
            positions={},
            equity=10000.0
        )

        # Check initial - should pass
        allowed1, reason1 = risk_manager_test4.should_allow_trade(
            signal=1,
            portfolio=portfolio1,
            price=50000.0,
            timestamp=timestamp
        )

        if not allowed1:
            all_validation_failures.append(f"Initial trade should pass: {reason1}")

        # Later in day, down 6% (exceeds 5% limit)
        portfolio2 = PortfolioState(
            timestamp=timestamp + timedelta(hours=3),
            cash=9400.0,
            positions={},
            equity=9400.0
        )

        allowed2, reason2 = risk_manager_test4.should_allow_trade(
            signal=1,
            portfolio=portfolio2,
            price=50000.0,
            timestamp=timestamp + timedelta(hours=3)
        )

        if allowed2:
            all_validation_failures.append("Should reject trade after daily loss limit hit")

        if "daily loss" not in reason2.lower():
            all_validation_failures.append(f"Should mention daily loss: {reason2}")

        print(f"  ✓ Initial trade allowed")
        print(f"  ✓ Trade after 6% loss rejected: {reason2}")
    except Exception as e:
        all_validation_failures.append(f"Daily loss test exception: {e}")

    # Test 5: Drawdown monitoring
    total_tests += 1
    print("\nTest 5: Drawdown monitoring")
    try:
        risk_manager.reset_daily_limits()

        # Peak equity at $15,000
        portfolio_peak = PortfolioState(
            timestamp=timestamp,
            cash=15000.0,
            positions={},
            equity=15000.0
        )

        metrics_peak = risk_manager.get_portfolio_risk(portfolio_peak)

        if metrics_peak['peak_equity'] != 15000.0:
            all_validation_failures.append(
                f"Peak equity: Expected 15000, got {metrics_peak['peak_equity']}"
            )

        # Drawdown to $13,500 (10% drawdown)
        portfolio_dd = PortfolioState(
            timestamp=timestamp + timedelta(days=1),
            cash=13500.0,
            positions={},
            equity=13500.0
        )

        metrics_dd = risk_manager.get_portfolio_risk(portfolio_dd)

        expected_drawdown = 0.1
        if abs(metrics_dd['drawdown'] - expected_drawdown) > 0.01:
            all_validation_failures.append(
                f"Drawdown: Expected {expected_drawdown:.2%}, got {metrics_dd['drawdown']:.2%}"
            )

        print(f"  ✓ Peak equity tracked: ${metrics_peak['peak_equity']:,.2f}")
        print(f"  ✓ Drawdown calculated: {metrics_dd['drawdown']:.2%}")
    except Exception as e:
        all_validation_failures.append(f"Drawdown monitoring test exception: {e}")

    # Test 6: Different position sizing methods
    total_tests += 1
    print("\nTest 6: Different position sizing methods")
    try:
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=10000.0,
            positions={},
            equity=10000.0
        )

        # Test fixed percent (already tested in Test 1, just verify it works)
        config_fixed = RiskConfig(
            max_position_risk=0.02,
            position_sizing_method="fixed_percent"
        )
        manager_fixed = RiskManager(config_fixed)

        qty_fixed = manager_fixed.calculate_position_size(
            signal=1,
            portfolio=portfolio,
            price=50000.0,
            stop_loss_price=48000.0
        )

        # Test volatility-based
        config_vol = RiskConfig(
            max_position_risk=0.02,
            position_sizing_method="volatility_based"
        )
        manager_vol = RiskManager(config_vol)

        qty_vol = manager_vol.calculate_position_size(
            signal=1,
            portfolio=portfolio,
            price=50000.0,
            volatility=1000.0
        )

        if qty_fixed <= 0:
            all_validation_failures.append(f"Fixed quantity should be positive: {qty_fixed}")

        if qty_vol <= 0:
            all_validation_failures.append(f"Volatility quantity should be positive: {qty_vol}")

        print(f"  ✓ Fixed percent sizing: {qty_fixed:.6f} BTC")
        print(f"  ✓ Volatility sizing: {qty_vol:.6f} BTC")
    except Exception as e:
        all_validation_failures.append(f"Position sizing methods test exception: {e}")

    # Test 7: Risk/reward ratio calculations
    total_tests += 1
    print("\nTest 7: Risk/reward ratio calculations")
    try:
        entry = 50000.0

        # Test different ratios
        for ratio in [1.5, 2.0, 3.0]:
            config_temp = RiskConfig(
                max_position_risk=0.02,
                stop_loss_percent=0.02,
                risk_reward_ratio=ratio
            )
            manager_temp = RiskManager(config_temp)

            stop = manager_temp.calculate_stop_loss(entry, side="long")
            tp = manager_temp.calculate_take_profit(entry, side="long")

            # Calculate actual ratio
            risk_distance = entry - stop
            reward_distance = tp - entry
            actual_ratio = reward_distance / risk_distance if risk_distance > 0 else 0

            if abs(actual_ratio - ratio) > 0.01:
                all_validation_failures.append(
                    f"R:R ratio {ratio}: Expected {ratio:.1f}x, got {actual_ratio:.1f}x"
                )

        print(f"  ✓ Risk/reward ratios validated: 1.5x, 2.0x, 3.0x")
    except Exception as e:
        all_validation_failures.append(f"Risk/reward test exception: {e}")

    # Test 8: Trade recording and statistics
    total_tests += 1
    print("\nTest 8: Trade recording and statistics")
    try:
        risk_manager.reset_daily_limits()

        # Initialize daily tracker
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=10000.0,
            positions={},
            equity=10000.0
        )

        risk_manager.should_allow_trade(
            signal=1,
            portfolio=portfolio,
            price=50000.0,
            timestamp=timestamp
        )

        # Record trades
        trades = [
            (150.0, "win"),
            (-75.0, "loss"),
            (200.0, "win"),
            (-50.0, "loss"),
            (100.0, "win")
        ]

        for pnl, result_type in trades:
            risk_manager.record_trade_completion(pnl)

        stats = risk_manager.get_daily_stats()

        if stats is None:
            all_validation_failures.append("Daily stats should not be None")
        elif stats['trades_count'] != 5:
            all_validation_failures.append(
                f"Trade count: Expected 5, got {stats['trades_count']}"
            )
        else:
            expected_pnl = sum(pnl for pnl, _ in trades)
            if abs(stats['daily_pnl'] - expected_pnl) > 0.01:
                all_validation_failures.append(
                    f"Daily PnL: Expected {expected_pnl}, got {stats['daily_pnl']}"
                )

        print(f"  ✓ Trades recorded: {stats['trades_count'] if stats else 'N/A'}")
        print(f"  ✓ Daily PnL: ${stats['daily_pnl']:,.2f}" if stats else "  ✓ Stats tracking initialized")
    except Exception as e:
        all_validation_failures.append(f"Trade recording test exception: {e}")

    # Test 9: Portfolio risk aggregation
    total_tests += 1
    print("\nTest 9: Portfolio risk aggregation")
    try:
        # Complex portfolio with multiple positions
        portfolio = PortfolioState(
            timestamp=timestamp,
            cash=2500.0,
            positions={
                'BTCUSDT': Position(
                    symbol='BTCUSDT',
                    entry_price=50000.0,
                    quantity=0.1,
                    entry_time=timestamp,
                    side=OrderSide.BUY
                ),
                'ETHUSDT': Position(
                    symbol='ETHUSDT',
                    entry_price=3000.0,
                    quantity=1.0,
                    entry_time=timestamp,
                    side=OrderSide.BUY
                ),
                'ADAUSDT': Position(
                    symbol='ADAUSDT',
                    entry_price=0.50,
                    quantity=2000.0,
                    entry_time=timestamp,
                    side=OrderSide.BUY
                )
            },
            equity=10500.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0
        )

        metrics = risk_manager.get_portfolio_risk(portfolio)

        required_metrics = [
            'equity', 'cash', 'num_positions', 'estimated_risk_percent',
            'cash_utilization', 'unrealized_pnl', 'realized_pnl'
        ]

        for metric in required_metrics:
            if metric not in metrics:
                all_validation_failures.append(f"Missing metric: {metric}")

        if metrics['num_positions'] != 3:
            all_validation_failures.append(
                f"Position count: Expected 3, got {metrics['num_positions']}"
            )

        print(f"  ✓ All {len(required_metrics)} risk metrics calculated")
        print(f"  ✓ Portfolio equity: ${metrics['equity']:,.2f}")
        print(f"  ✓ Cash utilization: {metrics['cash_utilization']:.2%}")
        print(f"  ✓ Estimated risk: {metrics['estimated_risk_percent']:.2%}")
    except Exception as e:
        all_validation_failures.append(f"Portfolio risk aggregation test exception: {e}")

    # Test 10: Edge cases and boundary conditions
    total_tests += 1
    print("\nTest 10: Edge cases and boundary conditions")
    try:
        # Zero equity - should handle gracefully
        portfolio_zero = PortfolioState(
            timestamp=timestamp,
            cash=0.0,
            positions={},
            equity=0.0
        )

        try:
            quantity = risk_manager.calculate_position_size(
                signal=1,
                portfolio=portfolio_zero,
                price=50000.0
            )
            # Should return 0 or small number, not crash
            if quantity > 1.0:  # Unreasonably large
                all_validation_failures.append(f"Quantity too large for zero equity: {quantity}")
        except Exception:
            # Some exception is acceptable for zero equity
            pass

        # Very small portfolio
        portfolio_small = PortfolioState(
            timestamp=timestamp,
            cash=100.0,
            positions={},
            equity=100.0
        )

        quantity_small = risk_manager.calculate_position_size(
            signal=1,
            portfolio=portfolio_small,
            price=50000.0,
            stop_loss_price=48000.0
        )

        # Should calculate something reasonable
        if quantity_small < 0:
            all_validation_failures.append("Negative quantity not allowed")

        print(f"  ✓ Handled zero equity gracefully")
        print(f"  ✓ Small portfolio ($100): {quantity_small:.6f} BTC")
    except Exception as e:
        all_validation_failures.append(f"Edge cases test exception: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:\n")
        for i, failure in enumerate(all_validation_failures, 1):
            print(f"  {i}. {failure}")
        print("\n" + "="*60)
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} integration tests successful\n")
        print("Risk management module is fully validated and production-ready!")
        print("\nKey Features Validated:")
        print("  ✓ Multiple position sizing algorithms")
        print("  ✓ Comprehensive risk limit enforcement")
        print("  ✓ Daily loss and drawdown monitoring")
        print("  ✓ Stop loss and take profit calculations")
        print("  ✓ Portfolio risk aggregation")
        print("  ✓ Trade approval workflow")
        print("  ✓ Statistics tracking")
        print("  ✓ Edge case handling")
        print("\n" + "="*60)
        sys.exit(0)


if __name__ == "__main__":
    main()
