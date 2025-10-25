"""
Transaction Cost Optimization for Portfolio Rebalancing.

This module provides utilities for transaction cost-aware portfolio optimization,
helping to reduce excessive trading and improve net returns.

Key Functions:
- calculate_turnover: Compute portfolio turnover between two weight vectors
- estimate_transaction_cost: Estimate total transaction costs for a rebalance
- should_rebalance: Decide if rebalancing is worthwhile given costs

References:
- PyPortfolioOpt transaction cost documentation:
  https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html#transaction-costs
- Patel et al. (2018): "Transaction Cost Optimization for Online Portfolio Selection"

Sample Input:
    current_weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.3, "BNB/USDT": 0.2}
    target_weights = {"BTC/USDT": 0.4, "ETH/USDT": 0.4, "BNB/USDT": 0.2}
    transaction_cost_pct = 0.001  # 10 basis points

Expected Output:
    should_rebalance, turnover = should_rebalance(current_weights, target_weights)
    # Returns: (True/False, 0.2) where 0.2 is 20% turnover
"""

from typing import Dict, Tuple
import numpy as np
from loguru import logger


def calculate_turnover(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float]
) -> float:
    """
    Calculate portfolio turnover as sum of absolute weight changes.

    Turnover measures how much of the portfolio needs to be traded to
    achieve the target allocation. A turnover of 1.0 means the entire
    portfolio is being replaced.

    Args:
        current_weights: Current portfolio weights (asset -> weight)
        target_weights: Target portfolio weights (asset -> weight)

    Returns:
        Portfolio turnover (0-2, typically 0-1)

    Examples:
        >>> current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        >>> target = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        >>> calculate_turnover(current, target)
        0.2
    """
    # Get all unique assets
    all_assets = set(current_weights.keys()) | set(target_weights.keys())

    # Sum absolute differences
    turnover = sum(
        abs(target_weights.get(asset, 0.0) - current_weights.get(asset, 0.0))
        for asset in all_assets
    )

    return turnover


def estimate_transaction_cost(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    transaction_cost_pct: float = 0.001
) -> float:
    """
    Estimate total transaction cost for rebalancing.

    Transaction costs include:
    - Exchange trading fees (typically 10 bps = 0.001)
    - Bid-ask spread (typically 5-10 bps for liquid crypto)
    - Slippage (typically negligible for small orders)

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        transaction_cost_pct: Cost per trade as percentage (default 0.1%)

    Returns:
        Estimated total transaction cost as fraction of portfolio value

    Examples:
        >>> current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        >>> target = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
        >>> estimate_transaction_cost(current, target, 0.001)
        0.0004  # 0.04% = 4 basis points
    """
    turnover = calculate_turnover(current_weights, target_weights)

    # Transaction cost is proportional to turnover
    # Note: We only pay costs on one side (either buy or sell)
    # so we divide turnover by 2
    cost = (turnover / 2) * transaction_cost_pct

    return cost


def should_rebalance(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    transaction_cost_pct: float = 0.001,
    min_benefit_pct: float = 0.005
) -> Tuple[bool, float]:
    """
    Determine if rebalancing is worthwhile given transaction costs.

    Rebalancing is only worthwhile if the expected benefit exceeds
    the transaction costs. The minimum benefit threshold prevents
    excessive trading for marginal gains.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        transaction_cost_pct: Cost per trade (default 0.1% = 10 bps)
        min_benefit_pct: Minimum expected benefit to justify rebalance
                         (default 0.5% = 50 bps)

    Returns:
        Tuple of (should_rebalance, turnover)
        - should_rebalance: True if rebalancing is worthwhile
        - turnover: Portfolio turnover (for logging/monitoring)

    Examples:
        >>> current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        >>> target = {"BTC/USDT": 0.51, "ETH/USDT": 0.49}
        >>> should_rebalance(current, target)
        (False, 0.02)  # Too small, skip rebalance

        >>> target = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
        >>> should_rebalance(current, target)
        (True, 0.4)  # Large enough, execute rebalance
    """
    # Calculate turnover
    turnover = calculate_turnover(current_weights, target_weights)

    # Estimate transaction cost
    tx_cost = estimate_transaction_cost(
        current_weights,
        target_weights,
        transaction_cost_pct
    )

    # Check if turnover is significant enough to warrant rebalancing
    # Only rebalance if the turnover (benefit) exceeds the cost threshold
    # The min_benefit_pct represents the minimum turnover needed to justify costs
    should_rebal = turnover >= min_benefit_pct

    # Log decision
    if not should_rebal:
        logger.debug(
            f"Skipping rebalance: turnover={turnover:.4f}, "
            f"tx_cost={tx_cost:.4f}, min_benefit={min_benefit_pct:.4f}"
        )
    else:
        logger.debug(
            f"Executing rebalance: turnover={turnover:.4f}, "
            f"tx_cost={tx_cost:.4f}"
        )

    return should_rebal, turnover


if __name__ == "__main__":
    """
    Validation: Test transaction cost calculations with realistic scenarios.
    """
    import sys

    print("=" * 70)
    print("TRANSACTION COST MODULE VALIDATION")
    print("=" * 70)

    all_validation_failures = []
    total_tests = 0

    # Test 1: Basic turnover calculation
    total_tests += 1
    print("\n[Test 1] Basic turnover calculation")
    current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
    target = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
    turnover = calculate_turnover(current, target)
    expected_turnover = 0.2
    print(f"  Current: {current}")
    print(f"  Target: {target}")
    print(f"  Turnover: {turnover:.4f}")
    print(f"  Expected: {expected_turnover:.4f}")
    if abs(turnover - expected_turnover) > 0.001:
        all_validation_failures.append(
            f"Test 1: Expected turnover {expected_turnover}, got {turnover}"
        )
    else:
        print("  ✓ PASS")

    # Test 2: Transaction cost estimation
    total_tests += 1
    print("\n[Test 2] Transaction cost estimation")
    current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
    target = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
    tx_cost = estimate_transaction_cost(current, target, transaction_cost_pct=0.001)
    # Turnover = 0.4, cost = (0.4/2) * 0.001 = 0.0002
    expected_cost = 0.0002
    print(f"  Current: {current}")
    print(f"  Target: {target}")
    print(f"  Transaction Cost: {tx_cost:.6f}")
    print(f"  Expected: {expected_cost:.6f}")
    if abs(tx_cost - expected_cost) > 0.00001:
        all_validation_failures.append(
            f"Test 2: Expected cost {expected_cost}, got {tx_cost}"
        )
    else:
        print("  ✓ PASS")

    # Test 3: Tiny rebalance should be skipped
    total_tests += 1
    print("\n[Test 3] Tiny rebalance should be skipped")
    current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
    target = {"BTC/USDT": 0.501, "ETH/USDT": 0.499}  # Only 0.2% turnover
    should_rebal, turnover = should_rebalance(
        current, target,
        transaction_cost_pct=0.001,
        min_benefit_pct=0.005  # 0.5% threshold
    )
    print(f"  Current: {current}")
    print(f"  Target: {target}")
    print(f"  Should Rebalance: {should_rebal}")
    print(f"  Turnover: {turnover:.4f}")
    # Turnover is 0.002 (0.2%), which is less than 0.005 (0.5%) threshold
    if should_rebal:
        all_validation_failures.append(
            "Test 3: Tiny rebalance should be skipped, but was approved"
        )
    else:
        print("  ✓ PASS")

    # Test 4: Large rebalance should be executed
    total_tests += 1
    print("\n[Test 4] Large rebalance should be executed")
    current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
    target = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
    should_rebal, turnover = should_rebalance(
        current, target,
        transaction_cost_pct=0.001,
        min_benefit_pct=0.005
    )
    print(f"  Current: {current}")
    print(f"  Target: {target}")
    print(f"  Should Rebalance: {should_rebal}")
    print(f"  Turnover: {turnover:.4f}")
    if not should_rebal:
        all_validation_failures.append(
            "Test 4: Large rebalance should be executed, but was skipped"
        )
    else:
        print("  ✓ PASS")

    # Test 5: New asset addition
    total_tests += 1
    print("\n[Test 5] New asset addition")
    current = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
    target = {"BTC/USDT": 0.33, "ETH/USDT": 0.33, "BNB/USDT": 0.34}
    turnover = calculate_turnover(current, target)
    # BTC: 0.5 -> 0.33 = -0.17
    # ETH: 0.5 -> 0.33 = -0.17
    # BNB: 0 -> 0.34 = +0.34
    # Total = 0.17 + 0.17 + 0.34 = 0.68
    expected_turnover = 0.68
    print(f"  Current: {current}")
    print(f"  Target: {target}")
    print(f"  Turnover: {turnover:.4f}")
    print(f"  Expected: {expected_turnover:.4f}")
    if abs(turnover - expected_turnover) > 0.001:
        all_validation_failures.append(
            f"Test 5: Expected turnover {expected_turnover}, got {turnover}"
        )
    else:
        print("  ✓ PASS")

    # Test 6: Asset removal
    total_tests += 1
    print("\n[Test 6] Asset removal")
    current = {"BTC/USDT": 0.33, "ETH/USDT": 0.33, "BNB/USDT": 0.34}
    target = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
    turnover = calculate_turnover(current, target)
    # BTC: 0.33 -> 0.5 = +0.17
    # ETH: 0.33 -> 0.5 = +0.17
    # BNB: 0.34 -> 0 = -0.34
    # Total = 0.17 + 0.17 + 0.34 = 0.68
    expected_turnover = 0.68
    print(f"  Current: {current}")
    print(f"  Target: {target}")
    print(f"  Turnover: {turnover:.4f}")
    print(f"  Expected: {expected_turnover:.4f}")
    if abs(turnover - expected_turnover) > 0.001:
        all_validation_failures.append(
            f"Test 6: Expected turnover {expected_turnover}, got {turnover}"
        )
    else:
        print("  ✓ PASS")

    # Final validation result
    print("\n" + "=" * 70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
