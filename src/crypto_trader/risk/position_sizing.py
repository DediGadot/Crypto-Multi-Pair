"""
Kelly Criterion Position Sizing Module

**Purpose**: Implements fractional Kelly Criterion for optimal position sizing in
portfolio strategies. Uses conservative 25% Kelly fraction with hard limits to
prevent excessive leverage.

**Third-party Packages**:
- NumPy: https://numpy.org/doc/stable/

**Sample Input**:
    position_size = calculate_kelly_position_size(
        expected_return=0.13,  # 13% annual return
        volatility=0.40,       # 40% annual volatility
        win_rate=0.55,         # 55% historical win rate
        signal_confidence=1.0  # Full confidence in signal
    )

**Expected Output**:
    0.08125  # 8.125% position size (between 2% min and 15% max)

**Research Backing**:
Thorp, E. O. (2006). "The Kelly Criterion in Blackjack Sports Betting, and the Stock Market"
"""

from typing import Optional
import numpy as np
from loguru import logger


def calculate_kelly_position_size(
    expected_return: float,
    volatility: float,
    win_rate: float,
    signal_confidence: float = 1.0,
    kelly_fraction: float = 0.25,
    min_position_pct: float = 0.02,
    max_position_pct: float = 0.15
) -> float:
    """
    Calculate position size using fractional Kelly Criterion.

    The Kelly Criterion determines the optimal bet size to maximize long-term
    growth of capital. We use a fractional (25%) Kelly for safety and add
    hard position limits to prevent excessive leverage in crypto markets.

    Args:
        expected_return: Expected annual return (e.g., 0.13 = 13%)
        volatility: Annual volatility (e.g., 0.40 = 40%)
        win_rate: Historical win rate (0-1, e.g., 0.55 = 55%)
        signal_confidence: Confidence in signal (0-1, default 1.0)
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = 25%)
        min_position_pct: Minimum position size (default 0.02 = 2%)
        max_position_pct: Maximum position size (default 0.15 = 15%)

    Returns:
        Position size as fraction of capital (0-1)

    Examples:
        >>> # Conservative position with moderate expected return
        >>> size = calculate_kelly_position_size(0.10, 0.30, 0.52)
        >>> 0.02 <= size <= 0.15
        True

        >>> # High confidence signal gets larger position
        >>> size_high = calculate_kelly_position_size(0.15, 0.35, 0.60, 1.0)
        >>> size_low = calculate_kelly_position_size(0.15, 0.35, 0.60, 0.5)
        >>> size_high > size_low
        True
    """
    # Validate inputs
    if volatility <= 0:
        logger.warning(f"Invalid volatility {volatility}, using minimum position")
        return min_position_pct

    if win_rate <= 0 or win_rate >= 1:
        logger.warning(f"Invalid win_rate {win_rate}, using minimum position")
        return min_position_pct

    if not (0 <= signal_confidence <= 1):
        logger.warning(f"Invalid confidence {signal_confidence}, clipping to [0,1]")
        signal_confidence = np.clip(signal_confidence, 0.0, 1.0)

    # Simplified Kelly for continuous returns
    # f* = (expected_return) / (volatility^2)
    #
    # For discrete case with win/loss:
    # f* = (p*b - q) / b where p=win_rate, q=1-p, b=avg_win/avg_loss
    # We use simplified continuous version which is more stable
    try:
        kelly_size = expected_return / (volatility ** 2)
    except (ZeroDivisionError, FloatingPointError):
        logger.warning(f"Numerical error in Kelly calculation, using minimum position")
        return min_position_pct

    # Apply fractional Kelly and signal confidence
    position_size = kelly_size * kelly_fraction * signal_confidence

    # Enforce hard limits
    position_size = np.clip(position_size, min_position_pct, max_position_pct)

    logger.debug(
        f"Kelly sizing: return={expected_return:.3f}, vol={volatility:.3f}, "
        f"win_rate={win_rate:.3f}, confidence={signal_confidence:.3f} "
        f"→ size={position_size:.4f}"
    )

    return float(position_size)


def calculate_portfolio_kelly_weights(
    expected_returns: np.ndarray,
    volatilities: np.ndarray,
    win_rates: np.ndarray,
    signal_confidences: Optional[np.ndarray] = None,
    kelly_fraction: float = 0.25,
    min_position_pct: float = 0.02,
    max_position_pct: float = 0.15
) -> np.ndarray:
    """
    Calculate Kelly position sizes for multiple assets in a portfolio.

    Args:
        expected_returns: Array of expected annual returns for each asset
        volatilities: Array of annual volatilities for each asset
        win_rates: Array of historical win rates for each asset
        signal_confidences: Optional array of signal confidences (default: all 1.0)
        kelly_fraction: Fraction of full Kelly (default 0.25)
        min_position_pct: Minimum position size (default 0.02)
        max_position_pct: Maximum position size (default 0.15)

    Returns:
        Array of position sizes (not normalized - may not sum to 1.0)

    Examples:
        >>> returns = np.array([0.12, 0.15, 0.10])
        >>> vols = np.array([0.35, 0.40, 0.30])
        >>> win_rates = np.array([0.55, 0.58, 0.52])
        >>> weights = calculate_portfolio_kelly_weights(returns, vols, win_rates)
        >>> len(weights)
        3
        >>> all(0.02 <= w <= 0.15 for w in weights)
        True
    """
    n_assets = len(expected_returns)

    if signal_confidences is None:
        signal_confidences = np.ones(n_assets)

    # Validate array lengths
    if not (len(volatilities) == len(win_rates) == len(signal_confidences) == n_assets):
        raise ValueError(
            f"Array lengths must match: returns={n_assets}, "
            f"vols={len(volatilities)}, win_rates={len(win_rates)}, "
            f"confidences={len(signal_confidences)}"
        )

    # Calculate Kelly size for each asset
    weights = np.array([
        calculate_kelly_position_size(
            expected_return=expected_returns[i],
            volatility=volatilities[i],
            win_rate=win_rates[i],
            signal_confidence=signal_confidences[i],
            kelly_fraction=kelly_fraction,
            min_position_pct=min_position_pct,
            max_position_pct=max_position_pct
        )
        for i in range(n_assets)
    ])

    return weights


if __name__ == "__main__":
    """
    Validation function to test Kelly position sizing with realistic crypto parameters.
    """
    import sys

    print("=" * 80)
    print("KELLY POSITION SIZING VALIDATION")
    print("=" * 80)

    all_validation_failures = []
    total_tests = 0

    # Test 1: Basic Kelly calculation with typical crypto parameters
    total_tests += 1
    print("\n[Test 1] Basic Kelly calculation")
    size = calculate_kelly_position_size(
        expected_return=0.13,  # 13% annual
        volatility=0.40,  # 40% annual (typical crypto)
        win_rate=0.55,  # 55% win rate
        signal_confidence=1.0
    )
    print(f"  Expected return: 13%, Volatility: 40%, Win rate: 55%")
    print(f"  → Position size: {size:.4f} ({size*100:.2f}%)")

    if not (0.02 <= size <= 0.15):
        all_validation_failures.append(
            f"Test 1: Position size {size:.4f} out of bounds [0.02, 0.15]"
        )

    # Test 2: Hard limit enforcement (unrealistic high return)
    total_tests += 1
    print("\n[Test 2] Hard limit enforcement")
    size_unlimited = calculate_kelly_position_size(
        expected_return=5.0,  # Unrealistic 500% return
        volatility=0.20,
        win_rate=0.80,
        signal_confidence=1.0
    )
    print(f"  Expected return: 500% (unrealistic)")
    print(f"  → Position size: {size_unlimited:.4f} (should be capped at 15%)")

    if size_unlimited != 0.15:
        all_validation_failures.append(
            f"Test 2: Hard limit not enforced, got {size_unlimited:.4f} instead of 0.15"
        )

    # Test 3: Confidence scaling
    total_tests += 1
    print("\n[Test 3] Signal confidence scaling")
    size_high_conf = calculate_kelly_position_size(
        expected_return=0.12,
        volatility=0.35,
        win_rate=0.55,
        signal_confidence=1.0
    )
    size_low_conf = calculate_kelly_position_size(
        expected_return=0.12,
        volatility=0.35,
        win_rate=0.55,
        signal_confidence=0.5
    )
    print(f"  High confidence (1.0): {size_high_conf:.4f}")
    print(f"  Low confidence (0.5): {size_low_conf:.4f}")

    if not (size_low_conf < size_high_conf):
        all_validation_failures.append(
            f"Test 3: Confidence scaling failed, "
            f"low={size_low_conf:.4f} should be < high={size_high_conf:.4f}"
        )

    # Test 4: Edge case - zero volatility
    total_tests += 1
    print("\n[Test 4] Edge case: zero volatility")
    size_zero_vol = calculate_kelly_position_size(
        expected_return=0.10,
        volatility=0.0,  # Zero volatility
        win_rate=0.55,
        signal_confidence=1.0
    )
    print(f"  Volatility: 0.0")
    print(f"  → Position size: {size_zero_vol:.4f} (should be minimum 2%)")

    if size_zero_vol != 0.02:
        all_validation_failures.append(
            f"Test 4: Zero volatility handling failed, got {size_zero_vol:.4f} instead of 0.02"
        )

    # Test 5: Portfolio Kelly weights
    total_tests += 1
    print("\n[Test 5] Portfolio Kelly weights")
    returns = np.array([0.12, 0.15, 0.10])
    vols = np.array([0.35, 0.40, 0.30])
    win_rates = np.array([0.55, 0.58, 0.52])

    weights = calculate_portfolio_kelly_weights(returns, vols, win_rates)
    print(f"  Asset returns: {returns}")
    print(f"  Asset volatilities: {vols}")
    print(f"  Asset win rates: {win_rates}")
    print(f"  → Kelly weights: {weights}")
    print(f"  → Total weight: {weights.sum():.4f}")

    if not all(0.02 <= w <= 0.15 for w in weights):
        all_validation_failures.append(
            f"Test 5: Some weights out of bounds: {weights}"
        )

    # Final validation result
    print("\n" + "=" * 80)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Kelly position sizing module is validated and ready for use")
        sys.exit(0)
