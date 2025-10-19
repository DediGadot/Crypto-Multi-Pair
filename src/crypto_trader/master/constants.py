"""
Master Analysis Constants

Centralized configuration constants extracted from master.py.
All magic numbers and configuration values should be defined here.

**Purpose**: Single source of truth for all configuration values,
eliminating magic numbers throughout the codebase.

**Third-party packages**: None (pure Python constants)

**Sample Usage**:
```python
from crypto_trader.master.constants import (
    MAX_DATA_LOSS_PERCENT,
    BINANCE_MAX_CANDLES,
    DEFAULT_WARMUP_MULTIPLIER
)

if data_loss_pct > MAX_DATA_LOSS_PERCENT:
    logger.warning(f"Data loss: {data_loss_pct}%")
```
"""

from typing import Dict

# ============================================================================
# DATA FETCHING CONSTANTS
# ============================================================================

# Maximum candles per API request (Binance limit)
BINANCE_MAX_CANDLES: int = 1000

# Maximum number of batches to fetch (safety limit)
MAX_HISTORICAL_BATCHES: int = 100  # ~11 years for hourly data

# Default warmup multiplier for horizon calculations
DEFAULT_WARMUP_MULTIPLIER: float = 1.5

# Maximum data loss percentage before warning
MAX_DATA_LOSS_PERCENT: float = 5.0


# ============================================================================
# WORKER POOL CONSTANTS
# ============================================================================

# Maximum workers for single-pair mode
MAX_SINGLE_PAIR_WORKERS: int = 8

# Maximum workers for multi-pair mode (limited due to memory)
MAX_MULTI_PAIR_WORKERS: int = 4

# Default number of workers
DEFAULT_WORKERS: int = 4


# ============================================================================
# BACKTEST CONFIGURATION
# ============================================================================

# Default initial capital for backtests
DEFAULT_INITIAL_CAPITAL: float = 10000.0

# Default trading fee (0.1% = 0.001)
DEFAULT_TRADING_FEE: float = 0.001

# Default slippage (0.05% = 0.0005)
DEFAULT_SLIPPAGE: float = 0.0005

# Default maximum position size (95% of capital)
DEFAULT_MAX_POSITION_SIZE: float = 0.95


# ============================================================================
# STRATEGY SCORING CONSTANTS
# ============================================================================

# Minimum Sharpe ratio for "good" quality
MIN_SHARPE_GOOD: float = 1.0

# Minimum Sharpe ratio for "excellent" quality
MIN_SHARPE_EXCELLENT: float = 2.0

# Minimum number of trades for valid backtest
MIN_TRADES_THRESHOLD: int = 5

# Weight for return in composite score
SCORE_WEIGHT_RETURN: float = 0.30

# Weight for Sharpe ratio in composite score
SCORE_WEIGHT_SHARPE: float = 0.25

# Weight for win rate in composite score
SCORE_WEIGHT_WIN_RATE: float = 0.20

# Weight for profit factor in composite score
SCORE_WEIGHT_PROFIT_FACTOR: float = 0.15

# Weight for consistency (low drawdown) in composite score
SCORE_WEIGHT_CONSISTENCY: float = 0.10


# ============================================================================
# REGIME DETECTION CONSTANTS
# ============================================================================

# Minimum samples required for HMM regime fitting
MIN_REGIME_SAMPLES: int = 30

# Default z-score window for mean reversion
DEFAULT_Z_SCORE_WINDOW: int = 90

# Minimum data points for statistical arbitrage
MIN_STAT_ARB_SAMPLES: int = 100


# ============================================================================
# REPORTING CONSTANTS
# ============================================================================

# Maximum error message length in reports
MAX_ERROR_MESSAGE_LENGTH: int = 500

# Number of top strategies to highlight in reports
TOP_STRATEGIES_COUNT: int = 10

# Default HTML report width
HTML_REPORT_WIDTH: str = "1400px"


# ============================================================================
# TIMEFRAME MAPPINGS
# ============================================================================

# Map timeframe strings to periods per year (for annualization)
PERIODS_PER_YEAR: Dict[str, float] = {
    "1m": 525600.0,    # 365.25 * 24 * 60
    "5m": 105120.0,    # 365.25 * 24 * 12
    "15m": 35040.0,    # 365.25 * 24 * 4
    "1h": 8760.0,      # 365.25 * 24
    "4h": 2190.0,      # 365.25 * 6
    "1d": 365.25,      # Standard year
    "1w": 52.0,        # Weeks per year
}

# Map timeframe strings to milliseconds duration
TIMEFRAME_DURATION_MS: Dict[str, int] = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
}

# Default timeframe if not specified
DEFAULT_TIMEFRAME: str = "1h"


# ============================================================================
# MULTI-PAIR STRATEGY CONFIGURATION
# ============================================================================

# Default asset pairs for portfolio strategies
DEFAULT_PORTFOLIO_ASSETS: list[str] = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

# Default pairs for statistical arbitrage
DEFAULT_STAT_ARB_PAIRS: list[tuple[str, str]] = [
    ("BTC/USDT", "ETH/USDT"),
    ("ETH/USDT", "BNB/USDT"),
]

# Minimum correlation for copula pairs
MIN_COPULA_CORRELATION: float = 0.5


# ============================================================================
# HORIZON CONFIGURATIONS
# ============================================================================

# Default horizons for analysis (in days)
DEFAULT_HORIZONS: list[int] = [30, 90, 180]

# Quick mode horizons (faster testing)
QUICK_MODE_HORIZONS: list[int] = [30, 90]

# Full analysis horizons
FULL_HORIZONS: list[int] = [30, 90, 180, 365]


if __name__ == "__main__":
    """
    Validation block for constants module.
    Verifies all constants are defined and have correct types.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: Verify numeric constants are positive
    total_tests += 1
    numeric_constants = [
        ("BINANCE_MAX_CANDLES", BINANCE_MAX_CANDLES),
        ("MAX_HISTORICAL_BATCHES", MAX_HISTORICAL_BATCHES),
        ("DEFAULT_WARMUP_MULTIPLIER", DEFAULT_WARMUP_MULTIPLIER),
        ("MAX_DATA_LOSS_PERCENT", MAX_DATA_LOSS_PERCENT),
        ("MAX_SINGLE_PAIR_WORKERS", MAX_SINGLE_PAIR_WORKERS),
        ("MAX_MULTI_PAIR_WORKERS", MAX_MULTI_PAIR_WORKERS),
    ]

    for name, value in numeric_constants:
        if value <= 0:
            all_validation_failures.append(f"{name} must be positive, got {value}")

    # Test 2: Verify score weights sum to 1.0
    total_tests += 1
    total_weight = (
        SCORE_WEIGHT_RETURN +
        SCORE_WEIGHT_SHARPE +
        SCORE_WEIGHT_WIN_RATE +
        SCORE_WEIGHT_PROFIT_FACTOR +
        SCORE_WEIGHT_CONSISTENCY
    )
    if abs(total_weight - 1.0) > 0.001:
        all_validation_failures.append(f"Score weights sum to {total_weight}, expected 1.0")

    # Test 3: Verify dictionaries are not empty
    total_tests += 1
    if not PERIODS_PER_YEAR:
        all_validation_failures.append("PERIODS_PER_YEAR is empty")
    if not TIMEFRAME_DURATION_MS:
        all_validation_failures.append("TIMEFRAME_DURATION_MS is empty")

    # Test 4: Verify timeframe keys match
    total_tests += 1
    if set(PERIODS_PER_YEAR.keys()) != set(TIMEFRAME_DURATION_MS.keys()):
        all_validation_failures.append("Timeframe keys don't match between dictionaries")

    # Test 5: Verify horizon lists are sorted
    total_tests += 1
    for horizon_list, name in [
        (DEFAULT_HORIZONS, "DEFAULT_HORIZONS"),
        (QUICK_MODE_HORIZONS, "QUICK_MODE_HORIZONS"),
        (FULL_HORIZONS, "FULL_HORIZONS"),
    ]:
        if horizon_list != sorted(horizon_list):
            all_validation_failures.append(f"{name} is not sorted")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Constants module is validated and ready for use")
        print(f"\nSample constants:")
        print(f"  • BINANCE_MAX_CANDLES: {BINANCE_MAX_CANDLES}")
        print(f"  • DEFAULT_WARMUP_MULTIPLIER: {DEFAULT_WARMUP_MULTIPLIER}")
        print(f"  • Score weights sum: {total_weight:.3f}")
        print(f"  • Timeframes configured: {len(PERIODS_PER_YEAR)}")
        sys.exit(0)
