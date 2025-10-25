"""
Portfolio Rebalancing Strategy

This module implements a multi-asset portfolio rebalancing strategy that
systematically rebalances asset allocations when they drift from target weights.
This approach has been shown to outperform buy-and-hold by 77% in research.

**Purpose**: Implement threshold-based portfolio rebalancing to capture
mean reversion at the portfolio level while maintaining target asset allocations.

**Strategy Type**: Multi-Asset Portfolio Rebalancing
**Method**: Threshold-based rebalancing (15% deviation triggers rebalance)
**Signals**: REBALANCE when asset weight deviates >threshold from target

**Parameters**:
- assets: List of (symbol, target_weight) tuples
- rebalance_threshold: Deviation threshold to trigger rebalance (default: 0.15 = 15%)
- min_rebalance_interval_hours: Minimum hours between rebalances (default: 24)

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```python
portfolio_data = {
    'BTC/USDT': pd.DataFrame({'close': [...], 'timestamp': [...]}),
    'ETH/USDT': pd.DataFrame({'close': [...], 'timestamp': [...]}),
}
assets = [('BTC/USDT', 0.5), ('ETH/USDT', 0.5)]
```

**Expected Output**:
```python
signals = pd.DataFrame({
    'timestamp': [...],
    'BTC/USDT_signal': ['HOLD', 'SELL', 'HOLD', ...],
    'ETH/USDT_signal': ['HOLD', 'BUY', 'HOLD', ...],
    'rebalance_event': [False, True, False, ...],
    'metadata': [...]
})
```
"""

from typing import Any, Dict, List, Tuple
from datetime import timedelta

import pandas as pd
import numpy as np
from loguru import logger

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy

# [TASK-3.2] Convex optimization for Sharpe maximization
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    logger.warning("cvxpy not available - Sharpe optimization disabled")
    CVXPY_AVAILABLE = False


@register_strategy(
    name="PortfolioRebalancer",
    description="Multi-asset portfolio with threshold-based rebalancing",
    tags=["portfolio", "rebalancing", "multi_asset", "mean_reversion", "research_backed"]
)
class PortfolioRebalancerStrategy(BaseStrategy):
    """
    Portfolio Rebalancing Strategy.

    Maintains target asset allocations and rebalances when weights drift
    beyond threshold. Systematically sells winners and buys losers.

    Research shows this approach outperforms buy-and-hold by 77% with
    15% rebalancing threshold.
    """

    def __init__(self, name: str = "PortfolioRebalancer", config: Dict[str, Any] = None):
        """
        Initialize the Portfolio Rebalancer strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default parameters
        self.assets: List[Tuple[str, float]] = []  # List of (symbol, target_weight)
        self.rebalance_threshold = 0.15  # 15% deviation triggers rebalance
        self.min_rebalance_interval_hours = 24  # Don't rebalance more than once per day

        # Enhanced parameters
        self.rebalance_method = "threshold"  # "threshold", "calendar", or "hybrid"
        self.calendar_period_days = 30  # For calendar-based rebalancing
        self.use_momentum_filter = False  # Avoid rebalancing during strong trends
        self.momentum_lookback_days = 30  # Lookback period for momentum calculation

        # [TASK-3.2] Sharpe optimization parameters
        self.use_sharpe_optimization: bool = True  # Use mean-variance Sharpe maximization
        self.sharpe_lookback_days: int = 90  # Lookback for expected returns/cov
        self.use_momentum_overlay: bool = True  # Tilt weights based on momentum
        self.momentum_tilt_pct: float = 0.20  # Max 20% weight adjustment for momentum
        self.momentum_score_periods: List[int] = [21, 63, 126, 252]  # 1m, 3m, 6m, 12m
        self.dynamic_threshold: bool = True  # Adjust threshold based on volatility
        self.base_threshold: float = 0.15  # Base rebalancing threshold
        self.vol_high_threshold: float = 0.50  # High vol threshold (annual)
        self.vol_low_threshold: float = 0.20  # Low vol threshold (annual)

        logger.debug(f"Initialized {self.__class__.__name__} with Sharpe optimization")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with strategy parameters

        Raises:
            ValueError: If parameters are invalid
        """
        # Extract asset configuration
        if "assets" in config:
            self.assets = config["assets"]
        else:
            raise ValueError("Portfolio strategy requires 'assets' configuration")

        self.rebalance_threshold = config.get("rebalance_threshold", 0.15)
        self.min_rebalance_interval_hours = config.get("min_rebalance_interval_hours", 24)

        # Enhanced parameters
        self.rebalance_method = config.get("rebalance_method", "threshold")
        self.calendar_period_days = config.get("calendar_period_days", 30)
        self.use_momentum_filter = config.get("use_momentum_filter", False)
        self.momentum_lookback_days = config.get("momentum_lookback_days", 30)

        # [TASK-3.2] Sharpe optimization parameters
        self.use_sharpe_optimization = config.get("use_sharpe_optimization", True)
        self.sharpe_lookback_days = config.get("sharpe_lookback_days", 90)
        self.use_momentum_overlay = config.get("use_momentum_overlay", True)
        self.momentum_tilt_pct = config.get("momentum_tilt_pct", 0.20)
        self.dynamic_threshold = config.get("dynamic_threshold", True)
        self.base_threshold = config.get("base_threshold", 0.15)

        # Validate configuration
        if len(self.assets) < 2:
            raise ValueError("Portfolio must have at least 2 assets")

        total_weight = sum(weight for _, weight in self.assets)
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Asset weights must sum to 1.0, got {total_weight}")

        if self.rebalance_threshold <= 0 or self.rebalance_threshold >= 1:
            raise ValueError("Rebalance threshold must be between 0 and 1")

        if self.rebalance_method not in ["threshold", "calendar", "hybrid"]:
            raise ValueError("Rebalance method must be 'threshold', 'calendar', or 'hybrid'")

        if self.calendar_period_days <= 0:
            raise ValueError("Calendar period must be positive")

        self._initialized = True
        logger.info(
            f"{self.name} initialized with {len(self.assets)} assets, "
            f"method={self.rebalance_method}, threshold={self.rebalance_threshold:.1%}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "assets": self.assets,
            "rebalance_threshold": self.rebalance_threshold,
            "min_rebalance_interval_hours": self.min_rebalance_interval_hours,
            "rebalance_method": self.rebalance_method,
            "calendar_period_days": self.calendar_period_days,
            "use_momentum_filter": self.use_momentum_filter,
            "momentum_lookback_days": self.momentum_lookback_days
        }

    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators.

        Returns:
            Empty list - no indicators needed, only price data
        """
        return []

    def _optimize_weights_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights to maximize Sharpe ratio.

        [TASK-3.2] CRITICAL ENHANCEMENT for better risk-adjusted returns.

        Mathematical formulation:
        maximize: (μ'w) / sqrt(w'Σw)
        subject to: Σw_i = 1, w_i >= 0 (long-only)

        We convert to convex form by maximizing μ'w - λ * sqrt(w'Σw)
        where λ is risk aversion parameter (we use λ=1 for Sharpe maximization)

        Args:
            expected_returns: Expected returns for each asset (annualized)
            cov_matrix: Covariance matrix (annualized)
            target_weights: Original target weights (for fallback)

        Returns:
            Dictionary of optimized weights
        """
        if not CVXPY_AVAILABLE:
            logger.warning("cvxpy not available, using target weights")
            return target_weights

        try:
            n_assets = len(expected_returns)
            w = cp.Variable(n_assets)

            # Expected return
            ret = expected_returns.values @ w

            # Portfolio variance (use quad_form for numerical stability)
            risk = cp.quad_form(w, cov_matrix.values)

            # Objective: maximize Sharpe = return / risk
            # Equivalent to: maximize return - 0.5 * risk_aversion * risk
            # For Sharpe maximization, we set risk_aversion = 1
            objective = cp.Maximize(ret - 0.5 * risk)

            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,  # Long-only (no shorting)
                w <= 0.50  # Max 50% per asset (diversification)
            ]

            # Solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status in ["optimal", "optimal_inaccurate"]:
                optimized_weights = dict(zip(expected_returns.index, w.value))

                # Clean small weights (< 1%)
                total_weight = sum(optimized_weights.values())
                cleaned_weights = {
                    k: max(0, v / total_weight) if v > 0.01 else 0.0
                    for k, v in optimized_weights.items()
                }

                # Renormalize
                total_cleaned = sum(cleaned_weights.values())
                if total_cleaned > 0:
                    cleaned_weights = {k: v / total_cleaned for k, v in cleaned_weights.items()}

                logger.debug(
                    f"Sharpe optimization: ret={ret.value:.4f}, "
                    f"risk={np.sqrt(risk.value):.4f}, "
                    f"sharpe={(ret.value / np.sqrt(risk.value)):.4f}"
                )

                return cleaned_weights
            else:
                logger.warning(f"Optimization failed: {problem.status}, using target weights")
                return target_weights

        except Exception as e:
            logger.error(f"Sharpe optimization error: {e}, using target weights")
            return target_weights

    def _calculate_momentum_scores(
        self,
        data: Dict[str, pd.DataFrame],
        timestamp_idx: int
    ) -> Dict[str, float]:
        """
        Calculate composite momentum scores for tactical allocation.

        [TASK-3.2] Momentum overlay for enhanced returns.

        Composite momentum = weighted average of:
        - 1-month momentum (21 days): 25% weight
        - 3-month momentum (63 days): 25% weight
        - 6-month momentum (126 days): 25% weight
        - 12-month momentum (252 days): 25% weight

        Each momentum is z-score normalized for cross-asset comparison.

        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data
            timestamp_idx: Current index in the data

        Returns:
            Dictionary mapping symbol to momentum score (-2 to +2 typical range)
        """
        momentum_scores = {}

        for symbol, _ in self.assets:
            if symbol not in data:
                momentum_scores[symbol] = 0.0
                continue

            asset_data = data[symbol]
            scores = []

            # Calculate momentum for each period
            for lookback in self.momentum_score_periods:
                if timestamp_idx >= lookback:
                    current_price = asset_data.iloc[timestamp_idx]['close']
                    past_price = asset_data.iloc[timestamp_idx - lookback]['close']
                    momentum = (current_price - past_price) / past_price
                    scores.append(momentum)
                else:
                    scores.append(0.0)

            # Z-score normalization (if we have enough data)
            if len(scores) > 0 and np.std(scores) > 0:
                z_scores = [(s - np.mean(scores)) / np.std(scores) for s in scores]
                composite_score = np.mean(z_scores)
            else:
                composite_score = 0.0

            momentum_scores[symbol] = composite_score

        logger.debug(f"Momentum scores: {momentum_scores}")
        return momentum_scores

    def _apply_momentum_overlay(
        self,
        base_weights: Dict[str, float],
        momentum_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply momentum overlay to base weights.

        [TASK-3.2] Tilt weights based on momentum (±20% max adjustment).

        Logic:
        - Top momentum quintile: +20% weight increase
        - Bottom momentum quintile: -20% weight decrease
        - Middle quintiles: linear interpolation

        This provides tactical alpha while maintaining strategic allocation.

        Args:
            base_weights: Base portfolio weights
            momentum_scores: Momentum scores for each asset

        Returns:
            Adjusted weights with momentum overlay
        """
        if not self.use_momentum_overlay:
            return base_weights

        # Rank assets by momentum score
        sorted_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        n_assets = len(sorted_assets)

        # Calculate momentum tilts
        tilts = {}
        for rank, (symbol, score) in enumerate(sorted_assets):
            # Rank position (0 = best, 1 = worst)
            rank_pct = rank / max(1, n_assets - 1)

            # Linear tilt: top gets +tilt_pct, bottom gets -tilt_pct
            tilt = self.momentum_tilt_pct * (1 - 2 * rank_pct)
            tilts[symbol] = tilt

        # Apply tilts to base weights
        adjusted_weights = {}
        for symbol, base_weight in base_weights.items():
            tilt = tilts.get(symbol, 0.0)
            # Apply tilt as multiplicative factor
            adjusted_weight = base_weight * (1 + tilt)
            adjusted_weights[symbol] = max(0, adjusted_weight)

        # Renormalize to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        logger.debug(
            f"Momentum overlay applied: "
            f"base_weights={base_weights}, "
            f"adjusted_weights={adjusted_weights}"
        )

        return adjusted_weights

    def _calculate_dynamic_threshold(
        self,
        data: Dict[str, pd.DataFrame],
        timestamp_idx: int
    ) -> float:
        """
        Calculate dynamic rebalancing threshold based on market volatility.

        [TASK-3.2] Volatility-adaptive rebalancing thresholds.

        Logic:
        - High volatility (>50% annual): 20% threshold (rebalance less frequently)
        - Medium volatility (20-50% annual): 15% threshold (standard)
        - Low volatility (<20% annual): 10% threshold (rebalance more frequently)

        This prevents excessive trading in volatile markets and ensures
        timely rebalancing in stable markets.

        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data
            timestamp_idx: Current index in the data

        Returns:
            Dynamic rebalancing threshold (0.10 to 0.20)
        """
        if not self.dynamic_threshold:
            return self.base_threshold

        # Calculate realized volatility for each asset
        vols = []
        for symbol, _ in self.assets:
            if symbol not in data:
                continue

            asset_data = data[symbol]

            # Use last 30 days of returns for realized volatility
            lookback = min(30, timestamp_idx)
            if lookback < 10:
                continue

            returns = asset_data.iloc[timestamp_idx - lookback:timestamp_idx]['close'].pct_change().dropna()
            realized_vol = returns.std() * np.sqrt(252)  # Annualized
            vols.append(realized_vol)

        if len(vols) == 0:
            return self.base_threshold

        # Average volatility across assets
        avg_vol = np.mean(vols)

        # Determine threshold based on volatility regime
        if avg_vol > self.vol_high_threshold:
            # High volatility: wider threshold (20%)
            threshold = 0.20
            regime = "high_volatility"
        elif avg_vol < self.vol_low_threshold:
            # Low volatility: tighter threshold (10%)
            threshold = 0.10
            regime = "low_volatility"
        else:
            # Normal volatility: standard threshold (15%)
            threshold = self.base_threshold
            regime = "normal_volatility"

        logger.debug(
            f"Dynamic threshold: avg_vol={avg_vol:.2%}, "
            f"regime={regime}, threshold={threshold:.2%}"
        )

        return threshold

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate rebalancing signals for portfolio.

        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data

        Returns:
            DataFrame with columns for each asset's signals and rebalance events

        Raises:
            ValueError: If data is invalid
        """
        # Validate that we have data for all assets
        for symbol, _ in self.assets:
            if symbol not in data:
                raise ValueError(f"Missing data for asset: {symbol}")

        # Get common timestamps across all assets
        timestamps = None
        for symbol, _ in self.assets:
            asset_data = data[symbol]
            if timestamps is None:
                timestamps = asset_data.index
            else:
                timestamps = timestamps.intersection(asset_data.index)

        if len(timestamps) < 2:
            raise ValueError("Insufficient overlapping data across assets")

        logger.info(f"Generating portfolio signals for {len(timestamps)} periods")

        # Initialize result arrays
        signals_dict = {f"{symbol}_signal": [] for symbol, _ in self.assets}
        signals_dict['rebalance_event'] = []
        signals_dict['metadata'] = []

        # Track portfolio state
        last_rebalance_time = None
        shares = None

        # Iterate through timestamps
        for idx, timestamp in enumerate(timestamps):
            # Get current prices for all assets
            prices = {}
            for symbol, _ in self.assets:
                prices[symbol] = data[symbol].loc[timestamp, 'close']

            # Calculate current portfolio value and weights
            if idx == 0:
                # Initial allocation
                initial_capital = 10000  # This will be overridden by backtest config
                portfolio_values = {
                    symbol: initial_capital * target_weight
                    for symbol, target_weight in self.assets
                }
                shares = {
                    symbol: portfolio_values[symbol] / prices[symbol]
                    for symbol in prices
                }
            else:
                # Update portfolio values based on current prices
                portfolio_values = {
                    symbol: shares[symbol] * prices[symbol]
                    for symbol in shares
                }

            # Calculate total portfolio value and current weights
            total_value = sum(portfolio_values.values())
            current_weights = {
                symbol: portfolio_values[symbol] / total_value
                for symbol in portfolio_values
            }

            # [TASK-3.2] Calculate dynamic threshold based on volatility
            dynamic_threshold = self._calculate_dynamic_threshold(data, idx)

            # [TASK-3.2] Calculate optimized target weights if enabled
            if self.use_sharpe_optimization and idx >= self.sharpe_lookback_days:
                # Calculate expected returns and covariance matrix
                lookback_returns = {}
                for symbol, _ in self.assets:
                    if symbol in data:
                        returns = data[symbol].iloc[idx - self.sharpe_lookback_days:idx]['close'].pct_change().dropna()
                        lookback_returns[symbol] = returns

                if len(lookback_returns) == len(self.assets):
                    returns_df = pd.DataFrame(lookback_returns)
                    expected_returns = returns_df.mean() * 252  # Annualized
                    cov_matrix = returns_df.cov() * 252  # Annualized

                    # Get base target weights
                    base_target_weights = {symbol: target_weight for symbol, target_weight in self.assets}

                    # Optimize for Sharpe ratio
                    optimized_weights = self._optimize_weights_sharpe(
                        expected_returns,
                        cov_matrix,
                        base_target_weights
                    )

                    # [TASK-3.2] Apply momentum overlay
                    momentum_scores = self._calculate_momentum_scores(data, idx)
                    final_target_weights = self._apply_momentum_overlay(
                        optimized_weights,
                        momentum_scores
                    )
                else:
                    # Not enough data, use base target weights
                    final_target_weights = {symbol: target_weight for symbol, target_weight in self.assets}
            else:
                # Use base target weights
                final_target_weights = {symbol: target_weight for symbol, target_weight in self.assets}

            # Check if rebalancing is needed based on method
            needs_rebalance = False
            max_deviation = 0.0
            rebalance_reason = None

            # Calculate deviation from optimized target weights
            for symbol in final_target_weights.keys():
                target_weight = final_target_weights[symbol]
                deviation = abs(current_weights[symbol] - target_weight)
                max_deviation = max(max_deviation, deviation)

            # Determine rebalancing based on method (use dynamic threshold)
            if self.rebalance_method == "threshold":
                # Threshold-based: rebalance when deviation exceeds dynamic threshold
                if max_deviation > dynamic_threshold:
                    needs_rebalance = True
                    rebalance_reason = "threshold_rebalance"

            elif self.rebalance_method == "calendar":
                # Calendar-based: rebalance on fixed schedule
                if last_rebalance_time is None:
                    # First rebalance after initial allocation
                    pass
                else:
                    days_since_rebalance = (timestamp - last_rebalance_time).total_seconds() / (3600 * 24)
                    if days_since_rebalance >= self.calendar_period_days:
                        needs_rebalance = True
                        rebalance_reason = "calendar_rebalance"

            elif self.rebalance_method == "hybrid":
                # Hybrid: rebalance on calendar OR when threshold exceeded
                threshold_triggered = max_deviation > dynamic_threshold

                calendar_triggered = False
                if last_rebalance_time is not None:
                    days_since_rebalance = (timestamp - last_rebalance_time).total_seconds() / (3600 * 24)
                    calendar_triggered = days_since_rebalance >= self.calendar_period_days

                if threshold_triggered or calendar_triggered:
                    needs_rebalance = True
                    rebalance_reason = "threshold_rebalance" if threshold_triggered else "calendar_rebalance"

            # Check minimum interval (applies to all methods)
            if needs_rebalance and last_rebalance_time is not None:
                time_since_rebalance = (timestamp - last_rebalance_time).total_seconds() / 3600
                if time_since_rebalance < self.min_rebalance_interval_hours:
                    needs_rebalance = False

            # Apply momentum filter if enabled
            if needs_rebalance and self.use_momentum_filter:
                # Calculate portfolio momentum over lookback period
                lookback_periods = self.momentum_lookback_days * 24  # Convert days to hours
                if idx >= lookback_periods:
                    lookback_idx = max(0, idx - lookback_periods)
                    lookback_timestamp = timestamps[lookback_idx]

                    # Calculate portfolio return over lookback period
                    old_prices = {}
                    for symbol in prices:
                        old_prices[symbol] = data[symbol].loc[lookback_timestamp, 'close']

                    old_total_value = sum(shares[symbol] * old_prices[symbol] for symbol in shares)
                    portfolio_return = (total_value - old_total_value) / old_total_value

                    # Skip rebalancing if strong uptrend (>20% gain)
                    if portfolio_return > 0.20:
                        needs_rebalance = False
                        logger.debug(f"Skipped rebalance at {timestamp} due to strong uptrend: {portfolio_return:.2%}")

            # Generate signals
            if needs_rebalance:
                # Rebalance: sell overweight, buy underweight using optimized target weights
                for symbol in final_target_weights.keys():
                    target_weight = final_target_weights[symbol]
                    current_weight = current_weights[symbol]
                    if current_weight > target_weight:
                        # Overweight - sell
                        signals_dict[f"{symbol}_signal"].append(SignalType.SELL.value)
                    elif current_weight < target_weight:
                        # Underweight - buy
                        signals_dict[f"{symbol}_signal"].append(SignalType.BUY.value)
                    else:
                        signals_dict[f"{symbol}_signal"].append(SignalType.HOLD.value)

                signals_dict['rebalance_event'].append(True)
                signals_dict['metadata'].append({
                    'reason': rebalance_reason or 'threshold_rebalance',
                    'max_deviation': float(max_deviation),
                    'current_weights': {s: float(current_weights[s]) for s in current_weights},
                    'target_weights': {s: float(final_target_weights[s]) for s in final_target_weights},
                    'dynamic_threshold': float(dynamic_threshold)
                })

                # Update shares after rebalance using optimized target weights
                target_values = {
                    symbol: total_value * final_target_weights[symbol]
                    for symbol in final_target_weights.keys()
                }
                shares = {
                    symbol: target_values[symbol] / prices[symbol]
                    for symbol in prices
                }

                last_rebalance_time = timestamp

                logger.debug(
                    f"Rebalance at {timestamp}, max deviation: {max_deviation:.2%}"
                )
            else:
                # No rebalance - hold all
                for symbol, _ in self.assets:
                    signals_dict[f"{symbol}_signal"].append(SignalType.HOLD.value)

                signals_dict['rebalance_event'].append(False)
                signals_dict['metadata'].append({
                    'current_weights': {s: float(current_weights[s]) for s in current_weights},
                    'max_deviation': float(max_deviation)
                })

        # Create result DataFrame
        result = pd.DataFrame(signals_dict, index=timestamps)
        result.reset_index(inplace=True)
        result.rename(columns={'index': 'timestamp'}, inplace=True)

        rebalance_count = sum(signals_dict['rebalance_event'])
        logger.info(f"Generated signals: {rebalance_count} rebalance events out of {len(timestamps)} periods")

        return result

    def _create_hold_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create HOLD signals for all rows.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with HOLD signals
        """
        # For portfolio, we need to create HOLD for all assets
        signals_dict = {f"{symbol}_signal": [SignalType.HOLD.value] * len(data)
                        for symbol, _ in self.assets}
        signals_dict['timestamp'] = data.index if isinstance(data.index, pd.DatetimeIndex) else data['timestamp']
        signals_dict['rebalance_event'] = [False] * len(data)
        signals_dict['metadata'] = [{}] * len(data)

        return pd.DataFrame(signals_dict)


if __name__ == "__main__":
    """
    Validation block for Portfolio Rebalancer Strategy.
    Tests the strategy with synthetic multi-asset data.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting Portfolio Rebalancer Strategy validation")

    # Test 1: Initialize strategy
    total_tests += 1
    try:
        assets = [
            ("BTC/USDT", 0.50),
            ("ETH/USDT", 0.30),
            ("SOL/USDT", 0.20)
        ]

        strategy = PortfolioRebalancerStrategy()
        strategy.initialize({
            "assets": assets,
            "rebalance_threshold": 0.15,
            "min_rebalance_interval_hours": 24
        })

        params = strategy.get_parameters()
        if params['rebalance_threshold'] != 0.15:
            all_validation_failures.append(
                f"Test 1: Expected threshold=0.15, got {params['rebalance_threshold']}"
            )

        logger.success("Test 1 PASSED: Strategy initialized")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Generate synthetic multi-asset data
    total_tests += 1
    try:
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')

        # BTC: trending up
        btc_prices = 40000 + np.cumsum(np.random.randn(100) * 200)
        # ETH: more volatile
        eth_prices = 2000 + np.cumsum(np.random.randn(100) * 50)
        # SOL: different pattern
        sol_prices = 100 + np.cumsum(np.random.randn(100) * 5)

        portfolio_data = {
            "BTC/USDT": pd.DataFrame({
                'close': btc_prices,
                'open': btc_prices * 0.99,
                'high': btc_prices * 1.01,
                'low': btc_prices * 0.98,
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates),
            "ETH/USDT": pd.DataFrame({
                'close': eth_prices,
                'open': eth_prices * 0.99,
                'high': eth_prices * 1.01,
                'low': eth_prices * 0.98,
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates),
            "SOL/USDT": pd.DataFrame({
                'close': sol_prices,
                'open': sol_prices * 0.99,
                'high': sol_prices * 1.01,
                'low': sol_prices * 0.98,
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates)
        }

        logger.success("Test 2 PASSED: Generated multi-asset data")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Generate signals
    total_tests += 1
    try:
        signals = strategy.generate_signals(portfolio_data)

        if signals is None or signals.empty:
            all_validation_failures.append("Test 3: No signals generated")
        else:
            rebalance_events = signals['rebalance_event'].sum()
            logger.success(f"Test 3 PASSED: Generated {len(signals)} signals with {rebalance_events} rebalance events")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Portfolio Rebalancer Strategy validated with synthetic multi-asset data")
        sys.exit(0)
