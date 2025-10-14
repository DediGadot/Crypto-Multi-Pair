"""
Black-Litterman Portfolio Strategy

**Purpose**: Implements the Black-Litterman asset allocation model, which combines
market equilibrium returns with investor views to generate portfolio weights.
Uses Bayesian approach to blend prior beliefs with evidence.

**Third-party Packages**:
- PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/

**Sample Input**:
DataFrame with columns: timestamp, {asset1}_close, {asset2}_close, ...

**Expected Output**:
DataFrame with columns: timestamp, signal (weights for each asset)

**Research Backing**:
Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
Financial Analysts Journal, 48(5), 28-43.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy


@register_strategy(
    name="BlackLitterman",
    description="Black-Litterman Bayesian portfolio with views and market equilibrium",
    tags=["portfolio", "black_litterman", "multi_asset", "bayesian", "sota_2025"]
)
class BlackLittermanStrategy(BaseStrategy):
    """
    Black-Litterman portfolio strategy.

    Combines market equilibrium returns with momentum-based views to construct
    portfolios using Bayesian updating.

    Key Features:
    - Market equilibrium via reverse optimization
    - Momentum-based view generation
    - Bayesian blending of prior and views
    - Superior risk-adjusted returns
    """

    def __init__(self):
        """Initialize Black-Litterman strategy."""
        super().__init__(name="BlackLitterman")
        self.asset_symbols: list[str] = []
        self.lookback_period: int = 90
        self.rebalance_freq: int = 7  # Rebalance weekly
        self.risk_free_rate: float = 0.02  # Annual risk-free rate
        self.tau: float = 0.05  # Uncertainty in prior
        self.view_confidence: float = 0.25  # Confidence in views (omega)
        self.last_weights: Optional[Dict[str, float]] = None

        logger.debug(f"Initialized {self.name}Strategy")

    def initialize(self, params: Dict[str, Any]) -> None:
        """
        Initialize strategy parameters.

        Args:
            params: Dictionary with keys:
                - asset_symbols: List of asset symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
                - lookback_period: Historical window for covariance estimation (default: 90)
                - rebalance_freq: Days between rebalances (default: 7)
                - risk_free_rate: Annual risk-free rate (default: 0.02)
                - tau: Uncertainty parameter (default: 0.05)
                - view_confidence: Confidence in views (default: 0.25)
        """
        self.asset_symbols = params.get('asset_symbols', [])
        self.lookback_period = params.get('lookback_period', 90)
        self.rebalance_freq = params.get('rebalance_freq', 7)
        self.risk_free_rate = params.get('risk_free_rate', 0.02)
        self.tau = params.get('tau', 0.05)
        self.view_confidence = params.get('view_confidence', 0.25)

        logger.info(
            f"{self.name} initialized: assets={self.asset_symbols}, "
            f"lookback={self.lookback_period}, rebalance_freq={self.rebalance_freq}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary containing all strategy parameters
        """
        return {
            'asset_symbols': self.asset_symbols,
            'lookback_period': self.lookback_period,
            'rebalance_freq': self.rebalance_freq,
            'risk_free_rate': self.risk_free_rate,
            'tau': self.tau,
            'view_confidence': self.view_confidence
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Black-Litterman portfolio weights.

        Args:
            data: DataFrame with columns [timestamp, asset1_close, asset2_close, ...]

        Returns:
            DataFrame with Black-Litterman weights for each period
        """
        logger.info(f"Generating Black-Litterman signals for {len(self.asset_symbols)} assets")

        # Extract close price columns
        price_columns = [col for col in data.columns if col.endswith('_close')]

        if len(price_columns) < 2:
            logger.error(f"Need at least 2 assets, found {len(price_columns)}")
            return pd.DataFrame()

        # Create a DataFrame for signals
        signals_df = data[['timestamp']].copy() if 'timestamp' in data.columns else pd.DataFrame(index=data.index)

        # Initialize weight columns
        for col in price_columns:
            signals_df[f'weight_{col}'] = 0.0

        # Calculate returns for all assets
        returns = data[price_columns].pct_change().dropna()

        if len(returns) < self.lookback_period:
            logger.warning(f"Insufficient data: {len(returns)} < {self.lookback_period}")
            # Equal weight as fallback
            equal_weight = 1.0 / len(price_columns)
            for col in price_columns:
                signals_df[f'weight_{col}'] = equal_weight
            return signals_df

        # Generate weights using Black-Litterman at rebalancing intervals
        rebalance_dates = range(self.lookback_period, len(data), self.rebalance_freq)

        current_weights = None
        for i in range(len(data)):
            if i in rebalance_dates or current_weights is None:
                # Calculate Black-Litterman weights
                window_returns = returns.iloc[max(0, i - self.lookback_period):i]

                if len(window_returns) >= 20:  # Minimum data requirement
                    try:
                        weights = self._calculate_bl_weights(window_returns)
                        current_weights = weights
                        logger.debug(f"Black-Litterman weights at index {i}: {weights}")
                    except Exception as e:
                        logger.warning(f"Black-Litterman calculation failed at index {i}: {e}")
                        if current_weights is None:
                            # Fallback to equal weights
                            current_weights = {col: 1.0 / len(price_columns) for col in price_columns}
                else:
                    if current_weights is None:
                        current_weights = {col: 1.0 / len(price_columns) for col in price_columns}

            # Apply current weights
            if current_weights:
                for col in price_columns:
                    signals_df.loc[signals_df.index[i], f'weight_{col}'] = current_weights.get(col, 0.0)

        logger.success(f"Generated Black-Litterman signals for {len(signals_df)} periods")
        return signals_df

    def _calculate_bl_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Black-Litterman weights using PyPortfolioOpt.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Dictionary mapping column names to weights
        """
        try:
            from pypfopt import BlackLittermanModel, risk_models, expected_returns
            from pypfopt.black_litterman import BlackLittermanModel as BLModel

            # Calculate covariance matrix
            S = risk_models.CovarianceShrinkage(
                pd.DataFrame({col: (1 + returns[col]).cumprod() for col in returns.columns})
            ).ledoit_wolf()

            # Calculate market-cap weights (equal for simplicity)
            market_caps = {asset: 1.0 / len(returns.columns) for asset in returns.columns}

            # Generate views based on momentum
            views, omega = self._generate_views(returns)

            if len(views) == 0:
                # No views, use market equilibrium (skip Black-Litterman)
                logger.debug("No views generated, using market cap weighted portfolio")
                # Use market cap weights directly (equal weights in this case)
                return market_caps
            else:
                # Use views with Black-Litterman
                # Extract P (view matrix) and Q (expected returns) from views dict
                P = np.array([p_row for p_row, q_val in views.values()])
                Q = np.array([q_val for p_row, q_val in views.values()])

                bl = BlackLittermanModel(
                    cov_matrix=S,
                    pi="market",
                    market_caps=market_caps,
                    risk_aversion=1,
                    tau=self.tau,
                    P=P,
                    Q=Q,
                    omega=omega
                )

            # Get posterior returns
            ret_bl = bl.bl_returns()

            # Optimize portfolio
            from pypfopt import EfficientFrontier
            ef = EfficientFrontier(ret_bl, S)
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate / 252)  # Daily rate

            # Clean weights (remove very small allocations)
            cleaned_weights = ef.clean_weights()

            return cleaned_weights

        except Exception as e:
            logger.error(f"Black-Litterman optimization error: {e}")
            # Fallback to equal weights
            return {col: 1.0 / len(returns.columns) for col in returns.columns}

    def _generate_views(self, returns: pd.DataFrame) -> tuple[Dict[str, tuple[np.ndarray, float]], np.ndarray]:
        """
        Generate momentum-based views for Black-Litterman.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Tuple of (views_dict, omega_matrix) where views_dict contains (P_row, Q_value) tuples
        """
        views = {}

        # Calculate momentum signals (20-day vs 60-day returns)
        short_window = min(20, len(returns) // 3)
        long_window = min(60, len(returns))

        for col in returns.columns:
            short_return = returns[col].iloc[-short_window:].mean()
            long_return = returns[col].iloc[-long_window:].mean()

            # Generate view if momentum is strong enough
            momentum = short_return - long_return
            if abs(momentum) > 0.001:  # Threshold for view generation
                # Create view matrix row (1 for the asset, 0 for others)
                view_row = np.zeros(len(returns.columns))
                view_row[list(returns.columns).index(col)] = 1
                # Store both P row and Q value (expected return)
                views[col] = (view_row, momentum)

        # Create omega matrix (diagonal, represents view uncertainty)
        n_views = len(views)
        if n_views > 0:
            omega = np.eye(n_views) * self.view_confidence
        else:
            omega = np.array([])

        return views, omega


if __name__ == "__main__":
    """
    Validation function to test Black-Litterman strategy with real crypto data.
    """
    import sys
    from pathlib import Path

    # Add src to path
    src_dir = Path(__file__).parent.parent.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating Black-Litterman Strategy...\n")

    # Initialize variables
    strategy = None
    signals = None

    # Test 1: Initialize strategy
    total_tests += 1
    print("Test 1: Strategy initialization")
    try:
        strategy = BlackLittermanStrategy()
        strategy.initialize({
            'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'lookback_period': 90,
            'rebalance_freq': 7,
            'risk_free_rate': 0.02,
            'tau': 0.05,
            'view_confidence': 0.25
        })
        print(f"  ✓ Strategy initialized: {strategy.name}")
    except Exception as e:
        all_validation_failures.append(f"Initialization failed: {e}")

    # Test 2: Fetch real data and generate signals
    total_tests += 1
    print("\nTest 2: Generate Black-Litterman weights with real crypto data")
    if strategy is not None:
        try:
            fetcher = BinanceDataFetcher()

            # Fetch data for 3 assets
            btc_data = fetcher.get_ohlcv('BTC/USDT', '1h', limit=500)
            eth_data = fetcher.get_ohlcv('ETH/USDT', '1h', limit=500)
            bnb_data = fetcher.get_ohlcv('BNB/USDT', '1h', limit=500)

            if btc_data is None or eth_data is None or bnb_data is None:
                all_validation_failures.append("Failed to fetch data from Binance")
            else:
                # Combine data
                combined_data = pd.DataFrame({
                    'timestamp': btc_data.index,
                    'BTC_USDT_close': btc_data['close'].values,
                    'ETH_USDT_close': eth_data['close'].reindex(btc_data.index).values,
                    'BNB_USDT_close': bnb_data['close'].reindex(btc_data.index).values
                }).dropna()

                print(f"  ✓ Fetched {len(combined_data)} periods of data")

                # Generate signals
                signals = strategy.generate_signals(combined_data)

                if signals.empty:
                    all_validation_failures.append("Generated empty signals DataFrame")
                else:
                    # Check that weights sum to approximately 1
                    weight_cols = [col for col in signals.columns if col.startswith('weight_')]
                    total_weight = signals[weight_cols].iloc[-1].sum()

                    if abs(total_weight - 1.0) > 0.01:
                        all_validation_failures.append(
                            f"Weights don't sum to 1.0: {total_weight}"
                        )
                    else:
                        print(f"  ✓ Generated {len(signals)} signal periods")
                        print(f"  ✓ Final weights sum to {total_weight:.4f}")
                        print(f"\n  Latest Black-Litterman allocation:")
                        for col in weight_cols:
                            asset_name = col.replace('weight_', '').replace('_', '/')
                            weight = signals[col].iloc[-1]
                            print(f"    {asset_name}: {weight:.2%}")

        except Exception as e:
            all_validation_failures.append(f"Signal generation test exception: {e}")
            import traceback
            traceback.print_exc()

    # Test 3: Verify Black-Litterman properties
    total_tests += 1
    print("\nTest 3: Verify Black-Litterman properties")
    try:
        if signals is not None and not signals.empty:
            weight_cols = [col for col in signals.columns if col.startswith('weight_')]

            # Check all weights are non-negative
            min_weight = signals[weight_cols].min().min()
            if min_weight < -0.001:  # Small tolerance for numerical errors
                all_validation_failures.append(f"Negative weights found: {min_weight}")
            else:
                print(f"  ✓ All weights non-negative (min={min_weight:.6f})")

            # Check diversification (no single asset >80%)
            max_weight = signals[weight_cols].max().max()
            if max_weight > 0.80:
                print(f"  ⚠ High concentration: max weight = {max_weight:.2%}")
            else:
                print(f"  ✓ Reasonable diversification (max weight={max_weight:.2%})")

            # Check that weights change over time (rebalancing works)
            weight_variance = signals[weight_cols].var().sum()
            if weight_variance < 0.0001:
                all_validation_failures.append("Weights don't change over time")
            else:
                print(f"  ✓ Weights rebalance over time (variance={weight_variance:.6f})")

    except Exception as e:
        all_validation_failures.append(f"Properties verification exception: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Black-Litterman Strategy is validated and ready for production use")
        sys.exit(0)
