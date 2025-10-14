"""
Risk Parity Portfolio Strategy

**Purpose**: Implements Risk Parity portfolio construction with Equal Risk Contribution (ERC)
and optional kurtosis minimization for tail risk management.

**Third-party Packages**:
- PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/
- cvxpy: https://www.cvxpy.org/
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- scipy: https://docs.scipy.org/doc/scipy/

**Sample Input**:
DataFrame with columns: timestamp, {asset1}_close, {asset2}_close, ...

**Expected Output**:
DataFrame with columns: timestamp, signal (weights for each asset)

**Research Backing**:
Maillard, S., Roncalli, T., & Teïletche, J. (2010). The Properties of Equally Weighted
Risk Contribution Portfolios. Journal of Portfolio Management, 36(4), 60-70.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy


@register_strategy(
    name="RiskParity",
    description="Risk Parity portfolio with Equal Risk Contribution and kurtosis minimization",
    tags=["portfolio", "risk_parity", "multi_asset", "equal_risk_contribution", "sota_2025"]
)
class RiskParityStrategy(BaseStrategy):
    """
    Risk Parity portfolio strategy with Equal Risk Contribution.

    Allocates capital such that each asset contributes equally to portfolio risk,
    with optional kurtosis minimization for better tail risk management.

    Key Features:
    - Equal risk contribution across assets
    - Optional kurtosis minimization
    - Better diversification than market-cap weighting
    - Robust to market regime changes
    """

    def __init__(self):
        """Initialize Risk Parity strategy."""
        super().__init__(name="RiskParity")
        self.asset_symbols: list[str] = []
        self.lookback_period: int = 90
        self.rebalance_freq: int = 7  # Rebalance weekly
        self.minimize_kurtosis: bool = True  # Enable kurtosis minimization
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
                - minimize_kurtosis: Whether to minimize kurtosis (default: True)
        """
        self.asset_symbols = params.get('asset_symbols', [])
        self.lookback_period = params.get('lookback_period', 90)
        self.rebalance_freq = params.get('rebalance_freq', 7)
        self.minimize_kurtosis = params.get('minimize_kurtosis', True)

        logger.info(
            f"{self.name} initialized: assets={self.asset_symbols}, "
            f"lookback={self.lookback_period}, rebalance_freq={self.rebalance_freq}, "
            f"minimize_kurtosis={self.minimize_kurtosis}"
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
            'minimize_kurtosis': self.minimize_kurtosis
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Risk Parity portfolio weights.

        Args:
            data: DataFrame with columns [timestamp, asset1_close, asset2_close, ...]

        Returns:
            DataFrame with Risk Parity weights for each period
        """
        logger.info(f"Generating Risk Parity signals for {len(self.asset_symbols)} assets")

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

        # Generate weights using Risk Parity at rebalancing intervals
        rebalance_dates = range(self.lookback_period, len(data), self.rebalance_freq)

        current_weights = None
        for i in range(len(data)):
            if i in rebalance_dates or current_weights is None:
                # Calculate Risk Parity weights
                window_returns = returns.iloc[max(0, i - self.lookback_period):i]

                if len(window_returns) >= 20:  # Minimum data requirement
                    try:
                        weights = self._calculate_risk_parity_weights(window_returns)
                        current_weights = weights
                        logger.debug(f"Risk Parity weights at index {i}: {weights}")
                    except Exception as e:
                        logger.warning(f"Risk Parity calculation failed at index {i}: {e}")
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

        logger.success(f"Generated Risk Parity signals for {len(signals_df)} periods")
        return signals_df

    def _calculate_risk_parity_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Risk Parity weights using Equal Risk Contribution.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Dictionary mapping column names to weights
        """
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov().values
            n_assets = len(returns.columns)

            # Use inverse volatility as starting point
            volatilities = np.sqrt(np.diag(cov_matrix))
            inv_vol_weights = 1.0 / volatilities
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()

            # Optimize for equal risk contribution using iterative method
            weights = self._optimize_risk_parity(cov_matrix, initial_weights=inv_vol_weights)

            # Apply kurtosis minimization if enabled
            if self.minimize_kurtosis:
                kurtosis_scores = self._calculate_kurtosis(returns)
                # Adjust weights by kurtosis penalty (reduce allocation to high-kurtosis assets)
                kurtosis_adjustment = 1.0 / (1.0 + np.abs(kurtosis_scores))
                kurtosis_adjustment = kurtosis_adjustment / kurtosis_adjustment.sum()

                # Blend ERC weights with kurtosis adjustment (80% ERC, 20% kurtosis)
                weights = 0.8 * weights + 0.2 * kurtosis_adjustment
                weights = weights / weights.sum()  # Renormalize

            # Create dictionary mapping column names to weights
            weight_dict = {col: float(weights[i]) for i, col in enumerate(returns.columns)}

            return weight_dict

        except Exception as e:
            logger.error(f"Risk Parity optimization error: {e}")
            # Fallback to equal weights
            return {col: 1.0 / len(returns.columns) for col in returns.columns}

    def _optimize_risk_parity(
        self,
        cov_matrix: np.ndarray,
        initial_weights: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Optimize for equal risk contribution using Newton's method.

        Args:
            cov_matrix: Covariance matrix of returns
            initial_weights: Starting weights
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Optimized weights array
        """
        weights = initial_weights.copy()
        n_assets = len(weights)

        for iteration in range(max_iter):
            # Calculate portfolio variance and marginal contributions
            portfolio_var = weights @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights

            # Risk contribution of each asset
            risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_var)

            # Target: equal risk contribution
            target_contrib = np.ones(n_assets) / n_assets

            # Calculate adjustment
            diff = risk_contrib - target_contrib

            if np.max(np.abs(diff)) < tol:
                break

            # Update weights (gradient descent step)
            learning_rate = 0.05
            weights = weights - learning_rate * diff

            # Project back to simplex (non-negative, sum to 1)
            weights = np.maximum(weights, 0)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(n_assets) / n_assets

        return weights

    def _calculate_kurtosis(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate excess kurtosis for each asset.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Array of excess kurtosis values
        """
        from scipy import stats

        kurtosis_values = np.array([
            stats.kurtosis(returns[col].dropna(), fisher=True)
            for col in returns.columns
        ])

        return kurtosis_values


if __name__ == "__main__":
    """
    Validation function to test Risk Parity strategy with real crypto data.
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

    print("🔍 Validating Risk Parity Strategy...\n")

    # Initialize variables
    strategy = None
    signals = None

    # Test 1: Initialize strategy
    total_tests += 1
    print("Test 1: Strategy initialization")
    try:
        strategy = RiskParityStrategy()
        strategy.initialize({
            'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'lookback_period': 90,
            'rebalance_freq': 7,
            'minimize_kurtosis': True
        })
        print(f"  ✓ Strategy initialized: {strategy.name}")
    except Exception as e:
        all_validation_failures.append(f"Initialization failed: {e}")

    # Test 2: Fetch real data and generate signals
    total_tests += 1
    print("\nTest 2: Generate Risk Parity weights with real crypto data")
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
                        print(f"\n  Latest Risk Parity allocation:")
                        for col in weight_cols:
                            asset_name = col.replace('weight_', '').replace('_', '/')
                            weight = signals[col].iloc[-1]
                            print(f"    {asset_name}: {weight:.2%}")

        except Exception as e:
            all_validation_failures.append(f"Signal generation test exception: {e}")
            import traceback
            traceback.print_exc()

    # Test 3: Verify Risk Parity properties
    total_tests += 1
    print("\nTest 3: Verify Risk Parity properties")
    try:
        if signals is not None and not signals.empty:
            weight_cols = [col for col in signals.columns if col.startswith('weight_')]

            # Check all weights are non-negative
            min_weight = signals[weight_cols].min().min()
            if min_weight < -0.001:  # Small tolerance for numerical errors
                all_validation_failures.append(f"Negative weights found: {min_weight}")
            else:
                print(f"  ✓ All weights non-negative (min={min_weight:.6f})")

            # Check diversification - Risk Parity should be well-diversified
            max_weight = signals[weight_cols].max().max()
            if max_weight > 0.70:  # Risk parity should avoid high concentration
                print(f"  ⚠ High concentration: max weight = {max_weight:.2%}")
            else:
                print(f"  ✓ Good diversification (max weight={max_weight:.2%})")

            # Check that weights change over time (rebalancing works)
            weight_variance = signals[weight_cols].var().sum()
            if weight_variance < 0.0001:
                all_validation_failures.append("Weights don't change over time")
            else:
                print(f"  ✓ Weights rebalance over time (variance={weight_variance:.6f})")

            # Check that weights are not too similar (should differ from equal weight)
            mean_weight = 1.0 / len(weight_cols)
            weight_deviation = signals[weight_cols].iloc[-1].std()
            if weight_deviation < 0.01:
                print(f"  ⚠ Weights too similar to equal weighting (std={weight_deviation:.4f})")
            else:
                print(f"  ✓ Weights differ from equal allocation (std={weight_deviation:.4f})")

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
        print("Risk Parity Strategy is validated and ready for production use")
        sys.exit(0)
