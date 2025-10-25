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

# PHASE 1: Risk management imports
from crypto_trader.risk.position_sizing import calculate_kelly_position_size

# PHASE 3: Transaction cost optimization
from crypto_trader.optimization.transaction_costs import should_rebalance as check_rebalance_threshold


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

        # PHASE 1: Kelly position sizing parameters
        self.use_kelly_sizing: bool = True  # Enable Kelly Criterion position sizing
        self.kelly_fraction: float = 0.25  # Conservative 25% of full Kelly
        self.min_position_pct: float = 0.02  # 2% minimum position
        self.max_position_pct: float = 0.15  # 15% maximum position

        # PHASE 3: Transaction cost optimization parameters
        self.transaction_cost_pct: float = 0.001  # 0.1% transaction cost (10 bps)
        self.min_rebalance_benefit: float = 0.005  # Only rebalance if benefit > 0.5%

        logger.debug(f"Initialized {self.name}Strategy with Kelly sizing and transaction cost optimization")

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
        Generate Risk Parity portfolio weights with graceful single-asset handling.

        BUGFIX: Now handles single-asset data gracefully instead of failing.

        Args:
            data: DataFrame with columns [timestamp, asset1_close, asset2_close, ...]

        Returns:
            DataFrame with Risk Parity weights for each period
        """
        logger.info(f"Generating Risk Parity signals for {len(self.asset_symbols)} assets")

        # Extract close price columns
        price_columns = [col for col in data.columns if col.endswith('_close')]

        # BUGFIX: Gracefully handle single-asset case
        if len(price_columns) < 2:
            logger.warning(
                f"Risk Parity requires ≥2 assets, found {len(price_columns)}. "
                f"Falling back to single-asset allocation."
            )
            return self._generate_single_asset_signals(data, price_columns)

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
            # PHASE 3 FIX: Convert to proper signal format
            return self._weights_to_signals(signals_df, price_columns)

        # Generate weights using Risk Parity at rebalancing intervals
        rebalance_dates = range(self.lookback_period, len(data), self.rebalance_freq)

        current_weights = None
        for i in range(len(data)):
            if i in rebalance_dates or current_weights is None:
                # Calculate Risk Parity weights
                window_returns = returns.iloc[max(0, i - self.lookback_period):i]

                if len(window_returns) >= 20:  # Minimum data requirement
                    try:
                        new_weights = self._calculate_risk_parity_weights(window_returns)

                        # PHASE 3: Check if rebalancing is worthwhile
                        if self._should_rebalance(new_weights, current_weights):
                            current_weights = new_weights
                            logger.debug(f"Risk Parity weights at index {i}: {new_weights}")
                        else:
                            logger.debug(f"Skipped rebalancing at index {i} due to transaction costs")

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

        # BUGFIX (Phase 2): Convert weight DataFrame to signal/confidence/metadata format
        # Required by backtesting engine's _signals_to_entries_exits() method
        return self._weights_to_signals(signals_df, price_columns)

    def _weights_to_signals(
        self,
        weights_df: pd.DataFrame,
        price_columns: list,
        weight_change_threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Convert portfolio weight DataFrame to signal/confidence/metadata format.

        BUGFIX (Phase 2): Portfolio strategies return weight allocations, but the
        backtesting engine expects BUY/SELL/HOLD signals. This method bridges the gap.

        Args:
            weights_df: DataFrame with 'weight_{asset}_close' columns
            price_columns: List of price column names
            weight_change_threshold: Minimum weight change to generate BUY/SELL signal (default 5%)

        Returns:
            DataFrame with columns: [timestamp, signal, confidence, metadata]
            - signal: 'BUY' for weight increases, 'SELL' for decreases, 'HOLD' otherwise
            - confidence: Magnitude of weight change (0-1 scale)
            - metadata: Dict containing current weights for all assets
        """
        signals = []
        confidences = []
        metadata_list = []

        # Track previous weights for change detection
        prev_weights = {col: 0.0 for col in price_columns}

        for idx, row in weights_df.iterrows():
            # Extract current weights
            current_weights = {}
            for col in price_columns:
                weight_col = f'weight_{col}'
                current_weights[col] = row.get(weight_col, 0.0) if weight_col in weights_df.columns else 0.0

            # Calculate total weight change
            total_weight_change = sum(abs(current_weights[col] - prev_weights[col]) for col in price_columns)

            # Determine signal based on dominant weight change
            if total_weight_change > weight_change_threshold:
                # Find asset with largest absolute weight increase
                max_increase = max(
                    (current_weights[col] - prev_weights[col], col)
                    for col in price_columns
                )
                weight_delta, dominant_asset = max_increase

                if weight_delta > weight_change_threshold:
                    signal = SignalType.BUY.value
                    confidence = min(abs(weight_delta), 1.0)  # Clip to [0, 1]
                elif weight_delta < -weight_change_threshold:
                    signal = SignalType.SELL.value
                    confidence = min(abs(weight_delta), 1.0)
                else:
                    signal = SignalType.HOLD.value
                    confidence = 0.0
            else:
                signal = SignalType.HOLD.value
                confidence = 0.0

            # Store metadata
            metadata = {
                'weights': current_weights.copy(),
                'total_weight_change': total_weight_change,
                'strategy': 'RiskParity'
            }

            signals.append(signal)
            confidences.append(confidence)
            metadata_list.append(metadata)

            # Update previous weights
            prev_weights = current_weights.copy()

        # Construct signal DataFrame
        result_df = pd.DataFrame({
            'timestamp': weights_df['timestamp'] if 'timestamp' in weights_df.columns else weights_df.index,
            'signal': signals,
            'confidence': confidences,
            'metadata': metadata_list
        })

        logger.debug(
            f"Converted {len(result_df)} weight periods to signals: "
            f"{sum(1 for s in signals if s == SignalType.BUY.value)} BUY, "
            f"{sum(1 for s in signals if s == SignalType.SELL.value)} SELL, "
            f"{sum(1 for s in signals if s == SignalType.HOLD.value)} HOLD"
        )

        return result_df

    def _generate_single_asset_signals(
        self,
        data: pd.DataFrame,
        price_columns: list
    ) -> pd.DataFrame:
        """
        Generate signals for single-asset case (graceful degradation).

        BUGFIX (Phase 3): Returns proper signal/confidence/metadata format
        required by backtesting engine.

        Args:
            data: DataFrame with OHLCV data
            price_columns: List of price column names

        Returns:
            DataFrame with columns: [timestamp, signal, confidence, metadata]
        """
        signals_df = pd.DataFrame({
            'timestamp': data['timestamp'] if 'timestamp' in data.columns else data.index
        })

        if len(price_columns) == 1:
            # 100% allocation to single asset
            signals_df[f'weight_{price_columns[0]}'] = 1.0
            logger.info(f"Generated single-asset signals: 100% to {price_columns[0]}")
        else:
            logger.warning("No price columns found, returning empty signals")

        # PHASE 3 FIX: Convert to proper signal format
        return self._weights_to_signals(signals_df, price_columns)

    def _calculate_risk_parity_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Risk Parity weights using Ledoit-Wolf shrinkage and ERC.

        CRITICAL ENHANCEMENTS for Sharpe improvement:
        1. Ledoit-Wolf covariance shrinkage (2024/2025 best practice)
        2. Improved numerical stability in ERC optimization
        3. Transaction cost awareness

        Args:
            returns: DataFrame of asset returns

        Returns:
            Dictionary mapping column names to weights
        """
        try:
            from pypfopt import risk_models

            # CRITICAL FIX: Use Ledoit-Wolf shrinkage instead of sample covariance
            prices_df = pd.DataFrame({
                col: (1 + returns[col]).cumprod() for col in returns.columns
            })
            S = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()
            cov_matrix = S.values  # Convert to numpy

            n_assets = len(returns.columns)

            # Use inverse volatility as starting point
            volatilities = np.sqrt(np.diag(cov_matrix))

            # NUMERICAL STABILITY: Check for zero/near-zero volatilities
            if np.any(volatilities < 1e-8):
                logger.warning("Near-zero volatility detected, using equal weights")
                return {col: 1.0 / n_assets for col in returns.columns}

            inv_vol_weights = 1.0 / volatilities
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()

            # Optimize for equal risk contribution with better numerical conditioning
            weights = self._optimize_risk_parity_robust(cov_matrix, inv_vol_weights)

            # Apply kurtosis minimization if enabled
            if self.minimize_kurtosis:
                kurtosis_scores = self._calculate_kurtosis(returns)
                # Penalize high-kurtosis assets (fat tails)
                kurtosis_adjustment = 1.0 / (1.0 + np.abs(kurtosis_scores))
                kurtosis_adjustment = kurtosis_adjustment / kurtosis_adjustment.sum()

                # Blend: 80% ERC, 20% kurtosis penalty
                weights = 0.8 * weights + 0.2 * kurtosis_adjustment
                weights = weights / weights.sum()

            # PHASE 1: Apply Kelly sizing to scale positions
            weight_dict = {col: float(weights[i]) for i, col in enumerate(returns.columns)}

            if self.use_kelly_sizing:
                try:
                    weight_dict = self._apply_kelly_sizing(
                        weights=weight_dict,
                        returns=returns,
                        cov_matrix=cov_matrix
                    )
                    # Convert back to array for transaction cost check
                    weights = np.array([weight_dict[col] for col in returns.columns])
                    logger.debug("Applied Kelly sizing to Risk Parity weights")
                except Exception as e:
                    logger.warning(f"Kelly sizing failed, using base weights: {e}")

            # Transaction cost check
            if self.last_weights is not None:
                prev_weights_array = np.array([self.last_weights.get(col, 0.0) for col in returns.columns])
                turnover = np.sum(np.abs(weights - prev_weights_array))
                tx_cost = turnover * 0.001  # 10 bps

                if tx_cost > 0.005:  # 50 bps threshold
                    logger.debug(f"Skipping rebalance: tx_cost={tx_cost:.4f}")
                    return self.last_weights

            # Save weights for next transaction cost check
            self.last_weights = weight_dict.copy()

            return weight_dict

        except Exception as e:
            logger.error(f"Risk Parity optimization error: {e}")
            return {col: 1.0 / len(returns.columns) for col in returns.columns}

    def _optimize_risk_parity_robust(
        self,
        cov_matrix: np.ndarray,
        initial_weights: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Optimize for equal risk contribution with improved numerical stability.

        BUGFIX: Added convergence checks and adaptive learning rate.

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

        # Adaptive learning rate
        learning_rate = 0.1
        prev_diff_norm = float('inf')

        for iteration in range(max_iter):
            # Calculate portfolio variance
            portfolio_var = weights @ cov_matrix @ weights

            # NUMERICAL STABILITY: Prevent division by zero
            if portfolio_var < 1e-12:
                logger.warning("Portfolio variance near zero, returning equal weights")
                return np.ones(n_assets) / n_assets

            # Marginal risk contribution
            marginal_contrib = cov_matrix @ weights

            # Risk contribution
            risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_var)

            # Target: equal risk
            target_contrib = np.ones(n_assets) / n_assets

            # Difference
            diff = risk_contrib - target_contrib
            diff_norm = np.max(np.abs(diff))

            # Check convergence
            if diff_norm < tol:
                logger.debug(f"Risk Parity converged in {iteration} iterations")
                break

            # Adaptive learning rate (reduce if not improving)
            if diff_norm > prev_diff_norm:
                learning_rate *= 0.5  # Reduce step size
            prev_diff_norm = diff_norm

            # Update weights
            weights = weights - learning_rate * diff

            # Project to simplex (non-negative, sum to 1)
            weights = np.maximum(weights, 0)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # Recovery: reset to equal weights
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

    def _apply_kelly_sizing(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame,
        cov_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Apply Kelly Criterion position sizing to scale portfolio weights.

        PHASE 1: Risk management enhancement.

        Uses Risk Parity base weights as signal confidence and calculates
        Kelly-optimal position sizes based on expected return and volatility.

        Args:
            weights: Base Risk Parity weights (from ERC optimization)
            returns: Historical returns DataFrame
            cov_matrix: Covariance matrix

        Returns:
            Kelly-scaled weights (normalized to sum to 1.0)
        """
        kelly_scaled_weights = {}

        for asset, base_weight in weights.items():
            # Skip negligible weights
            if base_weight < 0.01:
                kelly_scaled_weights[asset] = 0.0
                continue

            # Calculate expected return and volatility for this asset
            asset_returns = returns[asset]
            expected_return = asset_returns.mean() * 252  # Annualized
            volatility = asset_returns.std() * np.sqrt(252)  # Annualized

            # Estimate win rate from historical data
            win_rate = (asset_returns > 0).sum() / len(asset_returns)

            # Apply Kelly sizing using base weight as signal confidence
            kelly_size = calculate_kelly_position_size(
                expected_return=expected_return,
                volatility=volatility,
                win_rate=win_rate,
                signal_confidence=base_weight,  # RP weight = confidence
                kelly_fraction=self.kelly_fraction,
                min_position_pct=self.min_position_pct,
                max_position_pct=self.max_position_pct
            )

            kelly_scaled_weights[asset] = kelly_size

            logger.debug(
                f"Kelly sizing: {asset} return={expected_return:.3f}, "
                f"vol={volatility:.3f}, win_rate={win_rate:.3f}, "
                f"confidence={base_weight:.3f} → size={kelly_size:.4f}"
            )

        # Normalize weights to sum to 1.0
        total_weight = sum(kelly_scaled_weights.values())
        if total_weight > 0:
            kelly_scaled_weights = {
                asset: weight / total_weight
                for asset, weight in kelly_scaled_weights.items()
            }

        logger.debug(f"Kelly-scaled weights: {kelly_scaled_weights}")
        return kelly_scaled_weights

    def _should_rebalance(
        self,
        new_weights: Dict[str, float],
        current_weights: Optional[Dict[str, float]]
    ) -> bool:
        """
        Determine if rebalancing is beneficial after accounting for transaction costs.

        PHASE 3: Uses standardized transaction cost module for rebalancing decisions.

        Args:
            new_weights: Newly calculated optimal weights
            current_weights: Current portfolio weights (None if first allocation)

        Returns:
            True if rebalancing is beneficial, False otherwise
        """
        if current_weights is None:
            return True  # Initial allocation

        # PHASE 3: Use standardized transaction cost module
        should_rebal, turnover = check_rebalance_threshold(
            current_weights=current_weights,
            target_weights=new_weights,
            transaction_cost_pct=self.transaction_cost_pct,
            min_benefit_pct=self.min_rebalance_benefit
        )

        return should_rebal


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
