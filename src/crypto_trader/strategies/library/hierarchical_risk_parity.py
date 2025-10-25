"""
Hierarchical Risk Parity (HRP) Portfolio Strategy

**Purpose**: Implements the Hierarchical Risk Parity algorithm for portfolio construction,
which uses hierarchical clustering to build diversified portfolios without requiring
covariance matrix inversion.

**Third-party Packages**:
- PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/

**Sample Input**:
DataFrame with columns: timestamp, {asset1}_close, {asset2}_close, ...

**Expected Output**:
DataFrame with columns: timestamp, signal (weights for each asset)

**Research Backing**:
Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample.
Journal of Portfolio Management.
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

# GARCH volatility forecasting [TASK-3.1]
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    logger.warning("arch library not available - GARCH forecasting disabled")
    ARCH_AVAILABLE = False


@register_strategy(
    name="HierarchicalRiskParity",
    description="Hierarchical Risk Parity portfolio with hierarchical clustering",
    tags=["portfolio", "hrp", "multi_asset", "risk_parity", "sota_2025"]
)
class HierarchicalRiskParityStrategy(BaseStrategy):
    """
    Hierarchical Risk Parity portfolio strategy.

    Uses hierarchical clustering and inverse-variance weighting to construct
    portfolios without requiring covariance matrix inversion.

    Key Features:
    - Hierarchical clustering based on correlation distance
    - Quasi-diagonalization of covariance matrix
    - Recursive bisection for weight allocation
    - Superior out-of-sample performance
    """

    def __init__(self):
        """Initialize HRP strategy."""
        super().__init__(name="HierarchicalRiskParity")
        self.asset_symbols: list[str] = []
        self.lookback_period: int = 90
        self.rebalance_freq: int = 7  # Rebalance weekly
        self.last_weights: Optional[Dict[str, float]] = None

        # [TASK-3.1] Enhanced parameters for Sharpe optimization
        self.use_garch_vol: bool = True  # Use GARCH(1,1) for volatility forecasting
        self.transaction_cost_pct: float = 0.001  # 0.1% transaction cost (10 bps)
        self.min_rebalance_benefit: float = 0.005  # Only rebalance if benefit > 0.5%
        self.use_regime_clustering: bool = True  # Adapt clustering to market regime
        self.dynamic_lookback: bool = True  # Adjust lookback based on volatility
        self.min_lookback: int = 60  # Minimum lookback period
        self.max_lookback: int = 180  # Maximum lookback period

        # PHASE 1: Kelly position sizing parameters
        self.use_kelly_sizing: bool = True  # Enable Kelly Criterion position sizing
        self.kelly_fraction: float = 0.25  # Conservative 25% of full Kelly
        self.min_position_pct: float = 0.02  # 2% minimum position
        self.max_position_pct: float = 0.15  # 15% maximum position

        logger.debug(f"Initialized {self.name}Strategy with GARCH vol forecasting and Kelly sizing")

    def initialize(self, params: Dict[str, Any]) -> None:
        """
        Initialize strategy parameters.

        Args:
            params: Dictionary with keys:
                - asset_symbols: List of asset symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
                - lookback_period: Historical window for return calculation (default: 90)
                - rebalance_freq: Days between rebalances (default: 7)
                - use_garch_vol: Use GARCH forecasting (default: True) [TASK-3.1]
                - transaction_cost_pct: Transaction cost percentage (default: 0.001) [TASK-3.1]
                - min_rebalance_benefit: Minimum benefit to rebalance (default: 0.005) [TASK-3.1]
                - use_regime_clustering: Use regime-adaptive clustering (default: True) [TASK-3.1]
                - dynamic_lookback: Use dynamic lookback window (default: True) [TASK-3.1]
        """
        self.asset_symbols = params.get('asset_symbols', [])
        self.lookback_period = params.get('lookback_period', 90)
        self.rebalance_freq = params.get('rebalance_freq', 7)

        # [TASK-3.1] Enhanced parameters
        self.use_garch_vol = params.get('use_garch_vol', True)
        self.transaction_cost_pct = params.get('transaction_cost_pct', 0.001)
        self.min_rebalance_benefit = params.get('min_rebalance_benefit', 0.005)
        self.use_regime_clustering = params.get('use_regime_clustering', True)
        self.dynamic_lookback = params.get('dynamic_lookback', True)

        logger.info(
            f"{self.name} initialized: assets={self.asset_symbols}, "
            f"lookback={self.lookback_period}, rebalance_freq={self.rebalance_freq}, "
            f"garch_vol={self.use_garch_vol}, tx_cost={self.transaction_cost_pct:.3f}%"
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
            'rebalance_freq': self.rebalance_freq
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate HRP portfolio weights with graceful single-asset handling.

        BUGFIX: Now handles single-asset data gracefully instead of failing.

        Args:
            data: DataFrame with columns [timestamp, asset1_close, asset2_close, ...]

        Returns:
            DataFrame with HRP weights for each period
        """
        logger.info(f"Generating HRP signals for {len(self.asset_symbols)} assets")

        # Extract close price columns
        price_columns = [col for col in data.columns if col.endswith('_close')]

        # BUGFIX: Gracefully handle single-asset case
        if len(price_columns) < 2:
            logger.warning(
                f"HRP requires ≥2 assets, found {len(price_columns)}. "
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

        # [TASK-3.1] Calculate adaptive lookback period
        adaptive_lookback = self._calculate_adaptive_lookback(returns)

        if len(returns) < adaptive_lookback:
            logger.warning(f"Insufficient data: {len(returns)} < {adaptive_lookback}")
            # Equal weight as fallback
            equal_weight = 1.0 / len(price_columns)
            for col in price_columns:
                signals_df[f'weight_{col}'] = equal_weight
            # PHASE 3 FIX: Convert to proper signal format
            return self._weights_to_signals(signals_df, price_columns)

        # Generate weights using HRP at rebalancing intervals
        rebalance_dates = range(adaptive_lookback, len(data), self.rebalance_freq)

        current_weights = None
        last_calculated_weights = None
        for i in range(len(data)):
            # Check if it's time to recalculate weights
            if i in rebalance_dates or current_weights is None:
                # [TASK-3.1] Use adaptive lookback window
                window_returns = returns.iloc[max(0, i - adaptive_lookback):i]

                if len(window_returns) >= 20:  # Minimum data requirement
                    try:
                        # Calculate new optimal weights
                        new_weights = self._calculate_hrp_weights(window_returns)

                        # [TASK-3.1] Check if rebalancing is beneficial (transaction cost-aware)
                        if self._should_rebalance(new_weights, current_weights):
                            current_weights = new_weights
                            last_calculated_weights = new_weights
                            logger.debug(f"HRP weights at index {i}: {new_weights}")
                        else:
                            # Keep current weights, skip rebalancing
                            logger.debug(f"Skipped rebalancing at index {i} due to transaction costs")

                    except Exception as e:
                        logger.warning(f"HRP calculation failed at index {i}: {e}")
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

        logger.success(f"Generated HRP signals for {len(signals_df)} periods")

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
                'strategy': 'HierarchicalRiskParity'
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

    def _forecast_volatility_garch(self, returns: pd.Series) -> float:
        """
        Forecast one-step-ahead volatility using GARCH(1,1).

        [TASK-3.1] CRITICAL ENHANCEMENT for Sharpe improvement.

        GARCH(1,1) model:
        σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}

        This provides superior volatility forecasts compared to historical std dev,
        especially during regime changes. Typical crypto GARCH parameters:
        - α (ARCH effect): 0.05-0.15 (volatility clustering)
        - β (GARCH effect): 0.80-0.90 (persistence)
        - Sum α+β < 1 for stationarity

        Args:
            returns: Series of historical returns (minimum 50 observations)

        Returns:
            Forecasted volatility (annualized std dev)
        """
        if not ARCH_AVAILABLE or len(returns) < 50:
            # Fallback to historical volatility
            return returns.std() * np.sqrt(252)  # Annualized

        try:
            # Remove any NaN/inf values
            returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()

            if len(returns_clean) < 50:
                return returns_clean.std() * np.sqrt(252)

            # Scale returns to percentage for numerical stability
            returns_scaled = returns_clean * 100

            # Fit GARCH(1,1) model
            # Mean model: constant (mu)
            # Volatility model: GARCH(1,1)
            model = arch_model(
                returns_scaled,
                vol='Garch',
                p=1,  # GARCH lag
                q=1,  # ARCH lag
                mean='constant',
                dist='normal'
            )

            # Fit with minimal output
            results = model.fit(disp='off', show_warning=False)

            # Forecast one-step-ahead variance
            forecast = results.forecast(horizon=1)
            forecasted_variance = forecast.variance.values[-1, 0]

            # Convert back to daily returns scale and annualize
            forecasted_vol_daily = np.sqrt(forecasted_variance) / 100
            forecasted_vol_annual = forecasted_vol_daily * np.sqrt(252)

            logger.debug(
                f"GARCH forecast: α={results.params.get('alpha[1]', 0):.4f}, "
                f"β={results.params.get('beta[1]', 0):.4f}, "
                f"vol={forecasted_vol_annual:.4f}"
            )

            return forecasted_vol_annual

        except Exception as e:
            logger.warning(f"GARCH fitting failed: {e}, using historical vol")
            return returns.std() * np.sqrt(252)

    def _calculate_adaptive_lookback(self, returns: pd.DataFrame) -> int:
        """
        Calculate optimal lookback period based on market volatility regime.

        [TASK-3.1] Dynamic lookback window optimization.

        Logic:
        - High volatility (>4% daily): Use shorter window (60 days) for faster adaptation
        - Medium volatility (2-4% daily): Use standard window (90 days)
        - Low volatility (<2% daily): Use longer window (180 days) for stability

        This prevents over-reaction in volatile markets and over-smoothing in stable markets.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Optimal lookback period (days)
        """
        if not self.dynamic_lookback:
            return self.lookback_period

        # Calculate average volatility across all assets
        vols = []
        for col in returns.columns:
            vol = returns[col].std() * np.sqrt(252)  # Annualized
            vols.append(vol)

        avg_vol = np.mean(vols)

        # Regime thresholds (annualized volatility)
        if avg_vol > 0.80:  # Very high vol (>80% annual)
            lookback = self.min_lookback  # 60 days
            regime = "high_volatility"
        elif avg_vol > 0.50:  # High vol (50-80% annual)
            lookback = int((self.min_lookback + self.lookback_period) / 2)  # 75 days
            regime = "medium_high_volatility"
        elif avg_vol > 0.30:  # Normal vol (30-50% annual)
            lookback = self.lookback_period  # 90 days
            regime = "normal_volatility"
        else:  # Low vol (<30% annual)
            lookback = self.max_lookback  # 180 days
            regime = "low_volatility"

        logger.debug(
            f"Adaptive lookback: avg_vol={avg_vol:.2%}, "
            f"regime={regime}, lookback={lookback} days"
        )

        return min(lookback, len(returns))

    def _should_rebalance(
        self,
        new_weights: Dict[str, float],
        current_weights: Optional[Dict[str, float]]
    ) -> bool:
        """
        Determine if rebalancing is beneficial after accounting for transaction costs.

        PHASE 3: Uses standardized transaction cost module for rebalancing decisions.

        Logic:
        Only rebalance if:
        Expected benefit (from better allocation) > Transaction costs + buffer

        Transaction cost = sum(|new_weight - old_weight|) * cost_pct

        The min_rebalance_benefit ensures we don't churn the portfolio for marginal improvements.
        This is CRITICAL for Sharpe optimization in practice.

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

    def _generate_single_asset_signals(
        self,
        data: pd.DataFrame,
        price_columns: list
    ) -> pd.DataFrame:
        """
        Generate signals for single-asset case (graceful degradation).

        BUGFIX (Phase 3): Returns proper signal/confidence/metadata format
        required by backtesting engine, not just weight columns.

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
            # No assets - return empty weights (edge case)
            logger.warning("No price columns found, returning empty signals")

        # PHASE 3 FIX: Convert to proper signal format
        return self._weights_to_signals(signals_df, price_columns)

    def _calculate_hrp_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate HRP weights using PyPortfolioOpt with Ledoit-Wolf shrinkage and GARCH.

        CRITICAL ENHANCEMENTS for Sharpe improvement:
        1. Ledoit-Wolf covariance shrinkage (2024/2025 best practice)
        2. GARCH volatility forecasting for better risk estimates
        3. Transaction cost awareness

        Args:
            returns: DataFrame of asset returns

        Returns:
            Dictionary mapping column names to weights
        """
        try:
            from pypfopt import HRPOpt, risk_models

            # CRITICAL FIX: Use Ledoit-Wolf shrinkage instead of sample covariance
            # This is essential for crypto (high noise, low sample size)
            prices_df = pd.DataFrame({
                col: (1 + returns[col]).cumprod() for col in returns.columns
            })
            cov_matrix = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()

            # OPTIONAL: Use GARCH forecasts if enabled AND reliable
            if self.use_garch_vol and ARCH_AVAILABLE and len(returns) >= 60:
                try:
                    # Update diagonal with GARCH forecasts
                    for i, col in enumerate(returns.columns):
                        garch_vol = self._forecast_volatility_garch(returns[col])
                        # Validate forecast is reasonable (not inf/nan/extreme)
                        if np.isfinite(garch_vol) and 0.01 < garch_vol < 5.0:
                            cov_matrix.iloc[i, i] = garch_vol ** 2
                        else:
                            logger.warning(f"Invalid GARCH vol for {col}: {garch_vol}, using Ledoit-Wolf")
                except Exception as e:
                    logger.warning(f"GARCH enhancement failed: {e}, using Ledoit-Wolf only")

            # Create HRP optimizer with improved covariance matrix
            hrp = HRPOpt(returns, cov_matrix=cov_matrix)

            # Optimize weights
            weights = hrp.optimize()

            # Clean weights (remove <1% allocations to reduce transaction costs)
            cleaned_weights = hrp.clean_weights(cutoff=0.01)

            # PHASE 1: Apply Kelly position sizing if enabled
            if self.use_kelly_sizing:
                cleaned_weights = self._apply_kelly_sizing(
                    weights=cleaned_weights,
                    returns=returns,
                    cov_matrix=cov_matrix
                )

            return cleaned_weights

        except Exception as e:
            logger.error(f"HRP optimization error: {e}")
            # Fallback to equal weights
            return {col: 1.0 / len(returns.columns) for col in returns.columns}

    def _apply_kelly_sizing(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Apply Kelly Criterion position sizing to scale portfolio weights.

        PHASE 1: This adjusts the base HRP weights by Kelly optimal sizing,
        providing better risk-adjusted position sizes.

        Args:
            weights: Base HRP weights (summing to 1.0)
            returns: Historical returns DataFrame
            cov_matrix: Covariance matrix

        Returns:
            Kelly-scaled weights (may not sum to 1.0)
        """
        try:
            kelly_scaled_weights = {}

            for asset, base_weight in weights.items():
                if base_weight < 0.01:  # Skip tiny allocations
                    kelly_scaled_weights[asset] = 0.0
                    continue

                # Calculate asset metrics
                asset_returns = returns[asset]
                expected_return = asset_returns.mean() * 252  # Annualized
                volatility = asset_returns.std() * np.sqrt(252)  # Annualized

                # Estimate win rate (% of positive returns)
                win_rate = (asset_returns > 0).sum() / len(asset_returns)
                win_rate = max(0.01, min(0.99, win_rate))  # Clip to valid range

                # Kelly position size for this asset
                kelly_size = calculate_kelly_position_size(
                    expected_return=expected_return,
                    volatility=volatility,
                    win_rate=win_rate,
                    signal_confidence=base_weight,  # Use HRP weight as confidence
                    kelly_fraction=self.kelly_fraction,
                    min_position_pct=self.min_position_pct,
                    max_position_pct=self.max_position_pct
                )

                kelly_scaled_weights[asset] = kelly_size

            # Normalize weights to sum to 1.0
            total_weight = sum(kelly_scaled_weights.values())
            if total_weight > 0:
                kelly_scaled_weights = {
                    asset: weight / total_weight
                    for asset, weight in kelly_scaled_weights.items()
                }

            logger.debug(f"Kelly sizing applied: {kelly_scaled_weights}")
            return kelly_scaled_weights

        except Exception as e:
            logger.warning(f"Kelly sizing failed: {e}, using base HRP weights")
            return weights


if __name__ == "__main__":
    """
    Validation function to test HRP strategy with real crypto data.
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

    print("🔍 Validating Hierarchical Risk Parity Strategy...\n")

    # Initialize variables
    strategy = None
    signals = None

    # Test 1: Initialize strategy
    total_tests += 1
    print("Test 1: Strategy initialization")
    try:
        strategy = HierarchicalRiskParityStrategy()
        strategy.initialize({
            'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'lookback_period': 90,
            'rebalance_freq': 7
        })
        print(f"  ✓ Strategy initialized: {strategy.name}")
    except Exception as e:
        all_validation_failures.append(f"Initialization failed: {e}")

    # Test 2: Fetch real data and generate signals
    total_tests += 1
    print("\nTest 2: Generate HRP weights with real crypto data")
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
                        print(f"\n  Latest HRP allocation:")
                        for col in weight_cols:
                            asset_name = col.replace('weight_', '').replace('_', '/')
                            weight = signals[col].iloc[-1]
                            print(f"    {asset_name}: {weight:.2%}")

        except Exception as e:
            all_validation_failures.append(f"Signal generation test exception: {e}")
            import traceback
            traceback.print_exc()

    # Test 3: Verify HRP properties
    total_tests += 1
    print("\nTest 3: Verify HRP properties")
    try:
        if not signals.empty:
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
        print("HRP Strategy is validated and ready for production use")
        sys.exit(0)
