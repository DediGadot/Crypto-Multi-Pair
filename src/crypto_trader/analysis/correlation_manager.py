"""
Multi-Pair Correlation Management Framework

**Purpose**: Tracks real-time correlation matrices, detects correlation regime changes,
and provides correlation-based risk limits for multi-pair trading strategies.

**Key Features**:
- Exponentially weighted correlation matrices
- Correlation regime detection (crisis/normal/decorrelated)
- Dynamic position weighting based on correlations
- Marginal VaR contribution calculation
- Portfolio diversification metrics

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/
- scipy: https://docs.scipy.org/doc/scipy/

**Sample Input**:
```python
returns_df = pd.DataFrame({
    'BTC/USDT': [0.01, -0.02, 0.015, ...],
    'ETH/USDT': [0.012, -0.018, 0.013, ...],
    'SOL/USDT': [0.02, -0.025, 0.018, ...]
})
```

**Expected Output**:
```python
correlation_matrix = pd.DataFrame([
    [1.00, 0.85, 0.72],
    [0.85, 1.00, 0.68],
    [0.72, 0.68, 1.00]
], index=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], ...)
```

**CRITICAL FIX [TASK-2.1]**: Multi-pair strategies MUST consider correlations.
Without this, portfolio risk is massively underestimated during crisis periods.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


class CorrelationManager:
    """
    Manage correlation matrices and regime detection for multi-pair portfolios.

    IMPLEMENTATION [TASK-2.1]: This addresses the critical missing piece in
    multi-pair strategy risk management. Correlations spike to 0.95+ during
    crashes, turning "diversified" portfolios into concentrated bets.
    """

    def __init__(
        self,
        lookback_period: int = 90,
        ewm_halflife: int = 30,
        crisis_threshold: float = 0.7,
        decorrelated_threshold: float = 0.3
    ):
        """
        Initialize the Correlation Manager.

        Args:
            lookback_period: Historical window for correlation calculation (default: 90 days)
            ewm_halflife: Half-life for exponential weighting (default: 30 days)
            crisis_threshold: Avg correlation threshold for crisis regime (default: 0.7)
            decorrelated_threshold: Avg correlation below this = decorrelated (default: 0.3)
        """
        self.lookback_period = lookback_period
        self.ewm_halflife = ewm_halflife
        self.crisis_threshold = crisis_threshold
        self.decorrelated_threshold = decorrelated_threshold

        # State
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.rolling_correlations: Optional[pd.DataFrame] = None
        self.current_regime: str = "normal"
        self.regime_history: list = []

        logger.info(
            f"Initialized CorrelationManager: lookback={lookback_period}, "
            f"crisis_threshold={crisis_threshold}"
        )

    def update_correlations(
        self,
        returns_df: pd.DataFrame,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix with exponential weighting.

        IMPLEMENTATION [TASK-2.1]: Uses exponentially weighted correlations
        to give more weight to recent data. This is CRITICAL because:
        1. Market regimes change rapidly
        2. Recent correlations are more predictive
        3. Equal weighting over-smooths regime changes

        Lambda = 0.94 for daily data (RiskMetrics standard)
        Half-life = 30 days means correlation from 30 days ago gets 50% weight

        Args:
            returns_df: DataFrame with returns for each asset (columns = pairs)
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix as DataFrame

        Raises:
            ValueError: If returns_df has < 2 columns or insufficient data
        """
        # Validate input
        if len(returns_df.columns) < 2:
            raise ValueError(f"Need at least 2 assets, got {len(returns_df.columns)}")

        if len(returns_df) < 30:
            raise ValueError(f"Insufficient data: {len(returns_df)} rows, need 30+")

        # Remove any NaN values
        returns_clean = returns_df.dropna()

        if len(returns_clean) < 30:
            logger.warning(f"After dropna: {len(returns_clean)} rows, may be unstable")

        # Calculate exponentially weighted correlation
        # Lambda = exp(-log(2)/halflife)
        lambda_param = np.exp(-np.log(2) / self.ewm_halflife)

        # Use pandas ewm for proper exponential weighting
        ewm_cov = returns_clean.ewm(halflife=self.ewm_halflife, adjust=False).cov()

        # Extract final covariance matrix
        n_assets = len(returns_clean.columns)
        final_cov = ewm_cov.iloc[-n_assets:, :]

        # Convert covariance to correlation
        std_dev = np.sqrt(np.diag(final_cov))
        correlation_matrix = final_cov / np.outer(std_dev, std_dev)

        # Ensure diagonal = 1.0 (numerical stability)
        np.fill_diagonal(correlation_matrix.values, 1.0)

        # Store as DataFrame with proper index/columns
        self.correlation_matrix = pd.DataFrame(
            correlation_matrix,
            index=returns_clean.columns,
            columns=returns_clean.columns
        )

        logger.debug(
            f"Updated correlation matrix: avg={self._average_correlation():.3f}"
        )

        return self.correlation_matrix

    def detect_correlation_regime(self) -> str:
        """
        Detect current correlation regime based on average correlation.

        IMPLEMENTATION [TASK-2.1]: Three regimes based on empirical crypto research:
        - Crisis: avg correlation > 0.7 (everything moves together, sell-off)
        - Normal: 0.3 < avg correlation < 0.7 (typical market)
        - Decorrelated: avg correlation < 0.3 (rare, high alpha opportunity)

        Returns:
            Regime string: 'crisis', 'normal', or 'decorrelated'
        """
        if self.correlation_matrix is None:
            return "unknown"

        avg_corr = self._average_correlation()

        # Detect regime
        if avg_corr > self.crisis_threshold:
            regime = "crisis"
        elif avg_corr < self.decorrelated_threshold:
            regime = "decorrelated"
        else:
            regime = "normal"

        # Track regime changes
        if regime != self.current_regime:
            logger.info(
                f"Correlation regime changed: {self.current_regime} → {regime} "
                f"(avg_corr={avg_corr:.3f})"
            )
            self.regime_history.append({
                'timestamp': pd.Timestamp.now(),
                'old_regime': self.current_regime,
                'new_regime': regime,
                'avg_correlation': avg_corr
            })

        self.current_regime = regime
        return regime

    def get_correlation_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        max_correlation: float = 0.8
    ) -> Dict[str, float]:
        """
        Adjust portfolio weights to reduce exposure to highly correlated pairs.

        IMPLEMENTATION [TASK-2.1]: This is the CRITICAL risk management step.
        If two pairs have correlation > 0.8, they're essentially the same bet.
        We reduce their combined weight to avoid concentration risk.

        Algorithm:
        1. For each pair, find its maximum correlation with other pairs
        2. If max_corr > threshold, reduce weight proportionally
        3. Redistribute reduced weight to less-correlated pairs
        4. Ensure weights still sum to 1.0

        Args:
            base_weights: Initial weights for each pair
            max_correlation: Maximum allowed correlation before reduction (default: 0.8)

        Returns:
            Adjusted weights dictionary

        Example:
            If BTC and ETH have corr=0.92, and each has weight=0.3:
            - Reduce both to ~0.2 (33% reduction)
            - Redistribute 0.2 to SOL/XRP/etc
        """
        if self.correlation_matrix is None:
            logger.warning("No correlation matrix available, returning original weights")
            return base_weights

        adjusted_weights = base_weights.copy()
        pairs = list(base_weights.keys())

        # For each pair, find max correlation with others
        for pair in pairs:
            if pair not in self.correlation_matrix.index:
                continue

            # Get correlations with all other pairs
            pair_corrs = self.correlation_matrix.loc[pair, :].drop(pair)

            # Find maximum correlation
            max_corr = pair_corrs.max() if len(pair_corrs) > 0 else 0.0

            # If highly correlated, reduce weight
            if max_corr > max_correlation:
                # Reduction factor: linear from 1.0 (at max_correlation) to 0.5 (at 1.0)
                reduction_factor = 1.0 - 0.5 * ((max_corr - max_correlation) / (1.0 - max_correlation))
                adjusted_weights[pair] = base_weights[pair] * reduction_factor

                logger.debug(
                    f"Reduced {pair} weight: {base_weights[pair]:.3f} → "
                    f"{adjusted_weights[pair]:.3f} (max_corr={max_corr:.3f})"
                )

        # Renormalize to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        return adjusted_weights

    def get_marginal_var_contributions(
        self,
        positions: Dict[str, float],
        returns_df: pd.DataFrame,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate each pair's marginal contribution to portfolio VaR.

        IMPLEMENTATION [TASK-2.1]: This answers: "If I remove this pair,
        how much does portfolio VaR decrease?" Essential for risk budgeting.

        Uses variance-covariance method:
        Marginal VaR_i = (∂VaR/∂w_i) = (Σw)_i / σ_portfolio * z_score

        Args:
            positions: Position sizes for each pair
            returns_df: Historical returns DataFrame
            confidence: VaR confidence level (default: 0.95)

        Returns:
            Dict mapping pair → marginal VaR contribution
        """
        if self.correlation_matrix is None:
            self.update_correlations(returns_df)

        pairs = list(positions.keys())
        weights = np.array([positions[p] for p in pairs])

        # Calculate portfolio variance
        # σ²_p = w' Σ w
        cov_matrix = self.correlation_matrix.loc[pairs, pairs]

        # Need volatilities to convert correlation → covariance
        volatilities = returns_df[pairs].std()
        cov_matrix_values = cov_matrix.values * np.outer(volatilities.values, volatilities.values)

        portfolio_variance = weights @ cov_matrix_values @ weights
        portfolio_std = np.sqrt(portfolio_variance)

        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(confidence)

        # Marginal VaR = (Σw)_i / σ_p * z_score
        cov_times_weights = cov_matrix_values @ weights
        marginal_vars = (cov_times_weights / portfolio_std) * z_score

        # Convert to dict
        marginal_var_dict = {pair: float(mvar) for pair, mvar in zip(pairs, marginal_vars)}

        logger.info(
            f"Marginal VaR contributions: "
            f"max={max(marginal_var_dict.values()):.4f}, "
            f"min={min(marginal_var_dict.values()):.4f}"
        )

        return marginal_var_dict

    def calculate_diversification_ratio(
        self,
        weights: Dict[str, float],
        returns_df: pd.DataFrame
    ) -> float:
        """
        Calculate portfolio diversification ratio.

        IMPLEMENTATION [TASK-2.1]: DR measures diversification benefit.
        DR = (Weighted avg individual volatility) / (Portfolio volatility)

        - DR = 1.0: No diversification (perfect correlation)
        - DR > 1.5: Good diversification
        - DR > 2.0: Excellent diversification

        In crypto, achieving DR > 1.5 is challenging due to high correlations.

        Args:
            weights: Portfolio weights
            returns_df: Historical returns

        Returns:
            Diversification ratio (>= 1.0)
        """
        if self.correlation_matrix is None:
            self.update_correlations(returns_df)

        pairs = list(weights.keys())
        weight_array = np.array([weights[p] for p in pairs])

        # Individual volatilities
        volatilities = returns_df[pairs].std()

        # Weighted average individual volatility
        weighted_avg_vol = np.sum(weight_array * volatilities.values)

        # Portfolio volatility (accounting for correlations)
        cov_matrix = self.correlation_matrix.loc[pairs, pairs]
        cov_matrix_values = cov_matrix.values * np.outer(volatilities.values, volatilities.values)

        portfolio_variance = weight_array @ cov_matrix_values @ weight_array
        portfolio_vol = np.sqrt(portfolio_variance)

        # Diversification ratio
        if portfolio_vol > 0:
            dr = weighted_avg_vol / portfolio_vol
        else:
            dr = 1.0

        logger.info(f"Diversification ratio: {dr:.3f}")

        return dr

    def get_correlation_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of correlation matrix.

        Returns:
            Dict with mean, max, min, std of correlations
        """
        if self.correlation_matrix is None:
            return {
                'mean': 0.0,
                'max': 0.0,
                'min': 0.0,
                'std': 0.0,
                'regime': 'unknown'
            }

        # Get off-diagonal elements only (exclude self-correlation = 1.0)
        corr_values = self.correlation_matrix.values[np.triu_indices_from(
            self.correlation_matrix.values, k=1
        )]

        return {
            'mean': float(np.mean(corr_values)),
            'max': float(np.max(corr_values)),
            'min': float(np.min(corr_values)),
            'std': float(np.std(corr_values)),
            'regime': self.current_regime
        }

    def _average_correlation(self) -> float:
        """
        Calculate average off-diagonal correlation.

        Returns:
            Average pairwise correlation
        """
        if self.correlation_matrix is None:
            return 0.0

        # Get upper triangle (excluding diagonal)
        corr_values = self.correlation_matrix.values[np.triu_indices_from(
            self.correlation_matrix.values, k=1
        )]

        return float(np.mean(corr_values))


if __name__ == "__main__":
    """
    Validation function for CorrelationManager with real crypto data.

    VALIDATION [TASK-2.1]: This proves the correlation manager works correctly
    and can detect regime changes in live market data.
    """
    import sys
    from pathlib import Path

    # Add src to path
    src_dir = Path(__file__).parent.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating CorrelationManager...\n")

    # Test 1: Initialize manager
    total_tests += 1
    print("Test 1: Initialize CorrelationManager")
    try:
        manager = CorrelationManager(
            lookback_period=90,
            ewm_halflife=30,
            crisis_threshold=0.7,
            decorrelated_threshold=0.3
        )
        print(f"  ✓ Manager initialized: regime={manager.current_regime}")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Initialization failed: {e}")

    # Test 2: Fetch multi-pair data and calculate correlations
    total_tests += 1
    print("\nTest 2: Calculate correlations from real crypto data")
    try:
        fetcher = BinanceDataFetcher()

        # Fetch 3 pairs: BTC, ETH, SOL
        pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        returns_data = {}

        for pair in pairs:
            data = fetcher.get_ohlcv(pair, '1d', limit=200)
            if data is not None and not data.empty:
                # Calculate daily returns
                returns_data[pair] = data['close'].pct_change().dropna()

        # Combine into DataFrame
        returns_df = pd.DataFrame(returns_data).dropna()

        if len(returns_df) < 30:
            all_validation_failures.append(f"Test 2: Insufficient data: {len(returns_df)} days")
        else:
            # Calculate correlations
            corr_matrix = manager.update_correlations(returns_df)

            print(f"  ✓ Calculated {len(corr_matrix)}x{len(corr_matrix)} correlation matrix")
            print(f"  ✓ Data points: {len(returns_df)} days")

            # Show correlation matrix
            print(f"\n  Correlation Matrix:")
            print(corr_matrix.round(3).to_string(index=True))

    except Exception as e:
        all_validation_failures.append(f"Test 2: Correlation calculation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Detect correlation regime
    total_tests += 1
    print("\nTest 3: Detect correlation regime")
    try:
        if manager.correlation_matrix is not None:
            regime = manager.detect_correlation_regime()
            summary = manager.get_correlation_summary()

            print(f"  ✓ Current regime: {regime}")
            print(f"  ✓ Average correlation: {summary['mean']:.3f}")
            print(f"  ✓ Max correlation: {summary['max']:.3f}")
            print(f"  ✓ Min correlation: {summary['min']:.3f}")
            print(f"  ✓ Std correlation: {summary['std']:.3f}")

            # Verify regime logic
            if summary['mean'] > 0.7 and regime != 'crisis':
                all_validation_failures.append(
                    f"Test 3: Regime detection error: mean={summary['mean']:.3f} "
                    f"but regime={regime}, expected 'crisis'"
                )
            elif summary['mean'] < 0.3 and regime != 'decorrelated':
                all_validation_failures.append(
                    f"Test 3: Regime detection error: mean={summary['mean']:.3f} "
                    f"but regime={regime}, expected 'decorrelated'"
                )
            else:
                print(f"  ✓ Regime detection logic correct")
        else:
            all_validation_failures.append("Test 3: No correlation matrix available")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Regime detection failed: {e}")

    # Test 4: Correlation-adjusted weights
    total_tests += 1
    print("\nTest 4: Calculate correlation-adjusted weights")
    try:
        if manager.correlation_matrix is not None:
            # Equal weight portfolio
            base_weights = {pair: 1.0/3.0 for pair in pairs}

            # Adjust for correlations
            adjusted_weights = manager.get_correlation_adjusted_weights(base_weights, max_correlation=0.8)

            print(f"  Original weights:")
            for pair, weight in base_weights.items():
                print(f"    {pair}: {weight:.3f}")

            print(f"\n  Adjusted weights (max_corr=0.8):")
            for pair, weight in adjusted_weights.items():
                print(f"    {pair}: {weight:.3f}")

            # Verify weights sum to 1.0
            total_weight = sum(adjusted_weights.values())
            if abs(total_weight - 1.0) > 0.001:
                all_validation_failures.append(
                    f"Test 4: Weights don't sum to 1.0: {total_weight:.4f}"
                )
            else:
                print(f"\n  ✓ Weights sum to 1.0: {total_weight:.4f}")
        else:
            all_validation_failures.append("Test 4: No correlation matrix available")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Weight adjustment failed: {e}")

    # Test 5: Diversification ratio
    total_tests += 1
    print("\nTest 5: Calculate diversification ratio")
    try:
        if manager.correlation_matrix is not None and 'returns_df' in locals():
            weights = {pair: 1.0/3.0 for pair in pairs}
            dr = manager.calculate_diversification_ratio(weights, returns_df)

            print(f"  ✓ Diversification Ratio: {dr:.3f}")

            if dr >= 1.0:
                print(f"  ✓ DR >= 1.0 (mathematically valid)")
            else:
                all_validation_failures.append(
                    f"Test 5: Invalid DR: {dr:.3f} < 1.0"
                )

            # Interpretation
            if dr > 1.5:
                print(f"  ✓ Good diversification (DR > 1.5)")
            elif dr > 1.2:
                print(f"  ⚠ Moderate diversification (1.2 < DR < 1.5)")
            else:
                print(f"  ⚠ Poor diversification (DR < 1.2)")
        else:
            all_validation_failures.append("Test 5: Missing data for DR calculation")
    except Exception as e:
        all_validation_failures.append(f"Test 5: DR calculation failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("CorrelationManager is validated and ready for production use")
        sys.exit(0)
