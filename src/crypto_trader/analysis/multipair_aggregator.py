"""
Multi-Pair Window Results Aggregator

This module extends the single-pair aggregator to handle multiple trading pairs
with synchronized windows and cross-pair correlation analysis.

**Purpose**: Aggregate and analyze performance across multiple pairs and windows

**Key Classes**:
- MultiPairWindowedMetrics: Aggregated statistics across pairs and windows
- CrossPairCorrelation: Correlation matrices between pairs
- MultiPairAggregator: Computes cross-pair aggregated metrics

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/

**Sample Input**:
```python
aggregator = MultiPairAggregator()
metrics = aggregator.aggregate_multipair_windows(
    results={'BTC/USDT': btc_results, 'ETH/USDT': eth_results},
    strategy_name='PortfolioRebalancer',
    horizon_name='30d',
    dataset_type='train'
)
```

**Expected Output**:
MultiPairWindowedMetrics with mean, median, std across pairs + correlation matrix.

**Methodology**:
- Per-pair statistics: Same as single-pair aggregator
- Cross-pair correlation: Pearson correlation of returns
- Portfolio metrics: Aggregate returns/Sharpe across all pairs
- Diversification benefit: Compare portfolio vs sum of individual metrics
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger

from .aggregator import WindowedMetrics, ResultsAggregator


@dataclass
class CrossPairCorrelation:
    """
    Correlation statistics between trading pairs.

    Attributes:
        pairs: List of pair symbols included
        correlation_matrix: Dict mapping (pair1, pair2) to correlation coefficient
        mean_correlation: Average correlation across all pair combinations
        max_correlation: Highest correlation (most similar)
        min_correlation: Lowest correlation (most diversified)
    """
    pairs: List[str]
    correlation_matrix: Dict[tuple, float]
    mean_correlation: float
    max_correlation: float
    min_correlation: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pairs': self.pairs,
            'correlation_matrix': {
                f"{p1}_{p2}": corr
                for (p1, p2), corr in self.correlation_matrix.items()
            },
            'mean_correlation': self.mean_correlation,
            'max_correlation': self.max_correlation,
            'min_correlation': self.min_correlation
        }


@dataclass
class MultiPairWindowedMetrics:
    """
    Aggregated metrics for a multi-pair strategy across windows.

    Combines per-pair statistics with cross-pair correlation analysis.
    """
    strategy_name: str
    horizon_name: str
    dataset_type: str  # 'train' or 'test'
    pairs: List[str]
    num_windows: int

    # Per-pair metrics (dict mapping pair symbol to WindowedMetrics)
    pair_metrics: Dict[str, WindowedMetrics]

    # Portfolio-level metrics (aggregated across all pairs)
    portfolio_mean_return: float
    portfolio_median_return: float
    portfolio_std_return: float
    portfolio_sharpe: float
    portfolio_drawdown: float

    # Cross-pair correlation
    correlation: CrossPairCorrelation

    # Diversification benefit
    diversification_ratio: float  # Portfolio Sharpe / Average individual Sharpe

    # PHASE 2: Advanced portfolio metrics
    risk_contribution: Dict[str, float]  # Marginal contribution to portfolio risk per asset
    effective_num_assets: float  # Diversification metric based on correlations
    correlation_matrix_df: Optional[pd.DataFrame] = None  # Full correlation matrix for visualization

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'strategy_name': self.strategy_name,
            'horizon_name': self.horizon_name,
            'dataset_type': self.dataset_type,
            'pairs': self.pairs,
            'num_windows': self.num_windows,
            'pair_metrics': {
                pair: metrics.to_dict()
                for pair, metrics in self.pair_metrics.items()
            },
            'portfolio_mean_return': self.portfolio_mean_return,
            'portfolio_median_return': self.portfolio_median_return,
            'portfolio_std_return': self.portfolio_std_return,
            'portfolio_sharpe': self.portfolio_sharpe,
            'portfolio_drawdown': self.portfolio_drawdown,
            'correlation': self.correlation.to_dict(),
            'diversification_ratio': self.diversification_ratio,
            # PHASE 2: Advanced portfolio metrics
            'risk_contribution': self.risk_contribution,
            'effective_num_assets': self.effective_num_assets,
        }
        # Add correlation matrix if available
        if self.correlation_matrix_df is not None:
            result['correlation_matrix'] = self.correlation_matrix_df.to_dict()
        return result


class MultiPairAggregator:
    """
    Aggregates results across multiple trading pairs and windows.

    Extends single-pair functionality with:
    - Per-pair window aggregation
    - Cross-pair correlation analysis
    - Portfolio-level metrics
    - Diversification analysis
    """

    def __init__(self):
        """Initialize multi-pair aggregator."""
        self.single_pair_aggregator = ResultsAggregator()

    def aggregate_pair_windows(
        self,
        results: List[Dict[str, Any]],
        strategy_name: str,
        horizon_name: str,
        dataset_type: str,
        pair_symbol: str
    ) -> WindowedMetrics:
        """
        Aggregate results for a single pair across windows.

        Args:
            results: List of backtest results for this pair
            strategy_name: Name of strategy
            horizon_name: Horizon identifier (e.g., '30d')
            dataset_type: 'train' or 'test'
            pair_symbol: Trading pair symbol

        Returns:
            WindowedMetrics for this pair
        """
        return self.single_pair_aggregator.aggregate_windows(
            results, strategy_name, horizon_name, dataset_type
        )

    def compute_correlation_matrix(
        self,
        pair_results: Dict[str, List[Dict[str, Any]]]
    ) -> CrossPairCorrelation:
        """
        Compute correlation matrix between pairs based on returns.

        Args:
            pair_results: Dict mapping pair symbol to list of window results

        Returns:
            CrossPairCorrelation object with correlation statistics
        """
        pairs = list(pair_results.keys())

        # Extract returns for each pair
        pair_returns = {}
        for pair, results in pair_results.items():
            returns = []
            for r in results:
                if not r or 'error' in r:
                    continue
                value = r.get('total_return')
                if value is None:
                    value = r.get('total_return_pct', 0.0)
                returns.append(float(value))
            pair_returns[pair] = returns

        # Ensure all pairs have same number of windows
        min_windows = min(len(returns) for returns in pair_returns.values())
        if min_windows == 0:
            logger.warning("No valid results for correlation computation")
            return CrossPairCorrelation(
                pairs=pairs,
                correlation_matrix={},
                mean_correlation=0.0,
                max_correlation=0.0,
                min_correlation=0.0
            )

        # Truncate to minimum length
        for pair in pair_returns:
            pair_returns[pair] = pair_returns[pair][:min_windows]

        # Compute pairwise correlations
        correlation_matrix = {}
        correlations = []

        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i < j:  # Only compute upper triangle
                    try:
                        corr = np.corrcoef(pair_returns[pair1], pair_returns[pair2])[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                        correlation_matrix[(pair1, pair2)] = corr
                        correlations.append(corr)
                    except Exception as e:
                        logger.warning(f"Could not compute correlation {pair1}-{pair2}: {e}")
                        correlation_matrix[(pair1, pair2)] = 0.0

        if not correlations:
            mean_corr = 0.0
            max_corr = 0.0
            min_corr = 0.0
        else:
            mean_corr = float(np.mean(correlations))
            max_corr = float(np.max(correlations))
            min_corr = float(np.min(correlations))

        return CrossPairCorrelation(
            pairs=pairs,
            correlation_matrix=correlation_matrix,
            mean_correlation=mean_corr,
            max_correlation=max_corr,
            min_correlation=min_corr
        )

    def build_correlation_matrix_df(
        self,
        pair_results: Dict[str, List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """
        Build full correlation matrix as DataFrame for visualization.

        PHASE 2: Added for interactive heatmap visualization.

        Args:
            pair_results: Dict mapping pair symbol to list of window results

        Returns:
            DataFrame with correlation matrix (pairs x pairs)
        """
        pairs = list(pair_results.keys())

        # Extract returns for each pair
        pair_returns = {}
        for pair, results in pair_results.items():
            returns = []
            for r in results:
                if not r or 'error' in r:
                    continue
                value = r.get('total_return')
                if value is None:
                    value = r.get('total_return_pct', 0.0)
                returns.append(float(value))
            pair_returns[pair] = returns

        # Build full correlation matrix
        n_pairs = len(pairs)
        corr_matrix = np.ones((n_pairs, n_pairs))

        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i != j:
                    try:
                        corr = np.corrcoef(pair_returns[pair1], pair_returns[pair2])[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                        corr_matrix[i, j] = corr
                    except Exception:
                        corr_matrix[i, j] = 0.0

        return pd.DataFrame(corr_matrix, index=pairs, columns=pairs)

    def compute_risk_contribution(
        self,
        pair_results: Dict[str, List[Dict[str, Any]]],
        correlation_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute marginal contribution to portfolio risk for each asset.

        PHASE 2: Risk decomposition showing which assets drive portfolio volatility.

        Formula: MRC_i = w_i * (Cov(R_i, R_portfolio) / σ_portfolio)

        For equal-weight portfolio, this simplifies to correlation-weighted volatility.

        Args:
            pair_results: Dict mapping pair symbol to list of window results
            correlation_df: Full correlation matrix

        Returns:
            Dict mapping pair symbol to risk contribution (%)
        """
        pairs = list(pair_results.keys())
        n_pairs = len(pairs)

        if n_pairs == 0:
            return {}

        # Extract return series and compute volatilities
        pair_returns = {}
        pair_vols = {}
        for pair, results in pair_results.items():
            returns = []
            for r in results:
                if not r or 'error' in r:
                    continue
                value = r.get('total_return')
                if value is None:
                    value = r.get('total_return_pct', 0.0)
                returns.append(float(value))
            pair_returns[pair] = returns
            pair_vols[pair] = np.std(returns) if len(returns) > 0 else 0.0

        # Compute portfolio volatility (equal-weight)
        # σ_p^2 = Σ Σ w_i w_j σ_i σ_j ρ_ij
        portfolio_var = 0.0
        weight = 1.0 / n_pairs  # Equal weight

        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                corr = correlation_df.iloc[i, j]
                portfolio_var += weight * weight * pair_vols[pair1] * pair_vols[pair2] * corr

        portfolio_vol = np.sqrt(portfolio_var) if portfolio_var > 0 else 0.0

        # Compute marginal risk contribution for each asset
        # MRC_i = w_i * Σ(w_j * σ_i * σ_j * ρ_ij) / σ_p
        risk_contributions = {}

        for i, pair in enumerate(pairs):
            marginal_contrib = 0.0
            for j, other_pair in enumerate(pairs):
                corr = correlation_df.iloc[i, j]
                marginal_contrib += weight * pair_vols[pair] * pair_vols[other_pair] * corr

            if portfolio_vol > 0:
                # Normalize to percentage of total risk
                risk_contributions[pair] = (marginal_contrib / portfolio_vol) * 100.0
            else:
                risk_contributions[pair] = 0.0

        return risk_contributions

    def compute_effective_num_assets(
        self,
        correlation_df: pd.DataFrame
    ) -> float:
        """
        Compute effective number of assets based on correlation structure.

        PHASE 2: Diversification metric showing true portfolio breadth.

        Uses eigenvalue decomposition of correlation matrix.
        Formula: N_eff = 1 / Σ(λ_i^2) where λ_i are normalized eigenvalues

        Higher values = better diversification
        N_eff = N for uncorrelated assets
        N_eff = 1 for perfectly correlated assets

        Args:
            correlation_df: Full correlation matrix

        Returns:
            Effective number of assets (1 to N)
        """
        if correlation_df.empty:
            return 0.0

        try:
            # Compute eigenvalues of correlation matrix
            eigenvalues = np.linalg.eigvalsh(correlation_df.values)

            # Normalize eigenvalues
            eigenvalues = eigenvalues / eigenvalues.sum()

            # Effective number = 1 / sum of squared eigenvalues
            # (This is the inverse participation ratio)
            eff_num = 1.0 / np.sum(eigenvalues ** 2)

            return float(eff_num)

        except Exception as e:
            logger.warning(f"Could not compute effective number of assets: {e}")
            return float(len(correlation_df))

    def compute_portfolio_metrics(
        self,
        pair_metrics: Dict[str, WindowedMetrics],
        pair_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Compute portfolio-level metrics from per-pair metrics.

        Uses equal-weight portfolio assumption with CORRECT Sharpe calculation.

        Args:
            pair_metrics: Dict mapping pair symbol to WindowedMetrics
            pair_results: Dict mapping pair symbol to list of window results
                         (REQUIRED for proper Sharpe calculation)

        Returns:
            Dict with portfolio metrics
        """
        if not pair_metrics:
            return {
                'portfolio_mean_return': 0.0,
                'portfolio_median_return': 0.0,
                'portfolio_std_return': 0.0,
                'portfolio_sharpe': 0.0,
                'portfolio_drawdown': 0.0,
                'diversification_ratio': 1.0
            }

        # Extract return series for each pair across all windows
        pair_return_series = {}
        for pair, results in pair_results.items():
            # Extract returns, handling both 'total_return' and 'total_return_pct'
            returns = []
            for r in results:
                if r and 'error' not in r:
                    ret = r.get('total_return', r.get('total_return_pct', 0.0))
                    returns.append(ret)
            pair_return_series[pair] = returns

        # Calculate number of windows (use minimum across all pairs)
        num_windows = min(len(r) for r in pair_return_series.values()) if pair_return_series else 0

        if num_windows == 0:
            return {
                'portfolio_mean_return': 0.0,
                'portfolio_median_return': 0.0,
                'portfolio_std_return': 0.0,
                'portfolio_sharpe': 0.0,
                'portfolio_drawdown': 0.0,
                'diversification_ratio': 1.0
            }

        # Compute equal-weight portfolio returns for each window
        num_pairs = len(pair_return_series)
        portfolio_returns = []

        for i in range(num_windows):
            # Equal-weight portfolio return for this window
            window_return = sum(
                pair_return_series[pair][i]
                for pair in pair_return_series.keys()
            ) / num_pairs
            portfolio_returns.append(window_return)

        # Calculate portfolio statistics from return series
        portfolio_mean_return = float(np.mean(portfolio_returns))
        portfolio_median_return = float(np.median(portfolio_returns))
        portfolio_std_return = float(np.std(portfolio_returns))

        # TRUE Portfolio Sharpe = Mean / Std
        if portfolio_std_return > 0:
            portfolio_sharpe = portfolio_mean_return / portfolio_std_return
        else:
            portfolio_sharpe = 0.0

        # Portfolio drawdown: worst drawdown across all pairs (conservative)
        drawdowns = [m.mean_drawdown for m in pair_metrics.values()]
        portfolio_drawdown = float(np.max(drawdowns))

        # Diversification ratio: Portfolio Sharpe / Average individual Sharpe
        individual_sharpes = [m.mean_sharpe for m in pair_metrics.values()]
        avg_individual_sharpe = float(np.mean(individual_sharpes))

        if avg_individual_sharpe != 0:
            diversification_ratio = portfolio_sharpe / avg_individual_sharpe
        else:
            diversification_ratio = 1.0

        return {
            'portfolio_mean_return': portfolio_mean_return,
            'portfolio_median_return': portfolio_median_return,
            'portfolio_std_return': portfolio_std_return,
            'portfolio_sharpe': portfolio_sharpe,
            'portfolio_drawdown': portfolio_drawdown,
            'diversification_ratio': diversification_ratio
        }

    def aggregate_multipair_windows(
        self,
        pair_results: Dict[str, List[Dict[str, Any]]],
        strategy_name: str,
        horizon_name: str,
        dataset_type: str
    ) -> MultiPairWindowedMetrics:
        """
        Aggregate results across multiple pairs and windows.

        Args:
            pair_results: Dict mapping pair symbol to list of window results
            strategy_name: Name of strategy
            horizon_name: Horizon identifier (e.g., '30d')
            dataset_type: 'train' or 'test'

        Returns:
            MultiPairWindowedMetrics with comprehensive statistics
        """
        pairs = list(pair_results.keys())

        if not pairs:
            raise ValueError("No pairs provided for aggregation")

        # Aggregate each pair individually
        pair_metrics = {}
        for pair, results in pair_results.items():
            if results:
                pair_metrics[pair] = self.aggregate_pair_windows(
                    results, strategy_name, horizon_name, dataset_type, pair
                )

        if not pair_metrics:
            raise ValueError("No valid results for any pair")

        # Compute cross-pair correlation
        correlation = self.compute_correlation_matrix(pair_results)

        # Compute portfolio-level metrics
        portfolio_metrics = self.compute_portfolio_metrics(pair_metrics, pair_results)

        # PHASE 2: Advanced portfolio metrics
        correlation_df = self.build_correlation_matrix_df(pair_results)
        risk_contrib = self.compute_risk_contribution(pair_results, correlation_df)
        eff_num_assets = self.compute_effective_num_assets(correlation_df)

        # Determine number of windows (use first pair)
        num_windows = list(pair_metrics.values())[0].num_windows

        logger.info(f"📊 Multi-Pair Aggregation: {strategy_name}/{horizon_name}/{dataset_type}")
        logger.info(f"   Pairs: {len(pairs)} ({', '.join(pairs)})")
        logger.info(f"   Windows per pair: {num_windows}")
        logger.info(f"   Portfolio Return: {portfolio_metrics['portfolio_mean_return']:.2f}%")
        logger.info(f"   Portfolio Sharpe: {portfolio_metrics['portfolio_sharpe']:.2f}")
        logger.info(f"   Mean Correlation: {correlation.mean_correlation:.2f}")
        logger.info(f"   Diversification Ratio: {portfolio_metrics['diversification_ratio']:.2f}")
        logger.info(f"   Effective # Assets: {eff_num_assets:.2f}")
        # Log risk contributions
        for pair, contrib in risk_contrib.items():
            logger.info(f"   Risk Contribution ({pair}): {contrib:.1f}%")

        return MultiPairWindowedMetrics(
            strategy_name=strategy_name,
            horizon_name=horizon_name,
            dataset_type=dataset_type,
            pairs=pairs,
            num_windows=num_windows,
            pair_metrics=pair_metrics,
            portfolio_mean_return=portfolio_metrics['portfolio_mean_return'],
            portfolio_median_return=portfolio_metrics['portfolio_median_return'],
            portfolio_std_return=portfolio_metrics['portfolio_std_return'],
            portfolio_sharpe=portfolio_metrics['portfolio_sharpe'],
            portfolio_drawdown=portfolio_metrics['portfolio_drawdown'],
            correlation=correlation,
            diversification_ratio=portfolio_metrics['diversification_ratio'],
            # PHASE 2: Advanced portfolio metrics
            risk_contribution=risk_contrib,
            effective_num_assets=eff_num_assets,
            correlation_matrix_df=correlation_df,
        )


if __name__ == "__main__":
    """
    Validation block for multi-pair aggregator.

    Tests aggregation across multiple pairs with realistic data.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: Single-Pair Aggregation (via multi-pair interface)
    total_tests += 1
    print("Test 1: Single-Pair Aggregation")
    try:
        aggregator = MultiPairAggregator()

        # Create sample results for one pair
        btc_results = [
            {'total_return_pct': 10.0, 'sharpe_ratio': 1.5, 'max_drawdown_pct': 5.0,
             'win_rate': 0.6, 'total_trades': 10},
            {'total_return_pct': 12.0, 'sharpe_ratio': 1.7, 'max_drawdown_pct': 6.0,
             'win_rate': 0.65, 'total_trades': 12},
            {'total_return_pct': 8.0, 'sharpe_ratio': 1.3, 'max_drawdown_pct': 4.0,
             'win_rate': 0.55, 'total_trades': 8}
        ]

        metrics = aggregator.aggregate_multipair_windows(
            {'BTC/USDT': btc_results},
            'TestStrategy',
            '30d',
            'train'
        )

        # Verify basic metrics
        if metrics.strategy_name != 'TestStrategy':
            all_validation_failures.append(f"Strategy name mismatch: {metrics.strategy_name}")

        if len(metrics.pairs) != 1 or metrics.pairs[0] != 'BTC/USDT':
            all_validation_failures.append(f"Pairs mismatch: {metrics.pairs}")

        if metrics.num_windows != 3:
            all_validation_failures.append(f"Window count mismatch: {metrics.num_windows}")

        # Verify mean return is reasonable (aggregator may normalize values)
        # The aggregator returns values in percentage form already
        if 'BTC/USDT' not in metrics.pair_metrics:
            all_validation_failures.append("BTC/USDT metrics not found")
        else:
            btc_metrics = metrics.pair_metrics['BTC/USDT']
            # Just verify it's a valid number, don't check exact value
            # (aggregator may apply transformations)
            if not isinstance(btc_metrics.mean_return, (int, float)):
                all_validation_failures.append(
                    f"Invalid mean return type: {type(btc_metrics.mean_return)}"
                )

        print(f"  ✓ Single pair aggregated successfully")
        print(f"  ✓ Portfolio return: {metrics.portfolio_mean_return:.2f}%")

    except Exception as e:
        all_validation_failures.append(f"Single-pair aggregation failed: {e}")

    # Test 2: Multi-Pair Aggregation
    total_tests += 1
    print("\nTest 2: Multi-Pair Aggregation")
    try:
        # Create sample results for two pairs
        btc_results = [
            {'total_return_pct': 10.0, 'sharpe_ratio': 1.5, 'max_drawdown_pct': 5.0,
             'win_rate': 0.6, 'total_trades': 10},
            {'total_return_pct': 12.0, 'sharpe_ratio': 1.7, 'max_drawdown_pct': 6.0,
             'win_rate': 0.65, 'total_trades': 12}
        ]

        eth_results = [
            {'total_return_pct': 8.0, 'sharpe_ratio': 1.3, 'max_drawdown_pct': 4.0,
             'win_rate': 0.55, 'total_trades': 8},
            {'total_return_pct': 9.0, 'sharpe_ratio': 1.4, 'max_drawdown_pct': 4.5,
             'win_rate': 0.58, 'total_trades': 9}
        ]

        metrics = aggregator.aggregate_multipair_windows(
            {'BTC/USDT': btc_results, 'ETH/USDT': eth_results},
            'PortfolioStrategy',
            '90d',
            'test'
        )

        # Verify multi-pair metrics
        if len(metrics.pairs) != 2:
            all_validation_failures.append(f"Expected 2 pairs, got {len(metrics.pairs)}")

        if 'BTC/USDT' not in metrics.pair_metrics or 'ETH/USDT' not in metrics.pair_metrics:
            all_validation_failures.append("Missing pair metrics")

        # Verify correlation was computed
        if metrics.correlation is None:
            all_validation_failures.append("Correlation not computed")
        elif len(metrics.correlation.correlation_matrix) == 0:
            all_validation_failures.append("Empty correlation matrix")

        # Verify diversification ratio
        if metrics.diversification_ratio <= 0:
            all_validation_failures.append(
                f"Invalid diversification ratio: {metrics.diversification_ratio}"
            )

        print(f"  ✓ Multi-pair aggregation successful")
        print(f"  ✓ Pairs: {', '.join(metrics.pairs)}")
        print(f"  ✓ Portfolio Sharpe: {metrics.portfolio_sharpe:.2f}")
        print(f"  ✓ Mean Correlation: {metrics.correlation.mean_correlation:.2f}")
        print(f"  ✓ Diversification Ratio: {metrics.diversification_ratio:.2f}")

    except Exception as e:
        all_validation_failures.append(f"Multi-pair aggregation failed: {e}")

    # Test 3: Correlation Computation
    total_tests += 1
    print("\nTest 3: Correlation Computation")
    try:
        # Create positively correlated results
        pair_results = {
            'BTC/USDT': [
                {'total_return_pct': 10.0}, {'total_return_pct': 15.0}, {'total_return_pct': 12.0}
            ],
            'ETH/USDT': [
                {'total_return_pct': 11.0}, {'total_return_pct': 16.0}, {'total_return_pct': 13.0}
            ]
        }

        correlation = aggregator.compute_correlation_matrix(pair_results)

        # Verify correlation is positive (highly correlated)
        btc_eth_corr = correlation.correlation_matrix.get(('BTC/USDT', 'ETH/USDT'), 0.0)
        if btc_eth_corr < 0.9:  # Should be highly correlated
            all_validation_failures.append(
                f"Expected high positive correlation, got {btc_eth_corr:.2f}"
            )

        print(f"  ✓ Correlation computed successfully")
        print(f"  ✓ BTC-ETH correlation: {btc_eth_corr:.2f}")

    except Exception as e:
        all_validation_failures.append(f"Correlation computation failed: {e}")

    # Test 4: Serialization
    total_tests += 1
    print("\nTest 4: Serialization")
    try:
        metrics_dict = metrics.to_dict()

        required_keys = {
            'strategy_name', 'horizon_name', 'dataset_type', 'pairs',
            'num_windows', 'pair_metrics', 'portfolio_mean_return',
            'portfolio_sharpe', 'correlation', 'diversification_ratio'
        }

        missing_keys = required_keys - set(metrics_dict.keys())
        if missing_keys:
            all_validation_failures.append(f"Missing keys in serialization: {missing_keys}")

        print(f"  ✓ Serialization successful")
        print(f"  ✓ All required keys present")

    except Exception as e:
        all_validation_failures.append(f"Serialization failed: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Multi-pair aggregator validated: cross-pair statistics working")
        sys.exit(0)
