"""
Strategy Library - Pre-built Trading Strategies

This module contains a collection of pre-built, validated trading strategies
for cryptocurrency trading. All strategies are automatically registered with
the global strategy registry when this module is imported.

**Purpose**: Provide a library of ready-to-use trading strategies that implement
common technical analysis approaches for crypto markets.

**Available Strategies**:
1. SMA Crossover - Simple Moving Average crossover (Golden/Death Cross)
2. RSI Mean Reversion - RSI oversold/overbought mean reversion
3. MACD Momentum - MACD signal line crossover momentum
4. Bollinger Breakout - Bollinger Bands volatility breakout
5. Triple EMA - Triple EMA trend filter with reduced lag

**SOTA 2024 Strategies**:
6. Supertrend ATR - Advanced trend following with ATR-based stops and RSI confirmation
7. Ichimoku Cloud - Multi-dimensional Japanese indicator system for trend and momentum
8. VWAP Mean Reversion - Volume-weighted price action with mean reversion around VWAP

**Advanced Strategies**:
9. Statistical Arbitrage - Regime-aware pairs trading with cointegration and HMM

**SOTA 2025 Portfolio Strategies (Phase 1)**:
10. Hierarchical Risk Parity (HRP) - Hierarchical clustering portfolio without covariance inversion
11. Black-Litterman - Bayesian asset allocation combining market equilibrium with views
12. Risk Parity - Equal risk contribution with optional kurtosis minimization

**SOTA 2025 Advanced Strategies (Phase 2)**:
13. Copula Pairs Trading - Tail dependency modeling for pairs trading
14. Deep RL Portfolio - Deep reinforcement learning with PPO for portfolio management

**Usage Example**:
```python
from crypto_trader.strategies.library import (
    SMACrossoverStrategy,
    RSIMeanReversionStrategy,
    MACDMomentumStrategy,
    BollingerBreakoutStrategy,
    TripleEMAStrategy
)

# Get strategy from registry
from crypto_trader.strategies import get_strategy

SMAStrategy = get_strategy("SMA_Crossover")
strategy = SMAStrategy(name="my_sma", config={"fast_period": 50, "slow_period": 200})
strategy.initialize(strategy.config)

# Generate signals
signals = strategy.generate_signals(market_data)
```

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- pandas_ta: https://github.com/twopirllc/pandas-ta
- loguru: https://loguru.readthedocs.io/en/stable/
"""

from crypto_trader.strategies.library.sma_crossover import SMACrossoverStrategy
from crypto_trader.strategies.library.rsi_mean_reversion import RSIMeanReversionStrategy
from crypto_trader.strategies.library.macd_momentum import MACDMomentumStrategy
from crypto_trader.strategies.library.bollinger_breakout import BollingerBreakoutStrategy
from crypto_trader.strategies.library.triple_ema import TripleEMAStrategy
from crypto_trader.strategies.library.supertrend_atr import SupertrendATRStrategy
from crypto_trader.strategies.library.ichimoku_cloud import IchimokuCloudStrategy
from crypto_trader.strategies.library.vwap_mean_reversion import VWAPMeanReversionStrategy
from crypto_trader.strategies.library.portfolio_rebalancer import PortfolioRebalancerStrategy
from crypto_trader.strategies.library.statistical_arbitrage_pairs import StatisticalArbitrageStrategy
from crypto_trader.strategies.library.hierarchical_risk_parity import HierarchicalRiskParityStrategy
from crypto_trader.strategies.library.black_litterman import BlackLittermanStrategy
from crypto_trader.strategies.library.risk_parity import RiskParityStrategy
from crypto_trader.strategies.library.copula_pairs_trading import CopulaPairsTradingStrategy
from crypto_trader.strategies.library.deep_rl_portfolio import DeepRLPortfolioStrategy
from crypto_trader.strategies.library.onchain_analytics import OnChainAnalytics  # ensure registration

__all__ = [
    "SMACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "MACDMomentumStrategy",
    "BollingerBreakoutStrategy",
    "TripleEMAStrategy",
    "SupertrendATRStrategy",
    "IchimokuCloudStrategy",
    "VWAPMeanReversionStrategy",
    "PortfolioRebalancerStrategy",
    "StatisticalArbitrageStrategy",
    "HierarchicalRiskParityStrategy",
    "BlackLittermanStrategy",
    "RiskParityStrategy",
    "CopulaPairsTradingStrategy",
    "DeepRLPortfolioStrategy",
    "OnChainAnalytics",
]

# Version information
__version__ = "0.1.0"
__author__ = "Crypto Trader Team"


if __name__ == "__main__":
    """
    Validation block for the strategy library package.
    Tests that all strategies are properly exported and registered.
    """
    import sys
    from crypto_trader.strategies import get_registry

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("="*70)
    print("Strategy Library Validation")
    print("="*70)

    # Test 1: Verify all exports are available
    total_tests += 1
    try:
        expected_exports = {
            'SMACrossoverStrategy',
            'RSIMeanReversionStrategy',
            'MACDMomentumStrategy',
            'BollingerBreakoutStrategy',
            'TripleEMAStrategy',
            'SupertrendATRStrategy',
            'IchimokuCloudStrategy',
            'VWAPMeanReversionStrategy'
        }
        actual_exports = set(__all__)

        if actual_exports != expected_exports:
            all_validation_failures.append(
                f"Test 1: Exports mismatch - Expected {expected_exports}, got {actual_exports}"
            )
        else:
            print("✓ Test 1 PASSED: All strategy classes exported")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Verify all strategies are registered
    total_tests += 1
    try:
        registry = get_registry()
        expected_strategies = {
            'SMA_Crossover',
            'RSI_MeanReversion',
            'MACD_Momentum',
            'BollingerBreakout',
            'TripleEMA',
            'Supertrend_ATR',
            'Ichimoku_Cloud',
            'VWAP_MeanReversion'
        }

        registered_strategies = set(registry.get_strategy_names())

        # Check if all expected strategies are registered
        missing = expected_strategies - registered_strategies
        if missing:
            all_validation_failures.append(
                f"Test 2: Missing strategies in registry: {missing}"
            )
        else:
            print(f"✓ Test 2 PASSED: All {len(expected_strategies)} strategies registered")
            for strategy_name in sorted(expected_strategies):
                print(f"  - {strategy_name}")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Verify strategy classes can be instantiated
    total_tests += 1
    try:
        strategies_to_test = [
            (SMACrossoverStrategy, "SMA_Crossover"),
            (RSIMeanReversionStrategy, "RSI_MeanReversion"),
            (MACDMomentumStrategy, "MACD_Momentum"),
            (BollingerBreakoutStrategy, "BollingerBreakout"),
            (TripleEMAStrategy, "TripleEMA"),
            (SupertrendATRStrategy, "Supertrend_ATR"),
            (IchimokuCloudStrategy, "Ichimoku_Cloud"),
            (VWAPMeanReversionStrategy, "VWAP_MeanReversion")
        ]

        instantiation_failures = []
        for StrategyClass, expected_name in strategies_to_test:
            try:
                strategy = StrategyClass()
                if strategy.name != expected_name:
                    instantiation_failures.append(
                        f"{StrategyClass.__name__}: Expected name '{expected_name}', "
                        f"got '{strategy.name}'"
                    )
            except Exception as e:
                instantiation_failures.append(
                    f"{StrategyClass.__name__}: Failed to instantiate - {e}"
                )

        if instantiation_failures:
            all_validation_failures.append(
                f"Test 3: Strategy instantiation failures:\n  " +
                "\n  ".join(instantiation_failures)
            )
        else:
            print(f"✓ Test 3 PASSED: All {len(strategies_to_test)} strategies can be instantiated")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Verify strategy metadata
    total_tests += 1
    try:
        registry = get_registry()
        metadata_failures = []

        for strategy_name in ['SMA_Crossover', 'RSI_MeanReversion', 'MACD_Momentum',
                              'BollingerBreakout', 'TripleEMA', 'Supertrend_ATR',
                              'Ichimoku_Cloud', 'VWAP_MeanReversion']:
            try:
                info = registry.get_strategy_info(strategy_name)

                # Check required metadata fields
                if 'class_name' not in info:
                    metadata_failures.append(f"{strategy_name}: Missing 'class_name'")
                if 'description' not in info:
                    metadata_failures.append(f"{strategy_name}: Missing 'description'")
                if 'tags' not in info:
                    metadata_failures.append(f"{strategy_name}: Missing 'tags'")
                elif not info['tags']:
                    metadata_failures.append(f"{strategy_name}: Tags list is empty")
            except Exception as e:
                metadata_failures.append(f"{strategy_name}: {e}")

        if metadata_failures:
            all_validation_failures.append(
                f"Test 4: Metadata failures:\n  " + "\n  ".join(metadata_failures)
            )
        else:
            print("✓ Test 4 PASSED: All strategies have complete metadata")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception raised: {e}")

    # Test 5: Verify strategy tags
    total_tests += 1
    try:
        registry = get_registry()
        expected_tags = {
            'SMA_Crossover': ['trend_following', 'moving_average', 'crossover'],
            'RSI_MeanReversion': ['mean_reversion', 'rsi', 'oscillator'],
            'MACD_Momentum': ['momentum', 'macd', 'crossover'],
            'BollingerBreakout': ['volatility', 'bollinger_bands', 'breakout'],
            'TripleEMA': ['trend_following', 'ema', 'crossover', 'trend_filter'],
            'Supertrend_ATR': ['trend_following', 'supertrend', 'volatility', 'rsi', 'sota_2024'],
            'Ichimoku_Cloud': ['trend_following', 'ichimoku', 'multi_timeframe', 'sota_2024'],
            'VWAP_MeanReversion': ['mean_reversion', 'vwap', 'volume', 'rsi', 'sota_2024']
        }

        tag_failures = []
        for strategy_name, expected_tag_list in expected_tags.items():
            info = registry.get_strategy_info(strategy_name)
            actual_tags = set(info['tags'])
            expected_tag_set = set(expected_tag_list)

            if actual_tags != expected_tag_set:
                tag_failures.append(
                    f"{strategy_name}: Expected tags {expected_tag_set}, got {actual_tags}"
                )

        if tag_failures:
            all_validation_failures.append(
                f"Test 5: Tag verification failures:\n  " + "\n  ".join(tag_failures)
            )
        else:
            print("✓ Test 5 PASSED: All strategies have correct tags")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Test filtering by tags
    total_tests += 1
    try:
        registry = get_registry()

        # Test filtering by 'trend_following' tag
        trend_strategies = registry.list_strategies(tags=['trend_following'])
        if 'SMA_Crossover' not in trend_strategies or 'TripleEMA' not in trend_strategies:
            all_validation_failures.append(
                "Test 6: Trend following filter should return SMA_Crossover and TripleEMA"
            )

        # Test filtering by 'mean_reversion' tag
        mean_rev_strategies = registry.list_strategies(tags=['mean_reversion'])
        if 'RSI_MeanReversion' not in mean_rev_strategies:
            all_validation_failures.append(
                "Test 6: Mean reversion filter should return RSI_MeanReversion"
            )

        # Test filtering by 'momentum' tag
        momentum_strategies = registry.list_strategies(tags=['momentum'])
        if 'MACD_Momentum' not in momentum_strategies:
            all_validation_failures.append(
                "Test 6: Momentum filter should return MACD_Momentum"
            )

        if not all_validation_failures or len([f for f in all_validation_failures if 'Test 6' in f]) == 0:
            print("✓ Test 6 PASSED: Tag-based filtering works correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 6: Exception raised: {e}")

    # Test 7: Verify strategy retrieval
    total_tests += 1
    try:
        registry = get_registry()
        retrieval_failures = []

        strategy_mapping = {
            'SMA_Crossover': SMACrossoverStrategy,
            'RSI_MeanReversion': RSIMeanReversionStrategy,
            'MACD_Momentum': MACDMomentumStrategy,
            'BollingerBreakout': BollingerBreakoutStrategy,
            'TripleEMA': TripleEMAStrategy,
            'Supertrend_ATR': SupertrendATRStrategy,
            'Ichimoku_Cloud': IchimokuCloudStrategy,
            'VWAP_MeanReversion': VWAPMeanReversionStrategy
        }

        for strategy_name, expected_class in strategy_mapping.items():
            retrieved_class = registry.get_strategy(strategy_name)
            if retrieved_class is not expected_class:
                retrieval_failures.append(
                    f"{strategy_name}: Expected {expected_class.__name__}, "
                    f"got {retrieved_class.__name__}"
                )

        if retrieval_failures:
            all_validation_failures.append(
                f"Test 7: Strategy retrieval failures:\n  " + "\n  ".join(retrieval_failures)
            )
        else:
            print("✓ Test 7 PASSED: All strategies can be retrieved correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Test 8: Verify version information
    total_tests += 1
    try:
        if not __version__:
            all_validation_failures.append("Test 8: __version__ is empty")
        if not __author__:
            all_validation_failures.append("Test 8: __author__ is empty")

        if not all_validation_failures or len([f for f in all_validation_failures if 'Test 8' in f]) == 0:
            print(f"✓ Test 8 PASSED: Version information present (v{__version__})")
    except Exception as e:
        all_validation_failures.append(f"Test 8: Exception raised: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        print()
        for failure in all_validation_failures:
            print(failure)
            print()
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print()
        print("Strategy Library Summary:")
        print(f"  - Total Strategies: 8")
        print(f"  - Trend Following: 4 (SMA_Crossover, TripleEMA, Supertrend_ATR, Ichimoku_Cloud)")
        print(f"  - Mean Reversion: 2 (RSI_MeanReversion, VWAP_MeanReversion)")
        print(f"  - Momentum: 1 (MACD_Momentum)")
        print(f"  - Volatility: 1 (BollingerBreakout)")
        print(f"  - SOTA 2024: 3 (Supertrend_ATR, Ichimoku_Cloud, VWAP_MeanReversion)")
        print()
        print("All strategies are validated and ready for use!")
        sys.exit(0)
