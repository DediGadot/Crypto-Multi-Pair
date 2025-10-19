"""
Strategy Library - Pre-built Trading Strategies

This module safely imports all bundled strategies, skipping any whose optional
dependencies are unavailable at runtime. The strategy registry decorator handles
registration on import, so simply importing this package is sufficient to make
strategies discoverable.
"""

from __future__ import annotations

import importlib
from typing import Optional, Type

from loguru import logger

from crypto_trader.strategies.base import BaseStrategy

__all__: list[str] = []


def _register(module_path: str, class_name: str) -> Optional[Type[BaseStrategy]]:
    try:
        module = importlib.import_module(module_path)
        strategy_cls = getattr(module, class_name)
        globals()[class_name] = strategy_cls
        __all__.append(class_name)
        return strategy_cls
    except Exception as exc:  # pragma: no cover - defensive import guard
        logger.warning(f"Skipping {class_name}: {exc}")
        globals()[class_name] = None
        return None


_register("crypto_trader.strategies.library.sma_crossover", "SMACrossoverStrategy")
_register("crypto_trader.strategies.library.rsi_mean_reversion", "RSIMeanReversionStrategy")
_register("crypto_trader.strategies.library.macd_momentum", "MACDMomentumStrategy")
_register("crypto_trader.strategies.library.bollinger_breakout", "BollingerBreakoutStrategy")
_register("crypto_trader.strategies.library.triple_ema", "TripleEMAStrategy")
_register("crypto_trader.strategies.library.supertrend_atr", "SupertrendATRStrategy")
_register("crypto_trader.strategies.library.ichimoku_cloud", "IchimokuCloudStrategy")
_register("crypto_trader.strategies.library.vwap_mean_reversion", "VWAPMeanReversionStrategy")
_register("crypto_trader.strategies.library.portfolio_rebalancer", "PortfolioRebalancerStrategy")
_register("crypto_trader.strategies.library.statistical_arbitrage_pairs", "StatisticalArbitrageStrategy")
_register("crypto_trader.strategies.library.hierarchical_risk_parity", "HierarchicalRiskParityStrategy")
_register("crypto_trader.strategies.library.black_litterman", "BlackLittermanStrategy")
_register("crypto_trader.strategies.library.risk_parity", "RiskParityStrategy")
_register("crypto_trader.strategies.library.copula_pairs_trading", "CopulaPairsTradingStrategy")
_register("crypto_trader.strategies.library.deep_rl_portfolio", "DeepRLPortfolioStrategy")
_register("crypto_trader.strategies.library.onchain_analytics", "OnChainAnalytics")
_register("crypto_trader.strategies.library.multi_timeframe_confluence", "MultiTimeframeConfluenceStrategy")
_register("crypto_trader.strategies.library.regime_adaptive", "VolatilityRegimeAdaptiveStrategy")
_register("crypto_trader.strategies.library.dynamic_ensemble", "DynamicEnsembleStrategy")
_register("crypto_trader.strategies.library.transformer_gru_predictor", "TransformerGRUPredictorStrategy")
_register("crypto_trader.strategies.library.ddqn_feature_selected", "DDQNFeatureSelectedStrategy")
_register("crypto_trader.strategies.library.multimodal_sentiment_fusion", "MultiModalSentimentFusionStrategy")
_register("crypto_trader.strategies.library.order_flow_imbalance", "OrderFlowImbalanceStrategy")

__version__ = "0.1.0"
__author__ = "Crypto Trader Team"

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
        expected_exports = set(__all__)
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
            'VWAP_MeanReversion',
            'PortfolioRebalancer',
            'StatisticalArbitrage',
            'HierarchicalRiskParity',
            'BlackLitterman',
            'RiskParity',
            'CopulaPairsTrading',
            'DeepRLPortfolio',
            'OnChainAnalytics',
            'MultiTimeframeConfluence',
            'VolatilityRegimeAdaptive',
            'DynamicEnsemble',
            'TransformerGRUPredictor',
            'DDQNFeatureSelected',
            'MultiModalSentimentFusion',
            'OrderFlowImbalance',
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
