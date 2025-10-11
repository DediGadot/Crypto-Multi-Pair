"""
Trading Strategy Framework

This package provides a modular, plugin-based framework for implementing
and managing trading strategies in the crypto trading system.

**Purpose**: Provide a unified interface for strategy development with
automatic discovery, registration, and configuration loading capabilities.

**Key Components**:
- BaseStrategy: Abstract base class for all strategies
- StrategyRegistry: Plugin system for strategy management
- StrategyLoader: Configuration loading and strategy instantiation
- SignalType: Enumeration of signal types (BUY, SELL, HOLD)

**Usage Example**:
```python
from crypto_trader.strategies import (
    BaseStrategy,
    register_strategy,
    load_strategies_from_yaml,
    SignalType
)

# Define a strategy
@register_strategy(tags=["momentum"])
class MyStrategy(BaseStrategy):
    def initialize(self, config):
        self.period = config.get("period", 20)

    def generate_signals(self, data):
        # Strategy logic here
        return signals

    def get_parameters(self):
        return {"period": self.period}

# Load strategies from config
strategies = load_strategies_from_yaml("config/strategies.yaml")

# Use a strategy
strategy = strategies["MyStrategy"]
signals = strategy.generate_signals(market_data)
```

**Architecture**:
This framework uses a plugin pattern that allows strategies to be:
1. Defined as simple classes inheriting from BaseStrategy
2. Automatically discovered and registered using decorators
3. Loaded from YAML configuration files
4. Instantiated with validated parameters

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- pydantic: https://docs.pydantic.dev/latest/
- loguru: https://loguru.readthedocs.io/en/stable/
"""

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import (
    StrategyRegistry,
    get_registry,
    get_strategy,
    list_strategies,
    register_strategy,
)
from crypto_trader.strategies.loader import (
    StrategyConfig,
    StrategyLoader,
    load_strategies_from_yaml,
)

__all__ = [
    # Base classes and enums
    "BaseStrategy",
    "SignalType",
    # Registry functions
    "StrategyRegistry",
    "get_registry",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    # Loader classes and functions
    "StrategyConfig",
    "StrategyLoader",
    "load_strategies_from_yaml",
]

# Version information
__version__ = "0.1.0"
__author__ = "Crypto Trader Team"


if __name__ == "__main__":
    """
    Validation block for the strategies package.
    Tests the public API exports and basic integration.
    """
    import sys
    from typing import Dict, Any
    import pandas as pd

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    # Test 1: Verify all exports are available
    total_tests += 1
    try:
        expected_exports = {
            'BaseStrategy', 'SignalType', 'StrategyRegistry',
            'get_registry', 'get_strategy', 'list_strategies',
            'register_strategy', 'StrategyConfig', 'StrategyLoader',
            'load_strategies_from_yaml'
        }
        actual_exports = set(__all__)

        if actual_exports != expected_exports:
            all_validation_failures.append(
                f"Exports mismatch: Expected {expected_exports}, got {actual_exports}"
            )
    except Exception as e:
        all_validation_failures.append(f"Export verification failed: {e}")

    # Test 2: Import and use BaseStrategy
    total_tests += 1
    try:
        class TestIntegrationStrategy(BaseStrategy):
            """Test strategy for integration validation."""

            def initialize(self, config: Dict[str, Any]) -> None:
                self.config.update(config)

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                signals = pd.DataFrame({
                    'timestamp': data['timestamp'] if 'timestamp' in data.columns else pd.date_range('2024-01-01', periods=len(data)),
                    'signal': [SignalType.HOLD.value] * len(data),
                    'confidence': [0.5] * len(data),
                    'metadata': [{}] * len(data)
                })
                return signals

            def get_parameters(self) -> Dict[str, Any]:
                return self.config

        strategy = TestIntegrationStrategy(name="test", config={"param": "value"})
        if not isinstance(strategy, BaseStrategy):
            all_validation_failures.append(
                "BaseStrategy integration: Instance check failed"
            )
    except Exception as e:
        all_validation_failures.append(f"BaseStrategy integration failed: {e}")

    # Test 3: Test SignalType enum
    total_tests += 1
    try:
        if SignalType.BUY.value != "BUY":
            all_validation_failures.append(
                f"SignalType.BUY: Expected 'BUY', got {SignalType.BUY.value}"
            )
        if SignalType.SELL.value != "SELL":
            all_validation_failures.append(
                f"SignalType.SELL: Expected 'SELL', got {SignalType.SELL.value}"
            )
        if SignalType.HOLD.value != "HOLD":
            all_validation_failures.append(
                f"SignalType.HOLD: Expected 'HOLD', got {SignalType.HOLD.value}"
            )
    except Exception as e:
        all_validation_failures.append(f"SignalType verification failed: {e}")

    # Test 4: Test registry integration
    total_tests += 1
    try:
        # Clear registry for clean test
        registry = get_registry()
        registry.clear()

        # Register strategy using decorator
        @register_strategy(tags=["test", "integration"])
        class DecoratedIntegration(BaseStrategy):
            """Decorated strategy for integration test."""

            def initialize(self, config: Dict[str, Any]) -> None:
                pass

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame()

            def get_parameters(self) -> Dict[str, Any]:
                return {}

        # Verify registration
        if "DecoratedIntegration" not in registry:
            all_validation_failures.append(
                "Registry integration: DecoratedIntegration not registered"
            )

        # Retrieve strategy
        retrieved = get_strategy("DecoratedIntegration")
        if retrieved is not DecoratedIntegration:
            all_validation_failures.append(
                "Registry integration: Retrieved wrong strategy class"
            )
    except Exception as e:
        all_validation_failures.append(f"Registry integration failed: {e}")

    # Test 5: Test list_strategies
    total_tests += 1
    try:
        all_strats = list_strategies()
        if "DecoratedIntegration" not in all_strats:
            all_validation_failures.append(
                "list_strategies: DecoratedIntegration not in list"
            )

        # Filter by tag
        test_strats = list_strategies(tags=["test"])
        if len(test_strats) == 0:
            all_validation_failures.append(
                "list_strategies with tags: Expected at least 1 strategy"
            )
    except Exception as e:
        all_validation_failures.append(f"list_strategies failed: {e}")

    # Test 6: Test StrategyLoader instantiation
    total_tests += 1
    try:
        loader = StrategyLoader()
        if not isinstance(loader, StrategyLoader):
            all_validation_failures.append(
                "StrategyLoader instantiation: Type check failed"
            )
    except Exception as e:
        all_validation_failures.append(f"StrategyLoader instantiation failed: {e}")

    # Test 7: Test StrategyConfig model
    total_tests += 1
    try:
        config = StrategyConfig(
            name="TestConfig",
            class_name="DecoratedIntegration",
            enabled=True,
            parameters={"key": "value"}
        )

        if config.name != "TestConfig":
            all_validation_failures.append(
                f"StrategyConfig: Expected name='TestConfig', got {config.name}"
            )
        if config.class_name != "DecoratedIntegration":
            all_validation_failures.append(
                f"StrategyConfig: Expected class_name='DecoratedIntegration', got {config.class_name}"
            )
    except Exception as e:
        all_validation_failures.append(f"StrategyConfig creation failed: {e}")

    # Test 8: Test strategy instantiation from loader
    total_tests += 1
    try:
        config = StrategyConfig(
            name="LoaderTest",
            class_name="DecoratedIntegration",
            enabled=True
        )

        strategy_instance = loader.instantiate_strategy(config)
        if not isinstance(strategy_instance, BaseStrategy):
            all_validation_failures.append(
                "Loader instantiation: Expected BaseStrategy instance"
            )
        if strategy_instance.name != "LoaderTest":
            all_validation_failures.append(
                f"Loader instantiation: Expected name='LoaderTest', got {strategy_instance.name}"
            )
    except Exception as e:
        all_validation_failures.append(f"Loader strategy instantiation failed: {e}")

    # Test 9: Test complete workflow
    total_tests += 1
    try:
        # Create a simple strategy
        @register_strategy(name="WorkflowTest")
        class WorkflowStrategy(BaseStrategy):
            """Strategy for workflow test."""

            def initialize(self, config: Dict[str, Any]) -> None:
                self.threshold = config.get("threshold", 0.5)

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=3),
                    'signal': [SignalType.BUY.value, SignalType.HOLD.value, SignalType.SELL.value],
                    'confidence': [0.8, 0.5, 0.9],
                    'metadata': [{}, {}, {}]
                })

            def get_parameters(self) -> Dict[str, Any]:
                return {"threshold": self.threshold}

        # Instantiate and use
        workflow_config = StrategyConfig(
            name="workflow_instance",
            class_name="WorkflowTest",
            parameters={"threshold": 0.7}
        )

        workflow_strategy = loader.instantiate_strategy(workflow_config)

        # Generate signals
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })

        signals = workflow_strategy.generate_signals(test_data)

        if 'signal' not in signals.columns:
            all_validation_failures.append(
                "Workflow test: signals missing 'signal' column"
            )
        if len(signals) != 3:
            all_validation_failures.append(
                f"Workflow test: Expected 3 signals, got {len(signals)}"
            )

        params = workflow_strategy.get_parameters()
        if params.get("threshold") != 0.7:
            all_validation_failures.append(
                f"Workflow test: Expected threshold=0.7, got {params.get('threshold')}"
            )
    except Exception as e:
        all_validation_failures.append(f"Complete workflow test failed: {e}")

    # Test 10: Version information
    total_tests += 1
    try:
        if not __version__:
            all_validation_failures.append("Version information: __version__ is empty")
        if not __author__:
            all_validation_failures.append("Version information: __author__ is empty")
    except Exception as e:
        all_validation_failures.append(f"Version information check failed: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Strategy framework package is validated and ready for use")
        print(f"Version: {__version__}")
        sys.exit(0)
