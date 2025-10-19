"""
Strategy Factory Pattern

Centralized factory for creating and initializing trading strategies with
proper validation, logging, and error handling.

**Purpose**: Eliminate scattered strategy instantiation code and provide
a single, well-tested entry point for strategy creation.

**Third-party packages**:
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Usage**:
```python
from crypto_trader.factories import StrategyFactory

# Create with default config
strategy = StrategyFactory.create("SMA_Crossover")

# Create with custom config
strategy = StrategyFactory.create(
    "RSI_MeanReversion",
    config={"rsi_period": 14, "oversold": 30, "overbought": 70}
)

# Create with validation callback
def on_created(strategy):
    print(f"Strategy {strategy.name} ready!")

strategy = StrategyFactory.create(
    "MACD_Momentum",
    config={},
    on_created=on_created
)
```

**Expected Output**:
- Fully initialized strategy instance
- Logged creation with parameters
- Validated configuration
- Ready to generate signals
"""

from typing import Any, Callable, Dict, Optional
from loguru import logger

from crypto_trader.strategies.base import BaseStrategy
from crypto_trader.strategies.registry import get_registry


class StrategyFactory:
    """
    Factory for creating trading strategy instances.

    Provides centralized strategy creation with validation,
    logging, and lifecycle hooks.
    """

    @staticmethod
    def create(
        name: str,
        config: Optional[Dict[str, Any]] = None,
        validate: bool = True,
        on_created: Optional[Callable[[BaseStrategy], None]] = None
    ) -> BaseStrategy:
        """
        Create and initialize a trading strategy.

        Args:
            name: Strategy name from registry
            config: Configuration parameters (default: {})
            validate: Whether to validate config before creation
            on_created: Optional callback after successful creation

        Returns:
            Initialized strategy instance ready to use

        Raises:
            KeyError: If strategy name not found in registry
            ValueError: If configuration is invalid
            RuntimeError: If initialization fails

        Example:
            >>> strategy = StrategyFactory.create(
            ...     "SMA_Crossover",
            ...     config={"fast_period": 10, "slow_period": 20}
            ... )
            >>> strategy.name
            'SMA_Crossover'
        """
        if config is None:
            config = {}

        logger.debug(f"Creating strategy: {name}")

        # Get strategy class from registry
        registry = get_registry()
        try:
            strategy_class = registry.get_strategy(name)
        except KeyError:
            available = ", ".join(registry.list_strategies().keys())
            logger.error(f"Strategy '{name}' not found. Available: {available}")
            raise KeyError(
                f"Strategy '{name}' not found in registry. "
                f"Available strategies: {available}"
            )

        # Validate configuration if requested
        if validate and config:
            StrategyFactory._validate_config(name, config, strategy_class)

        # Create instance
        try:
            strategy = strategy_class(name=name, config=config)
            logger.debug(f"Instantiated {name}: {strategy_class.__name__}")
        except Exception as e:
            logger.error(f"Failed to instantiate {name}: {e}")
            raise RuntimeError(f"Strategy instantiation failed: {e}") from e

        # Initialize with config
        try:
            strategy.initialize(config)
            logger.info(
                f"✓ Created strategy: {name} with params {list(config.keys())}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            raise RuntimeError(f"Strategy initialization failed: {e}") from e

        # Invoke lifecycle hook if provided
        if on_created:
            try:
                on_created(strategy)
            except Exception as e:
                logger.warning(f"on_created callback failed: {e}")

        return strategy

    @staticmethod
    def _validate_config(
        name: str,
        config: Dict[str, Any],
        strategy_class: type
    ) -> None:
        """
        Validate strategy configuration before creation.

        Args:
            name: Strategy name
            config: Configuration to validate
            strategy_class: Strategy class for inspection

        Raises:
            ValueError: If configuration is invalid
        """
        # Basic validation: check for None values
        none_keys = [k for k, v in config.items() if v is None]
        if none_keys:
            raise ValueError(
                f"Configuration for {name} contains None values: {none_keys}"
            )

        # Check for empty strings
        empty_keys = [k for k, v in config.items() if isinstance(v, str) and not v]
        if empty_keys:
            raise ValueError(
                f"Configuration for {name} contains empty strings: {empty_keys}"
            )

        logger.debug(f"Configuration validated for {name}")

    @staticmethod
    def create_batch(
        strategy_names: list[str],
        config_dict: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, BaseStrategy]:
        """
        Create multiple strategies in batch.

        Args:
            strategy_names: List of strategy names to create
            config_dict: Dict mapping strategy names to their configs

        Returns:
            Dictionary mapping strategy names to instances

        Example:
            >>> strategies = StrategyFactory.create_batch(
            ...     ["SMA_Crossover", "RSI_MeanReversion"],
            ...     config_dict={"SMA_Crossover": {"fast_period": 10}}
            ... )
            >>> len(strategies)
            2
        """
        if config_dict is None:
            config_dict = {}

        strategies = {}
        failures = []

        for name in strategy_names:
            try:
                config = config_dict.get(name, {})
                strategy = StrategyFactory.create(name, config)
                strategies[name] = strategy
            except Exception as e:
                logger.warning(f"Skipping {name}: {e}")
                failures.append((name, str(e)))

        if failures:
            logger.warning(
                f"Created {len(strategies)}/{len(strategy_names)} strategies. "
                f"Failures: {len(failures)}"
            )
        else:
            logger.success(f"Created all {len(strategies)} strategies successfully")

        return strategies

    @staticmethod
    def get_default_config(name: str) -> Dict[str, Any]:
        """
        Get default configuration for a strategy.

        Args:
            name: Strategy name

        Returns:
            Default configuration dictionary

        Example:
            >>> config = StrategyFactory.get_default_config("SMA_Crossover")
            >>> "fast_period" in config
            True
        """
        # Create temporary instance to get defaults
        registry = get_registry()
        strategy_class = registry.get_strategy(name)

        try:
            temp_strategy = strategy_class(name=name)
            temp_strategy.initialize({})
            return temp_strategy.get_parameters()
        except Exception as e:
            logger.warning(f"Could not get defaults for {name}: {e}")
            return {}


if __name__ == "__main__":
    """
    Validation block for StrategyFactory.
    Tests factory methods with real strategy classes.
    """
    import sys
    from crypto_trader.strategies.registry import get_registry

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating StrategyFactory...\n")

    # Load strategies for testing
    registry = get_registry()
    from pathlib import Path
    strategies_path = Path(__file__).parent.parent / "strategies" / "library"
    if strategies_path.exists():
        registry.load_from_directory(strategies_path, recursive=False)
        print(f"Loaded {len(registry.list_strategies())} strategies for testing\n")

    # Test 1: Create strategy with default config
    total_tests += 1
    print("Test 1: Create strategy with default config")
    try:
        strategy = StrategyFactory.create("SMA_Crossover")

        if not isinstance(strategy, BaseStrategy):
            all_validation_failures.append(
                f"Expected BaseStrategy instance, got {type(strategy)}"
            )
        elif strategy.name != "SMA_Crossover":
            all_validation_failures.append(
                f"Expected name 'SMA_Crossover', got '{strategy.name}'"
            )
        else:
            print("  ✓ Strategy created successfully")
            print(f"  ✓ Name: {strategy.name}")
            print(f"  ✓ Parameters: {list(strategy.get_parameters().keys())}")
    except Exception as e:
        all_validation_failures.append(f"Test 1 failed: {e}")

    # Test 2: Create with custom config
    total_tests += 1
    print("\nTest 2: Create with custom config")
    try:
        config = {"fast_period": 5, "slow_period": 15}
        strategy = StrategyFactory.create("SMA_Crossover", config=config)

        params = strategy.get_parameters()
        if params.get("fast_period") != 5:
            all_validation_failures.append(
                f"Expected fast_period=5, got {params.get('fast_period')}"
            )
        elif params.get("slow_period") != 15:
            all_validation_failures.append(
                f"Expected slow_period=15, got {params.get('slow_period')}"
            )
        else:
            print("  ✓ Custom config applied")
            print(f"  ✓ fast_period: {params['fast_period']}")
            print(f"  ✓ slow_period: {params['slow_period']}")
    except Exception as e:
        all_validation_failures.append(f"Test 2 failed: {e}")

    # Test 3: Invalid strategy name
    total_tests += 1
    print("\nTest 3: Invalid strategy name")
    try:
        error_raised = False
        try:
            StrategyFactory.create("NonExistentStrategy")
        except KeyError:
            error_raised = True

        if not error_raised:
            all_validation_failures.append("Expected KeyError for invalid name")
        else:
            print("  ✓ KeyError raised correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 3 failed: {e}")

    # Test 4: Validation with None values
    total_tests += 1
    print("\nTest 4: Config validation with None values")
    try:
        error_raised = False
        try:
            StrategyFactory.create(
                "SMA_Crossover",
                config={"fast_period": None},
                validate=True
            )
        except ValueError:
            error_raised = True

        if not error_raised:
            all_validation_failures.append("Expected ValueError for None value")
        else:
            print("  ✓ ValueError raised for None values")
    except Exception as e:
        all_validation_failures.append(f"Test 4 failed: {e}")

    # Test 5: Batch creation
    total_tests += 1
    print("\nTest 5: Batch strategy creation")
    try:
        strategies = StrategyFactory.create_batch([
            "SMA_Crossover",
            "RSI_MeanReversion"
        ])

        if len(strategies) != 2:
            all_validation_failures.append(
                f"Expected 2 strategies, got {len(strategies)}"
            )
        elif "SMA_Crossover" not in strategies:
            all_validation_failures.append("SMA_Crossover not in batch result")
        elif "RSI_MeanReversion" not in strategies:
            all_validation_failures.append("RSI_MeanReversion not in batch result")
        else:
            print(f"  ✓ Created {len(strategies)} strategies")
            for name in strategies:
                print(f"    - {name}")
    except Exception as e:
        all_validation_failures.append(f"Test 5 failed: {e}")

    # Test 6: Get default config
    total_tests += 1
    print("\nTest 6: Get default configuration")
    try:
        defaults = StrategyFactory.get_default_config("SMA_Crossover")

        if not isinstance(defaults, dict):
            all_validation_failures.append(
                f"Expected dict, got {type(defaults)}"
            )
        elif not defaults:
            all_validation_failures.append("Default config is empty")
        else:
            print("  ✓ Retrieved default config")
            print(f"  ✓ Keys: {list(defaults.keys())}")
    except Exception as e:
        all_validation_failures.append(f"Test 6 failed: {e}")

    # Test 7: Lifecycle hook
    total_tests += 1
    print("\nTest 7: on_created lifecycle hook")
    try:
        # Use a mutable container to track hook execution from inner function
        hook_state = {"called": False}

        def test_hook(strategy):
            hook_state["called"] = True

        strategy = StrategyFactory.create(
            "SMA_Crossover",
            on_created=test_hook
        )

        if not hook_state["called"]:
            all_validation_failures.append("Lifecycle hook was not called")
        else:
            print("  ✓ Lifecycle hook executed")
    except Exception as e:
        all_validation_failures.append(f"Test 7 failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("StrategyFactory is validated and ready for use")
        sys.exit(0)
