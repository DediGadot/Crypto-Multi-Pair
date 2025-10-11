"""
Strategy Registry for Plugin System

This module implements a strategy registry pattern that allows dynamic registration
and retrieval of trading strategies. It supports both explicit registration via
decorator and automatic discovery from directories.

**Purpose**: Provide a centralized registry for trading strategies with decorator-based
registration, dynamic loading, and strategy discovery capabilities.

**Key Features**:
- Decorator-based strategy registration
- Dynamic strategy loading from directories
- Strategy metadata management
- Thread-safe registry operations

**Third-party packages**:
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```python
@register_strategy
class MyStrategy(BaseStrategy):
    pass

strategy_class = registry.get_strategy("MyStrategy")
all_strategies = registry.list_strategies()
```

**Expected Output**:
```python
{
    "MyStrategy": {
        "class": MyStrategy,
        "module": "strategies.library.my_strategy",
        "description": "My custom strategy"
    }
}
```
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
from threading import Lock

from loguru import logger

from crypto_trader.strategies.base import BaseStrategy


class StrategyRegistry:
    """
    Registry for managing trading strategy plugins.

    This class provides a thread-safe registry for strategy classes,
    supporting both explicit registration and dynamic discovery.

    Attributes:
        _strategies: Dictionary mapping strategy names to metadata
        _lock: Thread lock for safe concurrent access
    """

    def __init__(self):
        """Initialize an empty strategy registry."""
        self._strategies: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        logger.debug("Strategy registry initialized")

    def register(
        self,
        strategy_class: Type[BaseStrategy],
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a strategy class in the registry.

        Args:
            strategy_class: The strategy class to register
            name: Optional custom name (defaults to class name)
            description: Optional strategy description
            tags: Optional list of tags for categorization

        Raises:
            TypeError: If strategy_class is not a BaseStrategy subclass
            ValueError: If strategy name already exists
        """
        # Validate strategy class
        if not inspect.isclass(strategy_class):
            raise TypeError(f"Expected a class, got {type(strategy_class)}")

        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(
                f"Strategy class must inherit from BaseStrategy, "
                f"got {strategy_class.__bases__}"
            )

        # Determine strategy name
        strategy_name = name or strategy_class.__name__

        # Thread-safe registration
        with self._lock:
            if strategy_name in self._strategies:
                raise ValueError(f"Strategy '{strategy_name}' is already registered")

            # Extract description from docstring if not provided
            if description is None:
                doc = inspect.getdoc(strategy_class)
                description = doc.split('\n')[0] if doc else "No description"

            # Store strategy metadata
            self._strategies[strategy_name] = {
                "class": strategy_class,
                "module": strategy_class.__module__,
                "description": description,
                "tags": tags or [],
            }

            logger.info(f"Registered strategy: {strategy_name}")
            logger.debug(f"Strategy metadata: {self._strategies[strategy_name]}")

    def unregister(self, name: str) -> None:
        """
        Remove a strategy from the registry.

        Args:
            name: Name of the strategy to remove

        Raises:
            KeyError: If strategy name not found
        """
        with self._lock:
            if name not in self._strategies:
                raise KeyError(f"Strategy '{name}' not found in registry")

            del self._strategies[name]
            logger.info(f"Unregistered strategy: {name}")

    def get_strategy(self, name: str) -> Type[BaseStrategy]:
        """
        Retrieve a strategy class by name.

        Args:
            name: Name of the strategy to retrieve

        Returns:
            The strategy class

        Raises:
            KeyError: If strategy name not found
        """
        with self._lock:
            if name not in self._strategies:
                available = ", ".join(self._strategies.keys())
                raise KeyError(
                    f"Strategy '{name}' not found. "
                    f"Available strategies: {available}"
                )

            strategy_class = self._strategies[name]["class"]
            logger.debug(f"Retrieved strategy: {name}")
            return strategy_class

    def list_strategies(
        self,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        List all registered strategies with their metadata.

        Args:
            tags: Optional list of tags to filter strategies

        Returns:
            Dictionary mapping strategy names to their metadata
        """
        with self._lock:
            if tags is None:
                return dict(self._strategies)

            # Filter by tags
            filtered = {
                name: metadata
                for name, metadata in self._strategies.items()
                if any(tag in metadata["tags"] for tag in tags)
            }
            return filtered

    def get_strategy_names(self) -> List[str]:
        """
        Get list of all registered strategy names.

        Returns:
            List of strategy names
        """
        with self._lock:
            return list(self._strategies.keys())

    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a strategy.

        Args:
            name: Name of the strategy

        Returns:
            Dictionary with strategy metadata (excluding the class object)

        Raises:
            KeyError: If strategy name not found
        """
        with self._lock:
            if name not in self._strategies:
                raise KeyError(f"Strategy '{name}' not found in registry")

            metadata = self._strategies[name].copy()
            # Replace class with class name for serialization
            metadata["class_name"] = metadata["class"].__name__
            del metadata["class"]
            return metadata

    def clear(self) -> None:
        """Clear all registered strategies."""
        with self._lock:
            count = len(self._strategies)
            self._strategies.clear()
            logger.warning(f"Cleared {count} strategies from registry")

    def load_from_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> int:
        """
        Dynamically load all strategies from a directory.

        This method scans a directory for Python modules, imports them,
        and automatically registers any BaseStrategy subclasses found.

        Args:
            directory: Path to directory containing strategy modules
            recursive: Whether to scan subdirectories recursively

        Returns:
            Number of strategies loaded

        Raises:
            ValueError: If directory doesn't exist
        """
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        loaded_count = 0
        logger.info(f"Loading strategies from: {directory}")

        # Get all Python modules in directory
        if recursive:
            modules = directory.rglob("*.py")
        else:
            modules = directory.glob("*.py")

        for module_path in modules:
            # Skip __init__.py and private modules
            if module_path.name.startswith("_"):
                continue

            try:
                # Import module dynamically
                module_name = module_path.stem
                spec = importlib.util.spec_from_file_location(module_name, module_path)

                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find all BaseStrategy subclasses in module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            issubclass(obj, BaseStrategy)
                            and obj is not BaseStrategy
                            and obj.__module__ == module_name
                        ):
                            # Register strategy if not already registered
                            strategy_name = obj.__name__
                            if strategy_name not in self._strategies:
                                self.register(obj)
                                loaded_count += 1

            except Exception as e:
                logger.error(f"Failed to load module {module_path}: {e}")
                continue

        logger.info(f"Loaded {loaded_count} strategies from {directory}")
        return loaded_count

    def __len__(self) -> int:
        """Return number of registered strategies."""
        return len(self._strategies)

    def __contains__(self, name: str) -> bool:
        """Check if a strategy is registered."""
        return name in self._strategies

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"StrategyRegistry(strategies={len(self._strategies)})"


# Global registry instance
_global_registry = StrategyRegistry()


def register_strategy(
    strategy_class: Optional[Type[BaseStrategy]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Callable:
    """
    Decorator for registering strategy classes.

    Can be used as @register_strategy or @register_strategy(name="custom_name")

    Args:
        strategy_class: The strategy class (when used without parentheses)
        name: Optional custom name for the strategy
        description: Optional description
        tags: Optional list of tags

    Returns:
        Decorated class or decorator function

    Examples:
        >>> @register_strategy
        ... class MyStrategy(BaseStrategy):
        ...     pass

        >>> @register_strategy(name="custom", tags=["momentum"])
        ... class AnotherStrategy(BaseStrategy):
        ...     pass
    """
    def decorator(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
        """Inner decorator function."""
        _global_registry.register(
            cls,
            name=name,
            description=description,
            tags=tags
        )
        return cls

    # Support both @register_strategy and @register_strategy()
    if strategy_class is None:
        return decorator
    else:
        return decorator(strategy_class)


def get_strategy(name: str) -> Type[BaseStrategy]:
    """
    Get a strategy class from the global registry.

    Args:
        name: Name of the strategy

    Returns:
        Strategy class

    Raises:
        KeyError: If strategy not found
    """
    return _global_registry.get_strategy(name)


def list_strategies(tags: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    List all strategies in the global registry.

    Args:
        tags: Optional list of tags to filter by

    Returns:
        Dictionary of strategy metadata
    """
    return _global_registry.list_strategies(tags=tags)


def get_registry() -> StrategyRegistry:
    """
    Get the global strategy registry instance.

    Returns:
        The global StrategyRegistry instance
    """
    return _global_registry


if __name__ == "__main__":
    """
    Validation block for StrategyRegistry.
    Tests registration, retrieval, and decorator functionality.
    """
    import sys
    from typing import Dict, Any
    import pandas as pd

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    # Test 1: Create a test registry
    total_tests += 1
    try:
        test_registry = StrategyRegistry()
        if len(test_registry) != 0:
            all_validation_failures.append(
                f"Empty registry: Expected length 0, got {len(test_registry)}"
            )
    except Exception as e:
        all_validation_failures.append(f"Registry creation failed: {e}")

    # Test 2: Create test strategy classes
    total_tests += 1
    try:
        class TestStrategy1(BaseStrategy):
            """Test strategy for validation."""

            def initialize(self, config: Dict[str, Any]) -> None:
                self.config.update(config)

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame()

            def get_parameters(self) -> Dict[str, Any]:
                return self.config

        class TestStrategy2(BaseStrategy):
            """Another test strategy."""

            def initialize(self, config: Dict[str, Any]) -> None:
                pass

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame()

            def get_parameters(self) -> Dict[str, Any]:
                return {}

        # Verify classes are created
        if not issubclass(TestStrategy1, BaseStrategy):
            all_validation_failures.append(
                "Test strategy creation: TestStrategy1 not a BaseStrategy subclass"
            )
    except Exception as e:
        all_validation_failures.append(f"Test strategy creation failed: {e}")

    # Test 3: Register strategies explicitly
    total_tests += 1
    try:
        test_registry.register(TestStrategy1)
        test_registry.register(TestStrategy2, name="CustomName", tags=["test"])

        if len(test_registry) != 2:
            all_validation_failures.append(
                f"Registry after registration: Expected 2 strategies, got {len(test_registry)}"
            )

        if "TestStrategy1" not in test_registry:
            all_validation_failures.append(
                "Strategy registration: TestStrategy1 not found in registry"
            )

        if "CustomName" not in test_registry:
            all_validation_failures.append(
                "Custom name registration: CustomName not found in registry"
            )
    except Exception as e:
        all_validation_failures.append(f"Strategy registration failed: {e}")

    # Test 4: Retrieve strategies
    total_tests += 1
    try:
        retrieved_class = test_registry.get_strategy("TestStrategy1")
        if retrieved_class is not TestStrategy1:
            all_validation_failures.append(
                f"Strategy retrieval: Expected TestStrategy1, got {retrieved_class}"
            )

        custom_class = test_registry.get_strategy("CustomName")
        if custom_class is not TestStrategy2:
            all_validation_failures.append(
                f"Custom name retrieval: Expected TestStrategy2, got {custom_class}"
            )
    except Exception as e:
        all_validation_failures.append(f"Strategy retrieval failed: {e}")

    # Test 5: List strategies
    total_tests += 1
    try:
        all_strats = test_registry.list_strategies()
        if len(all_strats) != 2:
            all_validation_failures.append(
                f"List all strategies: Expected 2, got {len(all_strats)}"
            )

        # Filter by tags
        tagged_strats = test_registry.list_strategies(tags=["test"])
        if len(tagged_strats) != 1:
            all_validation_failures.append(
                f"List by tags: Expected 1 strategy, got {len(tagged_strats)}"
            )
        if "CustomName" not in tagged_strats:
            all_validation_failures.append(
                "List by tags: CustomName not in filtered results"
            )
    except Exception as e:
        all_validation_failures.append(f"List strategies failed: {e}")

    # Test 6: Get strategy info
    total_tests += 1
    try:
        info = test_registry.get_strategy_info("TestStrategy1")
        expected_keys = {'class_name', 'module', 'description', 'tags'}
        actual_keys = set(info.keys())

        if actual_keys != expected_keys:
            all_validation_failures.append(
                f"Strategy info: Expected keys {expected_keys}, got {actual_keys}"
            )

        if info['class_name'] != 'TestStrategy1':
            all_validation_failures.append(
                f"Strategy info class_name: Expected 'TestStrategy1', got {info['class_name']}"
            )
    except Exception as e:
        all_validation_failures.append(f"Get strategy info failed: {e}")

    # Test 7: Decorator registration
    total_tests += 1
    try:
        # Clear global registry for clean test
        _global_registry.clear()

        @register_strategy
        class DecoratedStrategy(BaseStrategy):
            """Decorated test strategy."""

            def initialize(self, config: Dict[str, Any]) -> None:
                pass

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame()

            def get_parameters(self) -> Dict[str, Any]:
                return {}

        if "DecoratedStrategy" not in _global_registry:
            all_validation_failures.append(
                "Decorator registration: DecoratedStrategy not in global registry"
            )

        retrieved = get_strategy("DecoratedStrategy")
        if retrieved is not DecoratedStrategy:
            all_validation_failures.append(
                f"Decorator retrieval: Expected DecoratedStrategy, got {retrieved}"
            )
    except Exception as e:
        all_validation_failures.append(f"Decorator registration failed: {e}")

    # Test 8: Decorator with parameters
    total_tests += 1
    try:
        @register_strategy(name="ParamDecorated", tags=["momentum", "test"])
        class ParameterizedDecorated(BaseStrategy):
            """Strategy with decorator parameters."""

            def initialize(self, config: Dict[str, Any]) -> None:
                pass

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame()

            def get_parameters(self) -> Dict[str, Any]:
                return {}

        if "ParamDecorated" not in _global_registry:
            all_validation_failures.append(
                "Parameterized decorator: ParamDecorated not in registry"
            )

        # Check tags
        info = _global_registry.get_strategy_info("ParamDecorated")
        if "momentum" not in info['tags']:
            all_validation_failures.append(
                f"Decorator tags: Expected 'momentum' in tags, got {info['tags']}"
            )
    except Exception as e:
        all_validation_failures.append(f"Parameterized decorator failed: {e}")

    # Test 9: Error handling - duplicate registration
    total_tests += 1
    try:
        test_registry2 = StrategyRegistry()
        test_registry2.register(TestStrategy1)

        # Attempt duplicate registration
        duplicate_error = False
        try:
            test_registry2.register(TestStrategy1)
        except ValueError as e:
            if "already registered" in str(e):
                duplicate_error = True

        if not duplicate_error:
            all_validation_failures.append(
                "Duplicate registration: Expected ValueError, but no error raised"
            )
    except Exception as e:
        all_validation_failures.append(f"Duplicate registration test failed: {e}")

    # Test 10: Error handling - invalid strategy retrieval
    total_tests += 1
    try:
        invalid_error = False
        try:
            test_registry.get_strategy("NonExistentStrategy")
        except KeyError as e:
            if "not found" in str(e):
                invalid_error = True

        if not invalid_error:
            all_validation_failures.append(
                "Invalid retrieval: Expected KeyError, but no error raised"
            )
    except Exception as e:
        all_validation_failures.append(f"Invalid retrieval test failed: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("StrategyRegistry is validated and ready for use")
        sys.exit(0)
