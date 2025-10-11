"""
Strategy Configuration Loader

This module handles loading strategy configurations from YAML files and
instantiating strategy objects with validated parameters.

**Purpose**: Load, validate, and instantiate trading strategies from configuration
files, supporting multiple configuration formats and parameter validation.

**Key Features**:
- Load strategy configs from YAML files
- Validate configuration parameters
- Instantiate strategies with configs
- Support multiple strategies per config file
- Parameter type validation

**Third-party packages**:
- pyyaml: https://pyyaml.org/wiki/PyYAMLDocumentation
- pydantic: https://docs.pydantic.dev/latest/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input (YAML)**:
```yaml
strategies:
  - name: "MovingAverageCross"
    class: "MovingAverageCrossover"
    enabled: true
    parameters:
      fast_period: 10
      slow_period: 30
      signal_threshold: 0.75
```

**Expected Output**:
```python
{
    "MovingAverageCross": <MovingAverageCrossover instance>
}
```
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from crypto_trader.strategies.base import BaseStrategy
from crypto_trader.strategies.registry import get_strategy, get_registry


class StrategyConfig(BaseModel):
    """
    Pydantic model for strategy configuration validation.

    Ensures that strategy configurations have all required fields
    and valid parameter types.
    """

    name: str = Field(..., description="Unique name for the strategy instance")
    class_name: str = Field(
        ..., alias="class", description="Strategy class name from registry"
    )
    enabled: bool = Field(True, description="Whether the strategy is enabled")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )
    tags: List[str] = Field(default_factory=list, description="Optional tags")
    description: Optional[str] = Field(None, description="Optional description")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate strategy name is not empty."""
        if not v or not v.strip():
            raise ValueError("Strategy name cannot be empty")
        return v.strip()

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        """Validate class name is not empty."""
        if not v or not v.strip():
            raise ValueError("Strategy class name cannot be empty")
        return v.strip()

    model_config = ConfigDict(populate_by_name=True)


class StrategyLoader:
    """
    Loader for strategy configurations.

    This class handles loading strategies from YAML files, validating
    configurations, and instantiating strategy objects.

    Attributes:
        config_path: Path to configuration file or directory
        _loaded_strategies: Cache of loaded strategy instances
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the strategy loader.

        Args:
            config_path: Optional path to config file or directory
        """
        self.config_path = Path(config_path) if config_path else None
        self._loaded_strategies: Dict[str, BaseStrategy] = {}
        logger.debug(f"Strategy loader initialized with path: {self.config_path}")

    def load_config_file(self, file_path: Union[str, Path]) -> List[StrategyConfig]:
        """
        Load strategy configurations from a YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            List of validated StrategyConfig objects

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid or missing required fields
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        logger.info(f"Loading strategy config from: {path}")

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                raise ValueError(f"Empty config file: {path}")

            # Support both single strategy and list of strategies
            if "strategies" in data:
                strategies_data = data["strategies"]
            elif "strategy" in data:
                strategies_data = [data["strategy"]]
            else:
                # Assume entire file is a single strategy config
                strategies_data = [data]

            # Validate each strategy config
            configs = []
            for idx, strategy_data in enumerate(strategies_data):
                try:
                    config = StrategyConfig(**strategy_data)
                    configs.append(config)
                except ValidationError as e:
                    logger.error(f"Invalid config at index {idx}: {e}")
                    raise ValueError(f"Strategy config validation failed at index {idx}: {e}")

            logger.info(f"Loaded {len(configs)} strategy configs from {path}")
            return configs

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            raise

    def load_from_directory(self, directory: Union[str, Path]) -> List[StrategyConfig]:
        """
        Load all strategy configs from a directory.

        Args:
            directory: Path to directory containing YAML config files

        Returns:
            List of all StrategyConfig objects found

        Raises:
            ValueError: If directory doesn't exist
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {dir_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        logger.info(f"Loading strategy configs from directory: {dir_path}")

        all_configs = []
        yaml_files = list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))

        for yaml_file in yaml_files:
            try:
                configs = self.load_config_file(yaml_file)
                all_configs.extend(configs)
            except Exception as e:
                logger.error(f"Skipping {yaml_file}: {e}")
                continue

        logger.info(f"Loaded {len(all_configs)} total configs from {dir_path}")
        return all_configs

    def instantiate_strategy(
        self,
        config: StrategyConfig,
        registry: Optional[Any] = None
    ) -> BaseStrategy:
        """
        Instantiate a strategy from a configuration.

        Args:
            config: Validated StrategyConfig object
            registry: Optional StrategyRegistry instance (uses global if None)

        Returns:
            Instantiated BaseStrategy object

        Raises:
            KeyError: If strategy class not found in registry
            ValueError: If strategy initialization fails
        """
        # Use global registry if none provided
        if registry is None:
            registry = get_registry()

        logger.info(f"Instantiating strategy: {config.name} (class: {config.class_name})")

        try:
            # Get strategy class from registry
            strategy_class = registry.get_strategy(config.class_name)

            # Create instance
            strategy = strategy_class(name=config.name, config=config.parameters)

            # Initialize with parameters
            strategy.initialize(config.parameters)

            logger.info(f"Successfully instantiated strategy: {config.name}")
            logger.debug(f"Strategy parameters: {config.parameters}")

            return strategy

        except KeyError as e:
            available = ", ".join(registry.get_strategy_names())
            raise KeyError(
                f"Strategy class '{config.class_name}' not found in registry. "
                f"Available: {available}"
            )
        except Exception as e:
            raise ValueError(f"Failed to instantiate strategy '{config.name}': {e}")

    def load_strategies(
        self,
        config_path: Optional[Union[str, Path]] = None,
        enabled_only: bool = True
    ) -> Dict[str, BaseStrategy]:
        """
        Load and instantiate all strategies from a config file or directory.

        Args:
            config_path: Path to config file or directory (uses self.config_path if None)
            enabled_only: Only load strategies where enabled=True

        Returns:
            Dictionary mapping strategy names to instantiated strategy objects

        Raises:
            ValueError: If no config path provided and none set in __init__
        """
        path = Path(config_path) if config_path else self.config_path

        if path is None:
            raise ValueError("No config path provided")

        # Load configurations
        if path.is_file():
            configs = self.load_config_file(path)
        else:
            configs = self.load_from_directory(path)

        # Filter by enabled status
        if enabled_only:
            configs = [c for c in configs if c.enabled]
            logger.info(f"Loading {len(configs)} enabled strategies")

        # Instantiate strategies
        strategies = {}
        for config in configs:
            try:
                strategy = self.instantiate_strategy(config)
                strategies[config.name] = strategy
                self._loaded_strategies[config.name] = strategy
            except Exception as e:
                logger.error(f"Failed to load strategy {config.name}: {e}")
                # Continue with other strategies
                continue

        logger.info(f"Successfully loaded {len(strategies)} strategies")
        return strategies

    def get_loaded_strategy(self, name: str) -> Optional[BaseStrategy]:
        """
        Get a previously loaded strategy by name.

        Args:
            name: Strategy instance name

        Returns:
            Strategy instance or None if not found
        """
        return self._loaded_strategies.get(name)

    def get_loaded_strategies(self) -> Dict[str, BaseStrategy]:
        """
        Get all loaded strategies.

        Returns:
            Dictionary of loaded strategy instances
        """
        return dict(self._loaded_strategies)

    def validate_config(self, config_path: Union[str, Path]) -> bool:
        """
        Validate a configuration file without loading strategies.

        Args:
            config_path: Path to config file

        Returns:
            True if valid, False otherwise
        """
        try:
            configs = self.load_config_file(config_path)

            # Check if all strategy classes exist in registry
            registry = get_registry()
            available_strategies = registry.get_strategy_names()

            for config in configs:
                if config.class_name not in available_strategies:
                    logger.error(
                        f"Strategy class '{config.class_name}' not found in registry"
                    )
                    return False

            logger.info(f"Config file {config_path} is valid")
            return True

        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False


def load_strategies_from_yaml(
    yaml_path: Union[str, Path],
    enabled_only: bool = True
) -> Dict[str, BaseStrategy]:
    """
    Convenience function to load strategies from a YAML file.

    Args:
        yaml_path: Path to YAML configuration file
        enabled_only: Only load enabled strategies

    Returns:
        Dictionary of strategy name to strategy instance

    Example:
        >>> strategies = load_strategies_from_yaml("config/strategies.yaml")
        >>> sma_strategy = strategies["SMA_Strategy"]
    """
    loader = StrategyLoader()
    return loader.load_strategies(yaml_path, enabled_only=enabled_only)


if __name__ == "__main__":
    """
    Validation block for StrategyLoader.
    Tests loading, validation, and instantiation from YAML configs.
    """
    import sys
    import tempfile
    from typing import Dict, Any
    import pandas as pd

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    # Test 1: Create a test strategy for validation
    total_tests += 1
    try:
        class ValidTestStrategy(BaseStrategy):
            """Test strategy for loader validation."""

            def initialize(self, config: Dict[str, Any]) -> None:
                self.config.update(config)
                self._initialized = True

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame()

            def get_parameters(self) -> Dict[str, Any]:
                return self.config

        # Register it
        from crypto_trader.strategies.registry import register_strategy

        register_strategy(ValidTestStrategy)

        if not isinstance(ValidTestStrategy("test", {}), BaseStrategy):
            all_validation_failures.append(
                "Test strategy creation failed: not a BaseStrategy instance"
            )
    except Exception as e:
        all_validation_failures.append(f"Test strategy creation failed: {e}")

    # Test 2: Create temporary YAML config
    total_tests += 1
    try:
        yaml_content = """
strategies:
  - name: "TestStrategy1"
    class: "ValidTestStrategy"
    enabled: true
    parameters:
      period: 20
      threshold: 0.75
    tags: ["test", "validation"]

  - name: "TestStrategy2"
    class: "ValidTestStrategy"
    enabled: false
    parameters:
      period: 50
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_yaml_path = f.name

        logger.debug(f"Created temporary YAML config: {temp_yaml_path}")
    except Exception as e:
        all_validation_failures.append(f"Temporary YAML creation failed: {e}")

    # Test 3: Load config file
    total_tests += 1
    try:
        loader = StrategyLoader()
        configs = loader.load_config_file(temp_yaml_path)

        if len(configs) != 2:
            all_validation_failures.append(
                f"Load config: Expected 2 configs, got {len(configs)}"
            )

        if configs[0].name != "TestStrategy1":
            all_validation_failures.append(
                f"Config name: Expected 'TestStrategy1', got {configs[0].name}"
            )

        if configs[0].parameters.get("period") != 20:
            all_validation_failures.append(
                f"Config parameters: Expected period=20, got {configs[0].parameters.get('period')}"
            )
    except Exception as e:
        all_validation_failures.append(f"Load config file failed: {e}")

    # Test 4: Validate StrategyConfig model
    total_tests += 1
    try:
        valid_config = StrategyConfig(
            name="TestConfig",
            class_name="ValidTestStrategy",
            enabled=True,
            parameters={"key": "value"}
        )

        if valid_config.name != "TestConfig":
            all_validation_failures.append(
                f"StrategyConfig validation: Expected name='TestConfig', got {valid_config.name}"
            )

        if valid_config.class_name != "ValidTestStrategy":
            all_validation_failures.append(
                f"StrategyConfig validation: Expected class_name='ValidTestStrategy', got {valid_config.class_name}"
            )
    except Exception as e:
        all_validation_failures.append(f"StrategyConfig validation failed: {e}")

    # Test 5: Instantiate strategy from config
    total_tests += 1
    try:
        config = configs[0]
        strategy = loader.instantiate_strategy(config)

        if not isinstance(strategy, BaseStrategy):
            all_validation_failures.append(
                f"Strategy instantiation: Expected BaseStrategy, got {type(strategy)}"
            )

        if strategy.name != "TestStrategy1":
            all_validation_failures.append(
                f"Instantiated strategy name: Expected 'TestStrategy1', got {strategy.name}"
            )

        params = strategy.get_parameters()
        if params.get("period") != 20:
            all_validation_failures.append(
                f"Instantiated strategy params: Expected period=20, got {params.get('period')}"
            )
    except Exception as e:
        all_validation_failures.append(f"Strategy instantiation failed: {e}")

    # Test 6: Load all strategies (enabled only)
    total_tests += 1
    try:
        strategies = loader.load_strategies(temp_yaml_path, enabled_only=True)

        if len(strategies) != 1:
            all_validation_failures.append(
                f"Load enabled strategies: Expected 1, got {len(strategies)}"
            )

        if "TestStrategy1" not in strategies:
            all_validation_failures.append(
                "Load enabled strategies: TestStrategy1 not found"
            )

        if "TestStrategy2" in strategies:
            all_validation_failures.append(
                "Load enabled strategies: TestStrategy2 should not be loaded (disabled)"
            )
    except Exception as e:
        all_validation_failures.append(f"Load enabled strategies failed: {e}")

    # Test 7: Load all strategies (including disabled)
    total_tests += 1
    try:
        loader2 = StrategyLoader()
        all_strategies = loader2.load_strategies(temp_yaml_path, enabled_only=False)

        if len(all_strategies) != 2:
            all_validation_failures.append(
                f"Load all strategies: Expected 2, got {len(all_strategies)}"
            )

        if "TestStrategy2" not in all_strategies:
            all_validation_failures.append(
                "Load all strategies: TestStrategy2 not found"
            )
    except Exception as e:
        all_validation_failures.append(f"Load all strategies failed: {e}")

    # Test 8: Get loaded strategy
    total_tests += 1
    try:
        loaded = loader.get_loaded_strategy("TestStrategy1")
        if loaded is None:
            all_validation_failures.append(
                "Get loaded strategy: TestStrategy1 not found in cache"
            )

        if loaded and not isinstance(loaded, BaseStrategy):
            all_validation_failures.append(
                f"Get loaded strategy: Expected BaseStrategy, got {type(loaded)}"
            )
    except Exception as e:
        all_validation_failures.append(f"Get loaded strategy failed: {e}")

    # Test 9: Validate config
    total_tests += 1
    try:
        is_valid = loader.validate_config(temp_yaml_path)
        if not is_valid:
            all_validation_failures.append(
                "Config validation: Expected True for valid config"
            )
    except Exception as e:
        all_validation_failures.append(f"Config validation failed: {e}")

    # Test 10: Convenience function
    total_tests += 1
    try:
        convenience_strategies = load_strategies_from_yaml(temp_yaml_path)

        if len(convenience_strategies) != 1:
            all_validation_failures.append(
                f"Convenience function: Expected 1 strategy, got {len(convenience_strategies)}"
            )

        if "TestStrategy1" not in convenience_strategies:
            all_validation_failures.append(
                "Convenience function: TestStrategy1 not found"
            )
    except Exception as e:
        all_validation_failures.append(f"Convenience function failed: {e}")

    # Cleanup
    try:
        Path(temp_yaml_path).unlink()
    except Exception:
        pass

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("StrategyLoader is validated and ready for use")
        sys.exit(0)
