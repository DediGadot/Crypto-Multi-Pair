"""
Integration Tests for Strategy Framework

Tests the complete strategy framework including:
- Base strategy interface
- Registry and registration
- Configuration loading
- Signal generation
- End-to-end workflows

**Purpose**: Verify that all strategy framework components work together correctly.

**Third-party packages**:
- pytest: https://docs.pytest.org/
- pandas: https://pandas.pydata.org/docs/
"""

import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
import pytest

from crypto_trader.strategies import (
    BaseStrategy,
    SignalType,
    StrategyRegistry,
    StrategyLoader,
    StrategyConfig,
    register_strategy,
    get_strategy,
    list_strategies,
    get_registry,
    load_strategies_from_yaml,
)


class TestBaseStrategy:
    """Test the BaseStrategy interface."""

    def test_signal_type_enum(self):
        """Test SignalType enum values."""
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"

    def test_base_strategy_initialization(self):
        """Test base strategy can be initialized via subclass."""

        class TestStrategy(BaseStrategy):
            def initialize(self, config):
                self.config.update(config)

            def generate_signals(self, data):
                return pd.DataFrame()

            def get_parameters(self):
                return self.config

        strategy = TestStrategy("test", {"param": "value"})
        assert strategy.name == "test"
        assert strategy.config == {"param": "value"}

    def test_data_validation(self):
        """Test data validation with valid and invalid data."""

        class TestStrategy(BaseStrategy):
            def initialize(self, config):
                pass

            def generate_signals(self, data):
                return pd.DataFrame()

            def get_parameters(self):
                return {}

        strategy = TestStrategy("test")

        # Valid data
        valid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100],
            'high': [105],
            'low': [99],
            'close': [103],
            'volume': [1000]
        })
        assert strategy.validate_data(valid_data)

        # Invalid data (missing columns)
        invalid_data = pd.DataFrame({'close': [100]})
        assert not strategy.validate_data(invalid_data)


class TestStrategyRegistry:
    """Test the StrategyRegistry."""

    def test_registry_initialization(self):
        """Test registry can be created."""
        registry = StrategyRegistry()
        assert len(registry) == 0

    def test_explicit_registration(self):
        """Test explicit strategy registration."""

        class TestStrategy(BaseStrategy):
            def initialize(self, config):
                pass

            def generate_signals(self, data):
                return pd.DataFrame()

            def get_parameters(self):
                return {}

        registry = StrategyRegistry()
        registry.register(TestStrategy)

        assert "TestStrategy" in registry
        assert len(registry) == 1

    def test_decorator_registration(self):
        """Test decorator-based registration."""
        test_registry = StrategyRegistry()

        # Clear global registry
        get_registry().clear()

        @register_strategy(tags=["test"])
        class DecoratedStrategy(BaseStrategy):
            def initialize(self, config):
                pass

            def generate_signals(self, data):
                return pd.DataFrame()

            def get_parameters(self):
                return {}

        # Should be in global registry
        assert "DecoratedStrategy" in get_registry()

    def test_strategy_retrieval(self):
        """Test retrieving registered strategies."""
        registry = StrategyRegistry()

        class StrategyA(BaseStrategy):
            def initialize(self, config):
                pass

            def generate_signals(self, data):
                return pd.DataFrame()

            def get_parameters(self):
                return {}

        registry.register(StrategyA)
        retrieved = registry.get_strategy("StrategyA")
        assert retrieved is StrategyA

    def test_list_strategies(self):
        """Test listing strategies with filtering."""
        registry = StrategyRegistry()

        class Strategy1(BaseStrategy):
            def initialize(self, config):
                pass

            def generate_signals(self, data):
                return pd.DataFrame()

            def get_parameters(self):
                return {}

        class Strategy2(BaseStrategy):
            def initialize(self, config):
                pass

            def generate_signals(self, data):
                return pd.DataFrame()

            def get_parameters(self):
                return {}

        registry.register(Strategy1, tags=["momentum"])
        registry.register(Strategy2, tags=["mean_reversion"])

        # List all
        all_strats = registry.list_strategies()
        assert len(all_strats) == 2

        # Filter by tags
        momentum = registry.list_strategies(tags=["momentum"])
        assert len(momentum) == 1
        assert "Strategy1" in momentum


class TestStrategyLoader:
    """Test the StrategyLoader."""

    def test_loader_initialization(self):
        """Test loader can be created."""
        loader = StrategyLoader()
        assert loader is not None

    def test_config_validation(self):
        """Test StrategyConfig validation."""
        # Valid config
        config = StrategyConfig(
            name="test",
            class_name="TestStrategy",
            parameters={"key": "value"}
        )
        assert config.name == "test"
        assert config.class_name == "TestStrategy"

    def test_load_yaml_config(self):
        """Test loading strategies from YAML."""
        # Create temporary YAML file
        yaml_content = """
strategies:
  - name: "TestStrategy1"
    class: "DummyStrategy"
    enabled: true
    parameters:
      param1: value1
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = StrategyLoader()
            configs = loader.load_config_file(temp_path)

            assert len(configs) == 1
            assert configs[0].name == "TestStrategy1"
            assert configs[0].class_name == "DummyStrategy"
            assert configs[0].parameters["param1"] == "value1"
        finally:
            Path(temp_path).unlink()

    def test_strategy_instantiation(self):
        """Test instantiating strategy from config."""
        # Register a test strategy
        get_registry().clear()

        @register_strategy
        class InstantiationTestStrategy(BaseStrategy):
            def initialize(self, config):
                self.test_param = config.get("test_param")
                self.config.update(config)

            def generate_signals(self, data):
                return pd.DataFrame()

            def get_parameters(self):
                return {"test_param": self.test_param}

        # Create config and instantiate
        config = StrategyConfig(
            name="test_instance",
            class_name="InstantiationTestStrategy",
            parameters={"test_param": "test_value"}
        )

        loader = StrategyLoader()
        strategy = loader.instantiate_strategy(config)

        assert isinstance(strategy, BaseStrategy)
        assert strategy.name == "test_instance"
        assert strategy.get_parameters()["test_param"] == "test_value"


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_workflow(self):
        """Test loading config, instantiating, and generating signals."""
        # Clear registry
        get_registry().clear()

        # Define test strategy
        @register_strategy(tags=["test"])
        class WorkflowTestStrategy(BaseStrategy):
            def initialize(self, config):
                self.threshold = config.get("threshold", 0.5)
                self.config.update(config)

            def generate_signals(self, data):
                signals = []
                for price in data['close']:
                    if price < self.threshold:
                        signals.append(SignalType.BUY.value)
                    elif price > self.threshold * 2:
                        signals.append(SignalType.SELL.value)
                    else:
                        signals.append(SignalType.HOLD.value)

                return pd.DataFrame({
                    'timestamp': data['timestamp'],
                    'signal': signals,
                    'confidence': [0.8] * len(data),
                    'metadata': [{}] * len(data)
                })

            def get_parameters(self):
                return {"threshold": self.threshold}

        # Create YAML config
        yaml_content = """
strategies:
  - name: "WorkflowTest"
    class: "WorkflowTestStrategy"
    enabled: true
    parameters:
      threshold: 100
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Load strategies
            strategies = load_strategies_from_yaml(temp_path)

            assert len(strategies) == 1
            assert "WorkflowTest" in strategies

            # Generate signals
            test_data = pd.DataFrame({
                'timestamp': [datetime.now() - timedelta(days=i) for i in range(5)],
                'open': [90, 100, 110, 120, 130],
                'high': [95, 105, 115, 125, 135],
                'low': [85, 95, 105, 115, 125],
                'close': [92, 102, 112, 122, 132],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })

            strategy = strategies["WorkflowTest"]
            signals = strategy.generate_signals(test_data)

            # Verify signals
            assert len(signals) == 5
            assert 'signal' in signals.columns
            assert 'confidence' in signals.columns
            assert 'metadata' in signals.columns

            # Check signal types
            assert signals['signal'].iloc[0] == SignalType.BUY.value  # 92 < 100
            # 132 is > 100 but < 200 (100*2), so it should be HOLD
            assert signals['signal'].iloc[-1] == SignalType.HOLD.value  # 132 is between 100 and 200

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    """Run tests with pytest."""
    import sys

    # Run pytest with verbose output
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
