"""
Base Strategy Interface for Crypto Trading System

This module defines the abstract base class for all trading strategies in the system.
It provides a consistent interface that all strategies must implement, ensuring
modularity and pluggability.

**Purpose**: Define the contract for trading strategies with signal generation,
parameter management, and data validation capabilities.

**Key Components**:
- BaseStrategy: Abstract base class for all strategies
- Signal types: BUY, SELL, HOLD
- Data validation and indicator requirements

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- abc: https://docs.python.org/3/library/abc.html

**Sample Input**:
```python
data = pd.DataFrame({
    'timestamp': [...],
    'open': [100, 101, 102],
    'high': [105, 106, 107],
    'low': [99, 100, 101],
    'close': [103, 104, 105],
    'volume': [1000, 1100, 1200]
})
```

**Expected Output**:
```python
signals = pd.DataFrame({
    'timestamp': [...],
    'signal': ['HOLD', 'BUY', 'SELL'],
    'confidence': [0.0, 0.85, 0.92],
    'metadata': [{}, {'reason': 'crossover'}, {'reason': 'overbought'}]
})
```
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum

import pandas as pd
from loguru import logger


class SignalType(str, Enum):
    """Signal types that a strategy can generate."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must inherit from this class and implement the required methods.
    This ensures a consistent interface across all strategies for the plugin system.

    Attributes:
        name: Strategy name (unique identifier)
        config: Strategy configuration parameters
        _initialized: Whether the strategy has been initialized
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base strategy.

        Args:
            name: Unique name for the strategy
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._initialized = False
        logger.debug(f"Base strategy initialized: {name}")

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the strategy with configuration parameters.

        This method should validate and set up all required parameters
        for the strategy. Called once before the strategy is used.

        Args:
            config: Dictionary containing strategy-specific parameters

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.

        This is the core method where the strategy logic is implemented.
        It should analyze the input data and return signals.

        Args:
            data: DataFrame with OHLCV data and any required indicators

        Returns:
            DataFrame with columns: ['timestamp', 'signal', 'confidence', 'metadata']
            - timestamp: DateTime of the signal
            - signal: One of SignalType (BUY, SELL, HOLD)
            - confidence: Float between 0.0 and 1.0
            - metadata: Dict with additional signal information

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary containing all strategy parameters with their current values
        """
        pass

    def get_required_indicators(self) -> List[str]:
        """
        Get list of required technical indicators for this strategy.

        Override this method if your strategy requires specific indicators
        to be calculated before signal generation.

        Returns:
            List of indicator names (e.g., ['SMA_20', 'RSI_14', 'MACD'])
        """
        return []

    def get_required_features(self) -> List[str]:
        """
        Get list of required alternative-data features for this strategy.

        Return fully-qualified feature names if applicable, e.g.,
        ['onchain.mvrv_z', 'sent.pos_24h']. Default: [] for backward compatibility.
        """
        return []

    def get_feature_lags(self) -> Dict[str, str]:
        """
        Optional per-feature lag to avoid look-ahead (e.g., {'onchain.*': '1d'}).

        Strings should be pandas Timedelta-parseable (e.g., '1d', '4h').
        Default: {} (no special lagging applied by the factory).
        """
        return {}

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that input data contains required columns and indicators.

        This method checks for basic OHLCV columns and any required indicators.
        Override for custom validation logic.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Check for basic OHLCV columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Check for required indicators
        required_indicators = self.get_required_indicators()
        missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
        if missing_indicators:
            logger.error(f"Missing required indicators: {missing_indicators}")
            return False

        # Check for empty data
        if len(data) == 0:
            logger.error("Data is empty")
            return False

        # Check for NaN values in critical columns
        critical_cols = ['close', 'volume']
        for col in critical_cols:
            if data[col].isna().any():
                logger.warning(f"Column '{col}' contains NaN values")

        logger.debug(f"Data validation passed for strategy: {self.name}")
        return True

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters.

        Args:
            parameters: Dictionary of parameters to update
        """
        self.config.update(parameters)
        logger.debug(f"Updated parameters for strategy {self.name}: {parameters}")

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}(name='{self.name}')"


if __name__ == "__main__":
    """
    Validation block for BaseStrategy interface.
    Tests the abstract base class and signal types.
    """
    import sys
    from datetime import datetime, timedelta

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    # Test 1: SignalType enum values
    total_tests += 1
    try:
        expected_signals = {'BUY', 'SELL', 'HOLD'}
        actual_signals = {signal.value for signal in SignalType}
        if actual_signals != expected_signals:
            all_validation_failures.append(
                f"SignalType enum: Expected {expected_signals}, got {actual_signals}"
            )
    except Exception as e:
        all_validation_failures.append(f"SignalType enum test failed: {e}")

    # Test 2: Create a concrete implementation for testing
    total_tests += 1
    try:
        class TestStrategy(BaseStrategy):
            """Concrete strategy for testing."""

            def initialize(self, config: Dict[str, Any]) -> None:
                self.config.update(config)
                self._initialized = True

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                signals = pd.DataFrame({
                    'timestamp': data['timestamp'],
                    'signal': [SignalType.HOLD.value] * len(data),
                    'confidence': [0.5] * len(data),
                    'metadata': [{}] * len(data)
                })
                return signals

            def get_parameters(self) -> Dict[str, Any]:
                return self.config

            def get_required_indicators(self) -> List[str]:
                return ['SMA_20']

        strategy = TestStrategy(name="test_strategy", config={"period": 20})
        if not isinstance(strategy, BaseStrategy):
            all_validation_failures.append(
                f"Strategy inheritance: Expected BaseStrategy instance, got {type(strategy)}"
            )
    except Exception as e:
        all_validation_failures.append(f"Concrete strategy creation failed: {e}")

    # Test 3: Data validation with valid data
    total_tests += 1
    try:
        # Create valid test data
        dates = [datetime.now() - timedelta(days=i) for i in range(10)]
        valid_data = pd.DataFrame({
            'timestamp': dates,
            'open': [100 + i for i in range(10)],
            'high': [105 + i for i in range(10)],
            'low': [99 + i for i in range(10)],
            'close': [103 + i for i in range(10)],
            'volume': [1000 + i*10 for i in range(10)],
            'SMA_20': [102 + i for i in range(10)]
        })

        validation_result = strategy.validate_data(valid_data)
        if not validation_result:
            all_validation_failures.append(
                "Data validation: Expected True for valid data, got False"
            )
    except Exception as e:
        all_validation_failures.append(f"Valid data validation test failed: {e}")

    # Test 4: Data validation with missing columns
    total_tests += 1
    try:
        invalid_data = pd.DataFrame({
            'timestamp': dates,
            'close': [103 + i for i in range(10)]
        })

        validation_result = strategy.validate_data(invalid_data)
        if validation_result:
            all_validation_failures.append(
                "Invalid data validation: Expected False for missing columns, got True"
            )
    except Exception as e:
        all_validation_failures.append(f"Invalid data validation test failed: {e}")

    # Test 5: Signal generation
    total_tests += 1
    try:
        signals = strategy.generate_signals(valid_data)
        expected_columns = {'timestamp', 'signal', 'confidence', 'metadata'}
        actual_columns = set(signals.columns)

        if actual_columns != expected_columns:
            all_validation_failures.append(
                f"Signal generation: Expected columns {expected_columns}, got {actual_columns}"
            )

        if len(signals) != len(valid_data):
            all_validation_failures.append(
                f"Signal generation: Expected {len(valid_data)} rows, got {len(signals)}"
            )
    except Exception as e:
        all_validation_failures.append(f"Signal generation test failed: {e}")

    # Test 6: Parameter management
    total_tests += 1
    try:
        initial_params = strategy.get_parameters()
        if initial_params.get('period') != 20:
            all_validation_failures.append(
                f"Initial parameters: Expected period=20, got {initial_params.get('period')}"
            )

        strategy.set_parameters({'period': 30, 'threshold': 0.7})
        updated_params = strategy.get_parameters()

        if updated_params.get('period') != 30:
            all_validation_failures.append(
                f"Updated parameters: Expected period=30, got {updated_params.get('period')}"
            )
        if updated_params.get('threshold') != 0.7:
            all_validation_failures.append(
                f"Updated parameters: Expected threshold=0.7, got {updated_params.get('threshold')}"
            )
    except Exception as e:
        all_validation_failures.append(f"Parameter management test failed: {e}")

    # Test 7: Required indicators
    total_tests += 1
    try:
        required = strategy.get_required_indicators()
        if required != ['SMA_20']:
            all_validation_failures.append(
                f"Required indicators: Expected ['SMA_20'], got {required}"
            )
    except Exception as e:
        all_validation_failures.append(f"Required indicators test failed: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("BaseStrategy interface is validated and ready for use")
        sys.exit(0)
