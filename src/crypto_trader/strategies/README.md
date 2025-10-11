# Strategy Framework

A modular, plugin-based framework for implementing and managing trading strategies in the crypto trading system.

## Overview

The strategy framework provides a unified interface for:
- Defining trading strategies as simple Python classes
- Automatically discovering and registering strategies
- Loading strategies from YAML configuration files
- Generating trading signals from market data
- Managing strategy parameters and metadata

## Architecture

### Core Components

1. **BaseStrategy** (`base.py`)
   - Abstract base class that all strategies must inherit from
   - Defines the contract for strategy implementation
   - Provides data validation and parameter management

2. **StrategyRegistry** (`registry.py`)
   - Plugin system for managing strategy classes
   - Decorator-based registration (`@register_strategy`)
   - Dynamic strategy discovery and loading
   - Thread-safe operations

3. **StrategyLoader** (`loader.py`)
   - Loads strategy configurations from YAML files
   - Validates parameters using Pydantic models
   - Instantiates strategy objects with validated configs
   - Supports enabled/disabled strategies

4. **SignalType** (`base.py`)
   - Enum defining trading signals: BUY, SELL, HOLD
   - Used consistently across all strategies

## Quick Start

### 1. Define a Strategy

```python
from crypto_trader.strategies import BaseStrategy, SignalType, register_strategy
import pandas as pd

@register_strategy(tags=["custom", "momentum"])
class MyStrategy(BaseStrategy):
    """My custom trading strategy."""

    def initialize(self, config):
        """Initialize with parameters."""
        self.threshold = config.get("threshold", 0.5)
        self.config.update(config)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        signals = []
        confidences = []
        metadata = []

        for idx, row in data.iterrows():
            # Your strategy logic here
            if some_condition:
                signals.append(SignalType.BUY.value)
                confidences.append(0.85)
                metadata.append({"reason": "buy_condition_met"})
            else:
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})

        return pd.DataFrame({
            'timestamp': data['timestamp'],
            'signal': signals,
            'confidence': confidences,
            'metadata': metadata
        })

    def get_parameters(self):
        """Return current parameters."""
        return {"threshold": self.threshold}
```

### 2. Create a Configuration File

Create `config/strategies/my_strategies.yaml`:

```yaml
strategies:
  - name: "MyStrategy_Conservative"
    class: "MyStrategy"
    enabled: true
    description: "Conservative variant with high threshold"
    tags: ["conservative"]
    parameters:
      threshold: 0.8

  - name: "MyStrategy_Aggressive"
    class: "MyStrategy"
    enabled: true
    description: "Aggressive variant with low threshold"
    tags: ["aggressive"]
    parameters:
      threshold: 0.3
```

### 3. Load and Use Strategies

```python
from crypto_trader.strategies import load_strategies_from_yaml

# Load strategies from config
strategies = load_strategies_from_yaml("config/strategies/my_strategies.yaml")

# Get a specific strategy
strategy = strategies["MyStrategy_Conservative"]

# Generate signals
signals = strategy.generate_signals(market_data)

# Analyze signals
buy_signals = signals[signals['signal'] == 'BUY']
print(f"Generated {len(buy_signals)} BUY signals")
```

## BaseStrategy Interface

All strategies must implement these methods:

### Required Methods

#### `initialize(config: Dict[str, Any]) -> None`
Initialize the strategy with configuration parameters.

```python
def initialize(self, config):
    self.period = config.get("period", 20)
    self.threshold = config.get("threshold", 0.7)
    self.config.update(config)
```

#### `generate_signals(data: pd.DataFrame) -> pd.DataFrame`
Generate trading signals from market data.

**Input**: DataFrame with at least these columns:
- `timestamp`: DateTime
- `open`, `high`, `low`, `close`, `volume`: OHLCV data
- Any required indicators

**Output**: DataFrame with these columns:
- `timestamp`: DateTime of the signal
- `signal`: One of "BUY", "SELL", "HOLD"
- `confidence`: Float between 0.0 and 1.0
- `metadata`: Dict with additional signal information

```python
def generate_signals(self, data):
    # Your strategy logic
    return pd.DataFrame({
        'timestamp': data['timestamp'],
        'signal': signals,
        'confidence': confidences,
        'metadata': metadata
    })
```

#### `get_parameters() -> Dict[str, Any]`
Return current strategy parameters.

```python
def get_parameters(self):
    return {
        "period": self.period,
        "threshold": self.threshold
    }
```

### Optional Methods

#### `get_required_indicators() -> List[str]`
Return list of required technical indicators.

```python
def get_required_indicators(self):
    return ['SMA_20', 'RSI_14', 'MACD']
```

#### `validate_data(data: pd.DataFrame) -> bool`
Validate input data. Override for custom validation.

```python
def validate_data(self, data):
    # Call parent validation
    if not super().validate_data(data):
        return False

    # Custom validation
    if 'custom_indicator' not in data.columns:
        logger.error("Missing custom indicator")
        return False

    return True
```

## Strategy Registry

The registry manages all available strategies.

### Registration

#### Decorator Registration
```python
@register_strategy
class MyStrategy(BaseStrategy):
    pass

# With parameters
@register_strategy(name="CustomName", tags=["momentum", "custom"])
class AnotherStrategy(BaseStrategy):
    pass
```

#### Explicit Registration
```python
from crypto_trader.strategies import get_registry

registry = get_registry()
registry.register(MyStrategy, name="custom_name", tags=["test"])
```

### Retrieval

```python
from crypto_trader.strategies import get_strategy, list_strategies

# Get a specific strategy class
StrategyClass = get_strategy("MyStrategy")

# List all strategies
all_strategies = list_strategies()

# Filter by tags
momentum_strategies = list_strategies(tags=["momentum"])
```

### Dynamic Loading

Load strategies from a directory:

```python
registry = get_registry()
count = registry.load_from_directory(
    Path("src/crypto_trader/strategies/library"),
    recursive=True
)
print(f"Loaded {count} strategies")
```

## Configuration Format

### Strategy Config Structure

```yaml
strategies:
  - name: "unique_strategy_name"      # Required: Unique identifier
    class: "StrategyClassName"        # Required: Class name in registry
    enabled: true                      # Optional: Default true
    description: "Strategy description" # Optional
    tags: ["tag1", "tag2"]            # Optional: For filtering
    parameters:                        # Required: Strategy-specific params
      param1: value1
      param2: value2
```

### Multiple Strategies

You can configure multiple instances of the same strategy class with different parameters:

```yaml
strategies:
  - name: "MA_Cross_Short"
    class: "MovingAverageCrossover"
    enabled: true
    parameters:
      fast_period: 5
      slow_period: 10

  - name: "MA_Cross_Long"
    class: "MovingAverageCrossover"
    enabled: true
    parameters:
      fast_period: 50
      slow_period: 200
```

### Enabled/Disabled Strategies

```yaml
strategies:
  - name: "Production_Strategy"
    enabled: true
    # ... config ...

  - name: "Experimental_Strategy"
    enabled: false  # Won't be loaded by default
    # ... config ...
```

Load all strategies (including disabled):
```python
strategies = load_strategies_from_yaml(
    "config.yaml",
    enabled_only=False
)
```

## Included Strategies

### MovingAverageCrossover

Classic momentum strategy based on MA crossovers.

**Parameters**:
- `fast_period` (int): Fast MA period (default: 10)
- `slow_period` (int): Slow MA period (default: 20)
- `signal_threshold` (float): Minimum confidence (default: 0.7)
- `ma_type` (str): 'SMA' or 'EMA' (default: 'SMA')

**Required Indicators**:
- `SMA_{fast_period}` and `SMA_{slow_period}` (if using SMA)
- `EMA_{fast_period}` and `EMA_{slow_period}` (if using EMA)

**Signals**:
- BUY: Fast MA crosses above slow MA (bullish crossover)
- SELL: Fast MA crosses below slow MA (bearish crossover)
- HOLD: No crossover

**Example Config**:
```yaml
strategies:
  - name: "Golden_Cross"
    class: "MovingAverageCrossover"
    enabled: true
    parameters:
      fast_period: 50
      slow_period: 200
      signal_threshold: 0.8
      ma_type: "SMA"
```

## Best Practices

### 1. Strategy Design

- **Single Responsibility**: Each strategy should implement one clear trading logic
- **Configurable**: Use parameters for values that might change
- **Validated**: Implement parameter validation in `initialize()`
- **Documented**: Include clear docstrings explaining the strategy logic

### 2. Signal Generation

- **Consistent Format**: Always return the expected DataFrame structure
- **Confidence Scores**: Use meaningful confidence values (0.0 to 1.0)
- **Metadata**: Include diagnostic information in metadata for debugging
- **Error Handling**: Validate data before processing

### 3. Testing

- **Real Data**: Test with actual market data, not synthetic
- **Edge Cases**: Test with missing data, NaN values, empty datasets
- **Validation Block**: Include `if __name__ == "__main__"` validation
- **Expected Results**: Verify signals against known outcomes

### 4. Configuration

- **Meaningful Names**: Use descriptive strategy instance names
- **Tags**: Tag strategies for easy filtering (e.g., "momentum", "mean_reversion")
- **Descriptions**: Document what each configuration variant does
- **Version Control**: Keep configs in version control

## Examples

See `/examples/strategy_framework_example.py` for comprehensive usage examples including:
- Loading strategies from YAML
- Creating custom strategies
- Generating signals
- Using the registry

Run the example:
```bash
uv run python examples/strategy_framework_example.py
```

## API Reference

### Classes

- `BaseStrategy`: Abstract base class for all strategies
- `StrategyRegistry`: Plugin system for strategy management
- `StrategyLoader`: Configuration loading and instantiation
- `StrategyConfig`: Pydantic model for config validation
- `SignalType`: Enum for trading signals (BUY, SELL, HOLD)

### Functions

- `register_strategy()`: Decorator for strategy registration
- `get_strategy(name)`: Retrieve a strategy class by name
- `list_strategies(tags)`: List registered strategies, optionally filtered
- `get_registry()`: Get the global registry instance
- `load_strategies_from_yaml(path)`: Load strategies from YAML file

## Advanced Topics

### Custom Validation

Override `validate_data()` for custom requirements:

```python
def validate_data(self, data):
    # Call parent validation first
    if not super().validate_data(data):
        return False

    # Custom checks
    if len(data) < self.min_periods:
        logger.error(f"Insufficient data: {len(data)} < {self.min_periods}")
        return False

    return True
```

### Dynamic Indicators

Specify required indicators dynamically:

```python
def get_required_indicators(self):
    return [
        f"{self.ma_type}_{self.fast_period}",
        f"{self.ma_type}_{self.slow_period}"
    ]
```

### State Management

Use instance variables for strategy state:

```python
def __init__(self, name, config=None):
    super().__init__(name, config)
    self.position = None  # Current position
    self.last_signal = None  # Last signal generated

def generate_signals(self, data):
    # Use state in signal generation
    if self.position == 'LONG':
        # Different logic for existing position
        pass
```

### Multi-Timeframe Strategies

Access multiple timeframes in your strategy:

```python
def generate_signals(self, data):
    # Assume data contains multiple timeframes
    daily = data[data['timeframe'] == '1d']
    hourly = data[data['timeframe'] == '1h']

    # Combine signals from multiple timeframes
    # ...
```

## Troubleshooting

### Strategy Not Found

```
KeyError: "Strategy 'MyStrategy' not found in registry"
```

**Solutions**:
1. Ensure the strategy module is imported before loading configs
2. Check that `@register_strategy` decorator is applied
3. Verify the class name matches the config file

### Missing Indicators

```
ValueError: Required indicators not found: SMA_20, RSI_14
```

**Solutions**:
1. Calculate required indicators before calling `generate_signals()`
2. Check indicator column names match exactly
3. Use `get_required_indicators()` to see what's needed

### Invalid Configuration

```
ValidationError: Strategy config validation failed
```

**Solutions**:
1. Check YAML syntax is valid
2. Ensure all required fields are present (name, class)
3. Verify parameter types match expectations

## Contributing

When adding new strategies to the framework:

1. Create strategy file in `src/crypto_trader/strategies/library/`
2. Inherit from `BaseStrategy`
3. Implement all required methods
4. Add `@register_strategy` decorator with appropriate tags
5. Include validation block (`if __name__ == "__main__"`)
6. Add documentation and example config
7. Update this README with strategy details

## License

Part of the crypto-trader project. See main project LICENSE for details.
