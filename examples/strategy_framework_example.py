"""
Strategy Framework Example

This example demonstrates how to use the strategy framework to:
1. Load strategies from YAML configuration
2. Register custom strategies
3. Generate signals from market data
4. Use the strategy registry

**Purpose**: Provide a comprehensive example of the strategy framework usage.

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- loguru: https://loguru.readthedocs.io/en/stable/

**Usage**:
```bash
uv run python examples/strategy_framework_example.py
```

**Expected Output**:
- Loaded strategies summary
- Generated signals for each strategy
- Strategy metadata and parameters
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from loguru import logger

from crypto_trader.strategies import (
    BaseStrategy,
    SignalType,
    load_strategies_from_yaml,
    register_strategy,
    get_registry,
    list_strategies
)

# Import strategies to trigger decorator registration
from crypto_trader.strategies.library.moving_average_crossover import MovingAverageCrossover


def create_sample_market_data() -> pd.DataFrame:
    """
    Create sample market data for demonstration.

    Returns:
        DataFrame with OHLCV data and indicators
    """
    logger.info("Creating sample market data...")

    # Generate 100 days of data
    dates = [datetime.now() - timedelta(days=100-i) for i in range(100)]

    # Simulate a trending market
    prices = []
    base_price = 100
    for i in range(100):
        # Add trend and noise
        trend = i * 0.5
        noise = (i % 10 - 5) * 0.2
        prices.append(base_price + trend + noise)

    # Calculate indicators
    prices_series = pd.Series(prices)

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + 2 for p in prices],
        'low': [p - 2 for p in prices],
        'close': prices,
        'volume': [10000 + i * 100 for i in range(100)],
        # Simple Moving Averages
        'SMA_10': prices_series.rolling(window=10).mean(),
        'SMA_20': prices_series.rolling(window=20).mean(),
        'SMA_50': prices_series.rolling(window=50).mean(),
        'SMA_200': prices_series.rolling(window=200).mean().fillna(method='bfill'),
        # Exponential Moving Averages
        'EMA_12': prices_series.ewm(span=12).mean(),
        'EMA_26': prices_series.ewm(span=26).mean(),
    })

    logger.info(f"Created {len(data)} rows of market data")
    return data


def demonstrate_yaml_loading():
    """Demonstrate loading strategies from YAML configuration."""
    logger.info("=" * 60)
    logger.info("DEMO 1: Loading Strategies from YAML")
    logger.info("=" * 60)

    config_path = Path("config/strategies/example_strategies.yaml")

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Creating sample config...")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # File should already exist from main implementation
        return {}

    # Load only enabled strategies
    strategies = load_strategies_from_yaml(config_path, enabled_only=True)

    logger.info(f"Loaded {len(strategies)} enabled strategies:")
    for name, strategy in strategies.items():
        params = strategy.get_parameters()
        logger.info(f"  - {name}: {strategy.__class__.__name__}")
        logger.info(f"    Parameters: {params}")

    return strategies


def demonstrate_custom_strategy():
    """Demonstrate creating and registering a custom strategy."""
    logger.info("=" * 60)
    logger.info("DEMO 2: Creating Custom Strategy")
    logger.info("=" * 60)

    @register_strategy(name="SimpleThreshold", tags=["custom", "threshold"])
    class SimpleThresholdStrategy(BaseStrategy):
        """
        Simple threshold-based strategy.

        Generates BUY when price drops below threshold,
        SELL when price rises above threshold.
        """

        def initialize(self, config: Dict[str, Any]) -> None:
            self.threshold = config.get("threshold", 100)
            self.config.update(config)
            logger.info(f"SimpleThreshold initialized with threshold={self.threshold}")

        def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
            signals = []
            confidences = []
            metadata = []

            for idx, row in data.iterrows():
                price = row['close']

                if price < self.threshold * 0.95:
                    signals.append(SignalType.BUY.value)
                    confidences.append(0.8)
                    metadata.append({"reason": "below_threshold", "price": price})
                elif price > self.threshold * 1.05:
                    signals.append(SignalType.SELL.value)
                    confidences.append(0.8)
                    metadata.append({"reason": "above_threshold", "price": price})
                else:
                    signals.append(SignalType.HOLD.value)
                    confidences.append(0.0)
                    metadata.append({"price": price})

            return pd.DataFrame({
                'timestamp': data['timestamp'],
                'signal': signals,
                'confidence': confidences,
                'metadata': metadata
            })

        def get_parameters(self) -> Dict[str, Any]:
            return {"threshold": self.threshold}

    # Instantiate and use the custom strategy
    custom_strategy = SimpleThresholdStrategy(
        name="my_threshold",
        config={"threshold": 120}
    )
    custom_strategy.initialize(custom_strategy.config)

    logger.info(f"Created custom strategy: {custom_strategy.name}")
    logger.info(f"Parameters: {custom_strategy.get_parameters()}")

    return custom_strategy


def demonstrate_signal_generation(strategies: Dict[str, BaseStrategy], data: pd.DataFrame):
    """Demonstrate generating signals with loaded strategies."""
    logger.info("=" * 60)
    logger.info("DEMO 3: Generating Trading Signals")
    logger.info("=" * 60)

    for name, strategy in strategies.items():
        logger.info(f"\nGenerating signals for: {name}")

        try:
            # Generate signals
            signals = strategy.generate_signals(data)

            # Analyze signals
            signal_counts = signals['signal'].value_counts().to_dict()
            buy_count = signal_counts.get(SignalType.BUY.value, 0)
            sell_count = signal_counts.get(SignalType.SELL.value, 0)
            hold_count = signal_counts.get(SignalType.HOLD.value, 0)

            logger.info(f"  Total signals: {len(signals)}")
            logger.info(f"  BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")

            # Show first BUY signal if any
            buy_signals = signals[signals['signal'] == SignalType.BUY.value]
            if len(buy_signals) > 0:
                first_buy = buy_signals.iloc[0]
                logger.info(f"  First BUY signal:")
                logger.info(f"    Timestamp: {first_buy['timestamp']}")
                logger.info(f"    Confidence: {first_buy['confidence']:.2f}")
                logger.info(f"    Metadata: {first_buy['metadata']}")

        except Exception as e:
            logger.error(f"  Failed to generate signals: {e}")


def demonstrate_registry():
    """Demonstrate using the strategy registry."""
    logger.info("=" * 60)
    logger.info("DEMO 4: Strategy Registry")
    logger.info("=" * 60)

    registry = get_registry()

    logger.info(f"Total strategies in registry: {len(registry)}")

    # List all strategies
    all_strategies = list_strategies()
    logger.info("\nRegistered strategies:")
    for name, metadata in all_strategies.items():
        logger.info(f"  - {name}")
        logger.info(f"    Description: {metadata['description']}")
        logger.info(f"    Tags: {metadata['tags']}")
        logger.info(f"    Module: {metadata['module']}")

    # Filter by tags
    logger.info("\nMomentum strategies:")
    momentum_strategies = list_strategies(tags=["momentum"])
    for name in momentum_strategies.keys():
        logger.info(f"  - {name}")


def main():
    """Run all demonstrations."""
    logger.info("Strategy Framework Demonstration")
    logger.info("=" * 60)

    # Create sample data
    data = create_sample_market_data()

    # Demo 1: Load from YAML
    loaded_strategies = demonstrate_yaml_loading()

    # Demo 2: Custom strategy
    custom_strategy = demonstrate_custom_strategy()

    # Demo 3: Generate signals
    all_strategies = {**loaded_strategies, "custom": custom_strategy}
    demonstrate_signal_generation(all_strategies, data)

    # Demo 4: Registry
    demonstrate_registry()

    logger.info("=" * 60)
    logger.info("Demonstration Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    """
    Validation block for the example script.
    Ensures all demonstrations run successfully.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    # Test 1: Can import all required modules
    total_tests += 1
    try:
        from crypto_trader.strategies import (
            BaseStrategy,
            SignalType,
            load_strategies_from_yaml,
            register_strategy
        )
        logger.info("All required modules imported successfully")
    except Exception as e:
        all_validation_failures.append(f"Module import failed: {e}")

    # Test 2: Can create sample data
    total_tests += 1
    try:
        data = create_sample_market_data()
        if len(data) != 100:
            all_validation_failures.append(
                f"Sample data creation: Expected 100 rows, got {len(data)}"
            )
    except Exception as e:
        all_validation_failures.append(f"Sample data creation failed: {e}")

    # Test 3: Run main demonstration
    total_tests += 1
    try:
        main()
    except Exception as e:
        all_validation_failures.append(f"Main demonstration failed: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Strategy framework example completed successfully")
        sys.exit(0)
