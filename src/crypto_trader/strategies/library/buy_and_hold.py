"""
Buy and Hold Strategy

This module implements the classic buy-and-hold investment strategy for cryptocurrency trading.
The strategy generates HOLD signals for every candle, simulating buying at the start of the
analysis window and holding the position through the entire period until selling at the end.

**Purpose**: Implement a passive investment strategy that serves as a baseline benchmark for
comparing active trading strategies.

**Strategy Type**: Passive Buy and Hold
**Indicators**: None required
**Entry Signal**: Buy at start (first candle)
**Exit Signal**: Sell at end (last candle)
**Behavior**: Generate HOLD signal for every candle in between

**Methodology**:
The buy-and-hold strategy is a passive investment approach where an investor:
1. Buys an asset at the beginning of the period
2. Holds the position regardless of market fluctuations
3. Sells only at the end of the analysis period

This strategy serves as a crucial benchmark - active strategies should ideally outperform
buy-and-hold after accounting for transaction costs and risk. In the backtesting context,
HOLD signals mean: maintain the position throughout the window.

**Parameters**:
None - this strategy has no configurable parameters

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```python
data = pd.DataFrame({
    'timestamp': [...],
    'open': [100, 101, 102, ...],
    'high': [105, 106, 107, ...],
    'low': [99, 100, 101, ...],
    'close': [103, 104, 105, ...],
    'volume': [1000, 1100, 1200, ...]
})
```

**Expected Output**:
```python
signals = pd.DataFrame({
    'timestamp': [...],
    'signal': ['BUY', 'HOLD', 'HOLD', ...],  # BUY at first candle, HOLD for rest
    'confidence': [1.0, 1.0, 1.0, ...],
    'metadata': [
        {'reason': 'buy_and_hold', 'strategy': 'passive', 'position': 'enter'},
        {'reason': 'buy_and_hold', 'strategy': 'passive', 'position': 'maintain'},
        {'reason': 'buy_and_hold', 'strategy': 'passive', 'position': 'maintain'},
        ...
    ]
})
```
"""

from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy


@register_strategy(
    name="BuyAndHold",
    description="Passive buy-and-hold strategy (benchmark)",
    tags=["passive", "benchmark", "baseline"]
)
class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy and Hold Strategy.

    Generates HOLD signals for every candle, simulating a passive investment
    approach where the asset is purchased at the start and held through the
    entire analysis period.

    This strategy serves as a baseline benchmark for evaluating active trading
    strategies. Any active strategy should ideally outperform buy-and-hold
    after accounting for transaction costs and risk adjustments.
    """

    def __init__(self, name: str = "BuyAndHold", config: Dict[str, Any] = None):
        """
        Initialize the Buy and Hold strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary (no parameters needed for this strategy)
        """
        super().__init__(name, config)
        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        This strategy has no configurable parameters, but we implement
        the method to satisfy the BaseStrategy interface.

        Args:
            config: Dictionary with configuration (ignored for this strategy)
        """
        # No parameters to validate for buy-and-hold
        self._initialized = True
        logger.info(f"{self.name} initialized (no parameters required)")

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Empty dictionary - this strategy has no parameters
        """
        return {}

    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators.

        Returns:
            Empty list - no indicators needed for buy-and-hold
        """
        return []

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for buy-and-hold strategy.

        This method returns a BUY signal for the first candle to enter the position,
        then HOLD signals for all remaining candles to maintain the position throughout
        the period.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with columns: ['timestamp', 'signal', 'confidence', 'metadata']
            First signal is BUY, remaining signals are HOLD, all with confidence 1.0

        Raises:
            ValueError: If data is invalid
        """
        # Handle empty data case first
        if len(data) == 0:
            logger.warning("Empty data provided to buy-and-hold strategy")
            return pd.DataFrame(columns=['timestamp', 'signal', 'confidence', 'metadata'])

        # Validate data
        if not self.validate_data(data):
            raise ValueError("Invalid data provided to generate_signals")

        # Create signals: BUY at first candle, HOLD for remaining
        num_rows = len(data)

        # First signal is BUY to enter position, rest are HOLD to maintain it
        signals = [SignalType.BUY.value] + [SignalType.HOLD.value] * (num_rows - 1)
        confidences = [1.0] * num_rows  # Full confidence in holding

        # Create metadata: first indicates entry, rest indicate maintain
        metadata = [{'reason': 'buy_and_hold', 'strategy': 'passive', 'position': 'enter'}]
        metadata.extend([
            {
                'reason': 'buy_and_hold',
                'strategy': 'passive',
                'position': 'maintain'
            }
        ] * (num_rows - 1))

        # Create result DataFrame
        result = pd.DataFrame({
            'timestamp': data.index if isinstance(data.index, pd.DatetimeIndex) else data['timestamp'],
            'signal': signals,
            'confidence': confidences,
            'metadata': metadata
        })

        logger.info(f"Generated 1 BUY + {num_rows - 1} HOLD signals for buy-and-hold strategy")

        return result


if __name__ == "__main__":
    """
    Validation block for Buy and Hold Strategy.
    Tests the strategy with sample OHLCV data and real BTC/USDT data.
    """
    import sys
    from datetime import datetime, timedelta

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting Buy and Hold Strategy validation")

    # Test 1: Initialize strategy
    total_tests += 1
    try:
        strategy = BuyAndHoldStrategy()
        strategy.initialize({})

        params = strategy.get_parameters()
        if params != {}:
            all_validation_failures.append(
                f"Test 1: Expected empty params, got {params}"
            )

        logger.success("Test 1 PASSED: Strategy initialized")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Verify no required indicators
    total_tests += 1
    try:
        required = strategy.get_required_indicators()
        if required != []:
            all_validation_failures.append(
                f"Test 2: Expected no indicators, got {required}"
            )

        logger.success("Test 2 PASSED: No indicators required")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Generate signals with sample data
    total_tests += 1
    try:
        # Create sample OHLCV data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'open': [100 + i for i in range(100)],
            'high': [105 + i for i in range(100)],
            'low': [99 + i for i in range(100)],
            'close': [103 + i for i in range(100)],
            'volume': [1000 + i*10 for i in range(100)]
        })

        signals = strategy.generate_signals(sample_data)

        if signals is None or signals.empty:
            all_validation_failures.append("Test 3: No signals generated")
        elif len(signals) != 100:
            all_validation_failures.append(
                f"Test 3: Expected 100 signals, got {len(signals)}"
            )
        else:
            logger.success(f"Test 3 PASSED: Generated {len(signals)} signals")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Verify first signal is BUY, rest are HOLD
    total_tests += 1
    try:
        if 'signals' in locals() and signals is not None and not signals.empty:
            first_signal = signals.iloc[0]['signal']
            remaining_signals = signals.iloc[1:]['signal'].unique()

            if first_signal != SignalType.BUY.value:
                all_validation_failures.append(
                    f"Test 4: Expected first signal to be BUY, got {first_signal}"
                )
            elif len(remaining_signals) != 1 or remaining_signals[0] != SignalType.HOLD.value:
                all_validation_failures.append(
                    f"Test 4: Expected remaining signals to be HOLD, got {remaining_signals.tolist()}"
                )
            else:
                logger.success("Test 4 PASSED: First signal is BUY, remaining are HOLD")
        else:
            all_validation_failures.append("Test 4: No signals available from Test 3")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception raised: {e}")

    # Test 5: Verify confidence values
    total_tests += 1
    try:
        if 'signals' in locals() and signals is not None and not signals.empty:
            unique_confidences = signals['confidence'].unique()

            if len(unique_confidences) != 1:
                all_validation_failures.append(
                    f"Test 5: Expected single confidence value, got {unique_confidences.tolist()}"
                )
            elif unique_confidences[0] != 1.0:
                all_validation_failures.append(
                    f"Test 5: Expected confidence=1.0, got {unique_confidences[0]}"
                )
            else:
                logger.success("Test 5 PASSED: All confidence values are 1.0")
        else:
            all_validation_failures.append("Test 5: No signals available")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Verify metadata structure
    total_tests += 1
    try:
        if 'signals' in locals() and signals is not None and not signals.empty:
            first_metadata = signals.iloc[0]['metadata']
            second_metadata = signals.iloc[1]['metadata'] if len(signals) > 1 else None

            # Check first signal metadata (should have position='enter')
            if 'reason' not in first_metadata:
                all_validation_failures.append("Test 6: First metadata missing 'reason'")
            elif first_metadata['reason'] != 'buy_and_hold':
                all_validation_failures.append(
                    f"Test 6: Expected reason='buy_and_hold', got '{first_metadata['reason']}'"
                )

            if 'strategy' not in first_metadata:
                all_validation_failures.append("Test 6: First metadata missing 'strategy'")
            elif first_metadata['strategy'] != 'passive':
                all_validation_failures.append(
                    f"Test 6: Expected strategy='passive', got '{first_metadata['strategy']}'"
                )

            if 'position' not in first_metadata:
                all_validation_failures.append("Test 6: First metadata missing 'position'")
            elif first_metadata['position'] != 'enter':
                all_validation_failures.append(
                    f"Test 6: Expected first position='enter', got '{first_metadata['position']}'"
                )

            # Check second signal metadata if it exists (should have position='maintain')
            if second_metadata is not None:
                if second_metadata.get('position') != 'maintain':
                    all_validation_failures.append(
                        f"Test 6: Expected second position='maintain', got '{second_metadata.get('position')}'"
                    )

            if not all_validation_failures or len([f for f in all_validation_failures if 'Test 6' in f]) == 0:
                logger.success("Test 6 PASSED: Metadata structure correct")
        else:
            all_validation_failures.append("Test 6: No signals available")
    except Exception as e:
        all_validation_failures.append(f"Test 6: Exception raised: {e}")

    # Test 7: Test with real BTC/USDT data
    total_tests += 1
    try:
        from crypto_trader.data.fetchers import BinanceDataFetcher

        logger.info("Fetching BTC/USDT data for real-world test...")
        fetcher = BinanceDataFetcher(use_storage=False, use_cache=False)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        real_data = fetcher.get_ohlcv(
            "BTC/USDT",
            "1d",
            start_date=start_date,
            end_date=end_date,
            limit=30
        )

        if real_data is None or real_data.empty:
            all_validation_failures.append("Test 7: Failed to fetch real data")
        else:
            real_data_reset = real_data.reset_index()
            real_signals = strategy.generate_signals(real_data_reset)

            if real_signals is None or real_signals.empty:
                all_validation_failures.append("Test 7: No signals from real data")
            elif len(real_signals) != len(real_data):
                all_validation_failures.append(
                    f"Test 7: Signal count mismatch - data: {len(real_data)}, signals: {len(real_signals)}"
                )
            elif real_signals.iloc[0]['signal'] != SignalType.BUY.value:
                all_validation_failures.append(
                    f"Test 7: First signal should be BUY, got {real_signals.iloc[0]['signal']}"
                )
            elif not all(s == SignalType.HOLD.value for s in real_signals.iloc[1:]['signal']):
                all_validation_failures.append(
                    "Test 7: Remaining signals should all be HOLD with real data"
                )
            else:
                logger.success(f"Test 7 PASSED: Generated {len(real_signals)} HOLD signals from real BTC/USDT data")
                logger.info(f"  Data range: {real_data.index.min()} to {real_data.index.max()}")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Test 8: Test with different timeframes
    total_tests += 1
    try:
        timeframe_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='4h'),
            'open': [200 + i*2 for i in range(50)],
            'high': [210 + i*2 for i in range(50)],
            'low': [195 + i*2 for i in range(50)],
            'close': [205 + i*2 for i in range(50)],
            'volume': [2000 + i*20 for i in range(50)]
        })

        tf_signals = strategy.generate_signals(timeframe_data)

        if tf_signals is None or tf_signals.empty:
            all_validation_failures.append("Test 8: No signals for different timeframe")
        elif len(tf_signals) != 50:
            all_validation_failures.append(
                f"Test 8: Expected 50 signals, got {len(tf_signals)}"
            )
        elif tf_signals.iloc[0]['signal'] != SignalType.BUY.value:
            all_validation_failures.append(
                f"Test 8: First signal should be BUY, got {tf_signals.iloc[0]['signal']}"
            )
        elif not all(s == SignalType.HOLD.value for s in tf_signals.iloc[1:]['signal']):
            all_validation_failures.append(
                "Test 8: Remaining signals should be HOLD for 4h timeframe"
            )
        else:
            logger.success("Test 8 PASSED: Works with different timeframes")
    except Exception as e:
        all_validation_failures.append(f"Test 8: Exception raised: {e}")

    # Test 9: Test with minimal data (edge case)
    total_tests += 1
    try:
        minimal_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1, freq='1d'),
            'open': [100],
            'high': [105],
            'low': [99],
            'close': [103],
            'volume': [1000]
        })

        minimal_signals = strategy.generate_signals(minimal_data)

        if minimal_signals is None or minimal_signals.empty:
            all_validation_failures.append("Test 9: No signals for minimal data")
        elif len(minimal_signals) != 1:
            all_validation_failures.append(
                f"Test 9: Expected 1 signal, got {len(minimal_signals)}"
            )
        elif minimal_signals.iloc[0]['signal'] != SignalType.BUY.value:
            all_validation_failures.append(
                f"Test 9: Expected BUY signal for single candle, got {minimal_signals.iloc[0]['signal']}"
            )
        else:
            logger.success("Test 9 PASSED: Handles minimal data (single row)")
    except Exception as e:
        all_validation_failures.append(f"Test 9: Exception raised: {e}")

    # Test 10: Test empty data handling (edge case)
    total_tests += 1
    try:
        empty_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        empty_signals = strategy.generate_signals(empty_data)

        if empty_signals is None:
            all_validation_failures.append("Test 10: Should return DataFrame, not None")
        elif not empty_signals.empty:
            all_validation_failures.append(
                f"Test 10: Expected empty signals, got {len(empty_signals)} rows"
            )
        else:
            logger.success("Test 10 PASSED: Handles empty data correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 10: Exception raised: {e}")

    # Test 11: Test strategy registration
    total_tests += 1
    try:
        from crypto_trader.strategies.registry import get_registry

        registry = get_registry()
        if "BuyAndHold" not in registry:
            all_validation_failures.append(
                "Test 11: Strategy not registered in global registry"
            )
        else:
            registered_class = registry.get_strategy("BuyAndHold")
            if registered_class is not BuyAndHoldStrategy:
                all_validation_failures.append(
                    "Test 11: Wrong class registered in registry"
                )
            else:
                logger.success("Test 11 PASSED: Strategy registered correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 11: Exception raised: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Buy and Hold Strategy validated with sample and real BTC/USDT data")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
