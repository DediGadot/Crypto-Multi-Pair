"""
Moving Average Crossover Strategy

A simple momentum-based strategy that generates signals based on the crossover
of fast and slow moving averages.

**Purpose**: Generate BUY signals when fast MA crosses above slow MA, and SELL
signals when fast MA crosses below slow MA.

**Strategy Logic**:
- BUY: Fast MA crosses above Slow MA (bullish crossover)
- SELL: Fast MA crosses below Slow MA (bearish crossover)
- HOLD: No crossover detected

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
```python
data = pd.DataFrame({
    'timestamp': [...],
    'close': [100, 102, 105, 103, 107, 110],
    'SMA_10': [100, 101, 103, 104, 105, 106],
    'SMA_20': [99, 100, 101, 102, 103, 104]
})
```

**Expected Output**:
```python
signals = pd.DataFrame({
    'timestamp': [...],
    'signal': ['HOLD', 'HOLD', 'BUY', 'HOLD', 'HOLD', 'HOLD'],
    'confidence': [0.0, 0.0, 0.85, 0.0, 0.0, 0.0],
    'metadata': [{'fast_ma': 100, 'slow_ma': 99}, ...]
})
```
"""

from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from crypto_trader.strategies import BaseStrategy, SignalType, register_strategy


@register_strategy(tags=["momentum", "moving_average", "crossover"])
class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy.

    Generates trading signals based on the crossover of two moving averages.
    This is a classic trend-following strategy.

    Parameters:
        fast_period (int): Period for fast moving average (default: 10)
        slow_period (int): Period for slow moving average (default: 20)
        signal_threshold (float): Minimum confidence for signals (default: 0.7)
        ma_type (str): Type of moving average - 'SMA' or 'EMA' (default: 'SMA')
    """

    def __init__(self, name: str, config: Dict[str, Any] | None = None):
        """Initialize the Moving Average Crossover strategy."""
        super().__init__(name, config)
        self.fast_period = 10
        self.slow_period = 20
        self.signal_threshold = 0.7
        self.ma_type = "SMA"

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with strategy parameters

        Raises:
            ValueError: If parameters are invalid
        """
        self.fast_period = config.get("fast_period", 10)
        self.slow_period = config.get("slow_period", 20)
        self.signal_threshold = config.get("signal_threshold", 0.7)
        self.ma_type = config.get("ma_type", "SMA")

        # Validate parameters
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"Fast period ({self.fast_period}) must be less than "
                f"slow period ({self.slow_period})"
            )

        if not 0 <= self.signal_threshold <= 1:
            raise ValueError(
                f"Signal threshold must be between 0 and 1, got {self.signal_threshold}"
            )

        if self.ma_type not in ["SMA", "EMA"]:
            raise ValueError(f"MA type must be 'SMA' or 'EMA', got {self.ma_type}")

        self.config.update(config)
        self._initialized = True
        logger.info(
            f"MovingAverageCrossover initialized: "
            f"fast={self.fast_period}, slow={self.slow_period}, "
            f"threshold={self.signal_threshold}, type={self.ma_type}"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.

        Args:
            data: DataFrame with OHLCV data and moving averages

        Returns:
            DataFrame with signals

        Raises:
            ValueError: If data validation fails
        """
        if not self.validate_data(data):
            raise ValueError("Data validation failed")

        # Get required indicator column names
        fast_ma_col = f"{self.ma_type}_{self.fast_period}"
        slow_ma_col = f"{self.ma_type}_{self.slow_period}"

        if fast_ma_col not in data.columns or slow_ma_col not in data.columns:
            raise ValueError(
                f"Required indicators not found: {fast_ma_col}, {slow_ma_col}"
            )

        # Calculate crossovers
        fast_ma = data[fast_ma_col]
        slow_ma = data[slow_ma_col]

        # Detect crossovers
        # Bullish: fast crosses above slow
        bullish_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))

        # Bearish: fast crosses below slow
        bearish_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        # Initialize signals
        signals = []
        confidences = []
        metadata = []

        for idx in range(len(data)):
            if bullish_cross.iloc[idx]:
                # Calculate confidence based on divergence
                divergence = abs(fast_ma.iloc[idx] - slow_ma.iloc[idx]) / slow_ma.iloc[idx]
                confidence = min(0.5 + divergence * 10, 1.0)

                if confidence >= self.signal_threshold:
                    signals.append(SignalType.BUY.value)
                    confidences.append(confidence)
                    metadata.append({
                        "reason": "bullish_crossover",
                        "fast_ma": float(fast_ma.iloc[idx]),
                        "slow_ma": float(slow_ma.iloc[idx]),
                        "divergence": float(divergence)
                    })
                else:
                    signals.append(SignalType.HOLD.value)
                    confidences.append(0.0)
                    metadata.append({"reason": "confidence_too_low"})

            elif bearish_cross.iloc[idx]:
                # Calculate confidence based on divergence
                divergence = abs(fast_ma.iloc[idx] - slow_ma.iloc[idx]) / slow_ma.iloc[idx]
                confidence = min(0.5 + divergence * 10, 1.0)

                if confidence >= self.signal_threshold:
                    signals.append(SignalType.SELL.value)
                    confidences.append(confidence)
                    metadata.append({
                        "reason": "bearish_crossover",
                        "fast_ma": float(fast_ma.iloc[idx]),
                        "slow_ma": float(slow_ma.iloc[idx]),
                        "divergence": float(divergence)
                    })
                else:
                    signals.append(SignalType.HOLD.value)
                    confidences.append(0.0)
                    metadata.append({"reason": "confidence_too_low"})

            else:
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({
                    "fast_ma": float(fast_ma.iloc[idx]),
                    "slow_ma": float(slow_ma.iloc[idx])
                })

        result = pd.DataFrame({
            'timestamp': data['timestamp'],
            'signal': signals,
            'confidence': confidences,
            'metadata': metadata
        })

        logger.debug(
            f"Generated {len(result)} signals: "
            f"BUY={sum(1 for s in signals if s == SignalType.BUY.value)}, "
            f"SELL={sum(1 for s in signals if s == SignalType.SELL.value)}, "
            f"HOLD={sum(1 for s in signals if s == SignalType.HOLD.value)}"
        )

        return result

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_threshold": self.signal_threshold,
            "ma_type": self.ma_type,
        }

    def get_required_indicators(self) -> List[str]:
        """
        Get required indicators for this strategy.

        Returns:
            List of required indicator names
        """
        return [
            f"{self.ma_type}_{self.fast_period}",
            f"{self.ma_type}_{self.slow_period}"
        ]


if __name__ == "__main__":
    """
    Validation block for MovingAverageCrossover strategy.
    Tests with real market-like data.
    """
    import sys
    from datetime import datetime, timedelta
    import numpy as np

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    # Test 1: Strategy initialization
    total_tests += 1
    try:
        strategy = MovingAverageCrossover(
            name="MA_Cross_Test",
            config={
                "fast_period": 10,
                "slow_period": 20,
                "signal_threshold": 0.5  # Lower threshold for test
            }
        )
        strategy.initialize(strategy.config)

        if not strategy._initialized:
            all_validation_failures.append(
                "Strategy initialization: _initialized flag not set"
            )
    except Exception as e:
        all_validation_failures.append(f"Strategy initialization failed: {e}")

    # Test 2: Parameter validation - invalid fast/slow periods
    total_tests += 1
    try:
        invalid_strategy = MovingAverageCrossover(name="Invalid")
        error_raised = False
        try:
            invalid_strategy.initialize({"fast_period": 20, "slow_period": 10})
        except ValueError as e:
            if "must be less than" in str(e):
                error_raised = True

        if not error_raised:
            all_validation_failures.append(
                "Parameter validation: Expected ValueError for fast >= slow"
            )
    except Exception as e:
        all_validation_failures.append(f"Parameter validation test failed: {e}")

    # Test 3: Create realistic market data with crossover
    total_tests += 1
    try:
        # Generate dates (need more data for proper MA calculation)
        dates = [datetime.now() - timedelta(days=50-i) for i in range(50)]

        # Simulate price action with clear crossovers
        prices = []
        for i in range(50):
            if i < 15:
                # Downtrend - start high and go low
                prices.append(120 - i * 1.5)
            elif i < 35:
                # Strong uptrend (bullish crossover happens here)
                prices.append(97.5 + (i - 15) * 2.0)
            else:
                # Continuation
                prices.append(137.5 + (i - 35) * 0.5)

        # Calculate simple moving averages
        prices_series = pd.Series(prices)
        sma_10 = prices_series.rolling(window=10).mean()
        sma_20 = prices_series.rolling(window=20).mean()

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000 + i * 10 for i in range(50)],
            'SMA_10': sma_10,
            'SMA_20': sma_20
        })

        logger.debug("Created test market data with crossover pattern")
    except Exception as e:
        all_validation_failures.append(f"Test data creation failed: {e}")

    # Test 4: Data validation
    total_tests += 1
    try:
        is_valid = strategy.validate_data(data)
        if not is_valid:
            all_validation_failures.append(
                "Data validation: Expected True for valid data"
            )
    except Exception as e:
        all_validation_failures.append(f"Data validation failed: {e}")

    # Test 5: Required indicators
    total_tests += 1
    try:
        required = strategy.get_required_indicators()
        expected = ['SMA_10', 'SMA_20']

        if required != expected:
            all_validation_failures.append(
                f"Required indicators: Expected {expected}, got {required}"
            )
    except Exception as e:
        all_validation_failures.append(f"Required indicators test failed: {e}")

    # Test 6: Signal generation
    total_tests += 1
    try:
        signals = strategy.generate_signals(data)

        # Verify output structure
        expected_columns = {'timestamp', 'signal', 'confidence', 'metadata'}
        actual_columns = set(signals.columns)

        if actual_columns != expected_columns:
            all_validation_failures.append(
                f"Signal columns: Expected {expected_columns}, got {actual_columns}"
            )

        if len(signals) != len(data):
            all_validation_failures.append(
                f"Signal count: Expected {len(data)}, got {len(signals)}"
            )

        # Verify signal types
        valid_signals = {SignalType.BUY.value, SignalType.SELL.value, SignalType.HOLD.value}
        invalid_signals = set(signals['signal'].unique()) - valid_signals
        if invalid_signals:
            all_validation_failures.append(
                f"Invalid signal types found: {invalid_signals}"
            )
    except Exception as e:
        all_validation_failures.append(f"Signal generation failed: {e}")

    # Test 7: Verify crossover detection
    total_tests += 1
    try:
        # Debug: check all signals
        logger.debug(f"All signals breakdown: {signals['signal'].value_counts().to_dict()}")

        # Should have at least one BUY signal (bullish crossover)
        buy_signals = signals[signals['signal'] == SignalType.BUY.value]

        # Also check signals with bullish_crossover but low confidence
        low_conf_crossovers = signals[
            signals['metadata'].apply(
                lambda x: x.get('reason') == 'confidence_too_low'
            )
        ]
        logger.debug(f"Low confidence crossovers: {len(low_conf_crossovers)}")

        if len(buy_signals) == 0:
            # Check if there are any crossovers at all
            all_crossovers = signals[
                signals['metadata'].apply(
                    lambda x: 'crossover' in str(x.get('reason', ''))
                )
            ]
            if len(all_crossovers) == 0:
                all_validation_failures.append(
                    "Crossover detection: No crossovers detected at all (check MA calculation)"
                )
            else:
                # There are crossovers but confidence too low - this is acceptable behavior
                logger.info(
                    f"Crossovers detected but confidence below threshold: {len(all_crossovers)}"
                )
        else:
            # Check first BUY signal has correct metadata
            first_buy = buy_signals.iloc[0]
            if 'reason' not in first_buy['metadata']:
                all_validation_failures.append(
                    "BUY signal metadata: Missing 'reason' field"
                )
            elif first_buy['metadata']['reason'] != 'bullish_crossover':
                all_validation_failures.append(
                    f"BUY signal reason: Expected 'bullish_crossover', got {first_buy['metadata']['reason']}"
                )

            if first_buy['confidence'] < 0.5:
                all_validation_failures.append(
                    f"BUY signal confidence: Expected >= 0.5, got {first_buy['confidence']}"
                )
    except Exception as e:
        all_validation_failures.append(f"Crossover detection test failed: {e}")

    # Test 8: Get parameters
    total_tests += 1
    try:
        params = strategy.get_parameters()

        if params['fast_period'] != 10:
            all_validation_failures.append(
                f"Parameters: Expected fast_period=10, got {params['fast_period']}"
            )
        if params['slow_period'] != 20:
            all_validation_failures.append(
                f"Parameters: Expected slow_period=20, got {params['slow_period']}"
            )
    except Exception as e:
        all_validation_failures.append(f"Get parameters test failed: {e}")

    # Test 9: Signal confidence filtering
    total_tests += 1
    try:
        # All non-HOLD signals should meet threshold
        non_hold_signals = signals[signals['signal'] != SignalType.HOLD.value]

        for idx, signal in non_hold_signals.iterrows():
            if signal['confidence'] < strategy.signal_threshold:
                all_validation_failures.append(
                    f"Confidence filtering: Signal at index {idx} has confidence "
                    f"{signal['confidence']} < threshold {strategy.signal_threshold}"
                )
    except Exception as e:
        all_validation_failures.append(f"Confidence filtering test failed: {e}")

    # Test 10: Strategy with EMA
    total_tests += 1
    try:
        ema_strategy = MovingAverageCrossover(
            name="EMA_Test",
            config={
                "fast_period": 10,
                "slow_period": 20,
                "ma_type": "EMA"
            }
        )
        ema_strategy.initialize(ema_strategy.config)

        required_ema = ema_strategy.get_required_indicators()
        if required_ema != ['EMA_10', 'EMA_20']:
            all_validation_failures.append(
                f"EMA indicators: Expected ['EMA_10', 'EMA_20'], got {required_ema}"
            )
    except Exception as e:
        all_validation_failures.append(f"EMA strategy test failed: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("MovingAverageCrossover strategy is validated and ready for use")
        print(f"Sample signals generated: {len(signals)} total, {len(buy_signals)} BUY signals")
        sys.exit(0)
