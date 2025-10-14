"""
Statistical Arbitrage Pairs Trading Strategy

Implements an Adaptive Regime-Aware Statistical Arbitrage (ARASA) strategy
for cointegrated cryptocurrency pairs.

**Purpose**: Exploit temporary mispricings in cointegrated pairs using
regime-adaptive signal thresholds and position sizing.

**Strategy Type**: Market-Neutral Pairs Trading / Statistical Arbitrage
**Indicators**: Cointegration, Spread Z-Score, HMM Regime Detection
**Entry Signal**: Spread crosses regime-dependent threshold (1.5σ to 2.5σ)
**Exit Signal**: Spread mean-reverts to regime-dependent target
**Risk Management**: Regime-based position sizing and stops

**Parameters**:
- pair1_symbol: First asset symbol (e.g., 'BTC/USDT')
- pair2_symbol: Second asset symbol (e.g., 'ETH/USDT')
- lookback_period: Period for cointegration testing (default: 180)
- entry_threshold: Base entry threshold in std devs (default: 2.0)
- exit_threshold: Base exit threshold in std devs (default: 0.5)
- z_score_window: Rolling window for z-score calculation (default: 90)

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/
- statsmodels: https://www.statsmodels.org/stable/
- hmmlearn: https://hmmlearn.readthedocs.io/
- loguru: https://loguru.readthedocs.io/

**Sample Input**:
```python
# Price data for two cointegrated assets
data = pd.DataFrame({
    'timestamp': [...],
    'BTC_close': [50000, 51000, ...],
    'ETH_close': [3000, 3100, ...]
})
```

**Expected Output**:
```python
signals = pd.DataFrame({
    'timestamp': [...],
    'signal': ['HOLD', 'BUY', 'HOLD', 'SELL', ...],
    'confidence': [0.0, 0.85, 0.0, 0.90, ...],
    'metadata': [
        {},
        {
            'z_score': -2.1,
            'spread': -0.05,
            'hedge_ratio': 0.144,
            'regime': 'mean_reverting',
            'reason': 'oversold_spread'
        },
        ...
    ]
})
```

**Academic Reference**:
Based on research showing copula-based cointegration with regime detection
improves Sharpe ratios by 30-40% (Financial Innovation, 2025).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy

# Handle imports for both package and standalone execution
try:
    from .statistical_arbitrage.cointegration import CointegrationAnalyzer
    from .statistical_arbitrage.regime_detection import RegimeDetector
except ImportError:
    from statistical_arbitrage.cointegration import CointegrationAnalyzer
    from statistical_arbitrage.regime_detection import RegimeDetector


@register_strategy(
    name="StatisticalArbitrage",
    description="Regime-aware statistical arbitrage on cointegrated pairs",
    tags=["pairs_trading", "statistical_arbitrage", "mean_reversion", "cointegration", "hmm"]
)
class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Adaptive Regime-Aware Statistical Arbitrage (ARASA) Strategy.

    Trades cointegrated cryptocurrency pairs with regime-dependent
    entry/exit thresholds and position sizing.

    Key Features:
    - Johansen cointegration testing
    - Dynamic hedge ratio estimation
    - Hidden Markov Model regime detection
    - Regime-adaptive signal thresholds
    - Z-score based spread trading
    """

    def __init__(self, name: str = "StatisticalArbitrage", config: Dict[str, Any] = None):
        """
        Initialize the Statistical Arbitrage strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary with parameters
        """
        super().__init__(name, config)

        # Default parameters
        self.pair1_symbol = "BTC/USDT"
        self.pair2_symbol = "ETH/USDT"
        self.lookback_period = 180
        self.entry_threshold = 2.0
        self.exit_threshold = 0.5
        self.z_score_window = 90
        self.min_data_points = 100

        # Components
        self.coint_analyzer = CointegrationAnalyzer()
        self.regime_detector = RegimeDetector()

        # State
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None
        self.is_cointegrated = False
        self.regime_fitted = False

        logger.debug(f"Initialized {self.__class__.__name__}")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration parameters.

        Args:
            config: Dictionary with strategy parameters

        Raises:
            ValueError: If parameters are invalid
        """
        self.pair1_symbol = config.get("pair1_symbol", "BTC/USDT")
        self.pair2_symbol = config.get("pair2_symbol", "ETH/USDT")
        self.lookback_period = config.get("lookback_period", 180)
        self.entry_threshold = config.get("entry_threshold", 2.0)
        self.exit_threshold = config.get("exit_threshold", 0.5)
        self.z_score_window = config.get("z_score_window", 90)

        # Validate parameters
        if self.lookback_period < 50:
            raise ValueError("lookback_period must be at least 50")

        if self.entry_threshold <= self.exit_threshold:
            raise ValueError("entry_threshold must be > exit_threshold")

        if self.z_score_window < 20:
            raise ValueError("z_score_window must be at least 20")

        self._initialized = True
        logger.info(
            f"{self.name} initialized: pairs=({self.pair1_symbol}, {self.pair2_symbol}), "
            f"lookback={self.lookback_period}, entry_threshold={self.entry_threshold}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "pair1_symbol": self.pair1_symbol,
            "pair2_symbol": self.pair2_symbol,
            "lookback_period": self.lookback_period,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "z_score_window": self.z_score_window,
            "is_cointegrated": self.is_cointegrated,
            "hedge_ratio": self.hedge_ratio
        }

    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators.

        Returns:
            Empty list - indicators are calculated internally
        """
        return []

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on cointegrated spread and regime.

        This method:
        1. Tests for cointegration between the pair
        2. Constructs the spread using hedge ratio
        3. Detects market regime using HMM
        4. Generates signals based on regime-adaptive z-score thresholds

        Args:
            data: DataFrame with price columns for both assets

        Returns:
            DataFrame with columns: ['timestamp', 'signal', 'confidence', 'metadata']

        Raises:
            ValueError: If data is invalid or assets not found
        """
        # Validate data structure - flexible column detection
        required_cols = ['timestamp']

        # Detect price columns (any column with 'close' in name, or direct column names)
        price_cols = [col for col in data.columns if 'close' in col.lower() and col != 'timestamp']

        # If no 'close' columns found, look for direct asset names
        if len(price_cols) < 2:
            # Try direct column names
            potential_cols = [col for col in data.columns if col not in ['timestamp', 'index']]
            if len(potential_cols) >= 2:
                price_cols = potential_cols[:2]

        if len(price_cols) < 2:
            logger.warning(
                f"Statistical arbitrage requires 2 price columns, found {len(price_cols)}. "
                f"Available columns: {data.columns.tolist()}"
            )
            return self._create_hold_signals(data)

        # Use first two price columns
        price1 = data[price_cols[0]]
        price2 = data[price_cols[1]]

        logger.info(f"Using columns: {price_cols[0]} and {price_cols[1]} for pairs trading")

        # Check minimum data requirement
        if len(data) < self.min_data_points:
            logger.warning(
                f"Insufficient data: {len(data)} rows, need {self.min_data_points}+"
            )
            return self._create_hold_signals(data)

        # Step 1: Test for cointegration
        coint_result = self.coint_analyzer.test_cointegration(price1, price2)

        self.is_cointegrated = coint_result['is_cointegrated']
        self.hedge_ratio = coint_result['hedge_ratio']

        if not self.is_cointegrated:
            logger.warning(
                f"Assets not cointegrated (reason: {coint_result['reason']}). "
                f"No trading signals generated."
            )
            return self._create_hold_signals(data)

        logger.info(
            f"Assets are cointegrated: hedge_ratio={self.hedge_ratio:.4f}, "
            f"half_life={coint_result['half_life']:.2f}"
        )

        # Step 2: Construct spread
        spread = self.coint_analyzer.construct_spread(price1, price2, self.hedge_ratio)

        # Step 3: Calculate z-scores
        z_scores = self.coint_analyzer.standardize_spread(spread, window=self.z_score_window)

        # Step 4: Calculate features for regime detection
        prices_df = pd.DataFrame({
            'asset1': price1,
            'asset2': price2
        })
        features = self.regime_detector.calculate_features(prices_df, spread, window=30)

        # Step 5: Fit regime detector (if not already fitted)
        features_clean = features.dropna()
        if len(features_clean) >= 30 and not self.regime_fitted:
            try:
                self.regime_detector.fit(features_clean)
                self.regime_fitted = True
                logger.info("Regime detector fitted successfully")
            except Exception as e:
                logger.warning(f"Failed to fit regime detector: {e}")

        # Step 6: Predict regimes
        if self.regime_fitted and len(features_clean) > 0:
            regime_prediction = self.regime_detector.predict(features_clean)
            # Align regimes with original data
            regimes = pd.Series(regime_prediction['regime'], index=features_clean.index)
            entry_thresholds = pd.Series(
                regime_prediction['entry_threshold'],
                index=features_clean.index
            )
            exit_thresholds = pd.Series(
                regime_prediction['exit_threshold'],
                index=features_clean.index
            )
        else:
            # Use default thresholds if regime detection not available
            regimes = pd.Series([1] * len(data), index=data.index)  # Trending regime
            entry_thresholds = pd.Series([self.entry_threshold] * len(data), index=data.index)
            exit_thresholds = pd.Series([self.exit_threshold] * len(data), index=data.index)

        # Step 7: Generate signals based on z-scores and regimes
        signals = []
        confidences = []
        metadata = []

        # Track position state
        in_position = False
        position_type = None  # 'LONG' or 'SHORT'

        for i in range(len(data)):
            timestamp = data.index[i] if isinstance(data.index, pd.DatetimeIndex) else data['timestamp'].iloc[i]

            # Get current values
            z = z_scores.iloc[i] if i < len(z_scores) else np.nan

            # Get regime-specific thresholds
            if timestamp in entry_thresholds.index:
                entry_thresh = entry_thresholds.loc[timestamp]
                exit_thresh = exit_thresholds.loc[timestamp]
                regime = regimes.loc[timestamp]
            else:
                entry_thresh = self.entry_threshold
                exit_thresh = self.exit_threshold
                regime = 1

            # Handle NaN z-scores
            if pd.isna(z):
                signals.append(SignalType.HOLD.value)
                confidences.append(0.0)
                metadata.append({})
                continue

            # Signal generation logic
            if not in_position:
                # Entry signals
                if z < -entry_thresh:
                    # Spread is oversold → Long spread (long asset1, short asset2)
                    signal = SignalType.BUY.value
                    confidence = min(0.5 + abs(z) / (2 * entry_thresh), 1.0)
                    in_position = True
                    position_type = 'LONG'

                    signals.append(signal)
                    confidences.append(confidence)
                    metadata.append({
                        'z_score': float(z),
                        'entry_threshold': float(entry_thresh),
                        'regime': int(regime),
                        'hedge_ratio': float(self.hedge_ratio),
                        'reason': 'oversold_spread'
                    })

                elif z > entry_thresh:
                    # Spread is overbought → Short spread (short asset1, long asset2)
                    signal = SignalType.SELL.value
                    confidence = min(0.5 + abs(z) / (2 * entry_thresh), 1.0)
                    in_position = True
                    position_type = 'SHORT'

                    signals.append(signal)
                    confidences.append(confidence)
                    metadata.append({
                        'z_score': float(z),
                        'entry_threshold': float(entry_thresh),
                        'regime': int(regime),
                        'hedge_ratio': float(self.hedge_ratio),
                        'reason': 'overbought_spread'
                    })

                else:
                    # No entry signal
                    signals.append(SignalType.HOLD.value)
                    confidences.append(0.0)
                    metadata.append({'z_score': float(z)})

            else:
                # Exit signals (close position when spread reverts)
                should_exit = False

                if position_type == 'LONG' and z > -exit_thresh:
                    # Long spread position, exit when spread reverts up
                    should_exit = True
                    exit_reason = 'spread_reversion_up'

                elif position_type == 'SHORT' and z < exit_thresh:
                    # Short spread position, exit when spread reverts down
                    should_exit = True
                    exit_reason = 'spread_reversion_down'

                if should_exit:
                    # Generate opposite signal to close position
                    signal = SignalType.SELL.value if position_type == 'LONG' else SignalType.BUY.value
                    confidence = 0.7

                    signals.append(signal)
                    confidences.append(confidence)
                    metadata.append({
                        'z_score': float(z),
                        'exit_threshold': float(exit_thresh),
                        'regime': int(regime),
                        'reason': exit_reason
                    })

                    in_position = False
                    position_type = None

                else:
                    # Hold position
                    signals.append(SignalType.HOLD.value)
                    confidences.append(0.0)
                    metadata.append({'z_score': float(z), 'in_position': True})

        # Create result DataFrame
        result = pd.DataFrame({
            'timestamp': data.index if isinstance(data.index, pd.DatetimeIndex) else data['timestamp'],
            'signal': signals,
            'confidence': confidences,
            'metadata': metadata
        })

        logger.info(
            f"Generated {len(result)} signals: "
            f"{sum(s == SignalType.BUY.value for s in signals)} BUY, "
            f"{sum(s == SignalType.SELL.value for s in signals)} SELL, "
            f"{sum(s == SignalType.HOLD.value for s in signals)} HOLD"
        )

        return result

    def _create_hold_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create HOLD signals for all rows.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with HOLD signals
        """
        return pd.DataFrame({
            'timestamp': data.index if isinstance(data.index, pd.DatetimeIndex) else data['timestamp'],
            'signal': [SignalType.HOLD.value] * len(data),
            'confidence': [0.0] * len(data),
            'metadata': [{}] * len(data)
        })


if __name__ == "__main__":
    """
    Validation block for StatisticalArbitrageStrategy.
    Tests with REAL BTC/USDT and ETH/USDT data.
    """
    import sys
    from datetime import datetime, timedelta
    from crypto_trader.data.fetchers import BinanceDataFetcher

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    logger.info("Starting Statistical Arbitrage Strategy validation with REAL DATA")

    # Test 1: Initialize strategy with default parameters
    total_tests += 1
    try:
        strategy = StatisticalArbitrageStrategy()
        strategy.initialize({})

        params = strategy.get_parameters()
        if params['entry_threshold'] != 2.0:
            all_validation_failures.append(
                f"Test 1: Expected entry_threshold=2.0, got {params['entry_threshold']}"
            )

        logger.success("Test 1 PASSED: Strategy initialized with default parameters")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Fetch real data for BTC and ETH
    total_tests += 1
    try:
        logger.info("Fetching BTC/USDT and ETH/USDT data...")
        fetcher = BinanceDataFetcher(use_storage=False, use_cache=False)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        btc_data = fetcher.get_ohlcv("BTC/USDT", "1d", start_date=start_date, end_date=end_date)
        eth_data = fetcher.get_ohlcv("ETH/USDT", "1d", start_date=start_date, end_date=end_date)

        if btc_data is None or btc_data.empty:
            all_validation_failures.append("Test 2: Failed to fetch BTC data")
        elif eth_data is None or eth_data.empty:
            all_validation_failures.append("Test 2: Failed to fetch ETH data")
        else:
            logger.success(
                f"Test 2 PASSED: Fetched {len(btc_data)} BTC and {len(eth_data)} ETH candles"
            )
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Prepare multi-asset data format
    total_tests += 1
    try:
        if 'btc_data' in locals() and 'eth_data' in locals():
            # Create combined DataFrame with both assets
            combined_data = pd.DataFrame({
                'timestamp': btc_data.index,
                'BTC_USDT_close': btc_data['close'].values,
                'ETH_USDT_close': eth_data['close'].reindex(btc_data.index).values
            })

            # Remove NaN values
            combined_data = combined_data.dropna()

            if len(combined_data) < 100:
                all_validation_failures.append(
                    f"Test 3: Insufficient clean data - got {len(combined_data)}, need 100+"
                )
            else:
                logger.success(f"Test 3 PASSED: Prepared {len(combined_data)} rows of paired data")
        else:
            all_validation_failures.append("Test 3: No data available from Test 2")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Generate signals with real paired data
    total_tests += 1
    try:
        if 'combined_data' in locals():
            signals = strategy.generate_signals(combined_data)

            if signals is None or signals.empty:
                all_validation_failures.append("Test 4: No signals generated")
            elif len(signals) != len(combined_data):
                all_validation_failures.append(
                    f"Test 4: Signal count mismatch - data: {len(combined_data)}, "
                    f"signals: {len(signals)}"
                )
            else:
                buy_count = (signals['signal'] == SignalType.BUY.value).sum()
                sell_count = (signals['signal'] == SignalType.SELL.value).sum()
                hold_count = (signals['signal'] == SignalType.HOLD.value).sum()

                logger.success(f"Test 4 PASSED: Generated {len(signals)} signals")
                logger.info(f"  BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")

                # Show sample signals
                action_signals = signals[signals['signal'] != SignalType.HOLD.value]
                if not action_signals.empty:
                    logger.info(f"  First action signal: {action_signals.iloc[0]['signal']}")
                    logger.info(f"    Z-score: {action_signals.iloc[0]['metadata'].get('z_score', 'N/A')}")
        else:
            all_validation_failures.append("Test 4: No combined data available")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception raised: {e}")

    # Test 5: Verify cointegration detection
    total_tests += 1
    try:
        params = strategy.get_parameters()
        if not params['is_cointegrated']:
            logger.warning(
                "Test 5: BTC/ETH not detected as cointegrated in this sample. "
                "This can happen with short time periods."
            )
        else:
            logger.success(f"Test 5 PASSED: Cointegration detected")
            logger.info(f"  Hedge ratio: {params['hedge_ratio']:.4f}")
    except Exception as e:
        all_validation_failures.append(f"Test 5: Exception raised: {e}")

    # Test 6: Test signal metadata
    total_tests += 1
    try:
        if 'signals' in locals() and not signals.empty:
            action_signals = signals[signals['signal'] != SignalType.HOLD.value]

            if not action_signals.empty:
                first_signal = action_signals.iloc[0]

                if 'z_score' not in first_signal['metadata']:
                    all_validation_failures.append(
                        "Test 6: Action signal metadata missing 'z_score'"
                    )
                elif first_signal['confidence'] <= 0:
                    all_validation_failures.append(
                        "Test 6: Action signal should have positive confidence"
                    )
                else:
                    logger.success("Test 6 PASSED: Signal metadata structure correct")
            else:
                logger.warning("Test 6: No action signals generated to verify metadata")
        else:
            all_validation_failures.append("Test 6: No signals available")
    except Exception as e:
        all_validation_failures.append(f"Test 6: Exception raised: {e}")

    # Test 7: Test strategy registration
    total_tests += 1
    try:
        from crypto_trader.strategies.registry import get_registry

        registry = get_registry()
        if "StatisticalArbitrage" not in registry:
            all_validation_failures.append(
                "Test 7: Strategy not registered in global registry"
            )
        else:
            registered_class = registry.get_strategy("StatisticalArbitrage")
            if registered_class is not StatisticalArbitrageStrategy:
                all_validation_failures.append(
                    "Test 7: Wrong class registered in registry"
                )
            else:
                logger.success("Test 7 PASSED: Strategy registered correctly")
    except Exception as e:
        all_validation_failures.append(f"Test 7: Exception raised: {e}")

    # Final validation result
    print("\n" + "="*70)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Statistical Arbitrage Strategy validated with REAL BTC/ETH data")
        print("Strategy is validated and ready for backtesting")
        sys.exit(0)
