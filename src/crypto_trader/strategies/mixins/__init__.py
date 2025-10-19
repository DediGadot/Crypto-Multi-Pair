"""
Strategy Mixins

Reusable behavior for trading strategies through composition.
Eliminates code duplication by providing common functionality as mixins.

**Purpose**: Extract common patterns (hold signal generation, validation,
indicator calculation) into reusable components that strategies can compose.

**Third-party packages**: None (pure Python patterns)

**Sample Usage**:
```python
from crypto_trader.strategies.base import BaseStrategy
from crypto_trader.strategies.mixins import HoldSignalMixin, ValidationMixin

class MyStrategy(ValidationMixin, HoldSignalMixin, BaseStrategy):
    \"\"\"Strategy with reusable validation and hold signal behavior.\"\"\"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Use mixin methods
        self.validate_required_columns(data, ['close', 'volume'])
        signals = self._create_signal_dataframe(data)

        # Generate BUY/SELL signals
        signals.loc[buy_condition, 'signal'] = 'BUY'
        signals.loc[sell_condition, 'signal'] = 'SELL'

        # Apply hold signals automatically
        return self.apply_hold_signals(data, signals)
```

**Expected Output**:
- Reduced code duplication by 40-60%
- Consistent behavior across strategies
- Easier to maintain and test
"""

from crypto_trader.strategies.mixins.hold_signal_mixin import HoldSignalMixin
from crypto_trader.strategies.mixins.validation_mixin import ValidationMixin
from crypto_trader.strategies.mixins.indicator_mixin import IndicatorMixin

__all__ = [
    "HoldSignalMixin",
    "ValidationMixin",
    "IndicatorMixin",
]
