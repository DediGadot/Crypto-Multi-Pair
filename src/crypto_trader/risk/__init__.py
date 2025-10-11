"""
Risk management module for crypto trading.

This module provides comprehensive risk management including position sizing,
risk limit enforcement, stop loss/take profit calculations, and portfolio
risk metrics.

**Main Components**:
- RiskManager: Main risk management coordinator
- Position sizing algorithms (Fixed, Kelly, Volatility, Risk Parity)
- Risk limit enforcement
- Daily loss tracking
- Drawdown monitoring

**Usage Example**:
```python
from crypto_trader.core.config import RiskConfig
from crypto_trader.risk import RiskManager
from crypto_trader.backtesting.portfolio import PortfolioState

# Initialize risk manager
config = RiskConfig(
    max_position_risk=0.02,
    max_portfolio_risk=0.10,
    position_sizing_method="fixed_percent"
)
risk_manager = RiskManager(config)

# Calculate position size
quantity = risk_manager.calculate_position_size(
    signal=1,
    portfolio=portfolio_state,
    price=50000.0,
    stop_loss_price=48000.0
)

# Check if trade is allowed
allowed, reason = risk_manager.should_allow_trade(
    signal=1,
    portfolio=portfolio_state,
    price=50000.0,
    timestamp=datetime.now()
)

# Calculate stops
stop_loss = risk_manager.calculate_stop_loss(50000.0, side="long")
take_profit = risk_manager.calculate_take_profit(50000.0, side="long")

# Get portfolio risk metrics
metrics = risk_manager.get_portfolio_risk(portfolio_state)
```

**Documentation**:
- Risk Management Concepts: https://www.investopedia.com/terms/r/riskmanagement.asp
- Position Sizing: https://www.investopedia.com/terms/p/positionsizing.asp
- Kelly Criterion: https://en.wikipedia.org/wiki/Kelly_criterion
"""

from crypto_trader.risk.limits import DailyRiskTracker, RiskLimitChecker
from crypto_trader.risk.manager import RiskManager
from crypto_trader.risk.sizing import (
    FixedFractionSizer,
    KellyCriterionSizer,
    PositionSizer,
    RiskParitySizer,
    VolatilityBasedSizer,
    create_position_sizer,
)

__all__ = [
    # Main risk manager
    "RiskManager",
    # Position sizers
    "PositionSizer",
    "FixedFractionSizer",
    "KellyCriterionSizer",
    "VolatilityBasedSizer",
    "RiskParitySizer",
    "create_position_sizer",
    # Risk limits
    "RiskLimitChecker",
    "DailyRiskTracker",
]
