"""
Configuration management for the crypto trading system using Pydantic.

This module provides type-safe configuration management with automatic
validation, environment variable substitution, and YAML file loading.
Uses Pydantic V2 for robust data validation and settings management.

Documentation:
- Pydantic V2: https://docs.pydantic.dev/latest/
- Pydantic Settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- PyYAML: https://pyyaml.org/wiki/PyYAMLDocumentation

Sample Input:
    config = DataConfig(
        exchange="binance",
        symbols=["BTCUSDT", "ETHUSDT"],
        default_timeframe="1h"
    )

Expected Output:
    Validated configuration objects with proper type checking and defaults
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from crypto_trader.core.exceptions import ConfigurationError
from crypto_trader.core.types import Timeframe


class DataConfig(BaseSettings):
    """
    Configuration for data fetching and storage.

    Attributes:
        exchange: Exchange name (binance, coinbase, kraken, etc.)
        symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        default_timeframe: Default candle timeframe
        cache_dir: Directory for caching historical data
        max_cache_age_hours: Maximum age of cached data in hours
        rate_limit_per_minute: API rate limit (requests per minute)
    """
    exchange: str = Field(default="binance", description="Exchange name")
    symbols: list[str] = Field(default=["BTCUSDT"], description="Trading pairs")
    default_timeframe: str = Field(default="1h", description="Default timeframe")
    cache_dir: Path = Field(default=Path("data/cache"), description="Cache directory")
    max_cache_age_hours: int = Field(default=24, ge=1, description="Cache age in hours")
    rate_limit_per_minute: int = Field(default=1200, ge=1, description="API rate limit")

    model_config = SettingsConfigDict(
        env_prefix="CRYPTO_DATA_",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("default_timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate that timeframe is supported."""
        try:
            Timeframe(v)  # Will raise ValueError if invalid
        except ValueError:
            valid_values = [tf.value for tf in Timeframe]
            raise ValueError(
                f"Invalid timeframe '{v}'. Must be one of: {valid_values}"
            )
        return v

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        """Validate symbol list is not empty and properly formatted."""
        if not v:
            raise ValueError("Symbols list cannot be empty")

        # Ensure symbols are uppercase and non-empty
        validated = []
        for symbol in v:
            if not symbol or not symbol.strip():
                raise ValueError("Symbol cannot be empty")
            validated.append(symbol.upper().strip())

        return validated


class StrategyConfig(BaseSettings):
    """
    Configuration for trading strategy parameters.

    Attributes:
        name: Strategy name
        indicator_params: Parameters for technical indicators
        signal_threshold: Minimum signal strength threshold (0-1)
        use_trailing_stop: Enable trailing stop loss
        trailing_stop_percent: Trailing stop percentage
        take_profit_percent: Take profit percentage
        lookback_periods: Historical periods to analyze
    """
    name: str = Field(default="default", description="Strategy name")
    indicator_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Technical indicator parameters"
    )
    signal_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Signal strength threshold"
    )
    use_trailing_stop: bool = Field(default=False, description="Enable trailing stop")
    trailing_stop_percent: float = Field(
        default=0.02,
        gt=0.0,
        le=0.5,
        description="Trailing stop percentage"
    )
    take_profit_percent: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Take profit percentage"
    )
    lookback_periods: int = Field(
        default=100,
        ge=10,
        description="Historical periods for analysis"
    )

    model_config = SettingsConfigDict(
        env_prefix="CRYPTO_STRATEGY_",
        case_sensitive=False,
        extra="ignore"
    )

    @model_validator(mode="after")
    def validate_stop_profit_relationship(self) -> "StrategyConfig":
        """Validate that take profit is greater than trailing stop."""
        if self.use_trailing_stop and self.take_profit_percent <= self.trailing_stop_percent:
            raise ValueError(
                f"Take profit ({self.take_profit_percent}) must be greater than "
                f"trailing stop ({self.trailing_stop_percent})"
            )
        return self


class BacktestConfig(BaseSettings):
    """
    Configuration for backtesting parameters.

    Attributes:
        initial_capital: Starting portfolio value
        trading_fee_percent: Trading fee as percentage (0.001 = 0.1%)
        slippage_percent: Slippage as percentage
        max_position_size: Maximum position size as % of capital
        enable_short_selling: Allow short positions
        compound_returns: Reinvest profits
        commission_type: Commission calculation type (fixed or percentage)
    """
    initial_capital: float = Field(
        default=10000.0,
        gt=0.0,
        description="Initial capital in USD"
    )
    trading_fee_percent: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Trading fee percentage"
    )
    slippage_percent: float = Field(
        default=0.0005,
        ge=0.0,
        le=0.1,
        description="Slippage percentage"
    )
    max_position_size: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Max position as % of capital"
    )
    enable_short_selling: bool = Field(
        default=False,
        description="Enable short positions"
    )
    compound_returns: bool = Field(
        default=True,
        description="Reinvest profits"
    )
    commission_type: str = Field(
        default="percentage",
        description="Commission type"
    )

    model_config = SettingsConfigDict(
        env_prefix="CRYPTO_BACKTEST_",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("commission_type")
    @classmethod
    def validate_commission_type(cls, v: str) -> str:
        """Validate commission type."""
        valid_types = ["fixed", "percentage"]
        if v.lower() not in valid_types:
            raise ValueError(
                f"Invalid commission_type '{v}'. Must be one of: {valid_types}"
            )
        return v.lower()


class RiskConfig(BaseSettings):
    """
    Configuration for risk management parameters.

    Attributes:
        max_position_risk: Max % of capital to risk per trade
        max_portfolio_risk: Max total portfolio risk exposure
        stop_loss_percent: Default stop loss percentage
        max_daily_loss_percent: Max daily loss before stopping
        max_open_positions: Maximum concurrent positions
        position_sizing_method: Method for calculating position size
        risk_reward_ratio: Minimum risk/reward ratio
    """
    max_position_risk: float = Field(
        default=0.02,
        gt=0.0,
        le=0.2,
        description="Max risk per position"
    )
    max_portfolio_risk: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Max total portfolio risk"
    )
    stop_loss_percent: float = Field(
        default=0.02,
        gt=0.0,
        le=0.5,
        description="Default stop loss"
    )
    max_daily_loss_percent: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
        description="Max daily loss"
    )
    max_open_positions: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Max concurrent positions"
    )
    position_sizing_method: str = Field(
        default="fixed_percent",
        description="Position sizing method"
    )
    risk_reward_ratio: float = Field(
        default=2.0,
        ge=1.0,
        description="Min risk/reward ratio"
    )

    model_config = SettingsConfigDict(
        env_prefix="CRYPTO_RISK_",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("position_sizing_method")
    @classmethod
    def validate_sizing_method(cls, v: str) -> str:
        """Validate position sizing method."""
        valid_methods = ["fixed_percent", "fixed_amount", "kelly", "volatility_based"]
        if v.lower() not in valid_methods:
            raise ValueError(
                f"Invalid position_sizing_method '{v}'. Must be one of: {valid_methods}"
            )
        return v.lower()

    @model_validator(mode="after")
    def validate_risk_hierarchy(self) -> "RiskConfig":
        """Validate that portfolio risk >= position risk."""
        if self.max_portfolio_risk < self.max_position_risk:
            raise ValueError(
                f"Max portfolio risk ({self.max_portfolio_risk}) must be >= "
                f"max position risk ({self.max_position_risk})"
            )
        return self


class TradingConfig(BaseSettings):
    """
    Main trading system configuration that combines all sub-configs.

    This is the root configuration object that aggregates all other
    configuration components.
    """
    data: DataConfig = Field(default_factory=DataConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    model_config = SettingsConfigDict(
        env_prefix="CRYPTO_",
        case_sensitive=False,
        extra="ignore"
    )

    @classmethod
    def from_yaml(cls, file_path: Path | str) -> "TradingConfig":
        """
        Load configuration from a YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            TradingConfig instance with loaded values

        Raises:
            ConfigurationError: If file cannot be read or parsed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                details={"path": str(file_path)}
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data is None:
                data = {}

            # Create config with data from YAML
            # Pydantic will automatically validate and merge with env vars
            return cls(**data)

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML configuration",
                details={"file": str(file_path)},
                original_error=e
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration",
                details={"file": str(file_path)},
                original_error=e
            )

    def to_yaml(self, file_path: Path | str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            file_path: Path where to save the configuration

        Raises:
            ConfigurationError: If file cannot be written
        """
        file_path = Path(file_path)

        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict for YAML serialization
            config_dict = {
                "data": self.data.model_dump(mode="json"),
                "strategy": self.strategy.model_dump(mode="json"),
                "backtest": self.backtest.model_dump(mode="json"),
                "risk": self.risk.model_dump(mode="json"),
            }

            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration",
                details={"file": str(file_path)},
                original_error=e
            )


if __name__ == "__main__":
    """
    Validation function to test configuration management with real data.
    Tests validation, defaults, and YAML loading/saving.
    """
    import sys
    import tempfile

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating config.py with real data...\n")

    # Test 1: DataConfig with defaults
    total_tests += 1
    print("Test 1: DataConfig with defaults")
    try:
        data_config = DataConfig()
        if data_config.exchange != "binance":
            all_validation_failures.append(f"Default exchange: Expected 'binance', got '{data_config.exchange}'")
        if data_config.default_timeframe != "1h":
            all_validation_failures.append(f"Default timeframe: Expected '1h', got '{data_config.default_timeframe}'")
        if data_config.symbols != ["BTCUSDT"]:
            all_validation_failures.append(f"Default symbols: Expected ['BTCUSDT'], got {data_config.symbols}")

        print(f"  ✓ Exchange: {data_config.exchange}")
        print(f"  ✓ Symbols: {data_config.symbols}")
        print(f"  ✓ Timeframe: {data_config.default_timeframe}")
    except Exception as e:
        all_validation_failures.append(f"DataConfig defaults test exception: {e}")

    # Test 2: DataConfig with custom values
    total_tests += 1
    print("\nTest 2: DataConfig with custom values")
    try:
        custom_data = DataConfig(
            exchange="coinbase",
            symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            default_timeframe="4h",
            rate_limit_per_minute=600
        )
        if custom_data.exchange != "coinbase":
            all_validation_failures.append(f"Custom exchange: Expected 'coinbase', got '{custom_data.exchange}'")
        if len(custom_data.symbols) != 3:
            all_validation_failures.append(f"Symbol count: Expected 3, got {len(custom_data.symbols)}")
        if custom_data.rate_limit_per_minute != 600:
            all_validation_failures.append(f"Rate limit: Expected 600, got {custom_data.rate_limit_per_minute}")

        print(f"  ✓ Exchange: {custom_data.exchange}")
        print(f"  ✓ Symbols: {custom_data.symbols}")
        print(f"  ✓ Rate limit: {custom_data.rate_limit_per_minute}")
    except Exception as e:
        all_validation_failures.append(f"DataConfig custom test exception: {e}")

    # Test 3: DataConfig validation - invalid timeframe
    total_tests += 1
    print("\nTest 3: DataConfig validation - invalid timeframe")
    try:
        invalid_timeframe_raised = False
        try:
            DataConfig(default_timeframe="2h")  # Invalid timeframe
        except ValueError as ve:
            invalid_timeframe_raised = True
            if "Invalid timeframe" not in str(ve):
                all_validation_failures.append(f"Invalid timeframe error message incorrect: {ve}")

        if not invalid_timeframe_raised:
            all_validation_failures.append("Invalid timeframe should raise ValueError")

        print(f"  ✓ Invalid timeframe properly rejected")
    except Exception as e:
        all_validation_failures.append(f"Invalid timeframe test exception: {e}")

    # Test 4: DataConfig validation - empty symbols
    total_tests += 1
    print("\nTest 4: DataConfig validation - empty symbols")
    try:
        empty_symbols_raised = False
        try:
            DataConfig(symbols=[])
        except ValueError as ve:
            empty_symbols_raised = True
            if "cannot be empty" not in str(ve):
                all_validation_failures.append(f"Empty symbols error message incorrect: {ve}")

        if not empty_symbols_raised:
            all_validation_failures.append("Empty symbols should raise ValueError")

        print(f"  ✓ Empty symbols properly rejected")
    except Exception as e:
        all_validation_failures.append(f"Empty symbols test exception: {e}")

    # Test 5: StrategyConfig with defaults
    total_tests += 1
    print("\nTest 5: StrategyConfig with defaults")
    try:
        strategy_config = StrategyConfig()
        if strategy_config.signal_threshold != 0.5:
            all_validation_failures.append(f"Signal threshold: Expected 0.5, got {strategy_config.signal_threshold}")
        if strategy_config.use_trailing_stop != False:
            all_validation_failures.append(f"Trailing stop: Expected False, got {strategy_config.use_trailing_stop}")

        print(f"  ✓ Signal threshold: {strategy_config.signal_threshold}")
        print(f"  ✓ Trailing stop: {strategy_config.use_trailing_stop}")
        print(f"  ✓ Lookback periods: {strategy_config.lookback_periods}")
    except Exception as e:
        all_validation_failures.append(f"StrategyConfig test exception: {e}")

    # Test 6: StrategyConfig validation - profit < stop
    total_tests += 1
    print("\nTest 6: StrategyConfig validation - profit < stop")
    try:
        profit_validation_raised = False
        try:
            StrategyConfig(
                use_trailing_stop=True,
                trailing_stop_percent=0.05,
                take_profit_percent=0.03  # Less than stop
            )
        except ValueError as ve:
            profit_validation_raised = True
            if "must be greater than" not in str(ve):
                all_validation_failures.append(f"Profit validation error message incorrect: {ve}")

        if not profit_validation_raised:
            all_validation_failures.append("Invalid profit/stop relationship should raise ValueError")

        print(f"  ✓ Invalid profit/stop relationship properly rejected")
    except Exception as e:
        all_validation_failures.append(f"Profit validation test exception: {e}")

    # Test 7: BacktestConfig with custom values
    total_tests += 1
    print("\nTest 7: BacktestConfig with custom values")
    try:
        backtest_config = BacktestConfig(
            initial_capital=50000.0,
            trading_fee_percent=0.002,
            enable_short_selling=True,
            compound_returns=False
        )
        if backtest_config.initial_capital != 50000.0:
            all_validation_failures.append(f"Initial capital: Expected 50000.0, got {backtest_config.initial_capital}")
        if not backtest_config.enable_short_selling:
            all_validation_failures.append(f"Short selling: Expected True, got {backtest_config.enable_short_selling}")

        print(f"  ✓ Initial capital: ${backtest_config.initial_capital:,.2f}")
        print(f"  ✓ Trading fee: {backtest_config.trading_fee_percent:.3%}")
        print(f"  ✓ Short selling: {backtest_config.enable_short_selling}")
    except Exception as e:
        all_validation_failures.append(f"BacktestConfig test exception: {e}")

    # Test 8: RiskConfig validation - hierarchy
    total_tests += 1
    print("\nTest 8: RiskConfig validation - risk hierarchy")
    try:
        risk_hierarchy_raised = False
        try:
            RiskConfig(
                max_position_risk=0.1,
                max_portfolio_risk=0.05  # Less than position risk
            )
        except ValueError as ve:
            risk_hierarchy_raised = True
            if "must be >=" not in str(ve):
                all_validation_failures.append(f"Risk hierarchy error message incorrect: {ve}")

        if not risk_hierarchy_raised:
            all_validation_failures.append("Invalid risk hierarchy should raise ValueError")

        print(f"  ✓ Invalid risk hierarchy properly rejected")
    except Exception as e:
        all_validation_failures.append(f"Risk hierarchy test exception: {e}")

    # Test 9: TradingConfig aggregation
    total_tests += 1
    print("\nTest 9: TradingConfig aggregation")
    try:
        trading_config = TradingConfig(
            data=DataConfig(exchange="kraken"),
            strategy=StrategyConfig(name="RSI Strategy"),
            backtest=BacktestConfig(initial_capital=25000.0),
            risk=RiskConfig(max_open_positions=5)
        )

        if trading_config.data.exchange != "kraken":
            all_validation_failures.append(f"Config exchange: Expected 'kraken', got '{trading_config.data.exchange}'")
        if trading_config.strategy.name != "RSI Strategy":
            all_validation_failures.append(f"Strategy name: Expected 'RSI Strategy', got '{trading_config.strategy.name}'")
        if trading_config.backtest.initial_capital != 25000.0:
            all_validation_failures.append(f"Initial capital: Expected 25000.0, got {trading_config.backtest.initial_capital}")
        if trading_config.risk.max_open_positions != 5:
            all_validation_failures.append(f"Max positions: Expected 5, got {trading_config.risk.max_open_positions}")

        print(f"  ✓ Exchange: {trading_config.data.exchange}")
        print(f"  ✓ Strategy: {trading_config.strategy.name}")
        print(f"  ✓ Capital: ${trading_config.backtest.initial_capital:,.2f}")
        print(f"  ✓ Max positions: {trading_config.risk.max_open_positions}")
    except Exception as e:
        all_validation_failures.append(f"TradingConfig test exception: {e}")

    # Test 10: YAML save and load
    total_tests += 1
    print("\nTest 10: YAML save and load")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        # Create and save config
        original_config = TradingConfig(
            data=DataConfig(exchange="binance", symbols=["BTCUSDT", "ETHUSDT"]),
            strategy=StrategyConfig(name="Test Strategy", signal_threshold=0.7),
            backtest=BacktestConfig(initial_capital=15000.0),
            risk=RiskConfig(max_position_risk=0.03)
        )
        original_config.to_yaml(tmp_path)

        # Load config
        loaded_config = TradingConfig.from_yaml(tmp_path)

        # Validate loaded values
        if loaded_config.data.exchange != "binance":
            all_validation_failures.append(f"Loaded exchange: Expected 'binance', got '{loaded_config.data.exchange}'")
        if loaded_config.strategy.signal_threshold != 0.7:
            all_validation_failures.append(f"Loaded threshold: Expected 0.7, got {loaded_config.strategy.signal_threshold}")
        if loaded_config.backtest.initial_capital != 15000.0:
            all_validation_failures.append(f"Loaded capital: Expected 15000.0, got {loaded_config.backtest.initial_capital}")
        if loaded_config.risk.max_position_risk != 0.03:
            all_validation_failures.append(f"Loaded risk: Expected 0.03, got {loaded_config.risk.max_position_risk}")

        # Clean up
        tmp_path.unlink()

        print(f"  ✓ Config saved to YAML")
        print(f"  ✓ Config loaded from YAML")
        print(f"  ✓ All values preserved correctly")
    except Exception as e:
        all_validation_failures.append(f"YAML test exception: {e}")

    # Test 11: YAML load - file not found
    total_tests += 1
    print("\nTest 11: YAML load - file not found")
    try:
        file_not_found_raised = False
        try:
            TradingConfig.from_yaml("/nonexistent/path/config.yaml")
        except ConfigurationError as ce:
            file_not_found_raised = True
            if "not found" not in str(ce):
                all_validation_failures.append(f"File not found error message incorrect: {ce}")

        if not file_not_found_raised:
            all_validation_failures.append("Missing file should raise ConfigurationError")

        print(f"  ✓ Missing file properly raises ConfigurationError")
    except Exception as e:
        all_validation_failures.append(f"File not found test exception: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
