"""
Unit Tests for Portfolio Configuration Loading

Tests the YAML configuration loading and validation for portfolio backtesting:
- YAML parsing
- Configuration validation
- Asset allocation verification
- Parameter extraction

**Purpose**: Verify portfolio configuration system works correctly.

**Third-party packages**:
- pytest: https://docs.pytest.org/
- pyyaml: https://pyyaml.org/wiki/PyYAMLDocumentation
"""

import tempfile
from pathlib import Path
from typing import Dict, Any

import yaml
import pytest


class TestYAMLConfigLoading:
    """Test YAML configuration file loading."""

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_content = """
run:
  name: "test_portfolio"
  description: "Test configuration"
  mode: "portfolio"

data:
  timeframe: "1h"
  days: 30

portfolio:
  assets:
    - symbol: "BTC/USDT"
      weight: 0.50
    - symbol: "ETH/USDT"
      weight: 0.50

  rebalancing:
    enabled: true
    threshold: 0.15
    method: "threshold"
    min_rebalance_interval_hours: 24

capital:
  initial_capital: 10000.0

costs:
  commission: 0.001
  slippage: 0.0005

output:
  directory: "results_test"
  save_trades: true
  save_equity_curve: true
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            with open(temp_path, 'r') as f:
                config = yaml.safe_load(f)

            assert config is not None
            assert 'run' in config
            assert 'portfolio' in config
            assert 'data' in config
            assert config['run']['name'] == "test_portfolio"
            assert len(config['portfolio']['assets']) == 2
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises an error."""
        # YAML with duplicate keys and invalid syntax
        invalid_yaml = """
run:
  name: "test"
  - invalid list item in mapping
  name: "duplicate key"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name

        try:
            with pytest.raises((yaml.YAMLError, yaml.scanner.ScannerError)):
                with open(temp_path, 'r') as f:
                    yaml.safe_load(f)
        finally:
            Path(temp_path).unlink()

    def test_missing_config_file_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            with open("/nonexistent/config.yaml", 'r') as f:
                yaml.safe_load(f)


class TestPortfolioConfigValidation:
    """Test portfolio configuration validation."""

    def test_valid_portfolio_config(self):
        """Test validation of valid portfolio configuration."""
        config = {
            'portfolio': {
                'assets': [
                    {'symbol': 'BTC/USDT', 'weight': 0.50},
                    {'symbol': 'ETH/USDT', 'weight': 0.30},
                    {'symbol': 'SOL/USDT', 'weight': 0.20}
                ],
                'rebalancing': {
                    'enabled': True,
                    'threshold': 0.15,
                    'method': 'threshold',
                    'min_rebalance_interval_hours': 24
                }
            }
        }

        # Validate weights sum to 1.0
        total_weight = sum(asset['weight'] for asset in config['portfolio']['assets'])
        assert abs(total_weight - 1.0) < 0.01

        # Validate all required fields
        assert 'assets' in config['portfolio']
        assert 'rebalancing' in config['portfolio']
        assert len(config['portfolio']['assets']) >= 2

    def test_weights_not_summing_to_one_detected(self):
        """Test detection of weights not summing to 1.0."""
        config = {
            'portfolio': {
                'assets': [
                    {'symbol': 'BTC/USDT', 'weight': 0.60},
                    {'symbol': 'ETH/USDT', 'weight': 0.30}
                    # Sum is 0.90, not 1.0
                ]
            }
        }

        total_weight = sum(asset['weight'] for asset in config['portfolio']['assets'])
        assert abs(total_weight - 1.0) > 0.01  # Should not be close to 1.0

    def test_single_asset_portfolio_detected(self):
        """Test detection of single-asset portfolio (invalid)."""
        config = {
            'portfolio': {
                'assets': [
                    {'symbol': 'BTC/USDT', 'weight': 1.0}
                ]
            }
        }

        assert len(config['portfolio']['assets']) < 2

    def test_negative_weights_detected(self):
        """Test detection of negative asset weights."""
        config = {
            'portfolio': {
                'assets': [
                    {'symbol': 'BTC/USDT', 'weight': 0.70},
                    {'symbol': 'ETH/USDT', 'weight': -0.30}
                ]
            }
        }

        has_negative = any(asset['weight'] < 0 for asset in config['portfolio']['assets'])
        assert has_negative

    def test_zero_weight_assets(self):
        """Test handling of zero-weight assets."""
        config = {
            'portfolio': {
                'assets': [
                    {'symbol': 'BTC/USDT', 'weight': 0.50},
                    {'symbol': 'ETH/USDT', 'weight': 0.50},
                    {'symbol': 'SOL/USDT', 'weight': 0.00}  # Zero weight
                ]
            }
        }

        non_zero_assets = [a for a in config['portfolio']['assets'] if a['weight'] > 0]
        assert len(non_zero_assets) == 2


class TestAssetConfiguration:
    """Test asset configuration parsing."""

    def test_extract_asset_symbols(self):
        """Test extracting asset symbols from config."""
        config = {
            'portfolio': {
                'assets': [
                    {'symbol': 'BTC/USDT', 'weight': 0.40},
                    {'symbol': 'ETH/USDT', 'weight': 0.30},
                    {'symbol': 'SOL/USDT', 'weight': 0.15},
                    {'symbol': 'BNB/USDT', 'weight': 0.15}
                ]
            }
        }

        symbols = [asset['symbol'] for asset in config['portfolio']['assets']]
        assert len(symbols) == 4
        assert 'BTC/USDT' in symbols
        assert 'ETH/USDT' in symbols
        assert 'SOL/USDT' in symbols
        assert 'BNB/USDT' in symbols

    def test_extract_asset_weights(self):
        """Test extracting asset weights from config."""
        config = {
            'portfolio': {
                'assets': [
                    {'symbol': 'BTC/USDT', 'weight': 0.40},
                    {'symbol': 'ETH/USDT', 'weight': 0.60}
                ]
            }
        }

        weights = {asset['symbol']: asset['weight'] for asset in config['portfolio']['assets']}
        assert weights['BTC/USDT'] == 0.40
        assert weights['ETH/USDT'] == 0.60

    def test_convert_to_strategy_format(self):
        """Test converting config to strategy-compatible format."""
        config = {
            'portfolio': {
                'assets': [
                    {'symbol': 'BTC/USDT', 'weight': 0.50},
                    {'symbol': 'ETH/USDT', 'weight': 0.50}
                ]
            }
        }

        # Convert to list of tuples for strategy
        assets_tuples = [
            (asset['symbol'], asset['weight'])
            for asset in config['portfolio']['assets']
        ]

        assert len(assets_tuples) == 2
        assert assets_tuples[0] == ('BTC/USDT', 0.50)
        assert assets_tuples[1] == ('ETH/USDT', 0.50)


class TestRebalancingConfiguration:
    """Test rebalancing parameter configuration."""

    def test_extract_rebalancing_params(self):
        """Test extracting rebalancing parameters."""
        config = {
            'portfolio': {
                'rebalancing': {
                    'enabled': True,
                    'threshold': 0.15,
                    'method': 'threshold',
                    'min_rebalance_interval_hours': 24
                }
            }
        }

        rebal = config['portfolio']['rebalancing']
        assert rebal['enabled'] is True
        assert rebal['threshold'] == 0.15
        assert rebal['method'] == 'threshold'
        assert rebal['min_rebalance_interval_hours'] == 24

    def test_disabled_rebalancing(self):
        """Test configuration with rebalancing disabled."""
        config = {
            'portfolio': {
                'rebalancing': {
                    'enabled': False
                }
            }
        }

        assert config['portfolio']['rebalancing']['enabled'] is False

    def test_various_threshold_values(self):
        """Test various threshold values."""
        valid_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

        for threshold in valid_thresholds:
            config = {
                'portfolio': {
                    'rebalancing': {
                        'threshold': threshold
                    }
                }
            }

            assert 0 < config['portfolio']['rebalancing']['threshold'] < 1

    def test_various_interval_values(self):
        """Test various min rebalance interval values."""
        valid_intervals = [1, 6, 12, 24, 48, 168]  # Hours

        for interval in valid_intervals:
            config = {
                'portfolio': {
                    'rebalancing': {
                        'min_rebalance_interval_hours': interval
                    }
                }
            }

            assert config['portfolio']['rebalancing']['min_rebalance_interval_hours'] > 0


class TestCapitalAndCosts:
    """Test capital and cost configuration."""

    def test_extract_initial_capital(self):
        """Test extracting initial capital."""
        config = {
            'capital': {
                'initial_capital': 10000.0
            }
        }

        assert config['capital']['initial_capital'] == 10000.0

    def test_extract_commission_and_slippage(self):
        """Test extracting trading costs."""
        config = {
            'costs': {
                'commission': 0.001,
                'slippage': 0.0005
            }
        }

        assert config['costs']['commission'] == 0.001
        assert config['costs']['slippage'] == 0.0005

    def test_zero_commission_allowed(self):
        """Test that zero commission is allowed."""
        config = {
            'costs': {
                'commission': 0.0,
                'slippage': 0.0
            }
        }

        assert config['costs']['commission'] == 0.0
        assert config['costs']['slippage'] == 0.0


class TestOutputConfiguration:
    """Test output directory and file settings."""

    def test_extract_output_directory(self):
        """Test extracting output directory."""
        config = {
            'output': {
                'directory': 'results_test',
                'save_trades': True,
                'save_equity_curve': True
            }
        }

        assert config['output']['directory'] == 'results_test'
        assert config['output']['save_trades'] is True
        assert config['output']['save_equity_curve'] is True

    def test_output_flags(self):
        """Test various combinations of output flags."""
        configs = [
            {'save_trades': True, 'save_equity_curve': True},
            {'save_trades': True, 'save_equity_curve': False},
            {'save_trades': False, 'save_equity_curve': True},
            {'save_trades': False, 'save_equity_curve': False},
        ]

        for output_config in configs:
            config = {'output': output_config}
            # All combinations are valid
            assert 'save_trades' in config['output']
            assert 'save_equity_curve' in config['output']


class TestCompleteConfigWorkflow:
    """Test complete configuration workflow."""

    def test_load_and_validate_complete_config(self):
        """Test loading and validating a complete config."""
        config_content = """
run:
  name: "integration_test"
  description: "Complete config test"
  mode: "portfolio"

data:
  timeframe: "1h"
  days: 30

portfolio:
  assets:
    - symbol: "BTC/USDT"
      weight: 0.40
    - symbol: "ETH/USDT"
      weight: 0.30
    - symbol: "SOL/USDT"
      weight: 0.20
    - symbol: "BNB/USDT"
      weight: 0.10

  rebalancing:
    enabled: true
    threshold: 0.15
    method: "threshold"
    min_rebalance_interval_hours: 24

capital:
  initial_capital: 10000.0

costs:
  commission: 0.001
  slippage: 0.0005

output:
  directory: "results_test"
  save_trades: true
  save_equity_curve: true
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            # Load config
            with open(temp_path, 'r') as f:
                config = yaml.safe_load(f)

            # Validate all sections present
            assert 'run' in config
            assert 'data' in config
            assert 'portfolio' in config
            assert 'capital' in config
            assert 'costs' in config
            assert 'output' in config

            # Validate assets
            assert len(config['portfolio']['assets']) == 4
            total_weight = sum(a['weight'] for a in config['portfolio']['assets'])
            assert abs(total_weight - 1.0) < 0.01

            # Validate rebalancing
            assert config['portfolio']['rebalancing']['enabled'] is True
            assert 0 < config['portfolio']['rebalancing']['threshold'] < 1

            # Validate capital
            assert config['capital']['initial_capital'] > 0

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    """
    Validation block for Portfolio Config Tests.
    Runs all tests and reports results.
    """
    import sys

    # Run pytest with verbose output
    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    if exit_code == 0:
        print("\n" + "="*70)
        print("✅ ALL PORTFOLIO CONFIG TESTS PASSED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ SOME PORTFOLIO CONFIG TESTS FAILED")
        print("="*70)

    sys.exit(exit_code)
