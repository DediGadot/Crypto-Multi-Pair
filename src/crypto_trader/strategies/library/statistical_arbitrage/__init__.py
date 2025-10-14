"""
Statistical Arbitrage Strategy Module

Implements the ARASA (Adaptive Regime-Aware Statistical Arbitrage) strategy
for cryptocurrency pairs trading.
"""

from .cointegration import CointegrationAnalyzer
from .regime_detection import RegimeDetector

__all__ = ['CointegrationAnalyzer', 'RegimeDetector']
