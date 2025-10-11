"""
Analysis and metrics module for backtesting results.

This module provides comprehensive tools for analyzing and comparing
trading strategy performance including metrics calculation, multi-strategy
comparison, and report generation.

Public APIs:
    MetricsCalculator: Calculate performance metrics
    StrategyComparison: Compare multiple strategies
    ReportGenerator: Generate reports and visualizations

Example Usage:
    from crypto_trader.analysis import MetricsCalculator, StrategyComparison, ReportGenerator

    # Calculate metrics
    calculator = MetricsCalculator(risk_free_rate=0.02)
    metrics = calculator.calculate_all_metrics(returns, trades, equity_curve, initial_capital)

    # Compare strategies
    comparison = StrategyComparison()
    df = comparison.compare_strategies([result1, result2, result3])
    best = comparison.best_performer([result1, result2, result3], metric="sharpe_ratio")

    # Generate report
    reporter = ReportGenerator()
    reporter.generate_html_report(result, "report.html")
    reporter.export_to_json(result, "results.json")
"""

from crypto_trader.analysis.comparison import StrategyComparison
from crypto_trader.analysis.metrics import MetricsCalculator
from crypto_trader.analysis.reporting import ReportGenerator

__all__ = [
    "MetricsCalculator",
    "StrategyComparison",
    "ReportGenerator",
]
