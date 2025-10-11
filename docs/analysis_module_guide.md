# Analysis and Metrics Module Guide

## Overview

The `crypto_trader.analysis` module provides comprehensive tools for analyzing and comparing trading strategy performance. It includes metrics calculation, multi-strategy comparison, statistical testing, and report generation with interactive visualizations.

## Components

### 1. MetricsCalculator

Calculate comprehensive performance metrics for backtesting results.

#### Features
- Risk-adjusted returns (Sharpe ratio, Sortino ratio)
- Drawdown metrics (max drawdown, Calmar ratio, recovery factor)
- Trade statistics (win rate, profit factor, expectancy)
- Consecutive wins/losses tracking
- Average trade duration

#### Usage

```python
from crypto_trader.analysis import MetricsCalculator
from crypto_trader.core.types import Trade
import pandas as pd

# Initialize calculator
calculator = MetricsCalculator(risk_free_rate=0.02)  # 2% annual risk-free rate

# Calculate all metrics
metrics = calculator.calculate_all_metrics(
    returns=returns_series,  # pandas Series of period returns
    trades=trades_list,      # List of Trade objects
    equity_curve=equity_curve,  # List of (timestamp, equity_value) tuples
    initial_capital=10000.0
)

# Access calculated metrics
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
print(f"Win Rate: {metrics.win_rate:.2%}")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
print(f"Expectancy: ${metrics.expectancy:.2f} per trade")
```

#### Individual Metric Methods

```python
# Calculate specific metrics
sharpe = calculator.sharpe_ratio(returns, risk_free_rate=0.02)
sortino = calculator.sortino_ratio(returns, risk_free_rate=0.02)
max_dd = calculator.max_drawdown(equity_curve)
pf = calculator.profit_factor(trades)
exp = calculator.expectancy(trades)

# Win/loss analysis
avg_win, avg_loss = calculator.average_win_loss(trades)
max_wins, max_losses = calculator.consecutive_wins_losses(trades)
avg_duration = calculator.average_trade_duration(trades)
```

### 2. StrategyComparison

Compare multiple strategies and perform statistical analysis.

#### Features
- Multi-strategy comparison tables
- Strategy ranking by any metric
- Correlation matrix between strategies
- Statistical significance testing
- Strategy filtering by criteria

#### Usage

```python
from crypto_trader.analysis import StrategyComparison

comparison = StrategyComparison()

# Compare all strategies
df = comparison.compare_strategies(results, normalize=False)
print(df[['strategy', 'total_return', 'sharpe_ratio', 'max_drawdown']])

# Rank strategies by metric
ranked = comparison.rank_strategies(
    results,
    metric="sharpe_ratio",
    ascending=False  # Higher is better
)

# Find best performer
best_sharpe = comparison.best_performer(results, metric="sharpe_ratio")
best_return = comparison.best_performer(results, metric="total_return")
best_dd = comparison.best_performer(results, metric="max_drawdown")

# Calculate correlation between strategies
corr_matrix = comparison.correlation_matrix(results, method="pearson")

# Test statistical significance
sig_test = comparison.statistical_significance(result1, result2, alpha=0.05)
print(f"P-value: {sig_test['p_value']:.4f}")
print(f"Significant: {sig_test['significant']}")

# Get comprehensive summary
summary = comparison.multi_strategy_summary(results)
print(f"Total Strategies: {summary['total_strategies']}")
print(f"Profitable: {summary['profitable_strategies']}")
print(f"Best Sharpe: {summary['best_performers']['sharpe_ratio']}")

# Filter strategies by criteria
filtered = comparison.filter_strategies(
    results,
    min_sharpe=1.5,
    max_drawdown=0.20,
    min_trades=30
)
```

### 3. ReportGenerator

Generate comprehensive reports and interactive visualizations.

#### Features
- HTML reports with embedded charts
- Interactive Plotly charts (equity curve, drawdown, monthly returns)
- Strategy comparison charts
- JSON and CSV exports
- Professional styling with metrics cards

#### Usage

```python
from crypto_trader.analysis import ReportGenerator

reporter = ReportGenerator()

# Generate HTML report
reporter.generate_html_report(
    result=backtest_result,
    output_path="reports/backtest_report.html",
    include_trades=True
)

# Create individual charts
equity_chart = reporter.create_equity_curve_chart(result)
drawdown_chart = reporter.create_drawdown_chart(result)
monthly_chart = reporter.create_monthly_returns_chart(result)

# Comparison chart for multiple strategies
comparison_chart = reporter.create_comparison_chart(
    results=[result1, result2, result3],
    metric="sharpe_ratio"
)

# Export data
reporter.export_to_json(result, "data/backtest_result.json")
reporter.export_to_csv(result, "data/trades.csv")

# Save charts
equity_chart.write_html("charts/equity_curve.html")
comparison_chart.write_html("charts/strategy_comparison.html")
```

## Complete Example

```python
from datetime import datetime, timedelta
from crypto_trader.analysis import (
    MetricsCalculator,
    StrategyComparison,
    ReportGenerator
)
from crypto_trader.core.types import BacktestResult

# Assume we have multiple backtest results
results = [result1, result2, result3]

# 1. Calculate metrics for each result
calculator = MetricsCalculator(risk_free_rate=0.02)
for result in results:
    returns = calculator.calculate_returns_from_equity(result.equity_curve)
    metrics = calculator.calculate_all_metrics(
        returns=returns,
        trades=result.trades,
        equity_curve=result.equity_curve,
        initial_capital=result.initial_capital
    )
    # Metrics are already in result.metrics

# 2. Compare strategies
comparison = StrategyComparison()

# Get comparison DataFrame
df = comparison.compare_strategies(results)
print("\nStrategy Comparison:")
print(df[['strategy', 'total_return', 'sharpe_ratio', 'win_rate']])

# Rank by Sharpe ratio
ranked = comparison.rank_strategies(results, metric="sharpe_ratio")
print(f"\nBest Strategy: {ranked.iloc[0]['strategy']}")

# Check correlation
corr = comparison.correlation_matrix(results)
print(f"\nStrategy Correlation Matrix:")
print(corr)

# Test significance
if len(results) >= 2:
    sig_test = comparison.statistical_significance(results[0], results[1])
    print(f"\nSignificance Test: {sig_test['message']}")

# 3. Generate reports
reporter = ReportGenerator()

for result in results:
    # HTML report
    html_path = f"reports/{result.strategy_name}_report.html"
    reporter.generate_html_report(result, html_path)

    # Export data
    json_path = f"data/{result.strategy_name}_result.json"
    csv_path = f"data/{result.strategy_name}_trades.csv"
    reporter.export_to_json(result, json_path)
    reporter.export_to_csv(result, csv_path)

# Create comparison chart
comparison_chart = reporter.create_comparison_chart(results, "sharpe_ratio")
comparison_chart.write_html("reports/comparison.html")

print("\n✅ Analysis complete! Reports generated.")
```

## Performance Metrics Reference

### Return Metrics
- **Total Return**: Percentage gain/loss over the period
- **Final Capital**: Ending portfolio value

### Risk-Adjusted Returns
- **Sharpe Ratio**: Excess return per unit of total risk
  - > 1.0: Good
  - > 2.0: Excellent
- **Sortino Ratio**: Excess return per unit of downside risk
  - Only considers negative volatility
  - Generally higher than Sharpe
- **Calmar Ratio**: Annual return divided by max drawdown
  - > 3.0: Excellent
- **Recovery Factor**: Net profit divided by max drawdown in dollars

### Risk Metrics
- **Max Drawdown**: Largest peak-to-trough decline
  - < 10%: Excellent
  - < 20%: Good
  - < 30%: Fair
  - > 30%: Poor

### Trade Statistics
- **Win Rate**: Percentage of profitable trades
  - > 60%: Good
  - > 70%: Excellent
- **Profit Factor**: Gross profit / Gross loss
  - > 1.0: Profitable
  - > 2.0: Excellent
- **Expectancy**: Average expected profit per trade
  - Should be positive for profitable strategy
- **Average Win/Loss**: Mean profit/loss per winning/losing trade
- **Max Consecutive Wins/Losses**: Longest streaks

### Other Metrics
- **Total Trades**: Number of completed trades
- **Total Fees**: Cumulative transaction costs
- **Average Trade Duration**: Mean time in position

## Quality Ratings

The system automatically assigns quality ratings based on risk-adjusted metrics:

- **Excellent**: Sharpe ≥ 2.0 AND Max Drawdown < 15%
- **Good**: Sharpe ≥ 1.5 AND Max Drawdown < 25%
- **Fair**: Sharpe ≥ 1.0 AND Max Drawdown < 35%
- **Poor**: Below Fair thresholds

## Statistical Testing

The module uses independent t-tests to determine if performance differences between strategies are statistically significant:

```python
sig_test = comparison.statistical_significance(result1, result2, alpha=0.05)

# Returns:
{
    'strategy1': 'Strategy A',
    'strategy2': 'Strategy B',
    'mean_return1': 0.0015,
    'mean_return2': 0.0012,
    't_statistic': 2.45,
    'p_value': 0.0145,
    'significant': True,  # if p_value < alpha
    'alpha': 0.05,
    'message': 'Performance difference is significant at 0.05 level'
}
```

## HTML Report Features

Generated HTML reports include:

1. **Summary Cards**: Key metrics with color-coded performance indicators
2. **Strategy Information Table**: Symbol, timeframe, dates, duration
3. **Performance Metrics Table**: Comprehensive metrics breakdown
4. **Interactive Charts**:
   - Equity curve with fill
   - Drawdown analysis
   - Monthly returns bar chart
5. **Trade Details Table**: Recent trades (up to 50)
6. **Professional Styling**: Modern design with responsive layout

## Best Practices

### 1. Risk-Free Rate Selection
- Use current treasury yields (e.g., 10-year T-bill rate)
- Default 2% is reasonable for most crypto strategies
- Higher for traditional finance (3-4%)

### 2. Statistical Significance
- Require at least 30 trades for meaningful statistics
- Use alpha=0.05 for 95% confidence
- Consider multiple testing corrections for many comparisons

### 3. Metric Selection
- Use Sharpe for overall risk-adjusted performance
- Use Sortino when downside risk is primary concern
- Use Calmar for drawdown-sensitive applications
- Use profit factor for trade-level performance

### 4. Strategy Filtering
- Set minimum Sharpe thresholds (e.g., > 1.5)
- Limit max drawdown (e.g., < 20%)
- Require minimum trades for statistical validity (e.g., > 30)

### 5. Report Generation
- Always include trades in HTML reports for transparency
- Export to JSON for programmatic analysis
- Export to CSV for spreadsheet analysis
- Create comparison charts when evaluating multiple strategies

## Module Files

```
crypto_trader/analysis/
├── __init__.py           # Public API exports
├── metrics.py            # MetricsCalculator class
├── comparison.py         # StrategyComparison class
└── reporting.py          # ReportGenerator class
```

## Dependencies

- **pandas**: DataFrame operations and time series
- **numpy**: Numerical calculations
- **scipy**: Statistical tests
- **plotly**: Interactive visualizations

All dependencies are included in the project's `pyproject.toml`.

## Examples

See `/home/fiod/crypto/examples/analysis_demo.py` for a complete working example demonstrating all module functionality.

Run the demo:
```bash
cd /home/fiod/crypto
uv run python examples/analysis_demo.py
```

This generates:
- HTML report with interactive charts
- JSON export of results
- CSV export of trades
- Comparison chart for multiple strategies

Output files are saved to `examples/output/`.
