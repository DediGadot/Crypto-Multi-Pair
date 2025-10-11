# Analysis Module Quick Reference

## Function Usage Table

### MetricsCalculator

| Function | Input | Output | Example |
|----------|-------|--------|---------|
| `__init__(risk_free_rate)` | `float` | `MetricsCalculator` | `calc = MetricsCalculator(0.02)` |
| `calculate_all_metrics(returns, trades, equity_curve, initial_capital)` | `pd.Series, list[Trade], list[tuple], float` | `PerformanceMetrics` | `metrics = calc.calculate_all_metrics(returns, trades, equity, 10000)` |
| `sharpe_ratio(returns, risk_free_rate)` | `pd.Series, float` | `float` | `sharpe = calc.sharpe_ratio(returns, 0.02)` |
| `sortino_ratio(returns, risk_free_rate)` | `pd.Series, float` | `float` | `sortino = calc.sortino_ratio(returns, 0.02)` |
| `max_drawdown(equity_curve)` | `list[tuple]` | `float` | `max_dd = calc.max_drawdown(equity_curve)` |
| `profit_factor(trades)` | `list[Trade]` | `float` | `pf = calc.profit_factor(trades)` |
| `average_win_loss(trades)` | `list[Trade]` | `tuple[float, float]` | `avg_win, avg_loss = calc.average_win_loss(trades)` |
| `consecutive_wins_losses(trades)` | `list[Trade]` | `tuple[int, int]` | `max_wins, max_losses = calc.consecutive_wins_losses(trades)` |
| `average_trade_duration(trades)` | `list[Trade]` | `float` | `avg_dur = calc.average_trade_duration(trades)` |
| `calmar_ratio(total_return, max_drawdown)` | `float, float` | `float` | `calmar = calc.calmar_ratio(0.25, 0.10)` |
| `recovery_factor(total_return, max_drawdown, initial, final)` | `float, float, float, float` | `float` | `recovery = calc.recovery_factor(0.25, 0.10, 10000, 12500)` |
| `expectancy(trades)` | `list[Trade]` | `float` | `exp = calc.expectancy(trades)` |
| `calculate_returns_from_equity(equity_curve)` | `list[tuple]` | `pd.Series` | `returns = calc.calculate_returns_from_equity(equity)` |

### StrategyComparison

| Function | Input | Output | Example |
|----------|-------|--------|---------|
| `compare_strategies(results, normalize)` | `list[BacktestResult], bool` | `pd.DataFrame` | `df = comp.compare_strategies(results, normalize=False)` |
| `rank_strategies(results, metric, ascending)` | `list[BacktestResult], str, bool` | `pd.DataFrame` | `ranked = comp.rank_strategies(results, "sharpe_ratio")` |
| `correlation_matrix(results, method)` | `list[BacktestResult], str` | `pd.DataFrame` | `corr = comp.correlation_matrix(results, "pearson")` |
| `best_performer(results, metric)` | `list[BacktestResult], str` | `BacktestResult` | `best = comp.best_performer(results, "sharpe_ratio")` |
| `statistical_significance(result1, result2, alpha)` | `BacktestResult, BacktestResult, float` | `dict` | `sig = comp.statistical_significance(r1, r2, 0.05)` |
| `multi_strategy_summary(results)` | `list[BacktestResult]` | `dict` | `summary = comp.multi_strategy_summary(results)` |
| `filter_strategies(results, min_sharpe, max_dd, min_trades)` | `list[BacktestResult], float, float, int` | `list[BacktestResult]` | `filtered = comp.filter_strategies(results, 1.5, 0.20, 30)` |

### ReportGenerator

| Function | Input | Output | Example |
|----------|-------|--------|---------|
| `generate_html_report(result, output_path, include_trades)` | `BacktestResult, str, bool` | `None` | `reporter.generate_html_report(result, "report.html", True)` |
| `create_equity_curve_chart(result)` | `BacktestResult` | `go.Figure` | `chart = reporter.create_equity_curve_chart(result)` |
| `create_drawdown_chart(result)` | `BacktestResult` | `go.Figure` | `chart = reporter.create_drawdown_chart(result)` |
| `create_monthly_returns_chart(result)` | `BacktestResult` | `go.Figure` | `chart = reporter.create_monthly_returns_chart(result)` |
| `create_comparison_chart(results, metric)` | `list[BacktestResult], str` | `go.Figure` | `chart = reporter.create_comparison_chart(results, "sharpe_ratio")` |
| `export_to_json(result, output_path)` | `BacktestResult, str` | `None` | `reporter.export_to_json(result, "result.json")` |
| `export_to_csv(result, output_path)` | `BacktestResult, str` | `None` | `reporter.export_to_csv(result, "trades.csv")` |

## Metric Reference Table

| Metric | Formula | Good Value | Excellent Value | Description |
|--------|---------|------------|-----------------|-------------|
| **Total Return** | `(final - initial) / initial` | > 0.10 (10%) | > 0.30 (30%) | Total percentage gain/loss |
| **Sharpe Ratio** | `(return - rf) / std_dev` | > 1.0 | > 2.0 | Risk-adjusted return |
| **Sortino Ratio** | `(return - rf) / downside_std` | > 1.5 | > 2.5 | Downside risk-adjusted return |
| **Max Drawdown** | `max((peak - trough) / peak)` | < 0.20 (20%) | < 0.10 (10%) | Largest peak-to-trough decline |
| **Calmar Ratio** | `return / max_drawdown` | > 2.0 | > 3.0 | Return per unit of drawdown |
| **Win Rate** | `winning_trades / total_trades` | > 0.55 (55%) | > 0.70 (70%) | Percentage of winning trades |
| **Profit Factor** | `gross_profit / gross_loss` | > 1.5 | > 2.0 | Ratio of profits to losses |
| **Expectancy** | `(win_rate × avg_win) - (loss_rate × avg_loss)` | > 0 | > 50 | Expected profit per trade |
| **Recovery Factor** | `net_profit / max_drawdown_dollars` | > 2.0 | > 5.0 | Profit per dollar of max drawdown |

## Common Use Cases

### 1. Analyze Single Strategy

```python
from crypto_trader.analysis import MetricsCalculator, ReportGenerator

# Calculate metrics
calculator = MetricsCalculator(risk_free_rate=0.02)
returns = calculator.calculate_returns_from_equity(result.equity_curve)
metrics = calculator.calculate_all_metrics(returns, result.trades, result.equity_curve, 10000)

# Generate report
reporter = ReportGenerator()
reporter.generate_html_report(result, "report.html")
reporter.export_to_json(result, "result.json")
```

### 2. Compare Multiple Strategies

```python
from crypto_trader.analysis import StrategyComparison, ReportGenerator

# Compare strategies
comparison = StrategyComparison()
df = comparison.compare_strategies([result1, result2, result3])
best = comparison.best_performer([result1, result2, result3], "sharpe_ratio")

# Create comparison chart
reporter = ReportGenerator()
chart = reporter.create_comparison_chart([result1, result2, result3], "sharpe_ratio")
chart.write_html("comparison.html")
```

### 3. Filter and Rank Strategies

```python
from crypto_trader.analysis import StrategyComparison

comparison = StrategyComparison()

# Filter by criteria
filtered = comparison.filter_strategies(
    results,
    min_sharpe=1.5,
    max_drawdown=0.20,
    min_trades=30
)

# Rank filtered strategies
ranked = comparison.rank_strategies(filtered, metric="total_return")
print(f"Best strategy: {ranked.iloc[0]['strategy']}")
```

### 4. Statistical Analysis

```python
from crypto_trader.analysis import StrategyComparison

comparison = StrategyComparison()

# Test significance
sig_test = comparison.statistical_significance(result1, result2, alpha=0.05)
print(f"P-value: {sig_test['p_value']:.4f}")
print(f"Significant: {sig_test['significant']}")

# Calculate correlation
corr_matrix = comparison.correlation_matrix([result1, result2, result3])
print(corr_matrix)
```

### 5. Custom Metric Calculation

```python
from crypto_trader.analysis import MetricsCalculator

calculator = MetricsCalculator(risk_free_rate=0.02)

# Calculate individual metrics
sharpe = calculator.sharpe_ratio(returns, 0.02)
sortino = calculator.sortino_ratio(returns, 0.02)
max_dd = calculator.max_drawdown(equity_curve)
pf = calculator.profit_factor(trades)
exp = calculator.expectancy(trades)

print(f"Sharpe: {sharpe:.2f}")
print(f"Sortino: {sortino:.2f}")
print(f"Max DD: {max_dd:.2%}")
print(f"Profit Factor: {pf:.2f}")
print(f"Expectancy: ${exp:.2f}")
```

## Chart Types

| Chart Type | Function | Best For | Interactive |
|------------|----------|----------|-------------|
| **Equity Curve** | `create_equity_curve_chart()` | Overall performance visualization | ✅ Yes |
| **Drawdown** | `create_drawdown_chart()` | Risk analysis | ✅ Yes |
| **Monthly Returns** | `create_monthly_returns_chart()` | Period-by-period analysis | ✅ Yes |
| **Comparison** | `create_comparison_chart()` | Multi-strategy comparison | ✅ Yes |

## Export Formats

| Format | Function | Best For | Size |
|--------|----------|----------|------|
| **HTML** | `generate_html_report()` | Human-readable reports | ~50 KB |
| **JSON** | `export_to_json()` | Programmatic analysis | ~15 KB |
| **CSV** | `export_to_csv()` | Spreadsheet analysis | ~3 KB |

## Metric Interpretation Guidelines

### Sharpe Ratio
- **< 0**: Strategy loses money or has negative risk-adjusted returns
- **0 - 1.0**: Subpar performance
- **1.0 - 2.0**: Good performance
- **> 2.0**: Excellent performance
- **> 3.0**: Exceptional (but verify data quality)

### Max Drawdown
- **< 5%**: Exceptional (very rare)
- **5% - 10%**: Excellent
- **10% - 20%**: Good
- **20% - 30%**: Acceptable for aggressive strategies
- **> 30%**: High risk, needs improvement

### Win Rate
- **< 40%**: Needs improvement (unless avg_win >> avg_loss)
- **40% - 50%**: Fair
- **50% - 60%**: Good
- **60% - 70%**: Excellent
- **> 70%**: Exceptional (verify not curve-fitted)

### Profit Factor
- **< 1.0**: Losing strategy
- **1.0 - 1.5**: Marginally profitable
- **1.5 - 2.0**: Good profitability
- **2.0 - 3.0**: Excellent profitability
- **> 3.0**: Exceptional (verify data quality)

## Time Complexity

| Function | Complexity | Notes |
|----------|------------|-------|
| `calculate_all_metrics()` | O(n) | n = number of trades/equity points |
| `sharpe_ratio()` | O(n) | n = number of returns |
| `max_drawdown()` | O(n) | n = number of equity points |
| `compare_strategies()` | O(m) | m = number of strategies |
| `correlation_matrix()` | O(m²×n) | m = strategies, n = returns |
| `statistical_significance()` | O(n) | n = number of returns |
| `generate_html_report()` | O(n) | n = total data points |

## Memory Usage

| Operation | Memory | Notes |
|-----------|--------|-------|
| Single strategy metrics | ~1 MB | For typical backtest (1000 trades) |
| Multiple strategy comparison | ~5 MB | For 10 strategies |
| HTML report generation | ~10 MB | Peak during chart generation |
| Large equity curves (10k points) | ~2 MB | Stored in memory during processing |

## Error Handling

All functions handle common error cases:
- ✅ Empty inputs (returns empty/zero results)
- ✅ Division by zero (returns 0.0 or inf appropriately)
- ✅ Missing data (skips gracefully)
- ✅ Invalid paths (creates directories as needed)
- ✅ Type mismatches (clear error messages)

## Performance Tips

1. **For large datasets**: Calculate metrics once and cache results
2. **For many strategies**: Use `compare_strategies()` instead of individual calculations
3. **For reports**: Generate HTML last after analysis is complete
4. **For correlations**: Use 'spearman' method for non-normal distributions
5. **For significance tests**: Ensure at least 30 data points for validity
