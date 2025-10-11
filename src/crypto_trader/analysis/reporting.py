"""
Report generation and visualization for backtest results.

This module provides functionality to generate comprehensive reports including
HTML reports, interactive charts, and data exports in various formats.

Documentation:
- Plotly: https://plotly.com/python/
- Pandas: https://pandas.pydata.org/docs/

Sample Input:
    reporter = ReportGenerator()
    reporter.generate_html_report(backtest_result, output_path="report.html")
    reporter.create_equity_curve_chart(backtest_result)
    reporter.export_to_json(backtest_result, "results.json")

Expected Output:
    - HTML report with charts and metrics
    - Interactive Plotly charts
    - JSON/CSV exports of results
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from crypto_trader.core.types import BacktestResult


class ReportGenerator:
    """
    Generate reports and visualizations for backtest results.

    This class creates comprehensive reports including HTML reports with
    embedded charts, standalone visualizations, and data exports.
    """

    def generate_html_report(
        self, result: BacktestResult, output_path: str, include_trades: bool = True
    ) -> None:
        """
        Generate comprehensive HTML report with charts and metrics.

        Args:
            result: BacktestResult to generate report from
            output_path: Path to save HTML file
            include_trades: Whether to include individual trade details
        """
        # Create charts
        equity_chart = self.create_equity_curve_chart(result)
        drawdown_chart = self.create_drawdown_chart(result)
        monthly_returns_chart = self.create_monthly_returns_chart(result)

        equity_chart_html = equity_chart.to_html(include_plotlyjs=False, full_html=False)
        drawdown_chart_html = drawdown_chart.to_html(include_plotlyjs=False, full_html=False)
        monthly_returns_chart_html = monthly_returns_chart.to_html(include_plotlyjs=False, full_html=False)

        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Backtest Report - {result.strategy_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #007bff;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                    border-bottom: 2px solid #ddd;
                    padding-bottom: 8px;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .metric-card.positive {{
                    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                }}
                .metric-card.negative {{
                    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
                }}
                .metric-label {{
                    font-size: 12px;
                    opacity: 0.9;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .metric-value {{
                    font-size: 28px;
                    font-weight: bold;
                    margin-top: 8px;
                }}
                .chart {{
                    margin: 20px 0;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 10px;
                }}
                .info-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .info-table th, .info-table td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .info-table th {{
                    background-color: #f8f9fa;
                    font-weight: 600;
                }}
                .info-table tr:hover {{
                    background-color: #f8f9fa;
                }}
                .quality-badge {{
                    display: inline-block;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                .quality-excellent {{ background-color: #28a745; color: white; }}
                .quality-good {{ background-color: #17a2b8; color: white; }}
                .quality-fair {{ background-color: #ffc107; color: black; }}
                .quality-poor {{ background-color: #dc3545; color: white; }}
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 2px solid #ddd;
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{result.strategy_name} - Backtest Report</h1>

                <div class="summary">
                    <div class="metric-card {self._get_return_class(result.metrics.total_return)}">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value">{result.metrics.total_return:.2%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{result.metrics.sharpe_ratio:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value">{result.metrics.max_drawdown:.2%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{result.metrics.win_rate:.2%}</div>
                    </div>
                </div>

                <h2>Strategy Information</h2>
                <table class="info-table">
                    <tr>
                        <th>Symbol</th>
                        <td>{result.symbol}</td>
                        <th>Timeframe</th>
                        <td>{result.timeframe.value}</td>
                    </tr>
                    <tr>
                        <th>Start Date</th>
                        <td>{result.start_date.strftime('%Y-%m-%d %H:%M')}</td>
                        <th>End Date</th>
                        <td>{result.end_date.strftime('%Y-%m-%d %H:%M')}</td>
                    </tr>
                    <tr>
                        <th>Duration</th>
                        <td>{result.duration_days} days</td>
                        <th>Quality</th>
                        <td>
                            <span class="quality-badge quality-{result.metrics.risk_adjusted_quality().lower()}">
                                {result.metrics.risk_adjusted_quality()}
                            </span>
                        </td>
                    </tr>
                </table>

                <h2>Performance Metrics</h2>
                <table class="info-table">
                    <tr>
                        <th>Initial Capital</th>
                        <td>${result.initial_capital:,.2f}</td>
                        <th>Final Capital</th>
                        <td>${result.metrics.final_capital:,.2f}</td>
                    </tr>
                    <tr>
                        <th>Total Trades</th>
                        <td>{result.metrics.total_trades}</td>
                        <th>Profit Factor</th>
                        <td>{result.metrics.profit_factor:.2f}</td>
                    </tr>
                    <tr>
                        <th>Winning Trades</th>
                        <td>{result.metrics.winning_trades}</td>
                        <th>Losing Trades</th>
                        <td>{result.metrics.losing_trades}</td>
                    </tr>
                    <tr>
                        <th>Average Win</th>
                        <td>${result.metrics.avg_win:,.2f}</td>
                        <th>Average Loss</th>
                        <td>${result.metrics.avg_loss:,.2f}</td>
                    </tr>
                    <tr>
                        <th>Sortino Ratio</th>
                        <td>{result.metrics.sortino_ratio:.2f}</td>
                        <th>Calmar Ratio</th>
                        <td>{result.metrics.calmar_ratio:.2f}</td>
                    </tr>
                    <tr>
                        <th>Expectancy</th>
                        <td>${result.metrics.expectancy:,.2f}</td>
                        <th>Total Fees</th>
                        <td>${result.metrics.total_fees:,.2f}</td>
                    </tr>
                </table>

                <h2>Equity Curve</h2>
                <div class="chart">
                    {equity_chart_html}
                </div>

                <h2>Drawdown Analysis</h2>
                <div class="chart">
                    {drawdown_chart_html}
                </div>

                <h2>Monthly Returns</h2>
                <div class="chart">
                    {monthly_returns_chart_html}
                </div>

                {self._generate_trades_section(result) if include_trades else ""}

                <div class="footer">
                    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                    Crypto Trader Analysis System
                </div>
            </div>
        </body>
        </html>
        """

        # Write HTML file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content)

    def create_equity_curve_chart(self, result: BacktestResult) -> go.Figure:
        """
        Create interactive equity curve chart.

        Args:
            result: BacktestResult to visualize

        Returns:
            Plotly Figure object
        """
        if len(result.equity_curve) == 0:
            return go.Figure()

        timestamps = [timestamp for timestamp, _ in result.equity_curve]
        equity = [equity_val for _, equity_val in result.equity_curve]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=equity,
                mode="lines",
                name="Equity",
                line=dict(color="#007bff", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 123, 255, 0.1)",
            )
        )

        # Add initial capital reference line
        fig.add_hline(
            y=result.initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital",
        )

        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            hovermode="x unified",
            template="plotly_white",
            height=400,
        )

        return fig

    def create_drawdown_chart(self, result: BacktestResult) -> go.Figure:
        """
        Create drawdown chart showing peak-to-trough declines.

        Args:
            result: BacktestResult to visualize

        Returns:
            Plotly Figure object
        """
        if len(result.equity_curve) == 0:
            return go.Figure()

        df = pd.DataFrame(result.equity_curve, columns=["timestamp", "equity"])
        df["running_max"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["running_max"]) / df["running_max"]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["drawdown"] * 100,  # Convert to percentage
                mode="lines",
                name="Drawdown",
                line=dict(color="#dc3545", width=2),
                fill="tozeroy",
                fillcolor="rgba(220, 53, 69, 0.2)",
            )
        )

        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            template="plotly_white",
            height=300,
        )

        return fig

    def create_monthly_returns_chart(self, result: BacktestResult) -> go.Figure:
        """
        Create monthly returns bar chart.

        Args:
            result: BacktestResult to visualize

        Returns:
            Plotly Figure object
        """
        if len(result.equity_curve) < 2:
            return go.Figure()

        df = pd.DataFrame(result.equity_curve, columns=["timestamp", "equity"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Resample to monthly and calculate returns
        monthly = df["equity"].resample("ME").last()
        monthly_returns = monthly.pct_change().dropna() * 100  # Convert to percentage

        colors = ["#28a745" if x > 0 else "#dc3545" for x in monthly_returns]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=monthly_returns.index,
                y=monthly_returns.values,
                marker_color=colors,
                name="Monthly Return",
            )
        )

        fig.update_layout(
            title="Monthly Returns",
            xaxis_title="Month",
            yaxis_title="Return (%)",
            hovermode="x unified",
            template="plotly_white",
            height=300,
        )

        return fig

    def create_comparison_chart(
        self, results: list[BacktestResult], metric: str = "total_return"
    ) -> go.Figure:
        """
        Create comparison chart for multiple strategies.

        Args:
            results: List of BacktestResult objects
            metric: Metric to compare (default: total_return)

        Returns:
            Plotly Figure object
        """
        if len(results) == 0:
            return go.Figure()

        strategies = [r.strategy_name for r in results]
        values = [getattr(r.metrics, metric) for r in results]

        # Convert to percentage if needed
        if metric in ["total_return", "max_drawdown", "win_rate"]:
            values = [v * 100 for v in values]
            y_title = f"{metric.replace('_', ' ').title()} (%)"
        else:
            y_title = metric.replace("_", " ").title()

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=strategies,
                y=values,
                marker_color="#007bff",
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
            )
        )

        fig.update_layout(
            title=f"Strategy Comparison - {metric.replace('_', ' ').title()}",
            xaxis_title="Strategy",
            yaxis_title=y_title,
            template="plotly_white",
            height=400,
        )

        return fig

    def export_to_json(self, result: BacktestResult, output_path: str) -> None:
        """
        Export backtest result to JSON format.

        Args:
            result: BacktestResult to export
            output_path: Path to save JSON file
        """
        export_data = {
            "strategy_name": result.strategy_name,
            "symbol": result.symbol,
            "timeframe": result.timeframe.value,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_capital": result.initial_capital,
            "metrics": {
                "total_return": result.metrics.total_return,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "sortino_ratio": result.metrics.sortino_ratio,
                "max_drawdown": result.metrics.max_drawdown,
                "calmar_ratio": result.metrics.calmar_ratio,
                "win_rate": result.metrics.win_rate,
                "profit_factor": result.metrics.profit_factor,
                "total_trades": result.metrics.total_trades,
                "winning_trades": result.metrics.winning_trades,
                "losing_trades": result.metrics.losing_trades,
                "avg_win": result.metrics.avg_win,
                "avg_loss": result.metrics.avg_loss,
                "expectancy": result.metrics.expectancy,
                "total_fees": result.metrics.total_fees,
                "final_capital": result.metrics.final_capital,
                "quality": result.metrics.risk_adjusted_quality(),
            },
            "trades": [
                {
                    "symbol": t.symbol,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "side": t.side.value,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_percent": t.pnl_percent,
                    "fees": t.fees,
                }
                for t in result.trades
            ],
            "equity_curve": [
                {"timestamp": ts.isoformat(), "equity": eq}
                for ts, eq in result.equity_curve
            ],
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(export_data, indent=2))

    def export_to_csv(self, result: BacktestResult, output_path: str) -> None:
        """
        Export trades to CSV format.

        Args:
            result: BacktestResult to export
            output_path: Path to save CSV file
        """
        if len(result.trades) == 0:
            return

        trades_data = []
        for trade in result.trades:
            trades_data.append(
                {
                    "symbol": trade.symbol,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "side": trade.side.value,
                    "quantity": trade.quantity,
                    "pnl": trade.pnl,
                    "pnl_percent": trade.pnl_percent,
                    "fees": trade.fees,
                    "duration_minutes": trade.duration_minutes,
                }
            )

        df = pd.DataFrame(trades_data)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

    def _get_return_class(self, total_return: float) -> str:
        """Helper to determine CSS class for return metric."""
        return "positive" if total_return > 0 else "negative"

    def _generate_trades_section(self, result: BacktestResult) -> str:
        """Generate HTML for trades table section."""
        if len(result.trades) == 0:
            return ""

        rows = []
        for trade in result.trades[:50]:  # Limit to first 50 trades for performance
            pnl_class = "positive" if trade.is_winning else "negative"
            rows.append(
                f"""
                <tr>
                    <td>{trade.entry_time.strftime('%Y-%m-%d %H:%M')}</td>
                    <td>{trade.exit_time.strftime('%Y-%m-%d %H:%M')}</td>
                    <td>{trade.side.value}</td>
                    <td>${trade.entry_price:,.2f}</td>
                    <td>${trade.exit_price:,.2f}</td>
                    <td style="color: {'green' if trade.is_winning else 'red'}; font-weight: bold;">
                        ${trade.pnl:,.2f} ({trade.pnl_percent:.2f}%)
                    </td>
                </tr>
                """
            )

        return f"""
        <h2>Recent Trades</h2>
        <table class="info-table">
            <thead>
                <tr>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Side</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>PnL</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """


if __name__ == "__main__":
    """
    Validation function to test report generation with real backtest data.
    """
    import sys
    from datetime import timedelta
    from tempfile import TemporaryDirectory

    from crypto_trader.core.types import (
        OrderSide,
        OrderType,
        PerformanceMetrics,
        Timeframe,
        Trade,
    )

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating reporting.py with real backtest data...\n")

    # Create sample backtest result for testing
    base_time = datetime(2024, 1, 1)
    end_time = datetime(2024, 12, 31)

    sample_trades = [
        Trade(
            symbol="BTCUSDT",
            entry_time=base_time + timedelta(days=i * 10),
            exit_time=base_time + timedelta(days=i * 10 + 2),
            entry_price=45000.0 + i * 500,
            exit_price=45000.0 + i * 500 + (300 if i % 2 == 0 else -200),
            side=OrderSide.BUY,
            quantity=0.1,
            pnl=30.0 if i % 2 == 0 else -20.0,
            pnl_percent=0.67 if i % 2 == 0 else -0.44,
            fees=10.0,
            order_type=OrderType.MARKET,
        )
        for i in range(10)
    ]

    equity_curve = [
        (base_time, 10000.0),
        (base_time + timedelta(days=30), 10500.0),
        (base_time + timedelta(days=60), 11200.0),
        (base_time + timedelta(days=90), 11800.0),
        (base_time + timedelta(days=120), 12200.0),
        (base_time + timedelta(days=150), 11900.0),
        (base_time + timedelta(days=180), 12400.0),
        (base_time + timedelta(days=210), 13000.0),
        (base_time + timedelta(days=240), 13200.0),
        (base_time + timedelta(days=270), 13100.0),
        (base_time + timedelta(days=300), 13500.0),
        (end_time, 13500.0),
    ]

    sample_result = BacktestResult(
        strategy_name="MA Crossover",
        symbol="BTCUSDT",
        timeframe=Timeframe.HOUR_4,
        start_date=base_time,
        end_date=end_time,
        initial_capital=10000.0,
        metrics=PerformanceMetrics(
            total_return=0.35,
            sharpe_ratio=2.5,
            sortino_ratio=3.0,
            max_drawdown=0.12,
            calmar_ratio=2.92,
            win_rate=0.68,
            profit_factor=2.1,
            total_trades=45,
            winning_trades=31,
            losing_trades=14,
            avg_win=85.5,
            avg_loss=-42.3,
            expectancy=78.5,
            total_fees=450.0,
            final_capital=13500.0,
        ),
        trades=sample_trades,
        equity_curve=equity_curve,
    )

    reporter = ReportGenerator()

    # Test 1: Create equity curve chart
    total_tests += 1
    print("Test 1: Create equity curve chart")
    try:
        equity_chart = reporter.create_equity_curve_chart(sample_result)

        if not isinstance(equity_chart, go.Figure):
            all_validation_failures.append(
                f"Equity chart: Expected go.Figure, got {type(equity_chart)}"
            )

        # Check that chart has data
        if len(equity_chart.data) == 0:
            all_validation_failures.append("Equity chart has no data traces")

        print(f"  ✓ Created equity curve chart")
        print(f"  ✓ Chart traces: {len(equity_chart.data)}")

    except Exception as e:
        all_validation_failures.append(f"Equity curve chart exception: {e}")

    # Test 2: Create drawdown chart
    total_tests += 1
    print("\nTest 2: Create drawdown chart")
    try:
        drawdown_chart = reporter.create_drawdown_chart(sample_result)

        if not isinstance(drawdown_chart, go.Figure):
            all_validation_failures.append(
                f"Drawdown chart: Expected go.Figure, got {type(drawdown_chart)}"
            )

        if len(drawdown_chart.data) == 0:
            all_validation_failures.append("Drawdown chart has no data traces")

        print(f"  ✓ Created drawdown chart")
        print(f"  ✓ Chart traces: {len(drawdown_chart.data)}")

    except Exception as e:
        all_validation_failures.append(f"Drawdown chart exception: {e}")

    # Test 3: Create monthly returns chart
    total_tests += 1
    print("\nTest 3: Create monthly returns chart")
    try:
        monthly_chart = reporter.create_monthly_returns_chart(sample_result)

        if not isinstance(monthly_chart, go.Figure):
            all_validation_failures.append(
                f"Monthly chart: Expected go.Figure, got {type(monthly_chart)}"
            )

        print(f"  ✓ Created monthly returns chart")
        print(f"  ✓ Chart traces: {len(monthly_chart.data)}")

    except Exception as e:
        all_validation_failures.append(f"Monthly returns chart exception: {e}")

    # Test 4: Create comparison chart
    total_tests += 1
    print("\nTest 4: Create comparison chart")
    try:
        # Create multiple results for comparison
        result2 = BacktestResult(
            strategy_name="RSI Strategy",
            symbol="ETHUSDT",
            timeframe=Timeframe.HOUR_1,
            start_date=base_time,
            end_date=end_time,
            initial_capital=10000.0,
            metrics=PerformanceMetrics(
                total_return=0.22,
                sharpe_ratio=1.8,
                max_drawdown=0.18,
                win_rate=0.55,
                profit_factor=1.6,
                total_trades=78,
                final_capital=12200.0,
            ),
            equity_curve=equity_curve,
        )

        comparison_chart = reporter.create_comparison_chart(
            [sample_result, result2], metric="sharpe_ratio"
        )

        if not isinstance(comparison_chart, go.Figure):
            all_validation_failures.append(
                f"Comparison chart: Expected go.Figure, got {type(comparison_chart)}"
            )

        print(f"  ✓ Created comparison chart")
        print(f"  ✓ Comparing 2 strategies")

    except Exception as e:
        all_validation_failures.append(f"Comparison chart exception: {e}")

    # Test 5: Export to JSON
    total_tests += 1
    print("\nTest 5: Export to JSON")
    try:
        with TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "result.json"
            reporter.export_to_json(sample_result, str(json_path))

            if not json_path.exists():
                all_validation_failures.append("JSON file was not created")
            else:
                # Verify JSON is valid
                with open(json_path) as f:
                    data = json.load(f)

                if data["strategy_name"] != "MA Crossover":
                    all_validation_failures.append(
                        f"JSON strategy: Expected 'MA Crossover', got '{data['strategy_name']}'"
                    )

                if "metrics" not in data:
                    all_validation_failures.append("JSON missing metrics")

                print(f"  ✓ Exported to JSON successfully")
                print(f"  ✓ File size: {json_path.stat().st_size} bytes")
                print(f"  ✓ Contains {len(data['trades'])} trades")

    except Exception as e:
        all_validation_failures.append(f"Export to JSON exception: {e}")

    # Test 6: Export to CSV
    total_tests += 1
    print("\nTest 6: Export to CSV")
    try:
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "trades.csv"
            reporter.export_to_csv(sample_result, str(csv_path))

            if not csv_path.exists():
                all_validation_failures.append("CSV file was not created")
            else:
                # Verify CSV is valid
                df = pd.read_csv(csv_path)

                if len(df) != len(sample_trades):
                    all_validation_failures.append(
                        f"CSV rows: Expected {len(sample_trades)}, got {len(df)}"
                    )

                expected_columns = ["symbol", "entry_time", "exit_time", "pnl"]
                for col in expected_columns:
                    if col not in df.columns:
                        all_validation_failures.append(f"CSV missing column: {col}")

                print(f"  ✓ Exported to CSV successfully")
                print(f"  ✓ Rows: {len(df)}")
                print(f"  ✓ Columns: {list(df.columns)}")

    except Exception as e:
        all_validation_failures.append(f"Export to CSV exception: {e}")

    # Test 7: Generate HTML report
    total_tests += 1
    print("\nTest 7: Generate HTML report")
    try:
        with TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "report.html"
            reporter.generate_html_report(sample_result, str(html_path))

            if not html_path.exists():
                all_validation_failures.append("HTML file was not created")
            else:
                html_content = html_path.read_text()

                # Verify essential elements
                required_elements = [
                    "MA Crossover",
                    "Total Return",
                    "Sharpe Ratio",
                    "Equity Curve",
                    "Drawdown Analysis",
                ]

                for element in required_elements:
                    if element not in html_content:
                        all_validation_failures.append(
                            f"HTML missing element: {element}"
                        )

                print(f"  ✓ Generated HTML report successfully")
                print(f"  ✓ File size: {html_path.stat().st_size} bytes")
                print(f"  ✓ Contains all required sections")

    except Exception as e:
        all_validation_failures.append(f"HTML report exception: {e}")

    # Test 8: Edge case - Empty equity curve
    total_tests += 1
    print("\nTest 8: Edge case - Empty equity curve")
    try:
        empty_result = BacktestResult(
            strategy_name="Empty Strategy",
            symbol="BTCUSDT",
            timeframe=Timeframe.HOUR_1,
            start_date=base_time,
            end_date=end_time,
            initial_capital=10000.0,
            metrics=PerformanceMetrics(),
            equity_curve=[],
            trades=[],
        )

        empty_chart = reporter.create_equity_curve_chart(empty_result)

        if not isinstance(empty_chart, go.Figure):
            all_validation_failures.append("Empty result should still return go.Figure")

        print("  ✓ Empty inputs handled correctly")
        print("  ✓ Returns empty Figure")

    except Exception as e:
        all_validation_failures.append(f"Empty equity curve exception: {e}")

    # Final validation result
    print("\n" + "=" * 60)
    if all_validation_failures:
        print(
            f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(
            f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results"
        )
        print("Function is validated and formal tests can now be written")
        sys.exit(0)
