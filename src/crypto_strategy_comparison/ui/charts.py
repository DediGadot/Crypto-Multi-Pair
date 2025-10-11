"""
Chart Rendering Component

Provides all Plotly-based visualizations:
- Equity curves (normalized and absolute)
- Drawdown charts (underwater plots)
- Returns distribution (box/violin plots)
- Rolling metrics (Sharpe, volatility, win rate)
- Correlation matrix heatmap
- Risk-return scatter plot

Documentation:
- Plotly Python: https://plotly.com/python/
- Plotly Express: https://plotly.com/python/plotly-express/

Sample Input:
- comparison_results: Dict with strategy data and metrics
- chart_type: str specifying chart type

Expected Output:
- Plotly Figure object ready for st.plotly_chart()
"""

from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from loguru import logger


# Color palette for strategies (colorblind-friendly)
STRATEGY_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]


def render_equity_curves(results: Dict) -> go.Figure:
    """
    Render equity curves for all strategies.

    Args:
        results: Comparison results dictionary

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if not results or "equity_curves" not in results:
        logger.warning("No equity curve data available")
        return fig

    equity_data = results["equity_curves"]

    for idx, (strategy, data) in enumerate(equity_data.items()):
        color = STRATEGY_COLORS[idx % len(STRATEGY_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=data["dates"],
                y=data["values"],
                mode="lines",
                name=strategy,
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f"<b>{strategy}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Value: %{y:,.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Update layout
    fig.update_layout(
        title="Equity Curves - Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=500,
    )

    # Add range selector
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )

    return fig


def render_drawdown_chart(results: Dict) -> go.Figure:
    """
    Render drawdown (underwater) chart.

    Args:
        results: Comparison results dictionary

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if not results or "drawdowns" not in results:
        logger.warning("No drawdown data available")
        return fig

    drawdown_data = results["drawdowns"]

    for idx, (strategy, data) in enumerate(drawdown_data.items()):
        color = STRATEGY_COLORS[idx % len(STRATEGY_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=data["dates"],
                y=data["values"],
                mode="lines",
                name=strategy,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(list(bytes.fromhex(color[1:])) + [0.3])}",
                hovertemplate=(
                    f"<b>{strategy}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Drawdown: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Update layout
    fig.update_layout(
        title="Drawdown Analysis - Underwater Plot",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=400,
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def render_returns_distribution(results: Dict) -> go.Figure:
    """
    Render returns distribution box plot.

    Args:
        results: Comparison results dictionary

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if not results or "returns" not in results:
        logger.warning("No returns data available")
        return fig

    returns_data = results["returns"]

    for idx, (strategy, returns) in enumerate(returns_data.items()):
        color = STRATEGY_COLORS[idx % len(STRATEGY_COLORS)]

        fig.add_trace(
            go.Box(
                y=returns,
                name=strategy,
                marker_color=color,
                boxmean="sd",  # Show mean and standard deviation
            )
        )

    # Update layout
    fig.update_layout(
        title="Returns Distribution - Daily Returns",
        yaxis_title="Daily Return (%)",
        template="plotly_white",
        height=400,
        showlegend=True,
    )

    return fig


def render_rolling_metrics(results: Dict, metric: str = "sharpe") -> go.Figure:
    """
    Render rolling metrics chart (Sharpe, volatility, etc.).

    Args:
        results: Comparison results dictionary
        metric: Metric to display ('sharpe', 'volatility', 'win_rate')

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if not results or "rolling_metrics" not in results:
        logger.warning("No rolling metrics data available")
        return fig

    rolling_data = results["rolling_metrics"].get(metric, {})

    for idx, (strategy, data) in enumerate(rolling_data.items()):
        color = STRATEGY_COLORS[idx % len(STRATEGY_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=data["dates"],
                y=data["values"],
                mode="lines",
                name=strategy,
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{strategy}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    f"{metric.title()}: %{{y:.2f}}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Metric-specific configuration
    metric_config = {
        "sharpe": {
            "title": "Rolling Sharpe Ratio (90 days)",
            "yaxis": "Sharpe Ratio"
        },
        "volatility": {
            "title": "Rolling Volatility (90 days)",
            "yaxis": "Volatility (%)"
        },
        "win_rate": {
            "title": "Rolling Win Rate (90 days)",
            "yaxis": "Win Rate (%)"
        }
    }

    config = metric_config.get(metric, {"title": f"Rolling {metric}", "yaxis": metric})

    # Update layout
    fig.update_layout(
        title=config["title"],
        xaxis_title="Date",
        yaxis_title=config["yaxis"],
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=400,
    )

    return fig


def render_correlation_matrix(results: Dict) -> go.Figure:
    """
    Render correlation matrix heatmap.

    Args:
        results: Comparison results dictionary

    Returns:
        Plotly Figure object
    """
    if not results or "correlation_matrix" not in results:
        logger.warning("No correlation data available")
        return go.Figure()

    corr_matrix = results["correlation_matrix"]
    strategies = list(corr_matrix.keys())

    # Build correlation matrix
    matrix = []
    for strat1 in strategies:
        row = []
        for strat2 in strategies:
            row.append(corr_matrix[strat1].get(strat2, 0))
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=strategies,
            y=strategies,
            colorscale="RdYlGn",
            zmid=0,
            text=matrix,
            texttemplate="%{text:.2f}",
            textfont={"size": 12},
            colorbar=dict(title="Correlation"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Strategy Return Correlations",
        template="plotly_white",
        height=500,
        xaxis=dict(side="bottom"),
    )

    return fig


def render_risk_return_scatter(results: Dict) -> go.Figure:
    """
    Render risk-return scatter plot.

    Args:
        results: Comparison results dictionary

    Returns:
        Plotly Figure object
    """
    if not results or "metrics" not in results:
        logger.warning("No metrics data available")
        return go.Figure()

    metrics = results["metrics"]

    # Extract data for scatter plot
    strategies = []
    volatilities = []
    returns = []
    sharpes = []
    trade_counts = []

    for strategy, data in metrics.items():
        strategies.append(strategy)
        volatilities.append(data.get("volatility", 0))
        returns.append(data.get("total_return", 0))
        sharpes.append(data.get("sharpe_ratio", 0))
        trade_counts.append(data.get("trade_count", 0))

    # Create scatter plot
    fig = go.Figure()

    for idx, strategy in enumerate(strategies):
        color = STRATEGY_COLORS[idx % len(STRATEGY_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=[volatilities[idx]],
                y=[returns[idx]],
                mode="markers+text",
                name=strategy,
                text=[strategy],
                textposition="top center",
                marker=dict(
                    size=np.sqrt(trade_counts[idx]) * 2,  # Size by trade count
                    color=color,
                    line=dict(width=2, color="white"),
                ),
                hovertemplate=(
                    f"<b>{strategy}</b><br>"
                    "Return: %{y:.1f}%<br>"
                    "Volatility: %{x:.1f}%<br>"
                    f"Sharpe: {sharpes[idx]:.2f}<br>"
                    f"Trades: {trade_counts[idx]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Update layout
    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Volatility (Risk) %",
        yaxis_title="Total Return %",
        template="plotly_white",
        height=500,
        showlegend=False,
    )

    # Add quadrant lines
    avg_vol = np.mean(volatilities)
    avg_ret = np.mean(returns)
    fig.add_vline(x=avg_vol, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_hline(y=avg_ret, line_dash="dash", line_color="gray", opacity=0.3)

    return fig


def render_detailed_charts(results: Dict) -> None:
    """
    Render all detailed charts in a grid layout.

    Args:
        results: Comparison results dictionary
    """
    # First row: Returns distribution and Rolling Sharpe
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Returns Distribution")
        fig = render_returns_distribution(results)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Rolling Sharpe Ratio")
        fig = render_rolling_metrics(results, metric="sharpe")
        st.plotly_chart(fig, use_container_width=True)

    # Second row: Risk-Return scatter and Correlation matrix
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Risk-Return Profile")
        fig = render_risk_return_scatter(results)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Strategy Correlations")
        fig = render_correlation_matrix(results)
        st.plotly_chart(fig, use_container_width=True)

    # Third row: Rolling volatility
    st.markdown("#### Rolling Volatility")
    fig = render_rolling_metrics(results, metric="volatility")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    # Validation function
    import sys

    print("🔍 Validating charts.py...")

    all_validation_failures = []
    total_tests = 0

    # Test 1: Equity curve generation
    total_tests += 1
    try:
        test_results = {
            "equity_curves": {
                "Strategy A": {
                    "dates": [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)],
                    "values": [100 + i * 5 for i in range(10)]
                },
                "Strategy B": {
                    "dates": [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)],
                    "values": [100 + i * 3 for i in range(10)]
                }
            }
        }
        fig = render_equity_curves(test_results)
        if not isinstance(fig, go.Figure):
            all_validation_failures.append("Equity curve: Expected go.Figure object")
        if len(fig.data) != 2:
            all_validation_failures.append(f"Equity curve: Expected 2 traces, got {len(fig.data)}")
    except Exception as e:
        all_validation_failures.append(f"Equity curve test failed: {e}")

    # Test 2: Drawdown chart generation
    total_tests += 1
    try:
        test_results = {
            "drawdowns": {
                "Strategy A": {
                    "dates": [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)],
                    "values": [-i for i in range(10)]
                }
            }
        }
        fig = render_drawdown_chart(test_results)
        if not isinstance(fig, go.Figure):
            all_validation_failures.append("Drawdown chart: Expected go.Figure object")
    except Exception as e:
        all_validation_failures.append(f"Drawdown chart test failed: {e}")

    # Test 3: Risk-return scatter
    total_tests += 1
    try:
        test_results = {
            "metrics": {
                "Strategy A": {
                    "volatility": 15.5,
                    "total_return": 45.2,
                    "sharpe_ratio": 2.1,
                    "trade_count": 100
                }
            }
        }
        fig = render_risk_return_scatter(test_results)
        if not isinstance(fig, go.Figure):
            all_validation_failures.append("Risk-return scatter: Expected go.Figure object")
    except Exception as e:
        all_validation_failures.append(f"Risk-return scatter test failed: {e}")

    # Final validation result
    if all_validation_failures:
        print(
            f"❌ VALIDATION FAILED - {len(all_validation_failures)} "
            f"of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests successful")
        print("Charts component is validated and ready for integration")
        sys.exit(0)
