"""
Main Streamlit Dashboard Application for Crypto Trading Strategy Comparison

This module provides the main entry point for the Streamlit web dashboard,
offering an interactive interface for comparing and analyzing crypto trading strategies.

**Purpose**: Multi-page Streamlit application for strategy backtesting, comparison,
and performance analysis with real-time interactive visualizations.

**Features**:
- Strategy comparison across multiple metrics
- Interactive Plotly charts (equity curves, drawdowns, returns)
- Time horizon analysis
- Performance metrics tables (sortable and filterable)
- Export functionality (HTML, PDF, CSV)
- Session state management for caching

**Third-party packages**:
- streamlit: https://docs.streamlit.io/
- plotly: https://plotly.com/python/
- pandas: https://pandas.pydata.org/docs/
- loguru: https://loguru.readthedocs.io/en/stable/

**Sample Input**:
Run with: `streamlit run src/crypto_trader/web/app.py`

**Expected Output**:
Interactive web dashboard running on localhost:8501 with:
- Strategy selection sidebar
- Multi-strategy comparison charts
- Performance metrics tables
- Export buttons
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger
from plotly.subplots import make_subplots

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from crypto_trader.analysis.comparison import StrategyComparison
from crypto_trader.analysis.metrics import MetricsCalculator
from crypto_trader.analysis.reporting import ReportGenerator
from crypto_trader.core.types import BacktestResult, Timeframe
from crypto_trader.strategies.registry import get_registry, list_strategies

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.3rem solid #1f77b4;
    }
    .metric-positive {
        border-left-color: #28a745 !important;
    }
    .metric-negative {
        border-left-color: #dc3545 !important;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
    }
    h2 {
        color: #555;
        margin-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables for caching."""
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = {}
    if "selected_strategies" not in st.session_state:
        st.session_state.selected_strategies = []
    if "comparison_data" not in st.session_state:
        st.session_state.comparison_data = None
    if "time_horizon" not in st.session_state:
        st.session_state.time_horizon = "1 Year"


def create_sidebar():
    """Create sidebar with strategy selection and configuration."""
    with st.sidebar:
        st.title("📊 Strategy Dashboard")
        st.markdown("---")

        # Time horizon selector
        st.subheader("⏱️ Time Horizon")
        time_horizons = ["1 Month", "3 Months", "6 Months", "1 Year", "All Time"]
        st.session_state.time_horizon = st.selectbox(
            "Select Period", time_horizons, index=3
        )

        st.markdown("---")

        # Strategy selection
        st.subheader("🎯 Available Strategies")

        # Load strategies from registry
        registry = get_registry()
        available_strategies = list_strategies()

        if not available_strategies:
            st.warning("No strategies registered. Please load strategies first.")
            # Show mock strategies for demo
            available_strategies = {
                "MA Crossover": {"description": "Moving Average Crossover"},
                "RSI Mean Reversion": {"description": "RSI-based mean reversion"},
                "MACD Momentum": {"description": "MACD momentum strategy"},
                "Bollinger Breakout": {"description": "Bollinger Bands breakout"},
                "Triple EMA": {"description": "Triple EMA crossover"},
            }

        # Multi-select for strategies
        strategy_names = list(available_strategies.keys())
        selected = st.multiselect(
            "Select Strategies (2-10)",
            strategy_names,
            default=strategy_names[:2] if len(strategy_names) >= 2 else strategy_names,
            help="Select 2-10 strategies to compare",
        )

        st.session_state.selected_strategies = selected

        # Validate selection
        if len(selected) < 2:
            st.error("Please select at least 2 strategies")
        elif len(selected) > 10:
            st.error("Maximum 10 strategies allowed")

        st.markdown("---")

        # Symbol and timeframe filters
        st.subheader("⚙️ Filters")
        symbols = st.multiselect(
            "Symbols", ["BTCUSDT", "ETHUSDT", "BNBUSDT"], default=["BTCUSDT"]
        )

        timeframes = st.multiselect(
            "Timeframes",
            ["15m", "1h", "4h", "1d"],
            default=["4h"],
        )

        st.markdown("---")

        # Export section
        st.subheader("💾 Export")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export HTML"):
                export_html_report()
        with col2:
            if st.button("Export CSV"):
                export_csv_data()


def create_metrics_overview(results: List[BacktestResult]):
    """Create overview metrics cards at the top of the page."""
    if not results:
        return

    comparison = StrategyComparison()

    # Find best performers
    best_sharpe = comparison.best_performer(results, "sharpe_ratio")
    best_return = comparison.best_performer(results, "total_return")
    best_dd = comparison.best_performer(results, "max_drawdown")

    st.subheader("🏆 Best Performers")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if best_sharpe:
            st.metric(
                "Best Sharpe Ratio",
                f"{best_sharpe.metrics.sharpe_ratio:.2f}",
                delta=best_sharpe.strategy_name,
            )

    with col2:
        if best_return:
            st.metric(
                "Best Total Return",
                f"{best_return.metrics.total_return:.2%}",
                delta=best_return.strategy_name,
            )

    with col3:
        if best_dd:
            st.metric(
                "Lowest Drawdown",
                f"{best_dd.metrics.max_drawdown:.2%}",
                delta=best_dd.strategy_name,
            )

    with col4:
        avg_return = sum(r.metrics.total_return for r in results) / len(results)
        st.metric("Average Return", f"{avg_return:.2%}", delta=f"{len(results)} strategies")


def create_equity_curves_chart(results: List[BacktestResult]) -> go.Figure:
    """Create overlaid equity curves for multiple strategies."""
    fig = go.Figure()

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, result in enumerate(results):
        if len(result.equity_curve) == 0:
            continue

        timestamps = [ts for ts, _ in result.equity_curve]
        equity = [eq for _, eq in result.equity_curve]

        # Normalize to percentage return
        initial = equity[0]
        normalized = [(e - initial) / initial * 100 for e in equity]

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=normalized,
                mode="lines",
                name=result.strategy_name,
                line=dict(width=2, color=colors[idx % len(colors)]),
                hovertemplate=f"{result.strategy_name}<br>Return: %{{y:.2f}}%<br>Date: %{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Strategy Equity Curves (Normalized Returns)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def create_drawdown_comparison_chart(results: List[BacktestResult]) -> go.Figure:
    """Create drawdown comparison chart."""
    fig = go.Figure()

    colors = [
        "#dc3545",
        "#ff6b6b",
        "#c92a2a",
        "#e03131",
        "#f03e3e",
        "#fa5252",
        "#ff8787",
        "#ffc9c9",
    ]

    for idx, result in enumerate(results):
        if len(result.equity_curve) == 0:
            continue

        df = pd.DataFrame(result.equity_curve, columns=["timestamp", "equity"])
        df["running_max"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["running_max"]) / df["running_max"] * 100

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["drawdown"],
                mode="lines",
                name=result.strategy_name,
                line=dict(width=2, color=colors[idx % len(colors)]),
                fill="tozeroy",
                fillcolor=colors[idx % len(colors)] + "20",
                hovertemplate=f"{result.strategy_name}<br>Drawdown: %{{y:.2f}}%<br>Date: %{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_returns_distribution_chart(results: List[BacktestResult]) -> go.Figure:
    """Create returns distribution violin plot."""
    fig = go.Figure()

    for result in results:
        if len(result.equity_curve) < 2:
            continue

        df = pd.DataFrame(result.equity_curve, columns=["timestamp", "equity"])
        df["returns"] = df["equity"].pct_change() * 100
        returns = df["returns"].dropna()

        fig.add_trace(
            go.Violin(
                y=returns,
                name=result.strategy_name,
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig.update_layout(
        title="Returns Distribution",
        yaxis_title="Returns (%)",
        template="plotly_white",
        height=400,
        showlegend=True,
    )

    return fig


def create_rolling_sharpe_chart(results: List[BacktestResult]) -> go.Figure:
    """Create rolling Sharpe ratio chart."""
    fig = go.Figure()

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    for idx, result in enumerate(results):
        if len(result.equity_curve) < 30:
            continue

        df = pd.DataFrame(result.equity_curve, columns=["timestamp", "equity"])
        df["returns"] = df["equity"].pct_change()

        # Calculate rolling Sharpe (30-period window)
        window = 30
        rolling_sharpe = (
            df["returns"].rolling(window).mean()
            / df["returns"].rolling(window).std()
            * (252 ** 0.5)  # Annualized
        )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=rolling_sharpe,
                mode="lines",
                name=result.strategy_name,
                line=dict(width=2, color=colors[idx % len(colors)]),
                hovertemplate=f"{result.strategy_name}<br>Sharpe: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Rolling Sharpe Ratio (30-period)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Add reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_hline(y=2.0, line_dash="dash", line_color="darkgreen", opacity=0.5)

    return fig


def create_correlation_heatmap(results: List[BacktestResult]) -> go.Figure:
    """Create correlation heatmap between strategies."""
    comparison = StrategyComparison()
    corr_matrix = comparison.correlation_matrix(results)

    if corr_matrix.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for correlation analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Strategy Return Correlation Matrix",
        template="plotly_white",
        height=500,
        xaxis=dict(tickangle=-45),
    )

    return fig


def create_risk_return_scatter(results: List[BacktestResult]) -> go.Figure:
    """Create risk-return scatter plot."""
    fig = go.Figure()

    returns = []
    volatilities = []
    names = []
    sharpes = []

    for result in results:
        if len(result.equity_curve) < 2:
            continue

        df = pd.DataFrame(result.equity_curve, columns=["timestamp", "equity"])
        df["returns"] = df["equity"].pct_change()
        period_returns = df["returns"].dropna()

        annual_return = result.metrics.total_return * 100  # Convert to percentage
        annual_vol = period_returns.std() * (252 ** 0.5) * 100  # Annualized volatility

        returns.append(annual_return)
        volatilities.append(annual_vol)
        names.append(result.strategy_name)
        sharpes.append(result.metrics.sharpe_ratio)

    # Color by Sharpe ratio
    fig.add_trace(
        go.Scatter(
            x=volatilities,
            y=returns,
            mode="markers+text",
            text=names,
            textposition="top center",
            marker=dict(
                size=15,
                color=sharpes,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
                line=dict(width=1, color="white"),
            ),
            hovertemplate="<b>%{text}</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Volatility (Annualized %)",
        yaxis_title="Return (Annualized %)",
        template="plotly_white",
        height=500,
        showlegend=False,
    )

    return fig


def create_performance_table(results: List[BacktestResult]) -> pd.DataFrame:
    """Create comprehensive performance metrics table."""
    comparison = StrategyComparison()
    df = comparison.compare_strategies(results)

    # Select and reorder columns
    columns = [
        "strategy",
        "symbol",
        "timeframe",
        "total_return",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
        "total_trades",
        "expectancy",
        "final_capital",
        "quality",
    ]

    # Filter to available columns
    available_cols = [col for col in columns if col in df.columns]
    df_display = df[available_cols].copy()

    # Format numeric columns
    format_dict = {
        "total_return": "{:.2%}",
        "sharpe_ratio": "{:.2f}",
        "sortino_ratio": "{:.2f}",
        "max_drawdown": "{:.2%}",
        "calmar_ratio": "{:.2f}",
        "win_rate": "{:.2%}",
        "profit_factor": "{:.2f}",
        "expectancy": "${:.2f}",
        "final_capital": "${:,.2f}",
    }

    for col, fmt in format_dict.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: fmt.format(x))

    # Rename columns for display
    rename_dict = {
        "strategy": "Strategy",
        "symbol": "Symbol",
        "timeframe": "Timeframe",
        "total_return": "Total Return",
        "sharpe_ratio": "Sharpe",
        "sortino_ratio": "Sortino",
        "max_drawdown": "Max DD",
        "calmar_ratio": "Calmar",
        "win_rate": "Win Rate",
        "profit_factor": "Profit Factor",
        "total_trades": "Trades",
        "expectancy": "Expectancy",
        "final_capital": "Final Capital",
        "quality": "Quality",
    }

    df_display = df_display.rename(columns=rename_dict)

    return df_display


def export_html_report():
    """Export comprehensive HTML report."""
    if not st.session_state.backtest_results:
        st.warning("No backtest results to export")
        return

    with st.spinner("Generating HTML report..."):
        reporter = ReportGenerator()
        output_dir = Path("exports")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for strategy_name, result in st.session_state.backtest_results.items():
            output_path = output_dir / f"report_{strategy_name}_{timestamp}.html"
            reporter.generate_html_report(result, str(output_path))

        st.success(f"Reports exported to {output_dir}")


def export_csv_data():
    """Export performance data to CSV."""
    if not st.session_state.comparison_data:
        st.warning("No comparison data to export")
        return

    with st.spinner("Exporting CSV..."):
        output_dir = Path("exports")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"comparison_{timestamp}.csv"

        st.session_state.comparison_data.to_csv(output_path, index=False)
        st.success(f"Data exported to {output_path}")


def load_sample_data() -> List[BacktestResult]:
    """
    Load sample backtest results for demonstration.
    In production, this would load from database or run actual backtests.
    """
    from crypto_trader.core.types import (
        OrderSide,
        OrderType,
        PerformanceMetrics,
        Trade,
    )

    base_time = datetime.now() - timedelta(days=365)
    end_time = datetime.now()

    # Generate sample data for demonstration
    sample_results = []

    strategies_config = [
        {
            "name": "MA Crossover",
            "return": 0.35,
            "sharpe": 2.5,
            "max_dd": 0.12,
            "win_rate": 0.68,
        },
        {
            "name": "RSI Mean Reversion",
            "return": 0.22,
            "sharpe": 1.8,
            "max_dd": 0.18,
            "win_rate": 0.55,
        },
        {
            "name": "MACD Momentum",
            "return": 0.28,
            "sharpe": 2.1,
            "max_dd": 0.15,
            "win_rate": 0.62,
        },
    ]

    for config in strategies_config:
        # Generate equity curve
        days = 365
        equity_curve = []
        initial_capital = 10000.0
        current_equity = initial_capital

        for day in range(days):
            timestamp = base_time + timedelta(days=day)
            # Simulate equity growth with volatility
            daily_return = config["return"] / days + (
                pd.np.random.randn() * 0.02
            )  # 2% daily volatility
            current_equity *= 1 + daily_return
            equity_curve.append((timestamp, current_equity))

        final_capital = current_equity

        result = BacktestResult(
            strategy_name=config["name"],
            symbol="BTCUSDT",
            timeframe=Timeframe.HOUR_4,
            start_date=base_time,
            end_date=end_time,
            initial_capital=initial_capital,
            metrics=PerformanceMetrics(
                total_return=config["return"],
                sharpe_ratio=config["sharpe"],
                sortino_ratio=config["sharpe"] * 1.2,
                max_drawdown=config["max_dd"],
                calmar_ratio=config["return"] / config["max_dd"],
                win_rate=config["win_rate"],
                profit_factor=1.5 + config["sharpe"] * 0.3,
                total_trades=45,
                winning_trades=int(45 * config["win_rate"]),
                losing_trades=int(45 * (1 - config["win_rate"])),
                avg_win=85.5,
                avg_loss=-42.3,
                expectancy=50.0,
                total_fees=450.0,
                final_capital=final_capital,
            ),
            trades=[],
            equity_curve=equity_curve,
        )

        sample_results.append(result)

    return sample_results


def main():
    """Main application entry point."""
    init_session_state()
    create_sidebar()

    # Main content
    st.title("📈 Crypto Trading Strategy Dashboard")
    st.markdown("### Compare and analyze multiple trading strategies")

    # Check if strategies are selected
    if len(st.session_state.selected_strategies) < 2:
        st.info(
            "👈 Please select at least 2 strategies from the sidebar to start comparison"
        )
        return

    # Load data (in production, this would be from database or live backtests)
    with st.spinner("Loading backtest results..."):
        results = load_sample_data()

        # Filter by selected strategies
        results = [
            r
            for r in results
            if r.strategy_name in st.session_state.selected_strategies
        ]

        if not results:
            st.warning("No results found for selected strategies")
            return

    # Store in session state
    for result in results:
        st.session_state.backtest_results[result.strategy_name] = result

    # Metrics overview
    create_metrics_overview(results)

    st.markdown("---")

    # Main charts in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📈 Equity Curves",
            "📉 Drawdowns & Risk",
            "📊 Distribution",
            "🔗 Correlation",
            "📋 Performance Table",
        ]
    )

    with tab1:
        st.plotly_chart(
            create_equity_curves_chart(results), use_container_width=True
        )
        st.plotly_chart(
            create_rolling_sharpe_chart(results), use_container_width=True
        )

    with tab2:
        st.plotly_chart(
            create_drawdown_comparison_chart(results), use_container_width=True
        )
        st.plotly_chart(
            create_risk_return_scatter(results), use_container_width=True
        )

    with tab3:
        st.plotly_chart(
            create_returns_distribution_chart(results), use_container_width=True
        )

    with tab4:
        st.plotly_chart(
            create_correlation_heatmap(results), use_container_width=True
        )

        # Statistical significance tests
        if len(results) >= 2:
            st.subheader("📊 Statistical Significance Tests")
            comparison = StrategyComparison()

            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    sig_test = comparison.statistical_significance(
                        results[i], results[j]
                    )

                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col1:
                        st.write(f"**{sig_test['strategy1']}** vs **{sig_test['strategy2']}**")
                    with col2:
                        st.write(f"p-value: {sig_test['p_value']:.4f}")
                    with col3:
                        if sig_test["significant"]:
                            st.success("✅ Significant difference")
                        else:
                            st.info("ℹ️ No significant difference")

    with tab5:
        st.subheader("Performance Metrics Comparison")

        df_display = create_performance_table(results)
        st.session_state.comparison_data = df_display

        # Make table sortable
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400,
        )

        # Download button
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="strategy_comparison.csv",
            mime="text/csv",
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Crypto Trading Strategy Dashboard | Built with Streamlit & Plotly</p>
            <p>Data is for demonstration purposes only. Past performance does not guarantee future results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
