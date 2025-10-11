"""
Strategy Comparison Page - Multi-strategy analysis and comparison

This module provides advanced multi-strategy comparison with statistical analysis,
correlation matrices, and comprehensive performance rankings.

**Purpose**: Multi-strategy comparison interface with advanced analytics including
correlation analysis, statistical significance tests, and side-by-side comparisons.

**Features**:
- Multi-strategy selector (2-10 strategies)
- Time horizon analysis
- Side-by-side comparison charts
- Correlation matrix visualization
- Statistical significance tests
- Performance rankings
- Export comparison reports

**Third-party packages**:
- streamlit: https://docs.streamlit.io/
- plotly: https://plotly.com/python/
- pandas: https://pandas.pydata.org/docs/
- scipy: https://docs.scipy.org/doc/scipy/

**Sample Input**:
Run with: `streamlit run src/crypto_trader/web/app.py` (accessed via page navigation)

**Expected Output**:
Comprehensive multi-strategy comparison interface with statistical analysis
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from crypto_trader.analysis.comparison import StrategyComparison
from crypto_trader.core.types import BacktestResult, PerformanceMetrics, Timeframe
from crypto_trader.strategies.registry import list_strategies

st.set_page_config(
    page_title="Strategy Comparison",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Multi-Strategy Comparison")
st.markdown("### Advanced comparison and statistical analysis of multiple strategies")


def load_comparison_data(strategies: list[str]) -> list[BacktestResult]:
    """
    Load backtest results for selected strategies.
    In production, this would query from database or session state.
    """
    from crypto_trader.core.types import OrderSide, OrderType, Trade

    base_time = datetime.now() - timedelta(days=365)
    end_time = datetime.now()

    # Generate sample data for selected strategies
    results = []

    # Pre-defined performance profiles for different strategies
    strategy_profiles = {
        "MA Crossover": {
            "return": 0.35,
            "sharpe": 2.5,
            "max_dd": 0.12,
            "win_rate": 0.68,
            "volatility": 0.02,
        },
        "RSI Mean Reversion": {
            "return": 0.22,
            "sharpe": 1.8,
            "max_dd": 0.18,
            "win_rate": 0.55,
            "volatility": 0.025,
        },
        "MACD Momentum": {
            "return": 0.28,
            "sharpe": 2.1,
            "max_dd": 0.15,
            "win_rate": 0.62,
            "volatility": 0.022,
        },
        "Bollinger Breakout": {
            "return": 0.18,
            "sharpe": 1.5,
            "max_dd": 0.22,
            "win_rate": 0.48,
            "volatility": 0.028,
        },
        "Triple EMA": {
            "return": 0.25,
            "sharpe": 1.9,
            "max_dd": 0.16,
            "win_rate": 0.58,
            "volatility": 0.023,
        },
        "SMA Crossover": {
            "return": 0.20,
            "sharpe": 1.7,
            "max_dd": 0.19,
            "win_rate": 0.52,
            "volatility": 0.024,
        },
    }

    for strategy_name in strategies:
        if strategy_name not in strategy_profiles:
            continue

        profile = strategy_profiles[strategy_name]

        # Generate equity curve
        days = 365
        equity_curve = []
        initial_capital = 10000.0
        current_equity = initial_capital

        for day in range(days):
            timestamp = base_time + timedelta(days=day)
            # Simulate returns with strategy-specific characteristics
            daily_return = profile["return"] / days + (
                np.random.randn() * profile["volatility"]
            )
            current_equity *= 1 + daily_return
            equity_curve.append((timestamp, current_equity))

        final_capital = current_equity

        result = BacktestResult(
            strategy_name=strategy_name,
            symbol="BTCUSDT",
            timeframe=Timeframe.HOUR_4,
            start_date=base_time,
            end_date=end_time,
            initial_capital=initial_capital,
            metrics=PerformanceMetrics(
                total_return=profile["return"],
                sharpe_ratio=profile["sharpe"],
                sortino_ratio=profile["sharpe"] * 1.2,
                max_drawdown=profile["max_dd"],
                calmar_ratio=profile["return"] / profile["max_dd"],
                win_rate=profile["win_rate"],
                profit_factor=1.5 + profile["sharpe"] * 0.3,
                total_trades=int(50 + np.random.randint(-10, 10)),
                winning_trades=int(50 * profile["win_rate"]),
                losing_trades=int(50 * (1 - profile["win_rate"])),
                avg_win=85.5,
                avg_loss=-42.3,
                expectancy=50.0 + np.random.randn() * 10,
                total_fees=450.0,
                final_capital=final_capital,
            ),
            trades=[],
            equity_curve=equity_curve,
        )

        results.append(result)

    return results


def create_comparison_metrics_table(results: list[BacktestResult]) -> pd.DataFrame:
    """Create detailed metrics comparison table."""
    comparison = StrategyComparison()
    df = comparison.compare_strategies(results)

    # Rank by Sharpe ratio
    df_ranked = comparison.rank_strategies(results, metric="sharpe_ratio")

    return df_ranked


def create_side_by_side_comparison(results: list[BacktestResult]) -> go.Figure:
    """Create side-by-side bar chart comparison of key metrics."""
    metrics_to_compare = {
        "Total Return": "total_return",
        "Sharpe Ratio": "sharpe_ratio",
        "Win Rate": "win_rate",
        "Profit Factor": "profit_factor",
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(metrics_to_compare.keys()),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    colors = px.colors.qualitative.Set2

    for idx, (title, metric) in enumerate(metrics_to_compare.items()):
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        strategy_names = [r.strategy_name for r in results]
        values = [getattr(r.metrics, metric) for r in results]

        # Convert to percentage if needed
        if metric in ["total_return", "win_rate"]:
            values = [v * 100 for v in values]

        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=values,
                name=title,
                marker_color=colors[idx % len(colors)],
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Update axes
        y_suffix = "%" if metric in ["total_return", "win_rate"] else ""
        fig.update_yaxes(title_text=title + (" (%)" if y_suffix else ""), row=row, col=col)

    fig.update_layout(
        title_text="Side-by-Side Performance Comparison",
        height=600,
        showlegend=False,
        template="plotly_white",
    )

    return fig


def create_ranking_chart(results: list[BacktestResult], metric: str) -> go.Figure:
    """Create ranking bar chart for a specific metric."""
    comparison = StrategyComparison()
    df_ranked = comparison.rank_strategies(results, metric=metric)

    metric_display = metric.replace("_", " ").title()

    # Get values
    strategies = df_ranked["strategy"].tolist()
    values = df_ranked[metric].tolist()

    # Convert to percentage if needed
    if metric in ["total_return", "max_drawdown", "win_rate"]:
        values = [v * 100 for v in values]
        y_title = f"{metric_display} (%)"
    else:
        y_title = metric_display

    # Color by rank (green for best, red for worst)
    colors = [
        f"rgb({int(255 * (1 - i / len(strategies)))}, {int(200 * i / len(strategies))}, 0)"
        for i in range(len(strategies))
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=strategies,
                y=values,
                marker_color=colors,
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=f"Strategy Ranking by {metric_display}",
        xaxis_title="Strategy",
        yaxis_title=y_title,
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    return fig


def create_pairwise_comparison_matrix(results: list[BacktestResult]) -> pd.DataFrame:
    """Create pairwise comparison matrix with statistical significance."""
    comparison = StrategyComparison()

    n = len(results)
    matrix_data = []

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append("—")
            elif i < j:
                sig_test = comparison.statistical_significance(results[i], results[j])
                if sig_test["significant"]:
                    # Determine which is better
                    if sig_test["mean_return1"] > sig_test["mean_return2"]:
                        row.append(f"✅ {results[i].strategy_name}")
                    else:
                        row.append(f"✅ {results[j].strategy_name}")
                else:
                    row.append("≈")
            else:
                # Mirror the upper triangle
                row.append("")

        matrix_data.append(row)

    df = pd.DataFrame(
        matrix_data,
        index=[r.strategy_name for r in results],
        columns=[r.strategy_name for r in results],
    )

    return df


def create_performance_radar(results: list[BacktestResult]) -> go.Figure:
    """Create radar chart comparing normalized performance metrics."""
    fig = go.Figure()

    # Metrics to compare (normalized to 0-1 scale)
    metrics = ["sharpe_ratio", "total_return", "win_rate", "profit_factor", "calmar_ratio"]
    metric_labels = ["Sharpe", "Return", "Win Rate", "Profit Factor", "Calmar"]

    colors = px.colors.qualitative.Set2

    for idx, result in enumerate(results):
        values = []
        for metric in metrics:
            value = getattr(result.metrics, metric)

            # Normalize to 0-1 scale based on reasonable ranges
            if metric == "sharpe_ratio":
                normalized = min(value / 3.0, 1.0)  # Sharpe of 3 is excellent
            elif metric == "total_return":
                normalized = min(value / 0.5, 1.0)  # 50% return is excellent
            elif metric == "win_rate":
                normalized = value  # Already 0-1
            elif metric == "profit_factor":
                normalized = min((value - 1) / 2.0, 1.0)  # PF of 3 is excellent
            elif metric == "calmar_ratio":
                normalized = min(value / 5.0, 1.0)  # Calmar of 5 is excellent
            else:
                normalized = value

            values.append(max(0, min(1, normalized)))  # Clamp to 0-1

        # Close the radar chart
        values.append(values[0])
        labels = metric_labels + [metric_labels[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels,
                fill="toself",
                name=result.strategy_name,
                line_color=colors[idx % len(colors)],
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Normalized Performance Radar",
        template="plotly_white",
        height=500,
    )

    return fig


def main():
    """Main comparison page logic."""
    # Strategy selection
    st.subheader("🎯 Select Strategies to Compare")

    strategies = list_strategies()
    if not strategies:
        strategies = {
            "MA Crossover": {"description": "Moving Average Crossover"},
            "RSI Mean Reversion": {"description": "RSI-based mean reversion"},
            "MACD Momentum": {"description": "MACD momentum strategy"},
            "Bollinger Breakout": {"description": "Bollinger Bands breakout"},
            "Triple EMA": {"description": "Triple EMA crossover"},
            "SMA Crossover": {"description": "Simple MA crossover"},
        }

    strategy_names = list(strategies.keys())

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_strategies = st.multiselect(
            "Select 2-10 strategies",
            strategy_names,
            default=strategy_names[:3] if len(strategy_names) >= 3 else strategy_names,
            help="Select between 2 and 10 strategies for comparison",
        )

    with col2:
        comparison_metric = st.selectbox(
            "Primary Metric",
            ["sharpe_ratio", "total_return", "max_drawdown", "win_rate", "profit_factor"],
            format_func=lambda x: x.replace("_", " ").title(),
        )

    # Validation
    if len(selected_strategies) < 2:
        st.warning("⚠️ Please select at least 2 strategies to compare")
        return
    elif len(selected_strategies) > 10:
        st.error("❌ Maximum 10 strategies allowed")
        return

    st.markdown("---")

    # Load data
    with st.spinner("Loading backtest results..."):
        results = load_comparison_data(selected_strategies)

    if not results:
        st.error("No results found for selected strategies")
        return

    # Overview metrics
    st.subheader("📊 Quick Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        comparison = StrategyComparison()
        best_sharpe = comparison.best_performer(results, "sharpe_ratio")
        st.metric(
            "Best Sharpe",
            f"{best_sharpe.metrics.sharpe_ratio:.2f}",
            delta=best_sharpe.strategy_name,
        )

    with col2:
        best_return = comparison.best_performer(results, "total_return")
        st.metric(
            "Best Return",
            f"{best_return.metrics.total_return:.2%}",
            delta=best_return.strategy_name,
        )

    with col3:
        best_dd = comparison.best_performer(results, "max_drawdown")
        st.metric(
            "Lowest DD",
            f"{best_dd.metrics.max_drawdown:.2%}",
            delta=best_dd.strategy_name,
        )

    with col4:
        avg_return = sum(r.metrics.total_return for r in results) / len(results)
        st.metric("Avg Return", f"{avg_return:.2%}", delta=f"{len(results)} strategies")

    st.markdown("---")

    # Main comparison tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Rankings", "📈 Charts", "🔗 Correlation", "📋 Statistics", "🎯 Details"]
    )

    with tab1:
        st.subheader("Strategy Rankings")

        # Ranking by selected metric
        st.plotly_chart(
            create_ranking_chart(results, comparison_metric), use_container_width=True
        )

        # Side-by-side comparison
        st.plotly_chart(
            create_side_by_side_comparison(results), use_container_width=True
        )

    with tab2:
        st.subheader("Performance Visualizations")

        # Performance radar
        st.plotly_chart(create_performance_radar(results), use_container_width=True)

        # Individual metric selection for detailed view
        col1, col2 = st.columns(2)

        with col1:
            metric1 = st.selectbox(
                "First Metric",
                ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="metric1",
            )

        with col2:
            metric2 = st.selectbox(
                "Second Metric",
                ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="metric2",
            )

        # Create scatter plot
        fig = go.Figure()

        x_values = [getattr(r.metrics, metric1) for r in results]
        y_values = [getattr(r.metrics, metric2) for r in results]
        names = [r.strategy_name for r in results]

        # Convert to percentage if needed
        if metric1 in ["total_return", "max_drawdown", "win_rate"]:
            x_values = [v * 100 for v in x_values]
            x_title = f"{metric1.replace('_', ' ').title()} (%)"
        else:
            x_title = metric1.replace("_", " ").title()

        if metric2 in ["total_return", "max_drawdown", "win_rate"]:
            y_values = [v * 100 for v in y_values]
            y_title = f"{metric2.replace('_', ' ').title()} (%)"
        else:
            y_title = metric2.replace("_", " ").title()

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                text=names,
                textposition="top center",
                marker=dict(size=15, color=px.colors.qualitative.Set2),
            )
        )

        fig.update_layout(
            title=f"{metric1.replace('_', ' ').title()} vs {metric2.replace('_', ' ').title()}",
            xaxis_title=x_title,
            yaxis_title=y_title,
            template="plotly_white",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Correlation Analysis")

        # Correlation matrix heatmap
        corr_matrix = comparison.correlation_matrix(results)

        if not corr_matrix.empty:
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

            st.plotly_chart(fig, use_container_width=True)

            # Interpretation
            st.info(
                """
                **Correlation Interpretation:**
                - Values close to 1: Strategies move together (highly correlated)
                - Values close to 0: Strategies are independent
                - Values close to -1: Strategies move opposite (negatively correlated)

                Lower correlation between strategies is generally better for portfolio diversification.
                """
            )
        else:
            st.warning("Insufficient data for correlation analysis")

    with tab4:
        st.subheader("Statistical Significance Tests")

        st.markdown(
            """
            Statistical significance tests help determine if performance differences between
            strategies are likely due to actual strategy superiority or just random chance.
            """
        )

        # Pairwise comparisons
        if len(results) >= 2:
            st.markdown("### Pairwise Comparisons")

            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    sig_test = comparison.statistical_significance(results[i], results[j])

                    with st.expander(
                        f"{results[i].strategy_name} vs {results[j].strategy_name}"
                    ):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                results[i].strategy_name,
                                f"{sig_test['mean_return1']:.4%}",
                                "Mean Return",
                            )

                        with col2:
                            st.metric(
                                results[j].strategy_name,
                                f"{sig_test['mean_return2']:.4%}",
                                "Mean Return",
                            )

                        with col3:
                            if sig_test["significant"]:
                                st.success(f"✅ Significant (p={sig_test['p_value']:.4f})")
                            else:
                                st.info(f"≈ Not significant (p={sig_test['p_value']:.4f})")

                        st.markdown(f"**T-statistic:** {sig_test['t_statistic']:.4f}")
                        st.markdown(f"**P-value:** {sig_test['p_value']:.4f}")
                        st.markdown(sig_test["message"])

        # Summary statistics
        st.markdown("### Summary Statistics")
        summary = comparison.multi_strategy_summary(results)

        if summary:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Strategies", summary["total_strategies"])
                st.metric("Profitable Strategies", summary["profitable_strategies"])

            with col2:
                st.metric("Profitable Rate", f"{summary['profitable_rate']:.2%}")

            with col3:
                st.write("**Best Performers:**")
                for metric, strategy in summary["best_performers"].items():
                    st.write(f"- {metric.replace('_', ' ').title()}: {strategy}")

    with tab5:
        st.subheader("Detailed Metrics Table")

        # Full comparison table
        df_comparison = create_comparison_metrics_table(results)

        st.dataframe(
            df_comparison,
            use_container_width=True,
            height=400,
        )

        # Download button
        csv = df_comparison.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Data",
            data=csv,
            file_name=f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Filter strategies by criteria
        st.markdown("---")
        st.subheader("🔍 Filter Strategies")

        col1, col2, col3 = st.columns(3)

        with col1:
            min_sharpe = st.number_input(
                "Min Sharpe Ratio",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
            )

        with col2:
            max_drawdown_filter = st.number_input(
                "Max Drawdown (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=5.0,
            )

        with col3:
            min_trades = st.number_input(
                "Min Trades", min_value=0, max_value=1000, value=30, step=10
            )

        filtered = comparison.filter_strategies(
            results,
            min_sharpe=min_sharpe,
            max_drawdown=max_drawdown_filter / 100,
            min_trades=min_trades,
        )

        st.metric(
            "Strategies Passing Filter",
            f"{len(filtered)} / {len(results)}",
            delta=f"{len(filtered) / len(results):.0%}",
        )

        if filtered:
            st.markdown("**Filtered Strategies:**")
            for result in filtered:
                st.write(
                    f"- **{result.strategy_name}**: Sharpe {result.metrics.sharpe_ratio:.2f}, "
                    f"Return {result.metrics.total_return:.2%}, "
                    f"DD {result.metrics.max_drawdown:.2%}"
                )


if __name__ == "__main__":
    main()
