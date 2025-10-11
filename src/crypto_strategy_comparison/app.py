"""
Main Streamlit Application for Crypto Strategy Comparison Dashboard

This is the entry point for the Streamlit web interface that allows users to:
- Select and compare multiple trading strategies (2-10 simultaneously)
- Analyze performance across different time horizons
- View interactive charts and performance metrics
- Export comparison reports

Documentation:
- Streamlit: https://docs.streamlit.io
- Plotly: https://plotly.com/python/

Sample Input:
- Selected strategies: ["momentum_eth", "mean_reversion_btc"]
- Time horizon: "6M" (6 months)
- Risk filters: max_dd=30%, min_sharpe=1.0

Expected Output:
- Interactive dashboard with equity curves, metrics table, and detailed charts
- Exportable comparison reports (PDF/HTML/CSV)
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import streamlit as st
from loguru import logger

# Import UI components
from crypto_strategy_comparison.ui.sidebar import render_sidebar
from crypto_strategy_comparison.ui.metrics_display import render_metrics_table
from crypto_strategy_comparison.ui.charts import render_equity_curves, render_detailed_charts
from crypto_strategy_comparison.ui.export import render_export_options
from crypto_strategy_comparison.comparison_engine import ComparisonEngine
from crypto_strategy_comparison.strategy_loader import StrategyLoader
from crypto_strategy_comparison.utils import apply_custom_css, get_strategy_icon


# Configure logging
logger.add("logs/app.log", rotation="10 MB", retention="7 days")


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "selected_strategies" not in st.session_state:
        st.session_state.selected_strategies = []

    if "time_horizon" not in st.session_state:
        st.session_state.time_horizon = "6M"

    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None

    if "show_advanced" not in st.session_state:
        st.session_state.show_advanced = False

    if "last_update" not in st.session_state:
        st.session_state.last_update = None


def render_header() -> None:
    """Render the main dashboard header with title and quick controls."""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("🎯 Crypto Strategy Comparison Dashboard")
        st.markdown("*Compare multiple trading strategies across different time horizons*")

    with col2:
        if st.session_state.last_update:
            st.metric(
                "Last Updated",
                st.session_state.last_update.strftime("%H:%M:%S"),
                delta=None
            )

    with col3:
        st.metric(
            "Strategies Selected",
            len(st.session_state.selected_strategies),
            delta=None
        )


def render_quick_controls() -> None:
    """Render quick control panel for strategy and time horizon selection."""
    st.markdown("### Quick Selection")

    col1, col2, col3 = st.columns([3, 2, 1])

    with col1:
        # Strategy multi-select
        available_strategies = StrategyLoader.get_available_strategies()
        selected = st.multiselect(
            "Select Strategies (2-10)",
            options=available_strategies,
            default=st.session_state.selected_strategies or available_strategies[:2],
            format_func=lambda x: f"{get_strategy_icon(x)} {x}",
            help="Choose 2 to 10 strategies to compare side-by-side",
            key="strategy_selector"
        )

        if 2 <= len(selected) <= 10:
            st.session_state.selected_strategies = selected
        elif len(selected) > 10:
            st.warning("⚠️ Please select maximum 10 strategies")
        elif len(selected) == 1:
            st.info("ℹ️ Select at least 2 strategies to compare")

    with col2:
        # Time horizon selector
        time_options = {
            "1W": "1 Week",
            "1M": "1 Month",
            "3M": "3 Months",
            "6M": "6 Months",
            "1Y": "1 Year",
            "ALL": "All Time"
        }

        selected_horizon = st.selectbox(
            "Time Horizon",
            options=list(time_options.keys()),
            format_func=lambda x: time_options[x],
            index=3,  # Default to 6M
            key="time_horizon_selector"
        )
        st.session_state.time_horizon = selected_horizon

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Vertical spacing
        run_analysis = st.button(
            "🔄 Run Analysis",
            type="primary",
            use_container_width=True,
            help="Run comparison analysis for selected strategies"
        )

        return run_analysis


def run_comparison_analysis() -> None:
    """
    Execute the comparison analysis for selected strategies.
    Updates session state with results.
    """
    if len(st.session_state.selected_strategies) < 2:
        st.error("❌ Please select at least 2 strategies to compare")
        return

    try:
        with st.spinner("🔄 Running comparison analysis..."):
            # Initialize components
            loader = StrategyLoader()
            engine = ComparisonEngine()

            # Load strategy data
            strategies_data = loader.load_strategies(
                st.session_state.selected_strategies
            )

            # Run comparison
            results = engine.compare(
                strategies_data,
                time_horizon=st.session_state.time_horizon
            )

            # Update session state
            st.session_state.comparison_results = results
            st.session_state.last_update = datetime.now()

            logger.info(
                f"Comparison completed for {len(st.session_state.selected_strategies)} "
                f"strategies over {st.session_state.time_horizon}"
            )

            st.success("✅ Analysis complete!")

    except Exception as e:
        logger.error(f"Comparison analysis failed: {e}")
        st.error(f"❌ Analysis failed: {str(e)}")


def render_performance_overview() -> None:
    """Render the main performance overview section with equity curves."""
    if not st.session_state.comparison_results:
        st.info("👆 Select strategies and click 'Run Analysis' to see results")
        return

    st.markdown("---")
    st.markdown("## 📊 Performance Overview")

    # Render equity curves
    fig = render_equity_curves(st.session_state.comparison_results)
    st.plotly_chart(fig, use_container_width=True)

    # Render metrics table
    st.markdown("### 📈 Key Performance Metrics")
    render_metrics_table(st.session_state.comparison_results)


def render_detailed_analysis() -> None:
    """Render tabbed interface for detailed analysis."""
    if not st.session_state.comparison_results:
        return

    st.markdown("---")
    st.markdown("## 🔍 Detailed Analysis")

    tabs = st.tabs([
        "📊 Charts",
        "📉 Drawdowns",
        "📋 Trades",
        "⚙️ Parameters",
        "📤 Export"
    ])

    with tabs[0]:  # Charts tab
        render_detailed_charts(st.session_state.comparison_results)

    with tabs[1]:  # Drawdowns tab
        st.markdown("### Drawdown Analysis")
        from crypto_strategy_comparison.ui.charts import render_drawdown_chart
        fig = render_drawdown_chart(st.session_state.comparison_results)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:  # Trades tab
        st.markdown("### Trade-Level Analysis")
        from crypto_strategy_comparison.ui.metrics_display import render_trades_table
        render_trades_table(st.session_state.comparison_results)

    with tabs[3]:  # Parameters tab
        st.markdown("### Strategy Parameters")
        from crypto_strategy_comparison.ui.sidebar import render_parameter_controls
        render_parameter_controls(st.session_state.selected_strategies)

    with tabs[4]:  # Export tab
        render_export_options(st.session_state.comparison_results)


def render_insights() -> None:
    """Render AI-generated insights section."""
    if not st.session_state.comparison_results:
        return

    st.markdown("---")
    st.markdown("## 💡 Insights & Recommendations")

    # Generate insights based on comparison results
    insights = generate_insights(st.session_state.comparison_results)

    for insight in insights:
        st.markdown(f"• {insight}")


def generate_insights(results: Dict) -> List[str]:
    """
    Generate insights based on comparison results.

    Args:
        results: Comparison results dictionary

    Returns:
        List of insight strings
    """
    insights = []

    # This is a placeholder - in production, this would use more sophisticated analysis
    if results and "metrics" in results:
        metrics = results["metrics"]

        # Find best performing strategy
        best_return_strategy = max(metrics.items(), key=lambda x: x[1].get("total_return", 0))
        insights.append(
            f"**{best_return_strategy[0]}** shows the highest absolute returns "
            f"at {best_return_strategy[1].get('total_return', 0):.1f}%"
        )

        # Find best risk-adjusted strategy
        best_sharpe_strategy = max(metrics.items(), key=lambda x: x[1].get("sharpe_ratio", 0))
        insights.append(
            f"**{best_sharpe_strategy[0]}** has the best risk-adjusted returns "
            f"(Sharpe: {best_sharpe_strategy[1].get('sharpe_ratio', 0):.2f})"
        )

        # Volatility comparison
        volatilities = {k: v.get("volatility", 0) for k, v in metrics.items()}
        if len(volatilities) >= 2:
            highest_vol = max(volatilities.items(), key=lambda x: x[1])
            lowest_vol = min(volatilities.items(), key=lambda x: x[1])
            vol_diff = highest_vol[1] - lowest_vol[1]
            insights.append(
                f"**{highest_vol[0]}** shows {vol_diff:.1f}pp more volatility "
                f"than **{lowest_vol[0]}**"
            )

    return insights


def main() -> None:
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Crypto Strategy Comparison",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS
    apply_custom_css()

    # Initialize session state
    initialize_session_state()

    # Render sidebar
    render_sidebar()

    # Main content area
    render_header()

    # Quick controls
    run_analysis = render_quick_controls()

    # Run analysis if button clicked
    if run_analysis:
        run_comparison_analysis()

    # Performance overview
    render_performance_overview()

    # Detailed analysis tabs
    render_detailed_analysis()

    # Insights
    render_insights()

    # Footer
    st.markdown("---")
    st.markdown(
        "*Built with Streamlit and Plotly | "
        f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_update else 'Never'}*"
    )


if __name__ == "__main__":
    main()
