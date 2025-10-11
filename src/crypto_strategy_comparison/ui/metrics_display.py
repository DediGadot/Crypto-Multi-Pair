"""
Metrics Display Component

Renders performance metrics tables:
- Summary metrics table (sortable, filterable)
- Detailed metrics (expandable rows)
- Trade-level table with pagination

Documentation:
- Streamlit DataFrame: https://docs.streamlit.io/library/api-reference/data/st.dataframe
- AgGrid: https://github.com/PablocFonseca/streamlit-aggrid

Sample Input:
- comparison_results: Dict with metrics for each strategy

Expected Output:
- Rendered tables with interactive sorting and filtering
"""

from typing import Dict, List, Optional, Any
import streamlit as st
import pandas as pd
from loguru import logger


def render_metrics_table(results: Dict) -> None:
    """
    Render the main performance metrics table.

    Args:
        results: Comparison results dictionary
    """
    if not results or "metrics" not in results:
        st.warning("No metrics data available")
        return

    metrics = results["metrics"]

    # Build DataFrame
    df = build_metrics_dataframe(metrics)

    # Display metrics with color coding
    st.dataframe(
        df.style.applymap(
            style_metric_cell,
            subset=["Total Return", "Sharpe Ratio", "Max Drawdown"]
        ).format({
            "Total Return": "{:.2f}%",
            "Sharpe Ratio": "{:.2f}",
            "Sortino Ratio": "{:.2f}",
            "Max Drawdown": "{:.2f}%",
            "Win Rate": "{:.2f}%",
            "Volatility": "{:.2f}%",
            "Calmar Ratio": "{:.2f}",
        }),
        use_container_width=True,
        height=400,
    )

    # Add metrics summary cards
    render_summary_cards(df)


def build_metrics_dataframe(metrics: Dict) -> pd.DataFrame:
    """
    Build a DataFrame from metrics dictionary.

    Args:
        metrics: Dictionary of strategy metrics

    Returns:
        pandas DataFrame with metrics
    """
    data = []

    for strategy, strategy_metrics in metrics.items():
        row = {
            "Strategy": strategy,
            "Total Return": strategy_metrics.get("total_return", 0),
            "Sharpe Ratio": strategy_metrics.get("sharpe_ratio", 0),
            "Sortino Ratio": strategy_metrics.get("sortino_ratio", 0),
            "Max Drawdown": strategy_metrics.get("max_drawdown", 0),
            "Win Rate": strategy_metrics.get("win_rate", 0),
            "Trade Count": strategy_metrics.get("trade_count", 0),
            "Volatility": strategy_metrics.get("volatility", 0),
            "Calmar Ratio": strategy_metrics.get("calmar_ratio", 0),
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by Total Return descending
    df = df.sort_values("Total Return", ascending=False)

    # Add rank
    df.insert(0, "Rank", range(1, len(df) + 1))

    return df


def style_metric_cell(val: float) -> str:
    """
    Apply color styling to metric cells.

    Args:
        val: Cell value

    Returns:
        CSS style string
    """
    if pd.isna(val):
        return ""

    # Color based on value
    if val > 0:
        color = "green"
    elif val < 0:
        color = "red"
    else:
        color = "gray"

    return f"color: {color}"


def render_summary_cards(df: pd.DataFrame) -> None:
    """
    Render summary metric cards.

    Args:
        df: Metrics DataFrame
    """
    st.markdown("---")
    st.markdown("### 📊 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        best_return = df["Total Return"].max()
        best_strategy = df.loc[df["Total Return"].idxmax(), "Strategy"]
        st.metric(
            "Best Return",
            f"{best_return:.2f}%",
            delta=best_strategy,
            delta_color="off"
        )

    with col2:
        best_sharpe = df["Sharpe Ratio"].max()
        best_sharpe_strategy = df.loc[df["Sharpe Ratio"].idxmax(), "Strategy"]
        st.metric(
            "Best Sharpe",
            f"{best_sharpe:.2f}",
            delta=best_sharpe_strategy,
            delta_color="off"
        )

    with col3:
        lowest_dd = df["Max Drawdown"].max()  # Max because drawdowns are negative
        lowest_dd_strategy = df.loc[df["Max Drawdown"].idxmax(), "Strategy"]
        st.metric(
            "Lowest Drawdown",
            f"{lowest_dd:.2f}%",
            delta=lowest_dd_strategy,
            delta_color="off"
        )

    with col4:
        avg_win_rate = df["Win Rate"].mean()
        st.metric(
            "Avg Win Rate",
            f"{avg_win_rate:.2f}%",
            delta=None
        )


def render_detailed_metrics(results: Dict) -> None:
    """
    Render detailed metrics with expandable sections.

    Args:
        results: Comparison results dictionary
    """
    if not results or "metrics" not in results:
        st.warning("No detailed metrics available")
        return

    metrics = results["metrics"]

    for strategy, strategy_metrics in metrics.items():
        with st.expander(f"📊 {strategy} - Detailed Metrics"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Return Metrics**")
                st.markdown(f"- Total Return: `{strategy_metrics.get('total_return', 0):.2f}%`")
                st.markdown(f"- CAGR: `{strategy_metrics.get('cagr', 0):.2f}%`")
                st.markdown(f"- Volatility: `{strategy_metrics.get('volatility', 0):.2f}%`")
                st.markdown(f"- Downside Deviation: `{strategy_metrics.get('downside_deviation', 0):.2f}%`")

                st.markdown("**Risk Metrics**")
                st.markdown(f"- Max Drawdown: `{strategy_metrics.get('max_drawdown', 0):.2f}%`")
                st.markdown(f"- Max Drawdown Duration: `{strategy_metrics.get('max_dd_duration', 0)} days`")
                st.markdown(f"- VaR (95%): `{strategy_metrics.get('var_95', 0):.2f}%`")
                st.markdown(f"- CVaR (95%): `{strategy_metrics.get('cvar_95', 0):.2f}%`")

            with col2:
                st.markdown("**Risk-Adjusted Returns**")
                st.markdown(f"- Sharpe Ratio: `{strategy_metrics.get('sharpe_ratio', 0):.2f}`")
                st.markdown(f"- Sortino Ratio: `{strategy_metrics.get('sortino_ratio', 0):.2f}`")
                st.markdown(f"- Calmar Ratio: `{strategy_metrics.get('calmar_ratio', 0):.2f}`")
                st.markdown(f"- Omega Ratio: `{strategy_metrics.get('omega_ratio', 0):.2f}`")

                st.markdown("**Trade Statistics**")
                st.markdown(f"- Total Trades: `{strategy_metrics.get('trade_count', 0)}`")
                st.markdown(f"- Win Rate: `{strategy_metrics.get('win_rate', 0):.2f}%`")
                st.markdown(f"- Profit Factor: `{strategy_metrics.get('profit_factor', 0):.2f}`")
                st.markdown(f"- Avg Trade: `{strategy_metrics.get('avg_trade', 0):.2f}%`")


def render_trades_table(results: Dict) -> None:
    """
    Render trade-level table with pagination.

    Args:
        results: Comparison results dictionary
    """
    if not results or "trades" not in results:
        st.warning("No trade data available")
        return

    # Strategy selector
    strategies = list(results["trades"].keys())
    selected_strategy = st.selectbox(
        "Select Strategy",
        options=strategies,
        key="trades_table_strategy"
    )

    if not selected_strategy:
        return

    trades = results["trades"][selected_strategy]

    # Build DataFrame
    df = pd.DataFrame(trades)

    if df.empty:
        st.info(f"No trades available for {selected_strategy}")
        return

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        side_filter = st.multiselect(
            "Trade Side",
            options=["LONG", "SHORT"],
            default=["LONG", "SHORT"],
            key="side_filter"
        )

    with col2:
        # Date range filter
        if "entry_date" in df.columns:
            min_date = pd.to_datetime(df["entry_date"]).min()
            max_date = pd.to_datetime(df["entry_date"]).max()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                key="date_filter"
            )

    with col3:
        show_winning = st.checkbox("Show only winning trades", value=False)

    # Apply filters
    filtered_df = df[df["side"].isin(side_filter)]

    if show_winning and "pnl_pct" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["pnl_pct"] > 0]

    # Pagination
    page_size = st.selectbox(
        "Trades per page",
        options=[20, 50, 100],
        index=0,
        key="page_size"
    )

    total_pages = (len(filtered_df) - 1) // page_size + 1
    current_page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        key="current_page"
    )

    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size

    # Display paginated data
    st.dataframe(
        filtered_df.iloc[start_idx:end_idx].style.applymap(
            lambda v: "color: green" if v > 0 else "color: red",
            subset=["pnl_pct"] if "pnl_pct" in filtered_df.columns else []
        ).format({
            "entry_price": "${:.2f}",
            "exit_price": "${:.2f}",
            "pnl_pct": "{:.2f}%",
            "duration_hours": "{:.1f}h"
        } if all(c in filtered_df.columns for c in ["entry_price", "exit_price", "pnl_pct", "duration_hours"]) else {}),
        use_container_width=True
    )

    # Summary for filtered trades
    if "pnl_pct" in filtered_df.columns:
        st.markdown(f"**Showing {len(filtered_df)} trades** "
                   f"(Page {current_page} of {total_pages})")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total PnL", f"{filtered_df['pnl_pct'].sum():.2f}%")
        with col2:
            st.metric("Avg PnL", f"{filtered_df['pnl_pct'].mean():.2f}%")
        with col3:
            win_rate = (filtered_df['pnl_pct'] > 0).sum() / len(filtered_df) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            st.metric("Best Trade", f"{filtered_df['pnl_pct'].max():.2f}%")


if __name__ == "__main__":
    # Validation function
    import sys
    import numpy as np

    print("🔍 Validating metrics_display.py...")

    all_validation_failures = []
    total_tests = 0

    # Test 1: DataFrame building
    total_tests += 1
    try:
        test_metrics = {
            "Strategy A": {
                "total_return": 45.2,
                "sharpe_ratio": 2.1,
                "sortino_ratio": 2.5,
                "max_drawdown": -15.3,
                "win_rate": 65.5,
                "trade_count": 100,
                "volatility": 18.4,
                "calmar_ratio": 2.95
            },
            "Strategy B": {
                "total_return": 32.1,
                "sharpe_ratio": 1.8,
                "sortino_ratio": 2.1,
                "max_drawdown": -12.1,
                "win_rate": 70.2,
                "trade_count": 85,
                "volatility": 15.2,
                "calmar_ratio": 2.65
            }
        }

        df = build_metrics_dataframe(test_metrics)

        expected_columns = [
            "Rank", "Strategy", "Total Return", "Sharpe Ratio",
            "Sortino Ratio", "Max Drawdown", "Win Rate",
            "Trade Count", "Volatility", "Calmar Ratio"
        ]

        if not all(col in df.columns for col in expected_columns):
            all_validation_failures.append(
                f"DataFrame columns: Expected {expected_columns}, got {df.columns.tolist()}"
            )

        if len(df) != 2:
            all_validation_failures.append(
                f"DataFrame rows: Expected 2, got {len(df)}"
            )

        # Check sorting (should be by Total Return descending)
        if df.iloc[0]["Total Return"] < df.iloc[1]["Total Return"]:
            all_validation_failures.append(
                "DataFrame sorting: Expected descending by Total Return"
            )

    except Exception as e:
        all_validation_failures.append(f"DataFrame building test failed: {e}")

    # Test 2: Style function
    total_tests += 1
    try:
        positive_style = style_metric_cell(10.5)
        if "green" not in positive_style:
            all_validation_failures.append(
                f"Style positive: Expected 'green' in style, got '{positive_style}'"
            )

        negative_style = style_metric_cell(-5.2)
        if "red" not in negative_style:
            all_validation_failures.append(
                f"Style negative: Expected 'red' in style, got '{negative_style}'"
            )

        zero_style = style_metric_cell(0)
        if "gray" not in zero_style:
            all_validation_failures.append(
                f"Style zero: Expected 'gray' in style, got '{zero_style}'"
            )

    except Exception as e:
        all_validation_failures.append(f"Style function test failed: {e}")

    # Test 3: NaN handling
    total_tests += 1
    try:
        nan_style = style_metric_cell(np.nan)
        if nan_style != "":
            all_validation_failures.append(
                f"NaN handling: Expected empty string, got '{nan_style}'"
            )
    except Exception as e:
        all_validation_failures.append(f"NaN handling test failed: {e}")

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
        print("Metrics display component is validated and ready for integration")
        sys.exit(0)
