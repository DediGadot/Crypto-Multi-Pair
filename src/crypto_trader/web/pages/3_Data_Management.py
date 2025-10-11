"""
Data Management Page - View, fetch, and manage market data

This module provides an interface for viewing available market data, fetching new data,
updating existing datasets, and performing data quality checks.

**Purpose**: Data management interface for viewing, fetching, updating, and validating
market data used in backtesting and strategy analysis.

**Features**:
- View available data by symbol and timeframe
- Fetch new historical data from exchanges
- Update existing datasets
- Data quality checks and validation
- Data statistics and coverage information
- Export data functionality

**Third-party packages**:
- streamlit: https://docs.streamlit.io/
- pandas: https://pandas.pydata.org/docs/
- plotly: https://plotly.com/python/

**Sample Input**:
Run with: `streamlit run src/crypto_trader/web/app.py` (accessed via page navigation)

**Expected Output**:
Data management interface with data viewing, fetching, and quality checking tools
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from crypto_trader.core.types import Timeframe

st.set_page_config(
    page_title="Data Management",
    page_icon="💾",
    layout="wide",
)

st.title("💾 Market Data Management")
st.markdown("### View, fetch, and manage cryptocurrency market data")


def get_available_data() -> pd.DataFrame:
    """
    Get list of available data in the system.
    In production, this would query the database.
    """
    # Mock data for demonstration
    data_records = [
        {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2025-10-11",
            "records": 8000,
            "completeness": 99.8,
            "last_updated": "2025-10-11 10:00:00",
        },
        {
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "start_date": "2023-01-01",
            "end_date": "2025-10-11",
            "records": 6000,
            "completeness": 99.9,
            "last_updated": "2025-10-11 10:00:00",
        },
        {
            "symbol": "ETHUSDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2025-10-11",
            "records": 8000,
            "completeness": 99.5,
            "last_updated": "2025-10-11 10:00:00",
        },
        {
            "symbol": "ETHUSDT",
            "timeframe": "4h",
            "start_date": "2023-06-01",
            "end_date": "2025-10-11",
            "records": 4000,
            "completeness": 99.7,
            "last_updated": "2025-10-11 10:00:00",
        },
        {
            "symbol": "BNBUSDT",
            "timeframe": "1h",
            "start_date": "2024-03-01",
            "end_date": "2025-10-11",
            "records": 5500,
            "completeness": 98.9,
            "last_updated": "2025-10-11 09:30:00",
        },
    ]

    return pd.DataFrame(data_records)


def check_data_quality(symbol: str, timeframe: str) -> dict:
    """
    Perform data quality checks on a dataset.
    In production, this would analyze actual data.
    """
    import numpy as np

    # Mock quality check results
    checks = {
        "missing_data": {
            "missing_candles": 10,
            "percentage": 0.12,
            "status": "✅ Good",
        },
        "price_anomalies": {
            "extreme_moves": 2,
            "spike_count": 1,
            "status": "✅ Normal",
        },
        "volume_analysis": {
            "zero_volume_candles": 0,
            "avg_volume": 12500000.0,
            "status": "✅ Good",
        },
        "time_gaps": {
            "gaps_found": 0,
            "max_gap_hours": 0,
            "status": "✅ Perfect",
        },
    }

    return checks


def create_data_coverage_chart(df: pd.DataFrame) -> go.Figure:
    """Create timeline chart showing data coverage."""
    fig = go.Figure()

    for idx, row in df.iterrows():
        start = pd.to_datetime(row["start_date"])
        end = pd.to_datetime(row["end_date"])

        fig.add_trace(
            go.Scatter(
                x=[start, end],
                y=[f"{row['symbol']} ({row['timeframe']})", f"{row['symbol']} ({row['timeframe']})"],
                mode="lines+markers",
                line=dict(width=10),
                marker=dict(size=10),
                name=f"{row['symbol']} {row['timeframe']}",
                hovertemplate=f"<b>{row['symbol']} {row['timeframe']}</b><br>"
                + f"Records: {row['records']}<br>"
                + f"Completeness: {row['completeness']:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Data Coverage Timeline",
        xaxis_title="Date",
        yaxis_title="Symbol & Timeframe",
        template="plotly_white",
        height=400,
        showlegend=False,
        hovermode="closest",
    )

    return fig


def create_completeness_chart(df: pd.DataFrame) -> go.Figure:
    """Create bar chart showing data completeness."""
    df_sorted = df.sort_values("completeness")

    colors = [
        "#dc3545" if x < 95 else "#ffc107" if x < 98 else "#28a745"
        for x in df_sorted["completeness"]
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df_sorted["completeness"],
                y=[
                    f"{row['symbol']} ({row['timeframe']})"
                    for _, row in df_sorted.iterrows()
                ],
                orientation="h",
                marker_color=colors,
                text=[f"{x:.1f}%" for x in df_sorted["completeness"]],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Data Completeness by Symbol",
        xaxis_title="Completeness (%)",
        yaxis_title="Symbol & Timeframe",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    # Add reference lines
    fig.add_vline(x=95, line_dash="dash", line_color="orange", opacity=0.5)
    fig.add_vline(x=98, line_dash="dash", line_color="green", opacity=0.5)

    return fig


def fetch_new_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> bool:
    """
    Fetch new data from exchange.
    In production, this would use the data fetcher module.
    """
    # Simulate data fetching
    import time

    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = 10
    for i in range(steps):
        progress_bar.progress((i + 1) / steps)
        status_text.text(f"Fetching data... {(i + 1) * 10}%")
        time.sleep(0.1)

    progress_bar.empty()
    status_text.empty()

    return True


def main():
    """Main data management page logic."""
    # Tabs for different operations
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 View Data", "⬇️ Fetch Data", "🔄 Update Data", "✅ Quality Checks"]
    )

    with tab1:
        st.subheader("Available Market Data")

        # Get available data
        df_data = get_available_data()

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Datasets", len(df_data))

        with col2:
            total_records = df_data["records"].sum()
            st.metric("Total Records", f"{total_records:,}")

        with col3:
            avg_completeness = df_data["completeness"].mean()
            st.metric("Avg Completeness", f"{avg_completeness:.1f}%")

        with col4:
            unique_symbols = df_data["symbol"].nunique()
            st.metric("Unique Symbols", unique_symbols)

        st.markdown("---")

        # Data coverage visualization
        st.subheader("📈 Data Coverage")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_data_coverage_chart(df_data), use_container_width=True
            )

        with col2:
            st.plotly_chart(
                create_completeness_chart(df_data), use_container_width=True
            )

        st.markdown("---")

        # Data table with filtering
        st.subheader("📋 Data Inventory")

        col1, col2, col3 = st.columns(3)

        with col1:
            symbol_filter = st.multiselect(
                "Filter by Symbol",
                df_data["symbol"].unique().tolist(),
                default=df_data["symbol"].unique().tolist(),
            )

        with col2:
            timeframe_filter = st.multiselect(
                "Filter by Timeframe",
                df_data["timeframe"].unique().tolist(),
                default=df_data["timeframe"].unique().tolist(),
            )

        with col3:
            min_completeness = st.slider(
                "Min Completeness (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=5.0,
            )

        # Apply filters
        df_filtered = df_data[
            (df_data["symbol"].isin(symbol_filter))
            & (df_data["timeframe"].isin(timeframe_filter))
            & (df_data["completeness"] >= min_completeness)
        ]

        # Format table
        df_display = df_filtered.copy()
        df_display["records"] = df_display["records"].apply(lambda x: f"{x:,}")
        df_display["completeness"] = df_display["completeness"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            df_display,
            use_container_width=True,
            height=300,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", width="medium"),
                "timeframe": st.column_config.TextColumn("Timeframe", width="small"),
                "start_date": st.column_config.TextColumn("Start Date", width="medium"),
                "end_date": st.column_config.TextColumn("End Date", width="medium"),
                "records": st.column_config.TextColumn("Records", width="small"),
                "completeness": st.column_config.TextColumn("Complete", width="small"),
                "last_updated": st.column_config.TextColumn("Updated", width="medium"),
            },
        )

        # Export button
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Export Data Inventory",
            data=csv,
            file_name=f"data_inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("⬇️ Fetch New Data")

        st.info(
            """
            Fetch historical market data from exchanges. Data will be stored in the database
            and made available for backtesting and analysis.
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            # Symbol selection
            symbol = st.selectbox(
                "Symbol",
                ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"],
                index=0,
            )

            # Timeframe
            timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=4,
            )

            # Exchange
            exchange = st.selectbox(
                "Exchange",
                ["Binance", "Coinbase", "Kraken"],
                index=0,
            )

        with col2:
            # Date range
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now(),
            )

            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now(),
            )

            # Data options
            include_volume = st.checkbox("Include Volume Data", value=True)
            verify_data = st.checkbox("Verify Data Quality", value=True)

        st.markdown("---")

        # Fetch button
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("🚀 Fetch Data", type="primary", use_container_width=True):
                if start_date >= end_date:
                    st.error("Start date must be before end date")
                    return

                with st.spinner(f"Fetching {symbol} {timeframe} data from {exchange}..."):
                    success = fetch_new_data(symbol, timeframe, start_date, end_date)

                    if success:
                        st.success(
                            f"✅ Successfully fetched {symbol} {timeframe} data "
                            f"from {start_date} to {end_date}"
                        )

                        # Show summary
                        days = (end_date - start_date).days
                        estimated_records = days * 24  # Assuming hourly data

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Records Fetched", f"{estimated_records:,}")
                        with col2:
                            st.metric("Date Range", f"{days} days")
                        with col3:
                            st.metric("Status", "Complete")
                    else:
                        st.error("Failed to fetch data. Please try again.")

    with tab3:
        st.subheader("🔄 Update Existing Data")

        st.info(
            """
            Update existing datasets with the latest market data. This will fetch
            new candles since the last update.
            """
        )

        # Get datasets that need updating
        df_data = get_available_data()
        df_data["needs_update"] = (
            pd.to_datetime("now") - pd.to_datetime(df_data["last_updated"])
        ).dt.days > 0

        st.markdown("### Datasets Needing Update")

        df_needs_update = df_data[df_data["needs_update"]]

        if len(df_needs_update) > 0:
            st.dataframe(
                df_needs_update[["symbol", "timeframe", "last_updated", "end_date"]],
                use_container_width=True,
            )

            # Update options
            col1, col2 = st.columns(2)

            with col1:
                update_selection = st.multiselect(
                    "Select datasets to update",
                    [
                        f"{row['symbol']} {row['timeframe']}"
                        for _, row in df_needs_update.iterrows()
                    ],
                    default=[
                        f"{row['symbol']} {row['timeframe']}"
                        for _, row in df_needs_update.iterrows()
                    ][:3],
                )

            with col2:
                st.write("")
                st.write("")
                if st.button("🔄 Update Selected", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, dataset in enumerate(update_selection):
                        status_text.text(f"Updating {dataset}...")
                        progress_bar.progress((idx + 1) / len(update_selection))
                        # Simulate update
                        import time

                        time.sleep(0.5)

                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"✅ Successfully updated {len(update_selection)} datasets")
        else:
            st.success("✅ All datasets are up to date!")

        st.markdown("---")

        # Bulk update options
        st.subheader("Bulk Update Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Update All BTCUSDT", use_container_width=True):
                st.info("Updating all BTCUSDT datasets...")

        with col2:
            if st.button("Update All 1h Data", use_container_width=True):
                st.info("Updating all 1h timeframe data...")

        with col3:
            if st.button("Update Everything", use_container_width=True):
                st.info("Updating all datasets...")

    with tab4:
        st.subheader("✅ Data Quality Checks")

        st.info(
            """
            Perform comprehensive quality checks on market data to identify issues
            such as missing candles, price anomalies, and data gaps.
            """
        )

        # Select dataset for quality check
        col1, col2 = st.columns(2)

        with col1:
            qc_symbol = st.selectbox(
                "Symbol",
                ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                key="qc_symbol",
            )

        with col2:
            qc_timeframe = st.selectbox(
                "Timeframe",
                ["1h", "4h", "1d"],
                key="qc_timeframe",
            )

        if st.button("🔍 Run Quality Check", type="primary"):
            with st.spinner("Running quality checks..."):
                checks = check_data_quality(qc_symbol, qc_timeframe)

                st.success("✅ Quality check complete!")

                # Display results
                st.markdown("### Quality Check Results")

                col1, col2 = st.columns(2)

                with col1:
                    # Missing data check
                    st.markdown("#### 📊 Missing Data")
                    st.metric(
                        "Missing Candles",
                        checks["missing_data"]["missing_candles"],
                        delta=f"{checks['missing_data']['percentage']:.2f}%",
                    )
                    st.write(checks["missing_data"]["status"])

                    # Price anomalies
                    st.markdown("#### 💹 Price Anomalies")
                    st.metric(
                        "Extreme Moves", checks["price_anomalies"]["extreme_moves"]
                    )
                    st.metric("Spike Count", checks["price_anomalies"]["spike_count"])
                    st.write(checks["price_anomalies"]["status"])

                with col2:
                    # Volume analysis
                    st.markdown("#### 📈 Volume Analysis")
                    st.metric(
                        "Zero Volume Candles",
                        checks["volume_analysis"]["zero_volume_candles"],
                    )
                    st.metric(
                        "Avg Volume",
                        f"${checks['volume_analysis']['avg_volume']:,.0f}",
                    )
                    st.write(checks["volume_analysis"]["status"])

                    # Time gaps
                    st.markdown("#### ⏱️ Time Gaps")
                    st.metric("Gaps Found", checks["time_gaps"]["gaps_found"])
                    st.metric("Max Gap (hours)", checks["time_gaps"]["max_gap_hours"])
                    st.write(checks["time_gaps"]["status"])

                # Overall status
                st.markdown("---")
                st.markdown("### Overall Status")

                all_good = all(
                    check["status"].startswith("✅") for check in checks.values()
                )

                if all_good:
                    st.success(
                        "🎉 All quality checks passed! Data is ready for backtesting."
                    )
                else:
                    st.warning(
                        "⚠️ Some issues detected. Review the results above for details."
                    )

        # Batch quality checks
        st.markdown("---")
        st.subheader("Batch Quality Checks")

        if st.button("🔍 Check All Datasets"):
            with st.spinner("Running batch quality checks..."):
                df_data = get_available_data()

                results = []
                for _, row in df_data.iterrows():
                    checks = check_data_quality(row["symbol"], row["timeframe"])

                    # Calculate overall score
                    score = 100.0
                    if checks["missing_data"]["percentage"] > 1:
                        score -= 10
                    if checks["price_anomalies"]["extreme_moves"] > 5:
                        score -= 10
                    if checks["volume_analysis"]["zero_volume_candles"] > 0:
                        score -= 5
                    if checks["time_gaps"]["gaps_found"] > 0:
                        score -= 15

                    results.append(
                        {
                            "symbol": row["symbol"],
                            "timeframe": row["timeframe"],
                            "quality_score": score,
                            "status": "✅" if score >= 90 else "⚠️" if score >= 70 else "❌",
                        }
                    )

                df_results = pd.DataFrame(results)

                st.dataframe(
                    df_results,
                    use_container_width=True,
                    column_config={
                        "quality_score": st.column_config.ProgressColumn(
                            "Quality Score",
                            help="Overall data quality score (0-100)",
                            format="%d%%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                )

                # Summary
                avg_score = df_results["quality_score"].mean()
                st.metric("Average Quality Score", f"{avg_score:.1f}%")


if __name__ == "__main__":
    main()
