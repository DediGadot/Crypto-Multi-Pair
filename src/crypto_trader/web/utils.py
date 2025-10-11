"""
Utility functions for Streamlit dashboard

This module provides shared utility functions used across multiple pages in the
Streamlit dashboard, including formatting, chart generation, and data processing helpers.

**Purpose**: Centralized utility functions for dashboard components to avoid code
duplication and ensure consistency across pages.

**Third-party packages**:
- streamlit: https://docs.streamlit.io/
- plotly: https://plotly.com/python/
- pandas: https://pandas.pydata.org/docs/

**Sample Input**:
```python
from crypto_trader.web.utils import format_percentage, create_metric_card
formatted = format_percentage(0.2543)  # Returns "25.43%"
```

**Expected Output**:
Various formatted strings, styled components, and helper functions
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.

    Args:
        value: Decimal value (e.g., 0.1234 for 12.34%)
        decimals: Number of decimal places (default: 2)

    Returns:
        Formatted percentage string (e.g., "12.34%")
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2, symbol: str = "$") -> str:
    """
    Format a value as currency with thousands separators.

    Args:
        value: Numeric value
        decimals: Number of decimal places (default: 2)
        symbol: Currency symbol (default: "$")

    Returns:
        Formatted currency string (e.g., "$1,234.56")
    """
    return f"{symbol}{value:,.{decimals}f}"


def format_number(value: Union[int, float], decimals: int = 0) -> str:
    """
    Format a number with thousands separators.

    Args:
        value: Numeric value
        decimals: Number of decimal places (default: 0)

    Returns:
        Formatted number string (e.g., "1,234,567")
    """
    if decimals == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def get_metric_color(value: float, metric_type: str) -> str:
    """
    Get color code for a metric based on its value and type.

    Args:
        value: Metric value
        metric_type: Type of metric ("return", "sharpe", "drawdown", etc.)

    Returns:
        CSS color code (hex or named color)
    """
    if metric_type in ["return", "sharpe", "sortino", "calmar", "profit_factor"]:
        # Higher is better
        if value > 0:
            return "#28a745"  # Green
        elif value < 0:
            return "#dc3545"  # Red
        else:
            return "#6c757d"  # Gray
    elif metric_type in ["drawdown"]:
        # Lower is better
        if value < 0.1:
            return "#28a745"  # Green
        elif value < 0.2:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red
    else:
        return "#007bff"  # Blue (default)


def create_metric_card_html(
    label: str,
    value: str,
    delta: Optional[str] = None,
    color: str = "#1f77b4",
) -> str:
    """
    Create HTML for a custom metric card.

    Args:
        label: Metric label
        value: Metric value (formatted string)
        delta: Optional delta/change indicator
        color: Border color (hex code)

    Returns:
        HTML string for the metric card
    """
    delta_html = f"<div class='metric-delta'>{delta}</div>" if delta else ""

    return f"""
    <div style='
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.3rem solid {color};
        margin: 0.5rem 0;
    '>
        <div style='
            font-size: 12px;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 1px;
        '>{label}</div>
        <div style='
            font-size: 28px;
            font-weight: bold;
            margin-top: 8px;
        '>{value}</div>
        {delta_html}
    </div>
    """


def create_quality_badge(quality: str) -> str:
    """
    Create HTML badge for strategy quality rating.

    Args:
        quality: Quality rating ("EXCELLENT", "GOOD", "FAIR", "POOR")

    Returns:
        HTML string for the badge
    """
    colors = {
        "EXCELLENT": "#28a745",
        "GOOD": "#17a2b8",
        "FAIR": "#ffc107",
        "POOR": "#dc3545",
    }

    bg_color = colors.get(quality, "#6c757d")

    return f"""
    <span style='
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        background-color: {bg_color};
        color: white;
    '>{quality}</span>
    """


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Main container */
        .main {
            padding: 0rem 1rem;
        }

        /* Metric styling */
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

        /* Headers */
        h1 {
            color: #1f77b4;
            padding-bottom: 1rem;
            border-bottom: 2px solid #1f77b4;
        }

        h2 {
            color: #555;
            margin-top: 2rem;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 0.5rem;
            font-weight: 600;
        }

        /* Tables */
        .stDataFrame {
            border: 1px solid #ddd;
            border-radius: 0.5rem;
        }

        /* Expander */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #333;
        }

        /* Info boxes */
        .stAlert {
            border-radius: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_empty_chart_message(message: str = "No data available") -> go.Figure:
    """
    Create an empty chart with a message.

    Args:
        message: Message to display

    Returns:
        Empty Plotly figure with message
    """
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#666"),
    )

    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        template="plotly_white",
        height=300,
    )

    return fig


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero

    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_cagr(initial: float, final: float, years: float) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).

    Args:
        initial: Initial value
        final: Final value
        years: Number of years

    Returns:
        CAGR as decimal (e.g., 0.15 for 15%)
    """
    if initial <= 0 or years <= 0:
        return 0.0

    return (final / initial) ** (1 / years) - 1


def filter_dataframe(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply multiple filters to a DataFrame.

    Args:
        df: DataFrame to filter
        filters: Dictionary of column names and filter conditions

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    for column, condition in filters.items():
        if column not in filtered_df.columns:
            continue

        if isinstance(condition, (list, tuple)):
            # Multiple values (inclusion)
            filtered_df = filtered_df[filtered_df[column].isin(condition)]
        elif isinstance(condition, dict):
            # Range filter with min/max
            if "min" in condition:
                filtered_df = filtered_df[filtered_df[column] >= condition["min"]]
            if "max" in condition:
                filtered_df = filtered_df[filtered_df[column] <= condition["max"]]
        else:
            # Single value
            filtered_df = filtered_df[filtered_df[column] == condition]

    return filtered_df


def create_download_link(df: pd.DataFrame, filename: str, label: str) -> None:
    """
    Create a download button for a DataFrame.

    Args:
        df: DataFrame to download
        filename: Name of the downloaded file
        label: Button label
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def add_footer():
    """Add a standard footer to the page."""
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
    """
    Validation function to test utility functions with real inputs.
    """
    import sys

    # Track all validation failures
    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating utils.py with real data...\n")

    # Test 1: Format percentage
    total_tests += 1
    print("Test 1: Format percentage")
    try:
        result = format_percentage(0.2543)
        expected = "25.43%"
        if result != expected:
            all_validation_failures.append(
                f"Format percentage: Expected '{expected}', got '{result}'"
            )
        print(f"  ✓ format_percentage(0.2543) = {result}")
    except Exception as e:
        all_validation_failures.append(f"Format percentage exception: {e}")

    # Test 2: Format currency
    total_tests += 1
    print("\nTest 2: Format currency")
    try:
        result = format_currency(1234.56)
        expected = "$1,234.56"
        if result != expected:
            all_validation_failures.append(
                f"Format currency: Expected '{expected}', got '{result}'"
            )
        print(f"  ✓ format_currency(1234.56) = {result}")
    except Exception as e:
        all_validation_failures.append(f"Format currency exception: {e}")

    # Test 3: Format number
    total_tests += 1
    print("\nTest 3: Format number")
    try:
        result = format_number(1234567)
        expected = "1,234,567"
        if result != expected:
            all_validation_failures.append(
                f"Format number: Expected '{expected}', got '{result}'"
            )
        print(f"  ✓ format_number(1234567) = {result}")
    except Exception as e:
        all_validation_failures.append(f"Format number exception: {e}")

    # Test 4: Get metric color
    total_tests += 1
    print("\nTest 4: Get metric color")
    try:
        color_positive = get_metric_color(0.5, "return")
        color_negative = get_metric_color(-0.2, "return")

        if not color_positive.startswith("#"):
            all_validation_failures.append(
                f"Metric color should be hex code, got '{color_positive}'"
            )

        print(f"  ✓ Positive return color: {color_positive}")
        print(f"  ✓ Negative return color: {color_negative}")
    except Exception as e:
        all_validation_failures.append(f"Get metric color exception: {e}")

    # Test 5: Safe divide
    total_tests += 1
    print("\nTest 5: Safe divide")
    try:
        result_normal = safe_divide(10, 2)
        result_zero = safe_divide(10, 0, default=0.0)

        if result_normal != 5.0:
            all_validation_failures.append(
                f"Safe divide: Expected 5.0, got {result_normal}"
            )

        if result_zero != 0.0:
            all_validation_failures.append(
                f"Safe divide by zero: Expected 0.0, got {result_zero}"
            )

        print(f"  ✓ safe_divide(10, 2) = {result_normal}")
        print(f"  ✓ safe_divide(10, 0) = {result_zero}")
    except Exception as e:
        all_validation_failures.append(f"Safe divide exception: {e}")

    # Test 6: Calculate CAGR
    total_tests += 1
    print("\nTest 6: Calculate CAGR")
    try:
        cagr = calculate_cagr(10000, 15000, 2)
        expected_cagr = 0.2247  # Approximately 22.47%

        if abs(cagr - expected_cagr) > 0.01:
            all_validation_failures.append(
                f"CAGR: Expected ~{expected_cagr:.4f}, got {cagr:.4f}"
            )

        print(f"  ✓ CAGR from $10,000 to $15,000 in 2 years: {cagr:.2%}")
    except Exception as e:
        all_validation_failures.append(f"Calculate CAGR exception: {e}")

    # Test 7: Create empty chart
    total_tests += 1
    print("\nTest 7: Create empty chart")
    try:
        fig = create_empty_chart_message("Test message")

        if not isinstance(fig, go.Figure):
            all_validation_failures.append(
                f"Empty chart: Expected go.Figure, got {type(fig)}"
            )

        print(f"  ✓ Created empty chart with message")
    except Exception as e:
        all_validation_failures.append(f"Create empty chart exception: {e}")

    # Test 8: Filter DataFrame
    total_tests += 1
    print("\nTest 8: Filter DataFrame")
    try:
        df = pd.DataFrame(
            {
                "strategy": ["A", "B", "C", "D"],
                "sharpe": [1.5, 2.0, 0.8, 2.5],
                "return": [0.15, 0.20, 0.05, 0.30],
            }
        )

        filters = {"sharpe": {"min": 1.5}, "return": {"min": 0.10}}

        df_filtered = filter_dataframe(df, filters)

        # Should keep strategies A, B, D
        expected_count = 3
        if len(df_filtered) != expected_count:
            all_validation_failures.append(
                f"Filter DataFrame: Expected {expected_count} rows, got {len(df_filtered)}"
            )

        print(f"  ✓ Filtered DataFrame from {len(df)} to {len(df_filtered)} rows")
    except Exception as e:
        all_validation_failures.append(f"Filter DataFrame exception: {e}")

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
        print("Function is validated and ready for use")
        sys.exit(0)
