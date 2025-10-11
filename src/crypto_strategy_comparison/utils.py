"""
Utility Functions

Common utility functions used throughout the application:
- Custom CSS styling
- Strategy icon mapping
- Date/time formatting
- Data transformation helpers

Documentation:
- Streamlit Markdown: https://docs.streamlit.io/library/api-reference/text/st.markdown

Sample Input:
- Various utility function calls with appropriate parameters

Expected Output:
- Formatted data, applied styles, helper function results
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import streamlit as st


def apply_custom_css() -> None:
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Main container styling */
        .main {
            padding: 0rem 1rem;
        }

        /* Metric cards */
        div[data-testid="metric-container"] {
            background-color: #f0f2f6;
            border: 1px solid #e0e0e0;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Tables */
        .dataframe {
            font-size: 14px;
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 5px;
            font-weight: 500;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }

        /* Headers */
        h1 {
            color: #1f77b4;
            font-weight: 700;
        }

        h2 {
            color: #2c3e50;
            font-weight: 600;
            margin-top: 2rem;
        }

        h3 {
            color: #34495e;
            font-weight: 600;
            margin-top: 1.5rem;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f0f2f6;
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
            font-weight: 500;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e0e2e6;
        }

        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #f0f2f6;
            border-radius: 5px;
            font-weight: 500;
        }

        /* Success/Error/Warning boxes */
        .stSuccess, .stError, .stWarning, .stInfo {
            padding: 1rem;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def get_strategy_icon(strategy_name: str) -> str:
    """
    Get icon emoji for a strategy based on its name.

    Args:
        strategy_name: Name of the strategy

    Returns:
        Emoji icon string
    """
    # Mapping of strategy keywords to icons
    icon_map = {
        "momentum": "🚀",
        "mean_reversion": "🔄",
        "mean reversion": "🔄",
        "grid": "📊",
        "dca": "💰",
        "dollar cost": "💰",
        "rsi": "📈",
        "macd": "📉",
        "bollinger": "📊",
        "arbitrage": "⚖️",
        "scalp": "⚡",
        "swing": "🎯",
        "trend": "📈",
        "breakout": "💥",
    }

    strategy_lower = strategy_name.lower()

    for keyword, icon in icon_map.items():
        if keyword in strategy_lower:
            return icon

    # Default icon
    return "📊"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as a percentage string.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"


def format_currency(value: float, currency: str = "$") -> str:
    """
    Format a value as currency.

    Args:
        value: Numeric value
        currency: Currency symbol

    Returns:
        Formatted currency string
    """
    return f"{currency}{value:,.2f}"


def format_duration(hours: float) -> str:
    """
    Format duration in hours to human-readable string.

    Args:
        hours: Duration in hours

    Returns:
        Formatted duration string
    """
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes}m"
    elif hours < 24:
        return f"{hours:.1f}h"
    else:
        days = hours / 24
        return f"{days:.1f}d"


def get_time_horizon_dates(horizon: str) -> tuple[datetime, datetime]:
    """
    Get start and end dates for a time horizon.

    Args:
        horizon: Time horizon code ("1W", "1M", "3M", "6M", "1Y", "ALL")

    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()

    horizon_map = {
        "1W": timedelta(weeks=1),
        "1M": timedelta(days=30),
        "3M": timedelta(days=90),
        "6M": timedelta(days=180),
        "1Y": timedelta(days=365),
        "ALL": timedelta(days=3650),  # ~10 years
    }

    delta = horizon_map.get(horizon, timedelta(days=180))
    start_date = end_date - delta

    return start_date, end_date


def calculate_rank(values: List[float], ascending: bool = False) -> List[int]:
    """
    Calculate ranks for a list of values.

    Args:
        values: List of numeric values
        ascending: If True, rank ascending (lower is better)

    Returns:
        List of ranks
    """
    if not values:
        return []

    # Create list of (value, original_index) tuples
    indexed_values = [(val, idx) for idx, val in enumerate(values)]

    # Sort by value
    indexed_values.sort(key=lambda x: x[0], reverse=not ascending)

    # Assign ranks
    ranks = [0] * len(values)
    for rank, (_, idx) in enumerate(indexed_values, start=1):
        ranks[idx] = rank

    return ranks


def color_by_rank(rank: int, total: int) -> str:
    """
    Get color code based on rank position.

    Args:
        rank: Rank position (1-based)
        total: Total number of items

    Returns:
        Color code string
    """
    # Top 20% - green
    if rank <= total * 0.2:
        return "#28a745"
    # Top 50% - yellow
    elif rank <= total * 0.5:
        return "#ffc107"
    # Bottom 50% - red
    else:
        return "#dc3545"


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


if __name__ == "__main__":
    # Validation function
    import sys

    print("🔍 Validating utils.py...")

    all_validation_failures = []
    total_tests = 0

    # Test 1: Strategy icon mapping
    total_tests += 1
    try:
        momentum_icon = get_strategy_icon("Momentum ETH")
        if momentum_icon != "🚀":
            all_validation_failures.append(
                f"Strategy icon: Expected '🚀' for Momentum, got '{momentum_icon}'"
            )

        mean_rev_icon = get_strategy_icon("Mean Reversion BTC")
        if mean_rev_icon != "🔄":
            all_validation_failures.append(
                f"Strategy icon: Expected '🔄' for Mean Reversion, got '{mean_rev_icon}'"
            )

        default_icon = get_strategy_icon("Unknown Strategy")
        if default_icon != "📊":
            all_validation_failures.append(
                f"Strategy icon: Expected default '📊', got '{default_icon}'"
            )

    except Exception as e:
        all_validation_failures.append(f"Strategy icon test failed: {e}")

    # Test 2: Percentage formatting
    total_tests += 1
    try:
        formatted = format_percentage(45.678, decimals=2)
        expected = "45.68%"
        if formatted != expected:
            all_validation_failures.append(
                f"Percentage format: Expected '{expected}', got '{formatted}'"
            )

        formatted_1dp = format_percentage(45.678, decimals=1)
        expected_1dp = "45.7%"
        if formatted_1dp != expected_1dp:
            all_validation_failures.append(
                f"Percentage format (1dp): Expected '{expected_1dp}', got '{formatted_1dp}'"
            )

    except Exception as e:
        all_validation_failures.append(f"Percentage formatting test failed: {e}")

    # Test 3: Currency formatting
    total_tests += 1
    try:
        formatted = format_currency(1234.56)
        expected = "$1,234.56"
        if formatted != expected:
            all_validation_failures.append(
                f"Currency format: Expected '{expected}', got '{formatted}'"
            )

        formatted_eur = format_currency(1234.56, currency="€")
        expected_eur = "€1,234.56"
        if formatted_eur != expected_eur:
            all_validation_failures.append(
                f"Currency format (EUR): Expected '{expected_eur}', got '{formatted_eur}'"
            )

    except Exception as e:
        all_validation_failures.append(f"Currency formatting test failed: {e}")

    # Test 4: Duration formatting
    total_tests += 1
    try:
        minutes_format = format_duration(0.5)  # 30 minutes
        if "m" not in minutes_format:
            all_validation_failures.append(
                f"Duration format (minutes): Expected 'm' in output, got '{minutes_format}'"
            )

        hours_format = format_duration(5.5)
        if "h" not in hours_format:
            all_validation_failures.append(
                f"Duration format (hours): Expected 'h' in output, got '{hours_format}'"
            )

        days_format = format_duration(48)
        if "d" not in days_format:
            all_validation_failures.append(
                f"Duration format (days): Expected 'd' in output, got '{days_format}'"
            )

    except Exception as e:
        all_validation_failures.append(f"Duration formatting test failed: {e}")

    # Test 5: Time horizon dates
    total_tests += 1
    try:
        start, end = get_time_horizon_dates("1M")

        if not isinstance(start, datetime) or not isinstance(end, datetime):
            all_validation_failures.append(
                "Time horizon dates: Expected datetime objects"
            )

        if start >= end:
            all_validation_failures.append(
                "Time horizon dates: Start date should be before end date"
            )

        delta = (end - start).days
        if not (25 <= delta <= 35):  # Allow some tolerance
            all_validation_failures.append(
                f"Time horizon dates (1M): Expected ~30 days, got {delta} days"
            )

    except Exception as e:
        all_validation_failures.append(f"Time horizon dates test failed: {e}")

    # Test 6: Rank calculation
    total_tests += 1
    try:
        values = [100, 50, 75, 25]
        ranks = calculate_rank(values)

        # Expected ranks (descending): 100=1, 75=2, 50=3, 25=4
        expected = [1, 3, 2, 4]
        if ranks != expected:
            all_validation_failures.append(
                f"Rank calculation: Expected {expected}, got {ranks}"
            )

        # Test ascending
        ranks_asc = calculate_rank(values, ascending=True)
        expected_asc = [4, 2, 3, 1]
        if ranks_asc != expected_asc:
            all_validation_failures.append(
                f"Rank calculation (ascending): Expected {expected_asc}, got {ranks_asc}"
            )

    except Exception as e:
        all_validation_failures.append(f"Rank calculation test failed: {e}")

    # Test 7: Safe division
    total_tests += 1
    try:
        result = safe_divide(10, 2)
        if result != 5.0:
            all_validation_failures.append(
                f"Safe divide (normal): Expected 5.0, got {result}"
            )

        result_zero = safe_divide(10, 0, default=0.0)
        if result_zero != 0.0:
            all_validation_failures.append(
                f"Safe divide (zero): Expected 0.0, got {result_zero}"
            )

        result_custom = safe_divide(10, 0, default=999.0)
        if result_custom != 999.0:
            all_validation_failures.append(
                f"Safe divide (custom default): Expected 999.0, got {result_custom}"
            )

    except Exception as e:
        all_validation_failures.append(f"Safe division test failed: {e}")

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
        print("Utils module is validated and ready for use")
        sys.exit(0)
