"""
Sidebar UI Component

Renders the left sidebar with:
- Strategy selection controls
- Time horizon filters
- Risk filters (max drawdown, min Sharpe ratio)
- Parameter exploration controls
- Export options

Documentation:
- Streamlit Sidebar: https://docs.streamlit.io/library/api-reference/layout/st.sidebar

Sample Input:
- Available strategies list
- Current session state

Expected Output:
- Rendered sidebar with all controls
- Updated session state based on user interactions
"""

from typing import Dict, List, Optional, Any
import streamlit as st
from loguru import logger


def render_sidebar() -> None:
    """Render the main sidebar with all control sections."""
    with st.sidebar:
        st.markdown("# 🎯 Controls")

        # Strategy selection section
        render_strategy_selection()

        st.markdown("---")

        # Time horizon section
        render_time_horizon_controls()

        st.markdown("---")

        # Risk filters section
        render_risk_filters()

        st.markdown("---")

        # Parameter explorer section
        render_parameter_explorer()

        st.markdown("---")

        # Export options section
        render_export_section()

        # Footer
        st.markdown("---")
        st.markdown("*📚 [Documentation](https://docs.example.com)*")
        st.markdown("*💬 [Support](https://support.example.com)*")


def render_strategy_selection() -> None:
    """Render strategy selection controls."""
    st.markdown("### 🎯 Strategy Selection")

    # Get available strategies
    from crypto_strategy_comparison.strategy_loader import StrategyLoader
    available = StrategyLoader.get_available_strategies()

    # Individual checkboxes for better control
    selected_strategies = []

    for strategy in available:
        from crypto_strategy_comparison.utils import get_strategy_icon
        icon = get_strategy_icon(strategy)

        if st.checkbox(
            f"{icon} {strategy}",
            value=strategy in st.session_state.get("selected_strategies", []),
            key=f"checkbox_{strategy}"
        ):
            selected_strategies.append(strategy)

    # Update session state
    st.session_state.selected_strategies = selected_strategies

    # Add custom strategy button
    if st.button("➕ Load Custom Strategy", use_container_width=True):
        st.info("Custom strategy loading coming soon!")

    # Show selection count
    count = len(selected_strategies)
    if count < 2:
        st.warning(f"⚠️ Select at least 2 strategies ({count}/2)")
    elif count > 10:
        st.error(f"❌ Maximum 10 strategies ({count}/10)")
    else:
        st.success(f"✅ {count} strategies selected")


def render_time_horizon_controls() -> None:
    """Render time horizon selection controls."""
    st.markdown("### ⏱️ Time Horizon")

    # Single selection radio buttons
    time_options = ["1W", "1M", "3M", "6M", "1Y", "ALL"]
    time_labels = {
        "1W": "1 Week",
        "1M": "1 Month",
        "3M": "3 Months",
        "6M": "6 Months",
        "1Y": "1 Year",
        "ALL": "All Time"
    }

    selected = st.radio(
        "Select period",
        options=time_options,
        format_func=lambda x: time_labels[x],
        index=3,  # Default to 6M
        key="sidebar_time_horizon"
    )

    st.session_state.time_horizon = selected

    # Option to compare multiple horizons
    compare_multiple = st.checkbox(
        "Compare multiple periods",
        value=False,
        help="Enable to compare performance across multiple time horizons"
    )

    if compare_multiple:
        multi_select = st.multiselect(
            "Select additional periods",
            options=[t for t in time_options if t != selected],
            format_func=lambda x: time_labels[x],
            key="multi_time_horizons"
        )
        if multi_select:
            st.session_state.multi_time_horizons = [selected] + multi_select


def render_risk_filters() -> None:
    """Render risk filter controls."""
    st.markdown("### 🎚️ Risk Filters")

    # Max drawdown filter
    max_dd = st.slider(
        "Max Acceptable Drawdown",
        min_value=0,
        max_value=50,
        value=30,
        step=5,
        format="%d%%",
        help="Filter strategies by maximum drawdown threshold",
        key="max_dd_filter"
    )
    st.session_state.max_dd_threshold = max_dd

    # Min Sharpe ratio filter
    min_sharpe = st.slider(
        "Minimum Sharpe Ratio",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Filter strategies by minimum Sharpe ratio",
        key="min_sharpe_filter"
    )
    st.session_state.min_sharpe = min_sharpe

    # Asset class filter
    asset_classes = ["BTC", "ETH", "Altcoins", "Stablecoins", "Mixed"]
    selected_assets = st.multiselect(
        "Asset Classes",
        options=asset_classes,
        default=["BTC", "ETH"],
        help="Filter strategies by underlying asset class",
        key="asset_class_filter"
    )
    st.session_state.asset_filter = selected_assets

    # Apply filters button
    if st.button("Apply Filters", type="primary", use_container_width=True):
        st.info("Filters applied! Re-run analysis to see filtered results.")


def render_parameter_explorer() -> None:
    """Render parameter exploration controls."""
    st.markdown("### ⚙️ Parameter Explorer")

    if not st.session_state.get("selected_strategies"):
        st.info("Select strategies to adjust parameters")
        return

    # Strategy selector for parameter adjustment
    strategy_to_adjust = st.selectbox(
        "Select strategy",
        options=st.session_state.selected_strategies,
        key="param_strategy_selector"
    )

    if strategy_to_adjust:
        st.markdown(f"**{strategy_to_adjust}**")

        # Get strategy parameters (this would come from strategy config)
        params = get_strategy_parameters(strategy_to_adjust)

        # Render controls for each parameter
        for param_name, param_config in params.items():
            render_parameter_control(
                strategy_to_adjust,
                param_name,
                param_config
            )

        # Apply button
        if st.button(
            "Apply Parameters",
            type="primary",
            use_container_width=True,
            key="apply_params_btn"
        ):
            st.success("Parameters updated! Re-run analysis to see changes.")


def render_parameter_control(
    strategy: str,
    param_name: str,
    param_config: Dict[str, Any]
) -> None:
    """
    Render a single parameter control based on its configuration.

    Args:
        strategy: Strategy name
        param_name: Parameter name
        param_config: Parameter configuration dict
    """
    param_type = param_config.get("type", "slider")
    label = param_config.get("label", param_name)
    key = f"{strategy}_{param_name}"

    if param_type == "slider":
        value = st.slider(
            label,
            min_value=param_config.get("min", 0),
            max_value=param_config.get("max", 100),
            value=param_config.get("default", 50),
            step=param_config.get("step", 1),
            help=param_config.get("help", ""),
            key=key
        )
    elif param_type == "select":
        value = st.selectbox(
            label,
            options=param_config.get("options", []),
            index=param_config.get("default_index", 0),
            help=param_config.get("help", ""),
            key=key
        )
    elif param_type == "number":
        value = st.number_input(
            label,
            min_value=param_config.get("min"),
            max_value=param_config.get("max"),
            value=param_config.get("default"),
            step=param_config.get("step", 1),
            help=param_config.get("help", ""),
            key=key
        )
    else:
        value = None

    # Store in session state
    if value is not None:
        if "strategy_params" not in st.session_state:
            st.session_state.strategy_params = {}
        if strategy not in st.session_state.strategy_params:
            st.session_state.strategy_params[strategy] = {}
        st.session_state.strategy_params[strategy][param_name] = value


def get_strategy_parameters(strategy: str) -> Dict[str, Dict[str, Any]]:
    """
    Get parameter configuration for a strategy.

    Args:
        strategy: Strategy name

    Returns:
        Dictionary of parameter configurations
    """
    # This is a placeholder - in production, load from strategy config
    default_params = {
        "window": {
            "type": "slider",
            "label": "Window Period",
            "min": 5,
            "max": 100,
            "default": 20,
            "step": 5,
            "help": "Lookback window for calculations"
        },
        "threshold": {
            "type": "slider",
            "label": "Signal Threshold",
            "min": 1.0,
            "max": 3.0,
            "default": 2.0,
            "step": 0.1,
            "help": "Threshold for entry signals"
        },
        "stop_loss": {
            "type": "select",
            "label": "Stop Loss %",
            "options": ["2%", "5%", "10%", "None"],
            "default_index": 1,
            "help": "Stop loss percentage"
        }
    }

    return default_params


def render_export_section() -> None:
    """Render export options in sidebar."""
    st.markdown("### 📤 Export Options")

    if not st.session_state.get("comparison_results"):
        st.info("Run analysis to enable export")
        return

    # Export format selection
    export_formats = st.multiselect(
        "Select formats",
        options=["PDF Report", "HTML", "CSV Data", "JSON"],
        default=["PDF Report"],
        key="export_formats"
    )

    # Include options
    include_charts = st.checkbox("Include charts", value=True)
    include_trades = st.checkbox("Include trade details", value=False)

    # Download button
    if st.button("📥 Download", type="primary", use_container_width=True):
        st.info("Export functionality will generate files with selected options")


def render_parameter_controls(strategies: List[str]) -> None:
    """
    Render parameter comparison view.

    Args:
        strategies: List of strategy names
    """
    st.markdown("### Parameter Comparison")

    if not strategies:
        st.info("Select strategies to view parameters")
        return

    # Create comparison table of parameters
    for strategy in strategies:
        with st.expander(f"📊 {strategy}", expanded=False):
            params = get_strategy_parameters(strategy)

            for param_name, param_config in params.items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{param_config.get('label', param_name)}**")
                with col2:
                    # Get current value from session state or use default
                    current_value = (
                        st.session_state.strategy_params
                        .get(strategy, {})
                        .get(param_name, param_config.get("default"))
                    )
                    st.markdown(f"`{current_value}`")


if __name__ == "__main__":
    # Validation function
    import sys

    print("🔍 Validating sidebar.py...")

    all_validation_failures = []
    total_tests = 0

    # Test 1: Parameter configuration retrieval
    total_tests += 1
    try:
        params = get_strategy_parameters("test_strategy")
        expected_keys = {"window", "threshold", "stop_loss"}
        if not expected_keys.issubset(params.keys()):
            all_validation_failures.append(
                f"Parameter config: Expected keys {expected_keys}, got {params.keys()}"
            )
    except Exception as e:
        all_validation_failures.append(f"Parameter config test failed: {e}")

    # Test 2: Parameter types validation
    total_tests += 1
    try:
        params = get_strategy_parameters("test_strategy")
        for param_name, param_config in params.items():
            if "type" not in param_config:
                all_validation_failures.append(
                    f"Parameter {param_name}: Missing 'type' field"
                )
            if "label" not in param_config:
                all_validation_failures.append(
                    f"Parameter {param_name}: Missing 'label' field"
                )
    except Exception as e:
        all_validation_failures.append(f"Parameter validation test failed: {e}")

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
        print("Sidebar component is validated and ready for integration")
        sys.exit(0)
