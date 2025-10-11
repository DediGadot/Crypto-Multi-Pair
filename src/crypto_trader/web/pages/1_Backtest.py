"""
Backtest Page - Run individual strategy backtests

This module provides an interactive interface for running backtests on individual
strategies with configurable parameters and real-time results visualization.

**Purpose**: Single-strategy backtesting interface with parameter configuration,
execution controls, and detailed results visualization.

**Features**:
- Strategy selection with parameter configuration
- Date range and symbol selection
- Real-time backtest execution
- Detailed results with interactive charts
- Trade-by-trade analysis
- Export individual results

**Third-party packages**:
- streamlit: https://docs.streamlit.io/
- plotly: https://plotly.com/python/
- pandas: https://pandas.pydata.org/docs/

**Sample Input**:
Run with: `streamlit run src/crypto_trader/web/app.py` (accessed via page navigation)

**Expected Output**:
Interactive backtest interface with strategy parameters, run button, and results
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from crypto_trader.analysis.metrics import MetricsCalculator
from crypto_trader.analysis.reporting import ReportGenerator
from crypto_trader.core.types import BacktestResult, PerformanceMetrics, Timeframe
from crypto_trader.strategies.registry import list_strategies

st.set_page_config(
    page_title="Backtest Strategy",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Strategy Backtesting")
st.markdown("### Configure and run individual strategy backtests")


def create_parameter_inputs(strategy_name: str) -> dict:
    """Create dynamic parameter input widgets based on strategy."""
    st.subheader("📊 Strategy Parameters")

    # Common parameters for all strategies
    params = {}

    col1, col2 = st.columns(2)

    with col1:
        # Symbol selection
        params["symbol"] = st.selectbox(
            "Symbol",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
            index=0,
        )

        # Timeframe
        params["timeframe"] = st.selectbox(
            "Timeframe",
            ["15m", "30m", "1h", "4h", "1d"],
            index=2,
        )

        # Initial capital
        params["initial_capital"] = st.number_input(
            "Initial Capital ($)",
            min_value=1000.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0,
        )

    with col2:
        # Date range
        params["start_date"] = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now(),
        )

        params["end_date"] = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now(),
        )

        # Position size
        params["position_size"] = st.slider(
            "Position Size (%)",
            min_value=10,
            max_value=100,
            value=95,
            step=5,
        )

    st.markdown("---")

    # Strategy-specific parameters
    st.subheader("⚙️ Strategy-Specific Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "MA" in strategy_name or "EMA" in strategy_name:
            params["fast_period"] = st.number_input(
                "Fast Period", min_value=5, max_value=50, value=10, step=1
            )
            params["slow_period"] = st.number_input(
                "Slow Period", min_value=20, max_value=200, value=50, step=5
            )

        elif "RSI" in strategy_name:
            params["rsi_period"] = st.number_input(
                "RSI Period", min_value=5, max_value=30, value=14, step=1
            )
            params["rsi_oversold"] = st.slider(
                "Oversold Level", min_value=10, max_value=40, value=30, step=5
            )
            params["rsi_overbought"] = st.slider(
                "Overbought Level", min_value=60, max_value=90, value=70, step=5
            )

        elif "MACD" in strategy_name:
            params["fast_period"] = st.number_input(
                "Fast EMA", min_value=5, max_value=20, value=12, step=1
            )
            params["slow_period"] = st.number_input(
                "Slow EMA", min_value=20, max_value=40, value=26, step=1
            )
            params["signal_period"] = st.number_input(
                "Signal Line", min_value=5, max_value=15, value=9, step=1
            )

        elif "Bollinger" in strategy_name:
            params["bb_period"] = st.number_input(
                "BB Period", min_value=10, max_value=50, value=20, step=5
            )
            params["bb_std"] = st.slider(
                "Standard Deviations", min_value=1.0, max_value=3.0, value=2.0, step=0.1
            )

    with col2:
        # Risk management
        st.markdown("**Risk Management**")
        params["stop_loss"] = st.number_input(
            "Stop Loss (%)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5,
        )
        params["take_profit"] = st.number_input(
            "Take Profit (%)",
            min_value=0.0,
            max_value=50.0,
            value=6.0,
            step=0.5,
        )

    with col3:
        # Fees and slippage
        st.markdown("**Trading Costs**")
        params["maker_fee"] = st.number_input(
            "Maker Fee (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.3f",
        )
        params["taker_fee"] = st.number_input(
            "Taker Fee (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.3f",
        )
        params["slippage"] = st.number_input(
            "Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.3f",
        )

    return params


def run_backtest_simulation(strategy_name: str, params: dict) -> BacktestResult:
    """
    Simulate backtest execution.
    In production, this would call the actual backtesting engine.
    """
    from crypto_trader.core.types import OrderSide, OrderType, Trade

    # Generate sample results for demonstration
    start_date = datetime.combine(params["start_date"], datetime.min.time())
    end_date = datetime.combine(params["end_date"], datetime.min.time())
    days = (end_date - start_date).days

    # Generate equity curve
    equity_curve = []
    initial_capital = params["initial_capital"]
    current_equity = initial_capital

    for day in range(days):
        timestamp = start_date + timedelta(days=day)
        # Simulate equity with realistic volatility
        daily_return = (
            pd.np.random.randn() * 0.02 + 0.0005
        )  # Mean positive return with noise
        current_equity *= 1 + daily_return
        equity_curve.append((timestamp, current_equity))

    final_capital = current_equity
    total_return = (final_capital - initial_capital) / initial_capital

    # Generate sample trades
    trades = []
    num_trades = pd.np.random.randint(30, 80)

    for i in range(num_trades):
        entry_time = start_date + timedelta(days=pd.np.random.randint(0, days))
        duration = pd.np.random.randint(1, 10)
        exit_time = entry_time + timedelta(days=duration)

        entry_price = 45000 + pd.np.random.randn() * 5000
        win = pd.np.random.random() > 0.4  # 60% win rate
        pnl_pct = pd.np.random.uniform(0.5, 5.0) if win else -pd.np.random.uniform(0.5, 3.0)
        exit_price = entry_price * (1 + pnl_pct / 100)

        quantity = 0.1
        pnl = (exit_price - entry_price) * quantity - (entry_price * quantity * params["taker_fee"] / 100 * 2)

        trade = Trade(
            symbol=params["symbol"],
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            side=OrderSide.BUY,
            quantity=quantity,
            pnl=pnl,
            pnl_percent=pnl_pct,
            fees=entry_price * quantity * params["taker_fee"] / 100 * 2,
            order_type=OrderType.MARKET,
        )
        trades.append(trade)

    # Calculate metrics
    calculator = MetricsCalculator()
    returns = calculator.calculate_returns_from_equity(equity_curve)
    metrics = calculator.calculate_all_metrics(
        returns=returns,
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=initial_capital,
    )

    # Create result
    result = BacktestResult(
        strategy_name=strategy_name,
        symbol=params["symbol"],
        timeframe=Timeframe.HOUR_4,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve,
    )

    return result


def display_backtest_results(result: BacktestResult):
    """Display comprehensive backtest results."""
    st.success("✅ Backtest completed successfully!")

    # Metrics overview
    st.subheader("📊 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Return",
            f"{result.metrics.total_return:.2%}",
            delta="Profit" if result.metrics.is_profitable() else "Loss",
        )
        st.metric("Sharpe Ratio", f"{result.metrics.sharpe_ratio:.2f}")

    with col2:
        st.metric("Max Drawdown", f"{result.metrics.max_drawdown:.2%}")
        st.metric("Sortino Ratio", f"{result.metrics.sortino_ratio:.2f}")

    with col3:
        st.metric("Win Rate", f"{result.metrics.win_rate:.2%}")
        st.metric("Profit Factor", f"{result.metrics.profit_factor:.2f}")

    with col4:
        st.metric("Total Trades", f"{result.metrics.total_trades}")
        st.metric("Expectancy", f"${result.metrics.expectancy:.2f}")

    st.markdown("---")

    # Quality badge
    quality = result.metrics.risk_adjusted_quality()
    quality_colors = {
        "EXCELLENT": "🟢",
        "GOOD": "🔵",
        "FAIR": "🟡",
        "POOR": "🔴",
    }
    st.markdown(
        f"### Strategy Quality: {quality_colors.get(quality, '')} **{quality}**"
    )

    st.markdown("---")

    # Charts
    tab1, tab2, tab3 = st.tabs(["📈 Equity Curve", "📉 Drawdown", "💹 Trade Analysis"])

    with tab1:
        # Equity curve
        reporter = ReportGenerator()
        equity_fig = reporter.create_equity_curve_chart(result)
        st.plotly_chart(equity_fig, use_container_width=True)

        # Monthly returns
        monthly_fig = reporter.create_monthly_returns_chart(result)
        st.plotly_chart(monthly_fig, use_container_width=True)

    with tab2:
        # Drawdown chart
        drawdown_fig = reporter.create_drawdown_chart(result)
        st.plotly_chart(drawdown_fig, use_container_width=True)

        # Drawdown statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Drawdown", f"{result.metrics.max_drawdown:.2%}")
        with col2:
            st.metric("Calmar Ratio", f"{result.metrics.calmar_ratio:.2f}")
        with col3:
            st.metric("Recovery Factor", f"{result.metrics.recovery_factor:.2f}")

    with tab3:
        # Trade statistics
        st.subheader("Trade Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Winning Trades", f"{result.metrics.winning_trades}")
            st.metric("Average Win", f"${result.metrics.avg_win:.2f}")
            st.metric("Max Consecutive Wins", f"{result.metrics.max_consecutive_wins}")

        with col2:
            st.metric("Losing Trades", f"{result.metrics.losing_trades}")
            st.metric("Average Loss", f"${result.metrics.avg_loss:.2f}")
            st.metric("Max Consecutive Losses", f"{result.metrics.max_consecutive_losses}")

        with col3:
            st.metric("Total Fees", f"${result.metrics.total_fees:.2f}")
            st.metric("Average Duration", f"{result.metrics.avg_trade_duration:.1f} min")
            st.metric("Expectancy", f"${result.metrics.expectancy:.2f}")

        # Recent trades table
        if result.trades:
            st.markdown("---")
            st.subheader("Recent Trades (Last 20)")

            trades_data = []
            for trade in result.trades[-20:]:
                trades_data.append(
                    {
                        "Entry Time": trade.entry_time.strftime("%Y-%m-%d %H:%M"),
                        "Exit Time": trade.exit_time.strftime("%Y-%m-%d %H:%M"),
                        "Side": trade.side.value,
                        "Entry Price": f"${trade.entry_price:,.2f}",
                        "Exit Price": f"${trade.exit_price:,.2f}",
                        "PnL": f"${trade.pnl:.2f}",
                        "PnL %": f"{trade.pnl_percent:.2f}%",
                        "Duration": f"{trade.duration_minutes:.0f} min",
                    }
                )

            df_trades = pd.DataFrame(trades_data)

            # Color code PnL
            st.dataframe(
                df_trades,
                use_container_width=True,
                height=400,
            )

    # Export section
    st.markdown("---")
    st.subheader("💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Export HTML Report", use_container_width=True):
            with st.spinner("Generating report..."):
                reporter = ReportGenerator()
                output_path = Path("exports") / f"backtest_{result.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                output_path.parent.mkdir(exist_ok=True)
                reporter.generate_html_report(result, str(output_path))
                st.success(f"Report saved to {output_path}")

    with col2:
        if st.button("Export Trades CSV", use_container_width=True):
            with st.spinner("Exporting..."):
                reporter = ReportGenerator()
                output_path = Path("exports") / f"trades_{result.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_path.parent.mkdir(exist_ok=True)
                reporter.export_to_csv(result, str(output_path))
                st.success(f"Trades saved to {output_path}")

    with col3:
        if st.button("Export JSON", use_container_width=True):
            with st.spinner("Exporting..."):
                reporter = ReportGenerator()
                output_path = Path("exports") / f"backtest_{result.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                output_path.parent.mkdir(exist_ok=True)
                reporter.export_to_json(result, str(output_path))
                st.success(f"Data saved to {output_path}")


def main():
    """Main backtest page logic."""
    # Strategy selection
    st.subheader("🎯 Select Strategy")

    strategies = list_strategies()
    if not strategies:
        strategies = {
            "MA Crossover": {"description": "Moving Average Crossover"},
            "RSI Mean Reversion": {"description": "RSI-based mean reversion"},
            "MACD Momentum": {"description": "MACD momentum strategy"},
            "Bollinger Breakout": {"description": "Bollinger Bands breakout"},
            "Triple EMA": {"description": "Triple EMA crossover"},
        }

    strategy_names = list(strategies.keys())
    selected_strategy = st.selectbox(
        "Strategy",
        strategy_names,
        help="Select a strategy to backtest",
    )

    # Show strategy description
    if selected_strategy in strategies:
        st.info(f"**Description:** {strategies[selected_strategy]['description']}")

    st.markdown("---")

    # Parameter configuration
    params = create_parameter_inputs(selected_strategy)

    st.markdown("---")

    # Run backtest button
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            # Validate parameters
            if params["start_date"] >= params["end_date"]:
                st.error("Start date must be before end date")
                return

            if params["initial_capital"] < 1000:
                st.error("Initial capital must be at least $1,000")
                return

            # Run backtest
            with st.spinner(f"Running backtest for {selected_strategy}..."):
                try:
                    result = run_backtest_simulation(selected_strategy, params)

                    # Store in session state
                    if "backtest_results" not in st.session_state:
                        st.session_state.backtest_results = {}
                    st.session_state.backtest_results[selected_strategy] = result

                    # Display results
                    st.markdown("---")
                    display_backtest_results(result)

                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")
                    st.exception(e)

    # Show previous results if available
    if "backtest_results" in st.session_state and st.session_state.backtest_results:
        st.markdown("---")
        st.subheader("📚 Previous Results")

        for strategy_name in st.session_state.backtest_results.keys():
            if st.button(f"View {strategy_name}", key=f"view_{strategy_name}"):
                result = st.session_state.backtest_results[strategy_name]
                display_backtest_results(result)


if __name__ == "__main__":
    main()
