"""
Interactive Plotly Visualization Module

This module generates interactive time series visualizations for strategy analysis
using Plotly.js with sliding window support for exploring different time periods.

**Purpose**: Create interactive HTML charts allowing users to:
- Select any strategy from dropdown
- Slide through different time periods with a range selector
- View equity curves vs buy-and-hold benchmark
- See trade markers on price charts
- Compare performance statistics dynamically

**Key Features**:
- Plotly.js interactive charts with zoom/pan/hover
- Sliding window for time period exploration
- Strategy selector dropdown
- Trade markers overlay on price charts
- Dynamic performance statistics table

**Third-party packages**:
- plotly: https://plotly.com/python/
- pandas: https://pandas.pydata.org/docs/

**Sample Usage**:
```python
from crypto_trader.reports.formatters.plotly_interactive import generate_interactive_section_html

interactive_html = generate_interactive_section_html(
    all_results={'strategy_name': {horizon: backtest_result}},
    horizons=[HorizonConfig('30d', 30, '30 days'), ...],
    symbol='BTC/USDT'
)
```

**Expected Output**:
HTML string with embedded Plotly charts and interactive controls.

Created for interactive strategy analysis feature.
"""

import sys
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


@dataclass
class HorizonConfig:
    """Configuration for time horizon."""
    name: str
    days: int
    description: str


def create_interactive_equity_chart(
    strategy_data: Dict[str, Any],
    strategy_name: str,
    symbol: str
) -> go.Figure:
    """
    Create interactive equity curve chart with sliding window.

    Args:
        strategy_data: Dict with 'timestamps', 'equity_strategy', 'equity_buyhold'
        strategy_name: Name of the strategy
        symbol: Trading pair symbol

    Returns:
        Plotly Figure object with equity curves and rangeslider
    """
    fig = go.Figure()

    timestamps = strategy_data['timestamps']
    equity_strategy = strategy_data['equity_strategy']
    equity_buyhold = strategy_data['equity_buyhold']

    # Strategy equity curve
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=equity_strategy,
        name=strategy_name,
        line=dict(color='#2E86DE', width=3),
        hovertemplate='<b>Strategy</b><br>Value: $%{y:,.2f}<br>Date: %{x}<extra></extra>'
    ))

    # Buy-and-hold benchmark
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=equity_buyhold,
        name='Buy & Hold',
        line=dict(color='#EE5A6F', width=2, dash='dash'),
        hovertemplate='<b>Buy & Hold</b><br>Value: $%{y:,.2f}<br>Date: %{x}<extra></extra>'
    ))

    # Layout configuration
    fig.update_layout(
        title=dict(
            text=f'{strategy_name} vs Buy-and-Hold ({symbol})',
            font=dict(size=20, color='#1a1a1a')
        ),
        xaxis=dict(
            title='Date',
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor='lightgray'
            ),
            type='date'
        ),
        yaxis=dict(
            title='Portfolio Value ($)',
            tickformat='$,.0f'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#BCCCDC',
            borderwidth=1
        )
    )

    return fig


def create_trade_markers_chart(
    price_data: Dict[str, Any],
    trades: List[Dict[str, Any]],
    symbol: str
) -> go.Figure:
    """
    Create price chart with buy/sell trade markers.

    Args:
        price_data: Dict with 'timestamps' and 'prices'
        trades: List of trade dicts with 'time', 'side', 'price'
        symbol: Trading pair symbol

    Returns:
        Plotly Figure object with price line and trade markers
    """
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=price_data['timestamps'],
        y=price_data['prices'],
        name='Price',
        line=dict(color='#34495e', width=1.5),
        hovertemplate='<b>Price</b><br>$%{y:,.2f}<br>%{x}<extra></extra>'
    ))

    # Separate buy and sell trades
    buy_trades = [t for t in trades if t.get('side', '').lower() == 'buy']
    sell_trades = [t for t in trades if t.get('side', '').lower() == 'sell']

    if buy_trades:
        # Use entry_time and entry_price for buy signals
        buy_times = [t['entry_time'] for t in buy_trades]
        buy_prices = [t['entry_price'] for t in buy_trades]

        fig.add_trace(go.Scatter(
            x=buy_times,
            y=buy_prices,
            mode='markers',
            name='Buy',
            marker=dict(
                symbol='triangle-up',
                size=14,
                color='green',
                line=dict(color='darkgreen', width=1)
            ),
            hovertemplate='<b>BUY</b><br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
        ))

    if sell_trades:
        # Use exit_time and exit_price for sell signals
        sell_times = [t['exit_time'] for t in sell_trades]
        sell_prices = [t['exit_price'] for t in sell_trades]

        fig.add_trace(go.Scatter(
            x=sell_times,
            y=sell_prices,
            mode='markers',
            name='Sell',
            marker=dict(
                symbol='triangle-down',
                size=14,
                color='red',
                line=dict(color='darkred', width=1)
            ),
            hovertemplate='<b>SELL</b><br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Price with Trade Signals',
            font=dict(size=18, color='#1a1a1a')
        ),
        xaxis=dict(
            title='Date',
            type='date'
        ),
        yaxis=dict(
            title='Price ($)',
            tickformat='$,.2f'
        ),
        hovermode='closest',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#BCCCDC',
            borderwidth=1
        )
    )

    return fig


def create_multi_strategy_selector(
    all_results: Dict[str, Dict],
    symbol: str
) -> go.Figure:
    """
    Create interactive chart with dropdown selector for all strategies.

    Args:
        all_results: Dict mapping strategy names to their data
        symbol: Trading pair symbol

    Returns:
        Plotly Figure with updatemenus for strategy selection
    """
    fig = go.Figure()

    strategy_names = sorted(all_results.keys())

    # Add traces for all strategies (initially all visible)
    for strategy_name in strategy_names:
        strategy_data = all_results[strategy_name]

        if 'timestamps' not in strategy_data or 'equity_strategy' not in strategy_data:
            continue

        # Strategy trace
        fig.add_trace(go.Scatter(
            x=strategy_data['timestamps'],
            y=strategy_data['equity_strategy'],
            name=f'{strategy_name}',
            line=dict(width=3),
            visible=True,
            hovertemplate=f'<b>{strategy_name}</b><br>Value: $%{{y:,.2f}}<br>%{{x}}<extra></extra>'
        ))

        # Buy-hold trace
        fig.add_trace(go.Scatter(
            x=strategy_data['timestamps'],
            y=strategy_data['equity_buyhold'],
            name=f'{strategy_name} - Buy & Hold',
            line=dict(width=2, dash='dash'),
            visible=True,
            hovertemplate='<b>Buy & Hold</b><br>Value: $%{y:,.2f}<br>%{x}<extra></extra>'
        ))

    # Create dropdown buttons
    buttons = []

    for i, strategy_name in enumerate(strategy_names):
        # Create visibility array: show only selected strategy's traces
        visibility = [False] * (len(strategy_names) * 2)
        visibility[i * 2] = True  # Strategy trace
        visibility[i * 2 + 1] = True  # Buy-hold trace

        buttons.append(
            dict(
                label=strategy_name,
                method='update',
                args=[
                    {'visible': visibility},
                    {'title': f'{strategy_name} vs Buy-and-Hold ({symbol})'}
                ]
            )
        )

    # Add "Show All" option
    buttons.insert(0, dict(
        label='Show All Strategies',
        method='update',
        args=[
            {'visible': [True] * (len(strategy_names) * 2)},
            {'title': f'All Strategies vs Buy-and-Hold ({symbol})'}
        ]
    ))

    # Update layout with dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction='down',
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.02,
                xanchor='left',
                y=1.15,
                yanchor='top',
                bgcolor='white',
                bordercolor='#BCCCDC',
                borderwidth=1
            )
        ],
        annotations=[
            dict(
                text='<b>Select Strategy:</b>',
                x=0,
                xref='paper',
                y=1.12,
                yref='paper',
                align='left',
                showarrow=False,
                font=dict(size=14)
            )
        ],
        xaxis=dict(
            title='Date',
            rangeslider=dict(visible=True, thickness=0.05),
            type='date'
        ),
        yaxis=dict(
            title='Portfolio Value ($)',
            tickformat='$,.0f'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=550,
        margin=dict(t=100)
    )

    return fig


def create_performance_stats_table(
    strategy_metrics: Dict[str, float],
    buyhold_metrics: Dict[str, float]
) -> str:
    """
    Create HTML table with performance statistics comparison.

    Args:
        strategy_metrics: Dict with strategy performance metrics
        buyhold_metrics: Dict with buy-hold performance metrics

    Returns:
        HTML string for statistics table
    """
    def format_pct(value: float) -> str:
        """Format percentage with color."""
        color = 'green' if value >= 0 else 'red'
        sign = '+' if value >= 0 else ''
        return f'<span style="color:{color}; font-weight:bold;">{sign}{value:.2f}%</span>'

    def format_num(value: float, decimals: int = 2) -> str:
        """Format number."""
        return f'{value:.{decimals}f}'

    # Extract metrics
    strategy_return = strategy_metrics.get('total_return', 0) * 100
    buyhold_return = buyhold_metrics.get('total_return', 0) * 100
    diff_return = strategy_return - buyhold_return

    strategy_sharpe = strategy_metrics.get('sharpe_ratio', 0)
    buyhold_sharpe = buyhold_metrics.get('sharpe_ratio', 0)
    diff_sharpe = strategy_sharpe - buyhold_sharpe

    strategy_dd = strategy_metrics.get('max_drawdown', 0) * 100
    buyhold_dd = buyhold_metrics.get('max_drawdown', 0) * 100
    diff_dd = strategy_dd - buyhold_dd  # Less negative is better

    win_rate = strategy_metrics.get('win_rate', 0) * 100
    total_trades = strategy_metrics.get('total_trades', 0)
    profit_factor = strategy_metrics.get('profit_factor', 0)

    html = f'''
    <table class="stats-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Strategy</th>
                <th>Buy-Hold</th>
                <th>Difference</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>Total Return</b></td>
                <td>{format_pct(strategy_return)}</td>
                <td>{format_pct(buyhold_return)}</td>
                <td>{format_pct(diff_return)}</td>
            </tr>
            <tr>
                <td><b>Sharpe Ratio</b></td>
                <td>{format_num(strategy_sharpe)}</td>
                <td>{format_num(buyhold_sharpe)}</td>
                <td style="color:{'green' if diff_sharpe >= 0 else 'red'}; font-weight:bold;">
                    {'+' if diff_sharpe >= 0 else ''}{format_num(diff_sharpe)}
                </td>
            </tr>
            <tr>
                <td><b>Max Drawdown</b></td>
                <td>{format_pct(strategy_dd)}</td>
                <td>{format_pct(buyhold_dd)}</td>
                <td style="color:{'green' if diff_dd > 0 else 'red'}; font-weight:bold;">
                    {'+' if diff_dd > 0 else ''}{format_num(diff_dd, 2)}%
                </td>
            </tr>
            <tr>
                <td><b>Win Rate</b></td>
                <td>{format_pct(win_rate)}</td>
                <td>N/A</td>
                <td>-</td>
            </tr>
            <tr>
                <td><b>Total Trades</b></td>
                <td>{int(total_trades)}</td>
                <td>0</td>
                <td>-</td>
            </tr>
            <tr>
                <td><b>Profit Factor</b></td>
                <td>{format_num(profit_factor)}</td>
                <td>N/A</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>
    '''

    return html


def create_correlation_heatmap(
    correlation_df: pd.DataFrame,
    title: str = 'Cross-Asset Correlation Matrix'
) -> go.Figure:
    """
    Create interactive correlation heatmap.

    PHASE 2: Visualize cross-pair correlation structure.

    Args:
        correlation_df: DataFrame with correlation matrix (pairs x pairs)
        title: Chart title

    Returns:
        Plotly Figure with correlation heatmap
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_df.values,
        x=correlation_df.columns.tolist(),
        y=correlation_df.index.tolist(),
        colorscale='RdBu_r',  # Diverging colorscale (red-white-blue)
        zmid=0,  # Center colorscale at 0
        zmin=-1,
        zmax=1,
        text=correlation_df.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(
            title='Correlation',
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
        ),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1a1a1a')),
        xaxis=dict(title='', side='bottom'),
        yaxis=dict(title=''),
        template='plotly_white',
        height=400,
        margin=dict(l=100, r=100, t=80, b=80)
    )

    return fig


def create_risk_contribution_chart(
    risk_contribution: Dict[str, float],
    title: str = 'Portfolio Risk Contribution by Asset'
) -> go.Figure:
    """
    Create risk contribution bar chart.

    PHASE 2: Show how much each asset contributes to portfolio volatility.

    Args:
        risk_contribution: Dict mapping pair symbol to risk contribution (%)
        title: Chart title

    Returns:
        Plotly Figure with horizontal bar chart
    """
    pairs = list(risk_contribution.keys())
    contributions = [risk_contribution[pair] for pair in pairs]

    # Color bars by contribution level
    colors = ['#EE5A6F' if c > 60 else '#F79F1F' if c > 40 else '#2E86DE' for c in contributions]

    fig = go.Figure(data=[
        go.Bar(
            x=contributions,
            y=pairs,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{c:.1f}%' for c in contributions],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Risk Contribution: %{x:.1f}%<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1a1a1a')),
        xaxis=dict(title='Risk Contribution (%)', range=[0, max(contributions) * 1.2]),
        yaxis=dict(title=''),
        template='plotly_white',
        height=max(200, len(pairs) * 60),  # Dynamic height based on number of pairs
        showlegend=False
    )

    return fig


def create_advanced_metrics_dashboard(
    metrics: Dict[str, float],
    title: str = 'Advanced Risk Metrics'
) -> go.Figure:
    """
    Create dashboard showing Phase 2 advanced metrics.

    PHASE 2: Display Omega ratio, Tail ratio, Ulcer index, Max consecutive DD days.

    Args:
        metrics: Dict with keys: omega_ratio, tail_ratio, ulcer_index, max_consecutive_drawdown_days
        title: Dashboard title

    Returns:
        Plotly Figure with metric cards
    """
    from plotly.subplots import make_subplots

    # Extract metrics
    omega = metrics.get('omega_ratio', 0.0)
    tail = metrics.get('tail_ratio', 1.0)
    ulcer = metrics.get('ulcer_index', 0.0)
    max_consec_dd = metrics.get('max_consecutive_drawdown_days', 0)

    # Create subplots with 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Omega Ratio', 'Tail Ratio', 'Ulcer Index', 'Max Consecutive DD Days'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Omega Ratio (>1 is good, >2 is excellent)
    omega_color = 'green' if omega > 2 else 'orange' if omega > 1 else 'red'
    fig.add_trace(go.Indicator(
        mode='number+delta',
        value=omega,
        number={'valueformat': '.2f', 'font': {'size': 48, 'color': omega_color}},
        delta={'reference': 1.0, 'valueformat': '.2f'},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=1)

    # Tail Ratio (>1 means right tail > left tail, asymmetry)
    tail_color = 'green' if tail > 1.2 else 'orange' if tail > 0.8 else 'red'
    fig.add_trace(go.Indicator(
        mode='number+delta',
        value=tail,
        number={'valueformat': '.2f', 'font': {'size': 48, 'color': tail_color}},
        delta={'reference': 1.0, 'valueformat': '.2f'},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=2)

    # Ulcer Index (lower is better, <5% is good)
    ulcer_color = 'green' if ulcer < 5 else 'orange' if ulcer < 10 else 'red'
    fig.add_trace(go.Indicator(
        mode='number',
        value=ulcer,
        number={'valueformat': '.2f', 'suffix': '%', 'font': {'size': 48, 'color': ulcer_color}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=2, col=1)

    # Max Consecutive DD Days (fewer is better)
    dd_color = 'green' if max_consec_dd < 30 else 'orange' if max_consec_dd < 90 else 'red'
    fig.add_trace(go.Indicator(
        mode='number',
        value=max_consec_dd,
        number={'valueformat': 'd', 'suffix': ' days', 'font': {'size': 40, 'color': dd_color}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1a1a1a')),
        template='plotly_white',
        height=500,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig


def generate_interactive_section_html(
    all_results: Dict[str, Dict],
    symbol: str = 'BTC/USDT'
) -> str:
    """
    Generate complete interactive section HTML with all charts and controls.

    Args:
        all_results: Dict mapping strategy names to their complete data:
            {
                'strategy_name': {
                    'timestamps': [...],
                    'equity_strategy': [...],
                    'equity_buyhold': [...],
                    'prices': [...],
                    'trades': [...],
                    'metrics': {...}
                }
            }
        symbol: Trading pair symbol

    Returns:
        Complete HTML string for interactive section
    """
    try:
        logger.info(f"Generating interactive section for {len(all_results)} strategies")

        # Create multi-strategy selector chart
        equity_fig = create_multi_strategy_selector(all_results, symbol)
        equity_html = equity_fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            config={'responsive': True, 'displayModeBar': True}
        )

        # Get first strategy for initial display
        first_strategy = sorted(all_results.keys())[0] if all_results else None

        price_html = ""
        stats_html = ""

        if first_strategy and first_strategy in all_results:
            strategy_data = all_results[first_strategy]

            # Create price chart with trade markers if available
            if 'prices' in strategy_data and 'trades' in strategy_data:
                price_data = {
                    'timestamps': strategy_data['timestamps'],
                    'prices': strategy_data['prices']
                }
                price_fig = create_trade_markers_chart(
                    price_data,
                    strategy_data['trades'],
                    symbol
                )
                price_html = price_fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={'responsive': True}
                )

            # Create stats table if metrics available
            if 'metrics' in strategy_data and 'buyhold_metrics' in strategy_data:
                stats_html = create_performance_stats_table(
                    strategy_data['metrics'],
                    strategy_data['buyhold_metrics']
                )

        # Combine into complete section
        html = f'''
        <div class="interactive-section">
            <h2>📊 Interactive Strategy Explorer</h2>
            <p><em>Select a strategy and use the range slider to explore different time periods</em></p>

            <div class="interactive-charts">
                <h3>Equity Curve Comparison</h3>
                {equity_html}

                {'<h3>Price Chart with Trade Signals</h3>' + price_html if price_html else ''}

                {'<h3>Performance Statistics</h3>' + stats_html if stats_html else ''}
            </div>
        </div>
        '''

        logger.success("Interactive section generated successfully")
        return html

    except Exception as e:
        logger.error(f"Failed to generate interactive section: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return '<div class="interactive-section"><p>⚠️ Failed to generate interactive visualizations</p></div>'


if __name__ == "__main__":
    """Validation block for plotly_interactive module."""

    all_validation_failures = []
    total_tests = 0

    print("🔍 Validating plotly_interactive module...\n")

    # Test 1: Create sample data and equity chart
    total_tests += 1
    print("Test 1: Create interactive equity chart")
    try:
        timestamps = pd.date_range('2024-01-01', periods=30, freq='D')
        strategy_data = {
            'timestamps': timestamps,
            'equity_strategy': [10000 + i * 50 for i in range(30)],
            'equity_buyhold': [10000 + i * 30 for i in range(30)]
        }

        fig = create_interactive_equity_chart(strategy_data, 'TestStrategy', 'BTC/USDT')

        if fig is None or not isinstance(fig, go.Figure):
            all_validation_failures.append("Equity chart creation failed: Invalid figure object")
        else:
            print("  ✓ Interactive equity chart created successfully")
            print(f"  ✓ Figure has {len(fig.data)} traces")
    except Exception as e:
        all_validation_failures.append(f"Test 1 failed: {e}")

    # Test 2: Create trade markers chart
    total_tests += 1
    print("\nTest 2: Create trade markers chart")
    try:
        price_data = {
            'timestamps': timestamps,
            'prices': [42000 + i * 100 for i in range(30)]
        }
        trades = [
            {'time': timestamps[5], 'side': 'buy', 'price': 42500},
            {'time': timestamps[15], 'side': 'sell', 'price': 43500},
        ]

        fig = create_trade_markers_chart(price_data, trades, 'BTC/USDT')

        if fig is None or not isinstance(fig, go.Figure):
            all_validation_failures.append("Trade markers chart creation failed")
        else:
            print("  ✓ Trade markers chart created successfully")
            print(f"  ✓ Figure has {len(fig.data)} traces (price + buy + sell)")
    except Exception as e:
        all_validation_failures.append(f"Test 2 failed: {e}")

    # Test 3: Create performance stats table
    total_tests += 1
    print("\nTest 3: Create performance stats table")
    try:
        strategy_metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.8,
            'max_drawdown': -0.05,
            'win_rate': 0.67,
            'total_trades': 10,
            'profit_factor': 2.3
        }
        buyhold_metrics = {
            'total_return': -0.03,
            'sharpe_ratio': -0.5,
            'max_drawdown': -0.12
        }

        html_table = create_performance_stats_table(strategy_metrics, buyhold_metrics)

        if not html_table or '<table' not in html_table:
            all_validation_failures.append("Stats table generation failed")
        elif '+15.00%' not in html_table:  # Check if strategy return is formatted
            all_validation_failures.append("Stats table missing expected return value")
        else:
            print("  ✓ Performance stats table created successfully")
            print(f"  ✓ Table length: {len(html_table)} characters")
    except Exception as e:
        all_validation_failures.append(f"Test 3 failed: {e}")

    # Test 4: Create multi-strategy selector
    total_tests += 1
    print("\nTest 4: Create multi-strategy selector")
    try:
        all_results = {
            'Strategy1': {
                'timestamps': timestamps,
                'equity_strategy': [10000 + i * 50 for i in range(30)],
                'equity_buyhold': [10000 + i * 30 for i in range(30)]
            },
            'Strategy2': {
                'timestamps': timestamps,
                'equity_strategy': [10000 + i * 40 for i in range(30)],
                'equity_buyhold': [10000 + i * 30 for i in range(30)]
            }
        }

        fig = create_multi_strategy_selector(all_results, 'BTC/USDT')

        if fig is None or not isinstance(fig, go.Figure):
            all_validation_failures.append("Multi-strategy selector creation failed")
        elif not fig.layout.updatemenus:
            all_validation_failures.append("Strategy selector missing updatemenus")
        else:
            print("  ✓ Multi-strategy selector created successfully")
            print(f"  ✓ Dropdown has {len(fig.layout.updatemenus[0].buttons)} options")
    except Exception as e:
        all_validation_failures.append(f"Test 4 failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("plotly_interactive module is validated and ready for use")
        sys.exit(0)
