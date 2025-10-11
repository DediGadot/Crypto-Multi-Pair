# Crypto Strategy Comparison Dashboard - UI Design Specification

## Overview

This document provides a comprehensive UI/UX design specification for the Streamlit-based Crypto Strategy Comparison Dashboard.

## Table of Contents

1. [Page Layout](#page-layout)
2. [UI Components](#ui-components)
3. [Interaction Flow](#interaction-flow)
4. [Visualizations](#visualizations)
5. [Data Tables](#data-tables)
6. [Controls & Filters](#controls--filters)
7. [Responsive Design](#responsive-design)
8. [Accessibility](#accessibility)

## Page Layout

### Overall Structure

```
┌─────────────────────────────────────────────────────────────┐
│  Header: Title + Quick Metrics                              │
├──────────────┬──────────────────────────────────────────────┤
│              │  Performance Overview                         │
│  Sidebar     │  - Equity Curves (main chart)                │
│  (300px)     │  - Key Metrics Table                         │
│              │                                               │
│  • Strategy  │  Detailed Analysis (Tabbed Interface)        │
│    Selection │  - Tab 1: Charts                             │
│  • Filters   │  - Tab 2: Drawdowns                          │
│  • Parameters│  - Tab 3: Trades                             │
│  • Export    │  - Tab 4: Parameters                         │
│              │  - Tab 5: Export                             │
│              │                                               │
│              │  Insights & Recommendations                   │
│              │  - AI-generated insights                     │
└──────────────┴──────────────────────────────────────────────┘
```

### Layout Specifications

- **Page Width**: Wide mode (full width)
- **Sidebar Width**: 300px fixed
- **Main Content**: Flexible width
- **Minimum Width**: 1024px
- **Maximum Width**: Unlimited

## UI Components

### 1. Header Section

**Components:**
- Main title with icon
- Last updated timestamp
- Strategy count metric
- Quick action buttons

**Styling:**
- Background: White
- Border-bottom: 1px solid #e0e0e0
- Padding: 20px
- Shadow: 0 2px 4px rgba(0,0,0,0.1)

### 2. Sidebar

**Sections:**

#### A. Strategy Selection
- Checkboxes for each available strategy
- Visual indicators (icons)
- Selection counter
- "Load Custom Strategy" button

#### B. Time Horizon
- Radio buttons for standard periods
- Option to compare multiple horizons
- Custom date range picker

#### C. Risk Filters
- Max Drawdown slider (0-50%)
- Min Sharpe slider (0-3.0)
- Asset class multi-select
- Apply filters button

#### D. Parameter Explorer
- Strategy dropdown selector
- Dynamic parameter controls based on selected strategy
- Apply parameters button

#### E. Export Options
- Format checkboxes (PDF, HTML, CSV, JSON)
- Include options (charts, trades, etc.)
- Download button

### 3. Main Content Area

#### A. Quick Controls Bar
- Strategy multi-select dropdown
- Time horizon pills
- Run Analysis button (primary action)

#### B. Performance Overview
- Large equity curve chart (full width)
- Metrics summary table
- Summary statistics cards

#### C. Tabbed Interface
- 5 tabs for detailed analysis
- Consistent padding and spacing
- Active tab highlighting

## Interaction Flow

### User Journey

```
Landing → Select Strategies → Choose Horizon → Run Analysis
    ↓
View Results → Explore Details → Adjust Parameters → Re-run
    ↓
Export Reports
```

### Key Interactions

1. **Strategy Selection**
   - Click checkbox to select/deselect
   - Visual feedback on selection
   - Validation messages (min 2, max 10)

2. **Time Horizon Selection**
   - Click radio button or pill
   - Immediate visual feedback
   - Option for custom range

3. **Run Analysis**
   - Button shows loading spinner
   - Progress indication
   - Success/error messages

4. **Chart Interactions**
   - Hover: Show tooltips with values
   - Click legend: Toggle strategy visibility
   - Range selector: Adjust time window
   - Zoom: Mouse wheel or pinch
   - Pan: Click and drag

5. **Table Interactions**
   - Click header: Sort by column
   - Hover row: Highlight effect
   - Click row: Expand for details (if applicable)

## Visualizations

### 1. Equity Curves (Main Dashboard)

**Type:** Multi-line time series chart

**Features:**
- Multiple strategies on same axes
- Distinct colors for each strategy
- Interactive legend
- Range selector (1M, 3M, 6M, YTD, 1Y, All)
- Zoom and pan
- Crosshair for value comparison
- Linear/Log scale toggle

**Plotly Configuration:**
```python
{
    "displayModeBar": True,
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "equity_curves",
        "height": 600,
        "width": 1200
    }
}
```

### 2. Drawdown Chart

**Type:** Area chart (filled to zero)

**Features:**
- Underwater plot showing drawdowns
- Shaded areas by severity
- Max drawdown markers
- Recovery period highlighting

### 3. Returns Distribution

**Type:** Box plot or Violin plot

**Features:**
- One box per strategy
- Show mean and std dev
- Quartile indicators
- Outlier markers

### 4. Rolling Metrics

**Type:** Multi-line time series

**Features:**
- Rolling Sharpe ratio
- Rolling volatility
- Rolling win rate
- Configurable window (30/60/90 days)

### 5. Correlation Matrix

**Type:** Heatmap

**Features:**
- Color scale (-1 to +1)
- Annotations showing values
- Interactive: click cell for detail

### 6. Risk-Return Scatter

**Type:** Scatter plot

**Features:**
- X-axis: Risk (volatility)
- Y-axis: Returns
- Bubble size: Trade count
- Quadrant lines
- Efficient frontier overlay

## Data Tables

### 1. Performance Metrics Table

**Columns:**
- Rank (1, 2, 3, ...)
- Strategy Name
- Total Return %
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown %
- Win Rate %
- Trade Count
- Volatility %
- Calmar Ratio

**Features:**
- Sortable by any column
- Color-coded values (green/red)
- Rank indicators (#1, #2, etc.)
- Expandable rows for details
- Export to CSV button

**Styling:**
```css
.dataframe {
    font-size: 14px;
    border: 1px solid #e0e0e0;
}

.dataframe thead {
    background-color: #1f77b4;
    color: white;
    font-weight: bold;
}

.dataframe tbody tr:hover {
    background-color: #f5f5f5;
}
```

### 2. Trade-Level Table

**Columns:**
- Date
- Symbol
- Side (LONG/SHORT)
- Entry Price
- Exit Price
- Return %
- Duration
- PnL $

**Features:**
- Pagination (20/50/100 per page)
- Filter by strategy, asset, date range
- Color-coded wins/losses
- Click row for detailed analysis
- Export to CSV

### 3. Detailed Metrics (Expandable)

**Additional Metrics:**
- Information Ratio
- Recovery Factor
- Expectancy
- Kelly Criterion %
- Value at Risk (VaR)
- Conditional VaR
- Max Consecutive Wins/Losses

## Controls & Filters

### 1. Strategy Multi-Select

**Component:** `st.multiselect()`

**Configuration:**
```python
st.multiselect(
    "Select Strategies (2-10)",
    options=available_strategies,
    format_func=lambda x: f"{get_icon(x)} {x}",
    help="Choose 2-10 strategies to compare",
    max_selections=10
)
```

### 2. Time Horizon Selector

**Component:** `st.radio()` with horizontal layout

**Options:**
- 1W, 1M, 3M, 6M, 1Y, ALL

**Styling:** Custom CSS to make pills

### 3. Risk Filters

**Components:**
- `st.slider()` for continuous values
- `st.multiselect()` for categorical filters

### 4. Parameter Controls

**Dynamic Generation:**
- Load parameter config from strategy
- Generate appropriate widget (slider/select/number)
- Store in session state
- Apply on button click

## Responsive Design

### Breakpoints

- **Desktop Large**: > 1440px - Full layout
- **Desktop**: 1024px - 1440px - Standard layout
- **Tablet**: 768px - 1024px - Sidebar collapses
- **Mobile**: < 768px - Stacked layout

### Mobile Optimizations

- Sidebar becomes collapsible menu
- Tables scroll horizontally
- Charts resize to fit width
- Touch-friendly controls
- Larger tap targets (min 44x44px)

## Accessibility

### WCAG 2.1 AA Compliance

**Color Contrast:**
- Text: Minimum 4.5:1 ratio
- Large text: Minimum 3:1 ratio
- Charts: Colorblind-friendly palette

**Keyboard Navigation:**
- All interactive elements accessible via Tab
- Enter/Space to activate buttons
- Arrow keys for sliders
- Escape to close modals

**Screen Reader Support:**
- ARIA labels on all controls
- Alt text for charts (via Plotly descriptions)
- Semantic HTML structure
- Skip to content link

**Focus Indicators:**
- Visible focus outline (2px solid #1f77b4)
- High contrast mode support

### Color Palette (Colorblind-Friendly)

**Strategy Colors:**
```python
COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]
```

**Semantic Colors:**
- Success: #28a745
- Warning: #ffc107
- Error: #dc3545
- Info: #17a2b8

## Performance Considerations

### Optimization Strategies

1. **Data Loading:**
   - Lazy load detailed data
   - Cache strategy data in session state
   - Pagination for large tables

2. **Chart Rendering:**
   - Limit data points (downsample if > 1000)
   - Use Plotly's streaming for real-time updates
   - Debounce interactive updates

3. **Session State:**
   - Minimize state variables
   - Clear unused data
   - Use callbacks for heavy computations

4. **Caching:**
   ```python
   @st.cache_data
   def load_strategy_data(strategy_name):
       # Expensive data loading
       return data
   ```

## Implementation Notes

### Streamlit Best Practices

1. **Layout:**
   - Use `st.columns()` for side-by-side content
   - Use `st.container()` for grouping
   - Use `st.expander()` for collapsible sections

2. **State Management:**
   - Initialize all state variables at app start
   - Use unique keys for all widgets
   - Handle state carefully in callbacks

3. **Performance:**
   - Cache data loading functions
   - Minimize reruns with `st.form()`
   - Use `st.spinner()` for long operations

4. **User Feedback:**
   - Always show loading indicators
   - Provide clear error messages
   - Confirm successful actions

## Future Enhancements

### Phase 2 Features

1. **Advanced Filtering:**
   - Save filter presets
   - Advanced query builder
   - Custom metric calculations

2. **Collaboration:**
   - Share comparison links
   - Comment on strategies
   - Team workspaces

3. **Notifications:**
   - Email reports
   - Slack integration
   - Alert on thresholds

4. **Machine Learning:**
   - Strategy recommendations
   - Anomaly detection
   - Predictive analytics

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Material Design](https://material.io/design)
