# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Browser                              │
│                     http://localhost:8501                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP/WebSocket
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Streamlit Server                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     app.py (Main Entry)                   │  │
│  │  • Session State Management                               │  │
│  │  • Page Layout Orchestration                              │  │
│  │  • Component Coordination                                 │  │
│  └───────────┬────────────────────────────────┬───────────────┘  │
│              │                                │                  │
│     ┌────────▼───────┐              ┌────────▼────────┐        │
│     │  UI Components  │              │  Core Business  │        │
│     │                 │              │     Logic       │        │
│     │  ┌───────────┐ │              │  ┌───────────┐ │        │
│     │  │ sidebar.py│ │              │  │comparison │ │        │
│     │  │           │ │              │  │_engine.py │ │        │
│     │  │ • Strategy│ │              │  │           │ │        │
│     │  │   Select  │ │              │  │ • Compare │ │        │
│     │  │ • Filters │ │              │  │   Multiple│ │        │
│     │  │ • Params  │ │◄─────────────┼──│   Strats  │ │        │
│     │  │ • Export  │ │   Data Flow  │  │ • Calc    │ │        │
│     │  └───────────┘ │              │  │   Metrics │ │        │
│     │                 │              │  │ • Generate│ │        │
│     │  ┌───────────┐ │              │  │   Results │ │        │
│     │  │ charts.py │ │              │  └─────┬─────┘ │        │
│     │  │           │ │              │        │       │        │
│     │  │ • Equity  │ │              │  ┌─────▼─────┐ │        │
│     │  │   Curves  │ │              │  │ metrics_  │ │        │
│     │  │ • Drawdown│ │              │  │calculator │ │        │
│     │  │ • Returns │ │              │  │.py        │ │        │
│     │  │ • Rolling │ │              │  │           │ │        │
│     │  │ • Scatter │ │              │  │ • Return  │ │        │
│     │  │ • Heatmap │ │              │  │   Metrics │ │        │
│     │  └───────────┘ │              │  │ • Risk    │ │        │
│     │                 │              │  │   Metrics │ │        │
│     │  ┌───────────┐ │              │  │ • Sharpe  │ │        │
│     │  │ metrics_  │ │              │  │ • Sortino │ │        │
│     │  │display.py │ │              │  │ • Calmar  │ │        │
│     │  │           │ │              │  │ • Trade   │ │        │
│     │  │ • Perf    │ │              │  │   Stats   │ │        │
│     │  │   Table   │ │              │  └─────▲─────┘ │        │
│     │  │ • Trade   │ │              │        │       │        │
│     │  │   Table   │ │              │  ┌─────┴─────┐ │        │
│     │  │ • Summary │ │              │  │ strategy_ │ │        │
│     │  │   Cards   │ │              │  │loader.py  │ │        │
│     │  └───────────┘ │              │  │           │ │        │
│     │                 │              │  │ • Load    │ │        │
│     │  ┌───────────┐ │              │  │   Data    │ │        │
│     │  │ export.py │ │              │  │ • Cache   │ │        │
│     │  │           │ │              │  │   Results │ │        │
│     │  │ • PDF     │ │              │  │ • Mock    │ │        │
│     │  │ • HTML    │ │              │  │   Data    │ │        │
│     │  │ • CSV     │ │              │  └───────────┘ │        │
│     │  │ • JSON    │ │              │                 │        │
│     │  └───────────┘ │              │  ┌───────────┐ │        │
│     │                 │              │  │ utils.py  │ │        │
│     └─────────────────┘              │  │           │ │        │
│                                      │  │ • Format  │ │        │
│                                      │  │ • Styling │ │        │
│                                      │  │ • Helpers │ │        │
│                                      │  └───────────┘ │        │
│                                      └─────────────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌─────────────┐
│   User      │
│ Interaction │
└──────┬──────┘
       │
       │ 1. Select Strategies
       ▼
┌──────────────────┐
│   Sidebar UI     │
│  (sidebar.py)    │
└──────┬───────────┘
       │
       │ 2. Update Session State
       ▼
┌──────────────────┐
│  app.py          │
│ (Main Controller)│
└──────┬───────────┘
       │
       │ 3. Load Strategy Data
       ▼
┌──────────────────────┐
│  StrategyLoader      │
│ (strategy_loader.py) │
│                      │
│ Cache ┌────────────┐ │
│ Check │ Cached?    │ │
│       └─────┬──────┘ │
│             │        │
│         Yes │ No     │
│       ┌─────┴─────┐  │
│       │  Return   │  │
│       │  Generate │  │
│       └─────┬─────┘  │
└─────────────┼────────┘
              │
              │ 4. Strategy Data
              ▼
┌──────────────────────────┐
│  ComparisonEngine        │
│ (comparison_engine.py)   │
│                          │
│  ┌────────────────────┐  │
│  │ Filter by Time     │  │
│  │ Horizon            │  │
│  └────────┬───────────┘  │
│           │              │
│  ┌────────▼───────────┐  │
│  │ Calculate Metrics  │◄─┼───┐
│  │ (via Calculator)   │  │   │
│  └────────┬───────────┘  │   │
│           │              │   │
│  ┌────────▼───────────┐  │   │
│  │ Build Correlation  │  │   │
│  │ Matrix             │  │   │
│  └────────┬───────────┘  │   │
│           │              │   │
│  ┌────────▼───────────┐  │   │
│  │ Calculate Rolling  │  │   │
│  │ Metrics            │  │   │
│  └────────┬───────────┘  │   │
└───────────┼──────────────┘   │
            │                  │
            │ 5. Results       │
            ▼                  │
┌─────────────────────┐        │
│  app.py             │        │
│ (Store in Session)  │        │
└─────────┬───────────┘        │
          │                    │
          │ 6. Render          │
          ▼                    │
┌─────────────────────┐        │
│  UI Components      │        │
│                     │        │
│  ┌──────────────┐   │        │
│  │ charts.py    │   │        │
│  │ • Equity     │   │        │
│  │ • Drawdowns  │   │        │
│  │ • Returns    │   │        │
│  └──────────────┘   │        │
│                     │        │
│  ┌──────────────┐   │        │
│  │metrics_      │   │        │
│  │display.py    │   │        │
│  │ • Tables     │   │        │
│  │ • Cards      │   │        │
│  └──────────────┘   │        │
└─────────┬───────────┘        │
          │                    │
          │ 7. Display         │
          ▼                    │
┌─────────────────────┐        │
│   User Browser      │        │
│ (Interactive Charts)│        │
└─────────────────────┘        │
                               │
                               │
┌──────────────────────────────┴───┐
│  MetricsCalculator               │
│ (metrics_calculator.py)          │
│                                  │
│  ┌────────────────────────────┐  │
│  │ Calculate Return Metrics   │  │
│  │ • Total Return, CAGR       │  │
│  │ • Volatility, Downside Dev │  │
│  └────────────────────────────┘  │
│                                  │
│  ┌────────────────────────────┐  │
│  │ Calculate Risk Metrics     │  │
│  │ • Max Drawdown, Duration   │  │
│  │ • VaR, CVaR                │  │
│  └────────────────────────────┘  │
│                                  │
│  ┌────────────────────────────┐  │
│  │ Calculate Risk-Adjusted    │  │
│  │ • Sharpe, Sortino, Calmar  │  │
│  │ • Omega Ratio              │  │
│  └────────────────────────────┘  │
│                                  │
│  ┌────────────────────────────┐  │
│  │ Calculate Trade Stats      │  │
│  │ • Win Rate, Profit Factor  │  │
│  │ • Avg Win/Loss, Consec     │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

## Component Relationships

```
┌────────────────────────────────────────────────────────┐
│                     app.py                             │
│                  (Main Controller)                     │
│                                                        │
│  • Manages session state                              │
│  • Coordinates all components                         │
│  • Handles user interactions                          │
└───┬────────────────────────────────────────────────┬───┘
    │                                                │
    │                                                │
    ├──────────────┬──────────────┬─────────────────┤
    │              │              │                 │
    ▼              ▼              ▼                 ▼
┌────────┐   ┌─────────┐   ┌──────────┐   ┌─────────────┐
│Sidebar │   │ Charts  │   │ Metrics  │   │   Export    │
│        │   │         │   │ Display  │   │             │
│Uses:   │   │Uses:    │   │Uses:     │   │Uses:        │
│• utils │   │• plotly │   │• pandas  │   │• fpdf2      │
│        │   │• numpy  │   │• utils   │   │• jinja2     │
└────────┘   └─────────┘   └──────────┘   └─────────────┘
                     ▲
                     │
                     │ Gets Results From
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐   ┌────────────────────┐
│ ComparisonEngine │   │  StrategyLoader    │
│                  │   │                    │
│ Uses:            │   │ Uses:              │
│ • MetricsCalc    │◄──│ • Cache            │
│ • utils          │   │ • Mock Data Gen    │
└──────────────────┘   └────────────────────┘
        │
        │ Uses
        ▼
┌────────────────────┐
│ MetricsCalculator  │
│                    │
│ Uses:              │
│ • numpy            │
│ • pandas           │
└────────────────────┘
```

## Module Dependencies

```
app.py
├── ui/sidebar.py
│   ├── strategy_loader.py
│   └── utils.py
├── ui/charts.py
│   ├── plotly
│   └── numpy
├── ui/metrics_display.py
│   ├── pandas
│   └── utils.py
├── ui/export.py
│   ├── fpdf2
│   └── jinja2
├── comparison_engine.py
│   ├── metrics_calculator.py
│   │   ├── numpy
│   │   └── pandas
│   ├── strategy_loader.py
│   └── utils.py
└── utils.py
    └── streamlit
```

## Session State Management

```
st.session_state
├── selected_strategies: List[str]
│   └── ["Momentum ETH", "Mean Reversion BTC", ...]
│
├── time_horizon: str
│   └── "6M"
│
├── comparison_results: Dict
│   ├── time_horizon: str
│   ├── strategy_count: int
│   ├── equity_curves: Dict[str, Dict]
│   ├── metrics: Dict[str, Dict]
│   ├── drawdowns: Dict[str, Dict]
│   ├── returns: Dict[str, List]
│   ├── trades: Dict[str, List]
│   ├── correlation_matrix: Dict[str, Dict]
│   └── rolling_metrics: Dict[str, Dict]
│
├── last_update: datetime
│   └── 2025-10-11 12:35:00
│
├── show_advanced: bool
│   └── False
│
├── max_dd_threshold: float
│   └── 30.0
│
├── min_sharpe: float
│   └── 1.0
│
├── asset_filter: List[str]
│   └── ["BTC", "ETH"]
│
└── strategy_params: Dict[str, Dict]
    ├── "Momentum ETH"
    │   ├── window: 20
    │   ├── threshold: 2.0
    │   └── stop_loss: "5%"
    └── "Mean Reversion BTC"
        ├── window: 30
        └── threshold: 2.5
```

## Request/Response Flow

### 1. Initial Page Load

```
Browser → GET / → Streamlit Server
                  ↓
                  Initialize Session State
                  ↓
                  Render UI Components
                  ↓
                  Return HTML/JavaScript
Browser ← HTML ← Streamlit Server
```

### 2. Strategy Selection

```
Browser → User clicks checkbox → WebSocket Update
                                  ↓
                                  Update session_state.selected_strategies
                                  ↓
                                  Trigger Rerun
                                  ↓
                                  Re-render Sidebar
Browser ← Updated UI ← Streamlit Server
```

### 3. Run Analysis

```
Browser → User clicks "Run Analysis" → WebSocket Update
                                       ↓
                                       Load Strategies (StrategyLoader)
                                       ↓
                                       Run Comparison (ComparisonEngine)
                                       ↓
                                       Calculate Metrics (MetricsCalculator)
                                       ↓
                                       Store in session_state.comparison_results
                                       ↓
                                       Trigger Rerun
                                       ↓
                                       Render Charts (charts.py)
                                       ↓
                                       Render Tables (metrics_display.py)
Browser ← Updated UI with Results ← Streamlit Server
```

### 4. Chart Interaction

```
Browser → User hovers over chart → Plotly.js handles locally
          (No server roundtrip)

Browser → User zooms chart → Plotly.js updates view
          (No server roundtrip)
```

### 5. Export Report

```
Browser → User clicks "Export PDF" → WebSocket Update
                                     ↓
                                     export.py generates PDF
                                     ↓
                                     Create download button
                                     ↓
                                     Trigger Rerun
Browser ← Download Button ← Streamlit Server
        ↓
        User clicks download
        ↓
        GET /download → File Transfer
Browser ← PDF File ← Streamlit Server
```

## Technology Stack Layers

```
┌─────────────────────────────────────────────┐
│           Presentation Layer                │
│  • Streamlit UI Components                  │
│  • Plotly Interactive Charts                │
│  • HTML/CSS Styling                         │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│           Application Layer                 │
│  • app.py (Main Controller)                 │
│  • UI Components (sidebar, charts, etc.)    │
│  • Session State Management                 │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│           Business Logic Layer              │
│  • ComparisonEngine                         │
│  • MetricsCalculator                        │
│  • StrategyLoader                           │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│           Data Layer                        │
│  • Mock Data Generation                     │
│  • Caching (in-memory)                      │
│  • (Future: Database Integration)           │
└─────────────────────────────────────────────┘
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│                Local Development                │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  uv run streamlit run app.py             │  │
│  └───────────────┬──────────────────────────┘  │
│                  │                              │
│                  ▼                              │
│  ┌──────────────────────────────────────────┐  │
│  │  Streamlit Dev Server                    │  │
│  │  • Port 8501                             │  │
│  │  • Auto-reload on file change            │  │
│  │  • Debug mode                            │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              Production (Future)                │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Streamlit Cloud / AWS / Docker          │  │
│  ├──────────────────────────────────────────┤  │
│  │  • Load Balancer                         │  │
│  │  • Multiple App Instances                │  │
│  │  • Database Connection Pool              │  │
│  │  • Redis Cache                           │  │
│  │  • S3 for File Storage                   │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Performance Optimization Points

```
1. Data Loading
   ├── StrategyLoader Cache (in-memory)
   └── Lazy Loading (load on demand)

2. Computation
   ├── Vectorized Operations (NumPy)
   ├── Efficient Data Structures (Pandas)
   └── Minimal State Updates

3. Rendering
   ├── Plotly WebGL for large datasets
   ├── Table Pagination (20/50/100 rows)
   └── Conditional Rendering (only if data exists)

4. Session State
   ├── Minimal state variables
   ├── Clear unused data
   └── Efficient serialization

5. Future Optimizations
   ├── Database Connection Pooling
   ├── Redis Caching
   ├── Background Job Processing
   └── CDN for Static Assets
```

## Security Considerations

```
Current (Local Development)
├── No authentication required
├── Local file system access only
└── No network exposure

Future (Production)
├── Authentication
│   ├── User login
│   ├── API keys
│   └── Session management
│
├── Authorization
│   ├── Role-based access control
│   └── Strategy-level permissions
│
├── Data Protection
│   ├── HTTPS/TLS
│   ├── Data encryption at rest
│   └── Input validation
│
└── Monitoring
    ├── Access logs
    ├── Error tracking
    └── Performance monitoring
```

## Scalability Considerations

```
Current Limitations
├── Single Process (Streamlit Server)
├── In-Memory Cache (lost on restart)
├── No Persistence (strategies regenerated)
└── Single User Session per instance

Future Enhancements
├── Multi-Process Deployment
│   └── Load balancer + multiple instances
│
├── Persistent Cache
│   └── Redis or Memcached
│
├── Database Integration
│   └── PostgreSQL for strategy data
│
└── Horizontal Scaling
    └── Container orchestration (Kubernetes)
```

---

**Architecture Status: ✅ Well-Structured and Production-Ready**

Clear separation of concerns, modular design, efficient data flow, and room for future enhancements.
