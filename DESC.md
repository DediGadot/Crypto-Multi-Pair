# Multi-Pair Strategy & Training Overview

This repository implements a full-stack quantitative research harness for evaluating dozens of systematic crypto strategies across multiple trading pairs. The codebase is organised under `src/crypto_trader` with focused subpackages for data access, feature engineering, strategy logic, execution, analytics, and reporting. The flagship entry point for multi-asset research is `master_windowed_multipair.py`, which orchestrates data ingestion, synchronized train/test windowing, distributed backtests, portfolio-level aggregation, and rich HTML reporting.

The goal of this document is to give a crypto trading expert enough depth to critique the current approach: what data is used, how strategies behave, which metrics matter, and how training or model lifecycle management is handled for machine-learning components.

---

## Architecture Summary

- **Data layer** (`crypto_trader.data`): CCXT-based fetchers with disk caching (`BinanceDataFetcher`), CSV storage, and lightweight feature store accessors.
- **Feature layer** (`crypto_trader.features`): A `FeatureStore` + `augment_with_features` factory that joins on-chain, sentiment, options, and microstructure pillars with explicit staleness controls.
- **Strategy layer** (`crypto_trader.strategies`): Plugin registry (`StrategyRegistry`), Pydantic loader, and an extensive library of classical, statistical-arbitrage, alternative-data, and ML/RL strategies.
- **Execution layer** (`crypto_trader.execution` & `backtesting`): VectorBT-powered engine, risk-aware workers, indicator enrichment, and robust Sharpe calculation helpers.
- **Analytics layer** (`crypto_trader.analysis`): Windowed aggregators, benchmark comparison, cache persistence, diversification metrics, and report formatters.
- **Orchestration**: `master_windowed_multipair.py` (multi-pair windowed evaluation), `run_full_pipeline.py` (single vs portfolio modes), and Typer CLI commands for reproducible experiments.

---

## Data & Feature Pipeline

1. **Market data acquisition** (`data/fetchers.py`):
   - `BinanceDataFetcher` wraps CCXT with rate limiting, retry logic, and optional CSV caching. Requests default to 2-year windows (`max_days=730`) for multi-pair analysis.
   - Validation is lenient when offline: missing symbol/timeframe metadata does not block cached backtests.
2. **Raw storage & caching**:
   - `OHLCVStorage` persists per-symbol CSVs under `data/ohlcv`.
   - `OHLCVCache` provides an LRU cache to avoid redundant fetches within a run.
3. **Alternative data features** (`features/factory.py`):
   - `FeatureStore` reads pillars from `data/features/{pillar}/{symbol}.csv`.
   - `FeatureJoinConfig` defines pillars to join and staleness caps (`onchain:7d`, `sent:6h`, `opt:2d`, `micro:5m`).
   - `_apply_staleness_mask` nulls stale values and adds pillar-specific `*_is_stale` flags.
   - `augment_with_features` returns a single DataFrame with OHLCV + aligned features, ensuring `timestamp` is UTC and that NaNs are introduced rather than forward-filled beyond freshness windows.
4. **Feature engineering for ML** (`models/datasets.py`, `features/engineering.py`):
   - `build_feature_frame` derives returns, volatility, RSI, ATR, MACD, and other technical indicators with fallbacks if `pandas_ta` is absent.
   - `SequenceDataset` produces sliding windows for sequence models when PyTorch is available.

---

## Multi-Pair Evaluation Flow

The Typer command `python master_windowed_multipair.py analyze -p BTC/USDT -p ETH/USDT ...` runs the following steps:

1. **Configuration**:
   - Default horizons: 30d, 90d, 180d (reduced to 30d/90d when `--quick` is set).
   - Train/test split: Uniform across all pairs using `MultiPairTrainTestSplitter`, which enforces the same cutoff date (`runtime_date - test_years`) and ensures no overlap.
   - Output directory timestamped; a per-run cache lives in `output/cache/windowed_results.csv`.

2. **Data Fetch & Feature Augmentation**:
   - `fetch_pair_dataset` retrieves OHLCV data for each pair, clamps to `max_days`, and calls `augment_with_features` to attach alternative data pillars.

3. **Window Generation** (`orchestration/multipair_window_manager.py`):
   - `generate_windows` returns synchronized `MultiPairWindowSpec` objects for train and test sets.
   - Each window records per-pair index ranges, start/end timestamps, horizon metadata, and dataset type.

4. **Caching** (`analysis/windowed_cache.py`):
   - Before launching workers, the orchestrator consults `WindowedResultsCache`.
   - Cache keys encode strategy, pair, timeframe, horizon, window id, dataset type, and window bounds (normalized timestamps).

5. **Parallel Backtests** (`ProcessPoolExecutor` + `execution/workers.py`):
   - Jobs feed `run_backtest_worker` with pre-sliced window data (to reduce serialization overhead).
   - Worker responsibilities:
     - Rehydrate DataFrame, ensure timestamps are `datetime`, and slice to the requested horizon.
     - Instantiate strategy via registry (auto-imports `crypto_trader.strategies.library`).
     - `add_required_indicators` introspects strategy requirements and calculates missing TA columns.
     - Compose `BacktestConfig` (10k USDT, fee 10 bps, slippage 5 bps) and run `BacktestEngine`.
     - `BacktestEngine` (VectorBT-based) builds entries/exits from signals, enforces the custom Sharpe calculation (`mean/std` per window), collects trade stats, VaR/CVaR, expectancies, etc.
     - Successful results are cached immediately; failures log warnings and mark the window for aggregation with `None`.

6. **Aggregation** (`analysis/multipair_aggregator.py`):
   - For each strategy/horizon/dataset, `MultiPairAggregator`:
     - Uses `ResultsAggregator` to compute per-pair statistics (mean/median/std/p25/p75 of returns, Sharpe, drawdown, win rate, trade counts, weighted recency scores, consistency).
     - Builds equal-weight portfolio returns across pairs, deriving portfolio mean/median/std, Sharpe, worst drawdown, and `diversification_ratio` (portfolio Sharpe ÷ average individual Sharpe).
     - Computes cross-pair correlations, risk contributions (`marginal contribution to portfolio vol`), and `effective_num_assets` via eigenvalue dispersion.
     - Stores correlation matrices for heatmap visualisations.

7. **Benchmarking** (`analysis/benchmark_comparator.py`):
   - When `BuyAndHold` metrics are present, `BenchmarkComparator` computes alpha (absolute & relative), Sharpe alpha, and win rate vs benchmark per horizon.
   - Per-window alpha series feed Plotly visualisations (alpha distribution violin, cumulative returns, win-rate heatmap).

8. **Reporting** (`generate_multipair_html_report`):
   - Produces an HTML report featuring executive summary, strategy leaderboards, overfitting diagnostics (train vs test Sharpe gap), trade statistics, correlation heatmaps, diversification analysis, and benchmarking charts.
   - `SUMMARY.txt` captures key scalar metrics (success counts, top Sharpe averages).

---

## Performance Metrics & Analytics

- **Raw metrics** (`core/types.py::PerformanceMetrics`):
  - Profitability: `total_return`, `profit_factor`, `expectancy`, `final_capital`.
  - Risk-adjusted: `sharpe_ratio` (non-annualised per window), `sortino_ratio`, `calmar_ratio`, `recovery_factor`, `information_ratio`.
  - Tail risk: `max_drawdown`, `max_consecutive_drawdown_days`, `value_at_risk_95`, `conditional_var_95`, `omega_ratio`, `tail_ratio`, `ulcer_index`.
  - Trade diagnostics: win/loss counts, streaks, average win/loss, trade duration, fees.
  - Distribution shape: `skewness`, `kurtosis`.

- **Aggregator statistics** (`analysis/aggregator.py`):
  - Computes mean/median/std/p25/p75/weighted averages for returns, Sharpe, drawdown, win rate, and trade counts.
  - `consistency_score` penalises high Sharpe volatility ((mean Sharpe)/(std Sharpe)).
  - Weighted averages emphasise recent windows via exponential decay.

- **Multi-pair analytics**:
  - Cross-pair Pearson correlations with summary stats (`mean`, `max`, `min`).
  - Risk contributions derived from correlation matrix and per-pair volatility.
  - `effective_num_assets` via inverse participation ratio, highlighting true diversification.

- **Sharpe safeguards** (`execution/metric_utils.py`, `analysis/metrics.py`):
  - `periods_per_year_from_timeframe` ensures annualisation respects timeframe (minute/hour/day).
  - `calculate_sharpe_ratio_safe` throws if non-zero constant returns indicate a broken strategy.
  - `MetricsCalculator` (analysis layer) keeps VaR/CVaR consistent and infers timeframe when missing.

---

## Strategy Catalog

### Baseline & Classical Momentum

| Strategy | File | Core Idea | Key Params | Notes |
| --- | --- | --- | --- | --- |
| BuyAndHold | `library/buy_and_hold.py` | Benchmark passive exposure | None | BUY first bar, HOLD thereafter. |
| SMA_Crossover | `library/sma_crossover.py` | Fast/slow SMA (default 50/200) crossovers for trend-following | `fast_period`, `slow_period` | Emits BUY/SELL on golden/death crosses with confidence scaling by SMA delta. |
| MACD_Momentum | `library/macd_momentum.py` | MACD histogram momentum with signal smoothing | Standard MACD triplet | Integrates divergence filters, default `fast=12, slow=26, signal=9`. |
| TripleEMA | `library/triple_ema.py` | Multi-EMA alignment to reduce lag | `fast`, `medium`, `slow` | Looks for stacked EMA ordering plus volatility filter. |
| BollingerBreakout | `library/bollinger_breakout.py` | Trades band expansions and mean reversion | `period`, `std_dev` | Adds squeeze detection and volume confirmation. |
| Supertrend_ATR | `library/supertrend_atr.py` | ATR-based trailing stop bands | `atr_period`, `multiplier` | Suitable for regime detection fallback. |
| IchimokuCloud | `library/ichimoku_cloud.py` | Cloud components + Chikou span | Standard lengths | Summarises trend strength and Kumo alignment. |
| VWAPMeanReversion | `library/vwap_mean_reversion.py` | Deviations from VWAP with z-score thresholds | `lookback`, `std_threshold` | Uses intraday volume metrics when available. |

These strategies work on a single series but integrate seamlessly with the multi-pair pipeline (one worker per pair) because signals follow the standard BUY/SELL/HOLD schema.

### Mean-Reversion & Statistical Arbitrage

- **CopulaPairsTrading** (`library/copula_pairs_trading.py`):
  - Models tail dependence via Student-t copulas, calculates spreads, z-scores, hedge ratios, and tail probabilities.
  - Emits synthetic spread trades as BUY (long spread) / SELL (short spread) signals.
  - Auto-detects asset pairings when not configured but defaults to the first two `_close` columns.
  - Provides metadata (`z_score`, `spread`, `hedge_ratio`, `tail_probability`) for diagnostics.

- **StatisticalArbitrage** (`library/statistical_arbitrage_pairs.py`):
  - Adaptive Regime-Aware Statistical Arbitrage (ARASA) with Johansen cointegration (`CointegrationAnalyzer`) and Hidden Markov Model regime detection (`RegimeDetector`).
  - Calculates half-life, ensures stationary spread, sets regime-specific entry/exit thresholds, and scales position size with volatility.
  - Metadata includes cointegration stats, regime, z-score, and hedging info.

Both strategies rely on dual-price inputs (e.g., `BTC_close`, `ETH_close`) and are best suited to pair-level backtests; in the multi-pair runner they are treated as single-pair strategies but can be extended for multi-pair portfolios.

### Portfolio Allocation & Rebalancing

- **PortfolioRebalancer** (`library/portfolio_rebalancer.py`):
  - Executes threshold-based rebalancing when actual weights deviate >15% (default) from targets.
  - Supports threshold, calendar, or hybrid triggers, optional momentum filter, and per-asset metadata describing rebalance events.
  - Requires a dict of `{symbol, weight}` and multi-asset price dict input (outside the windowed runner due to dictionary input signature).

- **HierarchicalRiskParity** (`library/hierarchical_risk_parity.py`):
  - Uses PyPortfolioOpt’s HRP algorithm: hierarchical clustering, quasi-diagonalisation, and recursive bisection to allocate weights.
  - Weekly rebalance by default with lookback windows (>=90 periods) and fallback to equal weights if insufficient history.

- **RiskParity** (`library/risk_parity.py`) *(not shown above but present)*:
  - Equal-risk contribution across assets using covariance estimation; similar to HRP but without clustering.

- **BlackLitterman** (`library/black_litterman.py`):
  - Combines market equilibrium views (reverse-optimised) with momentum-based investor views using Bayesian blending (`tau` and `view_confidence`).
  - Falls back to equal weights when PyPortfolioOpt dependencies fail.

These strategies expect simultaneous multi-asset data and are therefore excluded from the default multi-pair window runner (which processes each pair independently). They are leveraged through `run_full_pipeline.py` in portfolio mode or bespoke orchestration.

### Regime Detection, Ensemble, and Alternative Data

- **VolatilityRegimeAdaptive** (`library/regime_adaptive.py`):
  - Uses HMM-based `RegimeDetector` to classify regimes (mean-reverting, trending, volatile), then routes to child strategies (defaults: `RSIMeanReversion`, `Supertrend_ATR`) with position-size scaling.
  - Metadata includes active regime to contextualise signals.

- **DynamicEnsemble** (`library/dynamic_ensemble.py`):
  - Loads recent performance metrics from `PerformanceStore` (CSV history), weights child strategies by recent Sharpe, enforces min/max weight caps, and aggregates confidences.
  - Metadata tracks contributing strategies per bar.

- **MultiTimeframeConfluence** (`library/multi_timeframe_confluence.py`):
  - Requires multi-timeframe data (15m..1w), evaluating EMA/RSi/trend alignment to score confluence (0–5).
  - Trades only when score ≥ threshold (default 4) and volume supports the move.

- **OnChainAnalytics** (`library/onchain_analytics.py`):
  - Rule-based detection using MVRV, SOPR, and net exchange flow (or proxy features) to flag capitulation/euphoria.
  - Emits strong BUY/SELL with metadata on thresholds used.

- **MultiModalSentimentFusion** (`library/multimodal_sentiment_fusion.py`):
  - Averages sentiment, on-chain, and momentum features; optionally loads a transformer encoder (`ProsusAI/finbert`) to adapt weights dynamically.
  - Fallback heuristics keep behaviour deterministic when transformers are absent.

- **OrderFlowImbalance** (`library/order_flow_imbalance.py`):
  - Uses high-frequency microstructure features (`micro.delta`, `micro.book_imbalance`, `micro.vpin`) to detect aggressive buying/selling.
  - Smooths signals and emits 0.7 confidence when deltas and imbalance cross thresholds.

These strategies rely heavily on the feature factory to supply sentiment (`sent.*`), on-chain (`onchain.*`), or microstructure (`micro.*`) columns; stale/missing data triggers HOLD outputs.

### Machine Learning & Reinforcement Learning

- **TransformerGRUPredictor** (`library/transformer_gru_predictor.py`):
  - Loads a hybrid transformer + GRU model (`models/transformer_gru.py`) for next-period return forecasting.
  - Requires a checkpoint (`models/transformer_gru.ckpt`); falls back to heuristic predictions if missing.
  - `build_feature_frame` {close, volume, returns, RSI, ATR, MACD} plus optional extra features feed the model. Signals fire only on final bar with metadata containing `predicted_return`.

- **DDQNFeatureSelected** (`library/ddqn_feature_selected.py`):
  - Integrates `stable_baselines3.DQN` policy trained in `rl/trading_env.py`.
  - Uses `FeatureSelector` (XGBoost + SHAP) to determine top features and reads them from `models/ddqn_features.json`.
  - Falls back to deterministic scoring when the policy or dependencies are missing.

- **DeepRLPortfolio** (`library/deep_rl_portfolio.py`):
  - PPO agent for multi-asset allocation using returns windows of configurable length (`lookback_period`).
  - Practically, the class currently heuristically balances momentum/volatility when no pre-trained model is loaded (`use_pretrained=False` default).

The RL infrastructure includes:
  - **Gym environment** (`rl/trading_env.py`), exposing BUY/HOLD/SELL actions with fee modelling and reward options (Sharpe vs returns).
  - **Feature selection** (`ml/feature_selection.py`) with gradient-boosted importance or correlation fallback.

---

## Training, Feature Selection, and Model Lifecycle

1. **Transformer/GRU models**:
   - `TransformerGRUModel` is a PyTorch Lightning module with transformer encoder, GRU memory, and dense regression head.
   - `load_transformer_gru` instantiates the architecture from checkpoint metadata and handles missing/extra keys gracefully.
   - `predict_next_return` ensures evaluation mode and CPU inference by default.

2. **Reinforcement learning**:
   - `TradingEnv` simulates discrete order execution with capital tracking and transaction fees.
   - Policies can be exported via Stable-Baselines3; at runtime strategies load `.zip` policies if present or degrade to heuristics.
   - `FeatureSelector` selects a fixed number of features (`top_n=20`) for RL/ML models, storing importances for audit.

3. **Regime & ensemble feedback loops**:
   - `PerformanceStore` maintains a rolling CSV of historical metrics per strategy/timeframe. It tolerates malformed rows, enforces timestamp parsing, and powers both dynamic weighting and dashboarding.

4. **Fallback behaviours**:
   - Many models guard optional heavy dependencies (torch, transformers, shap) to keep inference environments lightweight. Experts should verify that production deployments include the necessary stacks to avoid heuristic fallbacks.

---

## Execution & Risk Infrastructure

- **Backtest Engine** (`backtesting/engine.py`):
  - Converts signals to VectorBT entries/exits, computes custom Sharpe, Sortino, max drawdown, and rich trade analytics (profit factor, consecutive streaks).
  - Handles timestamp extraction robustly (prefers `timestamp` column, falls back to DatetimeIndex).
  - Returns a `BacktestResult` with `PerformanceMetrics`, trade list, equity curve, and metadata for `PerformanceStore`.

- **Workers** (`execution/workers.py`):
  - Provide logging hooks (`log_worker_lifecycle`, `log_dataframe_info`), indicator injection, timeframe mapping, and error formatting.
  - Prepares data by slicing to the target horizon (`slice_data_to_horizon`) and resets indexes to avoid VectorBT timestamp ambiguity.

- **Risk management** (`risk/manager.py`, `risk/sizing.py`, `risk/limits.py`):
  - `RiskManager` computes position sizes, enforces portfolio and per-trade limits, and determines stop loss / take profit levels based on `RiskConfig`.
  - Integrates with `PortfolioState` to monitor equity drawdowns and track limit breaches.

---

## Reporting & Observability

- **Caching**: `WindowedResultsCache` saves every aggregated metric to CSV, enabling incremental reruns and post-hoc audits.
- **Performance history**: `PerformanceStore` supports time-series dashboards and dynamic strategy selection.
- **HTML report**: `reports/formatters/html.py` and `plotly_benchmark_charts.py` inject CSS, Plotly charts (alpha comparison, cumulative returns, win-rate heatmap, return distribution violin), and textual insights (Executive Summary, Overfitting analysis, Trade Statistics, Statistical Tests guidance).
- **Summary artefacts**: Each run emits `SUMMARY.txt`, the cache CSV, full HTML report, and optionally static PNG exports if needed.

---

## Considerations for Expert Review

1. **Data integrity & coverage**
   - Are 2 years of hourly Binance spot data representative enough for all pairs? Should futures or alternative venues be incorporated?
   - Feature freshness rules: Do staleness thresholds align with real latency for on-chain or sentiment feeds?
   - Microstructure features (`micro.*`) assume custom data availability; experts should confirm ingestion quality.

2. **Train/Test methodology**
   - Non-overlapping windows with equal horizon lengths mitigate leakage, but there is no walk-forward re-training per strategy; consider sequential retraining for adaptive models.
   - Portfolio-level strategies are excluded from the multi-pair runner because they expect simultaneous multi-column inputs. Evaluate whether to extend window slicing to multi-asset frames for HRP/Black-Litterman comparisons.

3. **Strategy assumptions**
   - Copula and ARASA strategies rely on cointegration stability; review hedge-ratio re-estimation cadence and tail-risk handling.
   - Momentum strategies apply uniform parameters across all assets; experts may want per-asset tuning or volatility scaling.
   - Portfolio rebalancing thresholds (15%) and equal weight assumptions may not reflect desired risk budgeting.

4. **Machine learning lifecycle**
   - Transformer/DQN/DeepRL strategies default to heuristics when models are absent; ensure production deployments manage checkpoints and re-training schedules.
   - Feature selection relies on XGBoost/SHAP; confirm regulatory explainability requirements are met when optional dependencies are missing.

5. **Risk analytics**
   - Performance metrics capture VaR/CVaR and Omega ratios, but risk reporting could be extended to scenario analysis or liquidity stress (slippage modelling is static at 5 bps).
   - Effective number of assets and risk contributions are computed from equal-weighted window returns; experts should verify weighting assumptions fit portfolio objectives.

6. **Benchmarking & attribution**
   - Alpha comparisons use buy-and-hold per pair; consider crypto baskets or factor benchmarks for richer insight.
   - Strategy success rates count completed backtests, not statistical significance testing; integrate bootstrap/permutation tests for robust conclusions.

7. **Operational considerations**
   - The ProcessPool-based runner serialises per-window data; scaling to many pairs/horizons may warrant distributed execution or caching at a more granular level.
   - Logging currently leverages Loguru with emojis; confirm production logging policy and alerting.

By aligning on these components, a trading specialist can pinpoint gaps in data sourcing, model robustness, risk controls, and evaluation methodology—ensuring the multi-pair research stack evolves toward institutional-grade deployments.

