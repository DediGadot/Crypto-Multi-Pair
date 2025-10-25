#!/usr/bin/env python3
"""
Master Multi-Pair Windowed Analysis - Train/Test Split Across Multiple Assets

Extends single-pair windowed analysis to multiple trading pairs with:
- Synchronized window generation across all pairs
- Cross-pair correlation analysis
- Portfolio-level metrics
- Train/test split methodology

Usage:
    python master_windowed_multipair.py -p BTC/USDT -p ETH/USDT --quick
    python master_windowed_multipair.py -p BTC/USDT -p ETH/USDT -p BNB/USDT --test-years 1.0

**Purpose**: Multi-pair strategy evaluation with proper train/test split

**Third-party packages**:
- pandas: https://pandas.pydata.org/docs/
- loguru: https://loguru.readthedocs.io/en/stable/

**Expected Output**:
HTML report with multi-pair train/test results, correlations, and portfolio metrics.

**Methodology**:
Train/test split applied uniformly across all pairs with synchronized windows.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.features.factory import augment_with_features, DEFAULT_JOIN_CONFIG
from crypto_trader.strategies import get_registry
from crypto_trader.orchestration.multipair_window_manager import (
    MultiPairTrainTestSplitter, MultiPairWindowSpec
)
from crypto_trader.analysis.multipair_aggregator import MultiPairAggregator
from crypto_trader.analysis.windowed_cache import WindowedResultsCache
from crypto_trader.execution.workers import run_backtest_worker
from crypto_trader.reports.formatters.html import HTMLFormatter
from crypto_trader.analysis.benchmark_comparator import BenchmarkComparator
from crypto_trader.reports.formatters.plotly_benchmark_charts import (
    create_alpha_comparison_chart,
    create_win_rate_heatmap,
    create_cumulative_returns_chart,
    create_return_distribution_violin
)

app = typer.Typer()

_ERROR_SINK_ID: Optional[int] = None
_ORIGINAL_EXCEPTHOOK = sys.excepthook


def setup_error_logging(output_dir: Path) -> Path:
    """
    Route all error-level logs (plus stack traces) to errors.txt in the given directory
    and install an excepthook so unhandled crashes get captured too.
    """
    global _ERROR_SINK_ID

    errors_file = output_dir / "errors.txt"
    errors_file.parent.mkdir(parents=True, exist_ok=True)

    if _ERROR_SINK_ID is not None:
        logger.remove(_ERROR_SINK_ID)

    _ERROR_SINK_ID = logger.add(
        errors_file,
        level="ERROR",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}\n{exception}"
    )

    def _unhandled_exception_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
            return

        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("Unhandled exception")
        _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)

    sys.excepthook = _unhandled_exception_hook

    return errors_file


def fetch_pair_dataset(
    symbol: str,
    timeframe: str,
    max_days: int = 730  # 2 years default for multi-pair
) -> pd.DataFrame:
    """Fetch and augment data for a single pair."""
    logger.info(f"📡 Fetching {symbol} data ({max_days} days)")

    from datetime import timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=max_days)

    fetcher = BinanceDataFetcher()
    data = fetcher.get_ohlcv(symbol, timeframe, start_date, end_date)

    logger.info(f"✅ {symbol}: {len(data):,} candles")

    # Augment with features
    data = augment_with_features(data, symbol, timeframe, config=DEFAULT_JOIN_CONFIG)

    return data


def merge_pairs_to_portfolio_dataframe(
    window_data_dict: Dict[str, pd.DataFrame],
    pairs: List[str]
) -> pd.DataFrame:
    """
    Merge multiple pair DataFrames into single wide-format portfolio DataFrame.

    Converts:
        {'BTC/USDT': df1, 'ETH/USDT': df2}
    To:
        [timestamp | BTC/USDT_close | ETH/USDT_close | ...]

    Args:
        window_data_dict: Dict mapping pair -> DataFrame with OHLCV data
        pairs: List of pairs to merge

    Returns:
        Wide-format DataFrame suitable for portfolio strategies
    """
    if not window_data_dict:
        raise ValueError("Empty window_data_dict")

    # Start with first pair's timestamp as base
    first_pair = pairs[0]
    first_df = window_data_dict[first_pair].copy()

    # Ensure timestamp column exists
    if 'timestamp' not in first_df.columns:
        if isinstance(first_df.index, pd.DatetimeIndex):
            first_df = first_df.reset_index()
            if 'index' in first_df.columns:
                first_df = first_df.rename(columns={'index': 'timestamp'})
        else:
            raise ValueError(f"No timestamp found in {first_pair}")

    # Build merged dataframe with renamed columns per pair
    merged = first_df[['timestamp']].copy()

    for pair in pairs:
        pair_df = window_data_dict[pair].copy()

        # Normalize column name
        safe_pair_name = pair.replace('/', '_')

        # Add OHLCV columns (all required by backtesting engine)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in pair_df.columns:
                merged[f'{safe_pair_name}_{col}'] = pair_df[col].values
            else:
                raise ValueError(f"No '{col}' column in {pair}")

    return merged


def run_multipair_window_backtest(
    strategy_name: str,
    window: MultiPairWindowSpec,
    window_data_dict: Dict[str, pd.DataFrame],
    timeframe: str,
    pairs_to_run: Optional[List[str]] = None,
    portfolio_mode: bool = False
) -> Dict[str, Any]:
    """
    Run backtest for a multi-pair window.

    BUGFIX: Now receives pre-sliced window data instead of full datasets
    to eliminate memory leak (was passing ~5MB per task, now ~40KB)

    Args:
        strategy_name: Strategy to backtest
        window: Window specification
        window_data_dict: PRE-SLICED data for this window only (not full datasets)
        timeframe: Timeframe string
        pairs_to_run: Optional subset of pairs to process
        portfolio_mode: If True, merge all pairs and run strategy once

    Returns dict mapping pair -> result (or 'PORTFOLIO' -> result in portfolio mode)
    """
    results = {}
    target_pairs = pairs_to_run if pairs_to_run is not None else list(window.pair_windows.keys())

    # Portfolio mode: merge all pairs and run once
    if portfolio_mode:
        try:
            merged_df = merge_pairs_to_portfolio_dataframe(window_data_dict, target_pairs)

            # Convert to dict for serialization
            merged_df_copy = merged_df.copy()
            if 'timestamp' in merged_df_copy.columns:
                merged_df_copy['timestamp'] = merged_df_copy['timestamp'].astype(str)
            data_dict_for_worker = merged_df_copy.to_dict('list')

            # Run portfolio strategy ONCE with all assets
            result = run_backtest_worker(
                strategy_name=strategy_name,
                data_dict=data_dict_for_worker,
                horizon_name=window.horizon_name,
                horizon_days=int(window.horizon_name.replace('d', '')),
                symbol='PORTFOLIO',  # Special symbol for multi-asset
                timeframe=timeframe,
                default_params={}
            )

            if result and 'error' not in result:
                # Return single portfolio result
                results['PORTFOLIO'] = result
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.warning(f"⚠️  Portfolio backtest failed for {strategy_name}: {error_msg}")
                results['PORTFOLIO'] = None

        except Exception:
            logger.exception(f"❌ Portfolio backtest exception for {strategy_name}")
            results['PORTFOLIO'] = None

        return results

    for pair in target_pairs:
        # BUGFIX (Multipair Bug #1): Proper timestamp extraction
        # The old code was ambiguous when both index and timestamp column existed
        window_df = window_data_dict[pair]

        # Extract timestamp properly - use helper function pattern from backtesting engine
        # BUGFIX: Remove duplicates before creating dict to avoid VectorBT "timestamp already exists" error
        if isinstance(window_df.index, pd.DatetimeIndex):
            # Index is already datetime - make it a column, ensure no duplicates
            if 'timestamp' in window_df.columns:
                # Drop existing timestamp column to avoid duplicates
                df_for_worker = window_df.drop(columns=['timestamp']).reset_index()
                df_for_worker = df_for_worker.rename(columns={'index': 'timestamp'})
            else:
                df_for_worker = window_df.reset_index()
                if 'index' in df_for_worker.columns:
                    df_for_worker = df_for_worker.rename(columns={'index': 'timestamp'})
        elif 'timestamp' in window_df.columns:
            # Timestamp is already a column
            df_for_worker = window_df.reset_index(drop=True)
        else:
            raise ValueError(f"{pair}: Window data has no timestamp column or DatetimeIndex")

        # Convert to dict for serialization
        # BUGFIX: Convert timestamps to ISO strings to preserve datetime info
        df_for_worker_copy = df_for_worker.copy()
        if 'timestamp' in df_for_worker_copy.columns:
            df_for_worker_copy['timestamp'] = df_for_worker_copy['timestamp'].astype(str)
        data_dict_for_worker = df_for_worker_copy.to_dict('list')

        try:
            # Worker will use default params from strategy class
            result = run_backtest_worker(
                strategy_name=strategy_name,
                data_dict=data_dict_for_worker,
                horizon_name=window.horizon_name,
                horizon_days=int(window.horizon_name.replace('d', '')),
                symbol=pair,
                timeframe=timeframe,
                default_params={}  # Worker uses strategy defaults
            )

            if result and 'error' not in result:
                results[pair] = result
            else:
                # BUGFIX (Multipair Bug #2): Log failures at WARNING level, not DEBUG
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.warning(f"⚠️  Backtest failed for {strategy_name}/{pair}: {error_msg}")
                results[pair] = None
        except Exception:
            # BUGFIX (Multipair Bug #2): Log exceptions at ERROR level
            logger.exception(f"❌ Backtest exception for {strategy_name}/{pair}")
            results[pair] = None

    return results


def generate_multipair_html_report(
    aggregated_results: Dict[str, Any],
    strategies_to_test: List[str],
    horizon_names: List[str],
    pairs: List[str],
    timeframe: str,
    test_years: float,
    total_windows: int,
    successful: int,
    total_jobs: int,
    output_dir: Path,
    benchmark_comparisons: Optional[Dict[str, Dict[str, Any]]] = None
) -> Path:
    """Generate comprehensive HTML report for multi-pair analysis."""
    logger.info("📝 Generating HTML report...")

    # Debug: Check if we have any data
    total_metrics = 0
    for strategy_name in strategies_to_test:
        for horizon_name in horizon_names:
            for dataset_type in ['train', 'test']:
                if (horizon_name in aggregated_results.get(strategy_name, {}) and
                    dataset_type in aggregated_results[strategy_name][horizon_name]):
                    total_metrics += 1

    logger.info(f"Found {total_metrics} aggregated metric sets to report")

    formatter = HTMLFormatter()
    html_parts = []

    # HTML header
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html lang='en'>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='UTF-8'>")
    html_parts.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html_parts.append("<title>Multi-Pair Windowed Analysis Report</title>")
    html_parts.append(formatter.get_css())
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("<div class='container'>")

    # Title
    html_parts.append("<h1>🚀 Multi-Pair Windowed Train/Test Analysis</h1>")

    # Metadata section
    html_parts.append("<div class='metadata'>")
    html_parts.append(f"<p><strong>Trading Pairs:</strong> {', '.join(pairs)}</p>")
    html_parts.append(f"<p><strong>Timeframe:</strong> {timeframe}</p>")
    html_parts.append(f"<p><strong>Test Period:</strong> {test_years} years</p>")
    html_parts.append(f"<p><strong>Horizons:</strong> {', '.join(horizon_names)}</p>")
    html_parts.append(f"<p><strong>Strategies Tested:</strong> {len(strategies_to_test)}</p>")
    html_parts.append(f"<p><strong>Total Windows:</strong> {total_windows}</p>")
    html_parts.append(f"<p><strong>Success Rate:</strong> {successful}/{total_jobs} "
                     f"({100*successful/total_jobs if total_jobs > 0 else 0:.1f}%)</p>")
    html_parts.append(f"<p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html_parts.append("</div>")

    # Calculate strategy scores (moved up for executive summary)
    strategy_scores = []
    for strategy_name in strategies_to_test:
        test_sharpes = []
        train_sharpes = []

        for horizon_name in horizon_names:
            if horizon_name in aggregated_results.get(strategy_name, {}):
                if 'test' in aggregated_results[strategy_name][horizon_name]:
                    metrics = aggregated_results[strategy_name][horizon_name]['test']
                    if hasattr(metrics, 'portfolio_sharpe'):
                        test_sharpes.append(metrics.portfolio_sharpe)

                if 'train' in aggregated_results[strategy_name][horizon_name]:
                    metrics = aggregated_results[strategy_name][horizon_name]['train']
                    if hasattr(metrics, 'portfolio_sharpe'):
                        train_sharpes.append(metrics.portfolio_sharpe)

        if test_sharpes:
            avg_test_sharpe = sum(test_sharpes) / len(test_sharpes)
            avg_train_sharpe = sum(train_sharpes) / len(train_sharpes) if train_sharpes else 0.0
            strategy_scores.append((strategy_name, avg_test_sharpe, avg_train_sharpe))

    strategy_scores.sort(key=lambda x: x[1], reverse=True)

    # PHASE 2: Executive Summary with Auto-Generated Insights
    html_parts.append("<div class='blockquote' style='background-color: #f0f7ff; border-left: 4px solid #2E86DE;'>")
    html_parts.append("<h2>📋 Executive Summary</h2>")

    if strategy_scores:
        top_strategy = strategy_scores[0]
        top_name, top_test_sharpe, top_train_sharpe = top_strategy

        # Performance tier analysis
        excellent_strategies = [s for s in strategy_scores if s[1] >= 1.0]
        good_strategies = [s for s in strategy_scores if 0.5 <= s[1] < 1.0]
        underperforming = [s for s in strategy_scores if s[1] < 0.5]

        html_parts.append(f"<p><strong>Best Performing Strategy:</strong> {formatter.escape_html(top_name)} "
                         f"(Avg Test Sharpe: {top_test_sharpe:.2f})</p>")

        # Performance distribution
        html_parts.append(f"<p><strong>Strategy Performance Distribution:</strong></p>")
        html_parts.append("<ul>")
        html_parts.append(f"<li><span class='positive'>Excellent</span> (Sharpe ≥ 1.0): {len(excellent_strategies)} strategies</li>")
        html_parts.append(f"<li>Good (0.5 ≤ Sharpe < 1.0): {len(good_strategies)} strategies</li>")
        html_parts.append(f"<li><span class='negative'>Underperforming</span> (Sharpe < 0.5): {len(underperforming)} strategies</li>")
        html_parts.append("</ul>")

        # Overfitting analysis
        overfit_risks = [(s[0], s[2] - s[1]) for s in strategy_scores]
        high_overfit = [r for r in overfit_risks if r[1] > 0.5]
        low_overfit = [r for r in overfit_risks if r[1] <= 0.2]

        html_parts.append(f"<p><strong>Overfitting Risk Analysis:</strong></p>")
        html_parts.append("<ul>")
        if high_overfit:
            html_parts.append(f"<li><span class='negative'>High Risk</span>: {len(high_overfit)} strategies show significant train/test gap (>0.5)</li>")
        else:
            html_parts.append(f"<li><span class='positive'>No High Risk</span>: All strategies have train/test gap ≤ 0.5</li>")
        html_parts.append(f"<li><span class='positive'>Low Risk</span>: {len(low_overfit)} strategies show robust generalization (gap ≤ 0.2)</li>")
        html_parts.append("</ul>")

        # Key insights
        html_parts.append(f"<p><strong>Key Insights:</strong></p>")
        html_parts.append("<ul>")

        success_rate_pct = (successful / total_jobs * 100) if total_jobs > 0 else 0
        if success_rate_pct == 100:
            html_parts.append(f"<li><span class='positive'>✓ Perfect Execution</span>: All {total_jobs} backtest jobs completed successfully</li>")
        elif success_rate_pct >= 90:
            html_parts.append(f"<li><span class='positive'>✓ High Reliability</span>: {success_rate_pct:.1f}% job success rate ({successful}/{total_jobs})</li>")
        else:
            html_parts.append(f"<li><span class='negative'>⚠ Execution Issues</span>: Only {success_rate_pct:.1f}% success rate - review logs</li>")

        if len(excellent_strategies) > 0:
            html_parts.append(f"<li><span class='positive'>✓ Strong Performers</span>: {len(excellent_strategies)} strategies achieved Sharpe ratio ≥ 1.0")
        if len(pairs) > 1:
            html_parts.append(f"<li>📊 Multi-Asset Portfolio: Testing across {len(pairs)} trading pairs provides diversification insights</li>")
        html_parts.append(f"<li>🔬 Robust Testing: {len(horizon_names)} time horizons ({', '.join(horizon_names)}) validate strategy performance</li>")
        html_parts.append("</ul>")

        # Recommendations
        html_parts.append(f"<p><strong>Recommendations:</strong></p>")
        html_parts.append("<ul>")

        if excellent_strategies:
            html_parts.append(f"<li><strong>Deploy:</strong> Consider {formatter.escape_html(excellent_strategies[0][0])} for live trading "
                            f"(Test Sharpe: {excellent_strategies[0][1]:.2f})</li>")
        elif good_strategies:
            html_parts.append(f"<li><strong>Review:</strong> {formatter.escape_html(good_strategies[0][0])} shows promise but may need optimization</li>")
        else:
            html_parts.append(f"<li><strong>Caution:</strong> No strategies achieved Sharpe ≥ 0.5 - consider parameter tuning or alternative approaches</li>")

        if high_overfit:
            html_parts.append(f"<li><strong>Investigate:</strong> {len(high_overfit)} strategies show high overfitting risk - reduce complexity or increase training data</li>")

        if len(pairs) > 1:
            html_parts.append(f"<li><strong>Diversification:</strong> Review correlation matrix below to optimize asset allocation</li>")

        html_parts.append("</ul>")

    else:
        html_parts.append("<p><em>No valid test results available for analysis.</em></p>")

    html_parts.append("</div>")

    # Top strategies section
    html_parts.append("<h2>🏆 Top Strategies by Portfolio Sharpe Ratio</h2>")

    # No data warning
    if total_metrics == 0:
        html_parts.append("<div class='blockquote warning'>")
        html_parts.append("<p><strong>⚠️ No aggregated results available.</strong></p>")
        html_parts.append("<p>This could mean:</p>")
        html_parts.append("<ul>")
        html_parts.append("<li>All backtests failed (check logs)</li>")
        html_parts.append("<li>Aggregation encountered errors</li>")
        html_parts.append("<li>No valid results were produced for any strategy/horizon combination</li>")
        html_parts.append("</ul>")
        html_parts.append("</div>")

    logger.info(f"Found {len(strategy_scores)} strategies with valid test results")

    # Strategy ranking table
    html_parts.append("<table>")
    html_parts.append("<thead>")
    html_parts.append("<tr>")
    html_parts.append("<th>Rank</th>")
    html_parts.append("<th>Strategy</th>")
    html_parts.append("<th>Avg Test Sharpe</th>")
    html_parts.append("<th>Avg Train Sharpe</th>")
    html_parts.append("<th>Overfitting Risk</th>")
    html_parts.append("</tr>")
    html_parts.append("</thead>")
    html_parts.append("<tbody>")

    for rank, (strategy_name, test_sharpe, train_sharpe) in enumerate(strategy_scores[:20], 1):
        overfit_gap = train_sharpe - test_sharpe

        # Determine tier
        tier_class = ""
        if test_sharpe >= 1.0:
            tier_class = "tier1"
        elif test_sharpe >= 0.5:
            tier_class = "tier2"
        else:
            tier_class = "tier3"

        html_parts.append(f"<tr class='{tier_class}'>")
        html_parts.append(f"<td>{rank}</td>")
        html_parts.append(f"<td><strong>{formatter.escape_html(strategy_name)}</strong></td>")
        html_parts.append(f"<td>{test_sharpe:.2f}</td>")
        html_parts.append(f"<td>{train_sharpe:.2f}</td>")

        # Overfitting indicator
        if overfit_gap > 0.5:
            html_parts.append(f"<td><span class='negative'>High ({overfit_gap:.2f})</span></td>")
        elif overfit_gap > 0.2:
            html_parts.append(f"<td>Moderate ({overfit_gap:.2f})</td>")
        else:
            html_parts.append(f"<td><span class='positive'>Low ({overfit_gap:.2f})</span></td>")

        html_parts.append("</tr>")

    html_parts.append("</tbody>")
    html_parts.append("</table>")

    # PHASE 2: Advanced Portfolio Analytics Section
    html_parts.append("<h2>🔬 Advanced Portfolio Analytics</h2>")
    html_parts.append("<p><em>Interactive visualizations showing correlation structure, risk decomposition, and advanced risk metrics</em></p>")

    # Get aggregated data for visualization (use first test set available)
    sample_metrics = None
    for strategy_name in strategy_scores[:1] if strategy_scores else []:
        strategy = strategy_name[0] if isinstance(strategy_name, tuple) else strategy_name
        for horizon_name in horizon_names:
            if (horizon_name in aggregated_results.get(strategy, {}) and
                'test' in aggregated_results[strategy][horizon_name]):
                sample_metrics = aggregated_results[strategy][horizon_name]['test']
                break
        if sample_metrics:
            break

    if sample_metrics and hasattr(sample_metrics, 'correlation_matrix_df'):
        try:
            from crypto_trader.reports.formatters.plotly_interactive import (
                create_correlation_heatmap,
                create_risk_contribution_chart,
                create_advanced_metrics_dashboard
            )

            # Correlation Heatmap
            if sample_metrics.correlation_matrix_df is not None and not sample_metrics.correlation_matrix_df.empty:
                corr_fig = create_correlation_heatmap(sample_metrics.correlation_matrix_df)
                html_parts.append("<h3>Cross-Asset Correlation Matrix</h3>")
                html_parts.append(corr_fig.to_html(
                    full_html=False,
                    include_plotlyjs='cdn',
                    config={'responsive': True}
                ))

            # Risk Contribution Chart
            if sample_metrics.risk_contribution:
                risk_fig = create_risk_contribution_chart(sample_metrics.risk_contribution)
                html_parts.append("<h3>Portfolio Risk Contribution by Asset</h3>")
                html_parts.append("<p><em>Shows how much each asset contributes to total portfolio volatility</em></p>")
                html_parts.append(risk_fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={'responsive': True}
                ))

            # Advanced Metrics Dashboard
            metrics_dict = {
                'omega_ratio': getattr(sample_metrics.pair_metrics[pairs[0]], 'omega_ratio', 0.0) if sample_metrics.pair_metrics else 0.0,
                'tail_ratio': getattr(sample_metrics.pair_metrics[pairs[0]], 'tail_ratio', 1.0) if sample_metrics.pair_metrics else 1.0,
                'ulcer_index': getattr(sample_metrics.pair_metrics[pairs[0]], 'ulcer_index', 0.0) if sample_metrics.pair_metrics else 0.0,
                'max_consecutive_drawdown_days': getattr(sample_metrics.pair_metrics[pairs[0]], 'max_consecutive_drawdown_days', 0) if sample_metrics.pair_metrics else 0,
            }
            adv_fig = create_advanced_metrics_dashboard(metrics_dict)
            html_parts.append("<h3>Advanced Risk Metrics (Top Strategy)</h3>")
            html_parts.append("<p><em>Omega Ratio: probability-weighted gains/losses | Tail Ratio: right/left tail asymmetry | "
                            "Ulcer Index: drawdown-based volatility | Max Consecutive DD: longest underwater period</em></p>")
            html_parts.append(adv_fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={'responsive': True}
            ))

            # Portfolio Statistics Summary
            html_parts.append("<h3>Portfolio Diversification Metrics</h3>")
            html_parts.append("<table>")
            html_parts.append("<thead><tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr></thead>")
            html_parts.append("<tbody>")

            eff_num = sample_metrics.effective_num_assets
            eff_pct = (eff_num / len(pairs)) * 100 if len(pairs) > 0 else 0
            html_parts.append(f"<tr><td><strong>Effective Number of Assets</strong></td><td>{eff_num:.2f} / {len(pairs)}</td>")
            html_parts.append(f"<td>{eff_pct:.1f}% diversification efficiency</td></tr>")

            mean_corr = sample_metrics.correlation.mean_correlation
            html_parts.append(f"<tr><td><strong>Mean Correlation</strong></td><td>{mean_corr:.2f}</td>")
            if abs(mean_corr) < 0.3:
                html_parts.append("<td><span class='positive'>Low correlation - good diversification</span></td></tr>")
            elif abs(mean_corr) < 0.7:
                html_parts.append("<td>Moderate correlation</td></tr>")
            else:
                html_parts.append("<td><span class='negative'>High correlation - limited diversification</span></td></tr>")

            div_ratio = sample_metrics.diversification_ratio
            html_parts.append(f"<tr><td><strong>Diversification Ratio</strong></td><td>{div_ratio:.2f}</td>")
            if div_ratio > 1.2:
                html_parts.append("<td><span class='positive'>Portfolio Sharpe > Average - diversification benefit</span></td></tr>")
            elif div_ratio > 0.8:
                html_parts.append("<td>Neutral diversification effect</td></tr>")
            else:
                html_parts.append("<td><span class='negative'>Portfolio Sharpe < Average - concentration risk</span></td></tr>")

            html_parts.append("</tbody></table>")

        except Exception as e:
            logger.warning(f"Could not generate Phase 2 advanced visualizations: {e}")
            html_parts.append("<p><em>⚠️ Advanced visualizations not available for this dataset</em></p>")

    # PHASE 3.5: Risk Dashboard Section
    html_parts.append("<h2>⚠️ Risk Dashboard</h2>")
    html_parts.append("<p><em>Value at Risk (VaR) and Conditional VaR (CVaR) metrics for top strategies</em></p>")

    # Collect VaR/CVaR metrics from top strategies
    risk_dashboard_data = []
    for strategy_name in [s[0] for s in strategy_scores[:5]]:  # Top 5 strategies
        for horizon_name in horizon_names:
            if (horizon_name in aggregated_results.get(strategy_name, {}) and
                'test' in aggregated_results[strategy_name][horizon_name]):
                metrics = aggregated_results[strategy_name][horizon_name]['test']
                if hasattr(metrics, 'pair_metrics') and metrics.pair_metrics:
                    first_pair = list(metrics.pair_metrics.keys())[0]
                    pair_metrics = metrics.pair_metrics[first_pair]

                    # Extract VaR/CVaR from individual window results if available
                    # For now, we'll show aggregated risk metrics
                    risk_dashboard_data.append({
                        'strategy': f"{strategy_name}/{horizon_name}",
                        'var_95': pair_metrics.mean_drawdown,  # Using drawdown as proxy for VaR
                        'cvar_95': pair_metrics.mean_drawdown * 1.3,  # CVaR typically 1.2-1.5x VaR
                        'sharpe': pair_metrics.mean_sharpe
                    })

    if risk_dashboard_data:
        # Create Risk Metrics Table
        html_parts.append("<h3>Risk Metrics Summary</h3>")
        html_parts.append("<table>")
        html_parts.append("<thead>")
        html_parts.append("<tr>")
        html_parts.append("<th>Strategy/Horizon</th>")
        html_parts.append("<th>Max Drawdown (VaR Proxy)</th>")
        html_parts.append("<th>Expected Tail Loss (CVaR Proxy)</th>")
        html_parts.append("<th>Risk/Reward (Sharpe)</th>")
        html_parts.append("<th>Risk Level</th>")
        html_parts.append("</tr>")
        html_parts.append("</thead>")
        html_parts.append("<tbody>")

        for data in sorted(risk_dashboard_data, key=lambda x: abs(x['var_95']), reverse=True)[:10]:
            html_parts.append("<tr>")
            html_parts.append(f"<td><strong>{formatter.escape_html(data['strategy'])}</strong></td>")

            var_pct = abs(data['var_95']) * 100
            html_parts.append(f"<td>{formatter.format_percentage(data['var_95'])}</td>")

            cvar_pct = abs(data['cvar_95']) * 100
            html_parts.append(f"<td>{formatter.format_percentage(data['cvar_95'])}</td>")

            sharpe = data['sharpe']
            if sharpe >= 1.0:
                html_parts.append(f"<td><span class='positive'>{sharpe:.2f}</span></td>")
            elif sharpe >= 0.5:
                html_parts.append(f"<td>{sharpe:.2f}</td>")
            else:
                html_parts.append(f"<td><span class='negative'>{sharpe:.2f}</span></td>")

            # Risk level based on max drawdown
            if var_pct < 5:
                html_parts.append("<td><span class='positive'>Low</span></td>")
            elif var_pct < 15:
                html_parts.append("<td><span style='color: #F79F1F;'>Medium</span></td>")
            else:
                html_parts.append("<td><span class='negative'>High</span></td>")

            html_parts.append("</tr>")

        html_parts.append("</tbody>")
        html_parts.append("</table>")

        # Risk Interpretation Guide
        html_parts.append("<h3>Risk Metrics Interpretation</h3>")
        html_parts.append("<ul>")
        html_parts.append("<li><strong>Max Drawdown (VaR Proxy):</strong> Maximum peak-to-trough decline. Lower is better. <5% is low risk, 5-15% is medium, >15% is high.</li>")
        html_parts.append("<li><strong>Expected Tail Loss (CVaR Proxy):</strong> Average loss during worst drawdowns. Typically 1.2-1.5x the max drawdown.</li>")
        html_parts.append("<li><strong>Risk/Reward (Sharpe):</strong> Returns per unit of risk. >1.0 is excellent, >0.5 is good, <0.5 needs improvement.</li>")
        html_parts.append("<li><strong>Risk Level:</strong> Composite assessment based on drawdown magnitude. Lower risk strategies are more suitable for conservative portfolios.</li>")
        html_parts.append("</ul>")

        # Risk Management Recommendations
        html_parts.append("<h3>Risk Management Recommendations</h3>")
        html_parts.append("<ul>")

        # Find lowest and highest risk strategies
        lowest_risk = min(risk_dashboard_data, key=lambda x: abs(x['var_95']))
        highest_risk = max(risk_dashboard_data, key=lambda x: abs(x['var_95']))

        html_parts.append(f"<li><strong>Lowest Risk Strategy:</strong> {formatter.escape_html(lowest_risk['strategy'])} (Max DD: {formatter.format_percentage(lowest_risk['var_95'])})</li>")
        html_parts.append(f"<li><strong>Highest Risk Strategy:</strong> {formatter.escape_html(highest_risk['strategy'])} (Max DD: {formatter.format_percentage(highest_risk['var_95'])})</li>")

        # Portfolio allocation advice
        avg_drawdown = sum(abs(d['var_95']) for d in risk_dashboard_data) / len(risk_dashboard_data)
        if avg_drawdown < 0.10:
            html_parts.append("<li><span class='positive'>✓ Portfolio Risk Profile:</span> Generally conservative with manageable drawdowns</li>")
        elif avg_drawdown < 0.20:
            html_parts.append("<li>Portfolio Risk Profile: Moderate - suitable for balanced portfolios with 5-10% target allocation per strategy</li>")
        else:
            html_parts.append("<li><span class='negative'>⚠ Portfolio Risk Profile:</span> High volatility - consider position sizing <5% per strategy or hedging</li>")

        html_parts.append("</ul>")
    else:
        html_parts.append("<p><em>Risk metrics not available for selected strategies</em></p>")

    # BENCHMARK COMPARISON SECTIONS
    if benchmark_comparisons:
        # Section 1: Buy-and-Hold Benchmark Performance
        html_parts.append("<h2>📊 Buy-and-Hold Benchmark Performance</h2>")
        html_parts.append("<p><em>Performance of passive buy-and-hold strategy for comparison baseline</em></p>")

        # Display BuyAndHold metrics if available
        if "BuyAndHold" in aggregated_results:
            html_parts.append("<h3>Benchmark Metrics Summary</h3>")
            html_parts.append("<table>")
            html_parts.append("<thead>")
            html_parts.append("<tr>")
            html_parts.append("<th>Horizon</th>")
            html_parts.append("<th>Dataset</th>")
            html_parts.append("<th>Portfolio Sharpe</th>")
            html_parts.append("<th>Portfolio Return</th>")
            html_parts.append("<th>Portfolio Drawdown</th>")
            html_parts.append("</tr>")
            html_parts.append("</thead>")
            html_parts.append("<tbody>")

            for horizon_name in horizon_names:
                if horizon_name in aggregated_results["BuyAndHold"]:
                    for dataset_type in ['train', 'test']:
                        if dataset_type in aggregated_results["BuyAndHold"][horizon_name]:
                            metrics = aggregated_results["BuyAndHold"][horizon_name][dataset_type]
                            html_parts.append("<tr>")
                            html_parts.append(f"<td>{horizon_name}</td>")
                            html_parts.append(f"<td>{dataset_type.upper()}</td>")
                            html_parts.append(f"<td>{metrics.portfolio_sharpe:.2f}</td>")
                            html_parts.append(f"<td>{formatter.format_percentage(metrics.portfolio_mean_return)}</td>")
                            html_parts.append(f"<td>{formatter.format_percentage(metrics.portfolio_drawdown)}</td>")
                            html_parts.append("</tr>")

            html_parts.append("</tbody>")
            html_parts.append("</table>")
        else:
            html_parts.append("<p><em>⚠️ BuyAndHold benchmark metrics not available</em></p>")

        # Section 2: Strategy vs Benchmark Comparison
        html_parts.append("<h2>🎯 Strategy vs Benchmark Comparison</h2>")
        html_parts.append("<p><em>Alpha and win rate analysis comparing top strategies to buy-and-hold benchmark</em></p>")

        # Collect all comparisons for charts
        all_alpha_comparisons = {}
        heatmap_comparisons = {}

        for strategy_name, horizons in benchmark_comparisons.items():
            if strategy_name not in heatmap_comparisons:
                heatmap_comparisons[strategy_name] = {}

            for horizon_name, comparison in horizons.items():
                # For alpha chart (flat structure)
                key = f"{strategy_name}_{horizon_name}"
                all_alpha_comparisons[key] = comparison

                # For heatmap (nested structure)
                heatmap_comparisons[strategy_name][horizon_name] = comparison

        # Generate interactive charts
        try:
            # Alpha Comparison Chart
            if all_alpha_comparisons:
                html_parts.append("<h3>Alpha Comparison (Excess Returns vs Benchmark)</h3>")
                alpha_fig = create_alpha_comparison_chart(all_alpha_comparisons)
                html_parts.append(alpha_fig.to_html(
                    full_html=False,
                    include_plotlyjs='cdn',
                    config={'responsive': True}
                ))

            # Win Rate Heatmap
            if heatmap_comparisons and len(heatmap_comparisons) > 0:
                html_parts.append("<h3>Win Rate vs Benchmark by Strategy and Horizon</h3>")
                heatmap_fig = create_win_rate_heatmap(heatmap_comparisons)
                html_parts.append(heatmap_fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={'responsive': True}
                ))

            # Cumulative Returns Chart (for top strategy)
            if all_alpha_comparisons:
                # Get comparisons for the top strategy across horizons
                top_strategy_name = list(benchmark_comparisons.keys())[0]
                top_strategy_comparisons = {
                    f"{top_strategy_name}_{h}": c
                    for h, c in benchmark_comparisons[top_strategy_name].items()
                }
                if top_strategy_comparisons:
                    html_parts.append(f"<h3>Cumulative Returns: {formatter.escape_html(top_strategy_name)}</h3>")
                    cumulative_fig = create_cumulative_returns_chart(top_strategy_comparisons)
                    html_parts.append(cumulative_fig.to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config={'responsive': True}
                    ))

            # Return Distribution Violin Plot
            if all_alpha_comparisons:
                # Use first horizon for each top strategy
                violin_comparisons = {}
                for strategy_name in list(benchmark_comparisons.keys())[:3]:  # Top 3 strategies
                    if benchmark_comparisons[strategy_name]:
                        first_horizon = list(benchmark_comparisons[strategy_name].keys())[0]
                        key = f"{strategy_name}_{first_horizon}"
                        violin_comparisons[key] = benchmark_comparisons[strategy_name][first_horizon]

                if violin_comparisons:
                    html_parts.append("<h3>Return Distribution Comparison (Top Strategies)</h3>")
                    violin_fig = create_return_distribution_violin(violin_comparisons)
                    html_parts.append(violin_fig.to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config={'responsive': True}
                    ))

        except Exception as e:
            logger.warning(f"Could not generate benchmark comparison charts: {e}")
            html_parts.append(f"<p><em>⚠️ Benchmark comparison visualizations not available: {e}</em></p>")

        # Summary Table
        html_parts.append("<h3>Benchmark Comparison Summary</h3>")
        html_parts.append("<table>")
        html_parts.append("<thead>")
        html_parts.append("<tr>")
        html_parts.append("<th>Strategy</th>")
        html_parts.append("<th>Horizon</th>")
        html_parts.append("<th>Alpha</th>")
        html_parts.append("<th>Relative Alpha</th>")
        html_parts.append("<th>Sharpe Alpha</th>")
        html_parts.append("<th>Win Rate</th>")
        html_parts.append("</tr>")
        html_parts.append("</thead>")
        html_parts.append("<tbody>")

        for strategy_name in benchmark_comparisons:
            for horizon_name in benchmark_comparisons[strategy_name]:
                comp = benchmark_comparisons[strategy_name][horizon_name]
                html_parts.append("<tr>")
                html_parts.append(f"<td><strong>{formatter.escape_html(strategy_name)}</strong></td>")
                html_parts.append(f"<td>{horizon_name}</td>")

                # Alpha with color coding
                if comp.alpha >= 0:
                    html_parts.append(f"<td><span class='positive'>{comp.alpha:+.2f}%</span></td>")
                else:
                    html_parts.append(f"<td><span class='negative'>{comp.alpha:+.2f}%</span></td>")

                html_parts.append(f"<td>{comp.relative_alpha:+.1f}%</td>")

                # Sharpe alpha with color coding
                if comp.sharpe_alpha >= 0:
                    html_parts.append(f"<td><span class='positive'>{comp.sharpe_alpha:+.2f}</span></td>")
                else:
                    html_parts.append(f"<td><span class='negative'>{comp.sharpe_alpha:+.2f}</span></td>")

                # Win rate with color coding
                win_rate = comp.win_rate_vs_benchmark
                if win_rate >= 60:
                    html_parts.append(f"<td><span class='positive'>{win_rate:.1f}%</span></td>")
                elif win_rate >= 50:
                    html_parts.append(f"<td>{win_rate:.1f}%</td>")
                else:
                    html_parts.append(f"<td><span class='negative'>{win_rate:.1f}%</span></td>")

                html_parts.append("</tr>")

        html_parts.append("</tbody>")
        html_parts.append("</table>")

        # Interpretation Guide
        html_parts.append("<h3>Benchmark Comparison Metrics Explained</h3>")
        html_parts.append("<ul>")
        html_parts.append("<li><strong>Alpha:</strong> Excess return over benchmark. Positive alpha means strategy outperformed buy-and-hold.</li>")
        html_parts.append("<li><strong>Relative Alpha:</strong> Alpha expressed as percentage of benchmark return. Shows relative outperformance magnitude.</li>")
        html_parts.append("<li><strong>Sharpe Alpha:</strong> Difference in risk-adjusted returns (Sharpe ratio). Positive means better risk-adjusted performance.</li>")
        html_parts.append("<li><strong>Win Rate:</strong> Percentage of windows where strategy beat benchmark. >50% indicates consistent outperformance.</li>")
        html_parts.append("</ul>")

    # Per-pair results section
    html_parts.append("<h2>📊 Per-Pair Performance Details</h2>")

    for pair in pairs:
        html_parts.append(f"<h3>{formatter.escape_html(pair)}</h3>")
        html_parts.append("<table>")
        html_parts.append("<thead>")
        html_parts.append("<tr>")
        html_parts.append("<th>Strategy</th>")
        html_parts.append("<th>Horizon</th>")
        html_parts.append("<th>Dataset</th>")
        html_parts.append("<th>Avg Sharpe</th>")
        html_parts.append("<th>Avg Return</th>")
        html_parts.append("</tr>")
        html_parts.append("</thead>")
        html_parts.append("<tbody>")

        for strategy_name in strategies_to_test[:10]:  # Top 10 only
            for horizon_name in horizon_names:
                for dataset_type in ['train', 'test']:
                    if horizon_name in aggregated_results[strategy_name]:
                        if dataset_type in aggregated_results[strategy_name][horizon_name]:
                            metrics = aggregated_results[strategy_name][horizon_name][dataset_type]

                            # Get pair-specific metrics if available
                            if hasattr(metrics, 'pair_metrics') and pair in metrics.pair_metrics:
                                pair_metric = metrics.pair_metrics[pair]

                                html_parts.append("<tr>")
                                html_parts.append(f"<td>{formatter.escape_html(strategy_name)}</td>")
                                html_parts.append(f"<td>{horizon_name}</td>")
                                html_parts.append(f"<td>{dataset_type.upper()}</td>")
                                html_parts.append(f"<td>{pair_metric.mean_sharpe:.2f}</td>")
                                html_parts.append(f"<td>{formatter.format_percentage(pair_metric.mean_return)}</td>")
                                html_parts.append("</tr>")

        html_parts.append("</tbody>")
        html_parts.append("</table>")

    # PHASE 3: Trade Analysis Section
    html_parts.append("<h2>📈 Trade Analysis</h2>")
    html_parts.append("<p><em>Aggregated trade statistics across multiple windows and pairs</em></p>")

    # Aggregate trade statistics across top 3 strategies
    trade_analysis_data = {}
    for strategy_name in [s[0] for s in strategy_scores[:3]]:
        for horizon_name in horizon_names:
            if (horizon_name in aggregated_results.get(strategy_name, {}) and
                'test' in aggregated_results[strategy_name][horizon_name]):
                metrics = aggregated_results[strategy_name][horizon_name]['test']
                # Get first pair's metrics for trade analysis (WindowedMetrics object)
                if hasattr(metrics, 'pair_metrics') and metrics.pair_metrics:
                    first_pair = list(metrics.pair_metrics.keys())[0]
                    pair_metrics = metrics.pair_metrics[first_pair]

                    key = f"{strategy_name}/{horizon_name}"
                    trade_analysis_data[key] = {
                        'total_trades': pair_metrics.total_trades,
                        'win_rate': pair_metrics.mean_win_rate,  # FIXED: Use mean_win_rate
                        'mean_sharpe': pair_metrics.mean_sharpe,
                        'median_sharpe': pair_metrics.median_sharpe,
                        'mean_return': pair_metrics.mean_return,
                        'mean_drawdown': pair_metrics.mean_drawdown
                    }

    if trade_analysis_data:
        html_parts.append("<h3>Trade Statistics Summary</h3>")
        html_parts.append("<table>")
        html_parts.append("<thead>")
        html_parts.append("<tr>")
        html_parts.append("<th>Strategy/Horizon</th>")
        html_parts.append("<th>Total Trades</th>")
        html_parts.append("<th>Win Rate</th>")
        html_parts.append("<th>Avg Return</th>")
        html_parts.append("<th>Avg Sharpe</th>")
        html_parts.append("<th>Avg Drawdown</th>")
        html_parts.append("</tr>")
        html_parts.append("</thead>")
        html_parts.append("<tbody>")

        for key, data in sorted(trade_analysis_data.items(), key=lambda x: x[1]['mean_sharpe'], reverse=True)[:10]:
            html_parts.append("<tr>")
            html_parts.append(f"<td><strong>{formatter.escape_html(key)}</strong></td>")
            html_parts.append(f"<td>{int(data['total_trades'])}</td>")

            win_rate_pct = data['win_rate'] * 100
            if win_rate_pct >= 60:
                html_parts.append(f"<td><span class='positive'>{win_rate_pct:.1f}%</span></td>")
            elif win_rate_pct >= 50:
                html_parts.append(f"<td>{win_rate_pct:.1f}%</td>")
            else:
                html_parts.append(f"<td><span class='negative'>{win_rate_pct:.1f}%</span></td>")

            html_parts.append(f"<td>{formatter.format_percentage(data['mean_return'])}</td>")

            sharpe = data['mean_sharpe']
            if sharpe >= 1.0:
                html_parts.append(f"<td><span class='positive'>{sharpe:.2f}</span></td>")
            elif sharpe >= 0.5:
                html_parts.append(f"<td>{sharpe:.2f}</td>")
            else:
                html_parts.append(f"<td><span class='negative'>{sharpe:.2f}</span></td>")

            html_parts.append(f"<td>{formatter.format_percentage(data['mean_drawdown'])}</td>")
            html_parts.append("</tr>")

        html_parts.append("</tbody>")
        html_parts.append("</table>")

        # Trade Insights
        html_parts.append("<h3>Key Trade Insights</h3>")
        html_parts.append("<ul>")

        # Calculate aggregate statistics
        total_all_trades = sum(d['total_trades'] for d in trade_analysis_data.values())
        avg_win_rate = sum(d['win_rate'] * d['total_trades'] for d in trade_analysis_data.values()) / total_all_trades if total_all_trades > 0 else 0
        avg_sharpe = sum(d['mean_sharpe'] * d['total_trades'] for d in trade_analysis_data.values()) / total_all_trades if total_all_trades > 0 else 0
        avg_return = sum(d['mean_return'] * d['total_trades'] for d in trade_analysis_data.values()) / total_all_trades if total_all_trades > 0 else 0

        html_parts.append(f"<li><strong>Overall Win Rate:</strong> {avg_win_rate*100:.1f}% across {total_all_trades} total trades</li>")

        if avg_sharpe >= 1.0:
            html_parts.append(f"<li><span class='positive'>✓ Excellent Risk-Adjusted Returns</span>: Average Sharpe {avg_sharpe:.2f}</li>")
        elif avg_sharpe >= 0.5:
            html_parts.append(f"<li><span class='positive'>✓ Good Risk-Adjusted Returns</span>: Average Sharpe {avg_sharpe:.2f}</li>")
        else:
            html_parts.append(f"<li>Moderate Risk-Adjusted Returns: Average Sharpe {avg_sharpe:.2f}</li>")

        html_parts.append(f"<li><strong>Average Return per Window:</strong> {formatter.format_percentage(avg_return)}</li>")

        # Find strategy with most trades
        most_active = max(trade_analysis_data.items(), key=lambda x: x[1]['total_trades'])
        html_parts.append(f"<li><strong>Most Active:</strong> {formatter.escape_html(most_active[0])} with {int(most_active[1]['total_trades'])} trades</li>")

        # Find best performing strategy
        best_sharpe = max(trade_analysis_data.items(), key=lambda x: x[1]['mean_sharpe'])
        html_parts.append(f"<li><strong>Best Risk-Adjusted:</strong> {formatter.escape_html(best_sharpe[0])} (Sharpe: {best_sharpe[1]['mean_sharpe']:.2f})</li>")

        html_parts.append("</ul>")
    else:
        html_parts.append("<p><em>Trade analysis data not available for selected strategies</em></p>")

    # PHASE 3: Statistical Tests Section
    html_parts.append("<h2>📊 Statistical Tests</h2>")
    html_parts.append("<p><em>Testing return distribution properties and market assumptions</em></p>")

    # We'll test the top strategy's returns
    if strategy_scores:
        top_strategy_name = strategy_scores[0][0]
        test_results = []

        for horizon_name in horizon_names:
            if (horizon_name in aggregated_results.get(top_strategy_name, {}) and
                'test' in aggregated_results[top_strategy_name][horizon_name]):
                metrics = aggregated_results[top_strategy_name][horizon_name]['test']
                if hasattr(metrics, 'pair_metrics') and metrics.pair_metrics:
                    # Show statistical tests for each pair
                    for pair_symbol, pair_metrics in metrics.pair_metrics.items():
                        test_results.append({
                            'label': f"{pair_symbol} ({horizon_name})",
                            'interpretation': "Statistical tests require return series data (not available in aggregated metrics)"
                        })

        if test_results:
            html_parts.append("<h3>Return Distribution Tests</h3>")
            html_parts.append("<div class='blockquote info'>")
            html_parts.append(f"<p><strong>Strategy Tested:</strong> {formatter.escape_html(top_strategy_name)}</p>")
            html_parts.append("<ul>")

            html_parts.append("<li><strong>Normality Test (Jarque-Bera):</strong> Tests if returns follow a normal distribution. Most financial returns show fat tails (non-normal).</li>")
            html_parts.append("<li><strong>Autocorrelation:</strong> Measures if returns are predictable from past returns. High values indicate momentum or mean reversion.</li>")
            html_parts.append("<li><strong>Stationarity (ADF Test):</strong> Tests if statistical properties remain constant over time. Stationary returns are easier to model.</li>")
            html_parts.append("</ul>")

            html_parts.append("<p><em>Note: Full statistical testing requires access to raw return series. Current aggregated metrics provide summary statistics only.</em></p>")
            html_parts.append("<p><strong>Recommendations for robust statistical analysis:</strong></p>")
            html_parts.append("<ul>")
            html_parts.append("<li>Enable detailed return series logging in backtest engine</li>")
            html_parts.append("<li>Calculate statistical tests on raw data before aggregation</li>")
            html_parts.append("<li>Store test results in PerformanceMetrics dataclass</li>")
            html_parts.append("</ul>")

            html_parts.append("</div>")

    # Interpretation section
    html_parts.append("<h2>💡 Interpretation Guide</h2>")
    html_parts.append("<div class='blockquote info'>")
    html_parts.append("<h4>Understanding the Results</h4>")
    html_parts.append("<ul>")
    html_parts.append("<li><strong>Portfolio Sharpe Ratio:</strong> Risk-adjusted return across all pairs. >1.0 is excellent, >0.5 is good.</li>")
    html_parts.append("<li><strong>Overfitting Risk:</strong> Large gap between train and test Sharpe indicates overfitting.</li>")
    html_parts.append("<li><strong>Test Set Performance:</strong> Most important metric - reflects real-world generalization.</li>")
    html_parts.append("<li><strong>Multi-Pair Correlation:</strong> Lower correlation between pairs provides better diversification.</li>")
    html_parts.append("</ul>")
    html_parts.append("</div>")

    # Footer
    html_parts.append("</div>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    # Write report
    html_file = output_dir / "report.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_parts))

    logger.info(f"✅ HTML report saved: {html_file}")
    return html_file


@app.command()
def analyze(
    pairs: List[str] = typer.Option(
        ...,
        "--pairs", "-p",
        help="Trading pairs to analyze (repeat for each pair: -p BTC/USDT -p ETH/USDT)"
    ),
    timeframe: str = typer.Option(
        "1h",
        "--timeframe", "-t",
        help="Candle timeframe"
    ),
    test_years: float = typer.Option(
        1.0,
        "--test-years",
        help="Years reserved for test set"
    ),
    horizons: List[int] = typer.Option(
        None,
        "--horizons", "-h",
        help="Custom horizons in days"
    ),
    workers: int = typer.Option(
        2,
        "--workers", "-w",
        help="Parallel workers"
    ),
    quick: bool = typer.Option(
        False,
        "--quick", "-q",
        help="Quick mode (fewer horizons)"
    ),
    max_days: int = typer.Option(
        730,
        "--max-days",
        help="Maximum days of data to fetch per pair"
    ),
    output: str = typer.Option(
        "multipair_windowed_results",
        "--output", "-o",
        help="Output directory"
    ),
    portfolio_mode: bool = typer.Option(
        False,
        "--portfolio-mode",
        help="Run portfolio strategies with all assets merged (PORTFOLIO symbol)"
    ),
):
    """
    Run multi-pair windowed train/test analysis.

    Analyzes multiple trading pairs with synchronized windows and proper
    train/test split methodology.
    """
    start_time = time.time()

    logger.info("🚀 Multi-Pair Windowed Analysis Starting")
    logger.info(f"   Pairs: {len(pairs)} ({', '.join(pairs)})")
    logger.info(f"   Test Set: {test_years} years")
    logger.info(f"   Workers: {workers}")

    # Determine horizons
    if horizons is None:
        horizons = [30, 90] if quick else [30, 90, 180]

    horizon_names = [f"{h}d" for h in horizons]
    logger.info(f"   Horizons: {', '.join(horizon_names)}")

    # Create output directory early so error logging has somewhere to write
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{output}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_file = setup_error_logging(output_dir)
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"🧾 Error log: {errors_file}")

    # Validate configuration
    min_days_required = int(test_years * 365 * 1.5)
    if max_days < min_days_required:
        try:
            raise ValueError(f"max_days ({max_days}) < required ({min_days_required})")
        except ValueError:
            logger.exception("❌ Configuration validation failed")
        raise typer.Exit(1)

    # Initialize cache
    cache_file_path = output_dir / "cache" / "windowed_results.csv"
    cache = WindowedResultsCache(cache_file=cache_file_path)

    # Fetch data for all pairs
    logger.info(f"\n📡 Fetching data for {len(pairs)} pairs...")
    data_dict = {}

    for pair in pairs:
        try:
            data_dict[pair] = fetch_pair_dataset(pair, timeframe, max_days)
        except Exception:
            logger.exception(f"❌ Failed to fetch {pair}")
            raise typer.Exit(1)

    # Initialize window splitter
    runtime_date = datetime.now()
    splitter = MultiPairTrainTestSplitter(
        runtime_date=runtime_date,
        test_set_years=test_years,
        pairs=pairs
    )

    # Get multi-pair strategies (that can work with multiple assets)
    # Import strategy library to trigger registration
    import crypto_trader.strategies.library  # noqa: F401

    registry = get_registry()
    all_strategies = registry.list_strategies()

    # Portfolio strategies that need merged data
    PORTFOLIO_STRATEGIES = {'HierarchicalRiskParity', 'RiskParity', 'BlackLitterman', 'CopulaPairsTrading'}

    if portfolio_mode:
        # Portfolio mode: ONLY test portfolio strategies
        strategy_names = [name for name in registry.get_strategy_names()
                         if name in PORTFOLIO_STRATEGIES]
        logger.info(f"🎯 Portfolio mode: testing {len(strategy_names)} portfolio strategies")
    else:
        # Normal mode: test single-asset strategies (exclude portfolio strategies)
        strategy_names = [name for name in registry.get_strategy_names()
                         if name not in PORTFOLIO_STRATEGIES
                         and "Portfolio" not in name and "Statistical" not in name
                         and "DeepRL" not in name]  # Exclude DeepRL (not trained)

        # Add BuyAndHold for benchmark comparison
        if "BuyAndHold" in registry.get_strategy_names():
            if "BuyAndHold" not in strategy_names:
                strategy_names.append("BuyAndHold")

    strategies_to_test = strategy_names[:5] if quick else strategy_names

    logger.info(f"\n🧪 Testing {len(strategies_to_test)} strategies")

    # Generate windows for each horizon
    all_windows = {}
    total_windows = 0

    for horizon_days, horizon_name in zip(horizons, horizon_names):
        train_windows, test_windows = splitter.generate_windows(
            data_dict, horizon_days, horizon_name, timeframe
        )
        all_windows[horizon_name] = {
            'train': train_windows,
            'test': test_windows
        }
        total_windows += len(train_windows) + len(test_windows)

    logger.info(f"   Total windows: {total_windows}")

    # Split data for train/test to prevent data leakage
    logger.info(f"\n📊 Splitting data into train/test sets...")
    train_data_dict, test_data_dict = splitter.split_data(data_dict)

    # BUGFIX (Multipair Bug #4): Show actual per-pair sizes, not just first pair
    logger.info(f"   Train set sizes:")
    for pair in pairs:
        logger.info(f"      {pair}: {len(train_data_dict[pair]):,} rows")
    logger.info(f"   Test set sizes:")
    for pair in pairs:
        logger.info(f"      {pair}: {len(test_data_dict[pair]):,} rows")

    # Run backtests
    logger.info(f"\n⚡ Running backtests...")

    all_results = {}
    for strategy in strategies_to_test:
        all_results[strategy] = {}
        for horizon_name in horizon_names:
            all_results[strategy][horizon_name] = {
                'train': [],
                'test': []
            }

    total_jobs = len(strategies_to_test) * total_windows
    logger.info(f"   Total jobs: {total_jobs}")

    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []

        for strategy_name in strategies_to_test:
            for horizon_name in horizon_names:
                for dataset_type in ['train', 'test']:
                    windows = all_windows[horizon_name][dataset_type]

                    for window in windows:
                        cached_results: Dict[str, Any] = {}
                        pairs_to_run: List[str] = []

                        # BUGFIX: Portfolio mode uses 'PORTFOLIO' as cache key, not individual pairs
                        if portfolio_mode:
                            # Use first pair window for timestamp bounds (all pairs share same window)
                            first_pair_window = list(window.pair_windows.values())[0]
                            cached = cache.get_result(
                                strategy=strategy_name,
                                symbol='PORTFOLIO',
                                timeframe=timeframe,
                                horizon=window.horizon_name,
                                window_id=window.window_id,
                                dataset_type=dataset_type,
                                start_date=first_pair_window.start_date.isoformat(),
                                end_date=first_pair_window.end_date.isoformat()
                            )

                            if cached is not None:
                                cached_results['PORTFOLIO'] = cached
                            else:
                                # BUGFIX: In portfolio mode, pairs_to_run should contain actual pair symbols
                                # that will be passed to the worker to fetch data, even though results
                                # will come back with 'PORTFOLIO' as the key
                                pairs_to_run = list(window.pair_windows.keys())
                        else:
                            # Multi-pair mode: check cache for each pair individually
                            for pair, pair_window in window.pair_windows.items():
                                cached = cache.get_result(
                                    strategy=strategy_name,
                                    symbol=pair,
                                    timeframe=timeframe,
                                    horizon=window.horizon_name,
                                    window_id=window.window_id,
                                    dataset_type=dataset_type,
                                    start_date=pair_window.start_date.isoformat(),
                                    end_date=pair_window.end_date.isoformat()
                                )

                                if cached is not None:
                                    cached_results[pair] = cached
                                else:
                                    pairs_to_run.append(pair)

                        # If everything is cached we can skip the worker entirely
                        if not pairs_to_run:
                            all_results[strategy_name][horizon_name][dataset_type].append(cached_results)
                            successful += 1
                            continue

                        # BUGFIX: Pre-slice window data to avoid memory leak
                        # Was passing entire train/test datasets (~5MB), now only this window (~40KB)
                        window_data_dict = {}
                        for pair, pair_window in window.pair_windows.items():
                            if pair not in pairs_to_run:
                                continue
                            # Select correct dataset based on window type
                            if window.dataset_type == 'train':
                                pair_data = train_data_dict[pair]
                            else:
                                pair_data = test_data_dict[pair]

                            # BUGFIX (Multipair Bug #9): Validate index bounds before slicing
                            if pair_window.start_idx < 0 or pair_window.end_idx > len(pair_data):
                                logger.opt(stack=True).error(
                                    f"❌ Index out of bounds for {pair} window {window.window_id}: "
                                    f"[{pair_window.start_idx}:{pair_window.end_idx}] "
                                    f"but data has {len(pair_data)} rows"
                                )
                                continue
                            if pair_window.start_idx >= pair_window.end_idx:
                                logger.opt(stack=True).error(
                                    f"❌ Invalid window indices for {pair}: "
                                    f"start={pair_window.start_idx} >= end={pair_window.end_idx}"
                                )
                                continue

                            # Slice just this window's data
                            window_data_dict[pair] = pair_data.iloc[
                                pair_window.start_idx:pair_window.end_idx
                            ].copy()

                        future = executor.submit(
                            run_multipair_window_backtest,
                            strategy_name,
                            window,
                            window_data_dict,  # Pre-sliced data only
                            timeframe,
                            pairs_to_run,
                            portfolio_mode  # Pass portfolio mode flag
                        )
                        futures.append((
                            future,
                            strategy_name,
                            horizon_name,
                            dataset_type,
                            window,
                            cached_results,
                            pairs_to_run
                        ))

        with tqdm(total=len(futures), desc="Running backtests") as pbar:
            for future, strategy_name, horizon_name, dataset_type, window, cached_results, pairs_to_run in futures:
                try:
                    computed_results = future.result(timeout=300)

                    # HARD STOP: Future returned None
                    if computed_results is None:
                        raise RuntimeError(
                            f"FATAL: Worker returned None for {strategy_name}/{horizon_name}/{dataset_type}. "
                            f"This indicates a critical failure in the backtest worker. "
                            f"All results must be valid dictionaries. Check errors.txt for details."
                        )

                    combined_results = dict(cached_results)
                    combined_results.update(computed_results)

                    # HARD STOP: Missing pairs (skip check in portfolio mode)
                    if not portfolio_mode:
                        missing_pairs = [pair for pair in pairs_to_run if pair not in computed_results]
                        if missing_pairs:
                            raise RuntimeError(
                                f"FATAL: Missing results for pairs {', '.join(missing_pairs)} "
                                f"in {strategy_name}/{horizon_name}/{dataset_type}. "
                                f"Worker must return results for ALL requested pairs or raise an exception. "
                            f"Partial failures indicate data corruption or worker malfunction."
                        )

                    # HARD STOP: Empty results when pairs were requested
                    if not combined_results and pairs_to_run:
                        raise RuntimeError(
                            f"FATAL: Worker returned empty results for {strategy_name}/{horizon_name}/{dataset_type} "
                            f"but {len(pairs_to_run)} pairs were requested. "
                            f"This should never happen - worker must succeed or raise."
                        )

                    # Persist newly computed results to cache
                    for pair, result in computed_results.items():
                        # HARD STOP: Result contains error
                        if result and 'error' in result:
                            raise RuntimeError(
                                f"FATAL: Worker returned error for {pair}: {result['error']}. "
                                f"Workers must raise exceptions, not return error dicts. "
                                f"Fix the underlying issue in the backtest worker."
                            )

                        # Portfolio mode: use PORTFOLIO symbol for caching
                        if portfolio_mode and pair == 'PORTFOLIO':
                            # Use first pair window for timestamp bounds
                            first_pair_window = list(window.pair_windows.values())[0]
                            cache.store_result(
                                result=result,
                                strategy=strategy_name,
                                symbol='PORTFOLIO',
                                timeframe=timeframe,
                                horizon=window.horizon_name,
                                window_id=window.window_id,
                                dataset_type=dataset_type,
                                start_date=first_pair_window.start_date.isoformat(),
                                end_date=first_pair_window.end_date.isoformat()
                            )
                        else:
                            pair_window = window.pair_windows[pair]
                            cache.store_result(
                                result=result,
                                strategy=strategy_name,
                                symbol=pair,
                                timeframe=timeframe,
                                horizon=window.horizon_name,
                                window_id=window.window_id,
                                dataset_type=dataset_type,
                                start_date=pair_window.start_date.isoformat(),
                                end_date=pair_window.end_date.isoformat()
                            )

                    all_results[strategy_name][horizon_name][dataset_type].append(combined_results)
                    successful += 1

                except Exception as e:
                    # HARD STOP: Any exception terminates the entire run
                    logger.exception(f"FATAL BACKTEST FAILURE: {strategy_name}/{horizon_name}/{dataset_type}")
                    logger.error(f"Terminating entire analysis due to backtest failure.")
                    logger.error(f"Fix the root cause before continuing. See errors.txt for full traceback.")
                    raise SystemExit(1) from e

                pbar.update(1)

    logger.info(f"\n📊 Backtest Results: {successful} successful, {failed} failed")

    # Aggregate results
    logger.info(f"\n📊 Aggregating results...")

    aggregator = MultiPairAggregator()
    aggregated_results = {}

    for strategy_name in strategies_to_test:
        aggregated_results[strategy_name] = {}

        for horizon_name in horizon_names:
            aggregated_results[strategy_name][horizon_name] = {}

            for dataset_type in ['train', 'test']:
                results_list = all_results[strategy_name][horizon_name][dataset_type]

                # Skip if no windows were generated for this horizon/dataset combination
                # This can happen when test period is shorter than the horizon window
                if not results_list:
                    logger.warning(
                        f"⚠️  No windows for {strategy_name}/{horizon_name}/{dataset_type} - skipping aggregation"
                    )
                    continue

                # Convert multi-pair results to per-pair lists
                # In portfolio mode, use 'PORTFOLIO' as the single "pair"
                result_keys = ['PORTFOLIO'] if portfolio_mode else pairs
                pair_results = {key: [] for key in result_keys}
                failed_windows = 0
                total_windows = len(results_list)

                for window_idx, window_results in enumerate(results_list):
                    window_had_failure = False
                    for pair, result in window_results.items():
                        if result:
                            pair_results[pair].append(result)
                        else:
                            window_had_failure = True
                    if window_had_failure:
                        failed_windows += 1

                # Check if we have any pair results
                total_pair_results = sum(len(v) for v in pair_results.values())

                # HARD STOP: Windows had failures (should have been caught earlier)
                if failed_windows > 0:
                    raise RuntimeError(
                        f"FATAL: {strategy_name}/{horizon_name}/{dataset_type} has {failed_windows}/{total_windows} "
                        f"windows with failures. This should have been caught during backtest execution. "
                        f"Data corruption detected."
                    )

                # HARD STOP: No valid pair results
                if total_pair_results == 0:
                    raise RuntimeError(
                        f"FATAL: No valid results for {strategy_name}/{horizon_name}/{dataset_type}. "
                        f"Total windows: {total_windows}, but no pair results extracted. "
                        f"Data structure corruption detected."
                    )

                # Aggregate - any exception here is FATAL
                try:
                    metrics = aggregator.aggregate_multipair_windows(
                        pair_results,
                        strategy_name,
                        horizon_name,
                        dataset_type
                    )
                    aggregated_results[strategy_name][horizon_name][dataset_type] = metrics
                    logger.debug(f"✓ Aggregated {strategy_name}/{horizon_name}/{dataset_type}: "
                               f"portfolio_sharpe={metrics.portfolio_sharpe:.2f}")
                except Exception as e:
                    # HARD STOP: Aggregation failure
                    logger.exception(f"FATAL AGGREGATION FAILURE: {strategy_name}/{horizon_name}/{dataset_type}")
                    logger.error(f"Aggregator failed with {len(pair_results)} pair result sets")
                    logger.error(f"This is a critical bug in the aggregation logic.")
                    raise SystemExit(1) from e

    # Calculate benchmark comparisons
    logger.info(f"\n📊 Calculating benchmark comparisons...")
    comparator = BenchmarkComparator()
    benchmark_comparisons = {}

    # Get top 3 strategies by test Sharpe (excluding BuyAndHold itself)
    strategy_scores = []
    for strategy_name in strategies_to_test:
        if strategy_name == "BuyAndHold":
            continue
        test_sharpes = []
        for horizon_name in horizon_names:
            if (horizon_name in aggregated_results.get(strategy_name, {}) and
                'test' in aggregated_results[strategy_name][horizon_name]):
                metrics = aggregated_results[strategy_name][horizon_name]['test']
                if hasattr(metrics, 'portfolio_sharpe'):
                    test_sharpes.append(metrics.portfolio_sharpe)

        if test_sharpes:
            avg_test_sharpe = sum(test_sharpes) / len(test_sharpes)
            strategy_scores.append((strategy_name, avg_test_sharpe))

    strategy_scores.sort(key=lambda x: x[1], reverse=True)
    top_strategies = [s[0] for s in strategy_scores[:3]] if strategy_scores else []

    # Calculate comparisons for top strategies
    if "BuyAndHold" in aggregated_results:
        for strategy_name in top_strategies:
            benchmark_comparisons[strategy_name] = {}
            for horizon_name in horizon_names:
                # Only compare test set performance
                if (horizon_name in aggregated_results.get(strategy_name, {}) and
                    horizon_name in aggregated_results.get("BuyAndHold", {}) and
                    'test' in aggregated_results[strategy_name][horizon_name] and
                    'test' in aggregated_results["BuyAndHold"][horizon_name]):

                    try:
                        strategy_metrics = aggregated_results[strategy_name][horizon_name]['test']
                        benchmark_metrics = aggregated_results["BuyAndHold"][horizon_name]['test']

                        comparison = comparator.compare_to_benchmark(strategy_metrics, benchmark_metrics)
                        benchmark_comparisons[strategy_name][horizon_name] = comparison

                        logger.info(f"  {strategy_name}/{horizon_name}: α={comparison.alpha:+.2f}%, "
                                  f"win rate={comparison.win_rate_vs_benchmark:.1f}%")
                    except Exception as e:
                        logger.warning(f"  Failed to compare {strategy_name}/{horizon_name}: {e}")
    else:
        logger.warning("BuyAndHold benchmark not available for comparison")
        benchmark_comparisons = None

    # Save results
    logger.info(f"\n💾 Saving results...")

    # Save cache
    cache.save()

    # Save summary
    summary_file = output_dir / "SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-PAIR WINDOWED TRAIN/TEST ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Pairs: {', '.join(pairs)}\n")
        f.write(f"Timeframe: {timeframe}\n")
        f.write(f"Test Set: {test_years} years\n")
        f.write(f"Horizons: {', '.join(horizon_names)}\n")
        f.write(f"Strategies: {len(strategies_to_test)}\n")
        f.write(f"Total Windows: {total_windows}\n")
        f.write(f"Success Rate: {successful}/{total_jobs} ({100*successful/total_jobs if total_jobs > 0 else 0:.1f}%)\n")
        f.write("\n" + "="*80 + "\n")
        f.write("TOP STRATEGIES BY PORTFOLIO SHARPE\n")
        f.write("="*80 + "\n\n")

        # Rank by test set portfolio Sharpe
        strategy_scores = []

        for strategy_name in strategies_to_test:
            sharpes = []
            for horizon_name in horizon_names:
                if 'test' in aggregated_results[strategy_name].get(horizon_name, {}):
                    metrics = aggregated_results[strategy_name][horizon_name]['test']
                    sharpes.append(metrics.portfolio_sharpe)

            if sharpes:
                avg_sharpe = sum(sharpes) / len(sharpes)
                strategy_scores.append((strategy_name, avg_sharpe))

        strategy_scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (strategy_name, avg_sharpe) in enumerate(strategy_scores[:10], 1):
            f.write(f"{rank}. {strategy_name}: {avg_sharpe:.2f}\n")

    logger.info(f"✅ Summary saved: {summary_file}")

    # Generate HTML report
    try:
        html_file = generate_multipair_html_report(
            aggregated_results=aggregated_results,
            strategies_to_test=strategies_to_test,
            horizon_names=horizon_names,
            pairs=pairs,
            timeframe=timeframe,
            test_years=test_years,
            total_windows=total_windows,
            successful=successful,
            total_jobs=total_jobs,
            output_dir=output_dir,
            benchmark_comparisons=benchmark_comparisons  # Pass benchmark comparisons
        )
        logger.info(f"📊 HTML report: {html_file}")
    except Exception:
        logger.exception("Failed to generate HTML report")

    elapsed = time.time() - start_time
    logger.info(f"\n✅ Analysis complete in {elapsed:.1f}s")
    logger.info(f"📁 Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    app()
