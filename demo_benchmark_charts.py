"""
Demo: Plotly Benchmark Charts

Demonstrates how to use the plotly_benchmark_charts module to create
interactive visualizations comparing trading strategies to benchmarks.

This script shows:
1. Creating mock benchmark comparison data
2. Generating all 4 chart types
3. Saving charts to HTML files
4. Combining charts into a complete report

Usage:
    uv run python demo_benchmark_charts.py
"""

from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import numpy as np

from crypto_trader.reports.formatters.plotly_benchmark_charts import (
    create_alpha_comparison_chart,
    create_win_rate_heatmap,
    create_cumulative_returns_chart,
    create_return_distribution_violin
)
from crypto_trader.analysis.benchmark_comparator import BenchmarkComparison


def create_realistic_comparison(
    strategy_name: str,
    horizon: str,
    alpha: float,
    win_rate: float,
    num_windows: int = 15
) -> BenchmarkComparison:
    """
    Create realistic mock BenchmarkComparison for demonstration.

    Args:
        strategy_name: Name of the strategy
        horizon: Time horizon (e.g., '30d', '90d')
        alpha: Expected alpha (excess return %)
        win_rate: Expected win rate (%)
        num_windows: Number of windows to simulate

    Returns:
        BenchmarkComparison object with realistic data
    """
    # Set seed for reproducibility based on strategy name
    np.random.seed(hash(strategy_name + horizon) % 2**32)

    # Simulate benchmark returns (mean ~10%, std ~3%)
    benchmark_mean = 10.0
    benchmark_std = 3.0
    benchmark_returns = np.random.normal(benchmark_mean, benchmark_std, num_windows).tolist()

    # Simulate strategy returns (mean = benchmark + alpha)
    strategy_mean = benchmark_mean + alpha
    strategy_std = 3.5  # Slightly higher volatility
    strategy_returns = np.random.normal(strategy_mean, strategy_std, num_windows).tolist()

    # Calculate per-window alphas
    window_alphas = [s - b for s, b in zip(strategy_returns, benchmark_returns)]

    # Calculate actual win rate from simulated data
    actual_wins = sum(1 for s, b in zip(strategy_returns, benchmark_returns) if s > b)
    actual_win_rate = (actual_wins / num_windows) * 100

    # Calculate Sharpe ratios (simplified)
    strategy_sharpe = (np.mean(strategy_returns) - 2.0) / np.std(strategy_returns)  # Risk-free = 2%
    benchmark_sharpe = (np.mean(benchmark_returns) - 2.0) / np.std(benchmark_returns)

    return BenchmarkComparison(
        strategy_name=strategy_name,
        horizon_name=horizon,
        dataset_type='test',
        strategy_return=float(np.mean(strategy_returns)),
        benchmark_return=float(np.mean(benchmark_returns)),
        alpha=float(np.mean(strategy_returns) - np.mean(benchmark_returns)),
        relative_alpha=float((np.mean(strategy_returns) - np.mean(benchmark_returns)) / np.mean(benchmark_returns) * 100),
        strategy_sharpe=float(strategy_sharpe),
        benchmark_sharpe=float(benchmark_sharpe),
        sharpe_alpha=float(strategy_sharpe - benchmark_sharpe),
        windows_beat_benchmark=actual_wins,
        total_windows=num_windows,
        win_rate_vs_benchmark=actual_win_rate,
        window_alphas=window_alphas,
        window_returns_strategy=strategy_returns,
        window_returns_benchmark=benchmark_returns
    )


def main():
    """Generate and save all benchmark comparison charts."""

    print("📊 Generating Benchmark Comparison Charts Demo\n")

    # Create realistic comparison data for multiple strategies and horizons
    strategies = {
        'Copula Pairs Trading': [
            create_realistic_comparison('Copula Pairs Trading', '30d', alpha=6.5, win_rate=75.0),
            create_realistic_comparison('Copula Pairs Trading', '90d', alpha=8.2, win_rate=78.0),
            create_realistic_comparison('Copula Pairs Trading', '180d', alpha=7.8, win_rate=76.0),
        ],
        'MACD Strategy': [
            create_realistic_comparison('MACD Strategy', '30d', alpha=4.2, win_rate=68.0),
            create_realistic_comparison('MACD Strategy', '90d', alpha=5.1, win_rate=70.0),
            create_realistic_comparison('MACD Strategy', '180d', alpha=4.8, win_rate=69.0),
        ],
        'RSI Mean Reversion': [
            create_realistic_comparison('RSI Mean Reversion', '30d', alpha=2.8, win_rate=62.0),
            create_realistic_comparison('RSI Mean Reversion', '90d', alpha=3.5, win_rate=65.0),
            create_realistic_comparison('RSI Mean Reversion', '180d', alpha=3.2, win_rate=63.0),
        ],
        'Bollinger Bands': [
            create_realistic_comparison('Bollinger Bands', '30d', alpha=-1.5, win_rate=42.0),
            create_realistic_comparison('Bollinger Bands', '90d', alpha=-0.8, win_rate=46.0),
            create_realistic_comparison('Bollinger Bands', '180d', alpha=-1.2, win_rate=44.0),
        ],
    }

    # Flatten for alpha chart and cumulative returns
    all_comparisons = {}
    for strategy_name, comps in strategies.items():
        for comp in comps:
            key = f"{strategy_name}_{comp.horizon_name}"
            all_comparisons[key] = comp

    # Prepare heatmap data (nested dict)
    heatmap_data = {}
    for strategy_name, comps in strategies.items():
        heatmap_data[strategy_name] = {
            comp.horizon_name: comp for comp in comps
        }

    output_dir = Path("benchmark_charts_demo")
    output_dir.mkdir(exist_ok=True)

    # 1. Alpha Comparison Chart
    print("1️⃣  Creating Alpha Comparison Chart...")
    fig_alpha = create_alpha_comparison_chart(all_comparisons)
    alpha_path = output_dir / "alpha_comparison.html"
    fig_alpha.write_html(str(alpha_path))
    print(f"   ✓ Saved to {alpha_path}")

    # 2. Win Rate Heatmap
    print("2️⃣  Creating Win Rate Heatmap...")
    fig_heatmap = create_win_rate_heatmap(heatmap_data)
    heatmap_path = output_dir / "win_rate_heatmap.html"
    fig_heatmap.write_html(str(heatmap_path))
    print(f"   ✓ Saved to {heatmap_path}")

    # 3. Cumulative Returns Chart (30d horizon only)
    print("3️⃣  Creating Cumulative Returns Chart...")
    cumulative_30d = {
        k: v for k, v in all_comparisons.items()
        if v.horizon_name == '30d'
    }
    fig_cumulative = create_cumulative_returns_chart(cumulative_30d)
    cumulative_path = output_dir / "cumulative_returns_30d.html"
    fig_cumulative.write_html(str(cumulative_path))
    print(f"   ✓ Saved to {cumulative_path}")

    # 4. Return Distribution Violin Plot (30d horizon)
    print("4️⃣  Creating Return Distribution Violin Plot...")
    fig_violin = create_return_distribution_violin(cumulative_30d)
    violin_path = output_dir / "return_distribution_30d.html"
    fig_violin.write_html(str(violin_path))
    print(f"   ✓ Saved to {violin_path}")

    # 5. Create combined HTML report
    print("5️⃣  Creating Combined Report...")
    combined_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
        }}
        .chart-section {{
            margin: 30px 0;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 20px;
            background-color: #fafafa;
        }}
        .description {{
            color: #666;
            font-size: 14px;
            margin-bottom: 15px;
            line-height: 1.6;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Benchmark Comparison Report</h1>
        <p style="font-size: 16px; color: #555;">
            Interactive analysis of trading strategy performance vs buy-and-hold benchmark
        </p>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Strategies Analyzed</div>
                <div class="metric-value">{len(strategies)}</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="metric-label">Time Horizons</div>
                <div class="metric-value">3</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="metric-label">Windows Analyzed</div>
                <div class="metric-value">15</div>
            </div>
        </div>

        <div class="chart-section">
            <h2>1. Alpha Comparison</h2>
            <p class="description">
                Excess returns (alpha) for each strategy compared to buy-and-hold benchmark.
                Green bars indicate outperformance, red bars indicate underperformance.
            </p>
            <iframe src="alpha_comparison.html" width="100%" height="600" frameborder="0"></iframe>
        </div>

        <div class="chart-section">
            <h2>2. Win Rate Heatmap</h2>
            <p class="description">
                Percentage of windows where each strategy beat the benchmark, organized by time horizon.
                Darker green indicates higher win rates.
            </p>
            <iframe src="win_rate_heatmap.html" width="100%" height="600" frameborder="0"></iframe>
        </div>

        <div class="chart-section">
            <h2>3. Cumulative Returns (30-day horizon)</h2>
            <p class="description">
                Tracking how strategy returns accumulate over consecutive 30-day windows.
                Shows consistency and trend compared to benchmark.
            </p>
            <iframe src="cumulative_returns_30d.html" width="100%" height="600" frameborder="0"></iframe>
        </div>

        <div class="chart-section">
            <h2>4. Return Distribution (30-day horizon)</h2>
            <p class="description">
                Statistical distribution of per-window returns. Violin plots show the full distribution
                including outliers, with box plots overlaid showing quartiles.
            </p>
            <iframe src="return_distribution_30d.html" width="100%" height="650" frameborder="0"></iframe>
        </div>

        <div style="margin-top: 50px; padding: 20px; background-color: #e8f4f8; border-left: 4px solid #3498db;">
            <h3 style="margin-top: 0; color: #2c3e50;">📌 Key Insights</h3>
            <ul style="color: #555; line-height: 1.8;">
                <li><strong>Copula Pairs Trading</strong> shows the strongest alpha across all horizons (6.5%-8.2%)</li>
                <li><strong>MACD Strategy</strong> demonstrates consistent positive alpha with 68-70% win rates</li>
                <li><strong>RSI Mean Reversion</strong> provides moderate alpha with stable performance</li>
                <li><strong>Bollinger Bands</strong> underperforms the benchmark, suggesting need for optimization</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

    combined_path = output_dir / "index.html"
    combined_path.write_text(combined_html)
    print(f"   ✓ Saved to {combined_path}")

    print(f"\n✅ Demo Complete!")
    print(f"\n📁 All files saved to: {output_dir.absolute()}")
    print(f"🌐 Open {combined_path.absolute()} in your browser to view the report")

    # Print summary statistics
    print("\n📈 Summary Statistics:")
    for strategy_name, comps in strategies.items():
        avg_alpha = np.mean([c.alpha for c in comps])
        avg_win_rate = np.mean([c.win_rate_vs_benchmark for c in comps])
        print(f"   {strategy_name}:")
        print(f"      Average Alpha: {avg_alpha:+.2f}%")
        print(f"      Average Win Rate: {avg_win_rate:.1f}%")


if __name__ == "__main__":
    main()
