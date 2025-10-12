#!/usr/bin/env python3
"""
Compare Portfolio Backtest Results

Compares results across different portfolio rebalancing configurations
to identify which approach performs best.

Usage:
    python compare_results.py
"""

from pathlib import Path
import pandas as pd


def load_summary(results_dir: str) -> dict:
    """Load summary from a results directory."""
    summary_path = Path(results_dir) / "PORTFOLIO_SUMMARY.txt"

    if not summary_path.exists():
        return None

    with open(summary_path, 'r') as f:
        content = f.read()

    # Parse key metrics
    metrics = {}
    for line in content.split('\n'):
        if 'Run Name:' in line:
            metrics['name'] = line.split(':', 1)[1].strip()
        elif 'Description:' in line:
            metrics['description'] = line.split(':', 1)[1].strip()
        elif 'REBALANCED PORTFOLIO:' in line:
            continue
        elif 'Final Value:' in line and 'Final' not in metrics:
            value = line.split('$')[1].replace(',', '').strip()
            metrics['final_value'] = float(value)
        elif 'Total Return:' in line and 'return' not in metrics:
            ret = line.split(':')[1].strip().replace('%', '')
            metrics['return'] = float(ret)
        elif 'Rebalance Events:' in line:
            events = line.split(':')[1].strip()
            metrics['rebalances'] = int(events)
        elif 'BUY & HOLD' in line:
            continue
        elif 'Final Value:' in line and 'buyhold_value' not in metrics:
            value = line.split('$')[1].replace(',', '').strip()
            metrics['buyhold_value'] = float(value)
        elif 'Total Return:' in line and 'buyhold_return' not in metrics:
            ret = line.split(':')[1].strip().replace('%', '')
            metrics['buyhold_return'] = float(ret)

    # Calculate outperformance
    if 'return' in metrics and 'buyhold_return' in metrics:
        metrics['outperformance'] = metrics['return'] - metrics['buyhold_return']
        metrics['relative_outperformance'] = (metrics['outperformance'] / metrics['buyhold_return']) * 100

    return metrics


def main():
    """Compare all backtest results."""
    results_dirs = [
        ('results', 'Original (15% threshold)'),
        ('results_5pct', '5% Threshold'),
        ('results_10pct', '10% Threshold'),
        ('results_calendar', 'Calendar (Monthly)'),
        ('results_hybrid', 'Hybrid (Calendar + Threshold + Momentum)'),
    ]

    print("=" * 100)
    print("PORTFOLIO REBALANCING STRATEGY - PERFORMANCE COMPARISON")
    print("=" * 100)
    print()

    all_results = []

    for results_dir, label in results_dirs:
        metrics = load_summary(results_dir)
        if metrics is None:
            print(f"⏳ {label:45s} - Not yet complete")
            continue

        all_results.append({
            'Strategy': label,
            'Final Value': f"${metrics['final_value']:,.2f}",
            'Return': f"{metrics['return']:.2f}%",
            'Rebalances': metrics['rebalances'],
            'Buy&Hold Return': f"{metrics['buyhold_return']:.2f}%",
            'Outperformance': f"{metrics.get('outperformance', 0):.2f}%",
            'Relative': f"{metrics.get('relative_outperformance', 0):.2f}%"
        })

    if all_results:
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))
        print()
        print("=" * 100)

        # Find best performer
        best_idx = df['Return'].apply(lambda x: float(x.replace('%', '').replace(',', ''))).argmax()
        best = all_results[best_idx]

        print(f"\n🏆 BEST PERFORMER: {best['Strategy']}")
        print(f"   Return: {best['Return']}")
        print(f"   Outperformance vs Buy&Hold: {best['Outperformance']}")
        print(f"   Rebalance Events: {best['Rebalances']}")
        print()
    else:
        print("No results available yet. Run backtests first.")
        print()


if __name__ == "__main__":
    main()
