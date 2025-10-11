"""
Basic Strategy Comparison Example

Demonstrates how to:
1. Load strategies
2. Run comparison
3. Display results

Usage:
    uv run python examples/basic_comparison.py
"""

from crypto_strategy_comparison.strategy_loader import StrategyLoader
from crypto_strategy_comparison.comparison_engine import ComparisonEngine
from loguru import logger


def main():
    """Run basic comparison example."""
    logger.info("Starting basic comparison example...")

    # Step 1: Initialize components
    loader = StrategyLoader()
    engine = ComparisonEngine()

    # Step 2: Get available strategies
    available = loader.get_available_strategies()
    print("\n📊 Available Strategies:")
    for i, strategy in enumerate(available, 1):
        print(f"  {i}. {strategy}")

    # Step 3: Select strategies to compare
    strategies_to_compare = available[:3]  # Compare first 3
    print(f"\n🎯 Comparing: {', '.join(strategies_to_compare)}")

    # Step 4: Load strategy data
    print("\n📥 Loading strategy data...")
    strategies_data = loader.load_strategies(strategies_to_compare)

    # Step 5: Run comparison
    print("\n⚡ Running comparison analysis...")
    results = engine.compare(strategies_data, time_horizon="6M")

    # Step 6: Display results
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)

    print(f"\nTime Horizon: {results['time_horizon']}")
    print(f"Strategies Compared: {results['strategy_count']}")

    print("\n📈 Performance Metrics:")
    print("-" * 60)
    print(f"{'Strategy':<25} {'Return':<12} {'Sharpe':<10} {'Max DD'}")
    print("-" * 60)

    for strategy, metrics in results["metrics"].items():
        print(
            f"{strategy:<25} "
            f"{metrics.get('total_return', 0):>10.2f}% "
            f"{metrics.get('sharpe_ratio', 0):>8.2f}  "
            f"{metrics.get('max_drawdown', 0):>10.2f}%"
        )

    print("-" * 60)

    # Best performers
    best_return = max(
        results["metrics"].items(),
        key=lambda x: x[1].get("total_return", 0)
    )
    best_sharpe = max(
        results["metrics"].items(),
        key=lambda x: x[1].get("sharpe_ratio", 0)
    )

    print("\n🏆 Best Performers:")
    print(f"  Highest Return: {best_return[0]} ({best_return[1]['total_return']:.2f}%)")
    print(f"  Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.2f})")

    print("\n✅ Example completed successfully!")
    print("\n💡 Next steps:")
    print("  1. Run the full Streamlit dashboard: streamlit run src/crypto_strategy_comparison/app.py")
    print("  2. Explore more strategies in the UI")
    print("  3. Export reports for detailed analysis")


if __name__ == "__main__":
    main()
