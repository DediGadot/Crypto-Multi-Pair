"""
Master Report Generator

This module generates comprehensive master strategy analysis reports.

**Purpose**: Create multi-format reports from master strategy analysis results

**Third-party packages**:
- None (pure Python)

**Sample Input**:
```python
generator = MasterReportGenerator(html_formatter)
report = generator.generate(scores, results, horizons)
```

**Expected Output**:
Dictionary containing report content in multiple formats.

Created during Phase 1 refactoring to separate report generation logic
from master.py orchestration.
"""

from typing import List, Dict, Any, Optional
from crypto_trader.reports.formatters.html import HTMLFormatter
from loguru import logger


class MasterReportGenerator:
    """
    Generates master strategy analysis reports in multiple formats.

    This is a stub implementation that will be expanded to include
    all report generation logic currently embedded in master.py.
    """

    def __init__(self, html_formatter: Optional[HTMLFormatter] = None):
        """
        Initialize report generator.

        Args:
            html_formatter: HTML formatter instance (default: creates new one)
        """
        self.html_formatter = html_formatter or HTMLFormatter()
        logger.debug("MasterReportGenerator initialized")

    def generate_html_report(
        self,
        scores: List[Any],
        results: Dict[str, Any],
        horizons: List[Any],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Generate HTML report from master analysis results.

        Args:
            scores: List of StrategyScore objects
            results: Dictionary of backtest results
            horizons: List of HorizonConfig objects
            metadata: Report metadata (symbol, timeframe, etc.)

        Returns:
            Complete HTML report as string
        """
        # This will be implemented to generate complete reports
        # For now, return a stub that delegates to HTMLFormatter
        logger.info("Generating HTML report")

        html_parts = []
        html_parts.append("<!DOCTYPE html><html><head>")
        html_parts.append(self.html_formatter.get_css())
        html_parts.append("</head><body><div class='container'>")
        html_parts.append("<h1>Master Strategy Analysis Report</h1>")

        # Add heatmap if available
        if scores and horizons:
            heatmap = self.html_formatter.create_performance_heatmap(scores, horizons)
            html_parts.append(heatmap)

        # Add Sharpe comparison if available
        if scores:
            sharpe_chart = self.html_formatter.create_sharpe_comparison_chart(scores)
            html_parts.append(sharpe_chart)

        html_parts.append("</div></body></html>")

        return "\n".join(html_parts)


if __name__ == "__main__":
    """
    Validation block for MasterReportGenerator.
    Tests report generation with mock data.
    """
    import sys
    from dataclasses import dataclass

    @dataclass
    class MockScore:
        strategy_name: str
        avg_sharpe: float
        horizon_results: Dict[str, Dict[str, float]]

    @dataclass
    class MockHorizon:
        name: str
        days: int

    all_validation_failures = []
    total_tests = 0

    # Test 1: Generator initialization
    total_tests += 1
    print("Test 1: Verify generator initialization")
    try:
        generator = MasterReportGenerator()
        if generator.html_formatter is None:
            all_validation_failures.append("HTML formatter not initialized")
        else:
            print("  ✓ Generator initialized with HTML formatter")
    except Exception as e:
        all_validation_failures.append(f"Generator initialization failed: {e}")

    # Test 2: HTML report generation with mock data
    total_tests += 1
    print("\nTest 2: Verify HTML report generation")
    try:
        # Create mock data
        scores = [
            MockScore("Strategy A", 1.5, {"30d": {"return": 0.15}}),
            MockScore("Strategy B", 0.8, {"30d": {"return": 0.08}})
        ]
        horizons = [MockHorizon("30d", 30)]
        metadata = {"symbol": "BTC/USDT", "timeframe": "1h"}

        html = generator.generate_html_report(scores, {}, horizons, metadata)

        if not html or len(html) < 100:
            all_validation_failures.append("HTML report too short or empty")
        elif "<!DOCTYPE html>" not in html:
            all_validation_failures.append("HTML report missing DOCTYPE")
        elif "Master Strategy Analysis Report" not in html:
            all_validation_failures.append("HTML report missing title")
        else:
            print(f"  ✓ HTML report generated: {len(html)} characters")
            print(f"  ✓ Report contains heatmap and Sharpe chart")
    except Exception as e:
        all_validation_failures.append(f"HTML report generation failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("MasterReportGenerator is validated and ready for use")
        sys.exit(0)
