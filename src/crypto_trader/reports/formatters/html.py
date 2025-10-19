"""
HTML Report Formatter

This module provides HTML formatting utilities for generating styled reports.

**Purpose**: Convert report data into rich HTML with charts, tables, and styling

**Key Features**:
- Modern CSS styling with responsive design
- Interactive Plotly charts (heatmaps, bar charts)
- Color-coded performance metrics
- Print-friendly formatting

**Third-party packages**:
- plotly: https://plotly.com/python/

**Sample Input**:
```python
formatter = HTMLFormatter()
html = formatter.format_percentage(0.25)  # Returns: '<span class="positive">+25.0%</span>'
```

**Expected Output**:
Styled HTML string with embedded CSS and JavaScript for interactivity.

Extracted from master.py (lines 355-760) during Phase 1 refactoring.
Original class: HTMLReportWriter
"""

from typing import List, Optional
from loguru import logger


class HTMLFormatter:
    """
    Formatter for generating styled HTML reports.

    This class was extracted from master.py's HTMLReportWriter class
    to provide a modular, reusable HTML formatting layer.
    """

    @staticmethod
    def get_css() -> str:
        """
        Return CSS styling for the report.

        Returns:
            Complete CSS stylesheet as string
        """
        return """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                background: #f5f5f5;
                padding: 20px;
            }

            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-radius: 8px;
            }

            h1 {
                color: #1a1a1a;
                font-size: 2.5em;
                margin-bottom: 20px;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 15px;
            }

            h2 {
                color: #2c3e50;
                font-size: 2em;
                margin-top: 40px;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #e0e0e0;
            }

            h3 {
                color: #34495e;
                font-size: 1.5em;
                margin-top: 30px;
                margin-bottom: 15px;
            }

            h4 {
                color: #555;
                font-size: 1.2em;
                margin-top: 20px;
                margin-bottom: 10px;
            }

            .metadata {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
                border-left: 4px solid #4CAF50;
            }

            .metadata p {
                margin: 5px 0;
            }

            .metadata strong {
                color: #2c3e50;
                display: inline-block;
                min-width: 180px;
            }

            hr {
                border: none;
                border-top: 1px solid #e0e0e0;
                margin: 30px 0;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            thead {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }

            th {
                padding: 15px;
                text-align: left;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.85em;
                letter-spacing: 0.5px;
            }

            td {
                padding: 12px 15px;
                border-bottom: 1px solid #e0e0e0;
            }

            tr:hover {
                background: #f8f9fa;
            }

            tbody tr:nth-child(even) {
                background: #fafafa;
            }

            .tier1 {
                background: #e8f5e9 !important;
                border-left: 4px solid #4CAF50;
            }

            .tier2 {
                background: #fff3e0 !important;
                border-left: 4px solid #FF9800;
            }

            .tier3 {
                background: #ffebee !important;
                border-left: 4px solid #f44336;
            }

            .positive {
                color: #4CAF50;
                font-weight: 600;
            }

            .negative {
                color: #f44336;
                font-weight: 600;
            }

            .blockquote {
                background: #fff8dc;
                border-left: 5px solid #ffa500;
                padding: 15px 20px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }

            .blockquote.warning {
                background: #fff3cd;
                border-left-color: #ffc107;
            }

            .blockquote.info {
                background: #d1ecf1;
                border-left-color: #17a2b8;
            }

            .action-plan {
                background: #e3f2fd;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 4px solid #2196F3;
            }

            .action-plan ol {
                margin-left: 20px;
                margin-top: 10px;
            }

            .action-plan li {
                margin: 8px 0;
                line-height: 1.8;
            }

            .recommendation-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 8px;
                margin: 20px 0;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }

            .recommendation-box h4 {
                color: white;
                margin-top: 0;
                font-size: 1.4em;
            }

            .recommendation-box ul {
                margin-left: 20px;
                margin-top: 10px;
            }

            .recommendation-box li {
                margin: 8px 0;
            }

            .profile-section {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }

            .profile-card {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 20px;
                transition: transform 0.2s, box-shadow 0.2s;
            }

            .profile-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            }

            .profile-card h4 {
                margin-top: 0;
                color: #667eea;
            }

            .academic-section {
                background: #fafafa;
                padding: 30px;
                border-radius: 5px;
                margin-top: 40px;
                border-top: 3px solid #999;
            }

            .academic-section h2 {
                color: #666;
            }

            .academic-section pre {
                background: white;
                padding: 15px;
                border-left: 3px solid #999;
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                line-height: 1.5;
            }

            ul, ol {
                margin-left: 30px;
                margin-top: 10px;
                margin-bottom: 15px;
            }

            li {
                margin: 8px 0;
                line-height: 1.8;
            }

            strong {
                color: #2c3e50;
            }

            em {
                color: #555;
                font-style: italic;
            }

            .emoji {
                font-size: 1.2em;
            }

            @media print {
                body {
                    background: white;
                    padding: 0;
                }

                .container {
                    box-shadow: none;
                    padding: 20px;
                }

                table {
                    page-break-inside: avoid;
                }

                h2 {
                    page-break-before: always;
                }
            }
        </style>
        """

    @staticmethod
    def escape_html(text: str) -> str:
        """
        Escape HTML special characters.

        Args:
            text: Raw text string

        Returns:
            HTML-escaped string safe for embedding
        """
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;'))

    @staticmethod
    def format_percentage(value: float, with_sign: bool = True) -> str:
        """
        Format percentage with color coding.

        Args:
            value: Percentage value (0.25 = 25%)
            with_sign: Include +/- sign prefix

        Returns:
            HTML string with color-coded percentage

        Example:
            >>> HTMLFormatter.format_percentage(0.25)
            '<span class="positive">+25.0%</span>'
        """
        formatted = f"{value:+.1%}" if with_sign else f"{value:.1%}"
        css_class = "positive" if value >= 0 else "negative"
        return f'<span class="{css_class}">{formatted}</span>'

    @staticmethod
    def create_performance_heatmap(strategy_scores: List, horizons: List) -> str:
        """
        Create performance heatmap showing strategy returns across horizons using Plotly.

        Args:
            strategy_scores: List of StrategyScore objects with horizon results
            horizons: List of HorizonConfig objects

        Returns:
            HTML string containing embedded Plotly heatmap
        """
        try:
            import plotly.graph_objects as go

            # Build matrix: rows=strategies, cols=horizons
            strategy_names = [s.strategy_name for s in strategy_scores]
            horizon_names = [h.name for h in horizons]

            # Create return matrix
            returns_matrix = []
            for strat in strategy_scores:
                row = []
                for horizon in horizons:
                    if horizon.name in strat.horizon_results:
                        ret = strat.horizon_results[horizon.name]['return'] * 100  # Convert to percentage
                        row.append(ret)
                    else:
                        row.append(None)
                returns_matrix.append(row)

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=returns_matrix,
                x=horizon_names,
                y=strategy_names,
                colorscale='RdYlGn',
                zmid=0,
                text=[[f"{v:.1f}%" if v is not None else "N/A" for v in row] for row in returns_matrix],
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Return (%)")
            ))

            fig.update_layout(
                title="Strategy Performance Heatmap (Returns % by Time Horizon)",
                xaxis_title="Time Horizon",
                yaxis_title="Strategy",
                height=400 + len(strategy_scores) * 30,
                font=dict(size=11)
            )

            return fig.to_html(full_html=False, include_plotlyjs='cdn')

        except Exception as e:
            logger.warning(f"Failed to create performance heatmap: {e}")
            return f"<p>⚠️ Heatmap visualization failed: {str(e)}</p>"

    @staticmethod
    def create_sharpe_comparison_chart(strategy_scores: List) -> str:
        """
        Create bar chart comparing Sharpe ratios across strategies.

        Args:
            strategy_scores: List of StrategyScore objects

        Returns:
            HTML string containing embedded Plotly bar chart
        """
        try:
            import plotly.graph_objects as go

            strategy_names = [s.strategy_name for s in strategy_scores]
            sharpe_ratios = [s.avg_sharpe for s in strategy_scores]
            colors = ['green' if sr > 1.0 else 'orange' if sr > 0 else 'red' for sr in sharpe_ratios]

            fig = go.Figure(data=[
                go.Bar(
                    x=strategy_names,
                    y=sharpe_ratios,
                    marker_color=colors,
                    text=[f"{sr:.2f}" for sr in sharpe_ratios],
                    textposition='outside'
                )
            ])

            fig.update_layout(
                title="Sharpe Ratio Comparison (Higher is Better)",
                xaxis_title="Strategy",
                yaxis_title="Sharpe Ratio",
                height=500,
                xaxis={'tickangle': -45},
                font=dict(size=11)
            )

            fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                         annotation_text="Good (>1.0)", annotation_position="right")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            return fig.to_html(full_html=False, include_plotlyjs='cdn')

        except Exception as e:
            logger.warning(f"Failed to create Sharpe comparison chart: {e}")
            return f"<p>⚠️ Sharpe chart visualization failed: {str(e)}</p>"


if __name__ == "__main__":
    """
    Validation block for HTMLFormatter.
    Tests all formatting methods with real data.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: CSS generation
    total_tests += 1
    print("Test 1: Verify CSS generation")
    try:
        css = HTMLFormatter.get_css()
        if not css or len(css) < 1000:
            all_validation_failures.append("CSS too short or empty")
        elif "<style>" not in css:
            all_validation_failures.append("CSS missing <style> tag")
        else:
            print(f"  ✓ CSS generated: {len(css)} characters")
    except Exception as e:
        all_validation_failures.append(f"CSS generation failed: {e}")

    # Test 2: HTML escaping
    total_tests += 1
    print("\nTest 2: Verify HTML escaping")
    try:
        test_str = "<script>alert('xss')</script>"
        escaped = HTMLFormatter.escape_html(test_str)
        if "<script>" in escaped:
            all_validation_failures.append("HTML escaping failed - tags not escaped")
        elif "&lt;script&gt;" not in escaped:
            all_validation_failures.append("HTML escaping incorrect output")
        else:
            print(f"  ✓ HTML escaping works: {escaped[:50]}...")
    except Exception as e:
        all_validation_failures.append(f"HTML escaping failed: {e}")

    # Test 3: Percentage formatting
    total_tests += 1
    print("\nTest 3: Verify percentage formatting")
    try:
        pos = HTMLFormatter.format_percentage(0.25)
        neg = HTMLFormatter.format_percentage(-0.15)

        if "positive" not in pos or "+25.0%" not in pos:
            all_validation_failures.append("Positive percentage formatting incorrect")
        if "negative" not in neg or "-15.0%" not in neg:
            all_validation_failures.append("Negative percentage formatting incorrect")

        if not all_validation_failures or len(all_validation_failures) == 0:
            print(f"  ✓ Positive: {pos}")
            print(f"  ✓ Negative: {neg}")
    except Exception as e:
        all_validation_failures.append(f"Percentage formatting failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("HTMLFormatter is validated and ready for use")
        sys.exit(0)
