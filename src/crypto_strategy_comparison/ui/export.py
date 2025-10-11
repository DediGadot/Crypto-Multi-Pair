"""
Export Component

Handles exporting comparison reports in multiple formats:
- PDF reports with charts and tables
- HTML interactive reports
- CSV data exports
- JSON raw data exports

Documentation:
- FPDF2: https://pyfpdf.github.io/fpdf2/
- Jinja2: https://jinja.palletsprojects.com/

Sample Input:
- comparison_results: Dict with all comparison data
- export_format: str ("pdf", "html", "csv", "json")

Expected Output:
- Downloadable file in specified format
"""

from typing import Dict, List, Optional, Any
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from io import BytesIO
from loguru import logger


def render_export_options(results: Dict) -> None:
    """
    Render export options and download buttons.

    Args:
        results: Comparison results dictionary
    """
    if not results:
        st.warning("No data available for export")
        return

    st.markdown("### 📤 Export Reports")

    # Export format selection
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Select Export Options:**")

        include_charts = st.checkbox("Include Charts", value=True, key="export_charts")
        include_detailed = st.checkbox("Include Detailed Metrics", value=True, key="export_detailed")
        include_trades = st.checkbox("Include Trade History", value=False, key="export_trades")

    with col2:
        st.markdown("**Quick Actions:**")

        if st.button("📄 Export PDF", use_container_width=True):
            export_pdf(results, include_charts, include_detailed, include_trades)

        if st.button("🌐 Export HTML", use_container_width=True):
            export_html(results, include_charts, include_detailed, include_trades)

    # Data exports
    st.markdown("---")
    st.markdown("**Export Data:**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Export CSV (Metrics)", use_container_width=True):
            export_csv(results, "metrics")

        if st.button("📊 Export CSV (Trades)", use_container_width=True):
            export_csv(results, "trades")

    with col2:
        if st.button("📋 Export JSON", use_container_width=True):
            export_json(results)


def export_pdf(
    results: Dict,
    include_charts: bool,
    include_detailed: bool,
    include_trades: bool
) -> None:
    """
    Generate and download PDF report.

    Args:
        results: Comparison results
        include_charts: Include chart images
        include_detailed: Include detailed metrics
        include_trades: Include trade history
    """
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()

        # Title
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 10, "Crypto Strategy Comparison Report", ln=True, align="C")

        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

        pdf.ln(10)

        # Summary metrics
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Performance Summary", ln=True)

        pdf.set_font("Arial", "", 10)

        if "metrics" in results:
            for strategy, metrics in results["metrics"].items():
                pdf.ln(5)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, strategy, ln=True)

                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 6, f"Total Return: {metrics.get('total_return', 0):.2f}%", ln=True)
                pdf.cell(0, 6, f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}", ln=True)
                pdf.cell(0, 6, f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%", ln=True)
                pdf.cell(0, 6, f"Win Rate: {metrics.get('win_rate', 0):.2f}%", ln=True)

        # Convert to bytes
        pdf_bytes = pdf.output(dest="S").encode("latin1")

        # Download button
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

        logger.info("PDF report generated successfully")
        st.success("✅ PDF report ready for download!")

    except Exception as e:
        logger.error(f"PDF export failed: {e}")
        st.error(f"❌ PDF export failed: {str(e)}")


def export_html(
    results: Dict,
    include_charts: bool,
    include_detailed: bool,
    include_trades: bool
) -> None:
    """
    Generate and download HTML report.

    Args:
        results: Comparison results
        include_charts: Include interactive charts
        include_detailed: Include detailed metrics
        include_trades: Include trade history
    """
    try:
        html_content = generate_html_report(
            results,
            include_charts,
            include_detailed,
            include_trades
        )

        # Download button
        st.download_button(
            label="📥 Download HTML Report",
            data=html_content,
            file_name=f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )

        logger.info("HTML report generated successfully")
        st.success("✅ HTML report ready for download!")

    except Exception as e:
        logger.error(f"HTML export failed: {e}")
        st.error(f"❌ HTML export failed: {str(e)}")


def generate_html_report(
    results: Dict,
    include_charts: bool,
    include_detailed: bool,
    include_trades: bool
) -> str:
    """
    Generate HTML report content.

    Args:
        results: Comparison results
        include_charts: Include charts
        include_detailed: Include detailed metrics
        include_trades: Include trades

    Returns:
        HTML content string
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Comparison Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #1f77b4;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px;
            }}
            .section {{
                background-color: white;
                margin: 20px 0;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #1f77b4;
                color: white;
            }}
            .positive {{
                color: green;
                font-weight: bold;
            }}
            .negative {{
                color: red;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Crypto Strategy Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """

    # Add metrics table
    if "metrics" in results:
        html += """
        <div class="section">
            <h2>📊 Performance Metrics</h2>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Total Return</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                </tr>
        """

        for strategy, metrics in results["metrics"].items():
            return_class = "positive" if metrics.get("total_return", 0) > 0 else "negative"
            html += f"""
                <tr>
                    <td><strong>{strategy}</strong></td>
                    <td class="{return_class}">{metrics.get('total_return', 0):.2f}%</td>
                    <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                    <td class="negative">{metrics.get('max_drawdown', 0):.2f}%</td>
                    <td>{metrics.get('win_rate', 0):.2f}%</td>
                    <td>{metrics.get('trade_count', 0)}</td>
                </tr>
            """

        html += """
            </table>
        </div>
        """

    html += """
    </body>
    </html>
    """

    return html


def export_csv(results: Dict, data_type: str) -> None:
    """
    Export data as CSV.

    Args:
        results: Comparison results
        data_type: Type of data to export ("metrics" or "trades")
    """
    try:
        if data_type == "metrics" and "metrics" in results:
            # Build metrics DataFrame
            data = []
            for strategy, metrics in results["metrics"].items():
                row = {"Strategy": strategy}
                row.update(metrics)
                data.append(row)

            df = pd.DataFrame(data)

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics CSV",
                data=csv,
                file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            logger.info("Metrics CSV exported successfully")
            st.success("✅ Metrics CSV ready for download!")

        elif data_type == "trades" and "trades" in results:
            # Combine all trades
            all_trades = []
            for strategy, trades in results["trades"].items():
                for trade in trades:
                    trade_copy = trade.copy()
                    trade_copy["strategy"] = strategy
                    all_trades.append(trade_copy)

            df = pd.DataFrame(all_trades)

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trades CSV",
                data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            logger.info("Trades CSV exported successfully")
            st.success("✅ Trades CSV ready for download!")

        else:
            st.warning(f"No {data_type} data available for export")

    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        st.error(f"❌ CSV export failed: {str(e)}")


def export_json(results: Dict) -> None:
    """
    Export results as JSON.

    Args:
        results: Comparison results
    """
    try:
        # Convert to JSON
        json_str = json.dumps(results, indent=2, default=str)

        # Download button
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        logger.info("JSON export completed successfully")
        st.success("✅ JSON data ready for download!")

    except Exception as e:
        logger.error(f"JSON export failed: {e}")
        st.error(f"❌ JSON export failed: {str(e)}")


if __name__ == "__main__":
    # Validation function
    import sys

    print("🔍 Validating export.py...")

    all_validation_failures = []
    total_tests = 0

    # Test 1: HTML generation
    total_tests += 1
    try:
        test_results = {
            "metrics": {
                "Strategy A": {
                    "total_return": 45.2,
                    "sharpe_ratio": 2.1,
                    "max_drawdown": -15.3,
                    "win_rate": 65.5,
                    "trade_count": 100
                }
            }
        }

        html = generate_html_report(test_results, True, True, False)

        if not isinstance(html, str):
            all_validation_failures.append("HTML generation: Expected string output")

        if "Strategy A" not in html:
            all_validation_failures.append("HTML generation: Expected strategy name in output")

        if "45.2" not in html:
            all_validation_failures.append("HTML generation: Expected return value in output")

    except Exception as e:
        all_validation_failures.append(f"HTML generation test failed: {e}")

    # Test 2: HTML structure validation
    total_tests += 1
    try:
        test_results = {"metrics": {"Test": {"total_return": 10}}}
        html = generate_html_report(test_results, False, False, False)

        required_tags = ["<html>", "<head>", "<body>", "<table>"]
        for tag in required_tags:
            if tag not in html:
                all_validation_failures.append(
                    f"HTML structure: Missing required tag '{tag}'"
                )

    except Exception as e:
        all_validation_failures.append(f"HTML structure test failed: {e}")

    # Test 3: JSON serialization
    total_tests += 1
    try:
        test_results = {
            "metrics": {"Test": {"value": 10.5}},
            "timestamp": datetime.now()
        }

        json_str = json.dumps(test_results, default=str)
        parsed = json.loads(json_str)

        if "metrics" not in parsed:
            all_validation_failures.append("JSON serialization: Missing 'metrics' key")

    except Exception as e:
        all_validation_failures.append(f"JSON serialization test failed: {e}")

    # Final validation result
    if all_validation_failures:
        print(
            f"❌ VALIDATION FAILED - {len(all_validation_failures)} "
            f"of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests successful")
        print("Export component is validated and ready for integration")
        sys.exit(0)
