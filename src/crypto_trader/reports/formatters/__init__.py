"""
Report Formatters - Format-specific rendering engines

This module contains formatters that convert report data into various output formats.

**Formatters**:
- HTMLFormatter: Rich HTML reports with charts and styling
- TextFormatter: Plain text reports for console/logs (TODO)
- JSONFormatter: Machine-readable JSON exports (TODO)
"""

from crypto_trader.reports.formatters.html import HTMLFormatter

__all__ = ["HTMLFormatter"]
