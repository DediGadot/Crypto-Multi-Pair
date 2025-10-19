"""
Reports Module - Modular Report Generation System

This module provides a clean, separated reporting layer for the crypto trading system.
Previously, all reporting logic was embedded in master.py (405+ lines). Now it's
properly modularized for maintainability and testability.

**Purpose**: Generate formatted reports (HTML, text, JSON) from backtest results

**Architecture**:
- formatters/: Format-specific rendering (HTML, text, JSON)
- generators/: Business logic for report content
- templates/: Reusable report templates

**Third-party packages**:
- plotly: https://plotly.com/python/ (for visualizations)
- loguru: https://loguru.readthedocs.io/ (for logging)

**Usage**:
```python
from crypto_trader.reports.formatters.html import HTMLFormatter
from crypto_trader.reports.generators.master_report import MasterReportGenerator

formatter = HTMLFormatter()
generator = MasterReportGenerator(formatter)
report = generator.generate(scores, results)
```

Extracted from master.py lines 355-760 during Phase 1 refactoring.
"""

from crypto_trader.reports.formatters.html import HTMLFormatter
from crypto_trader.reports.generators.master_report import MasterReportGenerator

__all__ = [
    "HTMLFormatter",
    "MasterReportGenerator",
]
