"""
CLI Layer for Crypto Trader

This module provides the command-line interface for the crypto trading
strategy analysis system.

**Purpose**: Clean separation of CLI from business logic

**Key Components**:
- commands: All CLI commands (analyze, etc.)

**Third-party packages**:
- typer: https://typer.tiangolo.com/

**Sample Usage**:
```python
from crypto_trader.cli import app

# Run CLI
if __name__ == "__main__":
    app()
```

**Expected Output**:
Command-line interface for strategy analysis.

Created during Phase 4 refactoring.
"""

from crypto_trader.cli.commands import app, analyze

__all__ = [
    'app',
    'analyze',
]
