#!/usr/bin/env python3
"""
Master Strategy Analysis - Entry Point

Thin wrapper that launches the crypto trading strategy analysis CLI.
All business logic has been extracted to modular components.

**Architecture** (after Phase 4 refactoring):
- CLI layer: src/crypto_trader/cli/
- Orchestration: src/crypto_trader/orchestration/
- Execution: src/crypto_trader/execution/
- Reports: src/crypto_trader/reports/
- Core, Data, Strategies, etc.: src/crypto_trader/

**Usage**:
```bash
# Standard analysis
python master.py --symbol BTC/USDT

# Quick analysis
python master.py --quick

# Custom configuration
python master.py --symbol ETH/USDT --workers 8 --horizons 30 90 180 365

# Multi-pair strategies
python master.py --multi-pair --quick
```

**Refactoring History**:
- Phase 1: Extracted reports module (777 lines)
- Phase 2: Created execution module structure (193 lines)
- Phase 2.5: Fully extracted execution module (2,204 lines total)
- Phase 3: Extracted orchestration module (2,714 lines)
- Phase 4: Extracted CLI module (this file reduced from 407 to <100 lines)

Original master.py: 4,588 lines
Current master.py: <100 lines
Total modular code: ~6,000+ lines across 4 modules

**No bullshit. Clean code. Working system.**
"""

import sys
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from crypto_trader.cli import app


if __name__ == "__main__":
    """
    Entry point for crypto trading strategy analysis.
    
    All CLI commands are defined in crypto_trader.cli.commands.
    All orchestration logic is in crypto_trader.orchestration.
    All execution logic is in crypto_trader.execution.
    """
    app()
