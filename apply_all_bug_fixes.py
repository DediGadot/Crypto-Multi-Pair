#!/usr/bin/env python3
"""
LINUS-STYLE BUG FIXER
====================
Fix ALL bugs with surgical precision and PROOF.

"Talk is cheap. Show me the code." - Linus Torvalds
"""

import re
import subprocess
from pathlib import Path
from typing import List, Tuple

class BugFixer:
    def __init__(self):
        self.fixes_applied = []
        self.errors_found = []

    def log_fix(self, file: str, description: str, before: str, after: str):
        """Log a fix with before/after proof"""
        self.fixes_applied.append({
            "file": file,
            "description": description,
            "before": before[:100],
            "after": after[:100]
        })
        print(f"✅ FIXED: {file} - {description}")

    def log_error(self, file: str, error: str):
        """Log an error encountered"""
        self.errors_found.append({"file": file, "error": error})
        print(f"❌ ERROR: {file} - {error}")

    def fix_pandas_iloc_issues(self) -> int:
        """
        Fix #1: Pandas .iloc[] assignment issues

        Problem: result.iloc[-1, result.columns.get_loc("col")] = value
        Solution: result.at[result.index[-1], "col"] = value
        """
        print("\n" + "="*80)
        print("FIX #1: Pandas .iloc[] Assignment Issues")
        print("="*80)

        fixed_count = 0
        file_path = Path("src/crypto_trader/strategies/library/ddqn_feature_selected.py")

        if not file_path.exists():
            self.log_error(str(file_path), "File not found")
            return 0

        content = file_path.read_text()
        original = content

        # Pattern 1: result.iloc[-1, result.columns.get_loc("signal")] = signal_value
        pattern1 = r'result\.iloc\[-1,\s*result\.columns\.get_loc\("(\w+)"\)\]\s*=\s*(\w+)'
        replacement1 = r'result.at[result.index[-1], "\1"] = \2'

        content, count1 = re.subn(pattern1, replacement1, content)
        fixed_count += count1

        if content != original:
            file_path.write_text(content)
            self.log_fix(
                str(file_path),
                f"Fixed {count1} .iloc[] assignments",
                "result.iloc[-1, result.columns.get_loc(col)] = val",
                "result.at[result.index[-1], col] = val"
            )

        return fixed_count

    def fix_sharpe_ratio_zero_variance(self) -> int:
        """
        Fix #2: Sharpe Ratio Zero Variance Handling

        Problem: Raises error for zero variance even when strategy makes no trades
        Solution: Return 0.0 for no trades, raise error only for trades with zero variance
        """
        print("\n" + "="*80)
        print("FIX #2: Sharpe Ratio Zero Variance Handling")
        print("="*80)

        file_path = Path("master.py")
        content = file_path.read_text()
        lines = content.split('\n')

        # Find the calculate_sharpe function
        fixed = False
        for i, line in enumerate(lines):
            if 'def calculate_sharpe' in line:
                # Look for the zero variance check
                for j in range(i, min(i+50, len(lines))):
                    if 'if std_return <=' in lines[j] and 'raise ValueError' in lines[j+1]:
                        # Found it! Fix it
                        indent = len(lines[j]) - len(lines[j].lstrip())

                        # Replace the raise with proper handling
                        lines[j] = ' ' * indent + 'if std_return <= 1e-8:'
                        lines[j+1] = ' ' * (indent + 4) + 'if num_trades == 0:'
                        lines.insert(j+2, ' ' * (indent + 8) + 'return 0.0  # No trades = Sharpe of 0')
                        lines.insert(j+3, ' ' * (indent + 4) + 'else:')
                        lines.insert(j+4, ' ' * (indent + 8) + 'raise ValueError(')
                        lines.insert(j+5, ' ' * (indent + 12) + 'f"Zero variance with {num_trades} trades = broken strategy"')
                        lines.insert(j+6, ' ' * (indent + 8) + ')')
                        fixed = True
                        break
                break

        if fixed:
            file_path.write_text('\n'.join(lines))
            self.log_fix(
                str(file_path),
                "Fixed Sharpe ratio to handle no-trade strategies",
                "if std <= 0: raise ValueError(...)",
                "if std <= 0: return 0.0 if no trades else raise"
            )
            return 1

        return 0

    def fix_process_pool_fallback(self) -> int:
        """
        Fix #3: Add ProcessPool → ThreadPool Fallback

        Problem: ProcessPoolExecutor fails silently, breaking all strategies
        Solution: Catch PermissionError and fall back to ThreadPoolExecutor
        """
        print("\n" + "="*80)
        print("FIX #3: ProcessPool → ThreadPool Fallback")
        print("="*80)

        file_path = Path("master.py")
        content = file_path.read_text()

        # Find the ProcessPoolExecutor usage
        pattern = r'with ProcessPoolExecutor\(max_workers=self\.workers\) as executor:'

        if pattern in content:
            # Already has try/except
            if 'except (PermissionError, OSError)' in content:
                print("  ℹ️  Fallback already implemented")
                return 0

            # Need to add fallback
            replacement = '''try:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:'''

            content = content.replace(
                'with ProcessPoolExecutor(max_workers=self.workers) as executor:',
                replacement
            )

            # Find the end of the with block and add except clause
            # This is complex, so let's do it manually in a separate step
            print("  ⚠️  Manual intervention needed for ProcessPool fallback")
            print("  →  Add: except (PermissionError, OSError) + ThreadPoolExecutor fallback")
            return 0

        return 0

    def fix_data_slicing_consistency(self) -> int:
        """
        Fix #4: Apply Consistent Data Slicing Across All Workers

        Problem: Multi-pair workers use _slice_data_to_horizon(), single-pair don't
        Solution: Apply slicing consistently to all workers
        """
        print("\n" + "="*80)
        print("FIX #4: Consistent Data Slicing")
        print("="*80)

        file_path = Path("master.py")
        content = file_path.read_text()

        # Check if single-pair worker slices data
        if '_slice_data_to_horizon' in content:
            # Count occurrences
            count = content.count('_slice_data_to_horizon')
            print(f"  ℹ️  Found {count} uses of _slice_data_to_horizon")

            # Check if it's used in run_backtest_worker
            if 'def run_backtest_worker' in content:
                worker_start = content.find('def run_backtest_worker')
                worker_end = content.find('\ndef ', worker_start + 1)
                worker_code = content[worker_start:worker_end]

                if '_slice_data_to_horizon' in worker_code:
                    print("  ✅ Single-pair worker already slices data")
                    return 0
                else:
                    print("  ⚠️  Single-pair worker missing data slicing")
                    print("  →  Need to add slicing in run_backtest_worker")
                    return 0

        return 0

    def generate_evidence_report(self):
        """Generate comprehensive evidence report"""
        print("\n" + "="*80)
        print("EVIDENCE REPORT")
        print("="*80)

        print(f"\n✅ Fixes Applied: {len(self.fixes_applied)}")
        for fix in self.fixes_applied:
            print(f"\n  File: {fix['file']}")
            print(f"  Fix: {fix['description']}")
            print(f"  Before: {fix['before']}...")
            print(f"  After: {fix['after']}...")

        print(f"\n❌ Errors Found: {len(self.errors_found)}")
        for error in self.errors_found:
            print(f"\n  File: {error['file']}")
            print(f"  Error: {error['error']}")

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║           LINUS-STYLE BUG FIXER - ULTRATHINK MODE          ║
    ║                                                            ║
    ║  "Talk is cheap. Show me the code."  - Linus Torvalds      ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    fixer = BugFixer()

    total_fixes = 0
    total_fixes += fixer.fix_pandas_iloc_issues()
    total_fixes += fixer.fix_sharpe_ratio_zero_variance()
    total_fixes += fixer.fix_process_pool_fallback()
    total_fixes += fixer.fix_data_slicing_consistency()

    fixer.generate_evidence_report()

    print("\n" + "="*80)
    print(f"TOTAL FIXES APPLIED: {total_fixes}")
    print("="*80)

    if total_fixes > 0:
        print("\n✅ Fixes applied successfully!")
        print("\nNext steps:")
        print("  1. Review changes with: git diff")
        print("  2. Test with: uv run python master.py -h 30 --quick --workers 2")
        print("  3. Compare results before/after")
    else:
        print("\n⚠️  No automatic fixes applied")
        print("  Some fixes require manual intervention")

if __name__ == "__main__":
    main()
