#!/usr/bin/env python3
"""
Quick Fix: ProcessPoolExecutor Permission Error
================================================

This script patches master.py to add ThreadPoolExecutor fallback
when ProcessPoolExecutor fails due to permission errors.

Issue: ProcessPoolExecutor raises PermissionError preventing 7 strategies
from initializing, causing them to fail with "Strategy not initialized"

Fix: Add graceful fallback to ThreadPoolExecutor when process pool fails

Usage:
    uv run python fix_process_pool_issue.py
"""

import re
from pathlib import Path
from loguru import logger

MASTER_PY_PATH = Path(__file__).parent / "master.py"


def backup_file(file_path: Path) -> Path:
    """Create backup of original file."""
    backup_path = file_path.with_suffix('.py.backup')
    backup_path.write_text(file_path.read_text())
    logger.info(f"Created backup: {backup_path}")
    return backup_path


def apply_fix(content: str) -> str:
    """
    Apply the ProcessPoolExecutor fallback fix.

    Wraps ProcessPoolExecutor usage in try-except with ThreadPoolExecutor fallback.
    """

    # Pattern to find the ProcessPoolExecutor usage
    pattern = r'(\s+)(def _run_parallel\(pbar_obj\) -> None:\s+)(with ProcessPoolExecutor\(max_workers=self\.workers\) as executor:)'

    # Replacement with fallback logic
    replacement = r'''\1\2try:
\1    \3
\1except (PermissionError, OSError) as pool_error:
\1    logger.warning(f"Process pool unavailable ({pool_error}); falling back to ThreadPoolExecutor")
\1    from concurrent.futures import ThreadPoolExecutor
\1    with ThreadPoolExecutor(max_workers=self.workers) as executor:'''

    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Verify fix was applied
    if fixed_content == content:
        logger.warning("Pattern not found - fix may need manual application")
        return None

    # Add proper indentation for the rest of the block
    # Find the executor block and ensure consistent indentation
    lines = fixed_content.split('\n')
    fixed_lines = []
    in_executor_block = False
    indent_level = 0

    for line in lines:
        if 'with ThreadPoolExecutor(max_workers=self.workers) as executor:' in line:
            in_executor_block = True
            indent_level = len(line) - len(line.lstrip()) + 4
            fixed_lines.append(line)
        elif in_executor_block and line.strip().startswith('futures = {}'):
            # Ensure proper indentation for executor block content
            fixed_lines.append(' ' * indent_level + line.lstrip())
            in_executor_block = False  # Only the first line after executor needs adjustment
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def verify_import_exists(content: str) -> tuple[str, bool]:
    """
    Verify ThreadPoolExecutor is imported at module level.
    Add import if missing.
    """
    if 'ThreadPoolExecutor' in content.split('from concurrent.futures import')[1].split('\n')[0]:
        logger.info("ThreadPoolExecutor already imported at module level")
        return content, False

    # Find the concurrent.futures import line and update it
    pattern = r'from concurrent\.futures import (.*?)(\n)'
    match = re.search(pattern, content)

    if match:
        imports = match.group(1)
        if 'ThreadPoolExecutor' not in imports:
            new_imports = imports + ', ThreadPoolExecutor'
            content = content.replace(
                f'from concurrent.futures import {imports}',
                f'from concurrent.futures import {new_imports}'
            )
            logger.info("Added ThreadPoolExecutor to module imports")
            return content, True
    else:
        logger.warning("Could not find concurrent.futures import")

    return content, False


def add_validation_test(content: str) -> str:
    """
    Add pre-flight strategy validation before worker submission.
    """

    validation_function = '''
def _verify_strategy_can_initialize(strategy_name: str, config: Dict[str, Any]) -> bool:
    """
    Pre-flight check to ensure strategy can initialize before submitting to worker pool.

    This prevents PermissionError cascades where strategies fail to initialize
    due to process pool issues.
    """
    try:
        from crypto_trader.strategies import get_strategy
        StrategyClass = get_strategy(strategy_name)
        if StrategyClass is None:
            logger.warning(f"Strategy {strategy_name} not found in registry")
            return False

        # Try to create and initialize
        strategy = StrategyClass()
        strategy.initialize(config)

        if not hasattr(strategy, '_initialized') or not strategy._initialized:
            logger.warning(f"Strategy {strategy_name} has _initialized=False after initialize()")
            return False

        logger.debug(f"Pre-flight check passed for {strategy_name}")
        return True

    except Exception as e:
        logger.error(f"Pre-flight check failed for {strategy_name}: {e}")
        return False

'''

    # Find a good place to insert (after imports, before first function)
    # Look for the first def or class definition
    lines = content.split('\n')
    insert_index = 0

    for i, line in enumerate(lines):
        if line.startswith('def ') or line.startswith('class '):
            insert_index = i
            break

    # Insert the validation function
    lines.insert(insert_index, validation_function)
    return '\n'.join(lines)


def main():
    """Apply the fix to master.py"""

    logger.info("="*70)
    logger.info("ProcessPoolExecutor Permission Error Fix")
    logger.info("="*70)

    # Check if master.py exists
    if not MASTER_PY_PATH.exists():
        logger.error(f"master.py not found at {MASTER_PY_PATH}")
        return False

    # Read current content
    logger.info(f"Reading {MASTER_PY_PATH}")
    content = MASTER_PY_PATH.read_text()

    # Check if already fixed
    if 'ThreadPoolExecutor(max_workers=self.workers) as executor' in content:
        logger.warning("Fix may already be applied - ThreadPoolExecutor fallback found in code")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted")
            return False

    # Create backup
    backup_path = backup_file(MASTER_PY_PATH)

    try:
        # Step 1: Verify/add ThreadPoolExecutor import
        logger.info("Step 1: Verifying imports...")
        content, import_added = verify_import_exists(content)

        # Step 2: Apply main fix
        logger.info("Step 2: Applying ProcessPoolExecutor fallback...")
        fixed_content = apply_fix(content)

        if fixed_content is None:
            logger.error("Failed to apply fix - pattern not found")
            logger.info("Manual fix required. See ERROR_ANALYSIS_REPORT.md for details")
            return False

        # Step 3: Add pre-flight validation
        logger.info("Step 3: Adding pre-flight strategy validation...")
        fixed_content = add_validation_test(fixed_content)

        # Write fixed content
        logger.info(f"Writing fixed code to {MASTER_PY_PATH}")
        MASTER_PY_PATH.write_text(fixed_content)

        logger.success("="*70)
        logger.success("Fix applied successfully!")
        logger.success("="*70)
        logger.info("")
        logger.info("Changes made:")
        logger.info("  1. Added ThreadPoolExecutor import (if needed)")
        logger.info("  2. Added ThreadPoolExecutor fallback for PermissionError")
        logger.info("  3. Added pre-flight strategy validation function")
        logger.info("")
        logger.info(f"Backup saved to: {backup_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review the changes: diff master.py master.py.backup")
        logger.info("  2. Test the fix: uv run python master.py --quick")
        logger.info("  3. Verify all strategies initialize successfully")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"Error during fix application: {e}")
        logger.info(f"Restoring from backup: {backup_path}")
        MASTER_PY_PATH.write_text(backup_path.read_text())
        logger.info("Backup restored")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
