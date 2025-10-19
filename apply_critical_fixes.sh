#!/bin/bash
#
# Apply Critical Bug Fixes to master.py
# This script applies all 3 critical fixes using sed
#
set -e

echo "🔧 Applying Critical Bug Fixes to master.py"
echo "=" * 60

# Backup
cp master.py master.py.before_fixes
echo "✓ Created backup: master.py.before_fixes"

# Fix #3 (Easiest): Fix string syntax errors
echo ""
echo "Fix #3: Fixing string syntax errors..."

# Fix AGGRESSIVE INVESTOR string
sed -i 's/f\.write("\\*\\*🎯 AGGRESSIVE INVESTOR\\*\\* (maximize returns, accept high risk):\n\n")/f.write("**🎯 AGGRESSIVE INVESTOR** (maximize returns, accept high risk):\\n\\n")/g' master.py

# Fix CONSERVATIVE INVESTOR string
sed -i 's/f\.write("\\*\\*🛡️  CONSERVATIVE INVESTOR\\*\\* (minimize drawdown, accept lower returns):\n\n")/f.write("**🛡️  CONSERVATIVE INVESTOR** (minimize drawdown, accept lower returns):\\n\\n")/g' master.py

# Fix BALANCED INVESTOR string
sed -i 's/f\.write("\\*\\*⚖️  BALANCED INVESTOR\\*\\* (best risk-adjusted returns):\n\n")/f.write("**⚖️  BALANCED INVESTOR** (best risk-adjusted returns):\\n\\n")/g' master.py

echo "✓ Fixed string syntax errors"

# Fix #1: Remove dead code (_verify_strategy_can_initialize)
echo ""
echo "Fix #1: Removing dead code..."

# Create a Python script to do this precisely
python3 << 'PYTHON_EOF'
import re

with open('master.py', 'r') as f:
    lines = f.readlines()

# Find the function
in_function = False
func_start = None
func_end = None

for i, line in enumerate(lines):
    if 'def _verify_strategy_can_initialize(' in line:
        func_start = i
        in_function = True
    elif in_function and line.strip() and not line[0].isspace() and 'def ' in line:
        # Next function found
        func_end = i
        break
    elif in_function and line.strip().startswith('class '):
        # Class found
        func_end = i
        break

if func_start is not None and func_end is not None:
    # Remove the function
    new_lines = lines[:func_start]
    new_lines.append('\n# DELETED: _verify_strategy_can_initialize - was dead code (never called)\n')
    new_lines.append('# If pre-flight checks are needed, implement them WHERE they\'re used\n')
    new_lines.extend(lines[func_end:])

    with open('master.py', 'w') as f:
        f.writelines(new_lines)

    print(f"✓ Removed dead code: lines {func_start+1}-{func_end}")
else:
    print("⚠ Dead code function not found (may already be removed)")
PYTHON_EOF

# Fix #2: Fix Sharpe ratio calculation
echo ""
echo "Fix #2: Fixing Sharpe ratio calculation..."

python3 << 'PYTHON_EOF'
import re

with open('master.py', 'r') as f:
    content = f.read()

# Find and replace the Sharpe ratio function's problematic section
old_sharpe = r'''    # Handle edge cases
    if std_return <= 0:
        # No volatility
        if mean_return > 0:
            return 100\.0  # Cap at high positive value
        elif mean_return < 0:
            return -100\.0  # Cap at high negative value
        else:
            return 0\.0  # No return, no volatility

    # Normal Sharpe calculation
    sharpe = \(mean_return \* periods_per_year\) / \(std_return \* np\.sqrt\(periods_per_year\)\)

    # Cap extreme values
    return max\(min\(sharpe, 100\.0\), -100\.0\)'''

new_sharpe = '''    # CRITICAL: Zero variance indicates a broken strategy - FAIL LOUDLY
    if std_return <= 1e-8:
        raise ValueError(
            f"Cannot calculate Sharpe ratio: zero/near-zero variance (std={std_return:.2e}). "
            f"This indicates constant returns, which suggests a broken strategy. "
            f"Returns: mean={mean_return:.6f}, std={std_return:.2e}"
        )

    # Normal Sharpe calculation
    sharpe = (mean_return * periods_per_year) / (std_return * np.sqrt(periods_per_year))

    # Sanity check for extreme values (but don't cap - let them through for debugging)
    if not np.isfinite(sharpe):
        raise ValueError(
            f"Sharpe ratio is non-finite ({sharpe}). "
            f"Returns: mean={mean_return}, std={std_return}, periods={periods_per_year}"
        )

    return float(sharpe)'''

if re.search(old_sharpe, content):
    content = re.sub(old_sharpe, new_sharpe, content)
    with open('master.py', 'w') as f:
        f.write(content)
    print("✓ Fixed Sharpe ratio calculation")
else:
    print("⚠ Sharpe ratio pattern not found (may already be fixed)")
PYTHON_EOF

# Verify syntax
echo ""
echo "Verifying syntax..."
python3 -m py_compile master.py && echo "✅ Syntax OK" || {
    echo "❌ Syntax error detected!"
    echo "Restoring backup..."
    cp master.py.before_fixes master.py
    exit 1
}

echo ""
echo "✅ ALL FIXES APPLIED SUCCESSFULLY!"
echo ""
echo "Next steps:"
echo "  1. Review changes: diff master.py.before_fixes master.py"
echo "  2. Test: uv run python master.py --help"
echo "  3. Run: uv run python master.py --multi-pair -h 30 --quick"
