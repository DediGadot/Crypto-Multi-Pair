#!/usr/bin/env python3
"""
Apply critical bug fixes to master.py
This script applies all fixes programmatically to avoid patch issues.
"""
import re

# Read the file
with open('master.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("Original file: {} lines".format(len(content.splitlines())))

# Fix #1: Delete dead code function
# Find and remove _verify_strategy_can_initialize function
pattern1 = r'\n\ndef _verify_strategy_can_initialize\(.*?\n        return False\n'
replacement1 = '\n# DELETED: _verify_strategy_can_initialize - was dead code (never called anywhere)\n# If pre-flight checks are needed in future, implement them WHERE they\'re actually used\n'

if re.search(pattern1, content, re.DOTALL):
    content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    print("✓ Fix #1: Deleted dead code function")
else:
    print("⚠ Fix #1: Pattern not found (may already be applied)")

# Fix #2: Fix string syntax errors (3 locations)
# These are the unclosed multi-line strings
fixes = [
    (r'f\.write\("(\*\*🎯 AGGRESSIVE INVESTOR\*\* \(maximize returns, accept high risk\):)\n\n"\)', r'f.write("\1\\n\\n")'),
    (r'f\.write\("(\*\*🛡️  CONSERVATIVE INVESTOR\*\* \(minimize drawdown, accept lower returns\):)\n\n"\)', r'f.write("\1\\n\\n")'),
    (r'f\.write\("(\*\*⚖️  BALANCED INVESTOR\*\* \(best risk-adjusted returns\):)\n\n"\)', r'f.write("\1\\n\\n")'),
]

for pattern, replacement in fixes:
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print(f"✓ Fixed string syntax")
    else:
        # Try alternate pattern (the broken one)
        broken_pattern = pattern.replace(r'\\n\\n', r'\n\n')
        if re.search(broken_pattern, content):
            content = re.sub(broken_pattern, replacement, content)
            print(f"✓ Fixed broken string syntax")

print("\nFixed file: {} lines".format(len(content.splitlines())))

# Write the fixed file
with open('master_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Fixes applied! Output written to master_fixed.py")
print("Review the file, then: mv master_fixed.py master.py")
