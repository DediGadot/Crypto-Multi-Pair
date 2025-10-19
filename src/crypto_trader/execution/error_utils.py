"""
Error Utilities for Backtest Execution

This module provides error formatting and handling utilities for
consistent error messaging across the execution layer.

**Purpose**: Format error messages with context and truncation

**Key Functions**:
- format_error_message: Format exceptions with context and truncation

**Third-party packages**:
None (standard library only)

**Sample Input**:
```python
try:
    risky_operation()
except Exception as e:
    message = format_error_message(e, context="Strategy: momentum", max_length=200)
```

**Expected Output**:
Formatted error message string with context.

Extracted from master.py (lines 503-527) during Phase 2.5 refactoring.
"""


def format_error_message(error: Exception, context: str = "", max_length: int = 500) -> str:
    """
    Format error messages consistently with optional truncation.

    Args:
        error: Exception object or error message
        context: Additional context (e.g., strategy name, operation)
        max_length: Maximum length for error message (0 = no truncation)

    Returns:
        Formatted error message string
    """
    error_str = str(error)

    # Add context if provided
    if context:
        full_message = f"{context}: {error_str}"
    else:
        full_message = error_str

    # Truncate if needed
    if max_length > 0 and len(full_message) > max_length:
        return full_message[:max_length-3] + "..."

    return full_message


if __name__ == "__main__":
    """
    Validation block for error utilities.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: Basic error formatting
    total_tests += 1
    print("Test 1: Basic error formatting")
    try:
        error = ValueError("Something went wrong")
        result = format_error_message(error)
        expected = "Something went wrong"

        if result != expected:
            all_validation_failures.append(f"Basic formatting: Expected '{expected}', got '{result}'")
        else:
            print(f"  ✓ Basic error: {result}")

    except Exception as e:
        all_validation_failures.append(f"Basic formatting failed: {e}")

    # Test 2: Error formatting with context
    total_tests += 1
    print("\nTest 2: Error formatting with context")
    try:
        error = ValueError("Invalid parameter")
        result = format_error_message(error, context="Strategy: momentum")
        expected = "Strategy: momentum: Invalid parameter"

        if result != expected:
            all_validation_failures.append(f"Context formatting: Expected '{expected}', got '{result}'")
        else:
            print(f"  ✓ With context: {result}")

    except Exception as e:
        all_validation_failures.append(f"Context formatting failed: {e}")

    # Test 3: Error formatting with truncation
    total_tests += 1
    print("\nTest 3: Error formatting with truncation")
    try:
        long_error = ValueError("A" * 1000)
        result = format_error_message(long_error, max_length=50)

        if len(result) != 50:
            all_validation_failures.append(f"Truncation: Expected length 50, got {len(result)}")
        elif not result.endswith("..."):
            all_validation_failures.append(f"Truncation: Should end with '...'")
        else:
            print(f"  ✓ Truncated to {len(result)} chars: {result[:30]}...")

    except Exception as e:
        all_validation_failures.append(f"Truncation failed: {e}")

    # Test 4: Context with truncation
    total_tests += 1
    print("\nTest 4: Context with truncation")
    try:
        long_error = ValueError("B" * 500)
        result = format_error_message(long_error, context="Long context", max_length=100)

        if len(result) != 100:
            all_validation_failures.append(f"Context+truncation: Expected length 100, got {len(result)}")
        elif not result.startswith("Long context:"):
            all_validation_failures.append(f"Context+truncation: Should start with context")
        elif not result.endswith("..."):
            all_validation_failures.append(f"Context+truncation: Should end with '...'")
        else:
            print(f"  ✓ Context + truncation: {len(result)} chars")

    except Exception as e:
        all_validation_failures.append(f"Context+truncation failed: {e}")

    # Test 5: No truncation (max_length=0)
    total_tests += 1
    print("\nTest 5: No truncation")
    try:
        long_error = ValueError("C" * 2000)
        result = format_error_message(long_error, max_length=0)

        if len(result) != 2000:
            all_validation_failures.append(f"No truncation: Expected length 2000, got {len(result)}")
        else:
            print(f"  ✓ No truncation (max_length=0): {len(result)} chars preserved")

    except Exception as e:
        all_validation_failures.append(f"No truncation failed: {e}")

    # Final validation result
    print("\n" + "="*60)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Error utilities are validated and ready for use")
        sys.exit(0)
