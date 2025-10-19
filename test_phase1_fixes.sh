#!/bin/bash
# Test script for Phase 1 multi-pair optimizations
# Verifies that fixes are working correctly

set -e  # Exit on error

echo "========================================================================"
echo "TESTING PHASE 1 MULTI-PAIR OPTIMIZATIONS"
echo "========================================================================"
echo ""
echo "This script will run master.py --multi-pair with timing to verify:"
echo "  1. Shared data pool is pre-fetching correctly"
echo "  2. Workers are using shared data (no redundant fetches)"
echo "  3. Data alignment warnings appear if needed"
echo "  4. Feature augmentation is applied"
echo "  5. Performance is 4-10x faster than before"
echo ""
echo "Starting test in 3 seconds..."
sleep 3
echo ""

# Record start time
START_TIME=$(date +%s)

echo "========================================================================"
echo "RUNNING: master.py --multi-pair --quick --workers 4"
echo "========================================================================"
echo ""

# Run master.py with multi-pair mode
uv run python master.py --symbol BTC/USDT --multi-pair --quick --workers 4

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================================================"
echo "TEST COMPLETE!"
echo "========================================================================"
echo ""
echo "Execution time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "✅ EXPECTED OUTCOMES:"
echo "  • Time: 2-5 minutes (if >10 min, shared pool may not be working)"
echo "  • Log should show 'PRE-FETCHING MULTI-PAIR DATA (Shared Data Pool)'"
echo "  • Log should show '✓ Pre-fetched N assets. Will share with all workers'"
echo "  • Log should show 'Memory optimization: ~X redundant API calls eliminated'"
echo "  • No individual worker should show 'Fetching data for...' messages"
echo ""
echo "📊 VERIFICATION CHECKLIST:"
echo "  [ ] Pre-fetch section appeared in logs?"
echo "  [ ] Execution completed in <5 minutes?"
echo "  [ ] No worker-level data fetching messages?"
echo "  [ ] Results generated successfully?"
echo ""
echo "Check the latest master_results_* directory for output files."
echo ""

# Find latest results directory
LATEST_DIR=$(ls -td master_results_* 2>/dev/null | head -1)

if [ -n "$LATEST_DIR" ]; then
    echo "Latest results: $LATEST_DIR"
    echo ""

    if [ -f "$LATEST_DIR/MASTER_REPORT.txt" ]; then
        echo "Preview of results:"
        echo "-------------------------------------------------------------------"
        head -30 "$LATEST_DIR/MASTER_REPORT.txt"
        echo "-------------------------------------------------------------------"
        echo ""
        echo "✅ SUCCESS: Master report generated"
        echo ""
        echo "Full report: $LATEST_DIR/MASTER_REPORT.txt"
        echo "Full report: $LATEST_DIR/MASTER_REPORT.html"
    else
        echo "⚠️  WARNING: MASTER_REPORT.txt not found"
    fi

    # Check log for key markers
    if [ -f "$LATEST_DIR/master_analysis.log" ]; then
        echo ""
        echo "Checking log for optimization markers..."

        if grep -q "PRE-FETCHING MULTI-PAIR DATA" "$LATEST_DIR/master_analysis.log"; then
            echo "  ✅ Shared data pool marker found"
        else
            echo "  ❌ Shared data pool marker NOT found - optimization may not be working"
        fi

        if grep -q "Pre-fetched.*assets.*Will share with all workers" "$LATEST_DIR/master_analysis.log"; then
            echo "  ✅ Data sharing confirmation found"
        else
            echo "  ❌ Data sharing confirmation NOT found"
        fi

        if grep -q "redundant API calls eliminated" "$LATEST_DIR/master_analysis.log"; then
            echo "  ✅ API call optimization confirmed"
        else
            echo "  ❌ API call optimization NOT confirmed"
        fi
    fi
else
    echo "⚠️  No results directory found - test may have failed"
fi

echo ""
echo "========================================================================"
echo "For detailed analysis, see: PHASE1_FIXES_SUMMARY.md"
echo "========================================================================"
