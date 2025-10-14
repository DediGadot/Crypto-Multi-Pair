# Maximum History Feature - Documentation

## Overview

The `--max-history` flag automatically calculates and uses the maximum available historical data for portfolio optimization.

## Usage

```bash
# Use maximum available history with quick mode
uv run python optimize_portfolio_optimized.py --max-history --quick

# Use maximum available history with specific timeframe
uv run python optimize_portfolio_optimized.py --max-history --timeframe 1d --test-windows 3 --quick

# Use maximum available history with 4-hour timeframe
uv run python optimize_portfolio_optimized.py --max-history --timeframe 4h --test-windows 2 --quick
```

## How It Works

1. **Fetches Maximum Data**: Requests up to 10,000 periods per asset from Binance
2. **Finds Limiting Asset**: Identifies which asset has the least historical data
3. **Calculates Usable Periods**: Uses 80% of minimum available (safety margin for timestamp alignment)
4. **Computes Maximum Windows**: Calculates: `max_window_days = usable_periods / (test_windows + 1) / periods_per_day`
5. **Overrides window_days**: Automatically sets the optimal window size

## Example Output

```
================================================================================
🔍 CALCULATING MAXIMUM AVAILABLE HISTORY
================================================================================

  Fetching data for 8 assets to determine available history...
    BTC/USDT: 2,980 periods
    ETH/USDT: 2,980 periods
    BNB/USDT: 2,899 periods
    SOL/USDT: 1,890 periods
    ADA/USDT: 2,737 periods
    XRP/USDT: 2,720 periods
    MATIC/USDT: 1,965 periods
    DOT/USDT: 1,883 periods

  Minimum periods across all assets: 1,883
  Limiting asset: DOT/USDT

✅ MAXIMUM HISTORY CALCULATED:
  Available periods: 1,883
  Usable periods (80%): 1,506
  Maximum window_days: 376 days
  This allows 3 test windows of 376 days each

  Setting window_days = 376
```

## Benefits

1. **No Manual Calculation**: Automatically determines optimal parameters
2. **Prevents Errors**: Won't run with insufficient data
3. **Maximizes Statistical Power**: Uses all available data for robust results
4. **Safety Margin Built-In**: 80% threshold accounts for timestamp misalignment

## Typical Results by Timeframe

### Daily (1d)
- **BTC/ETH**: ~2,980 days (~8 years)
- **DOT/SOL**: ~1,883 days (~5 years) ⚠️ Limiting
- **Max window with 3 tests**: ~376 days

### 4-Hour (4h)
- **Typical**: ~1,500-2,000 days of data
- **Max window with 3 tests**: ~200-350 days

### Hourly (1h)
- **Typical**: ~700-800 days of data
- **Max window with 3 tests**: ~140-160 days

## Comparison: Manual vs Auto

### Manual (Old Way)
```bash
# User must guess and might fail
uv run python optimize_portfolio_optimized.py --window-days 365 --test-windows 5 --quick
# Result: ❌ INSUFFICIENT DATA (shortfall: 7,469 periods)
```

### Auto (New Way)
```bash
# Automatically calculates and uses 376 days
uv run python optimize_portfolio_optimized.py --max-history --test-windows 5 --quick
# Result: ✅ Uses optimal 376-day windows
```

## When to Use

✅ **Use --max-history when:**
- You want the most robust optimization possible
- You don't know how much data is available
- You want to avoid "insufficient data" errors
- You're testing a new timeframe or asset universe

❌ **Don't use --max-history when:**
- You want to test specific window sizes
- You're doing comparative analysis with fixed parameters
- You want faster runs with shorter windows

## Tips

1. **Combine with --quick**: `--max-history --quick` for fast, data-maximized runs
2. **Adjust test_windows**: Fewer windows = larger window_days
   - `--test-windows 2`: Larger windows, less validation
   - `--test-windows 5`: Smaller windows, more validation

3. **Check output**: Always review calculated window_days to ensure it meets your needs

## Advanced Usage

### For Research Papers
```bash
# Maximum data with extensive validation
uv run python optimize_portfolio_optimized.py --max-history --test-windows 5
```

### For Quick Testing
```bash
# Maximum data with minimal configs
uv run python optimize_portfolio_optimized.py --max-history --test-windows 2 --quick
```

### For Production
```bash
# Maximum data with specific workers
uv run python optimize_portfolio_optimized.py --max-history --test-windows 3 --workers 6
```

## Technical Details

### Safety Margin Calculation
- Uses **80%** of minimum periods across all assets
- Accounts for timestamp misalignment between assets
- Prevents edge cases where data exists but isn't aligned

### Error Handling
- Exits early if `max_window_days < 30` (too small to be useful)
- Suggests reducing `--test-windows` or using larger timeframe
- Provides clear error messages with solutions

## CLI Integration

```bash
# View help
uv run python optimize_portfolio_optimized.py --help

# Options
--max-history, -m       Use maximum available historical data
--timeframe, -t TEXT    Timeframe (1h/4h/1d) [default: 1h]
--test-windows, -n INT  Number of test windows [default: 5]
--quick, -q            Quick mode (fewer configs)
--workers, -w INT      Number of parallel workers
```

## See Also

- [FIX_COMPLETE.md](FIX_COMPLETE.md) - Portfolio optimization fixes
- [COMPLETE_FIX_GUIDE.md](COMPLETE_FIX_GUIDE.md) - Comprehensive troubleshooting
- [OPTIMIZATION_COMPLETE.md](OPTIMIZATION_COMPLETE.md) - Performance optimization details
