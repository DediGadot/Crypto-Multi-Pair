# Portfolio Rebalancing Advisor

A pragmatic, research-backed portfolio rebalancing tool that implements the strategies described in `PR.md`. Analyzes your crypto portfolio and recommends when and how to rebalance to maintain target allocations.

## Key Features

- **Smart Rebalancing Strategies**: Threshold, calendar, or hybrid methods
- **Interactive Setup**: Creates config.yaml through guided questions
- **Real-Time Prices**: Fetches current prices from Binance (or other exchanges)
- **Detailed Recommendations**: Exact BUY/SELL amounts in both USD and shares
- **Transaction Cost Awareness**: Minimum interval controls prevent over-trading
- **Momentum Filter**: Optional feature to avoid rebalancing during strong trends
- **Multiple Output Formats**: Console display, text file, and JSON

## Quick Start

### First Time Use (No Config)

```bash
# Run without a config - it will ask questions to create one
uv run python portfolio_rebalancer_advisor.py check
```

The script will ask you:
1. How many assets?
2. Symbol and target weight for each asset
3. Current shares held for each asset
4. Rebalancing method (threshold/calendar/hybrid)
5. Threshold percentage (if applicable)
6. Calendar period in days (if applicable)
7. Minimum hours between rebalances
8. Whether to enable momentum filter

Your configuration will be saved to `rebalance_config.yaml`.

### With Existing Config

```bash
# Use existing config
uv run python portfolio_rebalancer_advisor.py check --config my_portfolio.yaml

# Check and don't save output files
uv run python portfolio_rebalancer_advisor.py check --no-save
```

## Example Output

```
================================================================================
PORTFOLIO REBALANCING ADVISOR
================================================================================
Timestamp: 2025-10-19 07:49:28
Total Value: $24,907.84
Max Deviation: 12.54%

[!] REBALANCING RECOMMENDED: threshold (12.5% > 15.0%)

--------------------------------------------------------------------------------
Asset        Action Current    Target     Trade $         Trade Shares   
--------------------------------------------------------------------------------
BTC/USDT     SELL      53.6%     50.0%  $     -887.00     -0.008311
ETH/USDT     SELL      39.0%     30.0%  $   -2,236.07     -0.575807
SOL/USDT     BUY        7.5%     20.0%  $    3,123.07     16.804244
--------------------------------------------------------------------------------

DETAILED BREAKDOWN:
--------------------------------------------------------------------------------

BTC/USDT:
  Current: 0.125000 shares @ $106,727.35 = $13,340.92 (53.56%)
  Target:  0.116689 shares @ $106,727.35 = $12,453.92 (50.00%)
  >> SELL 0.008311 shares ($887.00)
...
================================================================================
```

## Configuration File Structure

See `rebalance_config_example.yaml` for a fully documented example.

```yaml
portfolio:
  assets:
    - symbol: BTC/USDT
      target_weight: 0.5
    - symbol: ETH/USDT
      target_weight: 0.5
  holdings:
    BTC/USDT: 0.125
    ETH/USDT: 2.5

rebalancing:
  method: hybrid              # threshold | calendar | hybrid
  threshold: 0.15             # 15% deviation trigger
  min_interval_hours: 24      # Prevent over-trading
  calendar_period_days: 30    # Monthly rebalancing
  momentum_filter:
    enabled: false
    threshold: 0.20           # Skip if portfolio up 20%+
    lookback_days: 30

state:
  last_rebalance: null        # ISO timestamp of last rebalance
  initial_capital: 10000.0    # Starting capital if holdings are zero

exchange:
  name: binance
  quote_currency: USDT
```

## Rebalancing Methods

### Threshold Method
Rebalances **only** when any asset deviates from its target weight by more than the threshold.

**Best for**: Volatile markets, cost-conscious traders
**Example**: 15% threshold means rebalance when BTC allocation drifts from 50% to 65%+ or 35%-

### Calendar Method
Rebalances on a **fixed schedule** (e.g., every 30 days), regardless of deviation.

**Best for**: Stable markets, tax planning, predictable execution
**Example**: Rebalance on the 1st of every month

### Hybrid Method (Recommended)
Rebalances when **either** threshold is exceeded **or** calendar period elapses.

**Best for**: Most scenarios - combines responsiveness with regular review
**Example**: Rebalance if deviation > 15% OR 30 days have passed

## Workflow

1. **Check Portfolio**: Run the `check` command
   ```bash
   uv run python portfolio_rebalancer_advisor.py check
   ```

2. **Review Recommendations**: Examine the output
   - If `[OK] NO REBALANCING NEEDED`: You're done!
   - If `[!] REBALANCING RECOMMENDED`: Proceed to step 3

3. **Execute Trades**: Manually execute the recommended trades on your exchange
   - SELL overweight assets
   - BUY underweight assets

4. **Update Holdings**: Tell the script about your new balances
   ```bash
   uv run python portfolio_rebalancer_advisor.py update
   ```
   
   This will:
   - Prompt for new share counts for each asset
   - Update the last rebalance timestamp
   - Save the updated config

5. **Repeat**: Run `check` regularly (daily, weekly, etc.)

## Validation Tests

The script includes built-in validation tests. Run them anytime:

```bash
# Run validation tests
uv run python portfolio_rebalancer_advisor.py
```

Expected output:
```
[OK] VALIDATION PASSED - All 6 tests produced expected results
Script is validated and ready for use
```

## Theory & Research Basis

This tool implements the portfolio rebalancing strategy detailed in `PR.md`, which is based on:

1. **Mean Reversion Theory**: Selling winners, buying losers captures price oscillations
2. **Modern Portfolio Theory**: Maintains diversification and risk control
3. **Volatility Harvesting**: Exploits the gap between arithmetic and geometric mean returns
4. **Empirical Research**: 15% threshold shown to outperform buy-and-hold by 77% (Vanguard 2010)

## Recommended Settings

Based on academic research and crypto market characteristics:

- **Threshold**: 15% (0.15) - optimal risk/return tradeoff
- **Min Interval**: 24 hours - prevents over-trading in volatile crypto markets
- **Calendar Period**: 30 days - ensures regular review even in low-volatility periods
- **Method**: Hybrid - best of both worlds
- **Momentum Filter**: Disabled for most users, enabled (20%) for strong trend followers

## Advanced Usage

### Custom Exchange

Edit `exchange.name` in your config to use other exchanges supported by ccxt:
```yaml
exchange:
  name: kraken  # or coinbase, kucoin, etc.
```

### Multiple Portfolios

Create separate config files:
```bash
uv run python portfolio_rebalancer_advisor.py check --config conservative_portfolio.yaml
uv run python portfolio_rebalancer_advisor.py check --config aggressive_portfolio.yaml
```

### Automated Checks

Add to cron for daily checks:
```bash
# Run at 9 AM daily
0 9 * * * cd /path/to/crypto && uv run python portfolio_rebalancer_advisor.py check
```

## Troubleshooting

**Q: "Price fetch failed" error**
A: Check internet connection. Script falls back to example prices if exchange is unreachable.

**Q: Script asks for config even though I have one**
A: Ensure you're using `--config` flag or that `rebalance_config.yaml` exists in current directory.

**Q: Weights don't sum to 1.0**
A: Script auto-normalizes weights. Check the log output for warnings.

**Q: Rebalancing never triggers**
A: With `calendar` method and `last_rebalance: null`, it should trigger immediately. With `threshold` method, deviation must exceed threshold.

## Output Files

The script generates:
- `rebalance_rec_YYYYMMDD_HHMMSS.txt` - Human-readable text report
- `rebalance_rec_YYYYMMDD.json` - Machine-readable JSON (only when rebalancing recommended)
- `rebalance.log` - Detailed execution log

## Safety Features

- **Minimum Interval**: Prevents panic trading
- **Read-Only Price Fetching**: Never executes trades automatically
- **Config Backup**: Original config preserved when updated
- **Validation**: Built-in tests ensure correct calculations

## License & Disclaimer

This is a tool for **analysis and recommendations only**. It does NOT execute trades automatically.

**Trading crypto involves risk. This tool is for educational purposes. Always review recommendations before executing trades. Past performance does not guarantee future results.**

## Architecture

The code follows Linus Torvalds' philosophy:
- **Simple > Complex**: Functions over classes where possible
- **Modular**: Each function does one thing well
- **Tested**: Validation tests with real expected values
- **No Magic**: Clear, readable logic throughout

Total lines: ~450 (well under 500-line guideline)

## Contributing

This is a standalone tool. For bugs or improvements, modify the script directly. The validation block at the bottom ensures changes don't break core functionality.

---

**Built with**: Python 3.11+, typer, pyyaml, loguru, ccxt
**Based on**: Research-backed strategies from PR.md
**Philosophy**: Simple, pragmatic, effective
