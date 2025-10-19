# Daily Trading Recommendations - Complete Guide

## Overview

`daily_recommendations.py` is a production-ready script that generates actionable trading recommendations for today based on your selected strategy and current market conditions.

## Features

✅ **Real-time Analysis** - Fetches current market data and analyzes it
✅ **Multiple Strategies** - Support for all 25+ registered strategies
✅ **Multi-Asset Support** - Analyze one or multiple trading pairs
✅ **Risk Management** - Built-in position sizing and risk/reward calculations
✅ **Confidence Scoring** - Each recommendation has a confidence level (0-100%)
✅ **Entry/Exit Prices** - Specific price targets and stop losses
✅ **Rich Output** - Beautifully formatted console tables with colors
✅ **Export Options** - Save to JSON or CSV for record-keeping
✅ **Risk Levels** - Three pre-configured risk profiles (low, medium, high)

## Quick Start

### Basic Usage

```bash
# Get recommendations using RSI strategy for BTC
uv run python daily_recommendations.py -s RSI_MeanReversion

# Analyze multiple assets
uv run python daily_recommendations.py -s MACD_Momentum --symbols BTC/USDT,ETH/USDT,BNB/USDT

# Conservative trading (low risk)
uv run python daily_recommendations.py -s Supertrend_ATR --risk-level low

# Aggressive trading (high risk)
uv run python daily_recommendations.py -s BollingerBreakout --risk-level high

# Export recommendations to JSON
uv run python daily_recommendations.py -s TripleEMA --export json

# Use 4-hour timeframe instead of 1-hour
uv run python daily_recommendations.py -s VWAP_MeanReversion --timeframe 4h
```

## Command-Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--strategy` | `-s` | (required) | Strategy name to use (see list below) |
| `--symbols` | | `BTC/USDT` | Comma-separated symbols |
| `--timeframe` | `-t` | `1h` | Candle timeframe (1h, 4h, 1d) |
| `--multi-pair` | | `False` | Use multi-pair strategy |
| `--risk-level` | `-r` | `medium` | Risk level: low, medium, high |
| `--export` | `-e` | None | Export format: json, csv |
| `--lookback` | `-l` | `30` | Days of historical data |

## Available Strategies

### Single-Pair Strategies

1. **SMA_Crossover** - Simple Moving Average crossover (Golden/Death Cross)
2. **RSI_MeanReversion** - RSI oversold/overbought mean reversion
3. **MACD_Momentum** - MACD signal line crossover momentum
4. **BollingerBreakout** - Bollinger Bands volatility breakout
5. **TripleEMA** - Triple EMA trend filter strategy
6. **Supertrend_ATR** - Supertrend with ATR-based stops
7. **Ichimoku_Cloud** - Ichimoku Cloud comprehensive system
8. **VWAP_MeanReversion** - VWAP-based mean reversion
9. **OnChainAnalytics** - On-chain metrics strategy
10. **MultiTimeframeConfluence** - Multi-timeframe trend confluence
11. **VolatilityRegimeAdaptive** - Regime-aware adaptive strategy
12. **DynamicEnsemble** - Meta-strategy ensemble
13. **TransformerGRUPredictor** - ML-based predictor
14. **DDQNFeatureSelected** - Deep RL strategy
15. **MultiModalSentimentFusion** - Sentiment-based strategy
16. **OrderFlowImbalance** - Order flow microstructure

### Multi-Pair Strategies

Use with `--multi-pair` flag and multiple symbols:

1. **PortfolioRebalancer** - Multi-asset portfolio rebalancing
2. **StatisticalArbitrage** - Pairs trading with cointegration
3. **HierarchicalRiskParity** - HRP portfolio optimization
4. **BlackLitterman** - Bayesian portfolio with views
5. **RiskParity** - Equal risk contribution portfolio
6. **CopulaPairsTrading** - Copula-enhanced pairs trading
7. **DeepRLPortfolio** - Deep RL portfolio management

## Risk Levels Explained

### Low Risk (Conservative)
- Max position size: **10%** of portfolio
- Stop loss: **2%** from entry
- Min risk/reward: **2.0** (target 2x the risk)
- Min confidence: **70%**
- **Best for**: Capital preservation, retirement accounts

### Medium Risk (Balanced)
- Max position size: **20%** of portfolio
- Stop loss: **3%** from entry
- Min risk/reward: **1.5** (target 1.5x the risk)
- Min confidence: **60%**
- **Best for**: Most traders, steady growth

### High Risk (Aggressive)
- Max position size: **30%** of portfolio
- Stop loss: **5%** from entry
- Min risk/reward: **1.2** (target 1.2x the risk)
- Min confidence: **50%**
- **Best for**: Experienced traders, high conviction trades

## Output Format

### Console Output

The script generates a beautiful, color-coded table with:

```
╭────────────────────────────────╮
│ Daily Trading Recommendations  │
│ Strategy: RSI_MeanReversion    │
│ Generated: 2025-10-19 08:00:00 │
╰────────────────────────────────╯

┌──────────┬────────┬────────────┬──────────────┬──────────────┬──────────────┬────────┬──────┐
│ Symbol   │ Action │ Confidence │ Entry        │ Target       │ Stop Loss    │ Size % │ R:R  │
├──────────┼────────┼────────────┼──────────────┼──────────────┼──────────────┼────────┼──────┤
│ BTC/USDT │ BUY    │ 75.0%      │ $65,234.50   │ $68,500.00   │ $63,000.00   │ 15.0%  │ 1.50 │
│ ETH/USDT │ SELL   │ 68.5%      │ $3,245.80    │ $3,100.00    │ $3,350.00    │ 13.7%  │ 1.40 │
└──────────┴────────┴────────────┴──────────────┴──────────────┴──────────────┴────────┴──────┘

📋 Detailed Analysis:

BTC/USDT: Strategy: RSI_MeanReversion; Bullish signal detected; Price above long-term trend (bullish); RSI oversold (28.5)
ETH/USDT: Strategy: RSI_MeanReversion; Bearish signal detected; Price below long-term trend (bearish); RSI overbought (72.3)

✅ 2 actionable trade(s) recommended
```

### Color Coding

- **Green** - BUY signals, high confidence (≥70%)
- **Red** - SELL signals, low confidence (<60%)
- **Yellow** - HOLD signals, medium confidence (60-70%)
- **Cyan** - Informational text

### JSON Export

When using `--export json`, creates `daily_recommendations_YYYYMMDD.json`:

```json
{
  "date": "2025-10-19",
  "strategy_name": "RSI_MeanReversion",
  "recommendations": [
    {
      "symbol": "BTC/USDT",
      "action": "BUY",
      "confidence": 75.0,
      "entry_price": 65234.50,
      "target_price": 68500.00,
      "stop_loss": 63000.00,
      "position_size_pct": 15.0,
      "reasoning": "Strategy: RSI_MeanReversion; Bullish signal detected; Price above long-term trend (bullish); RSI oversold (28.5)",
      "risk_reward_ratio": 1.5,
      "timestamp": "2025-10-19T08:00:00"
    }
  ],
  "market_conditions": {
    "date": "2025-10-19",
    "time": "08:00:00",
    "symbols_analyzed": 1,
    "timeframe": "1h",
    "lookback_days": 30,
    "market_trend": "bullish"
  },
  "risk_level": "medium",
  "generated_at": "2025-10-19T08:00:00"
}
```

### CSV Export

When using `--export csv`, creates `daily_recommendations_YYYYMMDD.csv`:

```csv
symbol,action,confidence,entry_price,target_price,stop_loss,position_size_pct,reasoning,risk_reward_ratio,timestamp
BTC/USDT,BUY,75.0,65234.50,68500.00,63000.00,15.0,"Strategy: RSI_MeanReversion; Bullish signal...",1.5,2025-10-19 08:00:00
```

## Understanding Recommendations

### Action Types

- **BUY** - Enter long position (buy the asset)
- **SELL** - Enter short position (or close long)
- **HOLD** - No action recommended (market unclear)

### Confidence Score

The confidence score (0-100%) is calculated based on:

1. **Trend Alignment** (+15%) - Signal aligns with major trend
2. **RSI Confirmation** (+10%) - RSI supports the signal
3. **Volume Confirmation** (+10%) - Above-average volume
4. **Base Confidence** (50%) - Starting point

Higher confidence = stronger signal = larger position size

### Position Sizing

Position size is dynamically calculated as:

```
Position Size = Max Size × (Confidence / 100)
```

Examples:
- Medium risk, 75% confidence: 20% × 0.75 = 15% position
- High risk, 60% confidence: 30% × 0.60 = 18% position
- Low risk, 80% confidence: 10% × 0.80 = 8% position

### Risk/Reward Ratio

The R:R ratio shows potential reward vs risk:

```
R:R = (Target - Entry) / (Entry - Stop Loss)
```

- **R:R = 2.0** - Target is 2x farther than stop (good)
- **R:R = 1.5** - Target is 1.5x farther than stop (acceptable)
- **R:R = 1.0** - Target equals stop distance (risky)

Minimum R:R depends on risk level (see Risk Levels section).

## Example Workflows

### Morning Routine

```bash
#!/bin/bash
# morning_check.sh - Run this every morning

echo "Checking BTC with conservative approach..."
uv run python daily_recommendations.py -s RSI_MeanReversion --risk-level low --export json

echo "Checking ETH with balanced approach..."
uv run python daily_recommendations.py -s MACD_Momentum --symbols ETH/USDT --export json

echo "Multi-asset portfolio check..."
uv run python daily_recommendations.py -s PortfolioRebalancer \
  --symbols BTC/USDT,ETH/USDT,BNB/USDT \
  --multi-pair --export csv
```

### Scan Multiple Strategies

```bash
#!/bin/bash
# strategy_scan.sh - Compare multiple strategies

for strategy in RSI_MeanReversion MACD_Momentum Supertrend_ATR; do
  echo "=== $strategy ==="
  uv run python daily_recommendations.py -s $strategy --symbols BTC/USDT
  echo ""
done
```

### High-Frequency Updates

```bash
# Run every hour via cron
0 * * * * cd /path/to/crypto && uv run python daily_recommendations.py -s Supertrend_ATR --export json >> daily_log.txt 2>&1
```

## Integration with Trading

### Manual Trading

1. Run the script each morning
2. Review recommendations with >70% confidence
3. Verify the reasoning aligns with your analysis
4. Enter trades with specified parameters
5. Set alerts at target and stop loss prices

### Automated Trading (Future)

The JSON output can be consumed by trading bots:

```python
import json

# Load recommendations
with open('daily_recommendations_20251019.json') as f:
    report = json.load(f)

# Execute high-confidence trades
for rec in report['recommendations']:
    if rec['confidence'] >= 75 and rec['action'] in ['BUY', 'SELL']:
        execute_trade(
            symbol=rec['symbol'],
            side=rec['action'],
            entry=rec['entry_price'],
            stop_loss=rec['stop_loss'],
            target=rec['target_price'],
            size_pct=rec['position_size_pct']
        )
```

## Troubleshooting

### No Recommendations Generated

**Cause**: No clear signals in current market conditions

**Solution**:
- Try different strategies
- Adjust timeframe (1h → 4h → 1d)
- Lower risk level to accept more signals
- Market may genuinely be unclear (good!)

### Low Confidence Scores

**Cause**: Conflicting indicators or weak signals

**Solution**:
- Wait for clearer setup
- Use multiple strategies for confirmation
- Consider HOLD as valid recommendation

### Strategy Not Found

**Cause**: Typo in strategy name

**Solution**: Use exact names (case-sensitive):
- Correct: `RSI_MeanReversion`
- Wrong: `rsi_meanreversion`, `RSI MeanReversion`

List available strategies:
```bash
uv run python -c "from crypto_trader.strategies import get_registry; import crypto_trader.strategies.library; print(get_registry().get_strategy_names())"
```

## Best Practices

### 1. Daily Routine
- Run recommendations at the same time each day (e.g., 8 AM)
- Review before market open
- Don't force trades when no clear signals exist

### 2. Risk Management
- Never exceed recommended position sizes
- Always use stop losses
- Start with low risk level, increase gradually

### 3. Confirmation
- Use multiple strategies for major trades
- Verify signals align with your analysis
- Higher confidence (>75%) = stronger conviction

### 4. Record Keeping
- Use `--export json` to maintain history
- Track which strategies work best
- Review past recommendations for learning

### 5. Market Conditions
- Trending markets: Use momentum strategies (MACD, Supertrend)
- Ranging markets: Use mean reversion (RSI, Bollinger)
- Volatile markets: Lower risk level and position sizes

## Advanced Usage

### Custom Risk Parameters

Edit `_get_risk_parameters()` in the script to create custom profiles:

```python
'custom': {
    'max_position_size': 15.0,  # Your preference
    'stop_loss_pct': 2.5,
    'min_risk_reward': 1.8,
    'min_confidence': 65.0
}
```

### Backtesting Recommendations

Compare historical recommendations vs actual outcomes:

```bash
# Generate recommendations for past 30 days
for i in {1..30}; do
  date_offset=$((i * 24))  # hours
  # Use historical data and compare
done
```

### Multiple Timeframe Analysis

```bash
# Quick scan across timeframes
for tf in 1h 4h 1d; do
  echo "=== Timeframe: $tf ==="
  uv run python daily_recommendations.py -s TripleEMA --timeframe $tf
done
```

## File Structure

```
crypto/
├── daily_recommendations.py          # Main script
├── daily_recommendations.log         # Debug log (auto-generated)
├── daily_recommendations_YYYYMMDD.json  # Export (if --export json)
├── daily_recommendations_YYYYMMDD.csv   # Export (if --export csv)
└── src/crypto_trader/               # Core libraries
```

## Logging

Debug logs are automatically saved to `daily_recommendations.log`:

```bash
# View recent logs
tail -f daily_recommendations.log

# Search for errors
grep ERROR daily_recommendations.log

# View specific date
grep "2025-10-19" daily_recommendations.log
```

## Performance

- **Execution time**: 5-15 seconds per symbol
- **Data fetching**: Cached when possible
- **Memory usage**: ~500 MB typical
- **CPU**: Single-threaded analysis

## Future Enhancements

Planned features:
- [ ] Webhook notifications (Telegram, Discord, Email)
- [ ] Interactive mode with prompts
- [ ] Historical performance tracking
- [ ] Strategy combination recommendations
- [ ] Real-time price alerts
- [ ] Backtesting recommendations vs outcomes

## Support & Contact

For issues or questions:
1. Check this guide
2. Review logs in `daily_recommendations.log`
3. Test with simple examples first
4. Report bugs with full log output

---

**Remember**: These are recommendations, not financial advice. Always do your own analysis and never risk more than you can afford to lose.
