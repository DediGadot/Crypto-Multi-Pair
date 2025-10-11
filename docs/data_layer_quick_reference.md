# Data Layer Quick Reference

## Installation

```bash
# Already included in crypto-trader package
# Dependencies: ccxt, pandas, loguru
```

## Quick Start

```python
from crypto_trader.data import BinanceDataFetcher

# Initialize
fetcher = BinanceDataFetcher()

# Fetch data
df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)

# Update
fetcher.update_data("BTC/USDT", "1h")
```

## Common Operations

### Fetch Recent Data
```python
df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)
```

### Fetch Date Range
```python
from datetime import datetime, timedelta

end = datetime.now()
start = end - timedelta(days=30)
df = fetcher.get_ohlcv("BTC/USDT", "1h", start_date=start, end_date=end)
```

### Batch Fetch
```python
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
results = fetcher.fetch_batch(symbols, "1h", limit=50)
```

### Update Latest
```python
fetcher.update_data("BTC/USDT", "1h")
```

## DataFrame Structure

```python
df.columns = ['open', 'high', 'low', 'close', 'volume']
df.index   # DatetimeIndex (UTC)
```

## Timeframes

```
1m, 3m, 5m, 15m, 30m          # Minutes
1h, 2h, 4h, 6h, 8h, 12h       # Hours
1d, 3d                         # Days
1w                             # Week
1M                             # Month
```

## Storage

Data automatically saved to:
```
data/ohlcv/{SYMBOL}/{TIMEFRAME}.csv
```

Example:
```
data/ohlcv/BTC_USDT/1h.csv
```

## Cache

Default TTL: 300 seconds (5 minutes)

```python
# Get cache stats
stats = fetcher.get_cache_stats()
# {'size': 5, 'hits': 10, 'misses': 2, 'hit_rate': '83.33%'}
```

## Configuration

```python
fetcher = BinanceDataFetcher(
    use_storage=True,        # Save to CSV
    use_cache=True,          # Use memory cache
    storage_path="data/ohlcv",  # Storage location
    rate_limit=1200,         # Requests per minute
    max_retries=3            # Retry attempts
)
```

## Error Handling

```python
try:
    df = fetcher.get_ohlcv("BTC/USDT", "1h")
except ValueError as e:
    print(f"Invalid input: {e}")
except ConnectionError as e:
    print(f"Network error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Available Symbols

```python
symbols = fetcher.get_available_symbols()
# Returns list of ~4000 symbols
```

## Validation

```python
# Check symbol
if fetcher.validate_symbol("BTC/USDT"):
    print("Valid symbol")

# Check timeframe
if fetcher.validate_timeframe("1h"):
    print("Valid timeframe")
```

## Direct Storage Access

```python
from crypto_trader.data import OHLCVStorage

storage = OHLCVStorage(base_path="data/ohlcv")

# Save
storage.save_ohlcv(df, "BTC/USDT", "1h")

# Load
df = storage.load_ohlcv("BTC/USDT", "1h")

# Check exists
if storage.has_data("BTC/USDT", "1h"):
    print("Data exists")

# Get date range
start, end = storage.get_date_range("BTC/USDT", "1h")

# Delete
storage.delete_data("BTC/USDT", "1h")
```

## Direct Cache Access

```python
from crypto_trader.data import OHLCVCache

cache = OHLCVCache(max_size=100, default_ttl=300)

# Cache OHLCV
cache.set_ohlcv("BTC/USDT", "1h", df, ttl=600)
df = cache.get_ohlcv("BTC/USDT", "1h")

# Cache indicators
cache.set_indicator("BTC/USDT", "1h", "rsi", rsi_values)
rsi = cache.get_indicator("BTC/USDT", "1h", "rsi")

# Clear
cache.clear()
```

## Function Caching

```python
from crypto_trader.data import cached

@cached(ttl=300)
def expensive_calculation(symbol, timeframe):
    # Your code here
    return result
```

## Performance Tips

1. **Use batch fetching** for multiple symbols
2. **Initialize once** and reuse fetcher instance
3. **Use updates** instead of full refetch
4. **Monitor cache hit rate** (aim for >80%)
5. **Enable storage** to avoid redundant API calls

## Troubleshooting

### Rate Limit Errors
```python
# Lower rate limit
fetcher = BinanceDataFetcher(rate_limit=600)
```

### Slow Fetches
- First fetch is always slow (network)
- Subsequent fetches use cache (~1000x faster)

### Cache Not Working
- Check `use_cache=True`
- Verify TTL is appropriate
- Check stats: `fetcher.get_cache_stats()`

## Examples

### Real-time Monitoring
```python
import time

while True:
    fetcher.update_data("BTC/USDT", "1h")
    df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=1)
    print(f"BTC: ${df['close'].iloc[-1]:,.2f}")
    time.sleep(60)
```

### Multi-Asset Analysis
```python
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
results = fetcher.fetch_batch(symbols, "1h", limit=24)

for symbol, df in results.items():
    pct_change = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"{symbol}: {pct_change:+.2f}%")
```

### Historical Backtest Data
```python
from datetime import datetime, timedelta

end = datetime.now()
start = end - timedelta(days=365)

df = fetcher.get_ohlcv(
    "BTC/USDT",
    "1d",
    start_date=start,
    end_date=end
)

# Use df for backtesting
```

## Testing

```bash
# Run validations
uv run python src/crypto_trader/data/providers.py
uv run python src/crypto_trader/data/storage.py
uv run python src/crypto_trader/data/cache.py
uv run python src/crypto_trader/data/fetchers.py

# Run demo
uv run python examples/data_layer_demo.py
```

## Documentation

- Full Guide: `docs/data_layer_guide.md`
- Summary: `docs/data_layer_summary.md`
- This Reference: `docs/data_layer_quick_reference.md`

## Support

For issues or questions:
1. Check the full guide: `docs/data_layer_guide.md`
2. Review validation tests in source files
3. Run the demo: `examples/data_layer_demo.py`
