# Data Layer Guide

## Overview

The data layer provides comprehensive data management for cryptocurrency trading, including fetching, storage, and caching of OHLCV (Open, High, Low, Close, Volume) data from exchanges.

## Architecture

```
crypto_trader.data/
├── providers.py    # Abstract interface for data providers
├── fetchers.py     # CCXT-based exchange data fetchers
├── storage.py      # File-based CSV storage
├── cache.py        # In-memory caching with TTL
└── __init__.py     # Public API exports
```

## Components

### 1. DataProvider (Abstract Interface)

Defines the standard interface that all data providers must implement.

**Key Methods:**
- `get_ohlcv()` - Fetch OHLCV data for a symbol/timeframe
- `update_data()` - Update existing data with latest candles
- `get_available_symbols()` - List available trading pairs
- `validate_symbol()` - Check if symbol is valid
- `validate_timeframe()` - Check if timeframe is supported

### 2. BinanceDataFetcher

Production-ready fetcher for Binance exchange using CCXT library.

**Features:**
- Automatic rate limiting (respects exchange limits)
- Retry logic with exponential backoff
- Integrated storage and caching
- Batch fetching for multiple symbols
- Incremental data updates

**Usage:**
```python
from crypto_trader.data import BinanceDataFetcher

# Initialize fetcher
fetcher = BinanceDataFetcher(
    use_storage=True,
    use_cache=True,
    storage_path="data/ohlcv",
    rate_limit=1200
)

# Fetch recent data
df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)

# Update with latest data
fetcher.update_data("BTC/USDT", "1h")

# Batch fetch multiple symbols
results = fetcher.fetch_batch(["BTC/USDT", "ETH/USDT"], "1h", limit=50)
```

### 3. OHLCVStorage

File-based storage system using CSV format.

**Features:**
- Hierarchical directory structure by symbol/timeframe
- Incremental updates (append mode)
- Data validation
- Date range queries
- Duplicate removal

**Storage Structure:**
```
data/ohlcv/
├── BTC_USDT/
│   ├── 1h.csv
│   ├── 4h.csv
│   └── 1d.csv
└── ETH_USDT/
    ├── 1h.csv
    └── 4h.csv
```

**Usage:**
```python
from crypto_trader.data import OHLCVStorage

storage = OHLCVStorage(base_path="data/ohlcv")

# Save data
storage.save_ohlcv(df, "BTC/USDT", "1h", mode="overwrite")

# Load data
df = storage.load_ohlcv("BTC/USDT", "1h")

# Load with date filter
df = storage.load_ohlcv("BTC/USDT", "1h", start_date=start, end_date=end)

# Get date range
date_range = storage.get_date_range("BTC/USDT", "1h")
```

### 4. OHLCVCache

In-memory cache with TTL (Time To Live) support.

**Features:**
- Thread-safe LRU cache
- TTL-based expiration
- Cache statistics
- Specialized methods for OHLCV and indicators

**Usage:**
```python
from crypto_trader.data import OHLCVCache

cache = OHLCVCache(max_size=100, default_ttl=300)

# Cache OHLCV data
cache.set_ohlcv("BTC/USDT", "1h", df, ttl=600)

# Get cached data
df = cache.get_ohlcv("BTC/USDT", "1h")

# Cache indicator results
cache.set_indicator("BTC/USDT", "1h", "rsi", rsi_values)
rsi = cache.get_indicator("BTC/USDT", "1h", "rsi")

# Get statistics
stats = cache.get_stats()
# {'size': 5, 'max_size': 100, 'hits': 10, 'misses': 2, 'hit_rate': '83.33%'}
```

### 5. Cached Decorator

Function decorator for caching expensive calculations.

**Usage:**
```python
from crypto_trader.data import cached

@cached(ttl=300)
def calculate_indicators(symbol, timeframe):
    # Expensive calculation
    return results

# First call - computes and caches
result1 = calculate_indicators("BTC/USDT", "1h")

# Second call - returns cached result
result2 = calculate_indicators("BTC/USDT", "1h")
```

## Data Flow

1. **First Fetch**: Exchange → Storage → Cache → User
2. **Cached Fetch**: Cache → User (instant)
3. **Stored Fetch**: Storage → Cache → User (fast)
4. **Update**: Fetch new data → Merge with existing → Storage → Cache

```
┌─────────────┐
│  Exchange   │
│  (Binance)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ BinanceData │
│   Fetcher   │
└──────┬──────┘
       │
       ├──────────┐
       ▼          ▼
┌──────────┐  ┌────────┐
│ Storage  │  │ Cache  │
│  (CSV)   │  │ (RAM)  │
└────┬─────┘  └───┬────┘
     │            │
     └─────┬──────┘
           ▼
     ┌──────────┐
     │   User   │
     └──────────┘
```

## Supported Timeframes

- `1m` - 1 minute
- `3m` - 3 minutes
- `5m` - 5 minutes
- `15m` - 15 minutes
- `30m` - 30 minutes
- `1h` - 1 hour
- `2h` - 2 hours
- `4h` - 4 hours
- `6h` - 6 hours
- `8h` - 8 hours
- `12h` - 12 hours
- `1d` - 1 day
- `3d` - 3 days
- `1w` - 1 week
- `1M` - 1 month

## Error Handling

The data layer includes comprehensive error handling:

1. **Network Errors**: Automatic retry with exponential backoff
2. **Rate Limiting**: Automatic throttling to respect exchange limits
3. **Invalid Inputs**: Validation with clear error messages
4. **Data Integrity**: Validation of OHLCV data structure

**Example:**
```python
try:
    df = fetcher.get_ohlcv("INVALID/PAIR", "1h")
except ValueError as e:
    print(f"Invalid symbol: {e}")

try:
    df = fetcher.get_ohlcv("BTC/USDT", "99x")
except ValueError as e:
    print(f"Invalid timeframe: {e}")
```

## Performance Optimization

### Caching Strategy

1. **Hot Data**: Recently accessed data stays in cache (5 min TTL)
2. **Warm Data**: Stored on disk, quick reload
3. **Cold Data**: Fetch from exchange as needed

### Rate Limiting

- Default: 1200 requests/minute (Binance limit)
- Automatic spacing between requests
- Retry with exponential backoff on rate limit errors

### Batch Operations

Fetch multiple symbols efficiently:
```python
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT"]
results = fetcher.fetch_batch(symbols, "1h", limit=100)
```

## Best Practices

### 1. Initialize Once

```python
# Initialize at application startup
fetcher = BinanceDataFetcher()

# Reuse throughout application
df1 = fetcher.get_ohlcv("BTC/USDT", "1h")
df2 = fetcher.get_ohlcv("ETH/USDT", "1h")
```

### 2. Use Updates for Live Data

```python
# Initial fetch
df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=1000)

# Later, update with latest candles
fetcher.update_data("BTC/USDT", "1h")
```

### 3. Batch Fetch Related Data

```python
# Efficient
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
results = fetcher.fetch_batch(symbols, "1h")

# Less efficient
for symbol in symbols:
    df = fetcher.get_ohlcv(symbol, "1h")
```

### 4. Monitor Cache Statistics

```python
stats = fetcher.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']}")

# If hit rate is low, consider:
# - Increasing cache size
# - Increasing TTL
# - Checking access patterns
```

## Examples

### Basic Data Fetch

```python
from crypto_trader.data import BinanceDataFetcher

fetcher = BinanceDataFetcher()
df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)

print(f"Latest close: ${df['close'].iloc[-1]:,.2f}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
```

### Historical Data Analysis

```python
from datetime import datetime, timedelta

fetcher = BinanceDataFetcher()

# Get last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

df = fetcher.get_ohlcv(
    "BTC/USDT",
    "1h",
    start_date=start_date,
    end_date=end_date
)

# Calculate returns
df['returns'] = df['close'].pct_change()
print(f"Average hourly return: {df['returns'].mean():.2%}")
```

### Live Data Updates

```python
import time
from crypto_trader.data import BinanceDataFetcher

fetcher = BinanceDataFetcher()

# Initial fetch
fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)

# Update loop
while True:
    # Update with latest data
    fetcher.update_data("BTC/USDT", "1h")

    # Get updated data
    df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=100)
    print(f"Latest price: ${df['close'].iloc[-1]:,.2f}")

    # Wait 60 seconds
    time.sleep(60)
```

### Multi-Symbol Monitoring

```python
from crypto_trader.data import BinanceDataFetcher

fetcher = BinanceDataFetcher()

symbols = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT",
    "XRP/USDT", "ADA/USDT", "SOL/USDT"
]

# Fetch all symbols
results = fetcher.fetch_batch(symbols, "1h", limit=24)

# Analyze
for symbol, df in results.items():
    if not df.empty:
        price_change = (
            (df['close'].iloc[-1] - df['close'].iloc[0])
            / df['close'].iloc[0] * 100
        )
        print(f"{symbol}: {price_change:+.2f}% (24h)")
```

## Testing

Run the validation tests:

```bash
# Test providers interface
uv run python src/crypto_trader/data/providers.py

# Test storage
uv run python src/crypto_trader/data/storage.py

# Test cache
uv run python src/crypto_trader/data/cache.py

# Test fetchers (REAL Binance data)
uv run python src/crypto_trader/data/fetchers.py
```

Run the demo:

```bash
uv run python examples/data_layer_demo.py
```

## Troubleshooting

### Issue: Rate Limit Errors

**Solution:** Reduce request rate or increase wait time between requests.

```python
fetcher = BinanceDataFetcher(rate_limit=600)  # Lower rate
```

### Issue: Slow First Fetch

**Reason:** Data must be fetched from exchange, stored, and cached.

**Solution:** This is expected. Subsequent fetches will be much faster.

### Issue: Cache Not Working

**Check:**
1. Is caching enabled? `use_cache=True`
2. Is TTL appropriate? `default_ttl=300`
3. Check cache stats: `fetcher.get_cache_stats()`

### Issue: Storage Files Growing Large

**Solution:** Implement data retention policy or use database for large datasets.

```python
# Clean old data
storage = OHLCVStorage()
storage.delete_data("BTC/USDT", "1m")  # Delete 1-minute data
```

## Future Enhancements

- [ ] PostgreSQL/TimescaleDB support for large datasets
- [ ] Redis integration for distributed caching
- [ ] WebSocket support for real-time data
- [ ] Data compression for storage efficiency
- [ ] More exchange integrations (Coinbase, Kraken, etc.)
- [ ] Automatic data quality monitoring
- [ ] Advanced retry strategies
- [ ] Data versioning and rollback

## References

- CCXT Documentation: https://docs.ccxt.com/
- Binance API: https://binance-docs.github.io/apidocs/spot/en/
- Pandas Documentation: https://pandas.pydata.org/docs/
