"""
Debug script to test HierarchicalRiskParity signal generation.
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from crypto_trader.strategies.library.hierarchical_risk_parity import HierarchicalRiskParityStrategy
from crypto_trader.data.fetchers import BinanceDataFetcher

print("=" * 80)
print("DEBUGGING HIERARCHICAL RISK PARITY SIGNAL GENERATION")
print("=" * 80)

# Initialize strategy
strategy = HierarchicalRiskParityStrategy()
strategy.initialize({
    'asset_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
    'lookback_period': 90,
    'rebalance_freq': 7
})

print(f"\n✓ Strategy initialized")
print(f"  - Asset symbols: {strategy.asset_symbols}")
print(f"  - Transaction cost: {strategy.transaction_cost_pct}")
print(f"  - Min rebalance benefit: {strategy.min_rebalance_benefit}")

# Fetch real data
print(f"\n⏳ Fetching data...")
fetcher = BinanceDataFetcher()

# Fetch each asset separately and combine
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
dfs = []
for symbol in symbols:
    df = fetcher.get_ohlcv(
        symbol=symbol,
        timeframe='1h',
        limit=1500  # About 60 days
    )
    # Rename columns to include symbol
    df = df.rename(columns={
        'open': f'{symbol}_open',
        'high': f'{symbol}_high',
        'low': f'{symbol}_low',
        'close': f'{symbol}_close',
        'volume': f'{symbol}_volume'
    })
    dfs.append(df)

# Merge on timestamp
data = dfs[0]
for df in dfs[1:]:
    data = data.merge(df, left_index=True, right_index=True, how='inner')

data = data.reset_index()

print(f"✓ Data fetched: {len(data)} rows")
print(f"  Columns: {data.columns.tolist()}")

# Generate signals
print(f"\n⏳ Generating signals...")
try:
    signals = strategy.generate_signals(data)
    print(f"✓ Signals generated successfully!")
    print(f"\nSignal DataFrame shape: {signals.shape}")
    print(f"Signal DataFrame columns: {signals.columns.tolist()}")
    print(f"\nFirst 10 rows:")
    print(signals.head(10))
    print(f"\nSignal column unique values:")
    if 'signal' in signals.columns:
        print(signals['signal'].value_counts())
    else:
        print("❌ ERROR: 'signal' column is MISSING!")
        print(f"Available columns: {signals.columns.tolist()}")

    print(f"\n✅ SUCCESS: Signal generation completed")

except Exception as e:
    print(f"\n❌ ERROR during signal generation:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
