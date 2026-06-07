"""Profile feature_engine.update() time breakdown."""
import time
import pandas as pd
from loguru import logger
logger.remove()

from core.feature_engine import FeatureEngine, FeatureConfig
from core.data_structures import OrderBook, Trade, Side, PriceLevel

df = pd.read_parquet('./data/recorded/XRPUSDT_market_data_20260328.parquet')
df_small = df.iloc[:20000].reset_index(drop=True)
print(f"Testing {len(df_small)} rows")

engine = FeatureEngine(FeatureConfig())

times = {"build_inputs": 0.0, "update": 0.0, "prep": 0.0}

for i in range(len(df_small)):
    row = df_small.iloc[i]
    ts = pd.to_datetime(row["timestamp"]).to_pydatetime()

    t0 = time.time()
    ob = OrderBook(
        timestamp=ts,
        bids=[PriceLevel(price=row.get("bid_price", 1.0) - 0.001, size=row.get("bid_size", 100), timestamp=ts)],
        asks=[PriceLevel(price=row.get("ask_price", 1.0) + 0.001, size=row.get("ask_size", 100), timestamp=ts)],
    )
    trades = [Trade(timestamp=ts, price=row["trade_price"], size=row["trade_size"],
                     side=Side.BUY if str(row.get("trade_side", "buy")).lower() == "buy" else Side.SELL)]
    times["build_inputs"] += time.time() - t0

    t0 = time.time()
    state = engine.update(ob, trades, detect_patterns=(i % 25 == 0), compute_volume_profile=(i % 100 == 0))
    times["update"] += time.time() - t0

    if i > 0 and i % 5000 == 0:
        print(f"  {i}/{len(df_small)}: update={times['update']:.2f}s build={times['build_inputs']:.2f}s")

total = sum(times.values())
for k, v in times.items():
    print(f"{k}: {v:.3f}s ({v/total*100:.1f}%)")
print(f"Total: {total:.3f}s")
print(f"Ticks/sec: {len(df_small)/max(total, 0.01):.0f}")
