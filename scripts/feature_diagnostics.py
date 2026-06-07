"""
Feature Diagnostics Script
Computes percentile distributions for ALL strategy-relevant features.
Outputs to diagnostics/feature_stats.json, boolean_stats.json, window_stats.json.
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.feature_engine import FeatureEngine, FeatureConfig
from core.data_structures import OrderBook, PriceLevel, Trade, Side


NUMERIC_FEATURES = [
    "volume_acceleration", "price_change_pct_60s", "price_change_pct_300s",
    "abs_delta_60s", "depth_imbalance_10", "abs_depth_imbalance_10",
    "delta_pct_60s", "delta_pct_300s", "abs_delta_pct_60s",
    "book_trade_agreement", "spread_bps", "bid_depth_10", "ask_depth_10",
    "atr_60s", "recent_absorption_strength", "price_vs_poc_pct",
    "price_vs_vwap_pct", "slope_asymmetry", "trade_intensity_60s",
    "net_pressure", "footprint_imbalance_count", "exhaustion_score",
    "va_breakout_potential", "price_vs_vah_pct", "price_vs_val_pct",
    "delta_divergence_60s", "cvd_price_divergence", "pressure_confirmed",
    "in_value_area", "buying_exhaustion", "selling_exhaustion",
]

BOOLEAN_FEATURES = [
    "delta_divergence_60s", "cvd_price_divergence", "recent_sweep_detected",
    "book_trade_agreement", "pressure_confirmed", "in_value_area",
    "buying_exhaustion", "selling_exhaustion",
]

WINDOW_FEATURES = [
    "price_change_pct_60s", "price_change_pct_300s",
    "delta_pct_60s", "delta_pct_300s",
]


def compute_distribution(values):
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {}
    return {
        "P5": float(np.percentile(values, 5)),
        "P25": float(np.percentile(values, 25)),
        "P50": float(np.percentile(values, 50)),
        "P75": float(np.percentile(values, 75)),
        "P90": float(np.percentile(values, 90)),
        "P95": float(np.percentile(values, 95)),
        "P99": float(np.percentile(values, 99)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "zero_pct": float(np.mean(values == 0) * 100),
        "count": int(len(values)),
    }


def compute_boolean_freq(values):
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {}
    return {
        "freq_1": float(np.mean(values >= 0.5) * 100),
        "freq_0": float(np.mean(values < 0.5) * 100),
        "count": int(len(values)),
    }


def load_data(symbol: str, date: str):
    data_dir = project_root / "data" / "backtests"
    candidates = list(data_dir.glob(f"{symbol}*{date}*.parquet"))
    if not candidates:
        candidates = list(data_dir.glob(f"{symbol}*.parquet"))
    if not candidates:
        print(f"No data found for {symbol} {date}")
        sys.exit(1)
    path = candidates[0]
    print(f"Loading: {path.name}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows, columns: {len(df.columns)}")
    return df


def build_order_book(row, timestamp):
    bid_prices = []
    bid_sizes = []
    ask_prices = []
    ask_sizes = []

    for i in range(20):
        bp = row.get(f"bid_price_{i}", 0)
        bs = row.get(f"bid_size_{i}", 0)
        ap = row.get(f"ask_price_{i}", 0)
        a_s = row.get(f"ask_size_{i}", 0)
        if pd.notna(bp) and bp > 0 and pd.notna(bs) and bs > 0:
            bid_prices.append(float(bp))
            bid_sizes.append(float(bs))
        if pd.notna(ap) and ap > 0 and pd.notna(a_s) and a_s > 0:
            ask_prices.append(float(ap))
            ask_sizes.append(float(a_s))

    if not bid_prices or not ask_prices:
        mid = float(row.get("trade_price", 0))
        if mid <= 0:
            return None
        spread = mid * 0.0001
        bid_prices = [mid - spread / 2]
        ask_prices = [mid + spread / 2]
        bid_sizes = [float(row.get("trade_size", 1.0))]
        ask_sizes = [float(row.get("trade_size", 1.0))]

    bids = [PriceLevel(price=p, size=s, timestamp=timestamp) for p, s in zip(bid_prices, bid_sizes)]
    asks = [PriceLevel(price=p, size=s, timestamp=timestamp) for p, s in zip(ask_prices, ask_sizes)]
    bids.sort(key=lambda l: l.price, reverse=True)
    asks.sort(key=lambda l: l.price)
    return OrderBook(timestamp=timestamp, bids=bids, asks=asks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XRPUSDT", help="Trading symbol")
    parser.add_argument("--date", default="20260329", help="Date string YYYYMMDD")
    parser.add_argument("--max_rows", type=int, default=50000, help="Max rows to process")
    args = parser.parse_args()

    df = load_data(args.symbol, args.date)
    df = df.head(args.max_rows)

    feature_engine = FeatureEngine(FeatureConfig())

    feature_collector = defaultdict(list)
    window_move_collector = []

    ts_col = "timestamp"
    if ts_col not in df.columns:
        ts_cols = [c for c in df.columns if "time" in c.lower()]
        ts_col = ts_cols[0] if ts_cols else None

    print("Processing ticks...")
    for idx in range(len(df)):
        row = df.iloc[idx]
        timestamp = pd.to_datetime(row[ts_col]).to_pydatetime() if ts_col else datetime.now()

        trades = []
        tp = float(row.get("trade_price", 0))
        ts = float(row.get("trade_size", 0))
        tside = str(row.get("trade_side", "buy")).lower().strip()
        if pd.notna(tp) and tp > 0 and pd.notna(ts) and ts > 0:
            side = Side.BUY if tside == "buy" else Side.SELL
            trades.append(Trade(timestamp=timestamp, price=tp, size=ts, side=side))

        order_book = build_order_book(row, timestamp)
        if order_book is None:
            continue

        state = feature_engine.update(order_book, trades, detect_patterns=(idx % 25 == 0))

        features = state.features
        for fname in NUMERIC_FEATURES:
            if fname in features:
                feature_collector[fname].append(features[fname])

        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(df)} ticks...")

    print(f"\nProcessed {len(df)} ticks total")

    feature_stats = {}
    for fname in NUMERIC_FEATURES:
        vals = feature_collector.get(fname, [])
        if vals:
            feature_stats[fname] = compute_distribution(vals)
            print(f"  {fname}: P50={feature_stats[fname]['P50']:.4f}, P90={feature_stats[fname]['P90']:.4f}, zero_pct={feature_stats[fname]['zero_pct']:.1f}%")

    out_dir = project_root / "diagnostics"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=2)
    print(f"\nSaved: {out_dir / 'feature_stats.json'}")

    boolean_stats = {}
    for fname in BOOLEAN_FEATURES:
        vals = feature_collector.get(fname, [])
        if vals:
            boolean_stats[fname] = compute_boolean_freq(vals)
            print(f"  {fname}: freq_1={boolean_stats[fname]['freq_1']:.1f}%")

    with open(out_dir / "boolean_stats.json", "w") as f:
        json.dump(boolean_stats, f, indent=2)
    print(f"Saved: {out_dir / 'boolean_stats.json'}")

    window_stats = {}
    for fname in WINDOW_FEATURES:
        vals = feature_collector.get(fname, [])
        if vals:
            vals_arr = np.abs(np.array(vals))
            window_stats[fname] = {
                "abs_P5": float(np.percentile(vals_arr, 5)),
                "abs_P25": float(np.percentile(vals_arr, 25)),
                "abs_P50": float(np.percentile(vals_arr, 50)),
                "abs_P75": float(np.percentile(vals_arr, 75)),
                "abs_P90": float(np.percentile(vals_arr, 90)),
                "abs_P95": float(np.percentile(vals_arr, 95)),
                "abs_P99": float(np.percentile(vals_arr, 99)),
                "abs_mean": float(np.mean(vals_arr)),
            }

    with open(out_dir / "window_stats.json", "w") as f:
        json.dump(window_stats, f, indent=2)
    print(f"Saved: {out_dir / 'window_stats.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
