"""
Feature Precomputer
====================
Pre-computes ALL features from raw DataFrame in ONE vectorized pass.
Replaces per-tick FeatureEngine calls in the backtest loop with O(1) lookups.

Integration:
    precomputer = FeaturePrecomputer(feature_engine)
    precomputer.precompute_all(dataframe)
    for tick_idx in range(n):
        features = precomputer.get_features_for_tick(tick_idx)
        state.features.update(features)

Expected speedup: 5-20x in backtest loop.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger


class FeaturePrecomputer:
    """Pre-compute all features before the backtest loop for O(1) per-tick lookup."""

    # Feature name constants
    TRADE_FEATURE_KEYS = [
        "buy_volume", "sell_volume", "total_volume",
        "delta", "delta_pct", "abs_delta",
        "trade_count", "trade_intensity", "avg_trade_size", "vwap",
    ]

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or [15, 30, 60, 300, 600, 900]
        self.precomputed: Dict[str, np.ndarray] = {}
        self.n = 0

    def precompute_all(self, df: pd.DataFrame) -> None:
        """Precompute ALL features in one vectorized pass."""
        self.n = len(df)
        if self.n == 0:
            return

        n = self.n
        self.precomputed = {}

        # ─── Extract raw arrays ───
        prices = df["trade_price"].values.astype(np.float64)
        volumes = df["trade_size"].values.astype(np.float64)
        # [FIX] Fallback to _0 columns when bid_price/ask_price missing (raw parquet data)
        if 'bid_price' not in df.columns and 'bid_price_0' in df.columns:
            bid_prices = df['bid_price_0'].values.astype(np.float64)
            ask_prices = df['ask_price_0'].values.astype(np.float64)
        else:
            bid_prices = df.get("bid_price", pd.Series(np.zeros(n))).values.astype(np.float64)
            ask_prices = df.get("ask_price", pd.Series(np.zeros(n))).values.astype(np.float64)
        bid_sizes = df.get("bid_size", pd.Series(np.zeros(n))).values.astype(np.float64)
        ask_sizes = df.get("ask_size", pd.Series(np.zeros(n))).values.astype(np.float64)

        # ─── Compute bid_depth_10 and ask_depth_10 as SUM of 10 levels ───
        bid_depth_10 = np.zeros(n, dtype=np.float64)
        ask_depth_10 = np.zeros(n, dtype=np.float64)
        
        # Detect how many depth levels exist (bid_size_0, bid_size_1, ..., bid_size_9, etc.)
        n_depth_levels = 0
        while f'bid_size_{n_depth_levels}' in df.columns:
            n_depth_levels += 1
        
        if n_depth_levels > 0:
            # Sum up to 10 levels (or however many exist)
            levels_to_sum = min(10, n_depth_levels)
            for level in range(levels_to_sum):
                bid_col = f'bid_size_{level}'
                ask_col = f'ask_size_{level}'
                if bid_col in df.columns:
                    bid_depth_10 += df[bid_col].values.astype(np.float64)
                if ask_col in df.columns:
                    ask_depth_10 += df[ask_col].values.astype(np.float64)
            logger.info(f"[FeaturePrecomputer] Detected {n_depth_levels} depth levels; summing {levels_to_sum} levels for bid_depth_10/ask_depth_10")
        else:
            # Fallback: use best bid/ask only
            bid_depth_10 = bid_sizes.copy()
            ask_depth_10 = ask_sizes.copy()
            logger.warning(f"[FeaturePrecomputer] No depth level columns (bid_size_0, etc.) found; using best bid/ask only")
        
        # Validate depth (warn if unrealistic)
        valid_depths = bid_depth_10[bid_depth_10 > 0]
        if len(valid_depths) > 0:
            mean_depth = np.mean(valid_depths)
            if mean_depth < 1000:
                logger.warning(f"[FeaturePrecomputer] WARNING: Mean bid_depth_10 = {mean_depth:.0f} < 1000 (unrealistic for XRP/USDT). Check data loading.")
        else:
            logger.error(f"[FeaturePrecomputer] ERROR: All bid_depth_10 values are 0! Depth data may not be loaded correctly.")

        # Side as numeric: 1 = BUY, 0 = SELL
        if "trade_side" in df.columns:
            sides = (df["trade_side"].astype(str).str.lower()
                     .str.strip() == "buy").values.astype(np.int8)
        else:
            sides = np.ones(n, dtype=np.int8)

        # ─── 1. Price-derived features (vectorized) ───
        prev_prices = np.roll(prices, 1)
        prev_prices[0] = prices[0]
        returns = np.where(prev_prices > 0, (prices - prev_prices) / prev_prices, 0.0)
        self.precomputed["returns_1"] = returns
        self.precomputed["price_change_pct"] = returns

        # Mid price
        mid = np.where(
            (bid_prices > 0) & (ask_prices > 0),
            (bid_prices + ask_prices) / 2.0,
            prices,
        )
        self.precomputed["mid_price"] = mid

        # Spread in bps
        spread = np.where(
            (bid_prices > 0) & (ask_prices > 0),
            (ask_prices - bid_prices) / mid * 10000,
            0.0,
        )
        self.precomputed["spread_bps"] = spread

        # ─── 2. Cumulative arrays for O(1) window queries ───
        try:
            from core.numba_features import cumsum_by_side_nb
            cum_buy_vol, cum_sell_vol, cum_buy_cnt, cum_sell_cnt, cum_value, cum_buy_value = \
                cumsum_by_side_nb(sides, volumes, prices)
        except ImportError:
            # Pure Python fallback
            cum_buy_vol = np.zeros(n)
            cum_sell_vol = np.zeros(n)
            cum_buy_cnt = np.zeros(n)
            cum_sell_cnt = np.zeros(n)
            cum_value = np.zeros(n)
            cum_buy_value = np.zeros(n)
            for i in range(n):
                bv = cum_buy_vol[i - 1] if i > 0 else 0.0
                sv = cum_sell_vol[i - 1] if i > 0 else 0.0
                bc = cum_buy_cnt[i - 1] if i > 0 else 0.0
                sc = cum_sell_cnt[i - 1] if i > 0 else 0.0
                cv = cum_value[i - 1] if i > 0 else 0.0
                cbv = cum_buy_value[i - 1] if i > 0 else 0.0
                if sides[i] == 1:
                    bv += volumes[i]
                    bc += 1
                    cbv += prices[i] * volumes[i]
                else:
                    sv += volumes[i]
                    sc += 1
                cv += prices[i] * volumes[i]
                cum_buy_vol[i] = bv
                cum_sell_vol[i] = sv
                cum_buy_cnt[i] = bc
                cum_sell_cnt[i] = sc
                cum_value[i] = cv
                cum_buy_value[i] = cbv

        self._cum_buy_vol = cum_buy_vol
        self._cum_sell_vol = cum_sell_vol
        self._cum_buy_cnt = cum_buy_cnt
        self._cum_sell_cnt = cum_sell_cnt
        self._cum_value = cum_value
        self._cum_buy_value = cum_buy_value

        # ─── 3. Precompute window start indices ───
        timestamps = pd.to_datetime(df["timestamp"])
        timestamps_epoch = np.array([ts.timestamp() for ts in timestamps], dtype=np.float64)
        self._timestamps_epoch = timestamps_epoch

        try:
            from core.numba_features import find_window_starts_nb
            self._window_starts = find_window_starts_nb(timestamps_epoch, np.array(self.windows, dtype=np.float64))
        except ImportError:
            self._window_starts = np.zeros((n, len(self.windows)), dtype=np.int64)
            for w, window in enumerate(self.windows):
                idx = 0
                for i in range(n):
                    cutoff = timestamps_epoch[i] - window
                    while idx < i and timestamps_epoch[idx] < cutoff:
                        idx += 1
                    self._window_starts[i, w] = idx

        # ─── 4. ATR via Numba ───
        try:
            from core.numba_features import calculate_atr_nb
            self.precomputed["atr_60"] = calculate_atr_nb(
                np.maximum(bid_prices, ask_prices),
                np.minimum(bid_prices, ask_prices),
                prices, 14
            )
        except ImportError:
            atr = np.zeros(n)
            for i in range(1, n):
                high = max(bid_prices[i], ask_prices[i], prices[i])
                low = min(bid_prices[i], ask_prices[i], prices[i])
                tr = max(high - low, abs(high - prices[i-1]), abs(low - prices[i-1]))
                atr[i] = (atr[i-1] * 13 + tr) / 14 if i >= 14 else tr
            self.precomputed["atr_60"] = atr

        # ─── 5. Book features per tick ───
        self.precomputed["bid_depth_10"] = bid_depth_10
        self.precomputed["ask_depth_10"] = ask_depth_10
        total_depth = bid_depth_10 + ask_depth_10 + 1e-9
        self.precomputed["depth_imbalance_10"] = (bid_depth_10 - ask_depth_10) / total_depth
        self.precomputed["abs_depth_imbalance_10"] = np.abs(self.precomputed["depth_imbalance_10"])
        # net_pressure approximation: bid_volume - ask_volume over top 10 levels
        self.precomputed["net_pressure"] = bid_depth_10 - ask_depth_10
        self.precomputed["best_bid_size"] = bid_sizes
        self.precomputed["best_ask_size"] = ask_sizes
        denom = bid_sizes + ask_sizes
        best_level_imbalance = np.divide(
            bid_sizes - ask_sizes, denom,
            out=np.zeros_like(bid_sizes),
            where=denom > 0,
        )
        self.precomputed["best_level_imbalance"] = best_level_imbalance

        logger.info(
            f"[FeaturePrecomputer] Precomputed {len(self.precomputed)} base features "
            f"for {n} ticks in 1 pass"
        )

    def cumsum_delta(self, arr, start, end):
        """O(1) window sum from cumulative array."""
        if end < 0 or start > end:
            return 0.0
        return arr[end] - (arr[start - 1] if start > 0 else 0.0)

    def get_window_features(self, tick_idx: int, window_sec: int, w_idx: int) -> Dict[str, float]:
        """Get windowed features for a specific tick using O(1) cumsum lookups."""
        start = int(self._window_starts[tick_idx, w_idx])
        if tick_idx <= start:
            return {k: 0.0 for k in self.TRADE_FEATURE_KEYS}

        bv = self.cumsum_delta(self._cum_buy_vol, start, tick_idx)
        sv = self.cumsum_delta(self._cum_sell_vol, start, tick_idx)
        bc = self.cumsum_delta(self._cum_buy_cnt, start, tick_idx)
        sc = self.cumsum_delta(self._cum_sell_cnt, start, tick_idx)
        tv = self.cumsum_delta(self._cum_value, start, tick_idx)
        cbv = self.cumsum_delta(self._cum_buy_value, start, tick_idx)

        total_vol = bv + sv
        total_cnt = bc + sc
        delta = bv - sv

        features = {
            f"buy_volume_{window_sec}s": bv,
            f"sell_volume_{window_sec}s": sv,
            f"total_volume_{window_sec}s": total_vol,
            f"delta_{window_sec}s": delta,
            f"delta_pct_{window_sec}s": delta / (total_vol + 1e-9),
            f"abs_delta_pct_{window_sec}s": abs(delta) / (total_vol + 1e-9),
            f"abs_delta_{window_sec}s": abs(delta),
            f"trade_count_{window_sec}s": total_cnt,
            f"trade_intensity_{window_sec}s": total_cnt / window_sec if window_sec > 0 else 0.0,
            f"avg_trade_size_{window_sec}s": total_vol / (total_cnt + 1e-9),
            f"vwap_deviation_{window_sec}s": self._calc_vwap_dev(tv, total_vol, prices_at_tick=self.precomputed["mid_price"][tick_idx]),
            f"price_change_pct_{window_sec}s": self._calc_price_change(start, tick_idx),
            f"atr_{window_sec}s": self._calc_atr_window(start, tick_idx),
        }
        return features

    def _calc_vwap_dev(self, total_value, total_vol, prices_at_tick):
        if total_vol <= 0:
            return 0.0
        vwap = total_value / total_vol
        return (prices_at_tick - vwap) / vwap if vwap > 0 else 0.0

    def _calc_price_change(self, start, end):
        mid = self.precomputed["mid_price"]
        if end <= start or mid[start] <= 0:
            return 0.0
        return (mid[end] - mid[start]) / mid[start]

    def _calc_atr_window(self, start, end):
        atr = self.precomputed.get("atr_60", np.zeros(self.n))
        if end <= start:
            return 0.0
        return float(np.mean(atr[start:end + 1]))

    def get_features_for_tick(self, tick_idx: int) -> Dict[str, float]:
        """Get ALL features for a specific tick — O(1) per feature."""
        if tick_idx >= self.n:
            return {}

        features = {}

        # Book features (precomputed)
        for key in ["mid_price", "spread_bps", "best_bid_size", "best_ask_size",
                     "best_level_imbalance", "bid_depth_10", "ask_depth_10",
                     "depth_imbalance_10", "abs_depth_imbalance_10", "net_pressure"]:
            features[key] = float(self.precomputed[key][tick_idx])

        # ATR
        features["atr_60s"] = float(self.precomputed["atr_60"][tick_idx])
        features["price_range_60s"] = float(self.precomputed["atr_60"][tick_idx])  # proxy

        # Windower features
        for w_idx, window in enumerate(self.windows):
            features.update(self.get_window_features(tick_idx, window, w_idx))

        # CVD (cumulative delta)
        cum_delta = self._cum_buy_vol[tick_idx] - self._cum_sell_vol[tick_idx]
        features["cvd"] = cum_delta
        cvd_60 = cum_delta - (self._cum_buy_vol[max(0, tick_idx - 60)] - self._cum_sell_vol[max(0, tick_idx - 60)])
        features["cvd_60s"] = cvd_60

        # Book-trade agreement composite (fast estimate)
        book_imbal = features.get("depth_imbalance_10", 0)
        trade_imbal = features.get("delta_pct_60s", 0)
        if (book_imbal > 0.08 and trade_imbal > 0.1) or \
           (book_imbal < -0.08 and trade_imbal < -0.1):
            features["book_trade_agreement"] = 1.0
        else:
            features["book_trade_agreement"] = 0.0

        # Volume acceleration
        vol_60 = features.get("total_volume_60s", 0)
        vol_300 = features.get("total_volume_300s", 0)
        features["volume_acceleration"] = (vol_60 * 5) / (vol_300 + 1e-9)

        # Price vs VWAP
        vwap_dev = features.get("vwap_deviation_60s", 0)
        features["price_vs_vwap_pct"] = vwap_dev

        return features

    def reset(self) -> None:
        """Clear precomputed data."""
        self.precomputed.clear()
        self.n = 0
