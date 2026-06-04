import bisect
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .data_structures import (
    OrderBook, Trade, Side, FootprintBar, VolumeProfile,
    Imbalance, Absorption, LiquiditySweep, IcebergOrder,
    PriceLevel, OrderFlowState, Regime
)


@dataclass
class FeatureConfig:
    windows: List[int] = field(default_factory=lambda: [15, 30, 60, 300, 600, 900])
    book_depth_levels: int = 20
    volume_profile_lookback: int = 3600
    tick_size: float = 0.0001
    absorption_volume_multiplier: float = 1.5
    absorption_price_threshold_pct: float = 0.001
    absorption_min_duration_sec: float = 1.0
    imbalance_ratio_threshold: float = 3.0
    stacked_imbalance_min_levels: int = 3
    iceberg_refill_threshold: int = 3
    iceberg_size_threshold: float = 1.0
    sweep_speed_threshold_sec: float = 5.0
    sweep_reversal_threshold_pct: float = 0.002
    footprint_bar_duration_sec: int = 5
    regime_lookback_seconds: int = 1800
    regime_vol_low_threshold: float = 0.003
    regime_vol_high_threshold: float = 0.008
    regime_trend_strength_threshold: float = 0.5
    low_liquidity_spread_bps: float = 20.0
    low_liquidity_min_depth_10: float = 1.0
    high_volatility_multiplier: float = 1.5
    breakout_volume_accel_threshold: float = 1.0
    breakout_va_breakout_threshold: float = 0.5
    breakout_trend_strength_threshold: float = 0.4
    accum_dist_pressure_threshold: float = 0.5


class RegimeClassifier:
    def __init__(self, config: FeatureConfig):
        self.config = config

    def classify(
        self,
        trade_history: List[Trade],
        book_features: Dict[str, float],
        current_ts: datetime,
        timestamps_cache: Optional[List[datetime]] = None,
    ) -> Regime:
        if len(trade_history) < 100:
            return Regime.UNKNOWN
        cutoff = current_ts - timedelta(seconds=self.config.regime_lookback_seconds)
        if timestamps_cache is not None and len(timestamps_cache) == len(trade_history):
            timestamps = timestamps_cache
        else:
            timestamps = [t.timestamp for t in trade_history]
        idx = bisect.bisect_left(timestamps, cutoff)
        window_trades = trade_history[idx:]
        if len(window_trades) < 50:
            return Regime.UNKNOWN
        prices = [t.price for t in window_trades[:500]]
        if len(prices) < 10:
            return Regime.UNKNOWN
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                log_returns.append(np.log(prices[i] / prices[i - 1]))
        if len(log_returns) < 10:
            return Regime.UNKNOWN
        volatility = float(np.std(log_returns))
        start_px, end_px = prices[0], prices[-1]
        price_range = max(prices) - min(prices)
        net_move = abs(end_px - start_px)
        trend_strength = net_move / (price_range + 1e-9)
        price_dir = float(np.sign(end_px - start_px))
        spread_bps = book_features.get("spread_bps", 0)
        depth_10 = book_features.get("bid_depth_10", 0) + book_features.get("ask_depth_10", 0)
        net_pressure = book_features.get("net_pressure", 0)
        vol_accel = book_features.get("volume_acceleration", 1.0)
        cfg = self.config
        if spread_bps > cfg.low_liquidity_spread_bps:
            return Regime.LOW_LIQUIDITY
        if depth_10 < cfg.low_liquidity_min_depth_10 * 2:
            return Regime.LOW_LIQUIDITY
        if volatility > cfg.regime_vol_high_threshold * cfg.high_volatility_multiplier:
            return Regime.HIGH_VOLATILITY
        if (volatility >= cfg.regime_vol_low_threshold
                and trend_strength >= cfg.breakout_trend_strength_threshold
                and vol_accel > cfg.breakout_volume_accel_threshold):
            return Regime.BREAKOUT
        if (volatility >= cfg.regime_vol_low_threshold
                and trend_strength >= cfg.regime_trend_strength_threshold):
            return Regime.TRENDING_UP if price_dir > 0 else Regime.TRENDING_DOWN
        if volatility < cfg.regime_vol_low_threshold:
            if abs(net_pressure) > cfg.accum_dist_pressure_threshold:
                return Regime.ACCUMULATION if net_pressure > 0 else Regime.DISTRIBUTION
        if (volatility < cfg.regime_vol_high_threshold
                and trend_strength < cfg.regime_trend_strength_threshold):
            return Regime.RANGING
        return Regime.RANGING


class FeatureEngine:
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.trade_history: List[Trade] = []
        self.book_history: List[OrderBook] = []
        self.footprint_bars: List[FootprintBar] = []
        self.price_level_activity: Dict[float, Dict] = {}
        self.potential_icebergs: Dict[float, Dict] = {}
        self._volume_profile_cache: Optional[VolumeProfile] = None
        self._cache_timestamp: Optional[datetime] = None
        self._vp_features_cache: Optional[Dict[str, float]] = None
        self._cvd: float = 0.0
        self._trade_timestamps: List[datetime] = []
        self._cached_absorptions: List[Absorption] = []
        self._cached_sweeps: List[LiquiditySweep] = []
        self._cached_icebergs: List[IcebergOrder] = []
        self._current_timestamp: Optional[datetime] = None
        self._current_footprint_bar: Optional[FootprintBar] = None
        self._footprint_bar_start: Optional[datetime] = None
        self._feature_snapshots: List[Tuple[datetime, Dict[str, float]]] = []
        self._regime_classifier = RegimeClassifier(self.config)

    def reset(self) -> None:
        self.trade_history.clear()
        self.book_history.clear()
        self.footprint_bars.clear()
        self.price_level_activity.clear()
        self.potential_icebergs.clear()
        self._volume_profile_cache = None
        self._cache_timestamp = None
        self._vp_features_cache = None
        self._cvd = 0.0
        self._trade_timestamps.clear()
        self._cached_absorptions = []
        self._cached_sweeps = []
        self._cached_icebergs = []
        self._current_timestamp = None
        self._current_footprint_bar = None
        self._footprint_bar_start = None
        self._feature_snapshots.clear()

    def update(
        self,
        order_book: OrderBook,
        trades: List[Trade],
        detect_patterns: bool = True,
        compute_volume_profile: bool = True,
        precomputed_features: Optional[Dict[str, float]] = None,
    ) -> OrderFlowState:
        self._current_timestamp = order_book.timestamp
        self.book_history.append(order_book)
        for trade in trades:
            self.trade_history.append(trade)
            self._trade_timestamps.append(trade.timestamp)
            self._cvd += trade.size if trade.side == Side.BUY else -trade.size
            self._update_footprint_bars(trade)
        if len(self.trade_history) > 50000:
            self.trade_history = self.trade_history[-25000:]
            self._trade_timestamps = self._trade_timestamps[-25000:]
        if len(self.book_history) > 5000:
            self.book_history = self.book_history[-2500:]
        state = OrderFlowState(
            timestamp=order_book.timestamp,
            order_book=order_book,
            recent_trades=trades
        )
        if precomputed_features is not None:
            state.features = precomputed_features
        else:
            state.features = self._compute_all_features(
                order_book,
                trades,
                compute_volume_profile=compute_volume_profile
            )
        if detect_patterns:
            self._cached_absorptions = self._detect_absorptions()
            state.imbalances = self._detect_imbalances(order_book)
            self._cached_sweeps = self._detect_liquidity_sweeps()
            self._cached_icebergs = self._detect_icebergs()
        else:
            state.imbalances = self._detect_imbalances(order_book)
        state.absorptions = self._cached_absorptions
        state.sweeps = self._cached_sweeps
        state.icebergs = self._cached_icebergs
        if compute_volume_profile:
            state.volume_profile = self._compute_volume_profile()
        else:
            state.volume_profile = self._volume_profile_cache
        state.features.update(self._compute_pattern_features(state))
        state.features.update(self._compute_footprint_features())
        state.features.update(self._compute_composite_features(state.features))
        state.regime = self._regime_classifier.classify(
            self.trade_history,
            state.features,
            self._current_timestamp,
            timestamps_cache=self._trade_timestamps,
        )
        self._store_feature_snapshot(order_book.timestamp, state.features)
        if len(self._feature_snapshots) > 2000:
            self._feature_snapshots = self._feature_snapshots[-1000:]
        return state

    def _compute_all_features(
        self,
        book: OrderBook,
        trades: List[Trade],
        compute_volume_profile: bool = True
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        features.update(self._compute_book_features(book))
        trade_times = [t.timestamp for t in self.trade_history]
        for window in self.config.windows:
            cutoff = self._current_timestamp - timedelta(seconds=window)
            idx = bisect.bisect_left(trade_times, cutoff)
            window_trades = self.trade_history[idx:]
            window_features = self._compute_trade_features(window_trades, window)
            for key, value in window_features.items():
                features[f"{key}_{window}s"] = value
        features["cvd"] = self._cvd
        features.update(
            self._compute_volume_profile_features(use_cached=not compute_volume_profile)
        )
        features.update(self._compute_footprint_features())
        features.update(self._compute_composite_features(features))
        return features

    def _compute_book_features(self, book: OrderBook) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if not book.bids or not book.asks:
            return self._empty_book_features()
        features["spread_bps"] = book.spread_bps
        features["mid_price"] = book.mid_price
        features["microprice"] = book.microprice
        features["microprice_vs_mid"] = (
            (book.microprice - book.mid_price) / book.mid_price
            if book.mid_price else 0
        )
        features["best_bid_size"] = book.best_bid.size
        features["best_ask_size"] = book.best_ask.size
        features["best_level_imbalance"] = (
            (book.best_bid.size - book.best_ask.size) /
            (book.best_bid.size + book.best_ask.size + 1e-9)
        )
        bid_sizes = [level.size for level in book.bids[:self.config.book_depth_levels]]
        ask_sizes = [level.size for level in book.asks[:self.config.book_depth_levels]]
        for depth in [5, 10, 20]:
            bid_depth = sum(bid_sizes[:depth])
            ask_depth = sum(ask_sizes[:depth])
            total_depth = bid_depth + ask_depth
            features[f"bid_depth_{depth}"] = bid_depth
            features[f"ask_depth_{depth}"] = ask_depth
            features[f"depth_imbalance_{depth}"] = (bid_depth - ask_depth) / (total_depth + 1e-9)
        features["abs_depth_imbalance_10"] = abs(features["depth_imbalance_10"])
        features["bid_slope"] = self._compute_book_slope(book.bids)
        features["ask_slope"] = self._compute_book_slope(book.asks)
        features["slope_asymmetry"] = features["bid_slope"] - features["ask_slope"]
        features["bid_pressure"] = self._compute_pressure(book.bids, book.mid_price)
        features["ask_pressure"] = self._compute_pressure(book.asks, book.mid_price)
        features["net_pressure"] = features["bid_pressure"] - features["ask_pressure"]
        if book.bids[0].order_count > 0:
            features["bid_order_count"] = sum(l.order_count for l in book.bids[:10])
            features["ask_order_count"] = sum(l.order_count for l in book.asks[:10])
            features["avg_bid_order_size"] = features["bid_depth_10"] / (features["bid_order_count"] + 1e-9)
            features["avg_ask_order_size"] = features["ask_depth_10"] / (features["ask_order_count"] + 1e-9)
        features["bid_liquidity_concentration"] = bid_sizes[0] / (sum(bid_sizes) + 1e-9)
        features["ask_liquidity_concentration"] = ask_sizes[0] / (sum(ask_sizes) + 1e-9)
        features["bid_gap_exists"] = float(self._detect_gap(book.bids))
        features["ask_gap_exists"] = float(self._detect_gap(book.asks))
        return features

    @staticmethod
    def _compute_book_slope(levels: List[PriceLevel]) -> float:
        n = min(len(levels), 10)
        if n < 3:
            return 0.0
        sum_x = n * (n - 1) / 2.0
        sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0
        sum_y = 0.0
        sum_xy = 0.0
        for i in range(n):
            s = levels[i].size
            sum_y += s
            sum_xy += i * s
        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom

    def _compute_pressure(self, levels: List[PriceLevel], mid_price: float) -> float:
        if not levels or mid_price == 0:
            return 0.0
        pressure = 0.0
        for level in levels[:self.config.book_depth_levels]:
            distance = abs(level.price - mid_price) / mid_price
            weight = 1.0 / (1.0 + distance * 100)
            pressure += level.size * weight
        return pressure

    @staticmethod
    def _detect_gap(levels: List[PriceLevel], gap_multiplier: float = 3.0) -> bool:
        n = min(10, len(levels) - 1)
        if n < 1:
            return False
        total_gap = sum(abs(levels[i].price - levels[i + 1].price) for i in range(n))
        avg_gap = total_gap / n
        threshold = avg_gap * gap_multiplier
        for i in range(min(20, len(levels) - 1)):
            if abs(levels[i].price - levels[i + 1].price) > threshold:
                return True
        return False

    def _empty_book_features(self) -> Dict[str, float]:
        return {key: 0.0 for key in [
            "spread_bps", "mid_price", "microprice", "microprice_vs_mid",
            "best_bid_size", "best_ask_size", "best_level_imbalance",
            "bid_depth_5", "ask_depth_5", "depth_imbalance_5",
            "bid_depth_10", "ask_depth_10", "depth_imbalance_10", "abs_depth_imbalance_10",
            "bid_depth_20", "ask_depth_20", "depth_imbalance_20",
            "bid_slope", "ask_slope", "slope_asymmetry",
            "bid_pressure", "ask_pressure", "net_pressure",
            "bid_liquidity_concentration", "ask_liquidity_concentration",
            "bid_gap_exists", "ask_gap_exists"
        ]}

    def _compute_trade_features(
        self,
        window_trades: List[Trade],
        window_seconds: int
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if not window_trades:
            return self._empty_trade_features()
        buy_trades = [t for t in window_trades if t.side == Side.BUY]
        sell_trades = [t for t in window_trades if t.side == Side.SELL]
        buy_volume = sum(t.size for t in buy_trades)
        sell_volume = sum(t.size for t in sell_trades)
        total_volume = buy_volume + sell_volume
        features["buy_volume"] = buy_volume
        features["sell_volume"] = sell_volume
        features["total_volume"] = total_volume
        features["delta"] = buy_volume - sell_volume
        features["delta_pct"] = features["delta"] / (total_volume + 1e-9)
        features["abs_delta_pct"] = abs(features["delta_pct"])
        features["abs_delta"] = abs(features["delta"])
        features["trade_count"] = len(window_trades)
        features["buy_trade_count"] = len(buy_trades)
        features["sell_trade_count"] = len(sell_trades)
        features["trade_count_imbalance"] = (
            (len(buy_trades) - len(sell_trades)) / (len(window_trades) + 1e-9)
        )
        features["trade_intensity"] = len(window_trades) / window_seconds
        features["avg_trade_size"] = total_volume / (len(window_trades) + 1e-9)
        features["avg_buy_size"] = buy_volume / (len(buy_trades) + 1e-9) if buy_trades else 0
        features["avg_sell_size"] = sell_volume / (len(sell_trades) + 1e-9) if sell_trades else 0
        if len(window_trades) >= 10:
            sizes = sorted((t.size for t in window_trades), reverse=True)
            p90_idx = max(1, len(sizes) // 10)
            size_threshold = sizes[p90_idx - 1]
            large_count = 0
            large_vol = 0.0
            large_buy_vol = 0.0
            for t in window_trades:
                if t.size >= size_threshold:
                    large_count += 1
                    large_vol += t.size
                    if t.side == Side.BUY:
                        large_buy_vol += t.size
            features["large_trade_count"] = large_count
            features["large_trade_volume_pct"] = large_vol / (total_volume + 1e-9)
            features["large_trade_buy_pct"] = large_buy_vol / (large_vol + 1e-9)
        else:
            features["large_trade_count"] = 0
            features["large_trade_volume_pct"] = 0
            features["large_trade_buy_pct"] = 0
        if len(window_trades) >= 2:
            price_change = window_trades[-1].price - window_trades[0].price
            features["price_change"] = price_change
            features["price_change_pct"] = price_change / (window_trades[0].price + 1e-9)
        else:
            features["price_change"] = 0
            features["price_change_pct"] = 0
        if len(window_trades) >= 2:
            prices = [t.price for t in window_trades]
            tr = max(prices) - min(prices)
            features["atr"] = tr if tr > 0 else (window_trades[0].price * 0.005)
        else:
            features["atr"] = window_trades[0].price * 0.005 if window_trades else 0
        total_value = sum(t.value for t in window_trades)
        features["vwap"] = total_value / (total_volume + 1e-9)
        if len(window_trades) >= 2 and features["vwap"] > 0:
            features["vwap_deviation"] = (
                (window_trades[-1].price - features["vwap"]) / features["vwap"]
            )
        else:
            features["vwap_deviation"] = 0
        return features

    def _empty_trade_features(self) -> Dict[str, float]:
        features = {key: 0.0 for key in [
            "buy_volume", "sell_volume", "total_volume",
            "delta", "delta_pct", "abs_delta_pct", "abs_delta",
            "trade_count", "buy_trade_count", "sell_trade_count",
            "trade_count_imbalance", "trade_intensity",
            "avg_trade_size", "avg_buy_size", "avg_sell_size",
            "large_trade_count", "large_trade_volume_pct", "large_trade_buy_pct",
            "price_change", "price_change_pct", "atr", "vwap", "vwap_deviation"
        ]}
        features["cvd"] = self._cvd
        return features

    def _compute_volume_profile(self) -> VolumeProfile:
        cutoff = self._current_timestamp - timedelta(seconds=self.config.volume_profile_lookback)
        timestamps = [t.timestamp for t in self.trade_history]
        idx = bisect.bisect_left(timestamps, cutoff)
        recent_trades = self.trade_history[idx:]
        profile = VolumeProfile(
            timestamp=self._current_timestamp,
            lookback_seconds=self.config.volume_profile_lookback
        )
        if not recent_trades:
            return profile
        tick = self.config.tick_size
        for trade in recent_trades:
            bucket_price = round(trade.price / tick) * tick
            profile.volume_at_price[bucket_price] = (
                profile.volume_at_price.get(bucket_price, 0) + trade.size
            )
        profile.compute_value_area()
        self._volume_profile_cache = profile
        self._cache_timestamp = self._current_timestamp
        return profile

    def _compute_volume_profile_features(self, use_cached: bool = True) -> Dict[str, float]:
        profile = self._volume_profile_cache
        if not profile or not profile.volume_at_price:
            return {
                "poc": 0, "vah": 0, "val": 0,
                "value_area_width_pct": 0, "price_vs_poc_pct": 0,
                "price_vs_vah_pct": 0, "price_vs_val_pct": 0,
                "in_value_area": 0, "volume_profile_skew": 0
            }
        if use_cached and self._vp_features_cache and self._cache_timestamp:
            cache_age = (self._current_timestamp - self._cache_timestamp).total_seconds()
            if cache_age < 100:
                return self._vp_features_cache
        features = self._calculate_vp_features(profile)
        self._vp_features_cache = features
        return features

    def _calculate_vp_features(self, profile: VolumeProfile) -> Dict[str, float]:
        features: Dict[str, float] = {}
        current_price = self.book_history[-1].mid_price if self.book_history else 0
        features["poc"] = profile.poc
        features["vah"] = profile.vah
        features["val"] = profile.val
        features["value_area_width_pct"] = (
            (profile.vah - profile.val) / (profile.poc + 1e-9)
        )
        if current_price > 0:
            features["price_vs_poc_pct"] = (current_price - profile.poc) / profile.poc
            features["price_vs_vah_pct"] = (
                (current_price - profile.vah) / profile.vah if profile.vah else 0
            )
            features["price_vs_val_pct"] = (
                (current_price - profile.val) / profile.val if profile.val else 0
            )
            features["in_value_area"] = float(profile.val <= current_price <= profile.vah)
        else:
            features["price_vs_poc_pct"] = 0
            features["price_vs_vah_pct"] = 0
            features["price_vs_val_pct"] = 0
            features["in_value_area"] = 0
        prices = sorted(profile.volume_at_price.keys())
        volumes = [profile.volume_at_price[p] for p in prices]
        if len(volumes) > 2:
            total_vol = sum(volumes)
            weighted_price = sum(p * v for p, v in zip(prices, volumes)) / (total_vol + 1e-9)
            mid_price_range = (prices[0] + prices[-1]) / 2
            features["volume_profile_skew"] = (
                (weighted_price - mid_price_range) / (mid_price_range + 1e-9)
            )
        else:
            features["volume_profile_skew"] = 0
        return features

    def _update_footprint_bars(self, trade: Trade) -> None:
        bucket_price = round(trade.price / self.config.tick_size) * self.config.tick_size
        if (self._current_footprint_bar is None or
            (trade.timestamp - self._footprint_bar_start).total_seconds()
            >= self.config.footprint_bar_duration_sec):
            if self._current_footprint_bar:
                self.footprint_bars.append(self._current_footprint_bar)
                if len(self.footprint_bars) > 1000:
                    self.footprint_bars = self.footprint_bars[-500:]
            self._start_new_footprint_bar(trade)
        bar = self._current_footprint_bar
        bar.high_price = max(bar.high_price, trade.price)
        bar.low_price = min(bar.low_price, trade.price)
        bar.close_price = trade.price
        b_vol, s_vol = bar.volume_at_price.get(bucket_price, (0.0, 0.0))
        if trade.side == Side.BUY:
            bar.volume_at_price[bucket_price] = (b_vol + trade.size, s_vol)
        else:
            bar.volume_at_price[bucket_price] = (b_vol, s_vol + trade.size)

    def _start_new_footprint_bar(self, trade: Trade) -> None:
        self._footprint_bar_start = trade.timestamp
        self._current_footprint_bar = FootprintBar(
            timestamp=trade.timestamp,
            duration_seconds=self.config.footprint_bar_duration_sec,
            open_price=trade.price,
            high_price=trade.price,
            low_price=trade.price,
            close_price=trade.price,
            volume_at_price={}
        )

    def _compute_footprint_features(self) -> Dict[str, float]:
        features: Dict[str, float] = {
            "footprint_delta": 0, "highest_delta_level": 0,
            "lowest_delta_level": 0, "delta_concentration": 0,
            "footprint_imbalance_count": 0, "buying_exhaustion": 0,
            "selling_exhaustion": 0
        }
        if not self.footprint_bars:
            return features
        latest_bar = self.footprint_bars[-1]
        features["footprint_delta"] = latest_bar.delta
        if latest_bar.volume_at_price:
            delta_by_price = {
                price: buy - sell
                for price, (buy, sell) in latest_bar.volume_at_price.items()
            }
            features["highest_delta_level"] = max(delta_by_price.keys(),
                                                   key=lambda p: delta_by_price[p])
            features["lowest_delta_level"] = min(delta_by_price.keys(),
                                                  key=lambda p: delta_by_price[p])
            max_delta = max(delta_by_price.values())
            min_delta = min(delta_by_price.values())
            total_abs_delta = sum(abs(d) for d in delta_by_price.values())
            features["delta_concentration"] = (
                (abs(max_delta) + abs(min_delta)) / (total_abs_delta + 1e-9)
            )
            features["footprint_imbalance_count"] = len([
                price for price, (buy, sell) in latest_bar.volume_at_price.items()
                if (buy / (sell + 1e-9) >= self.config.imbalance_ratio_threshold or
                    sell / (buy + 1e-9) >= self.config.imbalance_ratio_threshold)
            ])
        if len(self.footprint_bars) >= 2:
            prev_bar = self.footprint_bars[-2]
            features["buying_exhaustion"] = float(
                latest_bar.high_price > prev_bar.high_price and
                latest_bar.delta < prev_bar.delta
            )
            features["selling_exhaustion"] = float(
                latest_bar.low_price < prev_bar.low_price and
                latest_bar.delta > prev_bar.delta
            )
        return features

    def _detect_absorptions(self) -> List[Absorption]:
        absorptions = []
        if len(self.trade_history) < 10:
            return absorptions
        recent_trades = self.trade_history[-200:]
        duration = (recent_trades[-1].timestamp - recent_trades[0].timestamp).total_seconds()
        avg_volume_per_second = sum(t.size for t in recent_trades) / (duration + 1)
        min_duration = self.config.absorption_min_duration_sec
        for i in range(len(recent_trades)):
            start_trade = recent_trades[i]
            target_time = start_trade.timestamp + timedelta(seconds=min_duration)
            end_idx = -1
            for j in range(i + 1, len(recent_trades)):
                if recent_trades[j].timestamp >= target_time:
                    end_idx = j
                    break
            if end_idx == -1:
                break
            window_trades = recent_trades[i:end_idx + 1]
            window_volume = sum(t.size for t in window_trades)
            window_duration = (window_trades[-1].timestamp - window_trades[0].timestamp).total_seconds()
            if window_duration < min_duration:
                continue
            price_change_pct = abs(
                window_trades[-1].price - window_trades[0].price
            ) / (window_trades[0].price + 1e-9)
            volume_rate = window_volume / (window_duration + 1e-9)
            if (volume_rate > avg_volume_per_second * self.config.absorption_volume_multiplier and
                price_change_pct < self.config.absorption_price_threshold_pct):
                buy_vol = sum(t.size for t in window_trades if t.side == Side.BUY)
                sell_vol = sum(t.size for t in window_trades if t.side == Side.SELL)
                if buy_vol > sell_vol:
                    absorbing_side = Side.BUY
                    strength = sell_vol / (buy_vol + 1e-9)
                else:
                    absorbing_side = Side.SELL
                    strength = buy_vol / (sell_vol + 1e-9)
                absorptions.append(Absorption(
                    timestamp=window_trades[-1].timestamp,
                    price=window_trades[-1].price,
                    absorbed_volume=window_volume,
                    price_change_pct=price_change_pct,
                    duration_seconds=window_duration,
                    absorbing_side=absorbing_side,
                    strength=min(strength, 1.0)
                ))
        return absorptions[-10:]

    def _detect_imbalances(self, book: OrderBook) -> List[Imbalance]:
        imbalances = []
        if not book.bids or not book.asks:
            return imbalances
        mid = book.mid_price
        bid_imbalances = self._find_side_imbalances(book.bids, book.asks, Side.BUY, mid)
        ask_imbalances = self._find_side_imbalances(book.asks, book.bids, Side.SELL, mid)
        imbalances.extend(bid_imbalances)
        imbalances.extend(ask_imbalances)
        self._mark_stacked_imbalances(imbalances)
        return imbalances

    def _find_side_imbalances(
        self,
        levels: List[PriceLevel],
        opposing_levels: List[PriceLevel],
        side: Side,
        mid_price: float
    ) -> List[Imbalance]:
        imbalances = []
        tolerance = self.config.tick_size * 1.5
        for level in levels[:self.config.book_depth_levels]:
            distance = abs(mid_price - level.price)
            mirror_price = mid_price + distance if side == Side.BUY else mid_price - distance
            matching_opp_size = 0
            for opp in opposing_levels:
                if abs(opp.price - mirror_price) <= tolerance:
                    matching_opp_size = opp.size
                    break
            ratio = level.size / matching_opp_size if matching_opp_size > 0 else float('inf')
            if ratio >= self.config.imbalance_ratio_threshold:
                imbalances.append(Imbalance(
                    timestamp=self._current_timestamp or datetime.now(),
                    price=level.price,
                    buy_volume=level.size if side == Side.BUY else matching_opp_size,
                    sell_volume=matching_opp_size if side == Side.BUY else level.size,
                    imbalance_ratio=ratio,
                    direction=side
                ))
        return imbalances

    def _mark_stacked_imbalances(self, imbalances: List[Imbalance]) -> None:
        if len(imbalances) < self.config.stacked_imbalance_min_levels:
            return
        sorted_imbalances = sorted(imbalances, key=lambda x: x.price)
        current_stack = [sorted_imbalances[0]]
        for i in range(1, len(sorted_imbalances)):
            if sorted_imbalances[i].direction == current_stack[-1].direction:
                current_stack.append(sorted_imbalances[i])
            else:
                if len(current_stack) >= self.config.stacked_imbalance_min_levels:
                    for imb in current_stack:
                        imb.is_stacked = True
                        imb.stack_count = len(current_stack)
                current_stack = [sorted_imbalances[i]]
        if len(current_stack) >= self.config.stacked_imbalance_min_levels:
            for imb in current_stack:
                imb.is_stacked = True
                imb.stack_count = len(current_stack)

    def _detect_liquidity_sweeps(self) -> List[LiquiditySweep]:
        sweeps = []
        if len(self.trade_history) < 50:
            return sweeps
        recent_trades = self.trade_history[-200:]
        window_size = 20
        for i in range(window_size * 2, len(recent_trades)):
            sweep_window = recent_trades[i-window_size*2:i-window_size]
            reversal_window = recent_trades[i-window_size:i]
            sweep_duration = (
                sweep_window[-1].timestamp - sweep_window[0].timestamp
            ).total_seconds()
            if sweep_duration > self.config.sweep_speed_threshold_sec:
                continue
            sweep_move = sweep_window[-1].price - sweep_window[0].price
            sweep_move_pct = abs(sweep_move) / (sweep_window[0].price + 1e-9)
            reversal_move = reversal_window[-1].price - reversal_window[0].price
            if (sweep_move_pct > self.config.sweep_reversal_threshold_pct and
                sweep_move * reversal_move < 0):
                reversal_pct = abs(reversal_move) / abs(sweep_move) if sweep_move != 0 else 0
                if reversal_pct > 0.5:
                    sweeps.append(LiquiditySweep(
                        timestamp=reversal_window[-1].timestamp,
                        sweep_price=sweep_window[-1].price,
                        reversal_price=reversal_window[-1].price,
                        direction=Side.BUY if sweep_move > 0 else Side.SELL,
                        volume_swept=sum(t.size for t in sweep_window),
                        speed_seconds=sweep_duration,
                        reversal_strength=reversal_pct
                    ))
        return sweeps[-5:]

    def _detect_icebergs(self) -> List[IcebergOrder]:
        icebergs = []
        if len(self.book_history) < 10:
            return icebergs
        recent_books = self.book_history[-30:]
        refill_counts: Dict[float, Dict] = {}
        for i in range(1, len(recent_books)):
            self._track_refills(
                recent_books[i-1].bids, recent_books[i].bids,
                refill_counts, Side.BUY
            )
            self._track_refills(
                recent_books[i-1].asks, recent_books[i].asks,
                refill_counts, Side.SELL
            )
        traded_prices: set = set()
        for trade in self.trade_history[-100:]:
            bucket = round(trade.price / self.config.tick_size) * self.config.tick_size
            traded_prices.add(bucket)
        for price, data in refill_counts.items():
            if data["refill_count"] >= self.config.iceberg_refill_threshold:
                price_bucket = round(price / self.config.tick_size) * self.config.tick_size
                if price_bucket not in traded_prices:
                    continue
                icebergs.append(IcebergOrder(
                    timestamp=self._current_timestamp or datetime.now(),
                    price=price,
                    visible_size=data["visible_size"],
                    estimated_hidden_size=data["total_refilled"],
                    refill_count=data["refill_count"],
                    side=data["side"],
                    confidence=min(data["refill_count"] / 10, 1.0)
                ))
        return icebergs

    def _track_refills(
        self,
        prev_levels: List[PriceLevel],
        curr_levels: List[PriceLevel],
        refill_counts: Dict[float, Dict],
        side: Side
    ) -> None:
        prev_by_price = {l.price: l.size for l in prev_levels[:20]}
        curr_by_price = {l.price: l.size for l in curr_levels[:20]}
        for price in prev_by_price:
            if price in curr_by_price:
                prev_size = prev_by_price[price]
                curr_size = curr_by_price[price]
                if prev_size < self.config.iceberg_size_threshold and curr_size > prev_size * 2:
                    if price not in refill_counts:
                        refill_counts[price] = {
                            "refill_count": 0,
                            "total_refilled": 0,
                            "visible_size": curr_size,
                            "side": side
                        }
                    refill_counts[price]["refill_count"] += 1
                    refill_counts[price]["total_refilled"] += curr_size - prev_size
                    refill_counts[price]["visible_size"] = curr_size

    def _compute_pattern_features(self, state: OrderFlowState) -> Dict[str, float]:
        features: Dict[str, float] = {}
        now = self._current_timestamp or datetime.now()
        recent_abs = [
            a for a in state.absorptions
            if (now - a.timestamp).total_seconds() < 60
        ]
        if recent_abs:
            latest = recent_abs[-1]
            features["recent_absorption_strength"] = latest.strength
            features["recent_absorption_detected"] = 1.0
            features["recent_absorption_direction"] = (
                1.0 if latest.absorbing_side == Side.BUY else -1.0
            )
            features["max_absorption_strength_60s"] = max(
                a.strength for a in recent_abs
            )
            features["absorption_count_60s"] = float(len(recent_abs))
            features["avg_absorption_volume"] = sum(
                a.absorbed_volume for a in recent_abs
            ) / len(recent_abs)
        else:
            features["recent_absorption_strength"] = 0.0
            features["recent_absorption_detected"] = 0.0
            features["recent_absorption_direction"] = 0.0
            features["max_absorption_strength_60s"] = 0.0
            features["absorption_count_60s"] = 0.0
            features["avg_absorption_volume"] = 0.0
        recent_sw = [
            s for s in state.sweeps
            if (now - s.timestamp).total_seconds() < 120
        ]
        if recent_sw:
            latest = recent_sw[-1]
            features["recent_sweep_detected"] = 1.0
            features["recent_sweep_reversal_strength"] = latest.reversal_strength
            features["recent_sweep_direction"] = (
                1.0 if latest.direction == Side.BUY else -1.0
            )
            features["recent_sweep_volume"] = latest.volume_swept
            features["sweep_count_120s"] = float(len(recent_sw))
        else:
            features["recent_sweep_detected"] = 0.0
            features["recent_sweep_reversal_strength"] = 0.0
            features["recent_sweep_direction"] = 0.0
            features["recent_sweep_volume"] = 0.0
            features["sweep_count_120s"] = 0.0
        stacked = [i for i in state.imbalances if i.is_stacked]
        if stacked:
            features["stacked_imbalance_count"] = float(len(stacked))
            features["max_stack_size"] = float(max(i.stack_count for i in stacked))
            buy_imb = sum(1 for i in stacked if i.direction == Side.BUY)
            sell_imb = sum(1 for i in stacked if i.direction == Side.SELL)
            features["imbalance_direction_ratio"] = (buy_imb - sell_imb) / (
                buy_imb + sell_imb + 1e-9
            )
        else:
            features["stacked_imbalance_count"] = 0.0
            features["max_stack_size"] = 0.0
            features["imbalance_direction_ratio"] = 0.0
        if state.imbalances:
            features["avg_imbalance_ratio"] = sum(
                i.imbalance_ratio for i in state.imbalances
            ) / len(state.imbalances)
        else:
            features["avg_imbalance_ratio"] = 0.0
        if state.icebergs:
            best = max(state.icebergs, key=lambda x: x.confidence)
            features["iceberg_detected"] = 1.0
            features["iceberg_confidence"] = best.confidence
            features["iceberg_direction"] = (
                1.0 if best.side == Side.BUY else -1.0
            )
            features["iceberg_estimated_size"] = best.estimated_hidden_size
            features["iceberg_count"] = float(len(state.icebergs))
        else:
            features["iceberg_detected"] = 0.0
            features["iceberg_confidence"] = 0.0
            features["iceberg_direction"] = 0.0
            features["iceberg_estimated_size"] = 0.0
            features["iceberg_count"] = 0.0
        return features

    def _compute_composite_features(self, base_features: Dict[str, float]) -> Dict[str, float]:
        features: Dict[str, float] = {}
        short_vol = base_features.get("total_volume_60s", 0)
        long_vol = base_features.get("total_volume_300s", 0)
        features["volume_acceleration"] = (short_vol * 5) / (long_vol + 1e-9)
        price_dir_60 = np.sign(base_features.get("price_change_pct_60s", 0))
        delta_dir_60 = np.sign(base_features.get("delta_pct_60s", 0))
        features["delta_divergence_60s"] = float(
            price_dir_60 != delta_dir_60 and price_dir_60 != 0
        )
        cvd_sign = np.sign(base_features.get("cvd", 0))
        price_dir_300 = np.sign(base_features.get("price_change_pct_300s", 0))
        features["cvd_price_divergence"] = float(
            cvd_sign != price_dir_300 and price_dir_300 != 0
        )
        buying_ex = base_features.get("buying_exhaustion", 0)
        selling_ex = base_features.get("selling_exhaustion", 0)
        features["exhaustion_score"] = max(buying_ex, selling_ex)
        net_pressure = base_features.get("net_pressure", 0)
        delta_60 = base_features.get("delta_60s", 0)
        features["pressure_confirmed"] = float(
            np.sign(net_pressure) == np.sign(delta_60) and net_pressure != 0
        )
        in_va = base_features.get("in_value_area", 0)
        vol_accel = features.get("volume_acceleration", 1.0)
        features["va_breakout_potential"] = float(in_va == 0 and vol_accel > 1.5)
        features["price_vs_vwap_pct"] = base_features.get("vwap_deviation_60s", 0)
        book_imbal = features.get("depth_imbalance_10", 0)
        trade_imbal = base_features.get("trade_count_imbalance_60s", 0)
        if (book_imbal > 0.08 and trade_imbal > 0.1) or \
           (book_imbal < -0.08 and trade_imbal < -0.1):
            features["book_trade_agreement"] = 1.0
        else:
            features["book_trade_agreement"] = 0.0
        return features

    def _store_feature_snapshot(
        self,
        timestamp: datetime,
        features: Dict[str, float]
    ) -> None:
        _TRACKED = [
            "recent_absorption_strength", "recent_absorption_direction",
            "depth_imbalance_10", "delta_60s", "delta_pct_60s",
            "footprint_imbalance_count", "volume_acceleration",
            "recent_sweep_detected", "recent_sweep_reversal_strength",
            "net_pressure", "spread_bps", "mid_price",
        ]
        snapshot = {k: features.get(k, 0.0) for k in _TRACKED}
        self._feature_snapshots.append((timestamp, snapshot))
        cutoff = timestamp - timedelta(seconds=60)
        while self._feature_snapshots and self._feature_snapshots[0][0] < cutoff:
            self._feature_snapshots.pop(0)

    def get_feature_at_lag(self, feature_name: str, seconds_ago: float) -> float:
        if not self._feature_snapshots or not self._current_timestamp:
            return 0.0
        target = self._current_timestamp - timedelta(seconds=seconds_ago)
        best_snap = None
        best_diff = float("inf")
        for ts, snap in self._feature_snapshots:
            diff = abs((ts - target).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_snap = snap
        if best_snap and best_diff < seconds_ago * 0.5:
            return best_snap.get(feature_name, 0.0)
        return 0.0
