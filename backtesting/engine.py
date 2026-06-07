"""
Backtesting Engine v5 — Integrated with Enhanced FeatureEngine
=================================================================
Fixes from v4:
1. Accepts FeatureConfig from caller (tick_size=0.5, regime params, etc.)
2. Removes duplicate footprint bar system (FeatureEngine is single source)
3. Removes duplicate regime detection (RegimeClassifier is single source)
4. WalkForwardValidator properly isolates trials for parallel execution
5. Min-trades filtering in OOS results

Performance optimisations from v4 preserved:
- Pre-extracted NumPy arrays
- Lightweight _Row container
- Pattern/volume profile throttling
- Equity curve downsampling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import copy
import math

from loguru import logger

from core.data_structures import (
    OrderFlowState, Signal, SignalType, Side, Trade,
    OrderBook, PriceLevel, FootprintBar, Regime
)
from core.feature_engine import FeatureEngine, FeatureConfig
from core.fee_aware_filter import FeeAwareFilter
from core.feature_precomputer import FeaturePrecomputer
from knowledge.strategy_library import StrategyDefinition
from execution.risk_manager import RiskManager, RiskLimits, RiskAction


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Track an open position"""
    entry_time: datetime
    entry_price: float
    allocated_capital: float
    size: float
    side: Side
    stop_loss: float
    take_profit: float
    signal: Signal
    strategy_name: str
    trailing_stop_activation_pct: float
    risk_action: str = ""

    entry_fee: float = 0.0
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0


@dataclass
class ClosedTrade:
    """Record of a completed trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    side: Side
    pnl: float
    pnl_pct: float
    exit_reason: str
    duration_seconds: float
    signal_confidence: float
    risk_action: str = ""
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    avg_trade_duration_seconds: float = 0.0
    max_consecutive_losses: int = 0

    trades_per_day: float = 0.0
    calmar_ratio: float = 0.0

    signals_rejected_by_risk: int = 0
    signals_reduced_by_risk: int = 0
    halts_triggered: int = 0
    suspicious_pnl_rejected: int = 0
    signals_rejected_by_fee_filter: int = 0


# ---------------------------------------------------------------------------
# Lightweight row container
# ---------------------------------------------------------------------------

class _Row:
    """Ultra-lightweight row accessor backed by raw numpy arrays."""
    __slots__ = (
        'timestamp', 'bid_price', 'ask_price', 'bid_size', 'ask_size',
        'trade_price', 'trade_size', 'trade_side',
        '_depth_bids_p', '_depth_bids_s', '_depth_asks_p', '_depth_asks_s',
        '_has_depth', '_n_depth',
    )

    def __init__(
        self,
        timestamp, bid_price, ask_price, bid_size, ask_size,
        trade_price, trade_size, trade_side,
        depth_bids_p=None, depth_bids_s=None,
        depth_asks_p=None, depth_asks_s=None,
    ):
        self.timestamp = timestamp
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.trade_price = trade_price
        self.trade_size = trade_size
        self.trade_side = trade_side
        self._depth_bids_p = depth_bids_p
        self._depth_bids_s = depth_bids_s
        self._depth_asks_p = depth_asks_p
        self._depth_asks_s = depth_asks_s
        self._has_depth = depth_bids_p is not None
        self._n_depth = len(depth_bids_p) if depth_bids_p is not None else 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven backtesting engine — integrated with enhanced FeatureEngine.
    """

    # Throttling knobs
    PATTERN_DETECTION_EVERY_N = 25
    EQUITY_SAMPLE_EVERY_N = 50
    VOLUME_PROFILE_EVERY_N = 100

    # Minimum hold before flow-based exits can fire (20 minutes)
    MIN_HOLD_BEFORE_FLOW_EXIT_SEC = 1200

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_pct: float = 0.0005,
        slippage_pct: float = 0.0003,
        sl_extra_slippage_pct: float = 0.0003,
        warmup_seconds: float = 60.0,
        min_time_between_trades_sec: float = 30.0,
        equity_floor_pct: float = 0.50,
        suspicious_pnl_pct: float = 0.30,
        risk_limits: Optional[RiskLimits] = None,
        feature_config: Optional[FeatureConfig] = None,  # NEW: accept config
    ):
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.sl_extra_slippage_pct = sl_extra_slippage_pct
        self.warmup_seconds = warmup_seconds
        self.min_time_between_trades_sec = min_time_between_trades_sec
        self.equity_floor_pct = equity_floor_pct
        self.equity_floor = initial_capital * equity_floor_pct
        self.suspicious_pnl_pct = suspicious_pnl_pct

        # FIX: Store feature config for creating FeatureEngine
        self._feature_config = feature_config or FeatureConfig()
        self.windows = self._feature_config.windows

        # Feature precomputer (optional, created during run())
        self.precomputer: Optional[FeaturePrecomputer] = None

        # Fee-aware filter (Futures fee structure: 0.02% maker, 0.05% taker)
        self.fee_filter = FeeAwareFilter(
            maker_fee_pct=0.0002,
            taker_fee_pct=0.0005,
            expected_spread_pct=0.0001,
            min_profit_target_pct=0.0002
        )

        self.risk_limits = risk_limits or RiskLimits(
            max_position_size=10000.0, #1.0 for BTC change to 10000 for XRP
            max_position_value_pct=0.25,
            max_daily_loss_pct=0.02,
            max_weekly_loss_pct=0.05,
            max_drawdown_pct=0.10,
            max_trades_per_day=50,
            max_trades_per_hour=10,
            min_time_between_trades_sec=int(min_time_between_trades_sec),
            max_consecutive_losses=20,
        )

        # State (initialized in reset())
        self.capital: float = 0.0
        self.position: Optional[Position] = None
        self.closed_trades: List[ClosedTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.feature_engine: Optional[FeatureEngine] = None
        self.risk_manager: Optional[RiskManager] = None

        self._last_trade_close_time: Optional[datetime] = None
        self._last_signal_strategy: Optional[str] = None
        self._last_signal_time: Optional[datetime] = None
        self._current_day = None

        # Tracking
        self._signals_rejected = 0
        self._signals_reduced = 0
        self._halts = 0
        self._suspicious_rejected = 0
        self._signals_rejected_by_fee_filter = 0

    def reset(self) -> None:
        """Reset state for new backtest."""
        self.capital = self.initial_capital
        self.position = None
        self.closed_trades = []
        self.equity_curve = []
        
        # FIX: Create FeatureEngine with proper config
        self.feature_engine = FeatureEngine(self._feature_config)
        
        self.risk_manager = RiskManager(
            limits=self.risk_limits,
            initial_equity=self.initial_capital,
        )
        self._last_trade_close_time = None
        self._last_signal_strategy = None
        self._last_signal_time = None
        self._current_day = None
        self._signals_rejected = 0
        self._signals_reduced = 0
        self._halts = 0
        self._suspicious_rejected = 0
        self._signals_rejected_by_fee_filter = 0
        self._entry_tick_idx: int = -1  # [FIXED] Track entry tick to prevent same-tick exits
        self._current_tick_idx: int = -1  # [FIXED] Track current tick for entry-tick guard

    # ------------------------------------------------------------------
    # DataFrame → numpy pre-extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(data: pd.DataFrame) -> List[_Row]:
        """Convert DataFrame to lightweight _Row objects."""
        
        # === FIX: ALWAYS Map _0 columns to base columns ===
        # - If base columns don't exist, copy from _0
        # - If base columns exist but are all zeros/NaN, replace with _0
        if 'bid_price_0' in data.columns:
            data = data.copy()
            
            # Check if bid_price is invalid (missing, all zeros, or all NaN)
            if 'bid_price' not in data.columns or \
               (data['bid_price'].fillna(0) == 0).all() or \
               data['bid_price'].isna().all():
                data['bid_price'] = data['bid_price_0']
                
            if 'ask_price' not in data.columns or \
               (data['ask_price'].fillna(0) == 0).all() or \
               data['ask_price'].isna().all():
                data['ask_price'] = data['ask_price_0']
                
            if 'bid_size' not in data.columns or \
               (data['bid_size'].fillna(0) == 0).all() or \
               data['bid_size'].isna().all():
                data['bid_size'] = data['bid_size_0']
                
            if 'ask_size' not in data.columns or \
               (data['ask_size'].fillna(0) == 0).all() or \
               data['ask_size'].isna().all():
                data['ask_size'] = data['ask_size_0']
        # ========================================================
        
        timestamps = pd.to_datetime(data['timestamp']).dt.to_pydatetime()

        bid_price = data['bid_price'].values if 'bid_price' in data.columns else np.zeros(len(data))
        ask_price = data['ask_price'].values if 'ask_price' in data.columns else np.zeros(len(data))
        bid_size = data['bid_size'].values if 'bid_size' in data.columns else np.zeros(len(data))
        ask_size = data['ask_size'].values if 'ask_size' in data.columns else np.zeros(len(data))

        trade_price = data['trade_price'].values if 'trade_price' in data.columns else np.full(len(data), np.nan)
        trade_size = data['trade_size'].values if 'trade_size' in data.columns else np.zeros(len(data))

        if 'trade_side' in data.columns:
            trade_side = data['trade_side'].astype(str).str.lower().values
        else:
            trade_side = np.full(len(data), 'buy')

        has_depth = 'bid_price_0' in data.columns
        depth_bids_p = depth_bids_s = depth_asks_p = depth_asks_s = None

        if has_depth:
            n_depth = 0
            while f'bid_price_{n_depth}' in data.columns:
                n_depth += 1

            zeros = np.zeros(len(data))
            depth_bids_p = np.column_stack([
                data.get(f'bid_price_{i}', zeros) for i in range(n_depth)
            ])
            depth_bids_s = np.column_stack([
                data.get(f'bid_size_{i}', zeros) for i in range(n_depth)
            ])
            depth_asks_p = np.column_stack([
                data.get(f'ask_price_{i}', zeros) for i in range(n_depth)
            ])
            depth_asks_s = np.column_stack([
                data.get(f'ask_size_{i}', zeros) for i in range(n_depth)
            ])

        n = len(data)
        rows: List[_Row] = []
        for i in range(n):
            ts = timestamps[i]
            rows.append(_Row(
                timestamp=ts,
                bid_price=float(bid_price[i]),
                ask_price=float(ask_price[i]),
                bid_size=float(bid_size[i]),
                ask_size=float(ask_size[i]),
                trade_price=float(trade_price[i]),
                trade_size=float(trade_size[i]),
                trade_side=str(trade_side[i]),
                depth_bids_p=depth_bids_p[i] if has_depth else None,
                depth_bids_s=depth_bids_s[i] if has_depth else None,
                depth_asks_p=depth_asks_p[i] if has_depth else None,
                depth_asks_s=depth_asks_s[i] if has_depth else None,
            ))

        return rows

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        strategy: StrategyDefinition,
        params: Optional[Dict[str, Any]] = None,
    ) -> BacktestMetrics:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with columns: timestamp, bid_price, ask_price,
                  bid_size, ask_size, trade_price, trade_size, trade_side
            strategy: Strategy to test
            params: Override strategy parameters

        Returns:
            BacktestMetrics
        """
        self.reset()

        if params:
            strategy = self._apply_params(strategy, params)

        rows = self._preprocess(data)
        n_rows = len(rows)
        logger.debug(f"Running backtest with {n_rows} data points")
        
        # Precompute features before loop (massive speedup)
        self.precomputer = FeaturePrecomputer(windows=self.windows)
        self.precomputer.precompute_all(data)
        
        # DEPTH CHECK - Verify data loading
        if len(rows) > 0:
            sample = rows[0]
            logger.info(f"DEPTH CHECK: Has depth={sample._has_depth}, Levels={sample._n_depth}")
            if not sample._has_depth:
                logger.error("WARNING: No depth data detected in backtest data! Order book will have only best bid/ask.")
            else:
                logger.info(f"✓ Depth data loaded successfully ({sample._n_depth} levels per row)")

        first_timestamp: Optional[datetime] = None
        state: Optional[OrderFlowState] = None
        timestamp = None

        pattern_every = self.PATTERN_DETECTION_EVERY_N
        equity_every = self.EQUITY_SAMPLE_EVERY_N
        vp_every = self.VOLUME_PROFILE_EVERY_N

        for tick_idx in range(n_rows):
            self._current_tick_idx = tick_idx  # [FIXED] Track current tick for entry-tick guard
            row = rows[tick_idx]
            timestamp = row.timestamp
            
            # DEBUG: First 5 iterations - verify order book construction
            if tick_idx < 5:
                logger.debug(f"DEBUG Row {tick_idx}: Has depth={row._has_depth}, Levels={row._n_depth}, "
                           f"Best bid={row.bid_price}, Best ask={row.ask_price}")

            if first_timestamp is None:
                first_timestamp = timestamp
                self.equity_curve.append((first_timestamp, self.initial_capital))

            # Daily reset
            self._handle_daily_reset(timestamp)

            # Equity floor check
            current_equity = self._calculate_equity_fast()
            if current_equity < self.equity_floor:
                logger.warning(f"Equity floor breached. Halting.")
                break

            # Build order book
            order_book = self._build_order_book_fast(row, timestamp)

            # Build trades
            trades = self._build_trades_fast(row, timestamp)

            # FIX: REMOVED _accumulate_footprint() — FeatureEngine handles this

            # Update feature engine (single source of truth)
            run_patterns = (tick_idx % pattern_every == 0)
            run_vp = (tick_idx % vp_every == 0)

            precomputed = self.precomputer.get_features_for_tick(tick_idx) if self.precomputer else None

            state = self.feature_engine.update(
                order_book, trades,
                detect_patterns=run_patterns,
                compute_volume_profile=run_vp,
                precomputed_features=precomputed,
            )

            # FIX: REMOVED state.regime override — FeatureEngine's classifier is used

            # Update open position
            if self.position:
                self._update_position(state)
                self._update_trailing_stop(state)

                if tick_idx > self._entry_tick_idx:  # Only gate the EXIT check
                    exit_reason = self._check_exit_conditions(state, timestamp)
                    if exit_reason:
                        self._close_position(state, timestamp, exit_reason, strategy)

            # Warmup gate
            elapsed = (timestamp - first_timestamp).total_seconds()
            if elapsed < self.warmup_seconds:
                if tick_idx % equity_every == 0:
                    self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                continue

            # Signal evaluation (only when flat)
            if not self.position:
                if not self._cooldown_passed(timestamp):
                    if tick_idx % equity_every == 0:
                        self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                    continue

                # Loss streak cooldown: skip signal after 2 consecutive losses
                recent_trades = self.closed_trades[-2:]
                if len(recent_trades) == 2 and all(t.pnl <= 0 for t in recent_trades):
                    if tick_idx % equity_every == 0:
                        self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                    continue

                signal = strategy.evaluate(state)

                if signal and signal.is_actionable:
                    # LONG-ONLY gate: reject SELL/STRONG_SELL signals
                    if signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL):
                        if tick_idx % equity_every == 0:
                            self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                        continue

                    if self._is_duplicate_signal(signal, strategy, timestamp):
                        if tick_idx % equity_every == 0:
                            self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                        continue

                    risk_action, adjusted_signal, reason = self.risk_manager.check_signal(
                        signal, state.order_book.mid_price, timestamp
                    )

                    if risk_action == RiskAction.HALT_TRADING:
                        self._halts += 1
                        continue
                    elif risk_action == RiskAction.REJECT:
                        self._signals_rejected += 1
                        continue

                    # Fee-aware filter check (before opening position)
                    signal_to_check = adjusted_signal or signal
                    # Use actual signal TP distance when available, not ATR estimate [FIX 5 & 7]
                    if signal_to_check.entry_price > 0 and signal_to_check.take_profit != signal_to_check.entry_price:
                        predicted_move_pct = abs(signal_to_check.take_profit - signal_to_check.entry_price) / signal_to_check.entry_price
                    else:
                        predicted_move_pct = self._estimate_predicted_move(state, strategy, signal_to_check)
                    if predicted_move_pct > 0:
                        should_ignore, fee_reason = self.fee_filter.should_ignore_signal(
                            signal_to_check, predicted_move_pct, signal_to_check.confidence
                        )
                        if should_ignore:
                            self._signals_rejected_by_fee_filter += 1
                            logger.info(f"Fee filter rejected signal: {fee_reason}")
                            if tick_idx % equity_every == 0:
                                self.equity_curve.append((timestamp, self._calculate_equity_fast()))
                            continue

                    if risk_action == RiskAction.REDUCE_SIZE:
                        self._signals_reduced += 1
                        self._open_position(adjusted_signal, state, timestamp, strategy, risk_action.name)
                        self._entry_tick_idx = tick_idx  # [FIXED] Record the tick we opened on
                    elif risk_action == RiskAction.ALLOW:
                        self._open_position(adjusted_signal or signal, state, timestamp, strategy, risk_action.name)
                        self._entry_tick_idx = tick_idx  # [FIXED] Record the tick we opened on

            # Equity curve (down-sampled)
            if tick_idx % equity_every == 0:
                self.equity_curve.append((timestamp, self._calculate_equity_fast()))

        # Close remaining position
        if self.position and state is not None and timestamp is not None:
            self._close_position(state, timestamp, "end_of_backtest", strategy)

        if timestamp is not None:
            self.equity_curve.append((timestamp, self._calculate_equity_fast()))

        return self._calculate_metrics()

    # ------------------------------------------------------------------
    # Parameter application
    # ------------------------------------------------------------------

    def _apply_params(
        self,
        strategy: StrategyDefinition,
        params: Dict[str, Any],
    ) -> StrategyDefinition:
        """
        Apply optimized parameters to strategy definition.
        CRITICAL: Handles param_key mapping and symmetric ranges correctly.
        DEBUG: Logs every parameter application for verification.
        """
        strategy = copy.deepcopy(strategy)
        applied_keys = set()
        
        logger.debug(f"[_apply_params] Applying {len(params)} parameters to {strategy.name}")
        
        for param_name, value in params.items():
            if value is None:
                logger.debug(f"[_apply_params] Skipping {param_name}: None value")
                continue
            
            # DEBUG: Log parameter being processed
            logger.debug(f"[_apply_params] Processing: {param_name} = {value}")
            
            # 1. Direct strategy attributes (regime multipliers, risk params)
            # CRITICAL: Check if this is a direct attribute of the strategy
            if hasattr(strategy, param_name):
                current_val = getattr(strategy, param_name)
                setattr(strategy, param_name, value)
                applied_keys.add(param_name)
                logger.debug(f"[_apply_params] Applied to attribute: {param_name} ({current_val} -> {value})")
                continue
            
            # 2. Entry conditions and filters via param_key
            all_conditions = list(strategy.entry_conditions) + list(strategy.filters)
            matched = False
            
            for cond in all_conditions:
                if getattr(cond, "param_key", None) != param_name:
                    continue
                
                # CRITICAL: Handle symmetric "between" ranges
                if param_name.endswith("_range") and cond.operator == "between":
                    abs_val = abs(value)
                    old_low, old_high = cond.threshold, cond.threshold_high
                    cond.threshold = -abs_val
                    cond.threshold_high = abs_val
                    logger.debug(f"[_apply_params] Applied symmetric range: {param_name} = ±{abs_val} (was {old_low} to {old_high})")
                # Handle asymmetric high threshold
                elif param_name.endswith("_high") and hasattr(cond, "threshold_high"):
                    cond.threshold_high = value
                    logger.debug(f"[_apply_params] Applied threshold_high: {param_name} = {value}")
                # Standard threshold
                else:
                    old_val = cond.threshold
                    cond.threshold = value
                    logger.debug(f"[_apply_params] Applied threshold: {param_name} = {value} (was {old_val})")
                
                applied_keys.add(param_name)
                matched = True
                break
            
            if not matched:
                logger.warning(f"[_apply_params] Parameter {param_name} not matched to any condition or attribute")
        
        # CRITICAL: Validate all conditions after modification
        logger.debug(f"[_apply_params] Validating {len(strategy.entry_conditions)} entry conditions and {len(strategy.filters)} filters")
        for cond in strategy.entry_conditions + strategy.filters:
            try:
                cond.validate()
            except ValueError as e:
                logger.error(f"[_apply_params] Validation failed for {cond.feature}: {e}")
                raise
        
        # Warn about unmapped parameters
        unmapped = set(params.keys()) - applied_keys
        if unmapped:
            logger.warning(f"[_apply_params] UNMAPPED PARAMETERS (will have no effect): {unmapped}")
        
        # Guard: min_conditions_satisfied must never exceed the number of actual
        # entry conditions \u2014 otherwise the strategy can never fire.  # [FIXED]
        entry_count = len(strategy.entry_conditions)
        if strategy.min_conditions_satisfied > entry_count:
            logger.warning(
                f"[_apply_params] min_conditions_satisfied={strategy.min_conditions_satisfied} "
                f"exceeds entry condition count={entry_count}. "
                f"Clamping to {entry_count}."
            )
            strategy.min_conditions_satisfied = entry_count
        
        logger.info(f"[_apply_params] Successfully applied {len(applied_keys)}/{len(params)} parameters to {strategy.name}")
        return strategy

    # ------------------------------------------------------------------
    # Fast order book builder
    # ------------------------------------------------------------------

    def _build_order_book_fast(self, row: _Row, timestamp: datetime) -> OrderBook:
        """Build OrderBook from pre-extracted _Row."""
        best_bid = row.bid_price
        best_ask = row.ask_price
        best_bid_size = row.bid_size
        best_ask_size = row.ask_size

        if best_bid == 0 and best_ask == 0:
            tp = row.trade_price
            if not math.isnan(tp) and tp > 0:
                spread_est = tp * 0.0001
                best_bid = tp - spread_est / 2
                best_ask = tp + spread_est / 2
                best_bid_size = row.trade_size if row.trade_size > 0 else 1.0
                best_ask_size = best_bid_size

        n_levels = 20
        bids: List[PriceLevel] = []
        asks: List[PriceLevel] = []

        if row._has_depth:
            for i in range(min(n_levels, row._n_depth)):
                bp = float(row._depth_bids_p[i])
                bs = float(row._depth_bids_s[i])
                ap = float(row._depth_asks_p[i])
                a_s = float(row._depth_asks_s[i])
                if bp > 0 and bs > 0:
                    bids.append(PriceLevel(price=bp, size=bs, timestamp=timestamp))
                if ap > 0 and a_s > 0:
                    asks.append(PriceLevel(price=ap, size=a_s, timestamp=timestamp))
        else:
            # NO SYNTHETIC DEPTH - Data missing real depth levels
            logger.warning(f"No depth data available at {timestamp}. Book contains only best bid/ask.")
            if best_bid > 0:
                bids.append(PriceLevel(price=best_bid, size=best_bid_size, timestamp=timestamp))
            if best_ask > 0:
                asks.append(PriceLevel(price=best_ask, size=best_ask_size, timestamp=timestamp))

        bids.sort(key=lambda l: l.price, reverse=True)
        asks.sort(key=lambda l: l.price)

        return OrderBook(timestamp=timestamp, bids=bids, asks=asks)

    # ------------------------------------------------------------------
    # Fast trade builder
    # ------------------------------------------------------------------

    def _build_trades_fast(self, row: _Row, timestamp: datetime) -> List[Trade]:
        """Build Trade list from pre-extracted _Row."""
        if math.isnan(row.trade_price) or row.trade_price <= 0:
            return []
        if math.isnan(row.trade_size) or row.trade_size <= 0:
            return []

        side = Side.BUY if str(row.trade_side).lower().strip() == 'buy' else Side.SELL
        return [Trade(
            timestamp=timestamp,
            price=row.trade_price,
            size=row.trade_size,
            side=side,
        )]

    # NOTE: _accumulate_footprint() REMOVED — FeatureEngine handles this

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _update_position(self, state: OrderFlowState) -> None:
        """Update position P&L using best bid/ask."""
        if not self.position:
            return

        book = state.order_book
        mid = book.mid_price

        if self.position.side == Side.BUY:
            mark_price = book.best_bid.price if book.best_bid else mid
            self.position.unrealized_pnl = (mark_price - self.position.entry_price) * self.position.size
        else:
            mark_price = book.best_ask.price if book.best_ask else mid
            self.position.unrealized_pnl = (self.position.entry_price - mark_price) * self.position.size

        self.position.highest_price = max(self.position.highest_price, mid)
        if self.position.lowest_price <= 0:
            self.position.lowest_price = mid
        else:
            self.position.lowest_price = min(self.position.lowest_price, mid)

    def _update_trailing_stop(self, state: OrderFlowState) -> None:
        """Activate and ratchet trailing stop."""
        if not self.position:
            return

        book = state.order_book
        activation = self.position.trailing_stop_activation_pct
        if activation <= 0:
            return

        if self.position.side == Side.BUY:
            mark = book.best_bid.price if book.best_bid else book.mid_price
            move_pct = (mark - self.position.entry_price) / self.position.entry_price
            if move_pct >= activation:
                self.position.trailing_stop_active = True
                trail_distance = activation * 0.5
                new_stop = mark * (1 - trail_distance)
                self.position.trailing_stop_price = max(
                    self.position.trailing_stop_price, new_stop
                )
                if self.position.trailing_stop_price > self.position.stop_loss:
                    self.position.stop_loss = self.position.trailing_stop_price
        else:
            mark = book.best_ask.price if book.best_ask else book.mid_price
            move_pct = (self.position.entry_price - mark) / self.position.entry_price
            if move_pct >= activation:
                self.position.trailing_stop_active = True
                trail_distance = activation * 0.5
                new_stop = mark * (1 + trail_distance)
                if self.position.trailing_stop_price <= 0:
                    self.position.trailing_stop_price = new_stop
                else:
                    self.position.trailing_stop_price = min(
                        self.position.trailing_stop_price, new_stop
                    )
                if self.position.trailing_stop_price < self.position.stop_loss:
                    self.position.stop_loss = self.position.trailing_stop_price

    # ------------------------------------------------------------------
    # Exit conditions
    # ------------------------------------------------------------------

    def _check_exit_conditions(
        self,
        state: OrderFlowState,
        timestamp: datetime,
    ) -> Optional[str]:
        """Check if position should be closed."""
        # Guard: Never exit on the same tick the position was opened.
        # Without this guard, a valid exit condition present at entry
        # (e.g., the same absorption event that triggered entry) will
        # immediately close the trade before the market can move.
        if hasattr(self, '_entry_tick_idx') and self._current_tick_idx <= self._entry_tick_idx:
            return None  # [FIXED] Too early to exit — still on entry tick
        
        # [APPLIED] No time-based exit logic exists here - exits are flow-based (SL/TP/absorption/divergence/etc)
        # max_holding_seconds=86400 in StrategyDefinition makes time exits effectively unlimited
        if not self.position:
            return None

        book = state.order_book
        features = state.features

        # [FIX] Minimum hold before flow-based exits: only SL/TP allowed before MIN_HOLD_BEFORE_FLOW_EXIT_SEC
        hold_duration = (timestamp - self.position.entry_time).total_seconds()
        flow_exits_allowed = hold_duration >= self.MIN_HOLD_BEFORE_FLOW_EXIT_SEC

        if self.position.side == Side.BUY:
            exit_check_price = book.best_bid.price if book.best_bid else book.mid_price
        else:
            exit_check_price = book.best_ask.price if book.best_ask else book.mid_price

        # 1) Hard stop loss (always active, no minimum hold required)
        if self.position.side == Side.BUY:
            if exit_check_price <= self.position.stop_loss:
                return "stop_loss"
        else:
            if exit_check_price >= self.position.stop_loss:
                return "stop_loss"

        # 2) Take profit (always active, no minimum hold required)
        if self.position.side == Side.BUY:
            if exit_check_price >= self.position.take_profit:
                return "take_profit"
        else:
            if exit_check_price <= self.position.take_profit:
                return "take_profit"

        # [DISABLED] 3a) Absorption against
        # [DISABLED] 3b) Delta divergence against

        # [DISABLED] 3c) Exhaustion

        # 3c) Sweep against (requires minimum hold)
        if flow_exits_allowed and state.sweeps:
            latest_sweep = state.sweeps[-1]
            if (self.position.side == Side.BUY and
                    latest_sweep.direction == Side.SELL and
                    latest_sweep.reversal_strength < 0.4):
                return "sweep_against_long"
            if (self.position.side == Side.SELL and
                    latest_sweep.direction == Side.BUY and
                    latest_sweep.reversal_strength < 0.4):
                return "sweep_against_short"

        # 3d) Book pressure collapse (requires minimum hold)
        if flow_exits_allowed:
            net_pressure = features.get("net_pressure", 0)
            if self.position.side == Side.BUY and net_pressure < -0.5:
                bid_depth = features.get("bid_depth_10", 0)
                ask_depth = features.get("ask_depth_10", 0)
                if ask_depth > 0 and bid_depth / (ask_depth + 1e-9) < 0.3:
                    return "book_pressure_collapse"
            if self.position.side == Side.SELL and net_pressure > 0.5:
                bid_depth = features.get("bid_depth_10", 0)
                ask_depth = features.get("ask_depth_10", 0)
                if bid_depth > 0 and ask_depth / (bid_depth + 1e-9) < 0.3:
                    return "book_pressure_collapse"

        return None

    # ------------------------------------------------------------------
    # Fee filter helper
    # ------------------------------------------------------------------

    def _estimate_predicted_move(
        self,
        state: OrderFlowState,
        strategy: StrategyDefinition,
        signal: Optional[Signal] = None,
    ) -> float:
        """
        Estimate the predicted price move based on signal TP distance or ATR + TP multiplier.
        Used by fee-aware filter to determine if signal covers trading costs.

        [CHANGED 2026-05-27] _estimate_atr now returns a fraction of price (atr_pct),
        so the formula simplifies: atr_pct * tp_mult (no more / mid_price division).
        Previously: (atr_dollar * tp_mult) / mid_price

        Priority:
          1. Use signal take_profit distance when available
          2. Fall back to atr_pct * tp_mult estimate
          3. Return min_profit from fee filter as absolute floor

        Returns:
            Predicted move as decimal (0.01 = 1%), or 0.0 if not calculable
        """
        # Priority 1: Use signal TP distance [FIX 5 & 7]
        if signal is not None and signal.entry_price > 0 and signal.take_profit != signal.entry_price:
            return abs(signal.take_profit - signal.entry_price) / signal.entry_price

        mid_price = state.order_book.mid_price
        if mid_price <= 0:
            return 0.0

        atr_pct = strategy._estimate_atr(state)
        if atr_pct <= 0:
            return 0.0

        # Use the regime-appropriate take profit multiplier
        regime = state.regime
        if regime == Regime.HIGH_VOLATILITY:
            tp_mult = strategy.tp_mult_high_vol
        elif regime in (Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION):
            tp_mult = strategy.tp_mult_low_vol
        else:
            tp_mult = strategy.tp_mult_trending

        predicted_move_pct = atr_pct * tp_mult
        return predicted_move_pct

    # ------------------------------------------------------------------
    # Open/Close position
    # ------------------------------------------------------------------

    def _open_position(
        self,
        signal: Signal,
        state: OrderFlowState,
        timestamp: datetime,
        strategy: StrategyDefinition,
        risk_action: str = "ALLOW",
    ) -> None:
        """Open a new position using best bid/ask fill model."""
        book = state.order_book
        best_bid = book.best_bid.price if book.best_bid else book.mid_price
        best_ask = book.best_ask.price if book.best_ask else book.mid_price

        # LONG-ONLY: Always go long
        entry_price = best_ask * (1 + self.slippage_pct)
        side = Side.BUY

        position_value = self.capital * signal.position_size
        size = position_value / entry_price

        entry_fee = position_value * self.fee_pct
        allocated = position_value + entry_fee

        if allocated > self.capital:
            allocated = self.capital * 0.95
            position_value = allocated / (1 + self.fee_pct)
            size = position_value / entry_price
            entry_fee = position_value * self.fee_pct

        self.capital -= allocated

        self.position = Position(
            entry_time=timestamp,
            entry_price=entry_price,
            allocated_capital=allocated,
            size=size,
            side=side,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal=signal,
            strategy_name=strategy.name,
            trailing_stop_activation_pct=strategy.trailing_stop_activation_pct,
            risk_action=risk_action,
            highest_price=entry_price,
            lowest_price=entry_price,
            entry_fee=entry_fee,
        )

        self.risk_manager.record_trade_opened(entry_price, size, side, timestamp)
        self._last_signal_time = timestamp
        self._last_signal_strategy = strategy.name

    def _close_position(
        self,
        state: OrderFlowState,
        timestamp: datetime,
        reason: str,
        strategy: StrategyDefinition,
    ) -> None:
        """Close current position using best bid/ask fill model."""
        if not self.position:
            return

        book = state.order_book
        best_bid = book.best_bid.price if book.best_bid else book.mid_price
        best_ask = book.best_ask.price if book.best_ask else book.mid_price

        adverse_slip = (self.slippage_pct + self.sl_extra_slippage_pct 
                       if reason == "stop_loss" else self.slippage_pct)

        if self.position.side == Side.BUY:
            exit_price = best_bid * (1 - adverse_slip)
            gross_pnl = (exit_price - self.position.entry_price) * self.position.size
        else:
            exit_price = best_ask * (1 + adverse_slip)
            gross_pnl = (self.position.entry_price - exit_price) * self.position.size

        exit_value = exit_price * self.position.size
        exit_fee = exit_value * self.fee_pct
        pnl = gross_pnl - exit_fee

        logger.info(
            f"EXIT: {reason} | Entry: {self.position.entry_price} | Exit: {exit_price} | PnL: {pnl}"
        )

        notional = self.position.entry_price * self.position.size
        pnl_pct = pnl / notional if notional > 0 else 0.0

        # Suspicious PnL rejection
        if abs(pnl_pct) > self.suspicious_pnl_pct and reason != "end_of_backtest":
            self._suspicious_rejected += 1
            self.capital += self.position.allocated_capital
            self.position = None
            return

        self.capital += (self.position.allocated_capital - self.position.entry_fee) + pnl

        duration = (timestamp - self.position.entry_time).total_seconds()

        self.closed_trades.append(ClosedTrade(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            size=self.position.size,
            side=self.position.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            duration_seconds=duration,
            signal_confidence=self.position.signal.confidence,
            risk_action=self.position.risk_action,
            stop_loss=self.position.stop_loss,
            take_profit=self.position.take_profit,
        ))

        self.risk_manager.record_trade_closed(pnl)
        self._last_trade_close_time = timestamp
        self.position = None

    # ------------------------------------------------------------------
    # Signal de-duplication / cooldown
    # ------------------------------------------------------------------

    def _is_duplicate_signal(
        self,
        signal: Signal,
        strategy: StrategyDefinition,
        timestamp: datetime,
    ) -> bool:
        if self._last_signal_time is None:
            return False
        if self._last_signal_strategy != strategy.name:
            return False
        elapsed = (timestamp - self._last_signal_time).total_seconds()
        return elapsed < 5.0

    def _cooldown_passed(self, timestamp: datetime) -> bool:
        if self._last_trade_close_time is None:
            return True
        elapsed = (timestamp - self._last_trade_close_time).total_seconds()
        return elapsed >= self.min_time_between_trades_sec

    # ------------------------------------------------------------------
    # Daily resets
    # ------------------------------------------------------------------

    def _handle_daily_reset(self, timestamp: datetime) -> None:
        day = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        if self._current_day is None:
            self._current_day = day
            return

        if day != self._current_day:
            self._current_day = day
            self.risk_manager.state.daily_pnl = 0.0
            self.risk_manager.state.trades_today = 0
            self.risk_manager.state.trades_this_hour = 0

            if (self.risk_manager.state.trading_halted and
                    "Daily" in self.risk_manager.state.halt_reason):
                self.risk_manager.resume_trading()
                logger.info("New trading day — daily halt lifted.")

    # ------------------------------------------------------------------
    # Equity helpers
    # ------------------------------------------------------------------

    def _calculate_equity_fast(self) -> float:
        equity = self.capital
        if self.position:
            equity += self.position.unrealized_pnl
            equity += (self.position.allocated_capital - self.position.entry_fee)
        return equity

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        metrics = BacktestMetrics()

        if not self.closed_trades:
            return metrics

        metrics.total_trades = len(self.closed_trades)

        final_equity = self._calculate_equity_fast()
        metrics.total_return = final_equity - self.initial_capital
        metrics.total_return_pct = metrics.total_return / self.initial_capital

        winning = [t for t in self.closed_trades if t.pnl > 0]
        losing = [t for t in self.closed_trades if t.pnl <= 0]

        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        metrics.win_rate = len(winning) / len(self.closed_trades)

        metrics.avg_win = float(np.mean([t.pnl for t in winning])) if winning else 0
        metrics.avg_loss = float(np.mean([t.pnl for t in losing])) if losing else 0

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        pnls = [t.pnl for t in self.closed_trades]
        metrics.expectancy = float(np.mean(pnls))

        durations = [t.duration_seconds for t in self.closed_trades]
        metrics.avg_trade_duration_seconds = float(np.mean(durations))

        max_consec = 0
        current_consec = 0
        for trade in self.closed_trades:
            if trade.pnl <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
        metrics.max_consecutive_losses = max_consec

        if len(self.equity_curve) > 1:
            timestamps_eq = [e[0] for e in self.equity_curve]
            equities = [e[1] for e in self.equity_curve]

            peak = equities[0]
            max_dd = 0.0
            for eq in equities:
                peak = max(peak, eq)
                dd = (peak - eq) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            metrics.max_drawdown_pct = max_dd
            metrics.max_drawdown = max_dd * self.initial_capital

            equity_series = pd.Series(
                equities,
                index=pd.DatetimeIndex(timestamps_eq),
            )
            daily_equity = equity_series.resample('D').last().dropna()

            if len(daily_equity) > 2:
                daily_returns = daily_equity.pct_change().dropna().values

                if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                    metrics.sharpe_ratio = float(
                        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                    )

                    downside = daily_returns[daily_returns < 0]
                    if len(downside) > 0:
                        downside_std = np.std(downside)
                        if downside_std > 0:
                            metrics.sortino_ratio = float(
                                np.mean(daily_returns) / downside_std * np.sqrt(252)
                            )

            if metrics.max_drawdown_pct > 0:
                metrics.calmar_ratio = metrics.total_return_pct / metrics.max_drawdown_pct

            total_seconds = (timestamps_eq[-1] - timestamps_eq[0]).total_seconds()
            total_days = total_seconds / 86400.0
            if total_days > 0:
                metrics.trades_per_day = metrics.total_trades / total_days
                metrics.annualized_return = metrics.total_return_pct * (365.0 / total_days)

        metrics.signals_rejected_by_risk = self._signals_rejected
        metrics.signals_reduced_by_risk = self._signals_reduced
        metrics.halts_triggered = self._halts
        metrics.suspicious_pnl_rejected = self._suspicious_rejected
        metrics.signals_rejected_by_fee_filter = self._signals_rejected_by_fee_filter

        return metrics


# ======================================================================
# Walk-Forward Validator — Fixed for Parallel Safety
# ======================================================================

class WalkForwardValidator:
    """
    Walk-forward validation with proper isolation for parallel optimization.
    """

    def __init__(
        self,
        train_days: int = 60,
        test_days: int = 14,
        step_days: int = 7,
        min_oos_trades: int = 10,  # NEW: filter out useless folds
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.min_oos_trades = min_oos_trades

    def generate_splits(
        self,
        data: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits"""
        data = data.sort_values('timestamp').reset_index(drop=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        start_date = data['timestamp'].min()
        end_date = data['timestamp'].max()

        splits = []
        current_start = start_date

        while True:
            train_end = current_start + timedelta(days=self.train_days)
            test_end = train_end + timedelta(days=self.test_days)

            if test_end > end_date:
                break

            train_data = data[
                (data['timestamp'] >= current_start) &
                (data['timestamp'] < train_end)
            ]

            test_data = data[
                (data['timestamp'] >= train_end) &
                (data['timestamp'] < test_end)
            ]

            if len(train_data) > 100 and len(test_data) > 10:
                splits.append((train_data, test_data))

            current_start += timedelta(days=self.step_days)

        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits

    def validate(
        self,
        data: pd.DataFrame,
        strategy: StrategyDefinition,
        optimizer,
        n_trials_per_fold: int = 50,
        feature_config: Optional[FeatureConfig] = None,  # NEW: pass config
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation with proper trial isolation.

        FIX: Each trial creates its own BacktestEngine to avoid race conditions
        when n_jobs > 1 in optimization.
        """
        splits = self.generate_splits(data)

        results: Dict[str, Any] = {
            "folds": [],
            "oos_metrics": [],
            "params_stability": {},
            "skipped_folds": 0,  # NEW: track filtered folds
        }

        all_params: List[Dict] = []
        valid_oos_sharpes: List[float] = []  # NEW: for proper averaging

        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Processing fold {i+1}/{len(splits)}")

            # FIX: Create engine INSIDE backtest_fn for thread safety
            # When n_jobs > 1, multiple trials run in parallel
            def train_backtest(params, _train_data=train_data, _fc=feature_config):
                # Each trial gets its own engine instance
                engine = BacktestEngine(feature_config=_fc)
                metrics = engine.run(_train_data, strategy, params)
                return {
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "profit_factor": metrics.profit_factor,
                    "win_rate": metrics.win_rate,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "total_trades": metrics.total_trades,
                }

            # Optimize with n_jobs=1 for walk-forward (safer, clearer logs)
            # The inner optimization can still be fast with TPE
            opt_result = optimizer.optimize(
                train_backtest, 
                n_trials=n_trials_per_fold,
                n_jobs=1  # FIX: Force sequential for fold stability
            )
            best_params = opt_result.best_params
            all_params.append(best_params)

            # OOS test with fresh engine
            oos_engine = BacktestEngine(feature_config=feature_config)
            oos_metrics = oos_engine.run(test_data, strategy, best_params)

            # FIX: Filter out folds with insufficient trades
            if oos_metrics.total_trades < self.min_oos_trades:
                logger.warning(
                    f"Fold {i+1} skipped: OOS trades ({oos_metrics.total_trades}) "
                    f"< min ({self.min_oos_trades})"
                )
                results["skipped_folds"] += 1
                continue

            fold_result = {
                "fold": i,
                "train_score": opt_result.best_score,
                "oos_sharpe": oos_metrics.sharpe_ratio,
                "oos_return": oos_metrics.total_return_pct,
                "oos_win_rate": oos_metrics.win_rate,
                "oos_trades": oos_metrics.total_trades,
                "oos_max_dd": oos_metrics.max_drawdown_pct,
                "best_params": best_params,
            }

            results["folds"].append(fold_result)
            results["oos_metrics"].append(oos_metrics)
            valid_oos_sharpes.append(oos_metrics.sharpe_ratio)

            logger.info(
                f"Fold {i+1}: Train={opt_result.best_score:.2f}, "
                f"OOS Sharpe={oos_metrics.sharpe_ratio:.2f}, "
                f"OOS Trades={oos_metrics.total_trades}"
            )

        # Parameter stability
        if all_params:
            param_names = list(all_params[0].keys())
            for param in param_names:
                values = [p.get(param, 0) for p in all_params]
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                results["params_stability"][param] = {
                    "mean": mean_val,
                    "std": std_val,
                    "cv": std_val / (abs(mean_val) + 1e-9),
                }

        # FIX: Use only valid folds for summary
        if valid_oos_sharpes:
            results["summary"] = {
                "avg_oos_sharpe": float(np.mean(valid_oos_sharpes)),
                "std_oos_sharpe": float(np.std(valid_oos_sharpes)),
                "min_oos_sharpe": float(np.min(valid_oos_sharpes)),
                "max_oos_sharpe": float(np.max(valid_oos_sharpes)),
                "pct_profitable_folds": sum(1 for s in valid_oos_sharpes if s > 0) / len(valid_oos_sharpes),
                "total_folds": len(splits),
                "valid_folds": len(valid_oos_sharpes),
                "skipped_folds": results["skipped_folds"],
            }
        else:
            results["summary"] = {
                "avg_oos_sharpe": 0.0,
                "std_oos_sharpe": 0.0,
                "pct_profitable_folds": 0.0,
                "total_folds": len(splits),
                "valid_folds": 0,
                "skipped_folds": results["skipped_folds"],
                "error": "No valid folds (all had insufficient OOS trades)",
            }

        logger.info(f"Walk-Forward Summary:")
        logger.info(f"  Valid folds: {results['summary']['valid_folds']}/{results['summary']['total_folds']}")
        logger.info(f"  Avg OOS Sharpe: {results['summary']['avg_oos_sharpe']:.2f}")
        logger.info(f"  Profitable folds: {results['summary']['pct_profitable_folds']*100:.1f}%")

        return results