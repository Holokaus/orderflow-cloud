from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
import numpy as np
from loguru import logger

from core.data_structures import (
    OrderFlowState, Signal, SignalType, Side, Regime
)


class StrategyCategory(Enum):
    ABSORPTION = auto()
    MOMENTUM = auto()
    REVERSAL = auto()
    BREAKOUT = auto()
    MEAN_REVERSION = auto()


@dataclass
class StrategyCondition:
    feature: str
    operator: str
    threshold: float
    threshold_high: Optional[float] = None
    weight: float = 1.0
    required: bool = False
    param_key: Optional[str] = None

    def validate(self) -> bool:
        if self.operator == "between":
            if self.threshold_high is None:
                raise ValueError(f"[{self.feature}] 'between' operator requires threshold_high")
            if self.threshold >= self.threshold_high:
                raise ValueError(f"[{self.feature}] threshold ({self.threshold}) must be < threshold_high ({self.threshold_high})")
        return True

    def evaluate(self, features: Dict[str, float]) -> tuple:
        if self.feature not in features:
            return (False, 0.0)
        value = features[self.feature]
        if self.operator == ">":
            satisfied = value > self.threshold
        elif self.operator == "<":
            satisfied = value < self.threshold
        elif self.operator == ">=":
            satisfied = value >= self.threshold
        elif self.operator == "<=":
            satisfied = value <= self.threshold
        elif self.operator == "==":
            satisfied = abs(value - self.threshold) < 1e-9
        elif self.operator == "between":
            satisfied = self.threshold <= value <= self.threshold_high
        else:
            satisfied = False
        if satisfied and self.operator in [">", ">="]:
            score = min((value - self.threshold) / (abs(self.threshold) + 1e-9), 2.0)
        elif satisfied and self.operator in ["<", "<="]:
            score = min((self.threshold - value) / (abs(self.threshold) + 1e-9), 2.0)
        elif satisfied:
            score = 1.0
        else:
            score = 0.0
        return (satisfied, score * self.weight)


@dataclass
class StrategyDefinition:
    name: str
    category: StrategyCategory
    description: str
    entry_conditions: List[StrategyCondition] = field(default_factory=list)
    min_conditions_satisfied: int = 3
    min_score_threshold: float = 2.0
    stop_loss_atr_mult: float = 2.5
    take_profit_atr_mult: float = 6.0
    max_holding_seconds: int = 86400
    trailing_stop_activation_pct: float = 0.005
    sl_mult_high_vol: float = 3.5
    sl_mult_low_vol: float = 1.8
    sl_mult_trending: float = 2.5
    tp_mult_high_vol: float = 7.0
    tp_mult_low_vol: float = 2.5
    tp_mult_trending: float = 5.0
    filters: List[StrategyCondition] = field(default_factory=list)
    allowed_regimes: List[Regime] = field(default_factory=lambda: list(Regime))
    ml_ensemble: Optional[object] = None
    ml_min_confidence: float = 0.7
    base_position_pct: float = 0.1
    max_position_pct: float = 0.25
    scale_with_score: bool = True

    def evaluate(self, state: OrderFlowState) -> Optional[Signal]:
        features = state.features
        if state.regime == Regime.LOW_LIQUIDITY:
            return None
        for cond in self.entry_conditions + self.filters:
            cond.validate()
        for i, filter_cond in enumerate(self.filters):
            satisfied, _ = filter_cond.evaluate(features)
            if satisfied:
                return None
        if state.regime not in self.allowed_regimes:
            return None
        satisfied_count = 0
        total_score = 0.0
        required_satisfied = True
        reasons = []
        for condition in self.entry_conditions:
            satisfied, score = condition.evaluate(features)
            if satisfied:
                satisfied_count += 1
                total_score += score
                reasons.append(f"{condition.feature} {condition.operator} {condition.threshold:.4f}")
            else:
                if condition.required:
                    required_satisfied = False
        if not required_satisfied:
            return None
        if satisfied_count < self.min_conditions_satisfied:
            return None
        if total_score < self.min_score_threshold:
            return None
        direction = self._determine_direction(state, total_score)
        if direction == SignalType.NEUTRAL:
            return None
        regime = state.regime
        if regime == Regime.HIGH_VOLATILITY:
            sl_mult, tp_mult = self.sl_mult_high_vol, self.tp_mult_high_vol
        elif regime in (Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION):
            sl_mult, tp_mult = self.sl_mult_low_vol, self.tp_mult_low_vol
        else:
            sl_mult, tp_mult = self.sl_mult_trending, self.tp_mult_trending
        atr_pct = self._estimate_atr(state)
        mid_price = state.order_book.mid_price
        stop_dist = mid_price * atr_pct * sl_mult
        tp_dist = mid_price * atr_pct * tp_mult
        min_tp_dist = mid_price * 0.0037
        if tp_dist < min_tp_dist:
            tp_dist = min_tp_dist
        if direction in (SignalType.BUY, SignalType.STRONG_BUY):
            stop_loss = mid_price - stop_dist
            take_profit = mid_price + tp_dist
        else:
            stop_loss = mid_price + stop_dist
            take_profit = mid_price - tp_dist
        position_pct = self.base_position_pct
        if self.scale_with_score:
            position_pct = min(self.base_position_pct * min(total_score / self.min_score_threshold, 2.0),
                             self.max_position_pct)
        confidence = min(total_score / (self.min_score_threshold * 2), 1.0)
        return Signal(
            timestamp=state.timestamp,
            signal_type=direction,
            confidence=confidence,
            entry_price=mid_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_pct,
            primary_reason=reasons[0] if reasons else self.name,
            supporting_factors=reasons[1:5],
            risk_reward_ratio=abs(tp_dist) / abs(stop_dist)
        )

    def _determine_direction(self, state: OrderFlowState, score: float) -> SignalType:
        features = state.features
        delta_pct = features.get("delta_pct_60s", 0)
        imbalance = features.get("depth_imbalance_10", 0)
        pressure = features.get("net_pressure", 0)
        abs_imbalance = features.get("abs_depth_imbalance_10", 0)
        abs_delta_pct = features.get("abs_delta_pct_60s", 0)
        directional_score = 0
        if abs(imbalance) > 0.1:
            directional_score += 1 if imbalance > 0 else -1
        if abs(imbalance) > 0.3:
            directional_score += 1 if imbalance > 0 else -1
        if abs_imbalance > 0.3:
            directional_score += 1 if imbalance > 0 else -1
        if abs(delta_pct) > 0.1:
            directional_score += 1 if delta_pct > 0 else -1
        if abs_delta_pct > 0.3:
            directional_score += 1 if delta_pct > 0 else -1
        if abs(pressure) > 50000:
            directional_score += 1 if pressure > 0 else -1
        if directional_score >= 2:
            return SignalType.STRONG_BUY if score > self.min_score_threshold * 1.5 else SignalType.BUY
        elif directional_score >= 0:
            return SignalType.BUY
        else:
            return SignalType.NEUTRAL

    def _estimate_atr(self, state: OrderFlowState, default: float = 100.0) -> float:
        mid_price = state.order_book.mid_price
        if mid_price <= 0:
            return default
        features = state.features
        if features.get("atr_60s", 0) > 0:
            atr_dollar = features["atr_60s"]
        elif features.get("price_range_60s", 0) > 0:
            atr_dollar = features["price_range_60s"]
        else:
            atr_dollar = mid_price * 0.001
        atr_pct = atr_dollar / mid_price
        return max(atr_pct, 0.001)


def create_absorption_strategy() -> StrategyDefinition:
    return StrategyDefinition(
        name="Absorption",
        category=StrategyCategory.ABSORPTION,
        description="Trade after detecting absorption of aggressive orders",
        entry_conditions=[
            StrategyCondition(feature="recent_absorption_strength", operator=">=", threshold=0.10, weight=2.0, required=False, param_key="abs__entry_str_min"),
            StrategyCondition(feature="volume_acceleration", operator=">", threshold=1.0, weight=1.5, param_key="abs__entry_vol_min"),
            StrategyCondition(feature="price_change_pct_60s", operator="<", threshold=0.001, weight=1.0, param_key="abs__entry_chg60_max"),
            StrategyCondition(feature="abs_delta_60s", operator=">", threshold=0, weight=1.5, param_key="abs__entry_delta_min"),
            StrategyCondition(feature="abs_depth_imbalance_10", operator=">", threshold=0.05, weight=1.0, param_key="abs__entry_imbal_min"),
            StrategyCondition(feature="price_vs_poc_pct", operator="between", threshold=-0.005, threshold_high=0.005, weight=0.5, param_key="abs__entry_poc_range"),
            StrategyCondition(feature="book_trade_agreement", operator="==", threshold=1.0, weight=1.5, required=False, param_key="abs__entry_agreement"),
        ],
        filters=[
            StrategyCondition(feature="spread_bps", operator=">", threshold=15.0),
            StrategyCondition(feature="bid_depth_10", operator="<", threshold=5000.0),
            StrategyCondition(feature="ask_depth_10", operator="<", threshold=5000.0),
        ],
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        sl_mult_high_vol=7.0,
        sl_mult_low_vol=7.0,
        sl_mult_trending=7.0,
        tp_mult_high_vol=100.0,
        tp_mult_low_vol=100.0,
        tp_mult_trending=100.0,
        trailing_stop_activation_pct=0.01,
        allowed_regimes=[Regime.RANGING, Regime.ACCUMULATION, Regime.DISTRIBUTION,
                        Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.BREAKOUT,
                        Regime.HIGH_VOLATILITY, Regime.LOW_LIQUIDITY]
    )


def create_stacked_imbalance_strategy() -> StrategyDefinition:
    return StrategyDefinition(
        name="Stacked Imbalance",
        category=StrategyCategory.MOMENTUM,
        description="Trade momentum when multiple price levels show same-side imbalance",
        entry_conditions=[
            StrategyCondition(feature="footprint_imbalance_count", operator=">=", threshold=2, weight=2.0, required=True),
            StrategyCondition(feature="abs_delta_pct_60s", operator=">", threshold=0.2, weight=1.5),
            StrategyCondition(feature="abs_depth_imbalance_10", operator=">", threshold=0.15, weight=1.5),
            StrategyCondition(feature="pressure_confirmed", operator="==", threshold=1.0, weight=1.0),
            StrategyCondition(feature="price_vs_vwap_pct", operator="between", threshold=-0.003, threshold_high=0.003, weight=0.5),
        ],
        filters=[
            StrategyCondition(feature="spread_bps", operator=">", threshold=15.0),
            StrategyCondition(feature="bid_depth_10", operator="<", threshold=5000.0),
            StrategyCondition(feature="ask_depth_10", operator="<", threshold=5000.0),
            StrategyCondition(feature="price_change_pct_300s", operator=">", threshold=0.05),
            StrategyCondition(feature="price_change_pct_300s", operator="<", threshold=-0.05),
        ],
        min_conditions_satisfied=2,
        min_score_threshold=2.5,
        sl_mult_high_vol=7.0,
        sl_mult_low_vol=7.0,
        sl_mult_trending=7.0,
        tp_mult_high_vol=100.0,
        tp_mult_low_vol=100.0,
        tp_mult_trending=100.0,
        trailing_stop_activation_pct=0.01,
        base_position_pct=0.95,
        scale_with_score=False,
        max_position_pct=0.95,
    )


STRATEGY_LIBRARY = {
    "absorption": create_absorption_strategy,
    "stacked_imbalance": create_stacked_imbalance_strategy,
}


def get_strategy(name: str) -> Optional[StrategyDefinition]:
    if name in STRATEGY_LIBRARY:
        return STRATEGY_LIBRARY[name]()
    return None


def get_all_strategies() -> Dict[str, StrategyDefinition]:
    return {name: factory() for name, factory in STRATEGY_LIBRARY.items()}
