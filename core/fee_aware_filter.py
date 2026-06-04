from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from core.data_structures import Signal


@dataclass
class FeeConfig:
    maker_fee_pct: float = 0.0002
    taker_fee_pct: float = 0.0005
    expected_spread_pct: float = 0.0001
    min_profit_target_pct: float = 0.0002


class FeeAwareFilter:
    def __init__(
        self,
        maker_fee_pct: float = 0.0002,
        taker_fee_pct: float = 0.0005,
        expected_spread_pct: float = 0.0001,
        min_profit_target_pct: float = 0.0002,
    ):
        self.entry_fee = maker_fee_pct
        self.exit_fee = taker_fee_pct
        self.expected_spread = expected_spread_pct
        self.min_profit = min_profit_target_pct
        self.total_cost = (
            self.entry_fee +
            self.exit_fee +
            self.expected_spread +
            self.min_profit
        )

    def should_ignore_signal(
        self,
        signal: Signal,
        predicted_price_move_pct: float,
        confidence: float = 0.7
    ) -> Tuple[bool, str]:
        confidence = max(0.3, min(confidence, 1.0))
        confidence_adjusted_threshold = self.total_cost / confidence
        if predicted_price_move_pct < confidence_adjusted_threshold:
            reason = (
                f"Predicted move {predicted_price_move_pct:.4%} < "
                f"threshold {confidence_adjusted_threshold:.4%} "
                f"(base_cost={self.total_cost:.4%}, confidence={confidence:.2f}). "
                f"Signal would likely lose money after fees."
            )
            return True, reason
        reason = (
            f"Signal passes fee-aware filter. "
            f"Predicted move {predicted_price_move_pct:.4%} >= "
            f"threshold {confidence_adjusted_threshold:.4%}"
        )
        return False, reason

    def should_reject_by_spread(
        self,
        actual_spread_pct: float
    ) -> Tuple[bool, str]:
        spread_multiplier = actual_spread_pct / self.expected_spread
        if spread_multiplier > 2.0:
            reason = (
                f"Spread {actual_spread_pct:.4%} is {spread_multiplier:.1f}x "
                f"expected {self.expected_spread:.4%}. Likely illiquidity. Reject."
            )
            return True, reason
        return False, ""


class LiveFeeAwareFilter(FeeAwareFilter):
    def should_trade_with_spread(
        self,
        signal: Signal,
        predicted_price_move_pct: float,
        confidence: float,
        current_bid: float,
        current_ask: float,
        last_price: float
    ) -> Tuple[bool, str]:
        actual_spread = current_ask - current_bid
        actual_spread_pct = actual_spread / last_price if last_price > 0 else 0
        should_reject, spread_reason = self.should_reject_by_spread(actual_spread_pct)
        if should_reject:
            return False, spread_reason
        should_ignore, fee_reason = self.should_ignore_signal(
            signal,
            predicted_price_move_pct,
            confidence
        )
        if should_ignore:
            return False, fee_reason
        return True, (
            f"Trade approved. Spread={actual_spread_pct:.4%}, "
            f"predicted_move={predicted_price_move_pct:.4%}, "
            f"confidence={confidence:.2f}"
        )
