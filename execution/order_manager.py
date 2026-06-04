from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum, auto
import asyncio
import uuid

from loguru import logger

from core.fee_aware_filter import LiveFeeAwareFilter


class OrderStatus(Enum):
    PENDING = auto()
    SUBMITTED = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


@dataclass
class Order:
    id: str
    symbol: str
    side: str
    order_type: OrderType
    size: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None
    parent_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None


@dataclass
class Fill:
    order_id: str
    fill_id: str
    price: float
    size: float
    fee: float
    timestamp: datetime


class OrderManager:
    def __init__(self, exchange_client=None):
        self.exchange = exchange_client
        self.orders: Dict[str, Order] = {}
        self.pending_orders: List[str] = []
        self.active_orders: List[str] = []
        self.fills: List[Fill] = []
        self.on_fill: Optional[Callable[[Fill], None]] = None
        self.on_order_update: Optional[Callable[[Order], None]] = None
        self.fee_filter = LiveFeeAwareFilter(
            maker_fee_pct=0.0002,
            taker_fee_pct=0.0005,
            expected_spread_pct=0.0001,
            min_profit_target_pct=0.0002
        )

    def create_market_order(self, symbol: str, side: str, size: float) -> Order:
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol, side=side,
            order_type=OrderType.MARKET, size=size
        )
        self.orders[order.id] = order
        self.pending_orders.append(order.id)
        return order

    def create_limit_order(self, symbol: str, side: str, size: float, price: float) -> Order:
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol, side=side,
            order_type=OrderType.LIMIT, size=size, price=price
        )
        self.orders[order.id] = order
        self.pending_orders.append(order.id)
        return order

    def create_bracket_order(self, symbol: str, side: str, size: float,
                              entry_price: Optional[float], stop_loss: float,
                              take_profit: float,
                              entry_type: OrderType = OrderType.MARKET) -> tuple:
        entry_order = Order(
            id=str(uuid.uuid4()), symbol=symbol, side=side,
            order_type=entry_type, size=size,
            price=entry_price if entry_type == OrderType.LIMIT else None
        )
        sl_side = "sell" if side == "buy" else "buy"
        sl_order = Order(
            id=str(uuid.uuid4()), symbol=symbol, side=sl_side,
            order_type=OrderType.STOP, size=size, stop_price=stop_loss,
            parent_order_id=entry_order.id
        )
        tp_order = Order(
            id=str(uuid.uuid4()), symbol=symbol, side=sl_side,
            order_type=OrderType.LIMIT, size=size, price=take_profit,
            parent_order_id=entry_order.id
        )
        entry_order.stop_loss_order_id = sl_order.id
        entry_order.take_profit_order_id = tp_order.id
        self.orders[entry_order.id] = entry_order
        self.orders[sl_order.id] = sl_order
        self.orders[tp_order.id] = tp_order
        self.pending_orders.append(entry_order.id)
        return (entry_order, sl_order, tp_order)

    def validate_signal_with_fee_filter(
        self, signal, predicted_move_pct: float, confidence: float,
        current_bid: float, current_ask: float, last_price: float
    ) -> dict:
        should_trade, reason = self.fee_filter.should_trade_with_spread(
            signal, predicted_move_pct, confidence,
            current_bid, current_ask, last_price
        )
        if not should_trade:
            logger.warning(f"[OrderManager] Fee filter REJECTED: {reason}")
            return {'status': 'REJECTED', 'reason': reason}
        logger.debug(f"[OrderManager] Fee filter APPROVED: {reason}")
        return {'status': 'APPROVED', 'reason': reason}

    async def submit_order(self, order: Order) -> bool:
        if not self.exchange:
            logger.warning("No exchange client configured - simulating submission")
            order.status = OrderStatus.SUBMITTED
            order.exchange_order_id = f"SIM_{order.id}"
            return True
        try:
            if order.order_type == OrderType.MARKET:
                result = await self.exchange.create_market_order(order.symbol, order.side, order.size)
            elif order.order_type == OrderType.LIMIT:
                result = await self.exchange.create_limit_order(order.symbol, order.side, order.size, order.price)
            elif order.order_type == OrderType.STOP:
                result = await self.exchange.create_stop_order(order.symbol, order.side, order.size, order.stop_price)
            else:
                logger.error(f"Unsupported order type: {order.order_type}")
                return False
            order.exchange_order_id = result.get('id')
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            if order.id in self.pending_orders:
                self.pending_orders.remove(order.id)
            self.active_orders.append(order.id)
            logger.info(f"Order submitted: {order.id} -> {order.exchange_order_id}")
            if self.on_order_update:
                self.on_order_update(order)
            return True
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            logger.error(f"Order submission failed: {e}")
            return False

    async def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.orders:
            return False
        order = self.orders[order_id]
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            return False
        if self.exchange and order.exchange_order_id:
            try:
                await self.exchange.cancel_order(order.exchange_order_id, order.symbol)
            except Exception as e:
                logger.error(f"Failed to cancel order on exchange: {e}")
                return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        if order.id in self.active_orders:
            self.active_orders.remove(order.id)
        if self.on_order_update:
            self.on_order_update(order)
        return True

    def process_fill(self, order_id: str, price: float, size: float, fee: float = 0.0):
        if order_id not in self.orders:
            return
        order = self.orders[order_id]
        fill = Fill(
            order_id=order_id, fill_id=str(uuid.uuid4()),
            price=price, size=size, fee=fee,
            timestamp=datetime.now()
        )
        self.fills.append(fill)
        old_filled = order.filled_size
        order.filled_size += size
        if order.filled_size > 0:
            order.avg_fill_price = (
                (old_filled * order.avg_fill_price + size * price) / order.filled_size
            )
        order.updated_at = datetime.now()
        if order.filled_size >= order.size:
            order.status = OrderStatus.FILLED
            if order.id in self.active_orders:
                self.active_orders.remove(order.id)
            self._activate_bracket_orders(order)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        if self.on_fill:
            self.on_fill(fill)
        if self.on_order_update:
            self.on_order_update(order)

    def _activate_bracket_orders(self, entry_order: Order):
        if entry_order.stop_loss_order_id:
            sl_order = self.orders.get(entry_order.stop_loss_order_id)
            if sl_order:
                self.pending_orders.append(sl_order.id)
        if entry_order.take_profit_order_id:
            tp_order = self.orders.get(entry_order.take_profit_order_id)
            if tp_order:
                self.pending_orders.append(tp_order.id)

    def cancel_bracket_orders(self, entry_order_id: str):
        entry_order = self.orders.get(entry_order_id)
        if not entry_order:
            return
        if entry_order.stop_loss_order_id:
            asyncio.create_task(self.cancel_order(entry_order.stop_loss_order_id))
        if entry_order.take_profit_order_id:
            asyncio.create_task(self.cancel_order(entry_order.take_profit_order_id))

    def get_open_orders(self) -> List[Order]:
        return [
            self.orders[oid] for oid in self.active_orders
            if self.orders[oid].status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)
