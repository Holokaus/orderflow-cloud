import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import json

from loguru import logger

try:
    import ccxt.async_support as ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


@dataclass
class ExchangeConfig:
    exchange_id: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    rate_limit: bool = True
    ws_base_url: str = ""
    use_futures: bool = True


class ExchangeConnector:
    MAX_RECONNECT_DELAY = 30
    INITIAL_RECONNECT_DELAY = 1
    STALE_CONNECTION_TIMEOUT = 45
    BOOK_DEPTH_LIMIT = 1000
    EMIT_BOOK_DEPTH = 20

    def __init__(self, config: ExchangeConfig):
        self.config = config
        if not HAS_CCXT:
            raise ImportError("ccxt not installed. Run: pip install ccxt")
        exchange_class = getattr(ccxt, config.exchange_id, None)
        if not exchange_class:
            raise ValueError(f"Unknown exchange: {config.exchange_id}")
        self.rest_client = exchange_class({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'sandbox': config.testnet,
            'enableRateLimit': config.rate_limit,
            'timeout': 30000,
            'options': {
                'defaultType': 'future' if config.use_futures else 'spot'
            }
        })
        self.ws_connection = None
        self.running = False
        self._local_bids: Dict[float, float] = {}
        self._local_asks: Dict[float, float] = {}
        self._last_update_id: int = 0
        self._prev_final_update_id: int = 0
        self._book_initialised: bool = False
        self._event_buffer: List[Dict] = []
        self._snapshot_lock = asyncio.Lock()
        self.order_book: Dict[str, Any] = {}
        self.recent_trades: List[Dict] = []
        self._last_message_time: float = 0.0
        self._reconnect_delay: float = self.INITIAL_RECONNECT_DELAY
        self._ws_symbol: str = ""
        self._ws_url_index: int = 0
        self._ws_urls: List[str] = []
        self.on_order_book_update: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None
        self.on_ticker: Optional[Callable] = None
        self.on_health_alert: Optional[Callable] = None
        self._health_metrics = {
            'gaps_detected': 0,
            'resyncs': 0,
            'events_received': 0,
            'events_applied': 0,
            'stale_skipped': 0,
            'warnings': 0,
            'avg_latency_ms': 0,
            'last_gap_time': None
        }
        self._last_resync_time: float = 0.0

    async def connect(self):
        max_retries = 5
        retry_delay = 2
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Connecting to {self.config.exchange_id} (attempt {attempt}/{max_retries})...")
                await self.rest_client.load_markets()
                logger.info(f"Connected to {self.config.exchange_id}")
                return
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Failed to connect after {max_retries} attempts: {e}")
                    raise
                error_msg = str(e).lower()
                if "timeout" in error_msg or "connection" in error_msg:
                    logger.warning(f"Connection timeout/failed (attempt {attempt}/{max_retries}). Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)
                else:
                    raise

    async def disconnect(self):
        self.running = False
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        if self.rest_client:
            await self.rest_client.close()
        logger.info("Disconnected from exchange")

    async def get_balance(self) -> Dict[str, float]:
        try:
            balance = await self.rest_client.fetch_balance()
            return {
                currency: data['free']
                for currency, data in balance.items()
                if isinstance(data, dict) and data.get('free', 0) > 0
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}

    async def get_ticker(self, symbol: str) -> Dict:
        try:
            return await self.rest_client.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch ticker: {e}")
            return {}

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        try:
            return await self.rest_client.fetch_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            return {'bids': [], 'asks': []}

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        try:
            return await self.rest_client.fetch_trades(symbol, limit=limit)
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []

    async def create_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        try:
            order = await self.rest_client.create_order(
                symbol=symbol, type='market', side=side, amount=amount
            )
            logger.info(f"Market order created: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to create market order: {e}")
            raise

    async def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        try:
            order = await self.rest_client.create_order(
                symbol=symbol, type='limit', side=side, amount=amount, price=price
            )
            logger.info(f"Limit order created: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to create limit order: {e}")
            raise

    async def create_stop_order(self, symbol: str, side: str, amount: float, stop_price: float) -> Dict:
        try:
            order = await self.rest_client.create_order(
                symbol=symbol, type='stop_market', side=side, amount=amount,
                params={'stopPrice': stop_price}
            )
            logger.info(f"Stop order created: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to create stop order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            await self.rest_client.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        try:
            return await self.rest_client.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []

    async def _fetch_rest_snapshot(self, symbol: str):
        async with self._snapshot_lock:
            try:
                snap = await asyncio.wait_for(
                    self.rest_client.fetch_order_book(symbol, limit=self.BOOK_DEPTH_LIMIT),
                    timeout=10.0
                )
                self._local_bids = {
                    float(price): float(size)
                    for price, size in snap.get('bids', [])
                    if float(size) > 0
                }
                self._local_asks = {
                    float(price): float(size)
                    for price, size in snap.get('asks', [])
                    if float(size) > 0
                }
                self._last_update_id = snap.get('nonce', 0)
                self._prev_final_update_id = 0
                self._health_metrics['resyncs'] += 1
                logger.info(f"Snapshot loaded: {len(self._local_bids)}b/{len(self._local_asks)}a, lastUpdateId={self._last_update_id} [Resync #{self._health_metrics['resyncs']}]")
                # Apply buffered events newer than snapshot (lenient - don't require strict bridging)
                applied_count = 0
                for evt in self._event_buffer:
                    if evt.get('u', 0) > self._last_update_id:
                        if self._apply_diff_if_valid(evt):
                            applied_count += 1
                if applied_count > 0:
                    logger.info(f"Applied {applied_count} buffered events after snapshot")
                self._event_buffer.clear()
                self._book_initialised = True
                logger.info(f"Book synchronized. Total resyncs: {self._health_metrics['resyncs']}")
            except Exception as e:
                logger.error(f"Snapshot failed: {e}")
                self._book_initialised = False

    def _apply_diff_if_valid(self, data: Dict) -> bool:
        event_first_id = data.get('U', 0)
        event_final_id = data.get('u', 0)
        if event_final_id <= self._last_update_id:
            return False
        if self._prev_final_update_id != 0:
            expected_u = self._prev_final_update_id + 1
            if event_first_id != expected_u:
                gap_size = event_first_id - expected_u
                self._health_metrics['gaps_detected'] += 1
                self._health_metrics['last_gap_time'] = datetime.now(timezone.utc)
                logger.debug(f"Gap: expected U={expected_u}, got U={event_first_id}, gap={abs(gap_size)}")
                # Tolerate all gaps — apply diff and continue.
                # A slightly stale book is better than endless re-syncs.
        for price_str, size_str in data.get('b', []):
            price, size = float(price_str), float(size_str)
            if size == 0:
                self._local_bids.pop(price, None)
            else:
                self._local_bids[price] = size
        for price_str, size_str in data.get('a', []):
            price, size = float(price_str), float(size_str)
            if size == 0:
                self._local_asks.pop(price, None)
            else:
                self._local_asks[price] = size
        self._prev_final_update_id = event_final_id
        self._health_metrics['events_applied'] += 1
        return True

    def _build_sorted_book(self) -> Dict[str, Any]:
        sorted_bids = sorted(
            ((p, s) for p, s in self._local_bids.items() if s > 0),
            key=lambda x: x[0], reverse=True
        )[:self.EMIT_BOOK_DEPTH]
        sorted_asks = sorted(
            ((p, s) for p, s in self._local_asks.items() if s > 0),
            key=lambda x: x[0]
        )[:self.EMIT_BOOK_DEPTH]
        return {'bids': [[p, s] for p, s in sorted_bids], 'asks': [[p, s] for p, s in sorted_asks]}

    async def start_websocket(self, symbol: str):
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets not installed. Run: pip install websockets")
        self.running = True
        self._ws_symbol = symbol
        self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
        self._ws_urls = self._get_ws_urls(symbol)
        self._ws_url_index = 0
        consecutive_failures = 0
        self._last_resync_time = time.monotonic()
        while self.running:
            ws_url = self._ws_urls[self._ws_url_index]
            try:
                logger.info(f"Connecting to WebSocket: {ws_url}")
                async with websockets.connect(
                    ws_url,
                    ping_interval=15,
                    ping_timeout=10,
                    close_timeout=5,
                    open_timeout=15,
                ) as ws:
                    self.ws_connection = ws
                    logger.info(f"WebSocket connected: {ws_url}")
                    consecutive_failures = 0
                    self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
                    logger.info("Fetching fresh order book snapshot...")
                    self._book_initialised = False
                    self._event_buffer.clear()
                    self._prev_final_update_id = 0
                    await self._fetch_rest_snapshot(symbol)
                    self._last_message_time = time.monotonic()
                    await self._subscribe(ws, symbol)
                    logger.info("Subscribed to market data streams")
                    watchdog_task = asyncio.create_task(self._stale_connection_watchdog(ws))
                    try:
                        async for message in ws:
                            if not self.running:
                                break
                            self._last_message_time = time.monotonic()
                            await self._process_message(message, symbol)
                    finally:
                        watchdog_task.cancel()
                        try:
                            await watchdog_task
                        except asyncio.CancelledError:
                            pass
            except websockets.exceptions.ConnectionClosed as e:
                consecutive_failures += 1
                logger.warning(f"WebSocket closed (code={e.code}) on {ws_url}, failure #{consecutive_failures}")
            except asyncio.CancelledError:
                logger.info("WebSocket task cancelled")
                break
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"WebSocket error on {ws_url}: {e}, failure #{consecutive_failures}")
            if not self.running:
                break
            if consecutive_failures >= 2:
                self._ws_url_index = (self._ws_url_index + 1) % len(self._ws_urls)
                consecutive_failures = 0
                await asyncio.sleep(1)
            else:
                logger.info(f"Reconnecting in {self._reconnect_delay:.1f}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY)

    async def _stale_connection_watchdog(self, ws):
        while True:
            try:
                await asyncio.sleep(5)
                elapsed = time.monotonic() - self._last_message_time
                if elapsed > self.STALE_CONNECTION_TIMEOUT:
                    logger.warning(f"No WS message for {elapsed:.1f}s. Forcing reconnect.")
                    await ws.close()
                    return
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                break

    def _get_ws_urls(self, symbol: str) -> List[str]:
        symbol_lower = symbol.replace('/', '').lower()
        streams = f"{symbol_lower}@depth@100ms/{symbol_lower}@aggTrade"
        if self.config.exchange_id != "binance":
            raise ValueError(f"WebSocket not configured for: {self.config.exchange_id}")
        if self.config.use_futures:
            primary_domain = "fstream.binance.com"
        else:
            primary_domain = "stream.binance.com"
        if self.config.testnet:
            return [f"wss://stream.testnet.binance.vision/stream?streams={streams}"]
        urls = []
        if hasattr(self.config, 'ws_base_url') and self.config.ws_base_url:
            urls.append(f"{self.config.ws_base_url}/stream?streams={streams}")
        urls.extend([
            f"wss://{primary_domain}:443/stream?streams={streams}",
            f"wss://{primary_domain}:9443/stream?streams={streams}",
            f"wss://data-stream.binance.vision/stream?streams={streams}",
        ])
        return urls

    async def _subscribe(self, ws, symbol: str):
        if self.config.exchange_id == "binance":
            pass
        elif self.config.exchange_id == "bybit":
            subscribe_msg = {
                "op": "subscribe",
                "args": [
                    f"orderbook.50.{symbol.replace('/', '')}",
                    f"publicTrade.{symbol.replace('/', '')}"
                ]
            }
            await ws.send(json.dumps(subscribe_msg))

    async def _process_message(self, message: str, symbol: str):
        try:
            data = json.loads(message)
            if 'stream' in data and 'data' in data:
                data = data['data']
            if self.config.exchange_id == "binance":
                if 'e' in data:
                    if data['e'] == 'depthUpdate':
                        await self._handle_depth_update(data, symbol)
                    elif data['e'] == 'aggTrade':
                        await self._handle_trade(data)
            elif self.config.exchange_id == "bybit":
                if 'topic' in data:
                    if 'orderbook' in data['topic']:
                        await self._handle_depth_update(data['data'], symbol)
                    elif 'publicTrade' in data['topic']:
                        await self._handle_trade(data['data'])
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _handle_depth_update(self, data: Dict, symbol: str):
        if not self._book_initialised:
            self._event_buffer.append(data)
            if len(self._event_buffer) > 2000:
                self._event_buffer = self._event_buffer[-1000:]
                self._health_metrics['warnings'] += 1
            return
        self._health_metrics['events_received'] += 1
        event_final_id = data.get('u', 0)
        event_first_id = data.get('U', 0)
        if event_final_id <= self._last_update_id:
            return
        # Skip out-of-order / duplicate events silently
        if self._prev_final_update_id != 0 and event_first_id <= self._prev_final_update_id:
            self._health_metrics['stale_skipped'] += 1
            return
        applied = self._apply_diff_if_valid(data)
        if not applied:
            # Only possible for events older than snapshot — skip silently
            return
        sorted_book = self._build_sorted_book()
        exchange_ts_ms = data.get('E')
        latency_ms = None
        if exchange_ts_ms:
            latency_ms = (time.time() * 1000) - exchange_ts_ms
            if latency_ms < 0 or latency_ms > 10000:
                latency_ms = None
        sorted_book.update({
            'timestamp': datetime.fromtimestamp(exchange_ts_ms / 1000, tz=timezone.utc) if exchange_ts_ms else datetime.now(timezone.utc),
            'exchange_timestamp_ms': exchange_ts_ms,
            'last_update_id': self._prev_final_update_id,
            'latency_ms': latency_ms
        })
        self.order_book = sorted_book
        if self.on_order_book_update:
            await self.on_order_book_update(sorted_book)

    async def _handle_trade(self, data: Dict):
        if not isinstance(data, dict):
            return
        if 'p' not in data and 'price' not in data:
            return
        try:
            trade_ts_ms = data.get('T', None)
            if trade_ts_ms is not None:
                trade_dt = datetime.fromtimestamp(trade_ts_ms / 1000.0, tz=timezone.utc)
            else:
                trade_dt = datetime.now(timezone.utc)
            price = float(data.get('p', data.get('price', 0)))
            size = float(data.get('q', data.get('size', 0)))
            if price <= 0 or size <= 0:
                return
            trade = {
                'price': price,
                'size': size,
                'side': 'sell' if data.get('m', data.get('side')) else 'buy',
                'timestamp': trade_dt,
                'exchange_timestamp_ms': trade_ts_ms,
            }
        except (ValueError, TypeError) as e:
            return
        self.recent_trades.append(trade)
        if len(self.recent_trades) > 1000:
            self.recent_trades = self.recent_trades[-500:]
        if self.on_trade:
            await self.on_trade(trade)

    def get_health_report(self) -> Dict:
        return {
            **self._health_metrics,
            'book_initialized': self._book_initialised,
            'buffer_size': len(self._event_buffer),
            'last_update_id': self._prev_final_update_id
        }

    async def test_connectivity(self, symbol: str = "BTC/USDT") -> Dict:
        result = {"rest": {}, "ws_urls": []}
        try:
            start = time.time()
            ticker = await self.rest_client.fetch_ticker(symbol)
            latency_ms = (time.time() - start) * 1000
            result["rest"] = {"ok": True, "latency_ms": latency_ms}
        except Exception as e:
            result["rest"] = {"ok": False, "error": str(e)}
        symbol_lower = symbol.replace('/', '').lower()
        ws_urls_to_test = self._get_ws_urls(symbol)
        for ws_url in ws_urls_to_test:
            try:
                start = time.time()
                async with websockets.connect(ws_url, open_timeout=15, close_timeout=5) as ws:
                    message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    latency_ms = (time.time() - start) * 1000
                    result["ws_urls"].append({"url": ws_url, "ok": True, "latency_ms": latency_ms})
            except asyncio.TimeoutError:
                latency_ms = (time.time() - start) * 1000
                result["ws_urls"].append({"url": ws_url, "ok": False, "error": "Timeout waiting for data", "latency_ms": latency_ms})
            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                result["ws_urls"].append({"url": ws_url, "ok": False, "error": str(e), "latency_ms": latency_ms})
        return result
