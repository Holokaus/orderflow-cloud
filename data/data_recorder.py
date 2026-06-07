import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Tuple
import json
import gzip
import time

import pandas as pd
from loguru import logger

from data.fee_aware_data_filter import HistoricalDataFilter, FilterConfig as DataFilterConfig


class DataRecorder:
    """
    Records order book and trade data to disk.
    Crash-safe append-only JSONL with proper write coordination.
    """

    def __init__(
        self,
        output_dir: str = "./data/recorded/",
        symbol: str = "BTCUSDT",
        compress: bool = True,
        flush_interval_seconds: int = 10,
        max_buffer_size: int = 100,
        gap_threshold_seconds: float = 5.0
    ):
        self.output_dir = Path(output_dir)
        self.symbol = symbol
        self.compress = compress
        self.flush_interval = flush_interval_seconds
        self.max_buffer_size = max_buffer_size
        self.gap_threshold_seconds = gap_threshold_seconds

        # Buffers - now with proper locking for thread safety
        self.order_book_buffer: List[Dict] = []
        self.trade_buffer: List[Dict] = []
        self._buffer_lock = asyncio.Lock()  # Protects buffer access
        self._flush_event = asyncio.Event()  # Signals flush needed
        self._flush_task: Optional[asyncio.Task] = None

        # State
        self.recording = False
        self.current_date: Optional[datetime] = None

        # Health metrics
        self._last_ob_record_time: Optional[float] = None
        self._last_trade_record_time: Optional[float] = None
        self._ob_gap_count: int = 0
        self._consecutive_gap_count: int = 0
        self._records_since_last_flush: int = 0
        self._total_records_written: int = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, data_type: str, date: datetime) -> Path:
        """Get file path for given date and data type."""
        date_str = date.strftime("%Y%m%d")
        ext = ".jsonl.gz" if self.compress else ".jsonl"
        return self.output_dir / f"{self.symbol}_{data_type}_{date_str}{ext}"

    async def start_recording(self):
        """Start recording with single dedicated flush loop."""
        self.recording = True
        self.current_date = datetime.now(timezone.utc)
        self._flush_event.clear()
        
        # Start SINGLE background flush task
        self._flush_task = asyncio.create_task(self._flush_loop())
        
        logger.info(f"Started recording {self.symbol} data to {self.output_dir}")

    async def stop_recording(self):
        """Stop recording gracefully."""
        self.recording = False
        self._flush_event.set()  # Signal flush loop to wake up
        
        if self._flush_task:
            try:
                await asyncio.wait_for(self._flush_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Flush task did not complete in time")
                self._flush_task.cancel()
        
        # Final flush any remaining data
        await self._flush_buffers()
        
        logger.info(
            f"Stopped recording. Total records written: {self._total_records_written}, "
            f"Gaps > {self.gap_threshold_seconds}s: {self._ob_gap_count}"
        )

    async def _flush_loop(self):
        """Single dedicated flush loop - prevents concurrent writes."""
        while self.recording:
            try:
                # Wait for either interval or explicit signal
                await asyncio.wait_for(
                    self._flush_event.wait(), 
                    timeout=self.flush_interval
                )
                self._flush_event.clear()
            except asyncio.TimeoutError:
                pass  # Normal interval flush
            
            if not self.recording and not self._has_data():
                break
                
            await self._flush_buffers()

    def _has_data(self) -> bool:
        """Check if buffers have data pending."""
        return bool(self.order_book_buffer or self.trade_buffer)

    # ==================== Record Methods ====================

    def record_order_book(self, order_book: Dict):
        """Thread-safe order book recording."""
        ts = order_book.get('timestamp', datetime.now(timezone.utc))
        ts_iso = ts.isoformat() if isinstance(ts, datetime) else ts
        
        exchange_ts_ms = order_book.get('exchange_timestamp_ms')
        
        # Gap detection logic...
        if exchange_ts_ms and self._last_ob_record_time:
            exchange_gap_ms = exchange_ts_ms - self._last_ob_record_time
            if exchange_gap_ms > (self.gap_threshold_seconds * 1000):
                self._ob_gap_count += 1
                logger.warning(f"Exchange gap: {exchange_gap_ms:.0f}ms")

        record = {
            'timestamp': ts_iso,
            'exchange_timestamp_ms': exchange_ts_ms,
            'bids': [[p, s] for p, s in order_book.get('bids', [])[:20] if s > 0],
            'asks': [[p, s] for p, s in order_book.get('asks', [])[:20] if s > 0],
            '_quality': {
                'latency_ms': order_book.get('latency_ms'),
                'last_update_id': order_book.get('last_update_id')
            }
        }
        
        # Use call_soon_threadsafe if called from different thread, else direct
        asyncio.create_task(self._add_to_buffer('ob', record))
        self._last_ob_record_time = exchange_ts_ms

    def record_trade(self, trade: Dict):
        """Thread-safe trade recording."""
        price = trade.get('price', 0)
        size = trade.get('size', 0)
        
        if price <= 0 or size <= 0:
            logger.debug(f"Skipping invalid trade: price={price}, size={size}")
            return
        
        ts = trade.get('timestamp')
        if isinstance(ts, datetime):
            ts_iso = ts.isoformat()
        elif isinstance(ts, str):
            ts_iso = ts
        else:
            ts_iso = datetime.now(timezone.utc).isoformat()

        record = {
            'timestamp': ts_iso,
            'exchange_timestamp_ms': trade.get('exchange_timestamp_ms'),
            'price': price,
            'size': size,
            'side': trade.get('side', 'unknown'),
        }
        
        asyncio.create_task(self._add_to_buffer('trade', record))

    async def _add_to_buffer(self, data_type: str, record: Dict):
        """Safely add to buffer and signal flush if needed."""
        async with self._buffer_lock:
            if data_type == 'ob':
                self.order_book_buffer.append(record)
                buffer_len = len(self.order_book_buffer)
            else:
                self.trade_buffer.append(record)
                buffer_len = len(self.trade_buffer)
            
            self._records_since_last_flush += 1
            
            # Signal flush needed - but don't spawn new task!
            if buffer_len >= self.max_buffer_size:
                self._flush_event.set()  # Wake up the single flush loop

    # ==================== Flush / Write ====================

    async def _flush_buffers(self):
        """Write buffers to disk - only called by _flush_loop."""
        # Swap buffers under lock to minimize lock time
        async with self._buffer_lock:
            ob_to_write = self.order_book_buffer
            self.order_book_buffer = []
            
            trades_to_write = self.trade_buffer
            self.trade_buffer = []

        flushed = 0

        if ob_to_write:
            ob_path = self._get_file_path('orderbook', self.current_date)
            await self._append_jsonl(ob_path, ob_to_write)
            flushed += len(ob_to_write)

        if trades_to_write:
            trade_path = self._get_file_path('trades', self.current_date)
            await self._append_jsonl(trade_path, trades_to_write)
            flushed += len(trades_to_write)

        if flushed:
            self._total_records_written += flushed
            logger.debug(f"Flushed {flushed} records (total: {self._total_records_written})")

    async def _append_jsonl(self, file_path: Path, records: List[Dict]):
        """
        Append records as JSONL lines.
        CRITICAL: Uses 'ab' (append binary) mode for gzip to ensure atomic appends.
        """
        def _do_write():
            if not records:
                return
            
            lines = [json.dumps(rec, default=str) + '\n' for rec in records]
            data = ''.join(lines).encode('utf-8')
            
            if self.compress:
                # Use 'ab' mode for atomic binary append
                # Each write is a complete gzip member for crash safety
                with gzip.open(file_path, 'ab', compresslevel=1) as f:
                    f.write(data)
            else:
                with open(file_path, 'ab') as f:
                    f.write(data)

        try:
            await asyncio.to_thread(_do_write)
            logger.debug(f"Appended {len(records)} records to {file_path}")
        except Exception as e:
            logger.error(f"Failed to append data to {file_path}: {e}")
            # Don't lose data - put it back in buffer for retry
            async with self._buffer_lock:
                if file_path.name.endswith('trades.jsonl.gz'):
                    self.trade_buffer = records + self.trade_buffer
                else:
                    self.order_book_buffer = records + self.order_book_buffer

    # ==================== Data Loading ====================

    def load_recorded_data(
        self,
        start_date: datetime,
        end_date: datetime,
        data_type: str = 'trades'
    ) -> pd.DataFrame:
        """Load recorded data for date range."""
        all_data = []

        current = start_date
        while current <= end_date:
            for ext in ('.jsonl.gz', '.jsonl', '.json.gz', '.json'):
                date_str = current.strftime("%Y%m%d")
                file_path = self.output_dir / f"{self.symbol}_{data_type}_{date_str}{ext}"

                if file_path.exists():
                    try:
                        data = self._read_file_auto(file_path)
                        all_data.extend(data)
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")

            current += timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='mixed')
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} records from {start_date.date()} to {end_date.date()}")
        return df

    def load_filtered_data(
        self,
        start_date: datetime,
        end_date: datetime,
        data_type: str = 'trades',
        filter_config: Optional[DataFilterConfig] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Load recorded data and apply fee-aware filtering.
        
        Returns filtered DataFrame and filtering statistics.
        """
        df = self.load_recorded_data(start_date, end_date, data_type)
        
        if df.empty:
            return df, {'original_size': 0, 'filtered_size': 0, 'retention_pct': 0}
        
        data_filter = HistoricalDataFilter(filter_config)
        filtered_df, stats = data_filter.filter_ticks_with_stats(df)
        
        logger.info(
            f"[load_filtered_data] Data filtering complete: "
            f"{stats['retention_pct']:.1f}% retained"
        )
        
        # Save filtered copy if configured
        if filter_config and filter_config.save_filtered_copy:
            output_path = self.output_dir / f"{self.symbol}_{data_type}_filtered.parquet"
            filtered_df.to_parquet(output_path)
            logger.info(f"[load_filtered_data] Saved filtered data to {output_path}")
        
        return filtered_df, stats

    def _read_file_auto(self, file_path: Path) -> List[Dict]:
        """Read data file, auto-detecting JSONL vs legacy JSON format."""
        is_compressed = file_path.suffix == '.gz'
        is_jsonl = '.jsonl' in file_path.suffixes

        opener = gzip.open if is_compressed else open

        if is_jsonl:
            return self._read_jsonl(file_path, opener)
        else:
            return self._read_legacy_json(file_path, opener)

    @staticmethod
    def _read_jsonl(file_path: Path, opener) -> List[Dict]:
        """Read JSONL file with corruption recovery."""
        records = []
        try:
            with opener(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line {line_no} in {file_path}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
        return records

    @staticmethod
    def _read_legacy_json(file_path: Path, opener) -> List[Dict]:
        """Read legacy monolithic JSON array file."""
        try:
            with opener(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except Exception as e:
            logger.error(f"Error reading legacy JSON {file_path}: {e}")
            return []