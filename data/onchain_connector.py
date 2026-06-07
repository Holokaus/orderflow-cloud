"""
On-Chain Data Connector & Regime Filter
========================================
PHASE 4 ENHANCEMENT

Integrates on-chain metrics from Glassnode API to detect regime changes and halt trading
during high-risk periods (selling pressure, euphoria, overvaluation).

Metrics tracked:
- exchange_inflow_volume: How much BTC flowing into exchanges (selling pressure)
- SOPR: Spent Output Profit Ratio (profit taking behavior)
- NUPL: Net Unrealized Profit/Loss (bull/bear cycle indicator)
- MVRV: Market Value / Realized Value (overvaluation metric)
- active_addresses: On-chain activity level

Decision logic:
- HIGH selling pressure → reject new longs (people exiting)
- HIGH SOPR → reject new trades (profit taking dominant)
- EUPHORIA zone (NUPL > 0.75) → high-risk area (prior to crashes)
- OVERVALUED (MVRV > 3.5) → extended rally, risky
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import requests
from datetime import datetime, timedelta
from loguru import logger
import time


class OnChainRegime(Enum):
    """On-chain market regime classification"""
    NORMAL = "normal"
    HIGH_SELLING_PRESSURE = "high_selling_pressure"
    PROFIT_TAKING = "profit_taking"
    EUPHORIA = "euphoria"
    OVERVALUED = "overvalued"
    PANIC = "panic"


@dataclass
class OnChainMetrics:
    """Container for on-chain metrics at a point in time"""
    timestamp: datetime
    exchange_inflow_volume: Optional[float] = None
    exchange_inflow_ratio: Optional[float] = None  # vs. average
    sopr: Optional[float] = None
    nupl: Optional[float] = None
    mvrv_ratio: Optional[float] = None
    active_addresses: Optional[float] = None
    regime: OnChainRegime = OnChainRegime.NORMAL
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'exchange_inflow_volume': self.exchange_inflow_volume,
            'exchange_inflow_ratio': self.exchange_inflow_ratio,
            'sopr': self.sopr,
            'nupl': self.nupl,
            'mvrv_ratio': self.mvrv_ratio,
            'active_addresses': self.active_addresses,
            'regime': self.regime.value,
        }


class GlassnodeConnector:
    """
    Fetch on-chain metrics from Glassnode API.
    
    Free tier available: some metrics are available without authentication.
    For production, use API key from https://glassnode.com
    """
    
    BASE_URL = "https://api.glassnode.com/v1/metrics"
    
    def __init__(self, api_key: Optional[str] = None, asset: str = "BTC"):
        """
        Initialize Glassnode connector.
        
        Args:
            api_key: Glassnode API key (optional, limited free tier available)
            asset: Asset to track (BTC, ETH, etc.)
        """
        self.api_key = api_key
        self.asset = asset
        self.cache = {}
        self.cache_ttl_sec = 3600  # 1 hour
        
        if not api_key:
            logger.warning(
                "[GlassnodeConnector] No API key provided — on-chain requests will fail. "
                "Set GLASSNODE_API_KEY env var or pass api_key to constructor."
            )
        else:
            logger.info(f"[GlassnodeConnector] Initialized for {asset} with API key")
    
    def _get_metric(
        self,
        metric_path: str,
        days_back: int = 1,
        resample: str = "1d"
    ) -> Optional[Dict]:
        """
        Fetch a single metric from Glassnode.
        
        Args:
            metric_path: API path (e.g., 'exchange/inflow_volume')
            days_back: How many days back to fetch
            resample: Resampling interval
        
        Returns:
            Latest metric value or None if error
        """
        cache_key = f"{metric_path}_{days_back}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl_sec:
                return cached_data
        
        try:
            params = {
                'a': self.asset,
                'since': int((datetime.now() - timedelta(days=days_back)).timestamp()),
                'until': int(datetime.now().timestamp()),
                'resample': resample,
            }
            
            if self.api_key:
                params['api_key'] = self.api_key
            
            url = f"{self.BASE_URL}/{metric_path}"
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.cache[cache_key] = (data, time.time())
                return data
            else:
                logger.warning(
                    f"[GlassnodeConnector] Failed to fetch {metric_path}: "
                    f"HTTP {response.status_code}"
                )
                return None
        
        except Exception as e:
            logger.error(f"[GlassnodeConnector] Error fetching {metric_path}: {e}")
            return None
    
    def get_latest_metrics(self) -> Optional[OnChainMetrics]:
        """
        Fetch latest on-chain metrics.
        
        Returns:
            OnChainMetrics object or None if error
        """
        try:
            exchange_inflow = self._get_metric('exchange/inflow_volume')
            sopr = self._get_metric('indicators/sopr')
            nupl = self._get_metric('indicators/nupl')
            mvrv = self._get_metric('indicators/mvrv_ratio')
            active_addr = self._get_metric('active_addresses/active_addresses_24h')
            
            # Extract latest values
            metrics = OnChainMetrics(timestamp=datetime.now())
            
            if exchange_inflow and 'data' in exchange_inflow and len(exchange_inflow['data']) > 0:
                metrics.exchange_inflow_volume = exchange_inflow['data'][-1]['v']
            
            if sopr and 'data' in sopr and len(sopr['data']) > 0:
                metrics.sopr = sopr['data'][-1]['v']
            
            if nupl and 'data' in nupl and len(nupl['data']) > 0:
                metrics.nupl = nupl['data'][-1]['v']
            
            if mvrv and 'data' in mvrv and len(mvrv['data']) > 0:
                metrics.mvrv_ratio = mvrv['data'][-1]['v']
            
            if active_addr and 'data' in active_addr and len(active_addr['data']) > 0:
                metrics.active_addresses = active_addr['data'][-1]['v']
            
            logger.debug(f"[GlassnodeConnector] Fetched metrics: {metrics.to_dict()}")
            return metrics
        
        except Exception as e:
            logger.error(f"[GlassnodeConnector] Failed to get latest metrics: {e}")
            return None


class OnChainRegimeFilter:
    """
    Filter trading signals based on on-chain metrics.
    
    Prevents trading during high-risk regimes:
    - Excessive selling pressure
    - Profit taking accumulation
    - Euphoria zone (prior to crashes)
    - Overvaluation
    """
    
    def __init__(self):
        """Initialize with threshold configuration"""
        # Thresholds for regime detection
        self.thresholds = {
            "high_selling_pressure": 2.0,   # 2x average inflow
            "high_profit_taking": 1.02,      # SOPR > 1.02
            "euphoria_zone": 0.75,           # NUPL > 0.75
            "overvalued": 3.5,               # MVRV > 3.5
            "panic_selling": 0.15,           # NUPL < 0.15 (deep underwater)
        }
        
        logger.info(
            f"[OnChainRegimeFilter] Initialized with thresholds: {self.thresholds}"
        )
    
    def classify_regime(self, metrics: OnChainMetrics) -> OnChainRegime:
        """
        Classify market regime based on metrics.
        
        Args:
            metrics: OnChainMetrics object
        
        Returns:
            OnChainRegime classification
        """
        if not metrics:
            return OnChainRegime.NORMAL
        
        # Check each condition (in order of severity)
        if metrics.nupl is not None and metrics.nupl < self.thresholds["panic_selling"]:
            return OnChainRegime.PANIC
        
        if metrics.exchange_inflow_ratio is not None and metrics.exchange_inflow_ratio > self.thresholds["high_selling_pressure"]:
            return OnChainRegime.HIGH_SELLING_PRESSURE
        
        if metrics.sopr is not None and metrics.sopr > self.thresholds["high_profit_taking"]:
            return OnChainRegime.PROFIT_TAKING
        
        if metrics.nupl is not None and metrics.nupl > self.thresholds["euphoria_zone"]:
            return OnChainRegime.EUPHORIA
        
        if metrics.mvrv_ratio is not None and metrics.mvrv_ratio > self.thresholds["overvalued"]:
            return OnChainRegime.OVERVALUED
        
        return OnChainRegime.NORMAL
    
    def should_halt_new_positions(
        self,
        metrics: OnChainMetrics
    ) -> Tuple[bool, str]:
        """
        Determine if new positions should be halted based on on-chain regime.
        
        Args:
            metrics: OnChainMetrics object
        
        Returns:
            Tuple[bool, str]: (should_halt, reason)
        """
        if not metrics:
            return False, "No metrics available"
        
        regime = self.classify_regime(metrics)
        
        if regime == OnChainRegime.PANIC:
            return True, (
                f"PANIC REGIME: NUPL {metrics.nupl:.3f} < {self.thresholds['panic_selling']:.3f}. "
                f"Market underwater - halt new longs."
            )
        
        if regime == OnChainRegime.HIGH_SELLING_PRESSURE:
            return True, (
                f"HIGH SELLING PRESSURE: Exchange inflow {metrics.exchange_inflow_ratio:.2f}x > "
                f"{self.thresholds['high_selling_pressure']:.2f}x. Halt longs."
            )
        
        if regime == OnChainRegime.PROFIT_TAKING:
            return True, (
                f"PROFIT TAKING: SOPR {metrics.sopr:.4f} > {self.thresholds['high_profit_taking']:.4f}. "
                f"Holders realizing gains - risky to enter."
            )
        
        if regime == OnChainRegime.EUPHORIA:
            return True, (
                f"EUPHORIA ZONE: NUPL {metrics.nupl:.3f} > {self.thresholds['euphoria_zone']:.3f}. "
                f"Extended rally prior to correction - halt new longs."
            )
        
        if regime == OnChainRegime.OVERVALUED:
            return True, (
                f"OVERVALUED: MVRV {metrics.mvrv_ratio:.2f} > {self.thresholds['overvalued']:.2f}. "
                f"Market significantly overvalued vs. realized price."
            )
        
        return False, "On-chain metrics normal - trading allowed"
    
    def log_metrics_state(self, metrics: OnChainMetrics) -> None:
        """
        Log current on-chain metrics state.
        
        Args:
            metrics: OnChainMetrics object
        """
        regime = self.classify_regime(metrics)
        halt, reason = self.should_halt_new_positions(metrics)
        
        logger.info(
            f"[OnChainRegimeFilter] Regime: {regime.value}, Halt: {halt}, "
            f"SOPR={metrics.sopr:.4f if metrics.sopr else 'N/A'}, "
            f"NUPL={metrics.nupl:.3f if metrics.nupl else 'N/A'}, "
            f"MVRV={metrics.mvrv_ratio:.2f if metrics.mvrv_ratio else 'N/A'}"
        )


class BacktestOnChainMetricsCollector:
    """
    Collect simulated on-chain metrics during backtests for regime analysis.
    
    Useful for post-analysis: "Which trades happened during euphoria vs. normal regimes?"
    """
    
    def __init__(self):
        """Initialize collector"""
        self.metrics_history: List[OnChainMetrics] = []
        self.regime_transition_log: List[Tuple[datetime, OnChainRegime]] = []
    
    def add_metrics(self, metrics: OnChainMetrics) -> None:
        """Record metrics snapshot"""
        self.metrics_history.append(metrics)
    
    def analyze_regime_distribution(self) -> Dict[str, float]:
        """
        Analyze percentage of backtest period spent in each regime.
        
        Returns:
            Dict mapping regime -> percentage of time in regime
        """
        if not self.metrics_history:
            return {}
        
        regime_counts = {}
        for metrics in self.metrics_history:
            regime = OnChainRegime.NORMAL  # Placeholder for backtests
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
        
        total = sum(regime_counts.values())
        return {
            regime: count / total * 100
            for regime, count in regime_counts.items()
        }
    
    def report(self) -> None:
        """Log summary report"""
        distribution = self.analyze_regime_distribution()
        
        logger.info("[BacktestOnChainMetricsCollector] Regime distribution during backtest:")
        for regime, pct in distribution.items():
            logger.info(f"  {regime}: {pct:.1f}%")
