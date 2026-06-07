"""
Configuration and Settings
All parameters, API keys, and system configuration
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()


class Exchange(str, Enum):
    BINANCE = "binance"
    COINBASE = "coinbasepro"
    BYBIT = "bybit"
    OKX = "okx"


class LLMProvider(str, Enum):
    # Native SDK providers
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    # OpenAI-compatible providers
    OPENROUTER = "openrouter"
    GITHUB_MODELS = "github_models"
    XAI = "xai"
    ZAI = "zai"
    GROQ = "groq"
    OLLAMA = "ollama"


class TradingConfig(BaseModel):
    """Core trading parameters"""
    symbol: str = "BTC/USDT"
    exchange: Exchange = Exchange.BINANCE
    
    # Position limits
    max_position_size: float = 10000.0  # Was 1 for BTC Now 10000 for XRP
    max_position_value_pct: float = 0.25  # 25% of account
    
    # Risk limits
    max_daily_loss_pct: float = 0.02  # 2%
    max_drawdown_pct: float = 0.05  # 5%
    max_consecutive_losses: int = 20
    
    # Execution
    min_time_between_trades_sec: int = 30
    slippage_estimate_pct: float = 0.0005  # 0.05%
    fee_pct: float = 0.0005   # 0.05% taker (Binance futures)
    
    #Add tick_size to the TradingConfig so it can be passed to the feature engine.
    tick_size: float = 0.0001  # <--- XRP typically uses 0.0001 or 0.01
    
    # Feature computation windows (seconds)
    feature_windows: List[int] = [15, 30, 60, 300, 600, 900]


class LLMConfig(BaseModel):
    """LLM API configuration"""
    provider: LLMProvider = LLMProvider.GEMINI
    model: Optional[str] = None  # None lets LLMAdvisor choose per-provider default
    
    # Native SDK API Keys (from environment)
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    
    # OpenAI-compatible providers API Keys (from environment)
    openrouter_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    github_token: Optional[str] = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))
    xai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("XAI_API_KEY"))
    zai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ZAI_API_KEY"))
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    
    # OpenRouter metadata (optional headers)
    openrouter_site_url: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", "http://localhost")
    )
    openrouter_app_name: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", "OrderFlowTrader")
    )
    
    # Rate limiting
    max_calls_per_minute: int = 10
    cache_responses: bool = True
    cache_ttl_seconds: int = 3600
    
    # Failover configuration
    enable_failover: bool = True
    failover_providers: List[LLMProvider] = Field(
        default_factory=lambda: [
            LLMProvider.GEMINI,
            LLMProvider.GROQ,
            LLMProvider.OPENROUTER,
            LLMProvider.OLLAMA,
        ]
    )
    
    def get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """Return API key for given provider"""
        key_map = {
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GEMINI: self.gemini_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENROUTER: self.openrouter_api_key,
            LLMProvider.GITHUB_MODELS: self.github_token,
            LLMProvider.XAI: self.xai_api_key,
            LLMProvider.ZAI: self.zai_api_key,
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.OLLAMA: "ollama",
        }
        return key_map.get(provider)
    
    def get_base_url(self, provider: LLMProvider) -> Optional[str]:
        """Return base URL for OpenAI-compatible providers"""
        urls = {
            LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1",
            LLMProvider.GITHUB_MODELS: "https://models.inference.ai.azure.com",
            LLMProvider.XAI: "https://api.x.ai/v1",
            LLMProvider.ZAI: "https://api.z.ai/api/paas/v4",
            LLMProvider.GROQ: "https://api.groq.com/openai/v1",
            LLMProvider.OLLAMA: "http://localhost:11434/v1",
        }
        return urls.get(provider)
    
    def get_failover_chain(self) -> List:
        """
        Returns ordered list of (provider, api_key, base_url) tuples to try.
        Priority: user preference first, then failover_providers list, then any remaining.
        """
        providers = []
        if self.provider not in self.failover_providers:
            providers.append(self.provider)
        providers.extend([p for p in self.failover_providers if p != self.provider])
        
        for p in LLMProvider:
            if p not in providers:
                providers.append(p)
        
        chain = []
        for provider in providers:
            api_key = self.get_api_key(provider)
            base_url = self.get_base_url(provider)
            
            if provider != LLMProvider.OLLAMA and not api_key:
                continue
                
            chain.append((provider, api_key, base_url))
        
        return chain


class OptunaConfig(BaseModel):
    """Optimization configuration - cloud optimized"""
    n_trials: int = 500
    n_startup_trials: int = 20  # Random exploration before Bayesian
    n_jobs: int = -1  # Auto-detect all CPU cores (use -1 for cloud)
    
    # Pruning
    enable_pruning: bool = True
    pruning_warmup_steps: int = 50
    
    # Study persistence
    # Use SQLite for single-instance, PostgreSQL for distributed cloud
    storage: str = "sqlite:///optuna_studies.db"
    study_name: str = "orderflow_optimization"
    
    # Objective
    primary_metric: str = "sharpe_ratio"
    min_trades_required: int = 100


class BacktestConfig(BaseModel):
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    
    # Walk-forward settings
    train_window_days: int = 60
    test_window_days: int = 14
    step_days: int = 7
    
    # Costs
    include_slippage: bool = True
    include_fees: bool = True
    
    # Data
    data_path: str = "./data/historical/"
    min_data_points: int = 100000


class FeeAwareFilterConfig(BaseModel):
    """Fee-aware signal filter configuration (Futures)"""
    enabled: bool = True
    maker_fee_pct: float = 0.0002
    taker_fee_pct: float = 0.0005
    expected_spread_pct: float = 0.0001
    min_profit_target_pct: float = 0.0002


class DataFilteringConfig(BaseModel):
    """Historical data pre-filtering configuration (Futures)"""
    enabled: bool = True
    maker_fee_pct: float = 0.0002
    taker_fee_pct: float = 0.0005
    min_spread_pct: float = 0.0001
    lookforward_window_ticks: int = 100
    save_filtered_copy: bool = True


class OnChainConfig(BaseModel):
    """On-chain data configuration"""
    enabled: bool = False
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GLASSNODE_API_KEY"))
    cache_ttl_seconds: int = 3600
    metrics: List[str] = [
        "exchange_inflow_volume", "sopr", "nupl",
        "mvrv_ratio", "active_addresses"
    ]


class MLEnsembleConfig(BaseModel):
    """ML ensemble prediction configuration"""
    enabled: bool = False
    model_dir: str = "models/ml_ensemble"
    weights: Dict[str, float] = {
        "xgboost": 0.4,
        "lightgbm": 0.4,
        "lstm": 0.2,
    }
    min_confidence: float = 0.7
    retrain_interval_days: int = 7


class Settings(BaseModel):
    """Master settings container"""
    trading: TradingConfig = TradingConfig()
    llm: LLMConfig = LLMConfig()
    optuna: OptunaConfig = OptunaConfig()
    backtest: BacktestConfig = BacktestConfig()
    fee_aware_filter: FeeAwareFilterConfig = FeeAwareFilterConfig()
    data_filtering: DataFilteringConfig = DataFilteringConfig()
    onchain: OnChainConfig = OnChainConfig()
    ml_ensemble: MLEnsembleConfig = MLEnsembleConfig()


# Global settings instance
settings = Settings()