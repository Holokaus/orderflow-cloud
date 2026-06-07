"""
Optuna Optimizer - Enhanced for Perfect Optimization
PHASE 1 FIXES:
1. Narrowed parameter ranges for XRP/USDT microstructure
2. Comprehensive error logging with tracebacks
3. Trial success rate monitoring
4. Per-trial timeout (60 seconds)
5. Warm-start parameter validation
6. POC range symmetric handling
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import traceback
import concurrent.futures

from loguru import logger

from knowledge.llm_advisor import LLMAdvisor, ABSORPTION_KNOWLEDGE, DELTA_DIVERGENCE_KNOWLEDGE
from knowledge.strategy_library import StrategyDefinition, get_strategy

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TimeoutException(Exception):
    """Raised when trial timeout is exceeded"""
    pass


@dataclass
class TrialSuccessStats:
    """Track trial success/failure statistics"""
    total_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0
    timeout_trials: int = 0
    pruned_trials: int = 0
    penalized_trials: int = 0
    error_log: List[Dict] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_trials == 0:
            return 0.0
        return (self.successful_trials / self.total_trials) * 100
    
    def log_every_n(self, n: int = 50) -> bool:
        """Return True if should log stats (every N trials)"""
        return self.total_trials > 0 and self.total_trials % n == 0
    
    def check_critical_threshold(self) -> Optional[str]:
        """Check if success rate is critically low"""
        if self.total_trials >= 10 and self.success_rate < 0.30:
            return (
                f"CRITICAL: Trial success rate {self.success_rate:.1f}% < 30% "
                f"({self.successful_trials}/{self.total_trials}). "
                f"Recommend tightening parameter ranges further or checking backtest fn."
            )
        return None


@dataclass
class OptimizationResult:
    """Results from optimization run"""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    study_name: str
    timestamp: datetime
    all_trials: List[Dict] = None
    objective_type: str = "robust"
    success_stats: Optional[TrialSuccessStats] = None


class StrategyOptimizer:
    """
    Enhanced strategy parameter optimizer.
    
    Features:
    1. Warm-start from LLM domain knowledge
    2. Multiple objective functions (sharpe, profit, robust)
    3. Minimum trade count enforcement
    4. Correlation-aware parameter constraints
    5. Advanced pruning
    """
    
    def __init__(
        self,
        strategy_name: str,
        llm_advisor: Optional[LLMAdvisor] = None,
        storage: str = "sqlite:///optuna_studies.db"
    ):
        self.strategy_name = strategy_name
        self.llm_advisor = llm_advisor
        self.storage = storage
        
        self.base_strategy = get_strategy(strategy_name)
        if not self.base_strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        self.param_ranges = self._get_parameter_ranges()
        self.warm_start_params = self._get_warm_start_params()
    
    def _get_parameter_ranges(self) -> Dict[str, Dict]:
        """
        PHASE 1 FIX: Apply narrowed parameter ranges for XRP/USDT microstructure.
        Previous ranges were absurdly wide (e.g., abs__entry_delta_min: 0-50,000).
        These tight ranges reflect actual microstructure bounds and reduce meaningless trials.
        
        All ranges validated and tested for XRP/USDT spot trading.
        Naming convention: {strategy}__{type}_{feature}_{bound}
        """
        
        logger.debug(f"[_get_parameter_ranges] Building NARROWED ranges for {self.strategy_name}")
        
        # NARROWED common parameters (Phase 1 spec table)
        common = {
            "trailing_stop_activation_pct": {"min": 0.002, "max": 0.03,  "step": 0.001, "default": 0.005},
            "min_conditions_satisfied":     {"min": 2,     "max": 5,      "step": 1,     "default": 2,      "type": "int"},
            "min_score_threshold":          {"min": 1.0,   "max": 6.0,    "step": 0.25,  "default": 2.5},
            "base_position_pct":            {"min": 0.02,  "max": 0.20,   "step": 0.01,  "default": 0.10},
        }
        
        # NARROWED absorption strategy parameters (Phase 1 spec table)
        strategy_params = {
            "absorption": {
                "abs__entry_str_min":       {"min": 0.2,   "max": 0.8,    "step": 0.05,   "default": 0.45},
                "abs__entry_vol_min":       {"min": 0.5,   "max": 3.0,    "step": 0.1,    "default": 1.0},
                "abs__entry_chg60_max":     {"min": 0.0005,"max": 0.005,  "step": 0.0005, "default": 0.002},
                "abs__entry_delta_min":     {"min": 0,     "max": 5000,   "step": 100,    "default": 0,      "type": "int"},
                "abs__entry_imbal_min":     {"min": 0.02,  "max": 0.40,   "step": 0.01,   "default": 0.08},
                "abs__entry_poc_range":     {"min": 0.001, "max": 0.015,  "step": 0.001,  "default": 0.006},
                "abs__filter_spread_max":   {"min": 5.0,   "max": 50.0,   "step": 1.0,    "default": 15.0},
                "abs__filter_bid_min":      {"min": 1000.0,"max": 20000.0,"step": 500.0,  "default": 2000.0},
                "abs__filter_ask_min":      {"min": 1000.0,"max": 20000.0,"step": 500.0,  "default": 2000.0},
                "abs__filter_chg300_range": {"min": 0.003, "max": 0.02,   "step": 0.001,  "default": 0.006},
            },
        }
        
        # Merge parameters
        result = common.copy()
        strat_key = self.strategy_name.lower()
        
        if strat_key in strategy_params:
            result.update(strategy_params[strat_key])
            logger.debug(f"[_get_parameter_ranges] Added {len(strategy_params[strat_key])} strategy-specific NARROWED params")
        else:
            logger.warning(f"[_get_parameter_ranges] No specific params defined for {strat_key}")
        
        logger.info(
            f"[_get_parameter_ranges] NARROWED parameter space: {len(result)} dimensions "
            f"(Phase 1 spec applied)"
        )
        return result
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial with proper type handling.
        DEBUG: Logs every suggestion for traceability.
        """
        ranges = self._get_parameter_ranges()
        params = {}
        
        logger.debug(f"[_suggest_params] Suggesting {len(ranges)} parameters for trial {trial.number}")
        
        for param_name, config in ranges.items():
            param_type = config.get("type", "float")
            step = config.get("step")
            
            try:
                if param_type == "int":
                    value = trial.suggest_int(
                        param_name,
                        config["min"],
                        config["max"],
                        step=step or 1
                    )
                elif param_type == "categorical":
                    value = trial.suggest_categorical(param_name, config["choices"])
                else:  # float
                    value = trial.suggest_float(
                        param_name,
                        config["min"],
                        config["max"],
                        step=step
                    )
                
                params[param_name] = value
                logger.debug(f"[_suggest_params] {param_name} = {value}")
                
            except Exception as e:
                logger.error(f"[_suggest_params] Failed to suggest {param_name}: {e}")
                raise
        
        return params
    
    def _get_warm_start_params(self) -> Dict[str, float]:
        """
        Get initial parameter values for warm start with validation.
        PHASE 1 FIX: Validate warm-start params are within ranges before returning.
        """
        warm_start = {}
        
        for param, range_dict in self.param_ranges.items():
            if "default" in range_dict:
                default_value = range_dict["default"]
                # Validate default is within range
                if range_dict["min"] <= default_value <= range_dict["max"]:
                    warm_start[param] = default_value
                else:
                    logger.warning(
                        f"[_get_warm_start_params] Default {default_value} for {param} "
                        f"outside range [{range_dict['min']}, {range_dict['max']}]. "
                        f"Clamping to range midpoint."
                    )
                    warm_start[param] = (range_dict["min"] + range_dict["max"]) / 2
        
        # Enforce logical constraints in warm start (regime-specific multipliers)
        regimes = ["high_vol", "low_vol", "trending"]
        for regime in regimes:
            sl_key = f"sl_mult_{regime}"
            tp_key = f"tp_mult_{regime}"
            if sl_key in warm_start and tp_key in warm_start:
                # Ensure 1.5:1 minimum risk/reward for each regime
                warm_start[tp_key] = max(
                    warm_start[tp_key],
                    warm_start[sl_key] * 1.5
                )
        
        if self.llm_advisor:
            try:
                llm_params = self.llm_advisor.get_initial_parameters(self.strategy_name)
                for param, value in llm_params.items():
                    if param in self.param_ranges:
                        range_dict = self.param_ranges[param]
                        # Validate LLM params before accepting
                        if range_dict["min"] <= value <= range_dict["max"]:
                            warm_start[param] = value
                        else:
                            logger.warning(
                                f"[_get_warm_start_params] LLM param {param}={value} "
                                f"outside range [{range_dict['min']}, {range_dict['max']}]. Rejected."
                            )
            except Exception as e:
                logger.warning(f"[_get_warm_start_params] Failed to get LLM initial params: {e}")
        
        logger.info(f"[_get_warm_start_params] Validated warm-start with {len(warm_start)} params")
        return warm_start
    
    def create_objective(
        self,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        objective_type: str = "robust",
        trial_timeout_sec: int = 60,
        success_stats: Optional[TrialSuccessStats] = None
    ) -> Callable[[optuna.Trial], float]:
        """
        Create Optuna objective function with PHASE 1 ENHANCEMENTS:
        - Comprehensive error logging (trial #, all params, exception, traceback, metrics)
        - Per-trial timeout (default 60 seconds)
        - POC range symmetric handling
        - Trial success rate monitoring
        - Penalized trial tracking
        
        Args:
            backtest_fn: Function to evaluate strategy params
            objective_type: Type of objective (robust, sharpe, profit)
            trial_timeout_sec: Maximum seconds per trial
            success_stats: Optional TrialSuccessStats object to track metrics
        """
        
        if success_stats is None:
            success_stats = TrialSuccessStats()
        
        def objective(trial: optuna.Trial) -> float:
            trial_num = trial.number
            
            # ===== PHASE 1 FIX: Start trial tracking =====
            logger.debug(f"[objective] Starting trial {trial_num}")
            
            try:
                # Step 1: Suggest parameters
                params = self._suggest_params(trial)
                
                # Step 2: PHASE 1 FIX - Enforce constraints including POC range validation
                params = self._enforce_constraints(params)
                
                # Step 3: PHASE 1 FIX - Validate POC range is symmetric (min > 0)
                if "abs__entry_poc_range" in params:
                    poc_range = params["abs__entry_poc_range"]
                    if poc_range <= 0:
                        logger.warning(
                            f"[objective] Trial {trial_num}: POC range {poc_range} <= 0. "
                            f"Clamping to minimum 0.001"
                        )
                        params["abs__entry_poc_range"] = 0.001
                
                logger.debug(
                    f"[objective] Trial {trial_num} params validated. "
                    f"Key params: trailing_stop={params.get('trailing_stop_activation_pct', 'N/A')}, "
                    f"min_conditions={params.get('min_conditions_satisfied', 'N/A')}, "
                    f"min_score={params.get('min_score_threshold', 'N/A')}"
                )
                
                # Step 4: PHASE 1 FIX - Run backtest with cross-platform timeout
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(backtest_fn, params)
                        try:
                            metrics = future.result(timeout=trial_timeout_sec)
                        except concurrent.futures.TimeoutError:
                            raise TimeoutException(f"Trial exceeded {trial_timeout_sec} second timeout")
                    
                    logger.debug(
                        f"[objective] Trial {trial_num} backtest complete. "
                        f"Metrics: sharpe={metrics.get('sharpe_ratio', 'N/A')}, "
                        f"trades={metrics.get('total_trades', 'N/A')}, "
                        f"return={metrics.get('total_return_pct', 'N/A'):.4f}"
                    )
                    
                except TimeoutException as e:
                    # PHASE 1 FIX: Log timeout
                    logger.error(
                        f"[objective] Trial {trial_num} TIMEOUT: {trial_timeout_sec}s exceeded. "
                        f"Params: {params}"
                    )
                    success_stats.timeout_trials += 1
                    success_stats.error_log.append({
                        "trial": trial_num,
                        "error_type": "TIMEOUT",
                        "message": str(e),
                        "params": params
                    })
                    return float("-inf")
                
                # Step 5: Check if penalized (too few trades)
                if metrics.get("_penalized", False):
                    logger.warning(
                        f"[objective] Trial {trial_num} PENALIZED: insufficient trades. "
                        f"Trades: {metrics.get('total_trades', 0)}"
                    )
                    success_stats.penalized_trials += 1
                    trial.set_user_attr("penalized", True)
                    return float("-inf")
                
                # Store all metrics as user attributes
                for key, value in metrics.items():
                    if not isinstance(value, (dict, list)):  # Skip nested structures
                        try:
                            trial.set_user_attr(key, value)
                        except Exception as e:
                            logger.debug(f"[objective] Could not set attr {key}: {e}")
                
                # Step 6: Calculate objective score
                if objective_type == "sharpe":
                    score = metrics.get("sharpe_ratio", 0)
                elif objective_type == "profit":
                    score = self._profit_objective(metrics)
                elif objective_type == "profit_dd_trades":
                    score = self._profit_dd_trades_objective(metrics)
                else:  # robust
                    score = self._robust_objective(metrics)
                
                logger.debug(f"[objective] Trial {trial_num} score ({objective_type}): {score:.4f}")
                
                # Step 7: Report for pruning
                trial.report(score, step=1)
                if trial.should_prune():
                    logger.debug(f"[objective] Trial {trial_num} pruned by Hyperband")
                    success_stats.pruned_trials += 1
                    raise optuna.TrialPruned()
                
                # SUCCESS
                success_stats.successful_trials += 1
                return score
            
            except optuna.TrialPruned:
                # Already logged above
                raise
            
            except Exception as e:
                # PHASE 1 FIX: COMPREHENSIVE ERROR LOGGING
                success_stats.failed_trials += 1
                
                error_entry = {
                    "trial": trial_num,
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                    "params": params if 'params' in locals() else {},
                    "timestamp": datetime.now().isoformat()
                }
                success_stats.error_log.append(error_entry)
                
                logger.error(
                    f"[objective] Trial {trial_num} FAILED: {type(e).__name__}: {e}\n"
                    f"Parameters: {params if 'params' in locals() else 'NOT_OBTAINED'}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                
                return float("-inf")
            
            finally:
                # PHASE 1 FIX: Update total trial count and check threshold
                success_stats.total_trials += 1
                
                if success_stats.log_every_n(n=50):
                    logger.info(
                        f"[objective] Trial {trial_num} milestone - "
                        f"Success rate: {success_stats.success_rate:.1f}% "
                        f"({success_stats.successful_trials}/{success_stats.total_trials}), "
                        f"Failed: {success_stats.failed_trials}, "
                        f"Timeout: {success_stats.timeout_trials}, "
                        f"Pruned: {success_stats.pruned_trials}, "
                        f"Penalized: {success_stats.penalized_trials}"
                    )
                    
                    critical_msg = success_stats.check_critical_threshold()
                    if critical_msg:
                        logger.critical(critical_msg)
        
        # Store reference to success_stats for later retrieval
        objective._success_stats = success_stats
        return objective
    
    def _enforce_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clamp parameters to prevent numerical crashes ONLY.
        PHASE 1 FIX: Added POC range symmetric handling (min > 0).
        CRITICAL: Never call trial.suggest_*() here - that creates duplicate parameters.
        """
        # Prevent ATR math crashes (keep within float-safe bounds)
        if "stop_loss_atr_mult" in params:
            params["stop_loss_atr_mult"] = max(0.5, min(20.0, params.get("stop_loss_atr_mult", 2.5)))
        if "take_profit_atr_mult" in params:
            params["take_profit_atr_mult"] = max(1.0, min(50.0, params.get("take_profit_atr_mult", 6.0)))
        
        # Prevent filter degeneracy (rejecting all signals or accepting all)
        params["abs__filter_spread_max"] = max(5.0, min(50.0, params.get("abs__filter_spread_max", 15.0)))
        params["abs__filter_bid_min"] = max(1000.0, min(20000.0, params.get("abs__filter_bid_min", 2000.0)))
        params["abs__filter_ask_min"] = max(1000.0, min(20000.0, params.get("abs__filter_ask_min", 2000.0)))
        params["abs__filter_chg300_range"] = max(0.003, min(0.02, params.get("abs__filter_chg300_range", 0.006)))
        
        # PHASE 1 FIX: Ensure POC range is symmetric (min > 0, not degenerate)
        if "abs__entry_poc_range" in params:
            poc = params["abs__entry_poc_range"]
            if poc <= 0:
                logger.warning(
                    f"[_enforce_constraints] POC range {poc} <= 0. Clamping to minimum 0.001."
                )
                params["abs__entry_poc_range"] = 0.001
            elif poc < 0.001:
                logger.debug(f"[_enforce_constraints] POC range {poc} very small, clamping to 0.001")
                params["abs__entry_poc_range"] = 0.001
        
        return params

    
    def _profit_objective(self, metrics: Dict[str, float]) -> float:
        """
        Profit-focused objective with drawdown penalty.
        
        Score = Return - 3 * MaxDrawdown
        """
        total_return = metrics.get("total_return_pct", 0) * 100  # Convert to %
        max_drawdown = metrics.get("max_drawdown_pct", 0) * 100
        
        return total_return - 3 * max_drawdown
    
    def _robust_objective(self, metrics: Dict[str, float]) -> float:
        """
        Robust multi-factor objective.
        
        Combines:
        - Sharpe ratio (risk-adjusted returns)
        - Profit factor (win/loss ratio)
        - Win rate (consistency)
        - Drawdown penalty (risk control)
        
        This objective prefers strategies that are:
        - Profitable (positive return)
        - Consistent (high win rate)
        - Risk-controlled (low drawdown)
        - Efficient (good profit factor)
        """
        sharpe = metrics.get("sharpe_ratio", 0)
        profit_factor = metrics.get("profit_factor", 0)
        win_rate = metrics.get("win_rate", 0)
        max_drawdown = metrics.get("max_drawdown_pct", 0)
        total_trades = metrics.get("total_trades", 0)
        
        # Component 1: Sharpe (normalized to 0-1 range, cap at 3)
        sharpe_component = min(sharpe / 3.0, 1.0)
        
        # Component 2: Profit factor (normalized, cap at 3)
        pf_component = min(profit_factor / 3.0, 1.0) if profit_factor > 0 else 0
        
        # Component 3: Win rate (direct, but weighted lower)
        win_rate_component = win_rate * 0.8  # 80% win rate = 0.64
        
        # Component 4: Drawdown penalty (exponential decay)
        # 0% DD = 1.0, 10% DD = 0.37, 20% DD = 0.14
        drawdown_penalty = np.exp(-max_drawdown * 10)
        
        # Component 5: Trade count sufficiency
        # Sigmoid function: approaches 1 around 100 trades
        trade_sufficiency = 1 / (1 + np.exp(-0.05 * (total_trades - 100)))
        
        # Weighted combination
        weights = {
            "sharpe": 0.30,
            "profit_factor": 0.25,
            "win_rate": 0.15,
            "drawdown": 0.20,
            "trade_count": 0.10
        }
        
        score = (
            weights["sharpe"] * sharpe_component +
            weights["profit_factor"] * pf_component +
            weights["win_rate"] * win_rate_component +
            weights["drawdown"] * drawdown_penalty +
            weights["trade_count"] * trade_sufficiency
        )
        
        # Hard penalty: negative return = bad
        if metrics.get("total_return_pct", 0) < 0:
            score *= 0.5
        
        return score
    
    def _profit_dd_trades_objective(self, metrics: Dict[str, float]) -> float:
        """
        Custom objective balancing winning trades, drawdown minimization, and average win size.
        
        Weights:
        - Winning trades: 35% (sigmoid-scaled trade count)
        - Drawdown penalty: 40% (exponential decay)
        - Average win size: 25% (normalized avg_win)
        
        This objective prioritizes strategies that:
        - Generate consistent winning trades
        - Maintain low drawdown risk
        - Achieve meaningful per-trade profitability
        """
        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0)
        winning_trades = total_trades * win_rate
        
        max_drawdown = metrics.get("max_drawdown_pct", 0)
        avg_win = metrics.get("avg_win", 0)  # Use avg_win not avg_win_pct
        
        # Component 1: Winning trades (sigmoid scaling, peaks at ~50 trades)
        # 10 winning trades = ~0.5, 50 winning trades = ~0.9
        winning_trades_component = 1 / (1 + np.exp(-0.15 * (winning_trades - 25)))
        
        # Component 2: Drawdown penalty (exponential decay, stronger than robust)
        # 0% DD = 1.0, 5% DD = 0.61, 10% DD = 0.37, 15% DD = 0.22
        drawdown_penalty = np.exp(-max_drawdown * 20)
        
        # Component 3: Average win size (normalized, assume $10-100 range is good)
        # Scale so $50 avg win = 1.0, $10 = 0.2, $100 = 2.0 (capped at 1.0)
        avg_win_component = min(avg_win / 50.0, 1.0) if avg_win > 0 else 0
        
        # Weighted combination
        weights = {
            "winning_trades": 0.35,
            "drawdown": 0.40,
            "avg_win": 0.25
        }
        
        score = (
            weights["winning_trades"] * winning_trades_component +
            weights["drawdown"] * drawdown_penalty +
            weights["avg_win"] * avg_win_component
        )
        
        # Hard penalty: no winning trades = very bad
        if winning_trades < 1:
            score *= 0.1
        
        return score
    
    def optimize(
        self,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: int = 200,
        n_jobs: int = -1,
        timeout: Optional[int] = None,
        objective_type: str = "robust",
        trial_timeout_sec: int = 60,
        fast_mode: bool = False,
    ) -> OptimizationResult:
        """
        Run optimization with warm start and enhanced objective.
        
        PHASE 1 ENHANCEMENT: Tracks trial success rate and logs comprehensive statistics.
        
        Args:
            backtest_fn: Function to evaluate strategy parameters
            n_trials: Number of optimization trials
            n_jobs: Number of parallel workers (-1 = all CPUs)
            timeout: Overall optimization timeout in seconds
            objective_type: Type of objective (robust, sharpe, profit, profit_dd_trades)
            trial_timeout_sec: Maximum seconds per individual trial
            fast_mode: If True, use 10 trials, 30s timeout each, 1 worker (for quick testing)
        
        Returns:
            OptimizationResult with best parameters, score, and success statistics
        """
        if fast_mode:
            n_trials = min(n_trials, 10)
            trial_timeout_sec = min(trial_timeout_sec, 30)
            n_jobs = 1
            logger.info(f"[optimize] FAST MODE: {n_trials} trials, {trial_timeout_sec}s timeout each")
        study_name = f"{self.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # PHASE 1 FIX: Create success stats tracker
        success_stats = TrialSuccessStats()
        
        # Enhanced sampler with warm start
        sampler = TPESampler(
            n_startup_trials=20,
            multivariate=True,  # Consider parameter correlations
            warn_independent_sampling=False
        )
        
        # Hyperband pruner for aggressive early stopping
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=1,
            reduction_factor=3
        )
        
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction="maximize",
            load_if_exists=True,  # CRITICAL: Resume instead of overwrite
            sampler=sampler,
            pruner=pruner
        )
        
        # Enqueue warm-start trial
        if self.warm_start_params:
            valid_warm_start = {
                k: v for k, v in self.warm_start_params.items()
                if k in self.param_ranges
            }
            if valid_warm_start:
                study.enqueue_trial(valid_warm_start)
                logger.info(f"[optimize] Enqueued warm-start trial with {len(valid_warm_start)} params")
        
        # PHASE 1 FIX: Pass success_stats to objective
        objective = self.create_objective(
            backtest_fn,
            objective_type,
            trial_timeout_sec=trial_timeout_sec,
            success_stats=success_stats
        )
        
        logger.info(
            f"[optimize] Starting optimization: {n_trials} trials, "
            f"objective={objective_type}, n_jobs={n_jobs}, "
            f"trial_timeout={trial_timeout_sec}s, "
            f"(NARROWED parameter ranges applied)"
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=False  # Cleaner logs
        )
        
        # Filter out penalized trials for analysis
        valid_trials = [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "user_attrs": t.user_attrs
            }
            for t in study.trials
            if t.value is not None and not t.user_attrs.get("_penalized", False)
        ]
        
        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study_name=study_name,
            timestamp=datetime.now(),
            all_trials=valid_trials,
            objective_type=objective_type,
            success_stats=success_stats
        )
        
        # PHASE 1 FIX: Log comprehensive results
        logger.info("=" * 80)
        logger.info("[optimize] Optimization complete - PHASE 1 RESULTS")
        logger.info("=" * 80)
        logger.info(f"Best score ({objective_type}): {result.best_score:.4f}")
        logger.info(f"Total trials: {result.n_trials}")
        logger.info(
            f"Trial breakdown - Successful: {success_stats.successful_trials}, "
            f"Failed: {success_stats.failed_trials}, "
            f"Timeout: {success_stats.timeout_trials}, "
            f"Pruned: {success_stats.pruned_trials}, "
            f"Penalized: {success_stats.penalized_trials}"
        )
        logger.info(f"Success rate: {success_stats.success_rate:.1f}%")
        logger.info(f"Valid trials for analysis: {len(valid_trials)} / {len(study.trials)}")
        
        if success_stats.error_log:
            logger.warning(f"[optimize] {len(success_stats.error_log)} errors logged during optimization:")
            # Log first 5 errors in detail
            for error in success_stats.error_log[:5]:
                logger.warning(
                    f"  Trial {error['trial']}: {error['error_type']} - {error['message']}"
                )
            if len(success_stats.error_log) > 5:
                logger.warning(f"  ... and {len(success_stats.error_log) - 5} more errors")
        
        logger.info(f"Best params: {result.best_params}")
        logger.info("=" * 80)
        
        return result
    
    def save_results(self, result: OptimizationResult, path: str) -> None:
        """
        Save optimization results to file, including PHASE 1 success stats.
        """
        output = {
            "strategy_name": self.strategy_name,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
            "valid_trials": len(result.all_trials) if result.all_trials else 0,
            "timestamp": result.timestamp.isoformat(),
            "objective_type": result.objective_type,
            "param_ranges": self.param_ranges,
            "warm_start_params": self.warm_start_params,
            # PHASE 1 FIX: Include success statistics
            "success_stats": {
                "total_trials": result.success_stats.total_trials if result.success_stats else 0,
                "successful_trials": result.success_stats.successful_trials if result.success_stats else 0,
                "failed_trials": result.success_stats.failed_trials if result.success_stats else 0,
                "timeout_trials": result.success_stats.timeout_trials if result.success_stats else 0,
                "pruned_trials": result.success_stats.pruned_trials if result.success_stats else 0,
                "penalized_trials": result.success_stats.penalized_trials if result.success_stats else 0,
                "success_rate_pct": result.success_stats.success_rate if result.success_stats else 0,
            }
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"[save_results] Saved results to {path}")
    
    def analyze_parameter_sensitivity(self, result: OptimizationResult) -> Dict[str, float]:
        """
        Analyze which parameters most affect the objective.
        Returns importance scores for each parameter.
        """
        if not result.all_trials or len(result.all_trials) < 10:
            return {}
        
        # Get parameter names
        param_names = list(result.all_trials[0]["params"].keys())
        
        importance = {}
        
        for param in param_names:
            # Get param values and corresponding scores
            pairs = [
                (t["params"].get(param), t["value"])
                for t in result.all_trials
                if param in t["params"] and t["value"] is not None
            ]
            
            if len(pairs) < 5:
                continue
            
            values = np.array([p[0] for p in pairs])
            scores = np.array([p[1] for p in pairs])
            
            # Calculate correlation as importance
            if np.std(values) > 0 and np.std(scores) > 0:
                correlation = np.corrcoef(values, scores)[0, 1]
                importance[param] = abs(correlation)
            else:
                importance[param] = 0.0
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance