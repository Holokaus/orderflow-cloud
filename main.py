"""
Main Entry Point - Enhanced for Perfect Optimization
Orchestrates the complete order flow trading system.
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Optional, Dict, Any
import sys

from loguru import logger
import pandas as pd

logger.remove()

logger.add(
    sys.stderr,
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}"
)

from config.settings import settings, Settings
from core.feature_engine import FeatureEngine, FeatureConfig
from core.data_structures import (
    OrderFlowState, Signal, PriceLevel, OrderBook, Side, Trade, SignalType,
    OrderType, Regime, FootprintBar, VolumeProfile, Imbalance, Absorption,
    LiquiditySweep, IcebergOrder
)
from knowledge.llm_advisor import LLMAdvisor, LLMProvider
from knowledge.strategy_library import get_all_strategies, get_strategy
from execution.risk_manager import RiskManager, RiskLimits, RiskAction
from execution.order_manager import OrderManager
from data.exchange_connector import ExchangeConnector, ExchangeConfig


class OrderFlowSystem:
    """
    Main system orchestrator with proper state management for optimization.
    """

    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()

        self.llm_advisor: Optional[LLMAdvisor] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.exchange: Optional[ExchangeConnector] = None
        self.data_recorder: Optional[Any] = None
        self.onchain_connector: Optional[Any] = None
        self.onchain_filter: Optional[Any] = None

        self.running = False
        self.current_state: Optional[OrderFlowState] = None

        self.paper_strategies: list = []
        self.paper_position = None
        self.paper_closed_trades: list = []
        self.paper_consecutive_losses = 0
        self.paper_last_trade_time = None
        self.paper_entry_timestamp = None
        self._pending_trades: list = []
        self.paper_initial_capital = 100.0
        self.paper_capital = 100.0

    def _init_llm(self) -> None:
        self.llm_advisor = LLMAdvisor.create_with_failover(self.settings.llm)
        if self.llm_advisor is None:
            logger.warning("LLM advisor not available - using hardcoded constants only")
        else:
            logger.info(f"LLM advisor ready: {self.llm_advisor.provider.value}")

    def _init_components(self, mode: str, testnet: bool = None) -> None:
        self.feature_engine = FeatureEngine(FeatureConfig(
            windows=self.settings.trading.feature_windows
        ))

        if mode in ['paper', 'live']:
            self.risk_manager = RiskManager(
                limits=RiskLimits(
                    max_position_size=self.settings.trading.max_position_size,
                    max_daily_loss_pct=self.settings.trading.max_daily_loss_pct
                ),
                initial_equity=self.settings.backtest.initial_capital
            )
            self.order_manager = OrderManager()

        if mode in ['record', 'paper', 'live']:
            use_testnet = testnet if testnet is not None else (mode == 'paper')
            self.exchange = ExchangeConnector(ExchangeConfig(
                exchange_id=self.settings.trading.exchange.value,
                testnet=use_testnet,
            ))

        if mode == 'record':
            from data.data_recorder import DataRecorder
            self.data_recorder = DataRecorder(
                symbol=self.settings.trading.symbol.replace('/', '')
            )

        if self.settings.onchain.enabled:
            try:
                from data.onchain_connector import GlassnodeConnector, OnChainRegimeFilter
                self.onchain_connector = GlassnodeConnector(
                    api_key=self.settings.onchain.api_key,
                    asset=self.settings.trading.symbol.split('/')[0]
                )
                self.onchain_filter = OnChainRegimeFilter()
                logger.info("On-chain regime filter enabled")
            except ImportError as e:
                logger.warning(f"On-chain filter disabled: {e}")

    def validate_optional_features(self) -> dict:
        status = {}
        if self.settings.onchain.enabled:
            if self.onchain_connector:
                status["onchain"] = "ready"
            else:
                status["onchain"] = "disabled (import failed)"
        else:
            status["onchain"] = "disabled (config)"
        if self.settings.ml_ensemble.enabled:
            try:
                from prediction.ml_ensemble import MLEnsemble
                status["ml_ensemble"] = "ready"
            except ImportError as e:
                status["ml_ensemble"] = f"missing deps: {e}"
        else:
            status["ml_ensemble"] = "disabled (config)"
        if self.settings.fee_aware_filter.enabled:
            status["fee_aware_filter"] = "enabled"
        else:
            status["fee_aware_filter"] = "disabled"
        if self.settings.data_filtering.enabled:
            status["data_filtering"] = "enabled"
        else:
            status["data_filtering"] = "disabled"
        return status

    def _print_startup_dashboard(self) -> None:
        print("\n" + "=" * 60)
        print("  ORDER FLOW TRADING SYSTEM \u2014 STARTUP DASHBOARD")
        print("=" * 60)
        print(f"  Trading Pair : {self.settings.trading.symbol}")
        print(f"  Exchange     : {self.settings.trading.exchange.value}")
        print(f"  Mode         : paper/live recording")
        print("-" * 60)
        status = self.validate_optional_features()
        for feature, state in status.items():
            if "disabled" in state:
                print(f"  [-] {feature:20s}  {state}")
            else:
                print(f"  [+] {feature:20s}  {state}")
        print("=" * 60 + "\n")

    def _get_feature_config(self) -> FeatureConfig:
        return FeatureConfig(
            windows=self.settings.trading.feature_windows,
            tick_size=self.settings.trading.tick_size
        )

    async def run_record(self, duration_hours: int = 24) -> None:
        logger.info(f"Starting data recording for {duration_hours} hours")
        self._init_components('record')
        try:
            await self.exchange.connect()
            self.exchange.on_order_book_update = self._on_order_book_record
            self.exchange.on_trade = self._on_trade_record
            record_task = asyncio.create_task(self.data_recorder.start_recording())
            ws_task = asyncio.create_task(
                self.exchange.start_websocket(self.settings.trading.symbol)
            )
            await asyncio.sleep(duration_hours * 3600)
            await self.data_recorder.stop_recording()
            self.exchange.running = False
            logger.info("Recording complete")
        finally:
            await self.exchange.disconnect()

    async def _on_order_book_record(self, order_book: dict) -> None:
        self.data_recorder.record_order_book(order_book)

    async def _on_trade_record(self, trade: dict) -> None:
        self.data_recorder.record_trade(trade)

    def run_backtest(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        params: dict = None,
        merged: bool = False
    ) -> dict:
        from backtesting.engine import BacktestEngine
        logger.info(f"Running backtest: {strategy_name} from {start_date} to {end_date}")
        data = self._load_backtest_data(start_date, end_date, merged)
        if data.empty:
            logger.error("No data available for backtest period")
            return {}
        strategy = get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Unknown strategy: {strategy_name}")
            return {}
        engine = BacktestEngine(
            initial_capital=self.settings.backtest.initial_capital,
            fee_pct=self.settings.trading.fee_pct,
            slippage_pct=self.settings.trading.slippage_estimate_pct,
            feature_config=self._get_feature_config(),
            risk_limits=RiskLimits(
                max_position_size=self.settings.trading.max_position_size,
                max_position_value_pct=self.settings.trading.max_position_value_pct,
            )
        )
        metrics = engine.run(data, strategy, params)
        logger.info("Backtest Results:")
        logger.info(f"  Total Return: {metrics.total_return_pct*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Win Rate: {metrics.win_rate*100:.1f}%")
        logger.info(f"  Max Drawdown: {metrics.max_drawdown_pct*100:.2f}%")
        logger.info(f"  Total Trades: {metrics.total_trades}")
        logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
        return {
            "total_return_pct": metrics.total_return_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "win_rate": metrics.win_rate,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "profit_factor": metrics.profit_factor,
            "total_trades": metrics.total_trades
        }

    def _load_backtest_data(
        self,
        start_date: datetime,
        end_date: datetime,
        merged: bool
    ) -> pd.DataFrame:
        if merged:
            symbol_clean = self.settings.trading.symbol.replace('/', '')
            merged_file = f"./data/backtests/{symbol_clean}_{start_date.strftime('%Y%m%d')}_merged.parquet"
            logger.info(f"Loading pre-merged data: {merged_file}")
            try:
                data = pd.read_parquet(merged_file)
                data = data[
                    (data['timestamp'] >= pd.Timestamp(start_date, tz='UTC')) &
                    (data['timestamp'] < pd.Timestamp(end_date + timedelta(days=1), tz='UTC'))
                ]
                return data
            except FileNotFoundError:
                logger.error(f"Merged file not found: {merged_file}")
                return pd.DataFrame()
        else:
            from data.data_recorder import DataRecorder
            recorder = DataRecorder()
            return recorder.load_recorded_data(start_date, end_date, 'trades')

    def run_optimization(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        n_trials: int = 200,
        merged: bool = False,
        objective: str = "robust"
    ) -> dict:
        from optimization.optuna_optimizer import StrategyOptimizer
        logger.info(f"Starting optimization: {strategy_name} (objective={objective})")
        self._init_llm()
        data = self._load_backtest_data(start_date, end_date, merged)
        if data.empty:
            logger.error("No data available")
            return {}
        strategy = get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Unknown strategy: {strategy_name}")
            return {}
        optimizer = StrategyOptimizer(
            strategy_name=strategy_name,
            llm_advisor=self.llm_advisor,
            storage=self.settings.optuna.storage
        )
        feature_config = self._get_feature_config()

        def backtest_fn(params: Dict[str, Any]) -> Dict[str, float]:
            from backtesting.engine import BacktestEngine
            engine = BacktestEngine(
                initial_capital=self.settings.backtest.initial_capital,
                fee_pct=self.settings.trading.fee_pct,
                slippage_pct=self.settings.trading.slippage_estimate_pct,
                feature_config=feature_config
            )
            metrics = engine.run(data, strategy, params)
            if metrics.total_trades < self.settings.optuna.min_trades_required:
                return {
                    "sharpe_ratio": -999, "profit_factor": 0,
                    "win_rate": 0, "max_drawdown_pct": 1.0,
                    "total_trades": 0, "total_return_pct": 0, "_penalized": True
                }
            return {
                "sharpe_ratio": metrics.sharpe_ratio, "profit_factor": metrics.profit_factor,
                "win_rate": metrics.win_rate, "max_drawdown_pct": metrics.max_drawdown_pct,
                "total_trades": metrics.total_trades, "total_return_pct": metrics.total_return_pct,
                "_penalized": False
            }

        result = optimizer.optimize(backtest_fn, n_trials=n_trials,
                                    n_jobs=self.settings.optuna.n_jobs,
                                    objective_type=objective)
        output_path = f"./results/{strategy_name}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        optimizer.save_results(result, output_path)
        sensitivity = optimizer.analyze_parameter_sensitivity(result)
        if sensitivity:
            logger.info("Parameter Sensitivity (importance):")
            for param, importance in sorted(sensitivity.items(), key=lambda x: -x[1]):
                logger.info(f"  {param}: {importance:.3f}")
        self._log_optimization_analysis(result)
        return {
            "best_params": result.best_params, "best_score": result.best_score,
            "n_trials": result.n_trials, "objective": objective, "output_path": output_path
        }

    def _log_optimization_analysis(self, result) -> None:
        if not result.all_trials:
            return
        valid_trials = [t for t in result.all_trials if not t.get("user_attrs", {}).get("_penalized", False)]
        logger.info(f"Optimization Analysis:")
        logger.info(f"  Valid trials: {len(valid_trials)} / {len(result.all_trials)}")
        if valid_trials:
            scores = [t["value"] for t in valid_trials if t["value"] is not None]
            trades = [t["user_attrs"].get("total_trades", 0) for t in valid_trials]
            if scores:
                logger.info(f"  Score range: {min(scores):.4f} to {max(scores):.4f}")
            if trades:
                logger.info(f"  Trade count range: {min(trades)} to {max(trades)}")

    def run_walk_forward(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        merged: bool = False,
        n_trials_per_fold: int = 50
    ) -> dict:
        from optimization.optuna_optimizer import StrategyOptimizer
        from backtesting.engine import WalkForwardValidator
        logger.info(f"Starting walk-forward validation: {strategy_name}")
        self._init_llm()
        data = self._load_backtest_data(start_date, end_date, merged)
        if data.empty:
            logger.error("No data available")
            return {}
        strategy = get_strategy(strategy_name)
        optimizer = StrategyOptimizer(
            strategy_name=strategy_name, llm_advisor=self.llm_advisor,
            storage=self.settings.optuna.storage
        )
        validator = WalkForwardValidator(
            train_days=self.settings.backtest.train_window_days,
            test_days=self.settings.backtest.test_window_days,
            step_days=self.settings.backtest.step_days,
            min_oos_trades=self.settings.optuna.min_trades_required
        )
        results = validator.validate(
            data=data, strategy=strategy, optimizer=optimizer,
            n_trials_per_fold=n_trials_per_fold,
            feature_config=self._get_feature_config()
        )
        summary = results.get('summary', {})
        logger.info("Walk-Forward Results:")
        logger.info(f"  Valid folds: {summary.get('valid_folds', 0)}/{summary.get('total_folds', 0)}")
        logger.info(f"  Avg OOS Sharpe: {summary.get('avg_oos_sharpe', 0):.2f}")
        logger.info(f"  Profitable Folds: {summary.get('pct_profitable_folds', 0)*100:.1f}%")
        if summary.get('skipped_folds', 0) > 0:
            logger.warning(f"  Skipped folds: {summary['skipped_folds']}")
        stability = results.get('params_stability', {})
        if stability:
            logger.info("Parameter Stability (CV - lower is more stable):")
            for param, stats in sorted(stability.items(), key=lambda x: x[1].get('cv', 999)):
                cv = stats.get('cv', 0)
                flag = "STABLE" if cv < 0.3 else "UNSTABLE" if cv > 0.7 else "MODERATE"
                logger.info(f"  {param}: CV={cv:.2f} [{flag}]")
        return results

    async def run_paper(self, strategy_names: list, params: dict = None, testnet: bool = None) -> None:
        logger.info(f"Starting paper trading: {strategy_names}")
        self._init_components('paper', testnet=testnet)
        self.paper_strategies = []
        for name in strategy_names:
            strat = get_strategy(name)
            if strat:
                self.paper_strategies.append((name, strat))
                logger.info(f"  Loaded strategy: {name}")
            else:
                logger.warning(f"Unknown strategy: {name}")
        if not self.paper_strategies:
            logger.error("No valid strategies loaded")
            return
        self.paper_feature_engine = FeatureEngine(
            FeatureConfig(
                windows=self.settings.trading.feature_windows,
                tick_size=self.settings.trading.tick_size,
            )
        )
        await self.exchange.connect()
        self.exchange.on_order_book_update = self._on_paper_order_book
        self.exchange.on_trade = self._on_paper_trade
        self.running = True
        await self.exchange.start_websocket(self.settings.trading.symbol)

    async def _on_paper_trade(self, trade: dict) -> None:
        self._pending_trades.append(trade)

    async def _on_paper_order_book(self, order_book: dict) -> None:
        if not self.running or not self.paper_strategies:
            return
        try:
            ts = datetime.now()
            bids_raw = order_book.get('bids', [])
            asks_raw = order_book.get('asks', [])
            bids = [PriceLevel(price=float(p), size=float(s), timestamp=ts)
                    for p, s in bids_raw[:20] if float(p) > 0 and float(s) > 0]
            asks = [PriceLevel(price=float(p), size=float(s), timestamp=ts)
                    for p, s in asks_raw[:20] if float(p) > 0 and float(s) > 0]
            bids.sort(key=lambda l: l.price, reverse=True)
            asks.sort(key=lambda l: l.price)
            ob = OrderBook(timestamp=ts, bids=bids, asks=asks)
            trades = []
            for t in self._pending_trades:
                price = float(t.get('price', 0))
                size = float(t.get('size', 0))
                side_str = str(t.get('side', 'buy')).lower().strip()
                side = Side.BUY if side_str == 'buy' else Side.SELL
                if price > 0 and size > 0:
                    trades.append(Trade(timestamp=ts, price=price, size=size, side=side))
            self._pending_trades.clear()
            if not ob.best_bid or not ob.best_ask or ob.mid_price <= 0:
                return
            state = self.paper_feature_engine.update(ob, trades)
            if self.paper_position:
                self._update_paper_position(state)
                self._update_paper_trailing_stop(state)
                if self.paper_entry_timestamp is not None:
                    exit_reason = self._check_paper_exit(state, ts)
                    if exit_reason:
                        self._close_paper_trade(state, ts, exit_reason)
            if self.paper_position:
                return
            if self.paper_last_trade_time is not None:
                elapsed = (ts - self.paper_last_trade_time).total_seconds()
                if elapsed < self.settings.trading.min_time_between_trades_sec:
                    return
            if self.paper_consecutive_losses >= 2:
                return
            mid = ob.mid_price
            for strat_name, strat in self.paper_strategies:
                signal = strat.evaluate(state)
                if not signal or not signal.is_actionable:
                    continue
                if signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL):
                    continue
                entry_price = ob.best_ask.price * (1 + self.settings.trading.slippage_estimate_pct)
                risk_action, adjusted_signal, reason = self.risk_manager.check_signal(signal, mid, ts)
                if risk_action in (RiskAction.HALT_TRADING, RiskAction.REJECT):
                    continue
                if signal.entry_price > 0 and signal.take_profit != signal.entry_price:
                    predicted_move = abs(signal.take_profit - signal.entry_price) / signal.entry_price
                else:
                    predicted_move = 0.002
                fee_check = self.order_manager.validate_signal_with_fee_filter(
                    signal, predicted_move, signal.confidence,
                    ob.best_bid.price, ob.best_ask.price, ob.mid_price
                )
                if fee_check['status'] == 'REJECTED':
                    continue
                sig = adjusted_signal or signal
                pos_value = self.paper_capital * sig.position_size
                size = pos_value / entry_price
                logger.info(f"[{strat_name}] PAPER ENTRY @ {entry_price:.4f} | "
                           f"SL: {sig.stop_loss:.4f} | TP: {sig.take_profit:.4f} | "
                           f"Size: {size:.4f}")
                self.paper_position = {
                    'strategy': strat_name, 'entry_time': ts,
                    'entry_price': entry_price, 'size': size,
                    'allocated': pos_value, 'stop_loss': sig.stop_loss,
                    'take_profit': sig.take_profit,
                    'trailing_activation': strat.trailing_stop_activation_pct,
                    'trailing_active': False, 'trailing_stop_price': 0.0,
                    'highest_price': entry_price,
                    'entry_fee': pos_value * self.settings.trading.fee_pct,
                }
                self.paper_entry_timestamp = ts
                self.paper_capital -= pos_value + (pos_value * self.settings.trading.fee_pct)
                break
        except Exception as e:
            logger.error(f"Paper trading error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _update_paper_position(self, state: OrderFlowState) -> None:
        pos = self.paper_position
        if not pos:
            return
        book = state.order_book
        mid = book.mid_price
        mark = book.best_bid.price if book.best_bid else mid
        pos['unrealized_pnl'] = (mark - pos['entry_price']) * pos['size']
        pos['highest_price'] = max(pos['highest_price'], mid)

    def _update_paper_trailing_stop(self, state: OrderFlowState) -> None:
        pos = self.paper_position
        if not pos or pos['trailing_activation'] <= 0:
            return
        book = state.order_book
        mark = book.best_bid.price if book.best_bid else book.mid_price
        move_pct = (mark - pos['entry_price']) / pos['entry_price']
        if move_pct >= pos['trailing_activation']:
            pos['trailing_active'] = True
            trail_distance = pos['trailing_activation'] * 0.5
            new_stop = mark * (1 - trail_distance)
            if new_stop > pos.get('trailing_stop_price', 0):
                pos['trailing_stop_price'] = new_stop
                pos['stop_loss'] = max(pos['stop_loss'], new_stop)

    def _check_paper_exit(self, state: OrderFlowState, ts: datetime) -> str:
        pos = self.paper_position
        if not pos:
            return None
        if self.paper_entry_timestamp is not None and ts == self.paper_entry_timestamp:
            return None
        book = state.order_book
        exit_price = book.best_bid.price if book.best_bid else book.mid_price
        hold_sec = (ts - pos['entry_time']).total_seconds()
        if exit_price <= pos['stop_loss']:
            return 'stop_loss'
        if exit_price >= pos['take_profit']:
            return 'take_profit'
        if hold_sec < 1200:
            return None
        net_pressure = state.features.get('net_pressure', 0)
        if net_pressure < -0.5:
            bid_depth = state.features.get('bid_depth_10', 0)
            ask_depth = state.features.get('ask_depth_10', 0)
            if ask_depth > 0 and bid_depth / (ask_depth + 1e-9) < 0.3:
                return 'book_pressure_collapse'
        return None

    def _close_paper_trade(self, state: OrderFlowState, ts: datetime, reason: str) -> None:
        pos = self.paper_position
        if not pos:
            return
        book = state.order_book
        best_bid = book.best_bid.price if book.best_bid else book.mid_price
        adverse_slip = (self.settings.trading.slippage_estimate_pct + 0.0003
                       if reason == 'stop_loss' else self.settings.trading.slippage_estimate_pct)
        exit_price = best_bid * (1 - adverse_slip)
        gross_pnl = (exit_price - pos['entry_price']) * pos['size']
        exit_value = exit_price * pos['size']
        exit_fee = exit_value * self.settings.trading.fee_pct
        net_pnl = gross_pnl - exit_fee
        notional = pos['entry_price'] * pos['size']
        pnl_pct = net_pnl / notional if notional > 0 else 0.0
        duration = (ts - pos['entry_time']).total_seconds()
        self.paper_capital += (pos['allocated'] - pos.get('entry_fee', 0)) + net_pnl
        trade_record = {
            'exit_time': ts, 'exit_price': exit_price,
            'entry_price': pos['entry_price'], 'exit_reason': reason,
            'pnl': net_pnl, 'pnl_pct': pnl_pct, 'strategy': pos['strategy'],
            'duration_seconds': duration, 'entry_time': pos['entry_time'],
        }
        self.paper_closed_trades.append(trade_record)
        if net_pnl <= 0:
            self.paper_consecutive_losses += 1
        else:
            self.paper_consecutive_losses = 0
        self.paper_last_trade_time = ts
        self.paper_position = None
        self.paper_entry_timestamp = None
        equity = self.paper_capital
        logger.info(f"[{pos['strategy']}] PAPER EXIT: {reason} | "
                   f"PnL: {net_pnl:.4f}$ ({pnl_pct*100:+.3f}%) | "
                   f"Eq: {equity:.2f}$ | Dur: {duration:.0f}s")

    async def run_live(self, strategy_name: str, params: dict) -> None:
        logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
        confirm = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
        if confirm != "CONFIRM LIVE TRADING":
            logger.info("Live trading cancelled")
            return
        self._init_components('live')


async def run_connectivity_test(symbol: str = "XRP/USDT") -> None:
    print("\n" + "="*70)
    print("CONNECTIVITY TEST")
    print("="*70 + "\n")
    exchange = ExchangeConnector(ExchangeConfig(exchange_id="binance", testnet=False))
    try:
        await exchange.connect()
        results = await exchange.test_connectivity(symbol)
        print(f"Symbol: {symbol}\n")
        print("REST API Connectivity:")
        rest_result = results.get("rest", {})
        if rest_result.get("ok"):
            print(f"  OK (latency: {rest_result.get('latency_ms', 0):.1f}ms)")
        else:
            print(f"  FAILED: {rest_result.get('error', 'Unknown error')}")
        print("\nWebSocket Connectivity:")
        ws_results = results.get("ws_urls", [])
        if ws_results:
            for i, ws_result in enumerate(ws_results, 1):
                url = ws_result.get("url", "")
                ok = ws_result.get("ok", False)
                latency = ws_result.get("latency_ms", 0)
                error = ws_result.get("error", "")
                try:
                    host = url.split('/')[2].split(':')[0]
                except:
                    host = "unknown"
                if ok:
                    print(f"  {host} OK (latency: {latency:.1f}ms)")
                else:
                    print(f"  {host} FAILED: {error}")
        else:
            print("  No WebSocket URLs tested")
        print("\n" + "="*70)
        all_ok = (results.get("rest", {}).get("ok", False) and
                  all(r.get("ok", False) for r in ws_results))
        if all_ok:
            print("ALL TESTS PASSED: Ready for production")
        else:
            print("SOME TESTS FAILED: Check network, firewall, or DNS")
        print("="*70 + "\n")
    finally:
        await exchange.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Order Flow Trading System")
    parser.add_argument('mode', choices=['record', 'backtest', 'optimize', 'validate', 'paper', 'live', 'test'],
                       help='Operation mode')
    parser.add_argument('--strategy', type=str, default='absorption', help='Strategy name')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--duration', type=int, default=24, help='Recording duration in hours')
    parser.add_argument('--trials', type=int, default=200, help='Number of optimization trials')
    parser.add_argument('--merged', action='store_true', help='Use pre-merged data')
    parser.add_argument('--objective', type=str, choices=['sharpe', 'profit', 'robust'], default='robust',
                       help='Optimization objective')
    parser.add_argument('--trials-per-fold', type=int, default=50,
                       help='Number of optimization trials per walk-forward fold')
    args = parser.parse_args()
    start_date = (datetime.strptime(args.start_date, '%Y-%m-%d')
                  if args.start_date else datetime.now() - timedelta(days=30))
    end_date = (datetime.strptime(args.end_date, '%Y-%m-%d')
                if args.end_date else datetime.now())
    system = OrderFlowSystem()
    if args.mode != 'test':
        system._print_startup_dashboard()
    if args.mode == 'record':
        asyncio.run(system.run_record(args.duration))
    elif args.mode == 'backtest':
        result = system.run_backtest(args.strategy, start_date, end_date, merged=args.merged)
        if result:
            print("\n" + "="*50)
            print("BACKTEST SUMMARY")
            print("="*50)
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print("="*50)
    elif args.mode == 'optimize':
        result = system.run_optimization(args.strategy, start_date, end_date,
                                         n_trials=args.trials, merged=args.merged,
                                         objective=args.objective)
        if result:
            print("\n" + "="*50)
            print("OPTIMIZATION SUMMARY")
            print("="*50)
            print(f"  Objective: {args.objective}")
            print(f"  Best Score: {result['best_score']:.4f}")
            print(f"  Best Params:")
            for key, value in result['best_params'].items():
                print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
            print(f"  Output: {result.get('output_path', 'N/A')}")
            print("="*50)
    elif args.mode == 'validate':
        result = system.run_walk_forward(args.strategy, start_date, end_date,
                                         merged=args.merged, n_trials_per_fold=args.trials_per_fold)
        if result:
            print("\n" + "="*50)
            print("WALK-FORWARD VALIDATION SUMMARY")
            print("="*50)
            summary = result.get('summary', {})
            print(f"  Valid Folds: {summary.get('valid_folds', 0)}/{summary.get('total_folds', 0)}")
            print(f"  Avg OOS Sharpe: {summary.get('avg_oos_sharpe', 0):.2f}")
            print(f"  Profitable Folds: {summary.get('pct_profitable_folds', 0)*100:.1f}%")
            if summary.get('error'):
                print(f"  ERROR: {summary['error']}")
            print("="*50)
    elif args.mode == 'paper':
        strategies = [s.strip() for s in args.strategy.split(',')]
        asyncio.run(system.run_paper(strategies, testnet=False))
    elif args.mode == 'live':
        asyncio.run(system.run_live(args.strategy, {}))
    elif args.mode == 'test':
        asyncio.run(run_connectivity_test(args.strategy))


if __name__ == "__main__":
    main()
