"""
PHASE 6: Walk-Forward Validation Verification
==============================================

This module documents the verification of walk-forward validation in the backtesting engine.

EXISTING IMPLEMENTATION:
- Located in: backtesting/engine.py (class WalkForwardValidator)
- Features already implemented:
  1. Proper train/test split generation
  2. Configurable windows (default: 60-day train, 14-day test, 7-day step)
  3. Minimum OOS trades filtering (min_oos_trades parameter)
  4. Parameter stability analysis (mean, std, CV)
  5. Out-of-sample Sharpe ratio tracking
  6. Profitable fold percentage reporting
  7. Thread-safe trial isolation (each fold creates own engine)
  8. Sequential optimization per fold (n_jobs=1 for stability)

VERIFICATION CHECKLIST:
✓ Class WalkForwardValidator exists in engine.py
✓ generate_splits() method creates proper train/test splits
✓ validate() method runs optimization on train, tests on OOS
✓ Minimum OOS trades enforcement (filters weak folds)
✓ Parameter stability computed (CV < 0.3 for core params)
✓ Summary statistics include:
  - avg_oos_sharpe
  - pct_profitable_folds (>50% target)
  - std_oos_sharpe
  - valid_folds / total_folds

PHASE 6 SUCCESS CRITERIA (from spec):
1. Walk-forward: 60-day train, 14-day test, 7-day step
   ✓ Implemented (configurable parameters)

2. Minimum 10 trades per OOS fold
   ✓ Implemented (min_oos_trades parameter)

3. Target: >50% profitable folds
   ✓ Can verify with pct_profitable_folds > 0.5

4. Avg OOS Sharpe > 0.5
   ✓ Can check against results['summary']['avg_oos_sharpe']

5. Parameter stability: CV < 0.3 for core params
   ✓ Can check results['params_stability'][param]['cv']

6. If poor results, return to Phase 1 and tighten ranges
   ✓ Instructions added to this file

TESTING PROTOCOL:
=================

1. Run walk-forward on 90 days of historical data:
   >>> from backtesting.engine import WalkForwardValidator
   >>> import pandas as pd
   >>> 
   >>> # Load data
   >>> df = pd.read_parquet('data/recorded/xrp_3months.parquet')
   >>> 
   >>> # Create validator
   >>> wfv = WalkForwardValidator(
   ...     train_days=60,
   ...     test_days=14,
   ...     step_days=7,
   ...     min_oos_trades=10
   ... )
   >>> 
   >>> # Run validation
   >>> results = wfv.validate(
   ...     data=df,
   ...     strategy=absorption_strategy,
   ...     optimizer=optimizer,
   ...     n_trials_per_fold=50,
   ...     feature_config=feature_config
   ... )

2. Check results:
   >>> print(results['summary'])
   >>> # Should show:
   >>> # - avg_oos_sharpe > 0.5
   >>> # - pct_profitable_folds > 0.5 (>50%)
   >>> # - valid_folds >= 3 (minimum)

3. Check parameter stability:
   >>> for param, stats in results['params_stability'].items():
   ...     print(f"{param}: CV={stats['cv']:.3f}")
   >>> # Core params should have CV < 0.3

4. Analyze individual folds:
   >>> for fold in results['folds']:
   ...     print(f"Fold {fold['fold']}: OOS Sharpe={fold['oos_sharpe']:.2f}, "
   ...           f"Trades={fold['oos_trades']}, WinRate={fold['oos_win_rate']:.1%}")

IF PHASE 6 FAILS:
==================

Action 1: Too many folds with <10 trades
   - Issue: Parameter ranges too restrictive
   - Solution: Go back to Phase 1, slightly widen key ranges
   - Example: Increase min_conditions_satisfied max from 5 to 6

Action 2: Avg OOS Sharpe < 0.5 or all negative
   - Issue: Strategy not trading enough or losing consistently
   - Solution: Phase 1 - check error logs for failed trials
   - Verify backtest_fn is working correctly (not returning -inf)

Action 3: >30% folds unprofitable
   - Issue: Parameters overfit to training period
   - Solution: Phase 1 - tighten ranges to prevent overfitting
   - Check parameter stability (CV should be < 0.3)

Action 4: Parameter CV > 0.3 (unstable parameters)
   - Issue: Parameters vary wildly between folds
   - Solution: Phase 1 - reduce step sizes or tighten ranges
   - Especially for trailing_stop_activation_pct and min_conditions_satisfied

INTEGRATION WITH PHASES 1-5:
=============================

Phase 1 (Fixed Optimizer):
   - Narrow ranges prevent noise → more stable parameters
   - Success rate monitoring catches broken ranges early
   - Example: If >3 consecutive folds fail, ranges too tight

Phase 2 (Fee-Aware Filter):
   - Backtest engine applies fee_aware_filter before opening positions
   - Tracks signals_rejected_by_fee_filter metric
   - OOS Sharpe should improve due to eliminated negative-edge trades

Phase 3 (Filtered Data):
   - Use filtered_xrp.parquet instead of raw data
   - 40-60% size reduction means faster backtests
   - Better signal quality (only strong patterns)

Phase 4 (On-Chain Filter):
   - Optional: Check on-chain regime before trading
   - In walk-forward: mark folds as "high_selling_pressure" etc.
   - Analyze: Do parameter sets differ by regime?

Phase 5 (ML Ensemble):
   - Once walk-forward passes (>50% profitable folds, Sharpe>0.5):
   - Train ML ensemble on filtered training folds
   - Validate on OOS folds
   - Add ML predictions as additional signal filter
   - Final integration: rule-based + ML + on-chain checks

SUCCESS METRICS (All must pass):
================================
✓ Phase 1: 50-trial opt completes <2 days, 30-50% success rate
✓ Phase 2: Fee filter rejects 30-50% of signals (keeps profitable ones)
✓ Phase 3: Filtered data 40-60% of original size
✓ Phase 4: On-chain filter halts trades during 3+ major selloffs
✓ Phase 5: ML ensemble agrees with rules 60-70% of time
✓ Phase 6: >50% profitable folds, avg Sharpe > 0.5
✓ Phase 6: Fee-adjusted returns positive in all test periods
✓ Phase 6: Parameter stability CV < 0.3 for core params

DEPLOYMENT READINESS:
=====================
Phase 6 PASS criteria for paper trading:
1. Run 3+ months walk-forward (minimum 10 folds)
2. ALL folds must have:
   - Sharpe > 0.2 (even if <0.5, must be positive)
   - Win rate > 40%
   - Profit factor > 1.0
   - Max drawdown < 15%
3. Parameter consistency: core params CV < 0.25
4. 3+ months profitable paper trading before live

Phase 6 PASS criteria for live trading:
1. All paper trading criteria met
2. 6+ months walk-forward with 99% confidence
3. Live spot trading only (no leverage)
4. Max position 1% of account per trade
5. Daily loss limit 2% of account
6. All fee filters enabled (Phase 2-5)
"""

# Example integration test
def example_walk_forward_test():
    """Example of running walk-forward validation"""
    import pandas as pd
    from backtesting.engine import WalkForwardValidator, BacktestEngine
    from knowledge.strategy_library import get_strategy
    from optimization.optuna_optimizer import StrategyOptimizer
    from core.feature_engine import FeatureConfig
    from config.settings import TradingConfig
    
    # Load configuration
    trading_config = TradingConfig()
    feature_config = FeatureConfig(
        tick_size=trading_config.tick_size,
        regime_change_threshold=0.15,
        minimum_ticks_between_patterns=20
    )
    
    # Load data (use filtered data from Phase 3)
    df = pd.read_parquet('data/recorded/xrp_filtered.parquet')
    
    # Get strategy
    strategy = get_strategy('absorption')
    if not strategy:
        raise ValueError("Absorption strategy not found")
    
    # Create optimizer
    optimizer = StrategyOptimizer(
        strategy_name='absorption',
        storage="sqlite:///optuna_studies.db"
    )
    
    # Create walk-forward validator
    wfv = WalkForwardValidator(
        train_days=60,
        test_days=14,
        step_days=7,
        min_oos_trades=10
    )
    
    # Run validation
    print("Starting walk-forward validation...")
    results = wfv.validate(
        data=df,
        strategy=strategy,
        optimizer=optimizer,
        n_trials_per_fold=50,
        feature_config=feature_config
    )
    
    # Analyze results
    summary = results['summary']
    print("\n" + "="*60)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("="*60)
    print(f"Total folds: {summary['total_folds']}")
    print(f"Valid folds: {summary['valid_folds']}")
    print(f"Skipped folds: {summary['skipped_folds']}")
    print(f"Avg OOS Sharpe: {summary['avg_oos_sharpe']:.3f}")
    print(f"Profitable folds: {summary['pct_profitable_folds']*100:.1f}%")
    print("="*60)
    
    # Check Phase 6 success criteria
    success = True
    if summary['avg_oos_sharpe'] < 0.5:
        print("❌ FAIL: Avg OOS Sharpe < 0.5")
        success = False
    else:
        print("✓ PASS: Avg OOS Sharpe >= 0.5")
    
    if summary['pct_profitable_folds'] < 0.5:
        print("❌ FAIL: Profitable folds < 50%")
        success = False
    else:
        print("✓ PASS: Profitable folds >= 50%")
    
    # Check parameter stability
    max_cv = max(
        (stats['cv'] for stats in results['params_stability'].values()),
        default=0.0
    )
    if max_cv > 0.3:
        print(f"⚠ WARNING: Parameter CV {max_cv:.3f} > 0.3 (unstable)")
    else:
        print("✓ PASS: Parameter stability (CV < 0.3)")
    
    return results if success else None


if __name__ == "__main__":
    # Uncomment to run example:
    # results = example_walk_forward_test()
    pass
