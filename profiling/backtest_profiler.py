"""Profile backtest engine to identify bottlenecks."""
import cProfile
import pstats
import io
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.engine import BacktestEngine
from core.feature_engine import FeatureConfig
from knowledge.strategy_library import get_strategy
from execution.risk_manager import RiskLimits


def find_data_file() -> str:
    """Find a parquet file with market data for profiling."""
    for p in sorted(Path("./data/recorded").glob("*_market_data_*.parquet")):
        return str(p)
    for p in sorted(Path("./data/backtests").glob("*.parquet")):
        return str(p)
    return None


def run_profile_backtest(duration_limit_sec: float = 30.0):
    """Run a short profiled backtest (1 hour or 30s wall time)."""
    data_path = find_data_file()
    if not data_path:
        logger.error("No data file found")
        return

    logger.info(f"Loading data from {data_path}")
    data = pd.read_parquet(data_path)
    n_total = len(data)
    n_use = min(n_total, 50000)  # Cap at 50K rows for profiling
    data = data.iloc[:n_use].reset_index(drop=True)
    logger.info(f"Using {len(data)} rows for profiling")

    feature_cfg = FeatureConfig()
    engine = BacktestEngine(
        initial_capital=100000.0,
        fee_pct=0.0004,
        slippage_pct=0.0005,
        feature_config=feature_cfg,
        risk_limits=RiskLimits(
            max_position_size=10000.0,
            max_position_value_pct=0.25,
        ),
    )

    strategy = get_strategy("absorption")

    profiler = cProfile.Profile()
    profiler.enable()
    engine.run(data, strategy)
    profiler.disable()

    profiler.dump_stats("profiling/backtest_profile.stats")

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumtime")
    stats.print_stats(30)
    output = stream.getvalue()
    print("\n=== TOP 30 FUNCTIONS BY CUMULATIVE TIME ===\n")
    print(output)

    with open("profiling/profile_output.txt", "w") as f:
        f.write(output)

    # Also save top 10 by tottime
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")
    stats2.print_stats(20)
    output2 = stream2.getvalue()
    print("\n=== TOP 20 FUNCTIONS BY TOTAL TIME ===\n")
    print(output2)

    with open("profiling/profile_tottime.txt", "w") as f:
        f.write(output2)

    elapsed = stats.total_tt
    ticks_per_sec = n_use / elapsed if elapsed > 0 else 0
    print(f"\n--- Summary ---")
    print(f"  Rows processed: {n_use}")
    print(f"  Profile duration: {elapsed:.2f}s")
    print(f"  Ticks/second: {ticks_per_sec:.0f}")
    print(f"  Estimated 24h ({24*3600} ticks): {24*3600/max(ticks_per_sec,1)/3600:.1f} hours")
    print(f"  Profile stats saved to profiling/profile_output.txt")


if __name__ == "__main__":
    run_profile_backtest()
