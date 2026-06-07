"""
ML Ensemble Training Script
===========================
PHASE 5 ENHANCEMENT

Trains XGBoost, LightGBM, and LSTM models on recorded market data.
Saves models to the directory configured in settings.ml_ensemble.model_dir.

Usage:
    python -m scripts.train_ml_models --start 2026-03-01 --end 2026-03-28

Requirements (pip install):
    xgboost lightgbm tensorflow pandas numpy scikit-learn
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger
import pandas as pd


def check_dependencies() -> bool:
    """Check that ML dependencies are installed."""
    missing = []
    try:
        import xgboost
    except ImportError:
        missing.append("xgboost")
    try:
        import lightgbm
    except ImportError:
        missing.append("lightgbm")
    try:
        import tensorflow
    except ImportError:
        missing.append("tensorflow")
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def load_training_data(start_date: datetime, end_date: datetime):
    """Load and prepare training data from recorded parquet files."""
    import pandas as pd
    from data.data_recorder import DataRecorder

    recorder = DataRecorder()
    data = recorder.load_recorded_data(start_date, end_date, 'trades')
    if data.empty:
        # Fall back to any parquet files
        parquet_dir = Path("./data/recorded")
        files = sorted(parquet_dir.glob("*_market_data_*.parquet"))
        if files:
            logger.info(f"Loading parquet file: {files[-1]}")
            data = pd.read_parquet(files[-1])
            # Filter by date range
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data[
                    (data['timestamp'] >= pd.Timestamp(start_date)) &
                    (data['timestamp'] < pd.Timestamp(end_date))
                ]
    return data


def prepare_features(data: pd.DataFrame, n_lags: int = 5):
    """Create feature matrix and target from raw market data."""
    import numpy as np
    import pandas as pd

    df = data.copy()

    # Price changes
    if 'price' in df.columns:
        df['ret_1'] = df['price'].pct_change()
    for lag in range(1, n_lags + 1):
        if 'price' in df.columns:
            df[f'ret_lag_{lag}'] = df['price'].pct_change(lag)

    # Volume features
    if 'volume' in df.columns:
        df['log_volume'] = np.log1p(df['volume'])
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5'].shift(1)

    # Time features
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        df['hour'] = ts.dt.hour
        df['dayofweek'] = ts.dt.dayofweek

    # Drop NaN rows from lag creation
    df = df.dropna()

    # Target: 1 if price increased over next N ticks, else 0
    forecast_horizon = 5
    if 'price' in df.columns:
        df['target'] = (df['price'].shift(-forecast_horizon) > df['price']).astype(int)
        df = df.dropna()

    # Separate features and target
    exclude_cols = {'target', 'timestamp', 'price', 'side'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not feature_cols:
        logger.error("No feature columns could be created from the data.")
        return None, None, None

    X = df[feature_cols].values
    y = df['target'].values if 'target' in df.columns else None
    feature_names = feature_cols

    return X, y, feature_names


def train_xgboost(X_train, y_train, X_val, y_val, feature_names, output_dir: Path):
    """Train and save XGBoost model."""
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    model_path = output_dir / "xgboost.json"
    model.save_model(str(model_path))
    logger.info(f"XGBoost model saved to {model_path}")
    return model_path


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names, output_dir: Path):
    """Train and save LightGBM model."""
    import lightgbm as lgb
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=200,
    )
    model_path = output_dir / "lightgbm.txt"
    model.save_model(str(model_path))
    logger.info(f"LightGBM model saved to {model_path}")
    return model_path


def train_lstm(X_train, y_train, X_val, y_val, output_dir: Path):
    """Train and save LSTM model (requires tensorflow)."""
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    # Reshape for LSTM: (samples, timesteps, features)
    timesteps = 10
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0] - timesteps + 1
    if n_samples < 100:
        logger.warning("Too few samples for LSTM training, skipping.")
        return None

    X_lstm = np.array([X_train[i:i + timesteps] for i in range(n_samples)])
    y_lstm = y_train[timesteps - 1:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(timesteps, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    model_path = output_dir / "lstm.keras"
    model.save(str(model_path))
    logger.info(f"LSTM model saved to {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train ML ensemble models")
    parser.add_argument('--start', type=str, default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='models/ml_ensemble',
                        help='Directory to save trained models')
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ML ENSEMBLE TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Date range : {start_date.date()} to {end_date.date()}")
    print(f"  Output dir : {output_dir.resolve()}")
    print("-" * 60)

    if not check_dependencies():
        sys.exit(1)

    # 1. Load data
    print("\n[1/4] Loading training data...")
    data = load_training_data(start_date, end_date)
    if data.empty:
        logger.error(f"No data found between {start_date.date()} and {end_date.date()}")
        print("  ✗ No data available. Record data first with: python main.py record")
        sys.exit(1)
    print(f"  ✓ Loaded {len(data):,} rows")

    # 2. Prepare features
    print("\n[2/4] Preparing features...")
    result = prepare_features(data)
    if result is None or result[0] is None:
        sys.exit(1)
    X, y, feature_names = result
    print(f"  ✓ {X.shape[1]} features, {X.shape[0]} samples")

    # 3. Train/validation split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")

    # 4. Train models
    print("\n[3/4] Training models...")
    models = []

    print("  Training XGBoost...")
    models.append(("xgboost", train_xgboost(X_train, y_train, X_val, y_val, feature_names, output_dir)))

    print("  Training LightGBM...")
    models.append(("lightgbm", train_lightgbm(X_train, y_train, X_val, y_val, feature_names, output_dir)))

    print("  Training LSTM...")
    models.append(("lstm", train_lstm(X_train, y_train, X_val, y_val, output_dir)))

    # 5. Summary
    print("\n[4/4] Training complete!")
    print("-" * 60)
    for name, path in models:
        if path:
            print(f"  ✓ {name:10s}  {path}")
        else:
            print(f"  ✗ {name:10s}  skipped (see logs)")
    print("=" * 60)

    print("\nTo use these models, set in .env or settings:")
    print(f"  ML_ENSEMBLE_ENABLED=true")
    print(f"  ML_ENSEMBLE_MODEL_DIR={args.output_dir}")
    print()


if __name__ == "__main__":
    main()
