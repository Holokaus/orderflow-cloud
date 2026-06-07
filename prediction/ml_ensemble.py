"""
ML Prediction Layer - Ensemble
===============================
PHASE 5 ENHANCEMENT

Ensemble of 3 ML models for price direction prediction:
1. XGBoost (40% weight) - fast, gradient boosting
2. LightGBM (40% weight) - even faster, histogram-based boosting
3. LSTM (20% weight) - captures temporal patterns

Training:
- Use filtered historical data (Phase 3)
- Features: 100+ from FeatureEngine
- Target: binary classification (profitable move or not)
- Walk-forward: train month N, validate month N+1
- Retrain weekly

Integration:
- Add ML prediction as additional signal filter
- Only trade if rule-based signal AND ML confidence > 0.7
- Log predictions for analysis and retraining

Confidence calculation:
- Consensus between models (lower std = higher confidence)
- Example: [0.6, 0.58, 0.59] from [XGB, LGBM, LSTM] → ensemble=0.59, confidence=0.99
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import pickle
from pathlib import Path


class ModelType(Enum):
    """Supported model types"""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LSTM = "lstm"


@dataclass
class PredictionResult:
    """Result from a single model prediction"""
    model_type: ModelType
    prob_up: float  # Probability of price going up
    prob_down: float  # Probability of price going down
    confidence: float  # Confidence in prediction (0.0-1.0)
    prediction_time: datetime = None
    metadata: Dict = None  # Model-specific metadata
    
    def __post_init__(self):
        if self.prediction_time is None:
            self.prediction_time = datetime.now()


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction"""
    prob_up: float
    confidence: float  # Agreement between models
    individual_predictions: Dict[str, PredictionResult]  # model_name -> PredictionResult
    timestamp: datetime = None
    decision: str = "HOLD"  # BUY, SELL, HOLD
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            'prob_up': self.prob_up,
            'confidence': self.confidence,
            'decision': self.decision,
            'timestamp': self.timestamp.isoformat(),
            'individual': {
                name: {
                    'prob_up': pred.prob_up,
                    'confidence': pred.confidence,
                    'model': pred.model_type.value
                }
                for name, pred in self.individual_predictions.items()
            }
        }


class XGBoostPredictor:
    """XGBoost predictor wrapper"""
    
    def __init__(self):
        """Initialize XGBoost predictor"""
        self.model = None
        self.model_path = None
        logger.debug("[XGBoostPredictor] Initialized")
    
    def load_model(self, path: str) -> None:
        """Load pre-trained model"""
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(path)
            self.model_path = path
            logger.info(f"[XGBoostPredictor] Loaded model from {path}")
        except Exception as e:
            logger.error(f"[XGBoostPredictor] Failed to load model: {e}")
            self.model = None
    
    def predict(self, features: Dict) -> PredictionResult:
        """
        Predict price direction.
        
        Args:
            features: Feature dictionary with numeric values
        
        Returns:
            PredictionResult
        """
        if self.model is None:
            logger.warning("[XGBoostPredictor] Model not loaded, returning neutral prediction")
            return PredictionResult(
                model_type=ModelType.XGBOOST,
                prob_up=0.5,
                prob_down=0.5,
                confidence=0.0
            )
        
        try:
            # Convert features dict to feature array (order must match training)
            feature_names = sorted(features.keys())
            X = np.array([[features[name] for name in feature_names]])
            
            # Get probability
            probs = self.model.predict_proba(X)[0]
            prob_up = probs[1]
            prob_down = probs[0]
            
            # Confidence is how far from 50-50
            confidence = abs(prob_up - 0.5) * 2  # 0.5 -> 0.0 confidence, 1.0 -> 1.0 confidence
            
            return PredictionResult(
                model_type=ModelType.XGBOOST,
                prob_up=prob_up,
                prob_down=prob_down,
                confidence=confidence
            )
        
        except Exception as e:
            logger.error(f"[XGBoostPredictor] Prediction error: {e}")
            return PredictionResult(
                model_type=ModelType.XGBOOST,
                prob_up=0.5,
                prob_down=0.5,
                confidence=0.0
            )


class LightGBMPredictor:
    """LightGBM predictor wrapper"""
    
    def __init__(self):
        """Initialize LightGBM predictor"""
        self.model = None
        self.model_path = None
        logger.debug("[LightGBMPredictor] Initialized")
    
    def load_model(self, path: str) -> None:
        """Load pre-trained model"""
        try:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=path)
            self.model_path = path
            logger.info(f"[LightGBMPredictor] Loaded model from {path}")
        except Exception as e:
            logger.error(f"[LightGBMPredictor] Failed to load model: {e}")
            self.model = None
    
    def predict(self, features: Dict) -> PredictionResult:
        """Predict price direction"""
        if self.model is None:
            logger.warning("[LightGBMPredictor] Model not loaded, returning neutral prediction")
            return PredictionResult(
                model_type=ModelType.LIGHTGBM,
                prob_up=0.5,
                prob_down=0.5,
                confidence=0.0
            )
        
        try:
            feature_names = sorted(features.keys())
            X = np.array([[features[name] for name in feature_names]])
            
            probs = self.model.predict(X)[0]
            prob_up = probs[1] if len(probs) > 1 else probs[0]
            prob_down = 1 - prob_up
            
            confidence = abs(prob_up - 0.5) * 2
            
            return PredictionResult(
                model_type=ModelType.LIGHTGBM,
                prob_up=prob_up,
                prob_down=prob_down,
                confidence=confidence
            )
        
        except Exception as e:
            logger.error(f"[LightGBMPredictor] Prediction error: {e}")
            return PredictionResult(
                model_type=ModelType.LIGHTGBM,
                prob_up=0.5,
                prob_down=0.5,
                confidence=0.0
            )


class LSTMPredictor:
    """LSTM predictor wrapper for temporal patterns"""
    
    def __init__(self, sequence_length: int = 60):
        """
        Initialize LSTM predictor.
        
        Args:
            sequence_length: Number of past timesteps to consider
        """
        self.model = None
        self.model_path = None
        self.sequence_length = sequence_length
        logger.debug(f"[LSTMPredictor] Initialized with sequence_length={sequence_length}")
    
    def load_model(self, path: str) -> None:
        """Load pre-trained model"""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(path)
            self.model_path = path
            logger.info(f"[LSTMPredictor] Loaded model from {path}")
        except Exception as e:
            logger.error(f"[LSTMPredictor] Failed to load model: {e}")
            self.model = None
    
    def predict(self, features: Dict, sequence: Optional[np.ndarray] = None) -> PredictionResult:
        """
        Predict price direction.
        
        Args:
            features: Feature dict (not used for LSTM, kept for interface consistency)
            sequence: Optional sequence array of shape (sequence_length, n_features)
        
        Returns:
            PredictionResult
        """
        if self.model is None:
            logger.warning("[LSTMPredictor] Model not loaded, returning neutral prediction")
            return PredictionResult(
                model_type=ModelType.LSTM,
                prob_up=0.5,
                prob_down=0.5,
                confidence=0.0
            )
        
        try:
            if sequence is None:
                logger.warning("[LSTMPredictor] No sequence provided")
                return PredictionResult(
                    model_type=ModelType.LSTM,
                    prob_up=0.5,
                    prob_down=0.5,
                    confidence=0.0
                )
            
            # Ensure shape (1, sequence_length, n_features) for batch prediction
            if len(sequence.shape) == 2:
                sequence = np.expand_dims(sequence, axis=0)
            
            probs = self.model.predict(sequence, verbose=0)[0]
            prob_up = probs[1] if len(probs) > 1 else probs[0]
            prob_down = 1 - prob_up
            
            confidence = abs(prob_up - 0.5) * 2
            
            return PredictionResult(
                model_type=ModelType.LSTM,
                prob_up=prob_up,
                prob_down=prob_down,
                confidence=confidence
            )
        
        except Exception as e:
            logger.error(f"[LSTMPredictor] Prediction error: {e}")
            return PredictionResult(
                model_type=ModelType.LSTM,
                prob_up=0.5,
                prob_down=0.5,
                confidence=0.0
            )


class MLEnsemble:
    """
    Ensemble predictor combining XGBoost, LightGBM, and LSTM.
    
    Weights:
    - XGBoost: 40%
    - LightGBM: 40%
    - LSTM: 20%
    
    Confidence = 1 - std(probabilities)
    This rewards agreement between models.
    """
    
    def __init__(
        self,
        xgb_model_path: Optional[str] = None,
        lgb_model_path: Optional[str] = None,
        lstm_model_path: Optional[str] = None,
    ):
        """
        Initialize ensemble with optional pre-trained models.
        
        Args:
            xgb_model_path: Path to XGBoost model file
            lgb_model_path: Path to LightGBM model file
            lstm_model_path: Path to LSTM model file
        """
        self.xgb = XGBoostPredictor()
        self.lgb = LightGBMPredictor()
        self.lstm = LSTMPredictor()
        
        # Model weights
        self.weights = {
            ModelType.XGBOOST: 0.40,
            ModelType.LIGHTGBM: 0.40,
            ModelType.LSTM: 0.20,
        }
        
        # Load models if paths provided
        if xgb_model_path:
            self.xgb.load_model(xgb_model_path)
        if lgb_model_path:
            self.lgb.load_model(lgb_model_path)
        if lstm_model_path:
            self.lstm.load_model(lstm_model_path)
        
        logger.info(
            f"[MLEnsemble] Initialized with weights: "
            f"XGB={self.weights[ModelType.XGBOOST]}, "
            f"LGBM={self.weights[ModelType.LIGHTGBM]}, "
            f"LSTM={self.weights[ModelType.LSTM]}"
        )
    
    def predict(
        self,
        features: Dict,
        lstm_sequence: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.0
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction.
        
        Args:
            features: Feature dictionary
            lstm_sequence: Optional LSTM sequence
            confidence_threshold: Minimum confidence to return BUY/SELL (else HOLD)
        
        Returns:
            EnsemblePrediction
        
        Example:
            >>> ensemble = MLEnsemble()
            >>> pred = ensemble.predict(features={'feat1': 0.5, 'feat2': 0.3})
            >>> print(f"Decision: {pred.decision}, Confidence: {pred.confidence:.2f}")
        """
        # Get individual predictions
        xgb_pred = self.xgb.predict(features)
        lgb_pred = self.lgb.predict(features)
        lstm_pred = self.lstm.predict(features, lstm_sequence)
        
        # Store in dict
        individual_preds = {
            "xgboost": xgb_pred,
            "lightgbm": lgb_pred,
            "lstm": lstm_pred,
        }
        
        # Calculate weighted ensemble probability
        probs = [
            xgb_pred.prob_up * self.weights[ModelType.XGBOOST],
            lgb_pred.prob_up * self.weights[ModelType.LIGHTGBM],
            lstm_pred.prob_up * self.weights[ModelType.LSTM],
        ]
        ensemble_prob_up = sum(probs)
        
        # Confidence = agreement (1 - std of probabilities)
        all_probs = [xgb_pred.prob_up, lgb_pred.prob_up, lstm_pred.prob_up]
        prob_std = np.std(all_probs)
        confidence = 1.0 - prob_std
        
        # Decision logic
        if confidence < confidence_threshold:
            decision = "HOLD"
        elif ensemble_prob_up > 0.6:
            decision = "BUY"
        elif ensemble_prob_up < 0.4:
            decision = "SELL"
        else:
            decision = "HOLD"
        
        ensemble_pred = EnsemblePrediction(
            prob_up=ensemble_prob_up,
            confidence=confidence,
            individual_predictions=individual_preds,
            decision=decision
        )
        
        logger.debug(
            f"[MLEnsemble] Prediction: {ensemble_pred.decision}, "
            f"prob_up={ensemble_prob_up:.3f}, "
            f"confidence={confidence:.3f} "
            f"(XGB={xgb_pred.prob_up:.3f}, LGBM={lgb_pred.prob_up:.3f}, LSTM={lstm_pred.prob_up:.3f})"
        )
        
        return ensemble_pred
    
    def should_trade(
        self,
        prediction: EnsemblePrediction,
        min_confidence: float = 0.7,
        bias: str = "neutral"  # "bullish", "bearish", "neutral"
    ) -> Tuple[bool, str]:
        """
        Determine if trade should proceed based on ensemble prediction.
        
        Args:
            prediction: EnsemblePrediction from predict()
            min_confidence: Minimum confidence required (0.0-1.0)
            bias: Directional bias ("bullish"=only longs, "bearish"=only shorts, "neutral"=both)
        
        Returns:
            Tuple[bool, str]: (should_trade, reason)
        """
        if prediction.confidence < min_confidence:
            return False, (
                f"Confidence {prediction.confidence:.2f} < threshold {min_confidence:.2f}"
            )
        
        if bias == "bullish" and prediction.decision != "BUY":
            return False, f"Bullish bias but decision is {prediction.decision}"
        
        if bias == "bearish" and prediction.decision != "SELL":
            return False, f"Bearish bias but decision is {prediction.decision}"
        
        if prediction.decision == "HOLD":
            return False, f"Ensemble decision is HOLD"
        
        return True, (
            f"ML ensemble {prediction.decision} signal. "
            f"Confidence: {prediction.confidence:.2f}, "
            f"prob_up: {prediction.prob_up:.3f}"
        )


class EnsembleTrainingPipeline:
    """
    Pipeline for training/retraining ensemble models.
    
    Typical usage:
    - Week 1: Train on month 0 historical data, validate on month 1
    - Week 2-4: Use trained models in production
    - Week 5: Retrain on new data (months 1-2), validate on month 3
    - Repeat weekly or monthly
    """
    
    def __init__(self, checkpoint_dir: str = "models/ml_ensemble"):
        """
        Initialize training pipeline.
        
        Args:
            checkpoint_dir: Directory to save trained models
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training_log = []
        
        logger.info(f"[EnsembleTrainingPipeline] Checkpoint dir: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        ensemble: MLEnsemble,
        metrics: Dict,
        tag: str = None
    ) -> str:
        """
        Save trained ensemble checkpoint.
        
        Args:
            ensemble: Trained MLEnsemble
            metrics: Training metrics (Sharpe, win_rate, etc.)
            tag: Optional tag for checkpoint (default: timestamp)
        
        Returns:
            Path to checkpoint
        """
        if tag is None:
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint_path = self.checkpoint_dir / f"ensemble_{tag}.pkl"
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'ensemble': ensemble,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            logger.info(f"[EnsembleTrainingPipeline] Saved checkpoint to {checkpoint_path}")
            return str(checkpoint_path)
        
        except Exception as e:
            logger.error(f"[EnsembleTrainingPipeline] Failed to save checkpoint: {e}")
            return None
    
    def log_training_result(
        self,
        model_type: ModelType,
        train_metrics: Dict,
        val_metrics: Dict
    ) -> None:
        """Log training results for analysis"""
        self.training_log.append({
            'model': model_type.value,
            'timestamp': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        })
        
        logger.info(
            f"[EnsembleTrainingPipeline] {model_type.value} training: "
            f"train_sharpe={train_metrics.get('sharpe', 'N/A')}, "
            f"val_sharpe={val_metrics.get('sharpe', 'N/A')}"
        )
