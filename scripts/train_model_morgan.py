"""
Train and evaluate XGBoost model using Morgan fingerprint features for drug permeability prediction.

This script implements a binary classification model for predicting drug permeability
using Morgan fingerprint features. It includes data loading, model training with early
stopping, comprehensive evaluation metrics, and visualization generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    roc_curve
)
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('data/logs/model_training.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for XGBoost model training and evaluation."""
    # Data paths
    data_dir: Path = Path('data')
    model_dir: Path = Path('models/morgan')
    viz_dir: Path = Path('data/visualizations_morgan')
    
    # Model parameters
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'random_state': 42
    })
    
    # Training parameters
    num_rounds: int = 1000
    early_stopping_rounds: int = 50
    verbose_eval: int = 100
    
    # Validation thresholds
    nan_threshold: float = 0.01  # Max allowed fraction of NaN values
    inf_threshold: float = 0.01  # Max allowed fraction of infinite values
    zero_threshold: float = 0.95  # Warning threshold for zero values
    
    # Model metadata
    model_version: str = '1.0.0'
    feature_type: str = 'morgan_fingerprints'
    
    def __post_init__(self):
        """Create necessary directories."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

def validate_features(features: np.ndarray, split: str, config: ModelConfig) -> None:
    """Validate feature array for potential issues.
    
    Args:
        features: Feature array to validate
        split: Name of the data split
        config: Configuration object with thresholds
        
    Raises:
        ValueError: If validation fails
    """
    # Check for NaN values
    nan_frac = np.isnan(features).mean()
    if nan_frac > config.nan_threshold:
        raise ValueError(f"{split} features contain {nan_frac:.1%} NaN values (threshold: {config.nan_threshold:.1%})")
    
    # Check for infinite values
    inf_frac = np.isinf(features).mean()
    if inf_frac > config.inf_threshold:
        raise ValueError(f"{split} features contain {inf_frac:.1%} infinite values (threshold: {config.inf_threshold:.1%})")
    
    # Check for zero values
    zero_frac = (features == 0).mean()
    if zero_frac > config.zero_threshold:
        logger.warning(f"{split} features contain {zero_frac:.1%} zero values (threshold: {config.zero_threshold:.1%})")

def load_data(config: ModelConfig) -> Dict[str, Dict[str, np.ndarray]]:
    """Load Morgan fingerprint features and labels.
    
    Args:
        config: Configuration object with data paths
        
    Returns:
        Dictionary containing features and labels for each split
        
    Raises:
        FileNotFoundError: If data files are missing
        ValueError: If data validation fails
    """
    data = {}
    
    for split in ['train', 'valid', 'test']:
        try:
            # Load features
            features_path = config.data_dir / 'features' / f'caco2_{split}_features.npy'
            features = np.load(features_path)
            
            # Load labels
            labels_path = config.data_dir / 'processed' / f'caco2_{split}.csv'
            df = pd.read_csv(labels_path)
            
            if 'Binary' not in df.columns:
                raise ValueError(f"'Binary' column not found in {labels_path}")
                
            labels = df['Binary'].values
            
            # Validate features
            validate_features(features, split, config)
            
            # Store data
            data[split] = {
                'features': features,
                'labels': labels
            }
            
            logger.info(f"Loaded {split} set: {len(labels)} samples, {features.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Error loading {split} data: {e}")
            raise
    
    return data

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    config: ModelConfig
) -> xgb.Booster:
    """Train XGBoost model with cross-validation and early stopping.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        config: Configuration object with model parameters
        
    Returns:
        Trained XGBoost model
        
    Raises:
        ValueError: If input shapes are inconsistent
    """
    # Validate input shapes
    if len(X_train) != len(y_train):
        raise ValueError(f"Training shapes mismatch: X={X_train.shape}, y={y_train.shape}")
    if len(X_valid) != len(y_valid):
        raise ValueError(f"Validation shapes mismatch: X={X_valid.shape}, y={y_valid.shape}")
    if X_train.shape[1] != X_valid.shape[1]:
        raise ValueError(f"Feature dimension mismatch: train={X_train.shape[1]}, valid={X_valid.shape[1]}")
    
    try:
        # Create DMatrix for faster training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        
        # Set up evaluation list
        evallist = [(dtrain, 'train'), (dvalid, 'valid')]
        
        # Train model with early stopping
        logger.info("Starting model training...")
        model = xgb.train(
            config.model_params,
            dtrain,
            config.num_rounds,
            evals=evallist,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=config.verbose_eval
        )
        
        logger.info(f"Training completed at iteration {model.best_iteration}")
        return model
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path
) -> None:
    """Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Path to save the plot
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Low', 'High'],
            yticklabels=['Low', 'High']
        )
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")

def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str,
    save_path: Path
) -> None:
    """Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        save_path: Path to save the plot
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {title}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to plot ROC curve: {e}")

def evaluate_model(
    model: xgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    config: ModelConfig
) -> Dict[str, float]:
    """Evaluate model performance and generate visualizations.
    
    Args:
        model: Trained XGBoost model
        X: Features to evaluate
        y: True labels
        split_name: Name of the data split
        config: Configuration object
        
    Returns:
        Dictionary of evaluation metrics
        
    Raises:
        ValueError: If input shapes are inconsistent
    """
    if len(X) != len(y):
        raise ValueError(f"Input shapes mismatch: X={X.shape}, y={y.shape}")
        
    try:
        # Convert to DMatrix for prediction
        dmat = xgb.DMatrix(X)
        
        # Get predictions
        y_pred_proba = model.predict(dmat)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred)),
            'recall': float(recall_score(y, y_pred)),
            'f1': float(f1_score(y, y_pred)),
            'roc_auc': float(roc_auc_score(y, y_pred_proba)),
            'avg_precision': float(average_precision_score(y, y_pred_proba))
        }
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y, y_pred,
            f'Morgan_{split_name}',
            config.viz_dir / f'confusion_matrix_{split_name}.png'
        )
        
        # Plot ROC curve
        plot_roc_curve(
            y, y_pred_proba,
            f'Morgan_{split_name}',
            config.viz_dir / f'roc_curve_{split_name}.png'
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed for {split_name} set: {e}")
        raise

def main() -> None:
    """Main training and evaluation pipeline."""
    try:
        config = ModelConfig()
        
        # Load data
        data = load_data(config)
        
        # Train model
        logger.info("Training XGBoost model with Morgan features...")
        model = train_model(
            data['train']['features'],
            data['train']['labels'],
            data['valid']['features'],
            data['valid']['labels'],
            config
        )
        
        # Save model bundle
        model_info = {
            'model_version': config.model_version,
            'training_date': datetime.now().isoformat(),
            'feature_dim': data['train']['features'].shape[1],
            'num_train_samples': len(data['train']['labels']),
            'feature_type': config.feature_type,
            'model_params': model.save_config()
        }
        
        # Save model files
        model.save_model(str(config.model_dir / 'xgboost_model.json'))  # Model weights
        with open(config.model_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)  # Model metadata
        
        logger.info("Model and metadata saved successfully!")
        
        # Evaluate model
        results = {'xgboost': {}}
        logger.info("Evaluating model performance...")
        
        for split in ['train', 'valid', 'test']:
            logger.info(f"Evaluating {split} set...")
            results['xgboost'][split] = evaluate_model(
                model,
                data[split]['features'],
                data[split]['labels'],
                split,
                config
            )
            logger.info(f"{split} set metrics: {results['xgboost'][split]}")
        
        # Save results
        with open(config.viz_dir / 'model_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
