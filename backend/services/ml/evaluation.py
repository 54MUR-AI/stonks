"""Model evaluation and cross-validation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from .models import BaseModel, ModelMetrics

@dataclass
class CrossValidationResult:
    """Results from cross-validation."""
    metrics: List[Dict[str, float]]
    feature_importance: pd.DataFrame
    fold_indices: List[Tuple[np.ndarray, np.ndarray]]
    training_times: List[float]

class ModelEvaluator:
    """Utilities for model evaluation and cross-validation."""
    
    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: str
    ) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_type: Type of model (classification/regression)
            
        Returns:
            Dictionary of performance metrics
        """
        if model_type == 'regression':
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        else:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred),
                'auc_roc': roc_auc_score(y_true, y_pred)
            }
    
    @staticmethod
    def cross_validate(
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        gap: int = 0
    ) -> CrossValidationResult:
        """Perform time series cross-validation.
        
        Args:
            model: Model instance to evaluate
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits
            gap: Gap between train and test sets
            
        Returns:
            CrossValidationResult with metrics and metadata
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        metrics = []
        feature_importance = []
        fold_indices = []
        training_times = []
        
        for train_idx, test_idx in tscv.split(X):
            # Split data
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Train model and time it
            start_time = datetime.now()
            model.scaler.fit(X_train)
            X_train_scaled = model.scaler.transform(X_train)
            X_test_scaled = model.scaler.transform(X_test)
            
            model.model = model.create_model()
            model.model.fit(X_train_scaled, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Generate predictions
            y_pred = model.model.predict(X_test_scaled)
            
            # Calculate metrics
            fold_metrics = ModelEvaluator.evaluate_predictions(
                y_test,
                y_pred,
                model.model_type
            )
            
            # Get feature importance
            importance = model.get_feature_importance()
            
            metrics.append(fold_metrics)
            feature_importance.append(importance)
            fold_indices.append((train_idx, test_idx))
            training_times.append(training_time)
            
        # Aggregate feature importance across folds
        feature_importance_df = pd.concat(feature_importance, axis=1)
        feature_importance_df.columns = [f'fold_{i+1}' for i in range(n_splits)]
        
        return CrossValidationResult(
            metrics=metrics,
            feature_importance=feature_importance_df,
            fold_indices=fold_indices,
            training_times=training_times
        )
    
    @staticmethod
    def plot_cv_metrics(cv_result: CrossValidationResult, figsize: Tuple[int, int] = (10, 6)):
        """Plot cross-validation metrics.
        
        Args:
            cv_result: Results from cross-validation
            figsize: Figure size
        """
        metrics_df = pd.DataFrame(cv_result.metrics)
        
        plt.figure(figsize=figsize)
        sns.boxplot(data=metrics_df)
        plt.title('Cross-Validation Metrics Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
    @staticmethod
    def plot_feature_importance(
        cv_result: CrossValidationResult,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """Plot feature importance from cross-validation.
        
        Args:
            cv_result: Results from cross-validation
            top_n: Number of top features to show
            figsize: Figure size
        """
        mean_importance = cv_result.feature_importance.mean(axis=1)
        std_importance = cv_result.feature_importance.std(axis=1)
        
        # Sort features by mean importance
        sorted_idx = mean_importance.argsort()[-top_n:]
        pos = np.arange(top_n) + .5
        
        plt.figure(figsize=figsize)
        plt.barh(pos, mean_importance[sorted_idx])
        plt.yticks(pos, mean_importance.index[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance (Mean across CV folds)')
        plt.tight_layout()
        
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, figsize: Tuple[int, int] = (8, 6)):
        """Plot ROC curve for classification models.
        
        Args:
            y_true: True target values
            y_prob: Predicted probabilities
            figsize: Figure size
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot Precision-Recall curve for classification models.
        
        Args:
            y_true: True target values
            y_prob: Predicted probabilities
            figsize: Figure size
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.tight_layout()
        
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot confusion matrix for classification models.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            figsize: Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
    @staticmethod
    def plot_regression_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot predicted vs actual values for regression models.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predicted vs Actual Values')
        plt.grid(True)
        plt.tight_layout()
        
    @staticmethod
    def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot residuals for regression models.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            figsize: Figure size
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=figsize)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True)
        plt.tight_layout()
