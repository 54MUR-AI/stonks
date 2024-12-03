"""AutoML tools for automated model selection and optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from .models import BaseModel
from .model_implementations import create_model
from .feature_selection import FeatureSelector
from .feature_engineering import FeatureEngineer

@dataclass
class AutoMLResult:
    """Results from AutoML optimization."""
    best_model: BaseModel
    best_params: Dict
    best_features: List[str]
    feature_importance: Dict[str, float]
    cv_scores: Dict[str, List[float]]
    optimization_history: pd.DataFrame

class AutoML:
    """Automated machine learning pipeline."""
    
    def __init__(
        self,
        task: str = 'classification',
        metric: Optional[str] = None,
        n_trials: int = 100,
        cv_splits: int = 5,
        feature_selection: bool = True
    ):
        """Initialize AutoML.
        
        Args:
            task: Either 'classification' or 'regression'
            metric: Metric to optimize (default: 'f1' for classification, 'rmse' for regression)
            n_trials: Number of optimization trials
            cv_splits: Number of cross-validation splits
            feature_selection: Whether to perform feature selection
        """
        self.task = task
        self.metric = metric or ('f1' if task == 'classification' else 'rmse')
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.feature_selection = feature_selection
        
        # Initialize components
        self.feature_selector = FeatureSelector()
        self.feature_engineer = FeatureEngineer()
        
    def _get_model_search_space(self, trial: optuna.Trial) -> Dict:
        """Define model search space for optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of model parameters
        """
        # Choose model type
        model_class = trial.suggest_categorical('model_class', [
            'random_forest',
            'xgboost',
            'lightgbm'
        ])
        
        # Common parameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'random_state': 42
        }
        
        # Model-specific parameters
        if model_class == 'random_forest':
            params.update({
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            })
        elif model_class == 'xgboost':
            params.update({
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True)
            })
        elif model_class == 'lightgbm':
            params.update({
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0)
            })
            
        return model_class, params
    
    def _evaluate_model(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[float, Dict[str, List[float]]]:
        """Evaluate model using cross-validation.
        
        Args:
            model: Model to evaluate
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Tuple of (mean metric score, dictionary of metric scores)
        """
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        } if self.task == 'classification' else {
            'mse': [], 'rmse': [], 'mae': [], 'r2': []
        }
        
        for train_idx, val_idx in cv.split(X):
            # Split data
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            
            # Train and predict
            model.train(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            if self.task == 'classification':
                scores['accuracy'].append(accuracy_score(y_val, y_pred))
                scores['precision'].append(precision_score(y_val, y_pred))
                scores['recall'].append(recall_score(y_val, y_pred))
                scores['f1'].append(f1_score(y_val, y_pred))
            else:
                mse = mean_squared_error(y_val, y_pred)
                scores['mse'].append(mse)
                scores['rmse'].append(np.sqrt(mse))
                scores['mae'].append(mean_absolute_error(y_val, y_pred))
                scores['r2'].append(r2_score(y_val, y_pred))
                
        # Return mean of target metric
        target_scores = scores[self.metric]
        return np.mean(target_scores), scores
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_sets: Optional[List[str]] = None
    ) -> AutoMLResult:
        """Run AutoML optimization.
        
        Args:
            X: Feature DataFrame
            y: Target series
            feature_sets: Optional list of feature sets to generate
            
        Returns:
            AutoMLResult with best model and optimization results
        """
        # Generate additional features if requested
        if feature_sets:
            X = self.feature_engineer.generate_features(X, feature_sets)
            
        # Perform feature selection if enabled
        if self.feature_selection:
            selection_result = self.feature_selector.ensemble_selection(
                X, y, task=self.task
            )
            X = X[selection_result.selected_features]
            
        # Initialize optimization history
        history = []
        
        def objective(trial):
            # Get model parameters
            model_class, params = self._get_model_search_space(trial)
            
            # Create and evaluate model
            model = create_model(
                name=f'automl_{model_class}',
                model_type=self.task,
                model_class=model_class,
                model_params=params
            )
            
            mean_score, cv_scores = self._evaluate_model(model, X, y)
            
            # Record trial
            history.append({
                'trial': trial.number,
                'model_class': model_class,
                'params': params,
                'score': mean_score,
                **{f'cv_{k}': np.mean(v) for k, v in cv_scores.items()}
            })
            
            return mean_score if self.metric != 'rmse' else -mean_score
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize' if self.metric != 'rmse' else 'minimize'
        )
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get best model
        model_class, params = self._get_model_search_space(study.best_trial)
        best_model = create_model(
            name=f'automl_{model_class}',
            model_type=self.task,
            model_class=model_class,
            model_params=params
        )
        
        # Train best model on full dataset
        best_model.train(X, y)
        
        # Get feature importance
        importance = best_model.get_feature_importance()
        
        return AutoMLResult(
            best_model=best_model,
            best_params=params,
            best_features=list(X.columns),
            feature_importance=importance,
            cv_scores=self._evaluate_model(best_model, X, y)[1],
            optimization_history=pd.DataFrame(history)
        )
