"""Machine learning models and training infrastructure."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import joblib
import json
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[pd.Series] = None
    training_time: float = 0.0
    
@dataclass
class ModelArtifacts:
    """Container for model artifacts."""
    model: object
    scaler: StandardScaler
    metadata: Dict
    metrics: ModelMetrics

class BaseModel(ABC):
    """Base class for all ML models."""
    
    def __init__(self, name: str, model_type: str, model_params: Dict = None):
        """Initialize model.
        
        Args:
            name: Model name
            model_type: Type of model (classification/regression)
            model_params: Model hyperparameters
        """
        self.name = name
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    @abstractmethod
    def create_model(self) -> object:
        """Create and return the underlying model object."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        pass
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            train_size: Proportion of data for training
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        split_idx = int(len(X) * train_size)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: float = 0.8
    ) -> ModelMetrics:
        """Train model and evaluate performance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            train_size: Proportion of data for training
            
        Returns:
            ModelMetrics with performance metrics
        """
        start_time = datetime.now()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, train_size)
        
        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        if self.model_type == 'regression':
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            conf_matrix = None
        else:
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred)
            }
            conf_matrix = confusion_matrix(y_test, y_pred)
            
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelMetrics(
            metrics=metrics,
            confusion_matrix=conf_matrix,
            feature_importance=feature_importance,
            training_time=training_time
        )
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate predictions for new data.
        
        Args:
            X: Feature DataFrame or array
            
        Returns:
            Array of predictions
        """
        if isinstance(X, pd.DataFrame):
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate probability predictions for new data.
        
        Args:
            X: Feature DataFrame or array
            
        Returns:
            Array of probability predictions
        """
        if self.model_type != 'classification':
            raise ValueError("Probability predictions only available for classification models")
            
        if isinstance(X, pd.DataFrame):
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        """Save model artifacts to disk.
        
        Args:
            path: Directory path to save artifacts
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, os.path.join(path, 'model.joblib'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
        
        # Save metadata
        metadata = {
            'name': self.name,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'created_at': datetime.now().isoformat()
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model artifacts from disk.
        
        Args:
            path: Directory path containing artifacts
            
        Returns:
            Loaded model instance
        """
        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        # Create model instance
        instance = cls(
            name=metadata['name'],
            model_type=metadata['model_type'],
            model_params=metadata['model_params']
        )
        
        # Load model and scaler
        instance.model = joblib.load(os.path.join(path, 'model.joblib'))
        instance.scaler = joblib.load(os.path.join(path, 'scaler.joblib'))
        instance.feature_names = metadata['feature_names']
        
        return instance
    
class ModelRegistry:
    """Registry for managing ML models."""
    
    def __init__(self, base_path: str):
        """Initialize registry.
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
    def save_model(
        self,
        model: BaseModel,
        artifacts: ModelArtifacts,
        version: str = None
    ):
        """Save model artifacts to registry.
        
        Args:
            model: Trained model instance
            artifacts: Model artifacts
            version: Model version (default: timestamp)
        """
        version = version or datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.base_path, model.name, version)
        
        # Save model artifacts
        model.save(model_path)
        
        # Save metrics
        metrics_path = os.path.join(model_path, 'metrics.json')
        metrics_dict = {
            'metrics': artifacts.metrics.metrics,
            'training_time': artifacts.metrics.training_time
        }
        
        if artifacts.metrics.confusion_matrix is not None:
            metrics_dict['confusion_matrix'] = artifacts.metrics.confusion_matrix.tolist()
            
        if artifacts.metrics.feature_importance is not None:
            metrics_dict['feature_importance'] = artifacts.metrics.feature_importance.to_dict()
            
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def load_model(self, model_name: str, version: str = 'latest') -> BaseModel:
        """Load model from registry.
        
        Args:
            model_name: Name of model to load
            version: Model version to load (default: latest)
            
        Returns:
            Loaded model instance
        """
        model_dir = os.path.join(self.base_path, model_name)
        
        if version == 'latest':
            versions = sorted(os.listdir(model_dir))
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            version = versions[-1]
            
        model_path = os.path.join(model_dir, version)
        if not os.path.exists(model_path):
            raise ValueError(f"Model version {version} not found for {model_name}")
            
        return BaseModel.load(model_path)
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all models in registry.
        
        Returns:
            Dictionary mapping model names to lists of versions
        """
        models = {}
        for model_name in os.listdir(self.base_path):
            model_dir = os.path.join(self.base_path, model_name)
            if os.path.isdir(model_dir):
                versions = sorted(os.listdir(model_dir))
                models[model_name] = versions
        return models
    
    def get_model_metrics(self, model_name: str, version: str) -> Dict:
        """Get metrics for a specific model version.
        
        Args:
            model_name: Name of model
            version: Model version
            
        Returns:
            Dictionary of model metrics
        """
        metrics_path = os.path.join(
            self.base_path,
            model_name,
            version,
            'metrics.json'
        )
        
        if not os.path.exists(metrics_path):
            raise ValueError(f"Metrics not found for {model_name} version {version}")
            
        with open(metrics_path, 'r') as f:
            return json.load(f)
