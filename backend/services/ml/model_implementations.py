"""Implementation of specific ML models."""

import pandas as pd
import numpy as np
from typing import Dict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from .models import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def create_model(self) -> object:
        """Create Random Forest model."""
        if self.model_type == 'classification':
            return RandomForestClassifier(**self.model_params)
        else:
            return RandomForestRegressor(**self.model_params)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names)

class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def create_model(self) -> object:
        """Create XGBoost model."""
        if self.model_type == 'classification':
            return XGBClassifier(**self.model_params)
        else:
            return XGBRegressor(**self.model_params)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names)

class LightGBMModel(BaseModel):
    """LightGBM model implementation."""
    
    def create_model(self) -> object:
        """Create LightGBM model."""
        if self.model_type == 'classification':
            return LGBMClassifier(**self.model_params)
        else:
            return LGBMRegressor(**self.model_params)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names)

class LinearModel(BaseModel):
    """Linear model implementation."""
    
    def create_model(self) -> object:
        """Create linear model."""
        if self.model_type == 'classification':
            return LogisticRegression(**self.model_params)
        else:
            return LinearRegression(**self.model_params)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
            if importance.ndim > 1:
                importance = importance.mean(axis=0)
            return pd.Series(importance, index=self.feature_names)
        else:
            return pd.Series(0, index=self.feature_names)

def create_model(
    name: str,
    model_type: str,
    model_class: str,
    model_params: Dict = None
) -> BaseModel:
    """Factory function to create model instances.
    
    Args:
        name: Model name
        model_type: Type of model (classification/regression)
        model_class: Class of model to create
        model_params: Model hyperparameters
        
    Returns:
        Model instance
    """
    model_classes = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'linear': LinearModel
    }
    
    if model_class not in model_classes:
        raise ValueError(f"Unknown model class: {model_class}")
        
    model_cls = model_classes[model_class]
    return model_cls(name, model_type, model_params)
