"""
Factor prediction service using machine learning models
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FactorPredictor:
    def __init__(self,
                 model_type: str = 'rf',  # 'rf' or 'gbm'
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 cv_splits: int = 5):
        """
        Initialize factor prediction model
        
        Args:
            model_type: Type of model ('rf' for Random Forest, 'gbm' for Gradient Boosting)
            n_estimators: Number of trees in the ensemble
            max_depth: Maximum depth of trees
            cv_splits: Number of cross-validation splits
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv_splits = cv_splits
        
        # Initialize models
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        
        self.scaler = StandardScaler()
        
    def prepare_features(self,
                        factor_returns: pd.DataFrame,
                        lookback: int = 21) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for factor prediction
        
        Args:
            factor_returns: DataFrame of factor returns
            lookback: Number of days to look back for features
            
        Returns:
            X: Feature matrix
            y: Target values
        """
        features = []
        targets = []
        
        for i in range(lookback, len(factor_returns)):
            # Historical returns
            hist_returns = factor_returns.iloc[i-lookback:i].values.flatten()
            
            # Rolling statistics
            rolling_mean = factor_returns.iloc[i-lookback:i].mean().values
            rolling_std = factor_returns.iloc[i-lookback:i].std().values
            rolling_skew = factor_returns.iloc[i-lookback:i].skew().values
            
            # Combine features
            feature_vector = np.concatenate([
                hist_returns,
                rolling_mean,
                rolling_std,
                rolling_skew
            ])
            
            features.append(feature_vector)
            targets.append(factor_returns.iloc[i].values)
        
        X = pd.DataFrame(features)
        y = pd.DataFrame(targets, columns=factor_returns.columns)
        
        return X, y
    
    def fit(self,
           factor_returns: pd.DataFrame,
           lookback: int = 21) -> Dict:
        """
        Fit prediction model using cross-validation
        
        Args:
            factor_returns: DataFrame of factor returns
            lookback: Number of days to look back for features
            
        Returns:
            Dictionary of cross-validation metrics
        """
        X, y = self.prepare_features(factor_returns, lookback)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series cross-validation
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        metrics = {
            'r2': [],
            'mse': [],
            'mae': []
        }
        
        for train_idx, test_idx in cv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit model
            self.model.fit(X_train, y_train)
            
            # Predict
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics['r2'].append(self.model.score(X_test, y_test))
            metrics['mse'].append(np.mean((y_test - y_pred) ** 2))
            metrics['mae'].append(np.mean(np.abs(y_test - y_pred)))
        
        # Calculate mean metrics
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def predict(self,
               factor_returns: pd.DataFrame,
               lookback: int = 21,
               horizon: int = 5) -> pd.DataFrame:
        """
        Generate factor return predictions
        
        Args:
            factor_returns: DataFrame of factor returns
            lookback: Number of days to look back for features
            horizon: Number of days to predict ahead
            
        Returns:
            DataFrame of predicted factor returns
        """
        if len(factor_returns) < lookback:
            raise ValueError(f"Not enough data for prediction. Need at least {lookback} periods.")
            
        predictions = []
        last_returns = factor_returns.iloc[-lookback-1:].copy()  # Include one extra period for feature calculation
        
        for i in range(horizon):
            # Prepare features
            X, _ = self.prepare_features(last_returns, lookback)
            
            # Ensure we have features for prediction
            if len(X) < 1:
                logger.warning(f"Not enough data for prediction at step {i}")
                continue
                
            X_scaled = self.scaler.transform(X.values)
            
            # Generate prediction
            pred = self.model.predict(X_scaled[-1:])
            predictions.append(pred[0])
            
            # Update returns for next prediction
            new_date = last_returns.index[-1] + pd.Timedelta(days=1)
            while new_date.weekday() > 4:  # Skip weekends
                new_date += pd.Timedelta(days=1)
                
            new_row = pd.DataFrame([pred[0]], columns=factor_returns.columns, index=[new_date])
            last_returns = pd.concat([last_returns, new_row])
        
        if not predictions:
            logger.warning("No predictions generated")
            return pd.DataFrame(columns=factor_returns.columns)
        
        # Create date range for predictions (business days only)
        pred_dates = pd.date_range(
            start=factor_returns.index[-1] + pd.Timedelta(days=1),
            periods=len(predictions),
            freq='B'
        )
        
        return pd.DataFrame(
            predictions,
            columns=factor_returns.columns,
            index=pred_dates
        )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Returns:
            DataFrame of feature importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(self.model.feature_importances_))],
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
