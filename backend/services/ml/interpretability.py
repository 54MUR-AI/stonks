"""Model interpretability tools."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

from .models import BaseModel

class ModelInterpreter:
    """Tools for interpreting model predictions."""
    
    @staticmethod
    def calculate_shap_values(
        model: BaseModel,
        X: pd.DataFrame,
        sample_size: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Calculate SHAP values for feature importance.
        
        Args:
            model: Trained model instance
            X: Feature DataFrame
            sample_size: Number of samples to use (for large datasets)
            
        Returns:
            Tuple of (shap_values, feature_names)
        """
        # Sample data if needed
        if sample_size and len(X) > sample_size:
            X = X.sample(sample_size, random_state=42)
            
        # Scale features
        X_scaled = model.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Create explainer based on model type
        if hasattr(model.model, 'predict_proba'):
            explainer = shap.TreeExplainer(model.model) if hasattr(model.model, 'apply') else \
                       shap.KernelExplainer(model.model.predict_proba, X_scaled_df)
        else:
            explainer = shap.TreeExplainer(model.model) if hasattr(model.model, 'apply') else \
                       shap.KernelExplainer(model.model.predict, X_scaled_df)
            
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_scaled_df)
        
        # For binary classification, we only need one class's SHAP values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
            
        return shap_values, list(X.columns)
    
    @staticmethod
    def plot_shap_summary(
        shap_values: np.ndarray,
        feature_names: List[str],
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values from calculate_shap_values
            feature_names: List of feature names
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        shap.summary_plot(
            shap_values,
            pd.DataFrame(np.zeros_like(shap_values), columns=feature_names),
            show=False
        )
        plt.tight_layout()
        
    @staticmethod
    def plot_shap_dependence(
        shap_values: np.ndarray,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """Plot SHAP dependence plot.
        
        Args:
            shap_values: SHAP values from calculate_shap_values
            X: Feature DataFrame
            feature: Feature to plot
            interaction_feature: Optional feature to show interactions with
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        if interaction_feature:
            shap.dependence_plot(
                feature,
                shap_values,
                X,
                interaction_index=interaction_feature,
                show=False
            )
        else:
            shap.dependence_plot(
                feature,
                shap_values,
                X,
                show=False
            )
        plt.tight_layout()
        
    @staticmethod
    def plot_partial_dependence(
        model: BaseModel,
        X: pd.DataFrame,
        features: Union[str, List[str]],
        figsize: Tuple[int, int] = (10, 6)
    ):
        """Plot partial dependence for features.
        
        Args:
            model: Trained model instance
            X: Feature DataFrame
            features: Feature(s) to plot
            figsize: Figure size
        """
        # Scale features
        X_scaled = pd.DataFrame(
            model.scaler.transform(X),
            columns=X.columns
        )
        
        # Convert single feature to list
        if isinstance(features, str):
            features = [features]
            
        # Create display
        plt.figure(figsize=figsize)
        display = PartialDependenceDisplay.from_estimator(
            model.model,
            X_scaled,
            features,
            kind='average',
            subsample=1000,
            n_jobs=-1,
            grid_resolution=50
        )
        plt.tight_layout()
        
    @staticmethod
    def analyze_feature_interactions(
        shap_values: np.ndarray,
        feature_names: List[str],
        top_n: int = 10
    ) -> pd.DataFrame:
        """Analyze feature interactions using SHAP values.
        
        Args:
            shap_values: SHAP values from calculate_shap_values
            feature_names: List of feature names
            top_n: Number of top interactions to return
            
        Returns:
            DataFrame with interaction strengths
        """
        n_features = len(feature_names)
        interactions = np.zeros((n_features, n_features))
        
        # Calculate interaction strengths
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction = np.abs(
                    np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                )
                interactions[i, j] = interaction
                interactions[j, i] = interaction
                
        # Convert to DataFrame
        interactions_df = pd.DataFrame(
            interactions,
            columns=feature_names,
            index=feature_names
        )
        
        # Get top interactions
        interaction_strengths = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction_strengths.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'strength': interactions[i, j]
                })
                
        results = pd.DataFrame(interaction_strengths)
        return results.nlargest(top_n, 'strength')
    
    @staticmethod
    def analyze_prediction(
        model: BaseModel,
        X: pd.DataFrame,
        index: int,
        top_n: int = 10
    ) -> Dict:
        """Analyze individual prediction.
        
        Args:
            model: Trained model instance
            X: Feature DataFrame
            index: Index of sample to analyze
            top_n: Number of top contributing features to return
            
        Returns:
            Dictionary with prediction analysis
        """
        # Get sample data
        sample = X.iloc[[index]]
        X_scaled = pd.DataFrame(
            model.scaler.transform(sample),
            columns=X.columns
        )
        
        # Get prediction
        if hasattr(model.model, 'predict_proba'):
            prediction = model.model.predict_proba(X_scaled)[0]
            pred_class = model.model.predict(X_scaled)[0]
        else:
            prediction = model.model.predict(X_scaled)[0]
            pred_class = prediction
            
        # Calculate SHAP values
        shap_values, _ = ModelInterpreter.calculate_shap_values(model, sample)
        
        # Get feature contributions
        contributions = pd.Series(
            shap_values[0],
            index=X.columns
        ).sort_values(ascending=False)
        
        return {
            'prediction': prediction,
            'predicted_class': pred_class,
            'feature_values': sample.iloc[0].to_dict(),
            'top_contributing_features': contributions.head(top_n).to_dict(),
            'bottom_contributing_features': contributions.tail(top_n).to_dict()
        }
    
    @staticmethod
    def plot_prediction_waterfall(
        shap_values: np.ndarray,
        X: pd.DataFrame,
        index: int,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """Plot SHAP waterfall plot for individual prediction.
        
        Args:
            shap_values: SHAP values from calculate_shap_values
            X: Feature DataFrame
            index: Index of sample to analyze
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[index],
                base_values=np.zeros(1),
                data=X.iloc[index],
                feature_names=list(X.columns)
            ),
            show=False
        )
        plt.tight_layout()
