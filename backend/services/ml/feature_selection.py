"""Advanced feature selection methods."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression,
    RFE,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from boruta import BorutaPy

@dataclass
class FeatureSelectionResult:
    """Results from feature selection."""
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_method: str
    n_features_selected: int

class FeatureSelector:
    """Advanced feature selection tools."""
    
    @staticmethod
    def select_mutual_information(
        X: pd.DataFrame,
        y: pd.Series,
        k: Optional[int] = None,
        task: str = 'classification'
    ) -> FeatureSelectionResult:
        """Select features using mutual information.
        
        Args:
            X: Feature DataFrame
            y: Target series
            k: Number of features to select (default: half of features)
            task: Either 'classification' or 'regression'
            
        Returns:
            FeatureSelectionResult with selected features and scores
        """
        k = k or X.shape[1] // 2
        mi_func = mutual_info_classif if task == 'classification' else mutual_info_regression
        
        selector = SelectKBest(score_func=mi_func, k=k)
        selector.fit(X, y)
        
        # Get selected features and scores
        mask = selector.get_support()
        scores = dict(zip(X.columns, selector.scores_))
        selected = list(X.columns[mask])
        
        return FeatureSelectionResult(
            selected_features=selected,
            feature_scores=scores,
            selection_method='mutual_information',
            n_features_selected=k
        )
    
    @staticmethod
    def select_f_score(
        X: pd.DataFrame,
        y: pd.Series,
        k: Optional[int] = None,
        task: str = 'classification'
    ) -> FeatureSelectionResult:
        """Select features using F-score.
        
        Args:
            X: Feature DataFrame
            y: Target series
            k: Number of features to select (default: half of features)
            task: Either 'classification' or 'regression'
            
        Returns:
            FeatureSelectionResult with selected features and scores
        """
        k = k or X.shape[1] // 2
        f_func = f_classif if task == 'classification' else f_regression
        
        selector = SelectKBest(score_func=f_func, k=k)
        selector.fit(X, y)
        
        # Get selected features and scores
        mask = selector.get_support()
        scores = dict(zip(X.columns, selector.scores_))
        selected = list(X.columns[mask])
        
        return FeatureSelectionResult(
            selected_features=selected,
            feature_scores=scores,
            selection_method='f_score',
            n_features_selected=k
        )
    
    @staticmethod
    def select_rfe(
        X: pd.DataFrame,
        y: pd.Series,
        k: Optional[int] = None,
        task: str = 'classification'
    ) -> FeatureSelectionResult:
        """Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target series
            k: Number of features to select (default: half of features)
            task: Either 'classification' or 'regression'
            
        Returns:
            FeatureSelectionResult with selected features and scores
        """
        k = k or X.shape[1] // 2
        
        # Initialize estimator based on task
        if task == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            
        selector = RFE(estimator=estimator, n_features_to_select=k)
        selector.fit(X, y)
        
        # Get selected features and ranking
        mask = selector.get_support()
        scores = dict(zip(X.columns, selector.ranking_))
        selected = list(X.columns[mask])
        
        return FeatureSelectionResult(
            selected_features=selected,
            feature_scores=scores,
            selection_method='recursive_feature_elimination',
            n_features_selected=k
        )
    
    @staticmethod
    def select_lasso(
        X: pd.DataFrame,
        y: pd.Series,
        alpha: float = 0.01,
        task: str = 'classification'
    ) -> FeatureSelectionResult:
        """Select features using Lasso regularization.
        
        Args:
            X: Feature DataFrame
            y: Target series
            alpha: L1 regularization strength
            task: Either 'classification' or 'regression'
            
        Returns:
            FeatureSelectionResult with selected features and scores
        """
        # Initialize model based on task
        if task == 'classification':
            model = LogisticRegression(penalty='l1', solver='liblinear', C=1/alpha)
        else:
            model = Lasso(alpha=alpha)
            
        selector = SelectFromModel(model, prefit=False)
        selector.fit(X, y)
        
        # Get selected features and coefficients
        mask = selector.get_support()
        scores = dict(zip(X.columns, np.abs(selector.estimator_.coef_)))
        selected = list(X.columns[mask])
        
        return FeatureSelectionResult(
            selected_features=selected,
            feature_scores=scores,
            selection_method='lasso',
            n_features_selected=len(selected)
        )
    
    @staticmethod
    def select_boruta(
        X: pd.DataFrame,
        y: pd.Series,
        task: str = 'classification'
    ) -> FeatureSelectionResult:
        """Select features using Boruta algorithm.
        
        Args:
            X: Feature DataFrame
            y: Target series
            task: Either 'classification' or 'regression'
            
        Returns:
            FeatureSelectionResult with selected features and scores
        """
        # Initialize random forest based on task
        if task == 'classification':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            
        # Initialize Boruta
        boruta = BorutaPy(
            estimator=rf,
            n_estimators='auto',
            verbose=0,
            random_state=42
        )
        
        # Fit Boruta
        boruta.fit(X.values, y.values)
        
        # Get selected features and ranking
        mask = boruta.support_
        scores = dict(zip(X.columns, boruta.ranking_))
        selected = list(X.columns[mask])
        
        return FeatureSelectionResult(
            selected_features=selected,
            feature_scores=scores,
            selection_method='boruta',
            n_features_selected=len(selected)
        )
    
    @staticmethod
    def ensemble_selection(
        X: pd.DataFrame,
        y: pd.Series,
        k: Optional[int] = None,
        task: str = 'classification',
        methods: Optional[List[str]] = None
    ) -> FeatureSelectionResult:
        """Combine multiple feature selection methods.
        
        Args:
            X: Feature DataFrame
            y: Target series
            k: Number of features to select (default: half of features)
            task: Either 'classification' or 'regression'
            methods: List of methods to use (default: all)
            
        Returns:
            FeatureSelectionResult with selected features and ensemble scores
        """
        k = k or X.shape[1] // 2
        methods = methods or ['mutual_information', 'f_score', 'rfe', 'lasso', 'boruta']
        
        # Initialize scores dictionary
        feature_scores = {feature: 0.0 for feature in X.columns}
        
        # Run each method
        for method in methods:
            if method == 'mutual_information':
                result = FeatureSelector.select_mutual_information(X, y, k, task)
            elif method == 'f_score':
                result = FeatureSelector.select_f_score(X, y, k, task)
            elif method == 'rfe':
                result = FeatureSelector.select_rfe(X, y, k, task)
            elif method == 'lasso':
                result = FeatureSelector.select_lasso(X, y, task=task)
            elif method == 'boruta':
                result = FeatureSelector.select_boruta(X, y, task)
                
            # Update ensemble scores
            for feature in result.selected_features:
                feature_scores[feature] += 1
                
        # Normalize scores
        for feature in feature_scores:
            feature_scores[feature] /= len(methods)
            
        # Select top k features
        selected = sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return FeatureSelectionResult(
            selected_features=[f[0] for f in selected],
            feature_scores=feature_scores,
            selection_method='ensemble',
            n_features_selected=k
        )
