import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

class PortfolioFactorAnalyzer:
    """Analyze portfolio risk factors and decompose returns"""
    
    def __init__(self, n_factors: int = 3):
        """
        Initialize factor analyzer
        
        Args:
            n_factors: Number of factors to extract (default: 3)
        """
        self.n_factors = n_factors
        self.pca = PCA(n_components=n_factors)
        self.fa = FactorAnalysis(n_components=n_factors, random_state=42)
        self.scaler = StandardScaler()
        
    def extract_statistical_factors(self,
                                  returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract statistical risk factors using PCA
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Tuple of (factor returns, factor loadings)
        """
        # Standardize returns
        scaled_returns = self.scaler.fit_transform(returns)
        
        # Extract factors using PCA
        self.pca.fit(scaled_returns)
        
        # Get factor loadings (standardized)
        loadings = self.pca.transform(scaled_returns)
        factor_loadings = pd.DataFrame(
            loadings / np.std(loadings, axis=0),  # Standardize loadings
            index=returns.index,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        # Get component weights
        components = pd.DataFrame(
            self.pca.components_,
            columns=returns.columns,
            index=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        return factor_loadings, components
        
    def analyze_factor_contribution(self,
                                  returns: pd.DataFrame,
                                  weights: pd.Series) -> Dict:
        """
        Analyze factor contribution to portfolio risk
        
        Args:
            returns: DataFrame of asset returns
            weights: Series of portfolio weights
            
        Returns:
            Dictionary with factor analysis results
        """
        # Extract factors
        factor_loadings, components = self.extract_statistical_factors(returns)
        
        # Calculate factor exposures
        factor_exposures = components @ weights
        
        # Calculate factor variances
        factor_variances = np.var(factor_loadings, axis=0)
        total_variance = np.sum(factor_variances)
        
        # Calculate risk contribution
        risk_contribution = factor_variances / total_variance
        
        # Calculate cumulative variance explained
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        
        return {
            'factor_exposures': pd.Series(factor_exposures, 
                                        index=[f'Factor_{i+1}' for i in range(self.n_factors)]),
            'risk_contribution': pd.Series(risk_contribution,
                                         index=[f'Factor_{i+1}' for i in range(self.n_factors)]),
            'cumulative_variance': pd.Series(cumulative_variance,
                                           index=[f'Factor_{i+1}' for i in range(self.n_factors)]),
            'components': components
        }
        
    def decompose_returns(self,
                         returns: pd.DataFrame,
                         weights: pd.Series) -> pd.DataFrame:
        """
        Decompose portfolio returns into factor contributions
        
        Args:
            returns: DataFrame of asset returns
            weights: Series of portfolio weights
            
        Returns:
            DataFrame of factor contributions to returns
        """
        # Extract factors
        factor_loadings, components = self.extract_statistical_factors(returns)
        
        # Calculate factor returns
        factor_returns = pd.DataFrame(
            np.zeros((len(returns), self.n_factors)),
            index=returns.index,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        # Calculate contribution of each factor
        for i in range(self.n_factors):
            factor_returns.iloc[:, i] = (
                factor_loadings.iloc[:, i] * 
                (components.iloc[i] @ weights)
            )
            
        # Add residual returns
        portfolio_returns = returns @ weights
        factor_sum = factor_returns.sum(axis=1)
        factor_returns['Residual'] = portfolio_returns - factor_sum
        
        return factor_returns
        
    def get_factor_correlations(self,
                              returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between assets and factors
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            DataFrame of factor-asset correlations
        """
        # Extract factors
        factor_loadings, _ = self.extract_statistical_factors(returns)
        
        # Calculate correlations
        combined = pd.concat([factor_loadings, returns], axis=1)
        return combined.corr().iloc[:self.n_factors, self.n_factors:]
        
    def get_factor_summary(self,
                          returns: pd.DataFrame,
                          weights: pd.Series) -> Dict:
        """
        Get comprehensive factor analysis summary
        
        Args:
            returns: DataFrame of asset returns
            weights: Series of portfolio weights
            
        Returns:
            Dictionary with factor analysis summary
        """
        # Get factor decomposition
        factor_returns = self.decompose_returns(returns, weights)
        
        # Calculate factor metrics
        factor_metrics = pd.DataFrame({
            'Annualized Return': factor_returns.mean() * 252,
            'Annualized Vol': factor_returns.std() * np.sqrt(252),
            'Sharpe Ratio': (factor_returns.mean() * 252) / (factor_returns.std() * np.sqrt(252)),
            'Contribution to Risk': factor_returns.std() / factor_returns.std().sum(),
            'Max Drawdown': (factor_returns.cumsum() - 
                           factor_returns.cumsum().expanding().max()).min()
        })
        
        # Get factor correlations
        factor_correlations = self.get_factor_correlations(returns)
        
        # Get factor analysis
        factor_analysis = self.analyze_factor_contribution(returns, weights)
        
        return {
            'factor_metrics': factor_metrics,
            'factor_correlations': factor_correlations,
            'factor_analysis': factor_analysis,
            'factor_returns': factor_returns
        }
