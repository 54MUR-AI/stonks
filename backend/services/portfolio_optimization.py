import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from scipy.optimize import minimize
import cvxopt as cv
from cvxopt import matrix, solvers

@dataclass
class OptimizationResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    metadata: Dict

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.03  # Annualized risk-free rate
        self.target_return = None
        self.target_risk = None
        self.constraints = None
        
    def mean_variance_optimization(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        constraints: Optional[Dict] = None
    ) -> OptimizationResult:
        """
        Perform mean-variance optimization using quadratic programming
        """
        n_assets = returns.shape[1]
        returns_mean = returns.mean() * 252  # Annualized returns
        cov_matrix = returns.cov() * 252  # Annualized covariance
        
        # Setup optimization parameters
        P = matrix(cov_matrix.values)
        q = matrix(np.zeros(n_assets))
        
        # Constraints
        # 1. Sum of weights = 1
        A = matrix(1.0, (1, n_assets))
        b = matrix(1.0)
        
        # 2. Long-only constraint (optional)
        if constraints and constraints.get("long_only", True):
            G = matrix(-np.eye(n_assets))
            h = matrix(np.zeros(n_assets))
        else:
            G = None
            h = None
            
        # 3. Target return constraint (optional)
        if target_return is not None:
            A = matrix(np.vstack([A, returns_mean.values]))
            b = matrix([1.0, target_return])
            
        # Solve optimization problem
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        
        if sol['status'] != 'optimal':
            raise ValueError("Optimization failed to converge")
            
        # Extract results
        weights = np.array(sol['x']).flatten()
        portfolio_return = np.sum(returns_mean * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            volatility=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            metadata={
                "optimization_type": "mean_variance",
                "target_return": target_return,
                "target_risk": target_risk,
                "constraints": constraints
            }
        )
        
    def risk_parity_optimization(
        self,
        returns: pd.DataFrame,
        risk_targets: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Perform risk parity optimization
        """
        n_assets = returns.shape[1]
        cov_matrix = returns.cov() * 252
        
        if risk_targets is None:
            risk_targets = np.ones(n_assets) / n_assets
            
        def risk_parity_objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            asset_contrib = weights * (np.dot(cov_matrix, weights)) / portfolio_risk
            return np.sum((asset_contrib - risk_targets)**2)
            
        # Optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights = 1
            {'type': 'ineq', 'fun': lambda x: x}  # Long-only constraint
        ]
        
        # Initial guess
        init_weights = np.ones(n_assets) / n_assets
        
        # Solve optimization
        result = minimize(
            risk_parity_objective,
            init_weights,
            method='SLSQP',
            constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError("Risk parity optimization failed to converge")
            
        weights = result.x
        returns_mean = returns.mean() * 252
        portfolio_return = np.sum(returns_mean * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            volatility=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            metadata={
                "optimization_type": "risk_parity",
                "risk_targets": risk_targets.tolist(),
                "convergence": result.success
            }
        )
        
    def black_litterman_optimization(
        self,
        returns: pd.DataFrame,
        market_caps: np.ndarray,
        views: List[Dict],
        confidence: List[float],
        tau: float = 0.025
    ) -> OptimizationResult:
        """
        Perform Black-Litterman optimization
        """
        n_assets = returns.shape[1]
        returns_mean = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Market-implied returns
        market_weights = market_caps / np.sum(market_caps)
        implied_returns = self.risk_free_rate + market_weights * \
                         np.sqrt(np.diag(cov_matrix))
        
        # Process views
        n_views = len(views)
        P = np.zeros((n_views, n_assets))
        q = np.zeros(n_views)
        
        for i, view in enumerate(views):
            P[i, returns.columns.get_loc(view['asset'])] = view['weight']
            q[i] = view['return']
            
        # Confidence matrix
        omega = np.diag(1 / np.array(confidence))
        
        # Prior covariance
        prior_cov = tau * cov_matrix
        
        # Posterior calculations
        tmp = np.dot(np.dot(P, prior_cov), P.T) + omega
        tmp_inv = np.linalg.inv(tmp)
        er = np.dot(prior_cov, np.dot(P.T, np.dot(tmp_inv, 
             (q - np.dot(P, implied_returns)))))
        posterior_returns = implied_returns + er
        
        # Optimize with posterior estimates
        return self.mean_variance_optimization(
            returns,
            constraints={"long_only": True},
            target_return=np.mean(posterior_returns)
        )
        
    def hierarchical_risk_parity(
        self,
        returns: pd.DataFrame
    ) -> OptimizationResult:
        """
        Perform Hierarchical Risk Parity optimization
        """
        cov_matrix = returns.cov() * 252
        corr_matrix = returns.corr()
        
        # Distance matrix
        dist = np.sqrt(0.5 * (1 - corr_matrix))
        
        # Clustering
        links = self._get_cluster_links(dist)
        sorted_assets = self._get_quasi_diag(links)
        weights = self._get_hrp_weights(cov_matrix, sorted_assets)
        
        # Calculate portfolio metrics
        returns_mean = returns.mean() * 252
        portfolio_return = np.sum(returns_mean * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            volatility=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            metadata={
                "optimization_type": "hierarchical_risk_parity",
                "clusters": links.tolist()
            }
        )
        
    def _get_cluster_links(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Helper function for hierarchical clustering"""
        from scipy.cluster.hierarchy import linkage
        return linkage(dist_matrix, method='single')
        
    def _get_quasi_diag(self, links: np.ndarray) -> List:
        """Helper function to sort clusters"""
        from scipy.cluster.hierarchy import leaves_list
        return leaves_list(links)
        
    def _get_hrp_weights(
        self,
        cov_matrix: np.ndarray,
        sorted_assets: List
    ) -> np.ndarray:
        """Calculate HRP weights"""
        weights = pd.Series(1, index=sorted_assets)
        clusters = [sorted_assets]
        
        while len(clusters) > 0:
            clusters = [c[start:end] for c in clusters
                      for start, end in ((0, len(c) // 2), (len(c) // 2, len(c)))
                      if len(c) > 1]
                      
            for c in clusters:
                c_cov = cov_matrix.iloc[c, c]
                c_vars = np.diag(c_cov)
                c_weights = 1 / c_vars
                c_weights /= c_weights.sum()
                weights[c] *= c_weights
                
        return weights.values

portfolio_optimizer = PortfolioOptimizer()
