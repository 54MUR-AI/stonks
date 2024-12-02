import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory"""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 target_return: Optional[float] = None,
                 max_position_size: float = 0.4):
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.max_position_size = max_position_size
        
    def calculate_portfolio_metrics(self, 
                                  weights: np.ndarray,
                                  returns: pd.DataFrame,
                                  cov_matrix: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        return portfolio_return, portfolio_vol, sharpe_ratio
        
    def optimize_portfolio(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Optimize portfolio weights using mean-variance optimization"""
        # Calculate daily returns
        returns = pd.DataFrame({
            symbol: df['close'].pct_change().dropna()
            for symbol, df in historical_data.items()
        })
        
        # Calculate covariance matrix and ensure it's positive definite
        cov_matrix = returns.cov()
        min_eigenval = np.min(np.linalg.eigvals(cov_matrix))
        if min_eigenval < 0:
            cov_matrix -= 1.2 * min_eigenval * np.eye(cov_matrix.shape[0])
        
        # Define optimization constraints with improved numerical stability
        num_assets = len(historical_data)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x.sum() - 0.9999},  # Lower bound for sum
            {'type': 'ineq', 'fun': lambda x: 1.0001 - x.sum()}  # Upper bound for sum
        ]
        
        # Add position size constraints with small buffer
        bounds = tuple((max(0, 1e-6), min(self.max_position_size, 1.0 - 1e-6)) for _ in range(num_assets))
        
        # Define objective function based on optimization goal
        if self.target_return is not None:
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                # Add penalty for numerical stability
                penalty = 100 * np.sum(np.maximum(0, weights - self.max_position_size)**2)
                return portfolio_vol + penalty
                
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.abs(np.sum(returns.mean() * x) * 252 - self.target_return) - 1e-6
            })
        else:
            def objective(weights):
                portfolio_return = np.sum(returns.mean() * weights) * 252
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                # Add small constant to avoid division by zero
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / (portfolio_vol + 1e-8)
                # Add penalty for numerical stability
                penalty = 100 * np.sum(np.maximum(0, weights - self.max_position_size)**2)
                return -sharpe_ratio + penalty
        
        # Try multiple initial guesses for better convergence
        best_result = None
        best_sharpe = float('-inf')
        
        initial_guesses = [
            np.array([1/num_assets] * num_assets),  # Equal weights
            np.random.dirichlet(np.ones(num_assets)),  # Random weights
            np.array([0.8/num_assets if i == 0 else 0.2/(num_assets-1) for i in range(num_assets)])  # Concentrated
        ]
        
        for initial_weights in initial_guesses:
            try:
                result = minimize(
                    objective,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-8}
                )
                
                if result.success:
                    # Normalize weights to ensure they sum to 1
                    weights = result.x / result.x.sum()
                    
                    # Calculate metrics
                    portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(
                        weights, returns, cov_matrix
                    )
                    
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_result = weights
                
            except Exception as e:
                logger.error(f"Optimization error with initial guess: {str(e)}")
                continue
        
        if best_result is None:
            logger.warning("All optimization attempts failed, returning equal weights")
            return dict(zip(historical_data.keys(), initial_guesses[0]))
        
        # Calculate final portfolio metrics
        portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(
            best_result, returns, cov_matrix
        )
        
        logger.info(f"Optimized Portfolio Metrics:")
        logger.info(f"Expected Return: {portfolio_return:.2%}")
        logger.info(f"Volatility: {portfolio_vol:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        return dict(zip(historical_data.keys(), best_result))
        
    def generate_efficient_frontier(self, 
                                  historical_data: Dict[str, pd.DataFrame],
                                  num_portfolios: int = 100) -> pd.DataFrame:
        """Generate efficient frontier points"""
        # Calculate daily returns
        returns = pd.DataFrame({
            symbol: df['close'].pct_change().dropna()
            for symbol, df in historical_data.items()
        })
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate minimum and maximum possible returns
        min_ret = min(returns.mean() * 252)
        max_ret = max(returns.mean() * 252)
        
        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        efficient_portfolios = []
        
        for target_ret in target_returns:
            self.target_return = target_ret
            weights = self.optimize_portfolio(historical_data)
            weights_array = np.array(list(weights.values()))
            
            portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(
                weights_array, returns, cov_matrix
            )
            
            efficient_portfolios.append({
                'Return': portfolio_return,
                'Volatility': portfolio_vol,
                'Sharpe Ratio': sharpe_ratio,
                **weights
            })
            
        return pd.DataFrame(efficient_portfolios)
        
class RiskParityOptimizer:
    """Risk Parity Portfolio Optimization"""
    
    def __init__(self, risk_target: float = 0.15):
        self.risk_target = risk_target
        
    def calculate_portfolio_risk(self, 
                               weights: np.ndarray,
                               cov_matrix: pd.DataFrame) -> Tuple[float, np.ndarray]:
        """Calculate portfolio risk and risk contributions"""
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        marginal_risk = np.dot(cov_matrix * 252, weights) / portfolio_vol
        risk_contribution = weights * marginal_risk
        return portfolio_vol, risk_contribution
        
    def risk_parity_objective(self, 
                            weights: np.ndarray,
                            cov_matrix: pd.DataFrame) -> float:
        """Objective function for risk parity optimization"""
        portfolio_vol, risk_contribution = self.calculate_portfolio_risk(weights, cov_matrix)
        target_risk_contrib = portfolio_vol / len(weights)
        return np.sum((risk_contribution - target_risk_contrib)**2)
        
    def optimize_portfolio(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Optimize portfolio weights using risk parity approach"""
        # Calculate daily returns
        returns = pd.DataFrame({
            symbol: df['close'].pct_change().dropna()
            for symbol, df in historical_data.items()
        })
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Define constraints
        num_assets = len(historical_data)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array([1/num_assets] * num_assets)
        
        try:
            # Optimize for risk parity
            result = minimize(
                lambda w: self.risk_parity_objective(w, cov_matrix),
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if not result.success:
                logger.warning(f"Risk parity optimization failed: {result.message}")
                return dict(zip(historical_data.keys(), initial_weights))
                
            optimal_weights = result.x
            
            # Scale weights to match target volatility
            portfolio_vol, _ = self.calculate_portfolio_risk(optimal_weights, cov_matrix)
            scaling_factor = self.risk_target / portfolio_vol
            optimal_weights = optimal_weights * scaling_factor
            
        except Exception as e:
            logger.error(f"Risk parity optimization error: {str(e)}")
            return dict(zip(historical_data.keys(), initial_weights))
            
        return dict(zip(historical_data.keys(), optimal_weights))

class BlackLittermanOptimizer:
    """Portfolio optimization using the Black-Litterman model"""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 market_risk_aversion: float = 2.5,
                 tau: float = 0.025,
                 max_position_size: float = 0.4):
        """
        Initialize Black-Litterman optimizer
        
        Args:
            risk_free_rate: Risk-free rate (default: 2%)
            market_risk_aversion: Market price of risk (default: 2.5)
            tau: Uncertainty in prior (default: 0.025)
            max_position_size: Maximum weight for any asset (default: 40%)
        """
        self.risk_free_rate = risk_free_rate
        self.market_risk_aversion = market_risk_aversion
        self.tau = tau
        self.max_position_size = max_position_size
        
    def calculate_market_implied_returns(self,
                                      market_weights: np.ndarray,
                                      cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate market implied returns using reverse optimization"""
        return self.risk_free_rate + self.market_risk_aversion * np.dot(cov_matrix, market_weights)
        
    def incorporate_views(self,
                        prior_returns: np.ndarray,
                        cov_matrix: np.ndarray,
                        view_matrix: np.ndarray,
                        view_returns: np.ndarray,
                        view_confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Incorporate investor views into prior beliefs
        
        Args:
            prior_returns: Market implied returns
            cov_matrix: Covariance matrix of returns
            view_matrix: Matrix P of views (k x n), where k is number of views and n is number of assets
            view_returns: Expected returns for each view
            view_confidences: Confidence in each view (diagonal elements of Ω)
            
        Returns:
            Tuple of posterior returns and covariance
        """
        # Calculate posterior covariance
        prior_cov = self.tau * cov_matrix
        view_cov = np.diag(view_confidences)
        
        temp = np.dot(np.dot(view_matrix, prior_cov), view_matrix.T)
        temp_inv = np.linalg.inv(temp + view_cov)
        
        posterior_cov = prior_cov - np.dot(
            np.dot(np.dot(prior_cov, view_matrix.T),
                  temp_inv),
            np.dot(view_matrix, prior_cov)
        )
        
        # Calculate posterior returns
        temp = np.dot(np.dot(prior_cov, view_matrix.T),
                     np.linalg.inv(temp + view_cov))
        posterior_returns = prior_returns + np.dot(
            temp, (view_returns - np.dot(view_matrix, prior_returns))
        )
        
        return posterior_returns, posterior_cov
        
    def optimize_portfolio(self,
                         historical_data: Dict[str, pd.DataFrame],
                         views: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Optimize portfolio using Black-Litterman model
        
        Args:
            historical_data: Dictionary of historical price data
            views: List of dictionaries containing views:
                  [{'assets': ['AAPL', 'MSFT'], 'weights': [1, -1],
                    'return': 0.1, 'confidence': 0.5}, ...]
                    
        Returns:
            Dictionary of optimal weights
        """
        # Calculate returns and covariance
        returns = pd.DataFrame({
            symbol: df['close'].pct_change().dropna()
            for symbol, df in historical_data.items()
        })
        
        cov_matrix = returns.cov().values
        
        # Use market cap weights as prior
        market_weights = np.array([
            df['close'].iloc[-1] * df['volume'].iloc[-1]
            for df in historical_data.values()
        ])
        market_weights = market_weights / market_weights.sum()
        
        # Calculate market implied returns
        prior_returns = self.calculate_market_implied_returns(market_weights, cov_matrix)
        
        # Incorporate views if provided
        if views:
            # Construct view matrix
            n_assets = len(historical_data)
            n_views = len(views)
            view_matrix = np.zeros((n_views, n_assets))
            view_returns = np.zeros(n_views)
            view_confidences = np.zeros(n_views)
            
            asset_list = list(historical_data.keys())
            
            for i, view in enumerate(views):
                for asset, weight in zip(view['assets'], view['weights']):
                    j = asset_list.index(asset)
                    view_matrix[i, j] = weight
                view_returns[i] = view['return']
                view_confidences[i] = 1 / view['confidence']
            
            # Calculate posterior distribution
            posterior_returns, posterior_cov = self.incorporate_views(
                prior_returns, cov_matrix, view_matrix, view_returns, view_confidences
            )
        else:
            posterior_returns = prior_returns
            posterior_cov = self.tau * cov_matrix
        
        # Optimize portfolio using posterior distribution
        def objective(weights):
            portfolio_return = np.dot(weights, posterior_returns)
            portfolio_vol = np.sqrt(np.dot(np.dot(weights, posterior_cov), weights))
            utility = portfolio_return - 0.5 * self.market_risk_aversion * portfolio_vol**2
            
            # Add stronger penalties for constraint violations
            max_weight_penalty = 1000 * np.sum(np.maximum(0, weights - self.max_position_size)**2)
            min_weight_penalty = 1000 * np.sum(np.maximum(0, -weights)**2)
            sum_penalty = 1000 * (np.sum(weights) - 1.0)**2
            
            return -utility + max_weight_penalty + min_weight_penalty + sum_penalty
        
        # Optimization constraints
        n_assets = len(historical_data)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # Tighter bounds to ensure max position size is respected
        bounds = tuple((0.0, min(self.max_position_size, 0.99)) for _ in range(n_assets))
        
        # Try multiple initial guesses with better initialization
        best_result = None
        best_utility = float('-inf')
        
        initial_guesses = [
            market_weights,  # Market weights
            np.array([1/n_assets] * n_assets),  # Equal weights
            np.random.dirichlet(np.ones(n_assets) * 5),  # More concentrated random weights
            np.minimum(market_weights, self.max_position_size)  # Capped market weights
        ]
        
        for initial_weights in initial_guesses:
            # Ensure initial guess satisfies constraints
            initial_weights = initial_weights / initial_weights.sum()
            initial_weights = np.minimum(initial_weights, self.max_position_size)
            initial_weights = initial_weights / initial_weights.sum()
            
            try:
                result = minimize(
                    objective,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-8}
                )
                
                if result.success:
                    utility = -objective(result.x)
                    if utility > best_utility:
                        best_utility = utility
                        best_result = result.x
                        
            except Exception as e:
                logger.error(f"Optimization error with initial guess: {str(e)}")
                continue
        
        if best_result is None:
            logger.warning("Black-Litterman optimization failed, returning market weights")
            return dict(zip(historical_data.keys(), market_weights))
            
        # Ensure final weights satisfy constraints
        final_weights = np.minimum(best_result, self.max_position_size)
        final_weights = final_weights / final_weights.sum()
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(final_weights, posterior_returns)
        portfolio_vol = np.sqrt(np.dot(np.dot(final_weights, posterior_cov), final_weights))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        logger.info("Black-Litterman Portfolio Metrics:")
        logger.info(f"Expected Return: {portfolio_return:.2%}")
        logger.info(f"Volatility: {portfolio_vol:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        return dict(zip(historical_data.keys(), final_weights))
