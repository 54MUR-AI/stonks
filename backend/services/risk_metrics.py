import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
import yfinance as yf

@dataclass
class RiskMetrics:
    symbol: str
    timestamp: datetime
    volatility: float
    value_at_risk: float
    expected_shortfall: float
    beta: float
    correlation: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    metadata: Dict

class RiskMetricsService:
    def __init__(self):
        self.risk_free_rate = 0.03  # Annualized risk-free rate
        self.confidence_level = 0.95
        self.lookback_window = 252  # One trading year
        
    def calculate_portfolio_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
        weights: Optional[np.ndarray] = None
    ) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        # Annualized metrics
        ann_factor = np.sqrt(252)  # Annualization factor for daily data
        
        # Calculate volatility
        volatility = portfolio_returns.std() * ann_factor
        
        # Calculate Value at Risk (VaR)
        var = self.calculate_var(portfolio_returns, self.confidence_level)
        
        # Calculate Expected Shortfall (CVaR)
        es = self.calculate_expected_shortfall(portfolio_returns, self.confidence_level)
        
        # Calculate beta and correlation
        beta, correlation = self.calculate_market_metrics(portfolio_returns, market_returns)
        
        # Calculate risk-adjusted returns
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe = self.calculate_sharpe_ratio(excess_returns, volatility)
        sortino = self.calculate_sortino_ratio(excess_returns)
        
        # Calculate maximum drawdown
        max_dd = self.calculate_max_drawdown(portfolio_returns)
        
        return RiskMetrics(
            symbol="PORTFOLIO",
            timestamp=datetime.now(),
            volatility=volatility,
            value_at_risk=var,
            expected_shortfall=es,
            beta=beta,
            correlation=correlation,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            metadata={
                "confidence_level": self.confidence_level,
                "risk_free_rate": self.risk_free_rate,
                "lookback_window": self.lookback_window
            }
        )
        
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float,
        method: str = "historical"
    ) -> float:
        """Calculate Value at Risk using different methods"""
        if method == "historical":
            return -np.percentile(returns, (1 - confidence_level) * 100)
        elif method == "parametric":
            z_score = stats.norm.ppf(confidence_level)
            return -(returns.mean() + z_score * returns.std())
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
            
    def calculate_expected_shortfall(
        self,
        returns: pd.Series,
        confidence_level: float
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self.calculate_var(returns, confidence_level)
        return -returns[returns <= -var].mean()
        
    def calculate_market_metrics(
        self,
        returns: pd.Series,
        market_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate beta and correlation with market"""
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        correlation = returns.corr(market_returns)
        
        beta = covariance / market_variance if market_variance != 0 else 0
        return beta, correlation
        
    def calculate_sharpe_ratio(
        self,
        excess_returns: pd.Series,
        volatility: float
    ) -> float:
        """Calculate Sharpe Ratio"""
        if volatility == 0:
            return 0
        return excess_returns.mean() * np.sqrt(252) / volatility
        
    def calculate_sortino_ratio(
        self,
        excess_returns: pd.Series
    ) -> float:
        """Calculate Sortino Ratio"""
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        
        if downside_std == 0:
            return 0
        return excess_returns.mean() * np.sqrt(252) / downside_std
        
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate Maximum Drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns.min()
        
    def calculate_component_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float
    ) -> pd.Series:
        """Calculate Component VaR for portfolio constituents"""
        portfolio_var = self.calculate_var(
            (returns * weights).sum(axis=1),
            confidence_level
        )
        
        # Calculate marginal VaR
        cov_matrix = returns.cov()
        portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_var = (cov_matrix @ weights) / portfolio_std
        
        # Component VaR
        component_var = weights * marginal_var * portfolio_var
        return pd.Series(component_var, index=returns.columns)
        
    def calculate_risk_contribution(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.Series:
        """Calculate risk contribution of each asset"""
        cov_matrix = returns.cov()
        portfolio_var = weights.T @ cov_matrix @ weights
        
        # Marginal risk contribution
        mrc = cov_matrix @ weights
        
        # Component risk contribution
        rc = weights * mrc
        prc = rc / portfolio_var
        
        return pd.Series(prc, index=returns.columns)
        
    def stress_test_portfolio(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Perform stress testing under different scenarios"""
        results = {}
        
        for scenario_name, scenario in scenarios.items():
            # Apply stress factors to returns
            stressed_returns = returns.copy()
            for asset, stress_factor in scenario.items():
                if asset in stressed_returns.columns:
                    stressed_returns[asset] *= (1 + stress_factor)
                    
            # Calculate portfolio return under stress
            portfolio_return = (stressed_returns * weights).sum(axis=1)
            
            # Calculate key metrics under stress
            results[scenario_name] = {
                "return": portfolio_return.mean(),
                "volatility": portfolio_return.std() * np.sqrt(252),
                "var": self.calculate_var(portfolio_return, self.confidence_level),
                "max_drawdown": self.calculate_max_drawdown(portfolio_return)
            }
            
        return results
        
    def calculate_tail_dependency(
        self,
        returns: pd.DataFrame,
        threshold: float = 0.05
    ) -> pd.DataFrame:
        """Calculate tail dependency matrix"""
        n_assets = returns.shape[1]
        tail_matrix = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(i, n_assets):
                # Calculate lower tail dependency
                asset1 = returns.iloc[:, i]
                asset2 = returns.iloc[:, j]
                
                tail_dep = self.calculate_tail_dependence_coefficient(
                    asset1, asset2, threshold
                )
                
                tail_matrix[i, j] = tail_dep
                tail_matrix[j, i] = tail_dep
                
        return pd.DataFrame(
            tail_matrix,
            index=returns.columns,
            columns=returns.columns
        )
        
    @staticmethod
    def calculate_tail_dependence_coefficient(
        x: pd.Series,
        y: pd.Series,
        threshold: float
    ) -> float:
        """Calculate lower tail dependence coefficient"""
        x_quantile = np.percentile(x, threshold * 100)
        y_quantile = np.percentile(y, threshold * 100)
        
        joint_exceedance = np.mean(
            (x <= x_quantile) & (y <= y_quantile)
        )
        
        if threshold == 0:
            return 0
            
        return joint_exceedance / threshold

risk_metrics_service = RiskMetricsService()
