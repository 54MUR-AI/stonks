import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from .risk_prediction import risk_predictor

@dataclass
class RebalanceRecommendation:
    symbol: str
    current_weight: float
    target_weight: float
    action: str  # 'buy', 'sell', or 'hold'
    quantity_change: int
    expected_impact: Dict[str, float]

class PortfolioRebalancer:
    def __init__(self):
        self.risk_free_rate = 0.02  # Assumed risk-free rate
        self.min_weight = 0.05      # Minimum position weight
        self.max_weight = 0.40      # Maximum position weight
        self.transaction_cost = 0.001  # 10 basis points per trade
        
    def _get_historical_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch historical data for multiple symbols"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
            
        data = pd.DataFrame()
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date)
                data[symbol] = hist['Close'].pct_change()
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
                
        return data.dropna()
        
    def _calculate_portfolio_metrics(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        portfolio_returns = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        )
        sharpe_ratio = (portfolio_returns - self.risk_free_rate) / portfolio_volatility
        return portfolio_returns, portfolio_volatility, sharpe_ratio
        
    def _optimize_portfolio(
        self,
        returns: pd.DataFrame,
        risk_predictions: Dict[str, float],
        current_weights: Dict[str, float],
        objective: str = 'sharpe'
    ) -> np.ndarray:
        """Optimize portfolio weights based on objective"""
        n_assets = len(returns.columns)
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess (current weights)
        initial_weights = np.array([current_weights[symbol] for symbol in returns.columns])
        
        # Adjust covariance matrix based on risk predictions
        predicted_vol_multipliers = np.array([
            risk_predictions.get(symbol, 1.0)
            for symbol in returns.columns
        ])
        base_cov = returns.cov().values
        adjusted_cov = base_cov * np.outer(predicted_vol_multipliers, predicted_vol_multipliers)
        
        def objective_function(weights):
            if objective == 'sharpe':
                portfolio_return = np.sum(returns.mean() * weights) * 252
                portfolio_vol = np.sqrt(
                    np.dot(weights.T, np.dot(adjusted_cov * 252, weights))
                )
                return -(portfolio_return - self.risk_free_rate) / portfolio_vol
            elif objective == 'min_variance':
                return np.sqrt(np.dot(weights.T, np.dot(adjusted_cov * 252, weights)))
            elif objective == 'max_diversification':
                portfolio_vol = np.sqrt(
                    np.dot(weights.T, np.dot(adjusted_cov * 252, weights))
                )
                asset_vols = np.sqrt(np.diag(adjusted_cov * 252))
                return -(np.dot(weights, asset_vols) / portfolio_vol)
                
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Add turnover constraint
        max_turnover = 0.2  # 20% maximum turnover
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: max_turnover - np.sum(np.abs(x - initial_weights))
        })
        
        result = minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
        
    def generate_rebalancing_recommendations(
        self,
        portfolio: Dict[str, Dict],
        cash: float,
        objective: str = 'sharpe'
    ) -> List[RebalanceRecommendation]:
        """Generate rebalancing recommendations for a portfolio"""
        # Extract portfolio information
        symbols = list(portfolio.keys())
        current_values = {
            symbol: info['quantity'] * info['current_price']
            for symbol, info in portfolio.items()
        }
        total_value = sum(current_values.values()) + cash
        current_weights = {
            symbol: value / total_value
            for symbol, value in current_values.items()
        }
        
        # Get historical data
        returns = self._get_historical_data(symbols)
        
        # Get risk predictions
        risk_predictions = {
            symbol: risk_predictor.predict_risk(symbol)['predicted_volatility']
            for symbol in symbols
        }
        
        # Optimize portfolio
        optimal_weights = self._optimize_portfolio(
            returns,
            risk_predictions,
            current_weights,
            objective
        )
        
        # Calculate metrics for current and optimal portfolio
        current_metrics = self._calculate_portfolio_metrics(
            returns,
            np.array(list(current_weights.values()))
        )
        optimal_metrics = self._calculate_portfolio_metrics(
            returns,
            optimal_weights
        )
        
        # Generate recommendations
        recommendations = []
        for i, symbol in enumerate(symbols):
            current_weight = current_weights[symbol]
            target_weight = optimal_weights[i]
            weight_diff = target_weight - current_weight
            
            # Calculate required quantity change
            current_price = portfolio[symbol]['current_price']
            target_value = target_weight * total_value
            current_value = current_weight * total_value
            value_change = target_value - current_value
            quantity_change = int(value_change / current_price)
            
            # Determine action
            if abs(weight_diff) < 0.01:  # 1% threshold
                action = 'hold'
            else:
                action = 'buy' if weight_diff > 0 else 'sell'
                
            # Calculate expected impact
            expected_impact = {
                'return_change': optimal_metrics[0] - current_metrics[0],
                'risk_change': optimal_metrics[1] - current_metrics[1],
                'sharpe_change': optimal_metrics[2] - current_metrics[2],
                'transaction_cost': abs(value_change) * self.transaction_cost
            }
            
            recommendations.append(RebalanceRecommendation(
                symbol=symbol,
                current_weight=current_weight,
                target_weight=target_weight,
                action=action,
                quantity_change=quantity_change,
                expected_impact=expected_impact
            ))
            
        return recommendations

portfolio_rebalancer = PortfolioRebalancer()
