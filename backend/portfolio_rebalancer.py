import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime

class PortfolioRebalancer:
    def __init__(self, positions: List, target_weights: Dict[str, float], tolerance: float = 0.05):
        """
        Initialize portfolio rebalancer
        
        Args:
            positions: List of portfolio positions
            target_weights: Dictionary of target weights for each symbol
            tolerance: Acceptable deviation from target weights before rebalancing
        """
        self.positions = positions
        self.target_weights = target_weights
        self.tolerance = tolerance
        self.current_prices = {}
        self.current_weights = {}
        self.total_value = 0
        
    def get_current_prices(self):
        """Get current market prices for all positions"""
        for position in self.positions:
            try:
                ticker = yf.Ticker(position.symbol)
                self.current_prices[position.symbol] = ticker.history(period='1d')['Close'].iloc[-1]
            except Exception as e:
                raise ValueError(f"Error fetching price for {position.symbol}: {str(e)}")
    
    def calculate_current_weights(self):
        """Calculate current portfolio weights"""
        self.total_value = 0
        position_values = {}
        
        # Calculate position values
        for position in self.positions:
            if position.symbol not in self.current_prices:
                continue
            position_values[position.symbol] = position.quantity * self.current_prices[position.symbol]
            self.total_value += position_values[position.symbol]
        
        # Calculate weights
        for symbol, value in position_values.items():
            self.current_weights[symbol] = value / self.total_value if self.total_value > 0 else 0
    
    def calculate_rebalancing_trades(self) -> Dict:
        """Calculate required trades to rebalance portfolio"""
        self.get_current_prices()
        self.calculate_current_weights()
        
        trades = {}
        for symbol in self.target_weights.keys():
            current_weight = self.current_weights.get(symbol, 0)
            target_weight = self.target_weights[symbol]
            
            # Check if rebalancing is needed
            if abs(current_weight - target_weight) > self.tolerance:
                # Calculate target position value
                target_value = self.total_value * target_weight
                
                # Find current position
                current_position = next(
                    (p for p in self.positions if p.symbol == symbol),
                    None
                )
                
                current_value = (
                    current_position.quantity * self.current_prices[symbol]
                    if current_position
                    else 0
                )
                
                # Calculate required trade
                value_difference = target_value - current_value
                quantity = int(value_difference / self.current_prices[symbol])
                
                if quantity != 0:
                    trades[symbol] = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "price": self.current_prices[symbol],
                        "type": "buy" if quantity > 0 else "sell",
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "value": abs(value_difference)
                    }
        
        return trades

    def get_rebalancing_summary(self) -> Dict:
        """Generate summary of portfolio rebalancing analysis"""
        trades = self.calculate_rebalancing_trades()
        
        total_trades_value = sum(trade["value"] for trade in trades.values())
        max_weight_deviation = max(
            abs(self.current_weights.get(symbol, 0) - target_weight)
            for symbol, target_weight in self.target_weights.items()
        )
        
        return {
            "total_portfolio_value": self.total_value,
            "total_rebalancing_value": total_trades_value,
            "rebalancing_trades": trades,
            "current_weights": self.current_weights,
            "target_weights": self.target_weights,
            "max_weight_deviation": max_weight_deviation,
            "positions_requiring_rebalancing": len(trades),
            "timestamp": datetime.now().isoformat()
        }

def optimize_portfolio_weights(
    positions: List,
    risk_tolerance: float = 0.5,
    min_weight: float = 0.05,
    max_weight: float = 0.4
) -> Dict[str, float]:
    """
    Optimize portfolio weights based on modern portfolio theory
    
    Args:
        positions: List of portfolio positions
        risk_tolerance: 0-1 scale where 1 is most aggressive
        min_weight: Minimum weight per position
        max_weight: Maximum weight per position
    """
    symbols = [p.symbol for p in positions]
    
    # Get historical data
    hist_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist_data[symbol] = ticker.history(period='1y')['Close'].pct_change().dropna()
        except Exception as e:
            raise ValueError(f"Error fetching data for {symbol}: {str(e)}")
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(hist_data)
    
    # Calculate expected returns and covariance
    exp_returns = returns_df.mean() * 252  # Annualized returns
    cov_matrix = returns_df.cov() * 252    # Annualized covariance
    
    # Generate random portfolios
    num_portfolios = 5000
    all_weights = np.zeros((num_portfolios, len(symbols)))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(len(symbols))
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / np.sum(weights)
        all_weights[i,:] = weights
        
        # Calculate portfolio return and volatility
        ret_arr[i] = np.sum(exp_returns * weights)
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_arr[i] = ret_arr[i] / vol_arr[i]
    
    # Find optimal portfolio based on risk tolerance
    if risk_tolerance < 0.5:
        # More conservative: minimize volatility
        optimal_idx = vol_arr.argmin()
    else:
        # More aggressive: maximize Sharpe ratio
        optimal_idx = sharpe_arr.argmax()
    
    # Create dictionary of optimal weights
    optimal_weights = {
        symbol: weight
        for symbol, weight in zip(symbols, all_weights[optimal_idx])
    }
    
    return optimal_weights
