import numpy as np
from scipy import stats, optimize
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def get_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error getting stock data for {symbol}: {e}")
        return None

def calculate_portfolio_metrics(positions, start_date=None, end_date=None):
    """
    Calculate portfolio metrics including returns, volatility, and Sharpe ratio.
    """
    try:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Get historical data for all positions
        returns_data = {}
        weights = {}
        total_value = sum(position.quantity * position.avg_price for position in positions)

        for position in positions:
            data = get_stock_data(position.symbol, start_date, end_date)
            if data is not None and not data.empty:
                returns_data[position.symbol] = data['Close'].pct_change().dropna()
                weights[position.symbol] = (position.quantity * position.avg_price) / total_value

        if not returns_data:
            return None

        # Create portfolio returns series
        portfolio_returns = pd.Series(0, index=next(iter(returns_data.values())).index)
        for symbol, returns in returns_data.items():
            portfolio_returns += returns * weights[symbol]

        # Calculate metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        max_drawdown = calculate_max_drawdown(portfolio_returns)

        return {
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown)
        }
    except Exception as e:
        print(f"Error calculating portfolio metrics: {e}")
        return None

def calculate_max_drawdown(returns):
    """Calculate the maximum drawdown from a series of returns."""
    try:
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return float(drawdowns.min())
    except Exception as e:
        print(f"Error calculating max drawdown: {e}")
        return 0.0

def calculate_correlation_matrix(positions, start_date=None, end_date=None):
    """
    Calculate correlation matrix for portfolio positions.
    """
    try:
        # Get historical data for all positions
        returns_data = {}
        for position in positions:
            ticker = position.symbol
            data = yf.download(ticker, start=start_date, end=end_date)
            if data is not None and not data.empty:
                returns_data[ticker] = data['Close'].pct_change().dropna()

        if not returns_data:
            return None

        # Create a DataFrame with aligned dates
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Convert to dictionary format
        return correlation_matrix.to_dict()
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return None

def optimize_portfolio_weights(positions, risk_tolerance=0.5, min_weight=0.1, max_weight=0.5):
    """
    Optimize portfolio weights based on modern portfolio theory.
    Args:
        positions: List of portfolio positions
        risk_tolerance: Float between 0 and 1, higher values prefer higher returns over lower risk
        min_weight: Minimum weight for any position
        max_weight: Maximum weight for any position
    Returns:
        Dictionary with optimized weights and metrics
    """
    try:
        # Get historical data and calculate returns
        returns_data = {}
        for position in positions:
            data = get_stock_data(position.symbol, None, None)
            if data is not None and not data.empty:
                returns_data[position.symbol] = data['Close'].pct_change().dropna()

        if not returns_data:
            return None

        returns_df = pd.DataFrame(returns_data)
        
        # Calculate expected returns and covariance
        exp_returns = returns_df.mean() * 252  # Annualized returns
        cov_matrix = returns_df.cov() * 252    # Annualized covariance
        
        n_assets = len(returns_df.columns)
        
        # Define optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x - min_weight},  # Minimum weight
            {'type': 'ineq', 'fun': lambda x: max_weight - x}   # Maximum weight
        ]
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(exp_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_std if portfolio_std != 0 else 0
            # Adjust objective based on risk tolerance
            return -(risk_tolerance * portfolio_return - (1 - risk_tolerance) * portfolio_std)
        
        # Optimize
        result = optimize.minimize(objective, initial_weights, method='SLSQP', constraints=constraints)
        
        if not result.success:
            print(f"Optimization failed: {result.message}")
            return None
            
        optimized_weights = result.x
        
        # Calculate portfolio metrics with optimized weights
        opt_return = np.sum(exp_returns * optimized_weights)
        opt_std = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
        opt_sharpe = opt_return / opt_std if opt_std != 0 else 0
        
        # Create results dictionary
        weights_dict = {symbol: float(weight) for symbol, weight in zip(returns_df.columns, optimized_weights)}
        
        return {
            "weights": weights_dict,
            "expected_return": float(opt_return),
            "volatility": float(opt_std),
            "sharpe_ratio": float(opt_sharpe)
        }
        
    except Exception as e:
        print(f"Error in portfolio optimization: {e}")
        return None
