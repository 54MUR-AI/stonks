import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def calculate_portfolio_metrics(positions, start_date=None, end_date=None):
    """Calculate portfolio performance metrics"""
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    total_value = 0
    total_cost = 0
    daily_returns = []
    position_metrics = []
    
    # Download market data for benchmark (S&P 500)
    benchmark = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    benchmark_returns = benchmark.pct_change().dropna()
    
    for position in positions:
        try:
            # Get historical data
            data = yf.download(position.symbol, start=start_date, end=end_date)
            
            if data.empty:
                continue
                
            current_price = data['Adj Close'][-1]
            position_value = position.quantity * current_price
            position_cost = position.quantity * position.average_price
            
            # Calculate returns
            returns = data['Adj Close'].pct_change().dropna()
            
            # Calculate metrics
            daily_return = returns.mean()
            volatility = returns.std()
            sharpe_ratio = (daily_return - 0.02/252) / volatility if volatility != 0 else 0
            
            # Calculate beta
            market_data = pd.DataFrame({
                'stock': returns,
                'market': benchmark_returns
            }).dropna()
            
            beta = stats.linregress(market_data['market'], market_data['stock']).slope
            
            # Maximum drawdown
            rolling_max = data['Adj Close'].expanding().max()
            drawdowns = (data['Adj Close'] - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            position_metrics.append({
                'symbol': position.symbol,
                'quantity': position.quantity,
                'current_price': current_price,
                'average_price': position.average_price,
                'market_value': position_value,
                'cost_basis': position_cost,
                'unrealized_pl': position_value - position_cost,
                'unrealized_pl_pct': ((position_value - position_cost) / position_cost * 100) if position_cost != 0 else 0,
                'daily_return': daily_return * 100,
                'volatility': volatility * np.sqrt(252) * 100,  # Annualized
                'sharpe_ratio': sharpe_ratio * np.sqrt(252),    # Annualized
                'beta': beta,
                'max_drawdown': max_drawdown * 100
            })
            
            total_value += position_value
            total_cost += position_cost
            daily_returns.append(returns * position_value)
            
        except Exception as e:
            print(f"Error calculating metrics for {position.symbol}: {str(e)}")
    
    if not position_metrics:
        return None
        
    # Calculate portfolio-level metrics
    portfolio_returns = pd.concat(daily_returns, axis=1).sum(axis=1)
    portfolio_return = portfolio_returns.mean()
    portfolio_volatility = portfolio_returns.std()
    portfolio_sharpe = (portfolio_return - 0.02/252) / portfolio_volatility if portfolio_volatility != 0 else 0
    
    # Calculate portfolio beta
    portfolio_data = pd.DataFrame({
        'portfolio': portfolio_returns,
        'market': benchmark_returns
    }).dropna()
    
    portfolio_beta = stats.linregress(portfolio_data['market'], portfolio_data['portfolio']).slope
    
    # Portfolio drawdown
    portfolio_values = (1 + portfolio_returns).cumprod()
    rolling_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    portfolio_max_drawdown = drawdowns.min()
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'unrealized_pl': total_value - total_cost,
        'unrealized_pl_pct': ((total_value - total_cost) / total_cost * 100) if total_cost != 0 else 0,
        'daily_return': portfolio_return * 100,
        'volatility': portfolio_volatility * np.sqrt(252) * 100,  # Annualized
        'sharpe_ratio': portfolio_sharpe * np.sqrt(252),          # Annualized
        'beta': portfolio_beta,
        'max_drawdown': portfolio_max_drawdown * 100,
        'positions': position_metrics
    }

def calculate_correlation_matrix(positions, start_date=None, end_date=None):
    """Calculate correlation matrix for portfolio positions"""
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    returns_data = {}
    
    for position in positions:
        try:
            data = yf.download(position.symbol, start=start_date, end=end_date)
            if not data.empty:
                returns_data[position.symbol] = data['Adj Close'].pct_change().dropna()
        except Exception as e:
            print(f"Error getting data for {position.symbol}: {str(e)}")
    
    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr().round(2)
        return correlation_matrix.to_dict()
    
    return None
