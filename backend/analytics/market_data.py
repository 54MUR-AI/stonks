import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_historical_data(
    symbols: List[str],
    days: int = 252,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Fetch historical market data for multiple symbols
    """
    if not end_date:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    def fetch_symbol(symbol: str) -> pd.Series:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            return hist['Close']
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.Series(dtype=float)
    
    # Fetch data in parallel
    with ThreadPoolExecutor(max_workers=min(len(symbols), 10)) as executor:
        future_to_symbol = {
            executor.submit(fetch_symbol, symbol): symbol
            for symbol in symbols
        }
        
        # Collect results
        data = {}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data[symbol] = future.result()
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                
    # Combine into DataFrame
    return pd.DataFrame(data)

def calculate_returns_metrics(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate return metrics for each symbol
    """
    returns = data.pct_change().dropna()
    
    metrics = {}
    for symbol in data.columns:
        symbol_returns = returns[symbol]
        
        metrics[symbol] = {
            "daily_return": symbol_returns.mean(),
            "annual_return": symbol_returns.mean() * 252,
            "volatility": symbol_returns.std() * np.sqrt(252),
            "sharpe_ratio": (symbol_returns.mean() * 252) / (symbol_returns.std() * np.sqrt(252))
            if symbol_returns.std() != 0 else 0
        }
    
    return metrics

def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix between assets
    """
    returns = data.pct_change().dropna()
    return returns.corr()

def calculate_portfolio_risk_metrics(
    weights: Dict[str, float],
    historical_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate portfolio-level risk metrics
    """
    returns = historical_data.pct_change().dropna()
    
    # Calculate portfolio returns
    portfolio_returns = sum(
        weight * returns[symbol]
        for symbol, weight in weights.items()
        if symbol in returns.columns
    )
    
    # Calculate metrics
    metrics = {
        "daily_return": portfolio_returns.mean(),
        "annual_return": portfolio_returns.mean() * 252,
        "volatility": portfolio_returns.std() * np.sqrt(252),
        "sharpe_ratio": (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
        if portfolio_returns.std() != 0 else 0,
        "var_95": portfolio_returns.quantile(0.05),  # 95% Value at Risk
        "cvar_95": portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean()  # Conditional VaR
    }
    
    return metrics

def get_market_summary(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Get current market summary for symbols
    """
    def fetch_summary(symbol: str) -> Dict[str, float]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "price": info.get("regularMarketPrice", 0),
                "change": info.get("regularMarketChangePercent", 0),
                "volume": info.get("regularMarketVolume", 0),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("forwardPE", 0),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
            }
        except Exception as e:
            print(f"Error fetching summary for {symbol}: {str(e)}")
            return {}
    
    # Fetch data in parallel
    with ThreadPoolExecutor(max_workers=min(len(symbols), 10)) as executor:
        future_to_symbol = {
            executor.submit(fetch_summary, symbol): symbol
            for symbol in symbols
        }
        
        # Collect results
        summary = {}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                summary[symbol] = future.result()
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                
    return summary

def estimate_transaction_costs(
    trades: List[Dict],
    commission_rate: float = 0.001
) -> Dict[str, float]:
    """
    Estimate transaction costs for proposed trades
    """
    total_value = sum(abs(trade['value']) for trade in trades)
    commission = total_value * commission_rate
    
    # Add spread costs (simplified estimate)
    spread_cost = total_value * 0.0005  # Assumes 5bps average spread
    
    return {
        "commission": commission,
        "spread_cost": spread_cost,
        "total_cost": commission + spread_cost
    }
