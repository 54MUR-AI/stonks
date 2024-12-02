import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

def get_benchmark_data(symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, float]]:
    """
    Get historical data for a benchmark index
    
    Args:
        symbol: Benchmark symbol (e.g., 'SPY' for S&P 500)
        start_date: Start date for historical data
        end_date: End date for historical data
    
    Returns:
        List of price points formatted for charting
    """
    try:
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        benchmark = yf.Ticker(symbol)
        data = benchmark.history(start=start_date, end=end_date)
        
        if data.empty:
            return []
            
        # Calculate percentage change from first value
        first_value = data['Close'].iloc[0]
        normalized_values = (data['Close'] / first_value - 1) * 100
        
        return [
            {
                "time": index.strftime("%Y-%m-%d"),
                "value": value
            }
            for index, value in normalized_values.items()
        ]
        
    except Exception as e:
        print(f"Error fetching benchmark data for {symbol}: {e}")
        return []

def get_benchmark_info(symbol: str) -> Dict[str, any]:
    """Get basic information about a benchmark index"""
    try:
        benchmark = yf.Ticker(symbol)
        info = benchmark.info
        
        return {
            "name": info.get("shortName", symbol),
            "description": info.get("longBusinessSummary", ""),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            "current_price": info.get("regularMarketPrice", 0),
            "day_change": info.get("regularMarketChangePercent", 0)
        }
    except Exception as e:
        print(f"Error fetching benchmark info for {symbol}: {e}")
        return {
            "name": symbol,
            "description": "",
            "currency": "USD",
            "exchange": "",
            "current_price": 0,
            "day_change": 0
        }
