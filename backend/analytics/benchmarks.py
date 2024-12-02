import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from ..schemas.benchmarks import BenchmarkInfo, BenchmarkMetrics, BenchmarkPerformance, BenchmarkData

# Standard benchmark definitions
BENCHMARK_DEFINITIONS = {
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "description": "Tracks the S&P 500 Index",
        "category": "Market"
    },
    "QQQ": {
        "name": "Invesco QQQ Trust",
        "description": "Tracks the Nasdaq-100 Index",
        "category": "Market"
    },
    "DIA": {
        "name": "SPDR Dow Jones Industrial Average ETF",
        "description": "Tracks the Dow Jones Industrial Average",
        "category": "Market"
    },
    "IWM": {
        "name": "iShares Russell 2000 ETF",
        "description": "Tracks the Russell 2000 Index",
        "category": "Market"
    }
}

def get_benchmark_info(symbol: str) -> Optional[BenchmarkInfo]:
    """Get benchmark information"""
    if symbol not in BENCHMARK_DEFINITIONS:
        return None
        
    info = BENCHMARK_DEFINITIONS[symbol]
    return BenchmarkInfo(
        symbol=symbol,
        name=info["name"],
        description=info["description"],
        category=info["category"]
    )

def calculate_benchmark_metrics(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> BenchmarkMetrics:
    """Calculate advanced benchmark comparison metrics"""
    # Calculate risk-free rate (using 3-month Treasury yield as proxy)
    risk_free_rate = 0.05  # This should be fetched from a reliable source
    
    # Calculate beta
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance
    
    # Calculate alpha (Jensen's Alpha)
    portfolio_return = portfolio_returns.mean() * 252
    benchmark_return = benchmark_returns.mean() * 252
    alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    # Calculate R-squared
    correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0][1]
    r_squared = correlation ** 2
    
    # Calculate tracking error
    tracking_diff = portfolio_returns - benchmark_returns
    tracking_error = np.std(tracking_diff) * np.sqrt(252)
    
    # Calculate information ratio
    excess_return = portfolio_return - benchmark_return
    information_ratio = excess_return / tracking_error if tracking_error != 0 else 0
    
    return BenchmarkMetrics(
        alpha=alpha,
        beta=beta,
        r_squared=r_squared,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        correlation=correlation
    )

def calculate_benchmark_performance(data: pd.Series) -> BenchmarkPerformance:
    """Calculate benchmark performance metrics"""
    current_price = data.iloc[-1]
    returns = {
        "ytd_return": calculate_ytd_return(data),
        "one_month": calculate_period_return(data, days=30),
        "three_month": calculate_period_return(data, days=90),
        "six_month": calculate_period_return(data, days=180),
        "one_year": calculate_period_return(data, days=365),
        "three_year": calculate_period_return(data, days=1095),
        "five_year": calculate_period_return(data, days=1825)
    }
    
    return BenchmarkPerformance(**returns)

def calculate_ytd_return(data: pd.Series) -> float:
    """Calculate year-to-date return"""
    current_year = datetime.now().year
    ytd_start = data[data.index.year == current_year].iloc[0]
    return (data.iloc[-1] / ytd_start - 1) * 100

def calculate_period_return(data: pd.Series, days: int) -> Optional[float]:
    """Calculate return for a specific period"""
    if len(data) < days:
        return None
    return (data.iloc[-1] / data.iloc[-days] - 1) * 100

def get_benchmark_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    normalize: bool = False
) -> Optional[BenchmarkData]:
    """Get comprehensive benchmark data including metrics and performance"""
    try:
        # Get benchmark info
        info = get_benchmark_info(symbol)
        if not info:
            return None
            
        # Fetch data from Yahoo Finance
        benchmark = yf.Ticker(symbol)
        hist = benchmark.history(start=start_date, end=end_date)
        if hist.empty:
            return None
            
        # Calculate daily returns
        prices = hist['Close']
        if normalize:
            prices = prices / prices.iloc[0] * 100
            
        # Format historical values
        historical_values = [
            {
                "time": index.strftime("%Y-%m-%d"),
                "value": float(value)
            }
            for index, value in prices.items()
        ]
        
        # Calculate performance metrics
        performance = calculate_benchmark_performance(prices)
        
        # Calculate comparison metrics if we have enough data
        returns = prices.pct_change().dropna()
        metrics = BenchmarkMetrics(
            alpha=0.0,
            beta=1.0,
            r_squared=1.0,
            tracking_error=0.0,
            information_ratio=0.0,
            correlation=1.0
        )
        
        return BenchmarkData(
            info=info,
            metrics=metrics,
            performance=performance,
            historical_values=historical_values,
            last_updated=datetime.now()
        )
        
    except Exception as e:
        print(f"Error fetching benchmark data for {symbol}: {str(e)}")
        return None

def compare_with_benchmarks(
    portfolio_values: pd.Series,
    benchmark_symbols: List[str],
    normalize: bool = True
) -> Dict[str, BenchmarkData]:
    """Compare portfolio performance with multiple benchmarks"""
    start_date = portfolio_values.index[0]
    end_date = portfolio_values.index[-1]
    
    results = {}
    portfolio_returns = portfolio_values.pct_change().dropna()
    
    for symbol in benchmark_symbols:
        benchmark_data = get_benchmark_data(symbol, start_date, end_date, normalize)
        if benchmark_data:
            # Calculate benchmark returns
            benchmark_values = pd.Series(
                [point['value'] for point in benchmark_data.historical_values],
                index=pd.to_datetime([point['time'] for point in benchmark_data.historical_values])
            )
            benchmark_returns = benchmark_values.pct_change().dropna()
            
            # Calculate comparison metrics
            benchmark_data.metrics = calculate_benchmark_metrics(
                portfolio_returns,
                benchmark_returns
            )
            
            results[symbol] = benchmark_data
            
    return results
