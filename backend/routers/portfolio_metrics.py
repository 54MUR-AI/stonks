from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ..database import get_db
from ..models import Portfolio, Position
from ..schemas.portfolio import PortfolioMetrics, IndicatorRequest, BenchmarkRequest
from ..analytics import calculate_portfolio_metrics, calculate_correlation_matrix
from ..analytics.indicators import calculate_all_indicators
from ..analytics.benchmarks import get_benchmark_data, get_benchmark_info, compare_with_benchmarks

router = APIRouter()

def get_time_range_dates(time_range: str) -> tuple[datetime, datetime]:
    """Convert time range string to start and end dates"""
    end_date = datetime.now()
    
    if time_range == '1D':
        start_date = end_date - timedelta(days=1)
    elif time_range == '1W':
        start_date = end_date - timedelta(weeks=1)
    elif time_range == '1M':
        start_date = end_date - timedelta(days=30)
    elif time_range == '3M':
        start_date = end_date - timedelta(days=90)
    elif time_range == '6M':
        start_date = end_date - timedelta(days=180)
    elif time_range == '1Y':
        start_date = end_date - timedelta(days=365)
    else:  # ALL
        start_date = end_date - timedelta(days=365*5)  # 5 years
        
    return start_date, end_date

@router.get("/portfolios/{portfolio_id}/metrics", response_model=PortfolioMetrics)
async def get_portfolio_metrics(
    portfolio_id: int,
    time_range: str = Query('1M', regex='^(1D|1W|1M|3M|6M|1Y|ALL)$'),
    benchmark: Optional[str] = Query(None, regex='^(SPY|QQQ|DIA|IWM)$'),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive metrics for a portfolio including:
    - Current value and daily change
    - Historical value data
    - Risk metrics (volatility, Sharpe ratio)
    - Returns (daily, annual)
    - Technical indicators
    - Benchmark comparison
    """
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Get positions
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    
    # Get date range
    start_date, end_date = get_time_range_dates(time_range)
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(positions, start_date, end_date)
    if not metrics:
        raise HTTPException(status_code=500, detail="Failed to calculate portfolio metrics")

    # Calculate correlation matrix
    correlation_matrix = calculate_correlation_matrix(positions, start_date, end_date)

    # Get historical portfolio values
    portfolio_values = pd.Series(
        [point['value'] for point in metrics['historical_values']],
        index=pd.to_datetime([point['date'] for point in metrics['historical_values']])
    )

    # Get benchmark data if requested
    benchmark_data = None
    if benchmark:
        benchmark_data = get_benchmark_data(benchmark, start_date, end_date)
        if not benchmark_data:
            raise HTTPException(status_code=500, detail=f"Failed to fetch benchmark data for {benchmark}")

    return {
        "currentValue": metrics['total_value'],
        "dayChange": metrics.get('daily_return', 0),
        "annualReturn": metrics.get('annual_return', 0) * 100,
        "volatility": metrics.get('volatility', 0) * 100,
        "sharpeRatio": metrics.get('sharpe_ratio', 0),
        "historicalValue": [
            {
                "time": point['date'],
                "value": point['value']
            }
            for point in metrics['historical_values']
        ],
        "correlationMatrix": correlation_matrix,
        "positions": [
            {
                "symbol": pos.symbol,
                "weight": pos.quantity * pos.avg_price / metrics['total_value'],
                "value": pos.quantity * pos.avg_price,
                "return": metrics.get('position_returns', {}).get(pos.symbol, 0) * 100,
                "volatility": metrics.get('position_volatility', {}).get(pos.symbol, 0) * 100
            }
            for pos in positions
        ],
        "benchmarkData": benchmark_data if benchmark else None
    }

@router.post("/portfolios/{portfolio_id}/indicators")
async def calculate_portfolio_indicators(
    portfolio_id: int,
    indicator_request: IndicatorRequest,
    time_range: str = Query('1M', regex='^(1D|1W|1M|3M|6M|1Y|ALL)$'),
    db: Session = Depends(get_db)
):
    """
    Calculate technical indicators for a portfolio with custom parameters
    """
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Get positions
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    
    # Get date range
    start_date, end_date = get_time_range_dates(time_range)
    
    # Calculate metrics to get historical values
    metrics = calculate_portfolio_metrics(positions, start_date, end_date)
    if not metrics:
        raise HTTPException(status_code=500, detail="Failed to calculate portfolio metrics")

    # Get historical portfolio values
    portfolio_values = pd.Series(
        [point['value'] for point in metrics['historical_values']],
        index=pd.to_datetime([point['date'] for point in metrics['historical_values']])
    )

    # Calculate indicators with custom parameters
    try:
        indicators = calculate_all_indicators(portfolio_values, indicator_request.indicators)
        return {"indicators": indicators}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate indicators: {str(e)}")

@router.post("/portfolios/{portfolio_id}/benchmark-comparison")
async def compare_portfolio_benchmarks(
    portfolio_id: int,
    benchmark_request: BenchmarkRequest,
    time_range: str = Query('1M', regex='^(1D|1W|1M|3M|6M|1Y|ALL)$'),
    db: Session = Depends(get_db)
):
    """
    Compare portfolio performance against multiple benchmarks
    """
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Get positions
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    
    # Get date range
    start_date, end_date = get_time_range_dates(time_range)
    
    # Calculate metrics to get historical values
    metrics = calculate_portfolio_metrics(positions, start_date, end_date)
    if not metrics:
        raise HTTPException(status_code=500, detail="Failed to calculate portfolio metrics")

    # Get historical portfolio values
    portfolio_values = pd.Series(
        [point['value'] for point in metrics['historical_values']],
        index=pd.to_datetime([point['date'] for point in metrics['historical_values']])
    )

    # Compare with benchmarks
    try:
        benchmark_results = compare_with_benchmarks(
            portfolio_values,
            benchmark_request.symbols,
            benchmark_request.normalize
        )
        
        # Filter results based on request
        if not benchmark_request.include_metrics:
            for data in benchmark_results.values():
                data.metrics = None
                
        if not benchmark_request.include_performance:
            for data in benchmark_results.values():
                data.performance = None
        
        return {
            "portfolio_value": metrics['total_value'],
            "portfolio_return": metrics.get('annual_return', 0) * 100,
            "benchmarks": benchmark_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare with benchmarks: {str(e)}")
