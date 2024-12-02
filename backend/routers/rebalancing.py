from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
from datetime import datetime, timedelta

from ..database import get_db
from ..models import Portfolio, Position
from ..schemas.rebalancing import (
    RebalanceRequest, RebalanceAnalysis, RebalanceStrategy,
    TargetAllocation
)
from ..analytics.rebalancing import (
    generate_rebalancing_plan, get_predefined_strategies
)
from ..analytics.market_data import get_historical_data

router = APIRouter()

@router.post("/portfolios/{portfolio_id}/rebalance", response_model=RebalanceAnalysis)
async def rebalance_portfolio(
    portfolio_id: int,
    rebalance_request: RebalanceRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze or execute portfolio rebalancing
    """
    # Validate portfolio
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get current positions
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    positions_data = [
        {
            "symbol": pos.symbol,
            "quantity": pos.quantity,
            "avg_price": pos.avg_price,
            "current_value": pos.quantity * pos.current_price
        }
        for pos in positions
    ]
    
    # Get historical data for analysis
    symbols = [t.symbol for t in rebalance_request.target_allocations]
    historical_data = get_historical_data(symbols, days=252)  # 1 year of data
    
    # Generate rebalancing plan
    try:
        analysis = generate_rebalancing_plan(
            positions_data,
            rebalance_request.target_allocations,
            rebalance_request.constraints,
            historical_data
        )
        
        # Execute trades if requested
        if not rebalance_request.analysis_only:
            # TODO: Implement trade execution
            pass
            
        return analysis
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate rebalancing plan: {str(e)}"
        )

@router.get("/portfolios/{portfolio_id}/rebalance/strategies", response_model=List[RebalanceStrategy])
async def get_portfolio_strategies(
    portfolio_id: int,
    risk_score: Optional[float] = Query(None, ge=1, le=10),
    db: Session = Depends(get_db)
):
    """
    Get predefined portfolio strategies
    """
    # Validate portfolio
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    strategies = get_predefined_strategies()
    
    # Filter by risk score if provided
    if risk_score is not None:
        strategies = [
            s for s in strategies
            if abs(s.risk_score - risk_score) <= 1
        ]
    
    return strategies

@router.get("/portfolios/{portfolio_id}/rebalance/drift", response_model=List[Dict])
async def get_portfolio_drift(
    portfolio_id: int,
    db: Session = Depends(get_db)
):
    """
    Calculate current portfolio drift from target allocations
    """
    # Validate portfolio
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get current positions and target allocations
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    
    # Calculate total value and current weights
    total_value = sum(pos.quantity * pos.current_price for pos in positions)
    
    drift_analysis = []
    for pos in positions:
        current_weight = (pos.quantity * pos.current_price / total_value) * 100
        target_weight = pos.target_weight if hasattr(pos, 'target_weight') else None
        
        drift_analysis.append({
            "symbol": pos.symbol,
            "current_weight": current_weight,
            "target_weight": target_weight,
            "drift": current_weight - target_weight if target_weight else None
        })
    
    return drift_analysis
