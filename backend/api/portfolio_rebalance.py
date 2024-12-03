from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict
from pydantic import BaseModel
from ..services.portfolio.rebalancer import (
    PortfolioRebalancer,
    PortfolioHolding,
    RebalanceResult,
    RebalanceAction
)
from ..services.market_data.provider import get_market_data_provider
from ..services.risk.metrics import get_risk_metrics

router = APIRouter(prefix="/api/portfolio")

class RebalanceRequest(BaseModel):
    holdings: List[PortfolioHolding]
    cash: float = 0.0
    constraints: Optional[Dict] = None
    rebalance_threshold: Optional[float] = None
    min_trade_value: Optional[float] = None
    max_turnover: Optional[float] = None

async def get_rebalancer(
    rebalance_threshold: Optional[float] = None,
    min_trade_value: Optional[float] = None,
    max_turnover: Optional[float] = None
) -> PortfolioRebalancer:
    market_data = await get_market_data_provider()
    risk_metrics = await get_risk_metrics()
    
    kwargs = {}
    if rebalance_threshold is not None:
        kwargs['rebalance_threshold'] = rebalance_threshold
    if min_trade_value is not None:
        kwargs['min_trade_value'] = min_trade_value
    if max_turnover is not None:
        kwargs['max_turnover'] = max_turnover
        
    return PortfolioRebalancer(
        market_data_provider=market_data,
        risk_metrics=risk_metrics,
        **kwargs
    )

@router.post("/rebalance", response_model=RebalanceResult)
async def rebalance_portfolio(
    request: RebalanceRequest,
    rebalancer: PortfolioRebalancer = Depends(get_rebalancer)
):
    """
    Calculate rebalancing actions for a portfolio
    """
    try:
        result = await rebalancer.calculate_rebalance_actions(
            request.holdings,
            request.cash
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize", response_model=RebalanceResult)
async def optimize_rebalance(
    request: RebalanceRequest,
    rebalancer: PortfolioRebalancer = Depends(get_rebalancer)
):
    """
    Optimize rebalancing actions with additional constraints
    """
    try:
        result = await rebalancer.optimize_rebalance(
            request.holdings,
            request.cash,
            request.constraints
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate", response_model=bool)
async def validate_rebalance(
    holdings: List[PortfolioHolding],
    actions: List[RebalanceAction],
    rebalancer: PortfolioRebalancer = Depends(get_rebalancer)
):
    """
    Validate that rebalancing actions are valid and executable
    """
    try:
        result = await rebalancer.validate_rebalance(holdings, actions)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=Dict)
async def analyze_portfolio(
    holdings: List[PortfolioHolding],
    rebalancer: PortfolioRebalancer = Depends(get_rebalancer)
):
    """
    Analyze current portfolio state and calculate metrics
    """
    try:
        weights, total_value = await rebalancer.analyze_portfolio(holdings)
        return {
            "current_weights": weights,
            "total_value": total_value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
