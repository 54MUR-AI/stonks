from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime
from ..services.risk.metrics import RiskMetrics, get_risk_metrics

router = APIRouter(prefix="/api/risk")

class PortfolioPosition(BaseModel):
    symbol: str
    quantity: float
    price: Optional[float] = None

class RiskRequest(BaseModel):
    positions: Dict[str, float]
    include_stress_tests: bool = True
    benchmark: Optional[str] = 'SPY'

class VaRRequest(BaseModel):
    positions: Dict[str, float]
    method: str = 'historical'  # 'historical', 'parametric', or 'modified'
    confidence_level: Optional[float] = None

@router.post("/metrics")
async def calculate_risk_metrics(
    request: RiskRequest,
    risk_metrics: RiskMetrics = Depends(get_risk_metrics)
):
    """Calculate comprehensive risk metrics for a portfolio"""
    try:
        metrics = await risk_metrics.calculate_risk_metrics(
            request.positions,
            request.include_stress_tests
        )
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/volatility")
async def calculate_volatility(
    positions: Dict[str, float],
    risk_metrics: RiskMetrics = Depends(get_risk_metrics)
):
    """Calculate portfolio volatility"""
    try:
        volatility = await risk_metrics.calculate_volatility(
            list(positions.keys()),
            positions
        )
        return {"volatility": volatility}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/var")
async def calculate_var(
    request: VaRRequest,
    risk_metrics: RiskMetrics = Depends(get_risk_metrics)
):
    """Calculate Value at Risk"""
    try:
        if request.confidence_level:
            risk_metrics.confidence_level = request.confidence_level
            
        var = await risk_metrics.calculate_var(
            list(request.positions.keys()),
            request.positions,
            request.method
        )
        return {"value_at_risk": var}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/expected-shortfall")
async def calculate_expected_shortfall(
    positions: Dict[str, float],
    risk_metrics: RiskMetrics = Depends(get_risk_metrics)
):
    """Calculate Expected Shortfall (Conditional VaR)"""
    try:
        es = await risk_metrics.calculate_es(
            list(positions.keys()),
            positions
        )
        return {"expected_shortfall": es}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/beta")
async def calculate_beta(
    positions: Dict[str, float],
    benchmark: str = 'SPY',
    risk_metrics: RiskMetrics = Depends(get_risk_metrics)
):
    """Calculate portfolio beta relative to benchmark"""
    try:
        beta = await risk_metrics.calculate_beta(
            list(positions.keys()),
            positions,
            benchmark
        )
        return {"beta": beta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sharpe-ratio")
async def calculate_sharpe_ratio(
    positions: Dict[str, float],
    risk_metrics: RiskMetrics = Depends(get_risk_metrics)
):
    """Calculate Sharpe Ratio"""
    try:
        sharpe = await risk_metrics.calculate_sharpe_ratio(
            list(positions.keys()),
            positions
        )
        return {"sharpe_ratio": sharpe}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk-contribution")
async def calculate_risk_contribution(
    positions: Dict[str, float],
    risk_metrics: RiskMetrics = Depends(get_risk_metrics)
):
    """Calculate risk contribution of each position"""
    try:
        risk_contrib = await risk_metrics.calculate_risk_contribution(
            list(positions.keys()),
            positions
        )
        return {"risk_contribution": risk_contrib}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stress-test")
async def run_stress_tests(
    positions: Dict[str, float],
    risk_metrics: RiskMetrics = Depends(get_risk_metrics)
):
    """Run stress test scenarios"""
    try:
        metrics = await risk_metrics.calculate_risk_metrics(
            positions,
            include_stress_tests=True
        )
        return {"stress_tests": metrics['stress_tests']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
