from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from decimal import Decimal
from datetime import datetime

class TargetAllocation(BaseModel):
    symbol: str = Field(..., description="Asset symbol")
    target_weight: float = Field(..., description="Target portfolio weight (0-100)", ge=0, le=100)
    tolerance: float = Field(1.0, description="Rebalancing tolerance in percentage points")

    @validator('target_weight')
    def validate_weight(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Weight must be between 0 and 100')
        return v

class RebalanceConstraints(BaseModel):
    min_trade_amount: float = Field(100.0, description="Minimum trade amount in USD")
    max_trade_amount: Optional[float] = Field(None, description="Maximum trade amount in USD")
    tax_loss_harvest: bool = Field(False, description="Consider tax loss harvesting")
    minimize_trades: bool = Field(True, description="Minimize number of trades")
    cash_buffer: float = Field(0.0, description="Minimum cash buffer to maintain (%)")

class TradeAction(BaseModel):
    symbol: str
    action: str = Field(..., description="buy or sell")
    shares: int
    estimated_value: float
    current_weight: float
    target_weight: float
    price_estimate: float

class RebalanceAnalysis(BaseModel):
    current_total: float
    target_total: float
    trades: List[TradeAction]
    estimated_commission: float = Field(0.0, description="Estimated trading commissions")
    tax_impact: Optional[float] = Field(None, description="Estimated tax impact")
    risk_score_before: float
    risk_score_after: float
    tracking_error_impact: float
    cash_position: float

class RebalanceRequest(BaseModel):
    portfolio_id: int
    target_allocations: List[TargetAllocation]
    constraints: RebalanceConstraints = Field(default_factory=RebalanceConstraints)
    analysis_only: bool = Field(True, description="Only analyze without executing trades")

    class Config:
        schema_extra = {
            "example": {
                "portfolio_id": 1,
                "target_allocations": [
                    {"symbol": "SPY", "target_weight": 60, "tolerance": 2.0},
                    {"symbol": "AGG", "target_weight": 40, "tolerance": 2.0}
                ],
                "constraints": {
                    "min_trade_amount": 100,
                    "max_trade_amount": 10000,
                    "tax_loss_harvest": True,
                    "minimize_trades": True,
                    "cash_buffer": 2.0
                },
                "analysis_only": True
            }
        }

class RebalanceStrategy(BaseModel):
    name: str = Field(..., description="Strategy name")
    description: str
    target_allocations: List[TargetAllocation]
    risk_score: float = Field(..., description="Strategy risk score (1-10)")
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Conservative Growth",
                "description": "60/40 portfolio with focus on stability",
                "target_allocations": [
                    {"symbol": "SPY", "target_weight": 60, "tolerance": 2.0},
                    {"symbol": "AGG", "target_weight": 40, "tolerance": 2.0}
                ],
                "risk_score": 5.0,
                "expected_return": 7.0,
                "expected_volatility": 10.0,
                "sharpe_ratio": 0.5
            }
        }
