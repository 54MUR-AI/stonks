from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime

class HistoricalValue(BaseModel):
    date: str
    value: float

class PositionMetrics(BaseModel):
    symbol: str
    weight: float
    value: float
    return_pct: float

class PortfolioMetrics(BaseModel):
    currentValue: float
    dayChange: float
    annualReturn: float
    volatility: float
    sharpeRatio: float
    historicalValue: List[HistoricalValue]
    correlationMatrix: Dict[str, Dict[str, float]]
    positions: List[PositionMetrics]
