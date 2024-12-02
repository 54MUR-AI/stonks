from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class BenchmarkInfo(BaseModel):
    symbol: str = Field(..., description="Benchmark symbol (e.g., SPY)")
    name: str = Field(..., description="Full name of the benchmark")
    description: Optional[str] = Field(None, description="Description of the benchmark")
    category: str = Field(..., description="Category (Market, Sector, Custom)")
    currency: str = Field("USD", description="Trading currency")
    
class BenchmarkMetrics(BaseModel):
    alpha: float = Field(..., description="Jensen's Alpha relative to benchmark")
    beta: float = Field(..., description="Beta coefficient relative to benchmark")
    r_squared: float = Field(..., description="R-squared value of regression")
    tracking_error: float = Field(..., description="Tracking error vs benchmark")
    information_ratio: float = Field(..., description="Information ratio")
    correlation: float = Field(..., description="Correlation with benchmark")
    
class BenchmarkPerformance(BaseModel):
    ytd_return: float = Field(..., description="Year-to-date return")
    one_month: float = Field(..., description="1-month return")
    three_month: float = Field(..., description="3-month return")
    six_month: float = Field(..., description="6-month return")
    one_year: float = Field(..., description="1-year return")
    three_year: Optional[float] = Field(None, description="3-year annualized return")
    five_year: Optional[float] = Field(None, description="5-year annualized return")
    
class BenchmarkData(BaseModel):
    info: BenchmarkInfo
    metrics: BenchmarkMetrics
    performance: BenchmarkPerformance
    historical_values: List[Dict[str, float]] = Field(..., description="Historical price data")
    last_updated: datetime = Field(..., description="Last data update timestamp")
    
class BenchmarkRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of benchmark symbols to compare against")
    normalize: bool = Field(True, description="Whether to normalize returns for comparison")
    include_metrics: bool = Field(True, description="Whether to include detailed metrics")
    include_performance: bool = Field(True, description="Whether to include performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "symbols": ["SPY", "QQQ"],
                "normalize": True,
                "include_metrics": True,
                "include_performance": True
            }
        }
