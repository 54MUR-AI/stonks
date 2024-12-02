from pydantic import BaseModel, Field
from typing import Dict, Optional, Union, List

class IndicatorParams(BaseModel):
    period: Optional[int] = Field(None, description="Period for calculations (e.g., 20 for SMA)")
    std_dev: Optional[int] = Field(None, description="Standard deviations for Bollinger Bands")
    fast_period: Optional[int] = Field(None, description="Fast period for MACD")
    slow_period: Optional[int] = Field(None, description="Slow period for MACD")
    signal_period: Optional[int] = Field(None, description="Signal period for MACD")
    rsi_period: Optional[int] = Field(None, description="Period for RSI calculation")
    ma_type: Optional[str] = Field(None, description="Moving average type (SMA/EMA)")
    
class TechnicalIndicator(BaseModel):
    name: str = Field(..., description="Indicator name (SMA/EMA/BB/RSI/MACD)")
    params: Optional[IndicatorParams] = Field(None, description="Indicator parameters")
    color: Optional[str] = Field(None, description="Display color for the indicator")
    visible: Optional[bool] = Field(True, description="Whether the indicator is visible")
    
class IndicatorRequest(BaseModel):
    indicators: List[TechnicalIndicator]
    
    class Config:
        schema_extra = {
            "example": {
                "indicators": [
                    {
                        "name": "SMA",
                        "params": {"period": 20},
                        "color": "#FF0000",
                        "visible": True
                    },
                    {
                        "name": "BB",
                        "params": {
                            "period": 20,
                            "std_dev": 2
                        },
                        "color": "#0000FF",
                        "visible": True
                    }
                ]
            }
        }
