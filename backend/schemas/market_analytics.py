from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class MarketImpact(BaseModel):
    price_impact: float = Field(..., description="Estimated price impact in basis points")
    volume_participation: float = Field(..., description="Percentage of ADV")
    market_impact_cost: float = Field(..., description="Estimated market impact cost")
    recovery_time: Optional[float] = Field(None, description="Estimated price recovery time in minutes")

class TransactionCosts(BaseModel):
    commission: float = Field(..., description="Trading commission")
    spread_cost: float = Field(..., description="Half-spread cost")
    market_impact: float = Field(..., description="Market impact cost")
    total_cost: float = Field(..., description="Total transaction cost")
    total_cost_bps: float = Field(..., description="Total cost in basis points")

class MarketMicrostructure(BaseModel):
    avg_trade_size: float
    volatility: float
    volume_profile: Dict[int, float]
    serial_correlation: float
    volume_volatility_corr: float
    bid_ask_bounce: float
    effective_spread: float
    price_impact: float

class LiquidityProfile(BaseModel):
    avg_daily_volume: float
    turnover: float
    volume_profile: Dict[int, float]
    peak_volume_hour: int
    volatility_adjusted_volume: float
    liquidity_cost_score: float
    spread_proxy: float

class TradingHour(BaseModel):
    hour: int
    score: float
    volume_percentile: float
    volatility_percentile: float

class TradingPatterns(BaseModel):
    volume_profile: Dict[int, float]
    volatility_profile: Dict[int, float]
    volume_volatility_correlation: float
    short_term_autocorr: float
    hourly_autocorr: float
    impact_decay: Dict[str, float]
    optimal_trading_hours: List[TradingHour]

class MarketDepth(BaseModel):
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    last_size: int
    spread: float
    timestamp: datetime

class MarketAnalytics(BaseModel):
    symbol: str
    microstructure: MarketMicrostructure
    liquidity_profile: LiquidityProfile
    trading_patterns: TradingPatterns
    market_depth: MarketDepth
    last_updated: datetime

    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "microstructure": {
                    "avg_trade_size": 100,
                    "volatility": 0.15,
                    "volume_profile": {
                        "9": 1000000,
                        "10": 1200000
                    },
                    "serial_correlation": 0.1,
                    "volume_volatility_corr": 0.3,
                    "bid_ask_bounce": 0.0001,
                    "effective_spread": 0.0002,
                    "price_impact": 0.00001
                },
                "liquidity_profile": {
                    "avg_daily_volume": 5000000,
                    "turnover": 1000000000,
                    "volume_profile": {
                        "9": 1000000,
                        "10": 1200000
                    },
                    "peak_volume_hour": 10,
                    "volatility_adjusted_volume": 4000000,
                    "liquidity_cost_score": 0.0005,
                    "spread_proxy": 0.0001
                },
                "trading_patterns": {
                    "volume_profile": {
                        "9": 1000000,
                        "10": 1200000
                    },
                    "volatility_profile": {
                        "9": 0.001,
                        "10": 0.002
                    },
                    "volume_volatility_correlation": 0.3,
                    "short_term_autocorr": 0.1,
                    "hourly_autocorr": 0.05,
                    "impact_decay": {
                        "lag_1": 0.00001,
                        "lag_2": 0.000005
                    },
                    "optimal_trading_hours": [
                        {
                            "hour": 10,
                            "score": 1.0,
                            "volume_percentile": 0.9,
                            "volatility_percentile": 0.5
                        }
                    ]
                },
                "market_depth": {
                    "bid_price": 150.0,
                    "ask_price": 150.1,
                    "bid_size": 100,
                    "ask_size": 100,
                    "last_price": 150.05,
                    "last_size": 100,
                    "spread": 0.0007,
                    "timestamp": "2024-01-01T10:00:00"
                },
                "last_updated": "2024-01-01T10:00:00"
            }
        }
