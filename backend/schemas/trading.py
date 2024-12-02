from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import datetime, time
from enum import Enum

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

class ExecutionStrategy(str, Enum):
    IMMEDIATE = "immediate"  # Execute all trades immediately
    TWAP = "twap"          # Time-Weighted Average Price
    VWAP = "vwap"          # Volume-Weighted Average Price
    SMART = "smart"        # Smart order routing

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class TradeOrder(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide
    quantity: int = Field(..., gt=0)
    order_type: OrderType = Field(OrderType.MARKET)
    limit_price: Optional[float] = Field(None, description="Required for limit orders")
    stop_price: Optional[float] = Field(None, description="Required for stop orders")
    time_in_force: TimeInForce = Field(TimeInForce.DAY)
    
    @validator('limit_price')
    def validate_limit_price(cls, v, values):
        if values.get('order_type') in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Limit price required for limit orders')
        return v
    
    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        if values.get('order_type') in [OrderType.STOP, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Stop price required for stop orders')
        return v

class ExecutionParams(BaseModel):
    strategy: ExecutionStrategy = Field(ExecutionStrategy.IMMEDIATE)
    start_time: Optional[time] = Field(None, description="Start time for scheduled execution")
    end_time: Optional[time] = Field(None, description="End time for scheduled execution")
    max_participation_rate: Optional[float] = Field(
        None,
        description="Maximum participation rate in market volume",
        ge=0.0,
        le=1.0
    )
    min_trade_size: Optional[int] = Field(None, description="Minimum trade size")
    price_limit: Optional[float] = Field(None, description="Maximum/minimum price limit")

class TradeExecution(BaseModel):
    portfolio_id: int
    orders: List[TradeOrder]
    execution_params: ExecutionParams = Field(default_factory=ExecutionParams)
    dry_run: bool = Field(True, description="Simulate execution without actual trades")

    class Config:
        schema_extra = {
            "example": {
                "portfolio_id": 1,
                "orders": [
                    {
                        "symbol": "SPY",
                        "side": "buy",
                        "quantity": 100,
                        "order_type": "market",
                        "time_in_force": "day"
                    }
                ],
                "execution_params": {
                    "strategy": "smart",
                    "max_participation_rate": 0.1
                },
                "dry_run": True
            }
        }

class OrderUpdate(BaseModel):
    order_id: str
    status: OrderStatus
    filled_quantity: int = Field(0, ge=0)
    average_price: Optional[float]
    last_price: Optional[float]
    last_quantity: Optional[int]
    remaining_quantity: int
    timestamp: datetime
    message: Optional[str]

class ExecutionResult(BaseModel):
    success: bool
    orders: List[Dict[str, OrderUpdate]]
    total_cost: float
    average_price: Dict[str, float]
    execution_time: float  # in seconds
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

class MarketHours(BaseModel):
    market: str = Field(..., description="Market identifier (e.g., 'NYSE')")
    is_open: bool
    next_open: datetime
    next_close: datetime
    trading_hours: List[Dict[str, time]]
    holidays: List[datetime]

class MarketCondition(BaseModel):
    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    volume: int
    vwap: float
    volatility: float
    spread: float
    timestamp: datetime
