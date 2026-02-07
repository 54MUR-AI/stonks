# Schemas package - consolidated from schemas.py
from pydantic import BaseModel, EmailStr, constr, confloat, conint, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    username: constr(min_length=3, max_length=50)
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: constr(min_length=8)

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Authentication schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

# Portfolio schemas
class PortfolioBase(BaseModel):
    name: constr(min_length=1, max_length=100)
    description: Optional[str] = None
    cash: confloat(ge=0) = 10000.0

class PortfolioCreate(PortfolioBase):
    pass

class PortfolioResponse(PortfolioBase):
    id: int
    owner_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Position schemas
class PositionBase(BaseModel):
    symbol: constr(min_length=1, max_length=10)
    quantity: confloat(gt=0)
    average_price: confloat(gt=0)

class PositionCreate(PositionBase):
    portfolio_id: int

class PositionResponse(PositionBase):
    id: int
    portfolio_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Trade schemas
class TradeBase(BaseModel):
    symbol: constr(min_length=1, max_length=10)
    quantity: confloat(gt=0)
    price: confloat(gt=0)

class TradeCreate(TradeBase):
    portfolio_id: int
    type: str
    status: str = "pending"

class TradeResponse(TradeBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

# Watchlist schemas
class WatchlistBase(BaseModel):
    symbol: constr(min_length=1, max_length=10)
    notes: Optional[str] = None

class WatchlistCreate(WatchlistBase):
    pass

class WatchlistResponse(WatchlistBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Alert schemas
class AlertBase(BaseModel):
    symbol: constr(min_length=1, max_length=10)
    condition: str
    price: confloat(gt=0)
    is_active: bool = True

class AlertCreate(AlertBase):
    pass

class AlertResponse(AlertBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Comment schemas
class CommentBase(BaseModel):
    content: constr(min_length=1)

class CommentCreate(CommentBase):
    pass

class CommentResponse(CommentBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Activity schemas
class ActivityResponse(BaseModel):
    id: int
    activity_type: str
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Notification schemas
class NotificationResponse(BaseModel):
    id: int
    notification_type: str
    content: str
    is_read: bool
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Market Data schemas
class MarketDataRequest(BaseModel):
    symbols: List[str]
    period: str = "1d"

# Target Weights
class TargetWeights(BaseModel):
    weights: Dict[str, float]

    @validator('weights')
    def weights_sum_to_one(cls, v):
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError('Weights must sum to 1.0')
        return v
