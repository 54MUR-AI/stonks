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
    is_public: bool = True

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
    portfolio_id: int
    symbol: constr(min_length=1, max_length=10)
    quantity: confloat(gt=0)
    average_price: Optional[float] = None

class PositionCreate(PositionBase):
    pass

class PositionResponse(PositionBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Trade schemas
class TradeBase(BaseModel):
    position_id: int
    type: constr(pattern='^(buy|sell)$')
    quantity: confloat(gt=0)
    price: confloat(gt=0)

class TradeCreate(TradeBase):
    pass

class TradeResponse(TradeBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

# Watchlist schemas
class WatchlistBase(BaseModel):
    name: constr(min_length=1, max_length=100)
    description: Optional[str] = None
    symbols: List[str]

class WatchlistCreate(WatchlistBase):
    pass

class WatchlistResponse(WatchlistBase):
    id: int
    owner_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Alert schemas
class AlertBase(BaseModel):
    symbol: constr(min_length=1, max_length=10)
    condition: constr(pattern='^(above|below)$')
    price: confloat(gt=0)
    is_active: bool = True

class AlertCreate(AlertBase):
    pass

class AlertResponse(AlertBase):
    id: int
    owner_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Comment schemas
class CommentBase(BaseModel):
    portfolio_id: int
    content: constr(min_length=1, max_length=1000)

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
class ActivityBase(BaseModel):
    activity_type: str
    target_id: Optional[int]
    target_type: Optional[str]
    data: Optional[Dict[str, Any]]

class ActivityCreate(ActivityBase):
    pass

class ActivityResponse(ActivityBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Notification schemas
class NotificationBase(BaseModel):
    notification_type: str
    data: Optional[Dict[str, Any]]
    is_read: bool = False

class NotificationCreate(NotificationBase):
    pass

class NotificationResponse(NotificationBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Market Data schemas
class MarketDataRequest(BaseModel):
    symbol: constr(min_length=1, max_length=10)

class TargetWeights(BaseModel):
    weights: Dict[str, confloat(ge=0, le=1)]

    @validator('weights')
    def validate_weights(cls, v):
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow for small floating point errors
            raise ValueError("Target weights must sum to 1.0")
        return v
