from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class PortfolioBase(BaseModel):
    name: str

class PortfolioCreate(PortfolioBase):
    pass

class Portfolio(PortfolioBase):
    id: int
    owner_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class PositionBase(BaseModel):
    symbol: str
    quantity: float
    average_price: float

class PositionCreate(PositionBase):
    portfolio_id: int

class Position(PositionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class TradeBase(BaseModel):
    type: str
    quantity: float
    price: float

class TradeCreate(TradeBase):
    position_id: int

class Trade(TradeBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class WatchlistBase(BaseModel):
    name: str
    symbols: List[str]

class WatchlistCreate(WatchlistBase):
    pass

class Watchlist(WatchlistBase):
    id: int
    owner_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class AlertBase(BaseModel):
    symbol: str
    condition: str
    price: float

class AlertCreate(AlertBase):
    pass

class Alert(AlertBase):
    id: int
    owner_id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
