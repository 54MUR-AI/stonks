from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
from enum import Enum

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    portfolios = relationship("Portfolio", back_populates="owner")
    watchlists = relationship("Watchlist", back_populates="owner")
    alerts = relationship("Alert", back_populates="owner")
    shared_portfolios = relationship("PortfolioShare", back_populates="shared_with")
    following = relationship("UserFollow", foreign_keys=[UserFollow.follower_id], back_populates="follower")
    followers = relationship("UserFollow", foreign_keys=[UserFollow.following_id], back_populates="followers")
    comments = relationship("Comment", back_populates="user")
    activities = relationship("Activity", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    notification_preferences = Column(JSON, default=lambda: {
        "email": True,
        "web": True,
        "price_alerts": True,
        "portfolio_updates": True,
        "social_notifications": True
    })

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_public = Column(Boolean, default=False)
    
    owner = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio")
    shares = relationship("PortfolioShare", back_populates="portfolio")
    comments = relationship("Comment", back_populates="portfolio")

class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    symbol = Column(String)
    quantity = Column(Float)
    average_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="positions")
    trades = relationship("Trade", back_populates="position")

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    position_id = Column(Integer, ForeignKey("positions.id"))
    type = Column(String)  # "buy" or "sell"
    quantity = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    position = relationship("Position", back_populates="trades")

class Watchlist(Base):
    __tablename__ = "watchlists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    symbols = Column(JSON)  # List of symbols
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="watchlists")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String)
    condition = Column(String)  # "above" or "below"
    price = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="alerts")

class PortfolioShare(Base):
    __tablename__ = "portfolio_shares"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"))
    shared_with_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    permission = Column(String, default="read")  # read, write
    created_at = Column(DateTime, default=datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="shares")
    shared_with = relationship("User", back_populates="shared_portfolios")

class UserFollow(Base):
    __tablename__ = "user_follows"
    
    id = Column(Integer, primary_key=True, index=True)
    follower_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    following_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    follower = relationship("User", foreign_keys=[follower_id], back_populates="following")
    following = relationship("User", foreign_keys=[following_id], back_populates="followers")

class Comment(Base):
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    content = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="comments")
    user = relationship("User", back_populates="comments")

class ActivityType(str, Enum):
    PORTFOLIO_CREATE = "portfolio_create"
    PORTFOLIO_UPDATE = "portfolio_update"
    POSITION_ADD = "position_add"
    POSITION_REMOVE = "position_remove"
    TRADE_EXECUTE = "trade_execute"
    PORTFOLIO_SHARE = "portfolio_share"
    FOLLOW_USER = "follow_user"
    COMMENT_ADD = "comment_add"

class Activity(Base):
    __tablename__ = "activities"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    activity_type = Column(String)
    target_id = Column(Integer)  # ID of the target object (portfolio, position, etc.)
    target_type = Column(String)  # Type of the target object
    data = Column(JSON)  # Additional activity data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="activities")

class NotificationType(str, Enum):
    PORTFOLIO_SHARED = "portfolio_shared"
    NEW_FOLLOWER = "new_follower"
    NEW_COMMENT = "new_comment"
    PRICE_ALERT = "price_alert"
    PORTFOLIO_MENTION = "portfolio_mention"
    TRADE_ALERT = "trade_alert"

class Notification(Base):
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    notification_type = Column(String)
    data = Column(JSON)  # Notification details
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="user")
