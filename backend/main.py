from fastapi import FastAPI, HTTPException, Depends, status, Request, WebSocket
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pydantic import BaseModel, constr, confloat, validator
from typing import List, Dict, Optional

from backend.database import SessionLocal, engine, Base
from backend.models import (
    Base, ActivityType, NotificationType, User, Portfolio, Position,
    Trade, Watchlist, Alert, PortfolioShare, UserFollow, Comment
)
from backend.schemas import (
    UserCreate, UserResponse, Token, PortfolioCreate, PortfolioResponse,
    PositionCreate, PositionResponse, TradeCreate, TradeResponse,
    WatchlistCreate, WatchlistResponse, AlertCreate, AlertResponse,
    CommentCreate, CommentResponse, MarketDataRequest, TargetWeights
)
from backend.auth_service import AuthService, get_current_active_user
from backend.activity_service import ActivityService, NotificationService
from backend.analytics import calculate_portfolio_metrics, calculate_correlation_matrix
from backend.email_service import send_alert_email, send_portfolio_summary, check_price_alerts
from backend.routers import (
    auth, users, portfolios, portfolio_metrics, websocket, rebalancing, trading, visualization, news
)
from backend.services.realtime_analytics import realtime_analytics_service

# Load environment variables
load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="STONKS Financial Platform",
    description="Advanced financial market analysis and portfolio management platform",
    version="1.0.0"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize services
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")))
    access_token = auth_service.create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=UserResponse)
async def register(user_create: UserCreate, db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    try:
        user = auth_service.create_user(
            email=user_create.email,
            password=user_create.password,
            username=user_create.username,
            full_name=user_create.full_name
        )
        return user
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create user")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stonks.log')
    ]
)
logger = logging.getLogger(__name__)

# Custom exceptions
class MarketDataError(Exception):
    pass

class AIAnalysisError(Exception):
    pass

class InvalidSymbolError(Exception):
    pass

# Input validation models
class MarketDataRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"

    @validator('symbol')
    def validate_symbol(cls, v):
        if not re.match(r'^[A-Za-z0-9.-]+$', v):
            raise InvalidSymbolError('Invalid symbol format')
        return v.upper()

    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']
        if v not in valid_timeframes:
            raise ValueError(f'Invalid timeframe. Must be one of: {", ".join(valid_timeframes)}')
        return v

class TargetWeights(BaseModel):
    weights: dict[str, float]
    tolerance: float = 0.05

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(users.router, prefix="/api", tags=["users"])
app.include_router(portfolios.router, prefix="/api", tags=["portfolios"])
app.include_router(portfolio_metrics.router, prefix="/api", tags=["portfolio-metrics"])
app.include_router(rebalancing.router, prefix="/api", tags=["portfolio-rebalancing"])
app.include_router(trading.router, prefix="/api", tags=["trading"])
app.include_router(visualization.router, prefix="/api", tags=["visualization"])
app.include_router(websocket.router, prefix="/api", tags=["websocket"])
app.include_router(news.router, tags=["news"])

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    response = await call_next(request)
    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds() * 1000
    logger.info(
        f"Path: {request.url.path} "
        f"Method: {request.method} "
        f"Status: {response.status_code} "
        f"Duration: {duration:.2f}ms"
    )
    return response

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}  # {client_id: {"websocket": WebSocket, "symbols": set()}}
        self.symbol_subscribers: dict = {}  # {symbol: set(client_ids)}
        self.background_tasks = set()

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        client_id = str(id(websocket))
        self.active_connections[client_id] = {"websocket": websocket, "symbols": set()}
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        return client_id

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            # Remove client from all symbol subscriptions
            for symbol in self.active_connections[client_id]["symbols"]:
                if symbol in self.symbol_subscribers:
                    self.symbol_subscribers[symbol].remove(client_id)
                    if not self.symbol_subscribers[symbol]:
                        del self.symbol_subscribers[symbol]
            
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def subscribe_to_symbol(self, client_id: str, symbol: str):
        if client_id not in self.active_connections:
            raise ValueError("Client not connected")
        
        # Add symbol to client's subscriptions
        self.active_connections[client_id]["symbols"].add(symbol)
        
        # Add client to symbol's subscribers
        if symbol not in self.symbol_subscribers:
            self.symbol_subscribers[symbol] = set()
        self.symbol_subscribers[symbol].add(client_id)
        
        logger.info(f"Client {client_id} subscribed to {symbol}")

    async def unsubscribe_from_symbol(self, client_id: str, symbol: str):
        if client_id in self.active_connections:
            self.active_connections[client_id]["symbols"].discard(symbol)
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(client_id)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]
            logger.info(f"Client {client_id} unsubscribed from {symbol}")

    async def broadcast_to_symbol(self, symbol: str, message: dict):
        if symbol in self.symbol_subscribers:
            for client_id in self.symbol_subscribers[symbol].copy():
                try:
                    websocket = self.active_connections[client_id]["websocket"]
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to client {client_id}: {str(e)}")
                    await self.disconnect(client_id)

manager = ConnectionManager()

async def fetch_real_time_data(symbol: str):
    """Background task to fetch real-time data for a symbol"""
    try:
        while symbol in manager.symbol_subscribers and len(manager.symbol_subscribers[symbol]) > 0:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1d', interval='1m').iloc[-1]
                
                message = {
                    "type": "market_update",
                    "symbol": symbol,
                    "data": {
                        "time": datetime.now().timestamp(),
                        "price": float(data["Close"]),
                        "volume": float(data["Volume"]),
                        "high": float(data["High"]),
                        "low": float(data["Low"]),
                        "open": float(data["Open"]),
                    }
                }
                
                await manager.broadcast_to_symbol(symbol, message)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                message = {
                    "type": "error",
                    "symbol": symbol,
                    "error": "Failed to fetch market data"
                }
                await manager.broadcast_to_symbol(symbol, message)
            
            await asyncio.sleep(60)  # Update every minute
            
    except Exception as e:
        logger.error(f"Background task error for {symbol}: {str(e)}")
    finally:
        logger.info(f"Stopping real-time updates for {symbol}")

@app.get("/")
async def root():
    return {"message": "Welcome to Stonks API"}

@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "1d"):
    try:
        # Validate input
        request = MarketDataRequest(symbol=symbol, timeframe=timeframe)
        
        logger.info(f"Fetching market data for {request.symbol} with timeframe {request.timeframe}")
        ticker = yf.Ticker(request.symbol)
        hist = ticker.history(period=request.timeframe)
        
        if hist.empty:
            logger.warning(f"No data found for symbol {request.symbol}")
            raise MarketDataError(f"No data found for symbol {request.symbol}")
        
        # Convert the data to a format suitable for charts
        candles = []
        for index, row in hist.iterrows():
            candles.append({
                "time": index.timestamp(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"])
            })
        
        logger.info(f"Successfully retrieved {len(candles)} candles for {request.symbol}")
        return {
            "symbol": request.symbol,
            "data": candles,
            "info": ticker.info
        }
    except InvalidSymbolError as e:
        logger.error(f"Invalid symbol error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except MarketDataError as e:
        logger.error(f"Market data error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_market_data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/ai/analysis/{symbol}")
async def get_ai_analysis(symbol: str):
    try:
        # Validate symbol
        request = MarketDataRequest(symbol=symbol)
        logger.info(f"Generating AI analysis for {request.symbol}")
        
        # Get market data
        ticker = yf.Ticker(request.symbol)
        hist = ticker.history(period="1mo")
        
        if hist.empty:
            logger.warning(f"No data found for symbol {request.symbol}")
            raise MarketDataError(f"No data found for symbol {request.symbol}")
        
        # Prepare prompt for Ollama
        prompt = f"""Analyze the following market data for {request.symbol}:
        Current Price: {hist['Close'][-1]:.2f}
        1 Month High: {hist['High'].max():.2f}
        1 Month Low: {hist['Low'].min():.2f}
        Volume: {hist['Volume'][-1]:.0f}
        
        Provide a brief market analysis and trading recommendation."""
        
        # Call Ollama API with timeout
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama2",
                        "prompt": prompt
                    }
                )
                
                if response.status_code == 200:
                    analysis = response.json()["response"]
                    logger.info(f"Successfully generated AI analysis for {request.symbol}")
                    return {
                        "symbol": request.symbol,
                        "analysis": analysis
                    }
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    raise AIAnalysisError("Failed to generate AI analysis")
                    
            except httpx.TimeoutException:
                logger.error("Ollama API timeout")
                raise AIAnalysisError("AI analysis service timeout")
                
    except InvalidSymbolError as e:
        logger.error(f"Invalid symbol error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except MarketDataError as e:
        logger.error(f"Market data error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except AIAnalysisError as e:
        logger.error(f"AI analysis error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_ai_analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/portfolios/", response_model=PortfolioResponse)
async def create_portfolio(
    portfolio: PortfolioCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    db_portfolio = Portfolio(**portfolio.dict(), owner_id=current_user.id)
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

@app.get("/portfolios/", response_model=List[PortfolioResponse])
async def get_portfolios(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return db.query(Portfolio).filter(Portfolio.owner_id == current_user.id).all()

@app.post("/positions/", response_model=PositionResponse)
async def create_position(
    position: PositionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == position.portfolio_id,
        Portfolio.owner_id == current_user.id
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    db_position = Position(**position.dict())
    db.add(db_position)
    db.commit()
    db.refresh(db_position)
    return db_position

@app.post("/trades/", response_model=TradeResponse)
async def create_trade(
    trade: TradeCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    position = db.query(Position).join(Portfolio).filter(
        Position.id == trade.position_id,
        Portfolio.owner_id == current_user.id
    ).first()
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")

    db_trade = Trade(**trade.dict())
    db.add(db_trade)

    if trade.type == "buy":
        new_quantity = position.quantity + trade.quantity
        new_cost = (position.quantity * position.average_price) + (trade.quantity * trade.price)
        position.average_price = new_cost / new_quantity
        position.quantity = new_quantity
    else:
        if trade.quantity > position.quantity:
            raise HTTPException(status_code=400, detail="Insufficient position quantity")
        position.quantity -= trade.quantity

    db.commit()
    db.refresh(db_trade)
    return db_trade

@app.post("/watchlists/", response_model=WatchlistResponse)
async def create_watchlist(
    watchlist: WatchlistCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    db_watchlist = Watchlist(**watchlist.dict(), owner_id=current_user.id)
    db.add(db_watchlist)
    db.commit()
    db.refresh(db_watchlist)
    return db_watchlist

@app.get("/watchlists/", response_model=List[WatchlistResponse])
async def get_watchlists(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return db.query(Watchlist).filter(Watchlist.owner_id == current_user.id).all()

@app.post("/alerts/", response_model=AlertResponse)
async def create_alert(
    alert: AlertCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    db_alert = Alert(**alert.dict(), owner_id=current_user.id)
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

@app.get("/alerts/", response_model=List[AlertResponse])
async def get_alerts(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return db.query(Alert).filter(Alert.owner_id == current_user.id).all()

@app.get("/portfolios/{portfolio_id}/analytics")
async def get_portfolio_analytics(
    portfolio_id: int,
    start_date: str = None,
    end_date: str = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    positions = db.query(Position).filter(
        Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    metrics = calculate_portfolio_metrics(positions, start_date, end_date)
    if not metrics:
        raise HTTPException(status_code=404, detail="Could not calculate metrics")
    
    return metrics

@app.get("/portfolios/{portfolio_id}/correlation")
async def get_portfolio_correlation(
    portfolio_id: int,
    start_date: str = None,
    end_date: str = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    positions = db.query(Position).filter(
        Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    correlation = calculate_correlation_matrix(positions, start_date, end_date)
    if not correlation:
        raise HTTPException(status_code=404, detail="Could not calculate correlation")
    
    return correlation

@app.post("/portfolios/{portfolio_id}/email-summary")
async def email_portfolio_summary(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    positions = db.query(Position).filter(
        Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    metrics = calculate_portfolio_metrics(positions)
    if not metrics:
        raise HTTPException(status_code=404, detail="Could not calculate metrics")
    
    try:
        await send_portfolio_summary(current_user.email, metrics)
        return {"message": "Portfolio summary email sent"}
    except Exception as e:
        logger.error(f"Error sending portfolio summary email: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send email")

@app.post("/portfolios/{portfolio_id}/rebalance")
async def rebalance_portfolio(
    portfolio_id: int,
    target_weights: TargetWeights,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    positions = db.query(Position).filter(
        Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    try:
        rebalancer = PortfolioRebalancer(
            positions=positions,
            target_weights=target_weights.weights,
            tolerance=target_weights.tolerance
        )
        summary = rebalancer.get_rebalancing_summary()
        return summary
    except Exception as e:
        logger.error(f"Error calculating rebalancing trades: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolios/{portfolio_id}/optimize")
async def optimize_portfolio(
    portfolio_id: int,
    risk_tolerance: float = 0.5,
    min_weight: float = 0.05,
    max_weight: float = 0.4,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    positions = db.query(Position).filter(
        Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    try:
        optimal_weights = optimize_portfolio_weights(
            positions=positions,
            risk_tolerance=risk_tolerance,
            min_weight=min_weight,
            max_weight=max_weight
        )
        return {"optimal_weights": optimal_weights}
    except Exception as e:
        logger.error(f"Error optimizing portfolio weights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolios/{portfolio_id}/email-summary")
async def email_portfolio_summary(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    positions = db.query(Position).filter(
        Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    metrics = calculate_portfolio_metrics(positions)
    if not metrics:
        raise HTTPException(status_code=404, detail="Could not calculate metrics")
    
    try:
        await send_portfolio_summary(current_user.email, metrics)
        return {"message": "Portfolio summary email sent"}
    except Exception as e:
        logger.error(f"Error sending portfolio summary email: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send email")

@app.post("/portfolios/{portfolio_id}/rebalance")
async def rebalance_portfolio(
    portfolio_id: int,
    target_weights: TargetWeights,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    positions = db.query(Position).filter(
        Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    try:
        rebalancer = PortfolioRebalancer(
            positions=positions,
            target_weights=target_weights.weights,
            tolerance=target_weights.tolerance
        )
        summary = rebalancer.get_rebalancing_summary()
        return summary
    except Exception as e:
        logger.error(f"Error calculating rebalancing trades: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolios/{portfolio_id}/optimize")
async def optimize_portfolio(
    portfolio_id: int,
    risk_tolerance: float = 0.5,
    min_weight: float = 0.05,
    max_weight: float = 0.4,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    positions = db.query(Position).filter(
        Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    try:
        optimal_weights = optimize_portfolio_weights(
            positions=positions,
            risk_tolerance=risk_tolerance,
            min_weight=min_weight,
            max_weight=max_weight
        )
        return {"optimal_weights": optimal_weights}
    except Exception as e:
        logger.error(f"Error optimizing portfolio weights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolios/{portfolio_id}/share")
async def share_portfolio(
    portfolio_id: int,
    share_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    shared_with = db.query(User).filter(
        User.id == share_data["user_id"]
    ).first()
    
    if not shared_with:
        raise HTTPException(status_code=404, detail="User not found")
    
    share = PortfolioShare(
        portfolio_id=portfolio.id,
        shared_with_id=shared_with.id,
        permission=share_data.get("permission", "read")
    )
    
    db.add(share)
    db.commit()
    db.refresh(share)
    return share

@app.get("/portfolios/shared")
async def get_shared_portfolios(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    shares = db.query(PortfolioShare).filter(
        PortfolioShare.shared_with_id == current_user.id
    ).all()
    
    portfolios = []
    for share in shares:
        portfolio = share.portfolio
        portfolio_dict = {
            "id": portfolio.id,
            "name": portfolio.name,
            "owner": portfolio.owner.email,
            "permission": share.permission
        }
        portfolios.append(portfolio_dict)
    
    return portfolios

@app.post("/users/{user_id}/follow")
async def follow_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")
    
    user_to_follow = db.query(User).filter(User.id == user_id).first()
    if not user_to_follow:
        raise HTTPException(status_code=404, detail="User not found")
    
    existing_follow = db.query(UserFollow).filter(
        UserFollow.follower_id == current_user.id,
        UserFollow.following_id == user_id
    ).first()
    
    if existing_follow:
        raise HTTPException(status_code=400, detail="Already following this user")
    
    follow = UserFollow(
        follower_id=current_user.id,
        following_id=user_id
    )
    
    db.add(follow)
    db.commit()
    db.refresh(follow)
    return follow

@app.delete("/users/{user_id}/unfollow")
async def unfollow_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    follow = db.query(UserFollow).filter(
        UserFollow.follower_id == current_user.id,
        UserFollow.following_id == user_id
    ).first()
    
    if not follow:
        raise HTTPException(status_code=404, detail="Not following this user")
    
    db.delete(follow)
    db.commit()
    return {"message": "Successfully unfollowed user"}

@app.get("/users/followers")
async def get_followers(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    followers = db.query(UserFollow).filter(
        UserFollow.following_id == current_user.id
    ).all()
    
    return [
        {
            "id": follow.follower.id,
            "email": follow.follower.email,
            "followed_at": follow.created_at
        }
        for follow in followers
    ]

@app.get("/users/following")
async def get_following(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    following = db.query(UserFollow).filter(
        UserFollow.follower_id == current_user.id
    ).all()
    
    return [
        {
            "id": follow.following.id,
            "email": follow.following.email,
            "followed_at": follow.created_at
        }
        for follow in following
    ]

@app.post("/portfolios/{portfolio_id}/comments")
async def create_comment(
    portfolio_id: int,
    comment_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    comment = Comment(
        portfolio_id=portfolio_id,
        user_id=current_user.id,
        content=comment_data["content"]
    )
    
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return comment

@app.get("/portfolios/{portfolio_id}/comments")
async def get_comments(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    comments = db.query(Comment).filter(
        Comment.portfolio_id == portfolio_id
    ).order_by(Comment.created_at.desc()).all()
    
    return [
        {
            "id": comment.id,
            "content": comment.content,
            "user_email": comment.user.email,
            "created_at": comment.created_at,
            "updated_at": comment.updated_at
        }
        for comment in comments
    ]

@app.get("/activities/feed")
async def get_activity_feed(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    activities = ActivityService.get_feed_activities(db, current_user.id, skip, limit)
    
    enriched_activities = []
    for activity in activities:
        activity_dict = {
            "id": activity.id,
            "user": {
                "id": activity.user.id,
                "email": activity.user.email
            },
            "activity_type": activity.activity_type,
            "target_id": activity.target_id,
            "target_type": activity.target_type,
            "data": activity.data,
            "created_at": activity.created_at
        }
        
        if activity.target_type == "portfolio":
            portfolio = db.query(Portfolio).get(activity.target_id)
            if portfolio:
                activity_dict["target"] = {
                    "id": portfolio.id,
                    "name": portfolio.name
                }
        
        enriched_activities.append(activity_dict)
    
    return enriched_activities

@app.get("/activities/user/{user_id}")
async def get_user_activities(
    user_id: int,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user_id != current_user.id:
        is_following = db.query(UserFollow).filter(
            UserFollow.follower_id == current_user.id,
            UserFollow.following_id == user_id
        ).first()
        if not is_following:
            raise HTTPException(status_code=403, detail="Not authorized to view activities")
    
    activities = ActivityService.get_user_activities(db, user_id, skip, limit)
    return activities

@app.get("/notifications")
async def get_notifications(
    unread_only: bool = False,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    notifications = NotificationService.get_user_notifications(
        db, current_user.id, unread_only, skip, limit
    )
    return notifications

@app.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    success = NotificationService.mark_as_read(db, notification_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"message": "Notification marked as read"}

@app.post("/notifications/read-all")
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    count = NotificationService.mark_all_as_read(db, current_user.id)
    return {"message": f"Marked {count} notifications as read"}

@app.put("/users/notification-preferences")
async def update_notification_preferences(
    preferences: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == current_user.id).first()
    user.notification_preferences.update(preferences)
    db.commit()
    return {"message": "Notification preferences updated"}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(check_price_alerts(SessionLocal(), Alert))

if __name__ == "__main__":
    import uvicorn
    import os
    from dotenv import load_dotenv

    load_dotenv()

    HOST = os.getenv("STONKS_HOST", "127.0.0.1")
    PORT = int(os.getenv("STONKS_PORT", "8000"))
    ENV = os.getenv("STONKS_ENV", "development")

    if ENV == "production" and HOST == "0.0.0.0":
        import logging
        logging.warning(
            "Warning: Server is configured to bind to all interfaces. "
            "Make sure this is intended for production use."
        )

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=(ENV == "development")
    )
