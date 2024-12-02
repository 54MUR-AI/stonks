from fastapi import FastAPI, WebSocket, HTTPException, Request, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import httpx
import json
import logging
from datetime import datetime, timedelta
import asyncio
from typing import Optional
from pydantic import BaseModel, validator
import re
from sqlalchemy.orm import Session
from database import engine, get_db
from models import Base, ActivityType, NotificationType
from schemas import UserCreate, User, Token, PortfolioCreate, Portfolio, PositionCreate, Position, TradeCreate, Trade, WatchlistCreate, Watchlist, AlertCreate, Alert
from auth import get_password_hash, verify_password, create_access_token, get_current_active_user, get_current_user
from analytics import calculate_portfolio_metrics, calculate_correlation_matrix
from email_service import send_alert_email, send_portfolio_summary, check_price_alerts
from portfolio_rebalancer import PortfolioRebalancer, optimize_portfolio_weights
from activity_service import ActivityService, NotificationService

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

# Create database tables
Base.metadata.create_all(bind=engine)

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

app = FastAPI(title="Stonks API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/register", response_model=User)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
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
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
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

@app.post("/portfolios/", response_model=Portfolio)
async def create_portfolio(
    portfolio: PortfolioCreate,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    db_portfolio = models.Portfolio(**portfolio.dict(), owner_id=current_user.id)
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

@app.get("/portfolios/", response_model=list[Portfolio])
async def get_portfolios(
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    return db.query(models.Portfolio).filter(models.Portfolio.owner_id == current_user.id).all()

@app.post("/positions/", response_model=Position)
async def create_position(
    position: PositionCreate,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == position.portfolio_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    db_position = models.Position(**position.dict())
    db.add(db_position)
    db.commit()
    db.refresh(db_position)
    return db_position

@app.post("/trades/", response_model=Trade)
async def create_trade(
    trade: TradeCreate,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify position ownership
    position = db.query(models.Position).join(models.Portfolio).filter(
        models.Position.id == trade.position_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")
    
    # Create trade
    db_trade = models.Trade(**trade.dict())
    db.add(db_trade)
    
    # Update position
    if trade.type == "buy":
        new_quantity = position.quantity + trade.quantity
        new_cost = (position.quantity * position.average_price) + (trade.quantity * trade.price)
        position.average_price = new_cost / new_quantity
        position.quantity = new_quantity
    else:  # sell
        if trade.quantity > position.quantity:
            raise HTTPException(status_code=400, detail="Insufficient position quantity")
        position.quantity -= trade.quantity
    
    db.commit()
    db.refresh(db_trade)
    return db_trade

@app.post("/watchlists/", response_model=Watchlist)
async def create_watchlist(
    watchlist: WatchlistCreate,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    db_watchlist = models.Watchlist(**watchlist.dict(), owner_id=current_user.id)
    db.add(db_watchlist)
    db.commit()
    db.refresh(db_watchlist)
    return db_watchlist

@app.get("/watchlists/", response_model=list[Watchlist])
async def get_watchlists(
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    return db.query(models.Watchlist).filter(models.Watchlist.owner_id == current_user.id).all()

@app.post("/alerts/", response_model=Alert)
async def create_alert(
    alert: AlertCreate,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    db_alert = models.Alert(**alert.dict(), owner_id=current_user.id)
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

@app.get("/alerts/", response_model=list[Alert])
async def get_alerts(
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    return db.query(models.Alert).filter(models.Alert.owner_id == current_user.id).all()

@app.get("/portfolios/{portfolio_id}/analytics")
async def get_portfolio_analytics(
    portfolio_id: int,
    start_date: str = None,
    end_date: str = None,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get positions
    positions = db.query(models.Position).filter(
        models.Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(positions, start_date, end_date)
    if not metrics:
        raise HTTPException(status_code=404, detail="Could not calculate metrics")
    
    return metrics

@app.get("/portfolios/{portfolio_id}/correlation")
async def get_portfolio_correlation(
    portfolio_id: int,
    start_date: str = None,
    end_date: str = None,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get positions
    positions = db.query(models.Position).filter(
        models.Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    # Calculate correlation matrix
    correlation = calculate_correlation_matrix(positions, start_date, end_date)
    if not correlation:
        raise HTTPException(status_code=404, detail="Could not calculate correlation")
    
    return correlation

@app.post("/portfolios/{portfolio_id}/email-summary")
async def email_portfolio_summary(
    portfolio_id: int,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get positions
    positions = db.query(models.Position).filter(
        models.Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(positions)
    if not metrics:
        raise HTTPException(status_code=404, detail="Could not calculate metrics")
    
    # Send email
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get positions
    positions = db.query(models.Position).filter(
        models.Position.portfolio_id == portfolio_id
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get positions
    positions = db.query(models.Position).filter(
        models.Position.portfolio_id == portfolio_id
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get positions
    positions = db.query(models.Position).filter(
        models.Position.portfolio_id == portfolio_id
    ).all()
    
    if not positions:
        raise HTTPException(status_code=404, detail="No positions found")
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(positions)
    if not metrics:
        raise HTTPException(status_code=404, detail="Could not calculate metrics")
    
    # Send email
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get positions
    positions = db.query(models.Position).filter(
        models.Position.portfolio_id == portfolio_id
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get positions
    positions = db.query(models.Position).filter(
        models.Position.portfolio_id == portfolio_id
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

@app.websocket("/ws/market")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = None,
    db: Session = Depends(get_db)
):
    if token:
        try:
            user = await get_current_user(token, db)
        except HTTPException:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    
    client_id = await manager.connect(websocket)
    
    try:
        while True:
            message = await websocket.receive_json()
            
            if "action" not in message or "symbol" not in message:
                await websocket.send_json({"error": "Invalid message format"})
                continue
            
            action = message["action"]
            symbol = message["symbol"].upper()
            
            try:
                # Validate symbol
                MarketDataRequest(symbol=symbol)
                
                if action == "subscribe":
                    await manager.subscribe_to_symbol(client_id, symbol)
                    
                    # Start background task for symbol if it's the first subscriber
                    if len(manager.symbol_subscribers.get(symbol, set())) == 1:
                        task = asyncio.create_task(fetch_real_time_data(symbol))
                        manager.background_tasks.add(task)
                        task.add_done_callback(manager.background_tasks.discard)
                    
                    await websocket.send_json({
                        "type": "subscription_success",
                        "symbol": symbol
                    })
                
                elif action == "unsubscribe":
                    await manager.unsubscribe_from_symbol(client_id, symbol)
                    await websocket.send_json({
                        "type": "unsubscription_success",
                        "symbol": symbol
                    })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Invalid action"
                    })
            
            except InvalidSymbolError as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
            
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
    finally:
        manager.disconnect(client_id)

# Social feature endpoints
@app.post("/portfolios/{portfolio_id}/share")
async def share_portfolio(
    portfolio_id: int,
    share_data: dict,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    shared_with = db.query(models.User).filter(
        models.User.id == share_data["user_id"]
    ).first()
    
    if not shared_with:
        raise HTTPException(status_code=404, detail="User not found")
    
    share = models.PortfolioShare(
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    shares = db.query(models.PortfolioShare).filter(
        models.PortfolioShare.shared_with_id == current_user.id
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")
    
    user_to_follow = db.query(models.User).filter(models.User.id == user_id).first()
    if not user_to_follow:
        raise HTTPException(status_code=404, detail="User not found")
    
    existing_follow = db.query(models.UserFollow).filter(
        models.UserFollow.follower_id == current_user.id,
        models.UserFollow.following_id == user_id
    ).first()
    
    if existing_follow:
        raise HTTPException(status_code=400, detail="Already following this user")
    
    follow = models.UserFollow(
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    follow = db.query(models.UserFollow).filter(
        models.UserFollow.follower_id == current_user.id,
        models.UserFollow.following_id == user_id
    ).first()
    
    if not follow:
        raise HTTPException(status_code=404, detail="Not following this user")
    
    db.delete(follow)
    db.commit()
    return {"message": "Successfully unfollowed user"}

@app.get("/users/followers")
async def get_followers(
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    followers = db.query(models.UserFollow).filter(
        models.UserFollow.following_id == current_user.id
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    following = db.query(models.UserFollow).filter(
        models.UserFollow.follower_id == current_user.id
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Check if portfolio is public or shared with the user
    if not portfolio.is_public and portfolio.owner_id != current_user.id:
        share = db.query(models.PortfolioShare).filter(
            models.PortfolioShare.portfolio_id == portfolio_id,
            models.PortfolioShare.shared_with_id == current_user.id
        ).first()
        if not share:
            raise HTTPException(status_code=403, detail="Not authorized to comment on this portfolio")
    
    comment = models.Comment(
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Check if portfolio is public or shared with the user
    if not portfolio.is_public and portfolio.owner_id != current_user.id:
        share = db.query(models.PortfolioShare).filter(
            models.PortfolioShare.portfolio_id == portfolio_id,
            models.PortfolioShare.shared_with_id == current_user.id
        ).first()
        if not share:
            raise HTTPException(status_code=403, detail="Not authorized to view comments")
    
    comments = db.query(models.Comment).filter(
        models.Comment.portfolio_id == portfolio_id
    ).order_by(models.Comment.created_at.desc()).all()
    
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

# Activity endpoints
@app.get("/activities/feed")
async def get_activity_feed(
    skip: int = 0,
    limit: int = 50,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get activity feed for current user"""
    activities = ActivityService.get_feed_activities(db, current_user.id, skip, limit)
    
    # Enrich activities with user and target object information
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
        
        # Add target object details based on type
        if activity.target_type == "portfolio":
            portfolio = db.query(models.Portfolio).get(activity.target_id)
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
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get activities for a specific user"""
    # Check if user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if user has permission to view activities
    if user_id != current_user.id:
        # Check if user is being followed
        is_following = db.query(models.UserFollow).filter(
            models.UserFollow.follower_id == current_user.id,
            models.UserFollow.following_id == user_id
        ).first()
        if not is_following:
            raise HTTPException(status_code=403, detail="Not authorized to view activities")
    
    activities = ActivityService.get_user_activities(db, user_id, skip, limit)
    return activities

# Notification endpoints
@app.get("/notifications")
async def get_notifications(
    unread_only: bool = False,
    skip: int = 0,
    limit: int = 50,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get notifications for current user"""
    notifications = NotificationService.get_user_notifications(
        db, current_user.id, unread_only, skip, limit
    )
    return notifications

@app.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark a notification as read"""
    success = NotificationService.mark_as_read(db, notification_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"message": "Notification marked as read"}

@app.post("/notifications/read-all")
async def mark_all_notifications_read(
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark all notifications as read"""
    count = NotificationService.mark_all_as_read(db, current_user.id)
    return {"message": f"Marked {count} notifications as read"}

@app.put("/users/notification-preferences")
async def update_notification_preferences(
    preferences: dict,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user's notification preferences"""
    user = db.query(models.User).filter(models.User.id == current_user.id).first()
    user.notification_preferences.update(preferences)
    db.commit()
    return {"message": "Notification preferences updated"}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(check_price_alerts(SessionLocal(), models.Alert))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
