from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set, Optional
import asyncio
import json
import yfinance as yf
from datetime import datetime
from ..database import get_db
from ..models import Portfolio, Position
from sqlalchemy.orm import Session

class MarketDataManager:
    def __init__(self):
        self.active_connections: Dict[int, Set[WebSocket]] = {}  # portfolio_id -> Set[WebSocket]
        self.symbol_subscriptions: Dict[str, Set[int]] = {}  # symbol -> Set[portfolio_id]
        self.last_prices: Dict[str, float] = {}
        self.update_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket, portfolio_id: int):
        await websocket.accept()
        if portfolio_id not in self.active_connections:
            self.active_connections[portfolio_id] = set()
        self.active_connections[portfolio_id].add(websocket)

    def disconnect(self, websocket: WebSocket, portfolio_id: int):
        self.active_connections[portfolio_id].remove(websocket)
        if not self.active_connections[portfolio_id]:
            del self.active_connections[portfolio_id]
            # Clean up symbol subscriptions
            for symbol, portfolios in self.symbol_subscriptions.items():
                portfolios.discard(portfolio_id)
                if not portfolios:
                    del self.symbol_subscriptions[symbol]

    async def subscribe_portfolio(self, portfolio_id: int, db: Session):
        """Subscribe to all symbols in a portfolio"""
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        for position in positions:
            if position.symbol not in self.symbol_subscriptions:
                self.symbol_subscriptions[position.symbol] = set()
            self.symbol_subscriptions[position.symbol].add(portfolio_id)

        # Start the update task if it's not running
        if not self.update_task or self.update_task.done():
            self.update_task = asyncio.create_task(self._update_prices())

    async def broadcast_to_portfolio(self, portfolio_id: int, message: dict):
        """Send update to all connections subscribed to a portfolio"""
        if portfolio_id in self.active_connections:
            dead_connections = set()
            for connection in self.active_connections[portfolio_id]:
                try:
                    await connection.send_json(message)
                except RuntimeError:
                    dead_connections.add(connection)
            
            # Clean up dead connections
            for dead in dead_connections:
                self.active_connections[portfolio_id].remove(dead)

    async def _update_prices(self):
        """Continuously update prices for subscribed symbols"""
        while self.symbol_subscriptions:
            try:
                for symbol in self.symbol_subscriptions.keys():
                    try:
                        ticker = yf.Ticker(symbol)
                        current_price = ticker.history(period='1d')['Close'].iloc[-1]
                        
                        # Check if price has changed
                        if symbol not in self.last_prices or self.last_prices[symbol] != current_price:
                            self.last_prices[symbol] = current_price
                            
                            # Broadcast to all subscribed portfolios
                            update = {
                                'type': 'price_update',
                                'symbol': symbol,
                                'price': current_price,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            for portfolio_id in self.symbol_subscriptions[symbol]:
                                await self.broadcast_to_portfolio(portfolio_id, update)
                    
                    except Exception as e:
                        print(f"Error updating price for {symbol}: {e}")
                
                await asyncio.sleep(5)  # Update every 5 seconds
            
            except Exception as e:
                print(f"Error in price update loop: {e}")
                await asyncio.sleep(5)

market_data_manager = MarketDataManager()

async def get_market_data_manager():
    return market_data_manager
