"""
Real-time market data integration service for factor monitoring
"""
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketUpdate:
    """Data class for market updates"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    metadata: Dict

class RealTimeDataService:
    def __init__(self,
                 websocket_url: str,
                 api_key: Optional[str] = None,
                 buffer_size: int = 1000):
        """
        Initialize real-time data service
        
        Args:
            websocket_url: WebSocket endpoint URL
            api_key: Optional API key for authentication
            buffer_size: Size of the data buffer for each symbol
        """
        self.websocket_url = websocket_url
        self.api_key = api_key
        self.buffer_size = buffer_size
        
        # Data structures
        self.price_buffers: Dict[str, pd.DataFrame] = {}
        self.callbacks: List[Callable] = []
        self.running = False
        self.websocket = None
        
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            if self.api_key:
                headers = {'Authorization': f'Bearer {self.api_key}'}
                self.websocket = await websockets.connect(self.websocket_url, extra_headers=headers)
            else:
                self.websocket = await websockets.connect(self.websocket_url)
            
            logger.info("WebSocket connection established")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            return False
    
    def subscribe(self, symbols: List[str]):
        """Subscribe to market data for specified symbols"""
        if self.websocket and self.websocket.open:
            message = {
                'type': 'subscribe',
                'symbols': symbols
            }
            asyncio.create_task(self.websocket.send(json.dumps(message)))
            logger.info(f"Subscribed to {len(symbols)} symbols")
    
    def add_callback(self, callback: Callable[[MarketUpdate], None]):
        """Add callback for real-time updates"""
        self.callbacks.append(callback)
    
    async def process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Create market update
            update = MarketUpdate(
                timestamp=datetime.fromtimestamp(data['timestamp']),
                symbol=data['symbol'],
                price=float(data['price']),
                volume=float(data['volume']),
                metadata=data.get('metadata', {})
            )
            
            # Update price buffer
            new_data = pd.DataFrame([{
                'timestamp': update.timestamp,
                'price': update.price,
                'volume': update.volume
            }])
            
            if update.symbol not in self.price_buffers:
                self.price_buffers[update.symbol] = new_data
            else:
                self.price_buffers[update.symbol] = pd.concat([
                    self.price_buffers[update.symbol],
                    new_data
                ], ignore_index=True).tail(self.buffer_size)
            
            # Trigger callbacks
            for callback in self.callbacks:
                callback(update)
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    async def start(self):
        """Start real-time data service"""
        self.running = True
        
        while self.running:
            if not self.websocket or not self.websocket.open:
                success = await self.connect()
                if not success:
                    await asyncio.sleep(5)  # Retry delay
                    continue
            
            try:
                message = await self.websocket.recv()
                await self.process_message(message)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed. Reconnecting...")
                self.websocket = None
            except Exception as e:
                logger.error(f"Error in data service: {str(e)}")
                await asyncio.sleep(1)
    
    def stop(self):
        """Stop real-time data service"""
        self.running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
    
    def get_latest_prices(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get latest prices for specified symbols
        
        Args:
            symbols: List of symbols to get prices for. If None, get all symbols.
            
        Returns:
            DataFrame with latest prices
        """
        if symbols is None:
            symbols = list(self.price_buffers.keys())
        
        latest_prices = {}
        for symbol in symbols:
            if symbol in self.price_buffers:
                buffer = self.price_buffers[symbol]
                if not buffer.empty:
                    latest_prices[symbol] = buffer.iloc[-1]
        
        return pd.DataFrame(latest_prices).T
    
    def get_price_history(self,
                         symbol: str,
                         lookback: Optional[timedelta] = None) -> pd.DataFrame:
        """
        Get price history for a symbol
        
        Args:
            symbol: Symbol to get history for
            lookback: Optional lookback period (timedelta or pd.Timedelta)
            
        Returns:
            DataFrame with price history, sorted by timestamp
        """
        if symbol not in self.price_buffers:
            return pd.DataFrame()
        
        buffer = self.price_buffers[symbol]
        if buffer.empty:
            return buffer
            
        # Always sort by timestamp first
        buffer = buffer.sort_values('timestamp', ascending=True)
            
        if lookback:
            cutoff = pd.Timestamp.now().floor('min') - pd.Timedelta(lookback)
            buffer = buffer[buffer['timestamp'] >= cutoff].copy()
            
        return buffer.reset_index(drop=True)
    
    def update_price(self, symbol: str, price: float, timestamp: datetime, volume: int) -> None:
        """
        Update price data for a symbol
        
        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Timestamp of the price update
            volume: Trading volume
        """
        new_data = pd.DataFrame({
            'timestamp': [pd.Timestamp(timestamp)],
            'price': [float(price)],  # Ensure consistent dtype
            'volume': [int(volume)]   # Ensure consistent dtype
        })
        
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = new_data
        else:
            self.price_buffers[symbol] = pd.concat([
                self.price_buffers[symbol],
                new_data
            ], ignore_index=True).sort_values('timestamp', ascending=True)
        
        # Trim buffer if it exceeds max size
        if len(self.price_buffers[symbol]) > self.buffer_size:
            self.price_buffers[symbol] = self.price_buffers[symbol].iloc[-self.buffer_size:]
