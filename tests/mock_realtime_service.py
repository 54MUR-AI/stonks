"""Mock RealTimeDataService for testing"""
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import random

from backend.services.realtime_data import RealTimeDataService, MarketUpdate

class MockRealTimeDataService(RealTimeDataService):
    """Mock implementation of real-time data service for testing"""
    
    def __init__(self):
        self.callbacks = []
        self.price_history = defaultdict(list)
        self.running = False
        self.update_task = None
        self.subscribed_symbols = set()
        
    def add_callback(self, callback: Callable) -> None:
        """Add callback for price updates"""
        self.callbacks.append(callback)
        
    def remove_callback(self, callback: Callable) -> None:
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    async def start(self) -> None:
        """Start the mock service"""
        if not self.running:
            self.running = True
            self.update_task = asyncio.create_task(self._simulate_updates())
            
    async def stop(self) -> None:
        """Stop the mock service"""
        self.running = False
        if self.update_task:
            try:
                self.update_task.cancel()
                await self.update_task
            except asyncio.CancelledError:
                pass
            self.update_task = None
            
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols"""
        self.subscribed_symbols.update(symbols)
        
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        self.subscribed_symbols.difference_update(symbols)
            
    async def _simulate_updates(self) -> None:
        """Simulate price updates"""
        base_prices = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0}
        
        while self.running:
            for symbol in self.subscribed_symbols:
                if symbol in base_prices:
                    # Generate random price movement
                    price_change = random.uniform(-1.0, 1.0)
                    new_price = base_prices[symbol] * (1 + price_change/100)
                    base_prices[symbol] = new_price
                    
                    # Create update
                    update = MarketUpdate(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        price=new_price,
                        volume=random.randint(100, 10000),
                        metadata={}
                    )
                    
                    # Store in history
                    self.price_history[symbol].append(update)
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        try:
                            await callback(update)
                        except Exception as e:
                            print(f"Error in callback: {e}")
                            
            await asyncio.sleep(0.1)  # Update every 100ms
            
    def get_price_history(
        self,
        symbol: str,
        lookback: timedelta = timedelta(minutes=5)
    ) -> pd.DataFrame:
        """Get price history for symbol"""
        now = datetime.now()
        cutoff = now - lookback
        
        # Filter history by lookback period
        history = [
            update for update in self.price_history[symbol]
            if update.timestamp >= cutoff
        ]
        
        if not history:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': update.timestamp,
                'price': update.price,
                'volume': update.volume
            }
            for update in history
        ])
        
        return df.set_index('timestamp')
