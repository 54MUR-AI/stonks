"""Mock classes for testing"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import random

from backend.services.realtime_data import RealTimeDataService, MarketUpdate

class MockRealTimeDataService(RealTimeDataService):
    """Mock implementation of RealTimeDataService"""

    def __init__(self, buffer_size: int = 1000):
        self._callbacks = []
        self._running = False
        self._price_history = {}
        self.buffer_size = buffer_size
        self._update_task = None
        self._subscribed_symbols = set()

    def add_callback(self, callback: Callable[[MarketUpdate], None]) -> None:
        """Add callback for market updates"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[MarketUpdate], None]) -> None:
        """Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def start(self) -> None:
        """Start the service"""
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())

    async def stop(self) -> None:
        """Stop the service"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data for symbols"""
        for symbol in symbols:
            if symbol not in self._price_history:
                self._price_history[symbol] = []
            self._subscribed_symbols.add(symbol)

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for symbols"""
        for symbol in symbols:
            self._subscribed_symbols.discard(symbol)

    async def _simulate_market_update(self, symbol: str, price: float, volume: int) -> None:
        """Simulate a market update for testing"""
        update = MarketUpdate(
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Add to price history
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(update)
        
        # Trim history if needed
        if len(self._price_history[symbol]) > self.buffer_size:
            self._price_history[symbol] = self._price_history[symbol][-self.buffer_size:]
            
        # Notify callbacks
        for callback in self._callbacks:
            try:
                await callback(update)
            except Exception as e:
                print(f"Error in callback: {e}")

    async def _update_loop(self) -> None:
        """Generate periodic market updates"""
        try:
            while self._running:
                # Generate updates for all subscribed symbols
                for symbol in list(self._subscribed_symbols):
                    price = 100.0 + random.uniform(-5, 5)
                    volume = random.randint(100, 1000)
                    await self._simulate_market_update(symbol, price, volume)
                await asyncio.sleep(0.1)  # Update every 100ms
        except asyncio.CancelledError:
            pass

    def get_price_history(self, symbol: str, lookback: timedelta) -> pd.DataFrame:
        """Get price history for a symbol"""
        if symbol not in self._price_history:
            return pd.DataFrame()
            
        history = self._price_history[symbol]
        if not history:
            return pd.DataFrame()
            
        cutoff = datetime.now() - lookback
        filtered = [
            update for update in history 
            if update.timestamp >= cutoff
        ]
        
        if not filtered:
            return pd.DataFrame()
            
        return pd.DataFrame([
            {
                'timestamp': update.timestamp,
                'price': update.price,
                'volume': update.volume
            }
            for update in filtered
        ])
