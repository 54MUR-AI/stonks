import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from .base import MarketDataProvider, MarketDataConfig

class MockProvider(MarketDataProvider):
    """Mock market data provider for testing and development"""
    
    def __init__(self, config: MarketDataConfig):
        super().__init__(config)
        self.subscribed_symbols = set()
        self.connected = False
        self._stop_streaming = False
        self._stream_task = None
        
    async def connect(self) -> None:
        """Simulate connection establishment"""
        if self.connected:
            raise RuntimeError("Already connected")
        await asyncio.sleep(0.1)  # Simulate network delay
        self.connected = True
        
    async def disconnect(self) -> None:
        """Simulate disconnection"""
        if not self.connected:
            raise RuntimeError("Not connected")
        if self._stream_task:
            self._stop_streaming = True
            await self._stream_task
        self.connected = False
        
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to mock market data stream"""
        if not self.connected:
            raise RuntimeError("Not connected to market data provider")
        
        self.subscribed_symbols.update(symbols)
        if not self._stream_task:
            self._stop_streaming = False
            self._stream_task = asyncio.create_task(self._stream_market_data())
            
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from mock market data stream"""
        self.subscribed_symbols.difference_update(symbols)
        if not self.subscribed_symbols and self._stream_task:
            self._stop_streaming = True
            await self._stream_task
            self._stream_task = None
            
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1min"
    ) -> pd.DataFrame:
        """Generate mock historical market data"""
        if not self.connected:
            raise RuntimeError("Not connected to market data provider")
            
        if not end_date:
            end_date = datetime.now()
            
        self._validate_interval(interval)
        if start_date > end_date:
            return pd.DataFrame()  # Return empty DataFrame for invalid range
            
        # Generate date range based on interval
        if interval == "1min":
            freq = "1min"
        elif interval == "5min":
            freq = "5min"
        elif interval == "1h":
            freq = "1h"
        elif interval == "1d":
            freq = "1D"
            
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate mock price data with random walk
        n_points = len(dates)
        base_price = 100.0
        returns = np.random.normal(0.0001, 0.001, n_points)
        prices = base_price * (1 + np.cumsum(returns))
        volumes = np.random.randint(1000, 10000, n_points)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.002, n_points)),
            'low': prices * (1 - np.random.uniform(0, 0.002, n_points)),
            'close': prices,
            'volume': volumes
        })
        
    async def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Get mock latest quote"""
        if not self.connected:
            raise RuntimeError("Not connected to market data provider")
            
        price = 100.0 + random.uniform(-5, 5)
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'bid': price - 0.01,
            'ask': price + 0.01,
            'last': price,
            'volume': random.randint(100, 1000)
        }
        
    async def _stream_market_data(self) -> None:
        """Generate mock market data stream"""
        while not self._stop_streaming:
            for symbol in self.subscribed_symbols:
                price = 100.0 + random.uniform(-5, 5)
                data = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'price': price,
                    'volume': random.randint(100, 1000)
                }
                if hasattr(self, 'on_data'):
                    await self.on_data(data)
            await asyncio.sleep(0.1)  # Stream more frequently for testing

    def _validate_interval(self, interval: str) -> None:
        """Validate the requested interval"""
        valid_intervals = ["1min", "5min", "1h", "1d"]
        if interval not in valid_intervals:
            raise ValueError(f"Unsupported interval: {interval}")

    def _validate_dates(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Validate the date range"""
        if start_date > end_date:
            return pd.DataFrame()  # Return empty DataFrame for invalid range
