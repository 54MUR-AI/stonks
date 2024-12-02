import asyncio
import json
from typing import Dict, Set, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque

from ..analytics.market_analytics import (
    analyze_market_microstructure,
    analyze_liquidity_profile,
    get_market_depth
)

class RealtimeAnalyticsService:
    def __init__(self):
        self.subscribers: Dict[str, Set[str]] = {}  # symbol -> set of connection ids
        self.price_history: Dict[str, deque] = {}  # symbol -> deque of price data
        self.volume_history: Dict[str, deque] = {}  # symbol -> deque of volume data
        self.analytics_cache: Dict[str, Dict] = {}  # symbol -> cached analytics
        self.cache_expiry: Dict[str, datetime] = {}  # symbol -> cache expiry time
        self.CACHE_DURATION = timedelta(minutes=5)
        self.HISTORY_LENGTH = 1000  # Number of data points to keep
        
    async def subscribe(self, symbol: str, connection_id: str):
        """Subscribe a connection to real-time analytics for a symbol"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = set()
            self.price_history[symbol] = deque(maxlen=self.HISTORY_LENGTH)
            self.volume_history[symbol] = deque(maxlen=self.HISTORY_LENGTH)
            
        self.subscribers[symbol].add(connection_id)
        
    async def unsubscribe(self, symbol: str, connection_id: str):
        """Unsubscribe a connection from a symbol"""
        if symbol in self.subscribers:
            self.subscribers[symbol].discard(connection_id)
            if not self.subscribers[symbol]:
                del self.subscribers[symbol]
                del self.price_history[symbol]
                del self.volume_history[symbol]
                
    def update_market_data(self, symbol: str, price: float, volume: int):
        """Update market data for a symbol"""
        if symbol in self.price_history:
            self.price_history[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'price': price
            })
            self.volume_history[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'volume': volume
            })
            
    async def get_analytics_update(self, symbol: str) -> Dict:
        """Get real-time analytics update for a symbol"""
        now = datetime.now()
        cache_valid = (
            symbol in self.cache_expiry and 
            now < self.cache_expiry[symbol]
        )
        
        if not cache_valid:
            # Get fresh market data
            depth = get_market_depth(symbol)
            
            # Calculate real-time metrics
            price_data = pd.DataFrame(list(self.price_history[symbol]))
            volume_data = pd.DataFrame(list(self.volume_history[symbol]))
            
            if not price_data.empty and not volume_data.empty:
                # Calculate real-time volatility
                returns = price_data['price'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252 * 390)  # Annualized
                
                # Calculate volume metrics
                volume_mean = volume_data['volume'].mean()
                volume_std = volume_data['volume'].std()
                
                # Calculate price momentum
                momentum_1min = returns.tail(60).mean()
                momentum_5min = returns.tail(300).mean()
                
                # Calculate real-time spread
                spread = (depth['ask_price'] - depth['bid_price']) / depth['last_price']
                
                # Update cache
                self.analytics_cache[symbol] = {
                    'real_time_metrics': {
                        'price': depth['last_price'],
                        'bid': depth['bid_price'],
                        'ask': depth['ask_price'],
                        'spread': spread,
                        'volatility': volatility,
                        'volume_mean': volume_mean,
                        'volume_std': volume_std,
                        'momentum_1min': momentum_1min,
                        'momentum_5min': momentum_5min
                    },
                    'market_depth': depth,
                    'timestamp': now.isoformat()
                }
                self.cache_expiry[symbol] = now + self.CACHE_DURATION
                
        return self.analytics_cache.get(symbol, {})
    
    async def get_visualization_data(self, symbol: str, metric: str = 'price') -> Dict:
        """Get data for visualization"""
        if symbol not in self.price_history:
            return {'data': []}
            
        if metric == 'price':
            data = list(self.price_history[symbol])
        elif metric == 'volume':
            data = list(self.volume_history[symbol])
        else:
            return {'data': []}
            
        # Calculate moving averages
        df = pd.DataFrame(data)
        if not df.empty:
            if metric == 'price':
                df['MA5'] = df['price'].rolling(window=5).mean()
                df['MA20'] = df['price'].rolling(window=20).mean()
                df['Upper_BB'], df['Lower_BB'] = self.calculate_bollinger_bands(df['price'])
            elif metric == 'volume':
                df['MA5'] = df['volume'].rolling(window=5).mean()
                
        return {
            'data': data,
            'moving_averages': {
                'MA5': df['MA5'].dropna().tolist(),
                'MA20': df['MA20'].dropna().tolist() if metric == 'price' else None,
            },
            'bollinger_bands': {
                'upper': df['Upper_BB'].dropna().tolist(),
                'lower': df['Lower_BB'].dropna().tolist()
            } if metric == 'price' else None
        }
    
    @staticmethod
    def calculate_bollinger_bands(price_data: pd.Series, window: int = 20, num_std: float = 2):
        """Calculate Bollinger Bands"""
        ma = price_data.rolling(window=window).mean()
        std = price_data.rolling(window=window).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        return upper_band, lower_band
    
    async def broadcast_analytics(self):
        """Broadcast analytics updates to subscribers"""
        while True:
            for symbol in list(self.subscribers.keys()):
                if not self.subscribers[symbol]:
                    continue
                    
                analytics = await self.get_analytics_update(symbol)
                message = {
                    'type': 'analytics_update',
                    'symbol': symbol,
                    'data': analytics
                }
                
                # Broadcast to all subscribers
                for connection_id in self.subscribers[symbol]:
                    yield (connection_id, json.dumps(message))
                    
            await asyncio.sleep(1)  # Update frequency

realtime_analytics_service = RealtimeAnalyticsService()
