"""Cached market data provider implementation."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
import logging
import pandas as pd

from .base import MarketDataProvider, MarketDataConfig
from .cache import MarketDataCache

logger = logging.getLogger(__name__)

class CachedMarketDataProvider(MarketDataProvider):
    """Market data provider with caching support."""
    
    def __init__(
        self,
        provider: MarketDataProvider,
        config: MarketDataConfig,
        cache_size_bytes: int = 100 * 1024 * 1024,  # 100MB
        quote_ttl: timedelta = timedelta(seconds=10),
        historical_ttl: timedelta = timedelta(hours=1)
    ):
        """Initialize provider with caching.
        
        Args:
            provider: Underlying provider to cache
            config: Provider configuration
            cache_size_bytes: Maximum cache size
            quote_ttl: Time-to-live for quote data
            historical_ttl: Time-to-live for historical data
        """
        super().__init__(config)
        self._provider = provider
        self._cache = MarketDataCache(
            max_size_bytes=cache_size_bytes,
            default_ttl=quote_ttl
        )
        self._quote_ttl = quote_ttl
        self._historical_ttl = historical_ttl
        self._subscribed_symbols: Set[str] = set()
        
    @property
    def cache_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        return self._cache.metrics
        
    async def connect(self) -> None:
        """Connect to the provider and start cache."""
        await self._cache.start()
        await self._provider.connect()
        
    async def disconnect(self) -> None:
        """Disconnect from provider and stop cache."""
        await self._cache.stop()
        await self._provider.disconnect()
        
    def _make_quote_key(self, symbol: str) -> str:
        """Make cache key for quote data."""
        return f"quote:{symbol}"
        
    def _make_historical_key(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime],
        interval: str
    ) -> str:
        """Make cache key for historical data."""
        end_str = end_date.isoformat() if end_date else "now"
        return f"historical:{symbol}:{start_date.isoformat()}:{end_str}:{interval}"
        
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data stream."""
        # Subscribe through provider
        await self._provider.subscribe(symbols)
        self._subscribed_symbols.update(symbols)
        
        # Invalidate quote cache for subscribed symbols
        for symbol in symbols:
            await self._cache.invalidate(self._make_quote_key(symbol))
            
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data stream."""
        await self._provider.unsubscribe(symbols)
        self._subscribed_symbols.difference_update(symbols)
        
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1min"
    ) -> pd.DataFrame:
        """Get historical market data with caching."""
        cache_key = self._make_historical_key(
            symbol, start_date, end_date, interval
        )
        
        # Try cache first
        cached_data = await self._cache.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data)
            
        # Get from provider
        df = await self._provider.get_historical_data(
            symbol, start_date, end_date, interval
        )
        
        # Cache the result
        await self._cache.set(
            cache_key,
            df.to_dict('records'),
            ttl=self._historical_ttl
        )
        
        return df
        
    async def get_quote(self, symbol: str) -> float:
        """Get current quote with caching."""
        cache_key = self._make_quote_key(symbol)
        
        # For subscribed symbols, always get fresh data
        if symbol in self._subscribed_symbols:
            price = await self._provider.get_quote(symbol)
            await self._cache.set(cache_key, price, ttl=self._quote_ttl)
            return price
            
        # Try cache first
        cached_price = await self._cache.get(cache_key)
        if cached_price is not None:
            return cached_price
            
        # Get from provider
        price = await self._provider.get_quote(symbol)
        await self._cache.set(cache_key, price, ttl=self._quote_ttl)
        return price
