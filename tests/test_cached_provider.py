"""Tests for cached market data provider."""

import asyncio
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock

from backend.services.market_data.cached_provider import CachedMarketDataProvider
from backend.services.market_data.base import MarketDataProvider, MarketDataConfig

class MockBaseProvider(MarketDataProvider):
    """Mock provider for testing cache wrapper."""
    
    def __init__(self, config: MarketDataConfig):
        """Initialize mock provider."""
        super().__init__(config)
        self.connect_called = AsyncMock()
        self.disconnect_called = AsyncMock()
        self.subscribe_called = AsyncMock()
        self.unsubscribe_called = AsyncMock()
        self.get_historical_called = AsyncMock()
        self.get_quote_called = AsyncMock()
        
    async def connect(self) -> None:
        await self.connect_called()
        
    async def disconnect(self) -> None:
        await self.disconnect_called()
        
    async def subscribe(self, symbols: list) -> None:
        await self.subscribe_called(symbols)
        
    async def unsubscribe(self, symbols: list) -> None:
        await self.unsubscribe_called(symbols)
        
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime = None,
        interval: str = "1min"
    ) -> pd.DataFrame:
        await self.get_historical_called(symbol, start_date, end_date, interval)
        dates = pd.date_range(start=start_date, end=end_date or datetime.now(), freq=interval)
        return pd.DataFrame({
            'timestamp': dates,
            'close': np.random.random(len(dates)) * 100
        })
        
    async def get_quote(self, symbol: str) -> float:
        await self.get_quote_called(symbol)
        return 100.0

@pytest.fixture
def config():
    """Create test config."""
    return MarketDataConfig(
        credentials=Mock(),
        base_url="http://test",
        websocket_url="ws://test"
    )

@pytest.fixture
async def cached_provider(config):
    """Create cached provider for testing."""
    base_provider = MockBaseProvider(config)
    provider = CachedMarketDataProvider(
        provider=base_provider,
        config=config,
        quote_ttl=timedelta(seconds=0.1),
        historical_ttl=timedelta(seconds=0.2)
    )
    await provider.connect()
    yield provider
    await provider.disconnect()

@pytest.mark.asyncio
async def test_quote_caching(cached_provider):
    """Test quote caching behavior."""
    # First call should hit provider
    quote1 = await cached_provider.get_quote("AAPL")
    assert cached_provider._provider.get_quote_called.call_count == 1
    
    # Second call should use cache
    quote2 = await cached_provider.get_quote("AAPL")
    assert cached_provider._provider.get_quote_called.call_count == 1
    assert quote1 == quote2
    
    # Wait for TTL to expire
    await asyncio.sleep(0.15)
    
    # Should hit provider again
    quote3 = await cached_provider.get_quote("AAPL")
    assert cached_provider._provider.get_quote_called.call_count == 2

@pytest.mark.asyncio
async def test_historical_caching(cached_provider):
    """Test historical data caching."""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2)
    
    # First call should hit provider
    df1 = await cached_provider.get_historical_data("AAPL", start, end)
    assert cached_provider._provider.get_historical_called.call_count == 1
    
    # Second call should use cache
    df2 = await cached_provider.get_historical_data("AAPL", start, end)
    assert cached_provider._provider.get_historical_called.call_count == 1
    assert df1.equals(df2)
    
    # Wait for TTL to expire
    await asyncio.sleep(0.25)
    
    # Should hit provider again
    df3 = await cached_provider.get_historical_data("AAPL", start, end)
    assert cached_provider._provider.get_historical_called.call_count == 2

@pytest.mark.asyncio
async def test_subscription_cache_invalidation(cached_provider):
    """Test cache invalidation on subscription."""
    # Get quote (cached)
    quote1 = await cached_provider.get_quote("AAPL")
    assert cached_provider._provider.get_quote_called.call_count == 1
    
    # Subscribe to symbol
    await cached_provider.subscribe(["AAPL"])
    
    # Quote should be refreshed
    quote2 = await cached_provider.get_quote("AAPL")
    assert cached_provider._provider.get_quote_called.call_count == 2
    
    # Subsequent quotes for subscribed symbol should always refresh
    quote3 = await cached_provider.get_quote("AAPL")
    assert cached_provider._provider.get_quote_called.call_count == 3

@pytest.mark.asyncio
async def test_cache_metrics(cached_provider):
    """Test cache metrics collection."""
    # Generate some cache activity
    await cached_provider.get_quote("AAPL")
    await cached_provider.get_quote("AAPL")  # Cache hit
    await cached_provider.get_quote("GOOGL")
    
    metrics = cached_provider.cache_metrics
    assert metrics['hits'] == 1
    assert metrics['misses'] == 2
    assert 'avg_latencies' in metrics

@pytest.mark.asyncio
async def test_provider_delegation(cached_provider):
    """Test proper delegation to base provider."""
    # Connect/disconnect handled by fixture
    assert cached_provider._provider.connect_called.called
    
    # Test subscribe
    await cached_provider.subscribe(["AAPL", "GOOGL"])
    cached_provider._provider.subscribe_called.assert_called_once_with(
        ["AAPL", "GOOGL"]
    )
    
    # Test unsubscribe
    await cached_provider.unsubscribe(["AAPL"])
    cached_provider._provider.unsubscribe_called.assert_called_once_with(
        ["AAPL"]
    )
