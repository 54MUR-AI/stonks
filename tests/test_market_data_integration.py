import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any

from backend.services.market_data.mock_provider import MockProvider
from backend.services.market_data.adapter import MarketDataAdapter
from backend.services.market_data.base import MarketDataConfig, MarketDataCredentials
from backend.services.realtime_data import RealTimeDataService

@pytest.fixture
def mock_config():
    return MarketDataConfig(
        credentials=MarketDataCredentials(api_key="test_key"),
        base_url="https://test.example.com",
        websocket_url="wss://test.example.com/ws"
    )

@pytest.fixture
def realtime_service(mock_config):
    service = RealTimeDataService(
        websocket_url=mock_config.websocket_url,
        api_key=mock_config.credentials.api_key
    )
    return service

@pytest.fixture
async def mock_provider(mock_config):
    provider = MockProvider(mock_config)
    await provider.connect()
    return provider

@pytest.fixture
async def market_data_adapter(mock_config, realtime_service):
    adapter = MarketDataAdapter(
        provider_class=MockProvider,
        config=mock_config,
        realtime_service=realtime_service
    )
    await adapter.start()
    return adapter

@pytest.mark.asyncio
async def test_adapter_provider_integration(market_data_adapter):
    """Test integration between adapter and provider"""
    adapter = await market_data_adapter
    
    # Subscribe through adapter
    symbols = ["AAPL", "GOOGL"]
    await adapter.subscribe(symbols)
    
    # Get quotes
    quotes = await adapter.get_quotes(symbols)
    assert len(quotes) == len(symbols)
    
    # Unsubscribe
    await adapter.unsubscribe(symbols)
    await adapter.stop()

@pytest.mark.asyncio
async def test_multi_provider_data_aggregation(market_data_adapter, mock_config, realtime_service):
    """Test data aggregation from multiple providers"""
    adapter = await market_data_adapter
    
    # Create a second adapter with different provider
    second_adapter = MarketDataAdapter(
        provider_class=MockProvider,
        config=mock_config,
        realtime_service=realtime_service
    )
    await second_adapter.start()
    
    symbols = ["AAPL", "GOOGL"]
    await adapter.subscribe(symbols)
    await second_adapter.subscribe(symbols)
    
    # Get quotes from both adapters
    quotes = await asyncio.gather(
        adapter.get_quotes(symbols),
        second_adapter.get_quotes(symbols)
    )
    
    assert len(quotes) == 2
    assert all(len(q) == len(symbols) for q in quotes)
    
    # Cleanup
    await adapter.unsubscribe(symbols)
    await second_adapter.unsubscribe(symbols)
    await adapter.stop()
    await second_adapter.stop()

@pytest.mark.asyncio
async def test_provider_failover(market_data_adapter, mock_config, realtime_service):
    """Test adapter failover behavior when a provider fails"""
    adapter = await market_data_adapter
    
    # Create backup adapter
    backup_adapter = MarketDataAdapter(
        provider_class=MockProvider,
        config=mock_config,
        realtime_service=realtime_service
    )
    await backup_adapter.start()
    
    symbols = ["AAPL"]
    await adapter.subscribe(symbols)
    await backup_adapter.subscribe(symbols)
    
    # Simulate primary adapter failure
    await adapter.stop()
    
    # Verify backup still works
    quotes = await backup_adapter.get_quotes(symbols)
    assert len(quotes) == len(symbols)
    
    # Cleanup
    await backup_adapter.stop()

@pytest.mark.asyncio
async def test_concurrent_data_requests(market_data_adapter):
    """Test handling of concurrent data requests"""
    adapter = await market_data_adapter
    
    # Create multiple concurrent historical data requests
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    async def get_data(symbol: str) -> pd.DataFrame:
        return await adapter.get_historical_data(symbol, start_date, end_date)
    
    # Execute requests concurrently
    tasks = [get_data(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == len(symbols)
    assert all(isinstance(df, pd.DataFrame) for df in results)
    
    # Cleanup
    await adapter.stop()
