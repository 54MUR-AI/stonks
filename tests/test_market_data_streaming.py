"""Tests for market data streaming functionality"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock
import pandas as pd

from backend.services.market_data.mock_provider import MockProvider
from backend.services.market_data.adapter import MarketDataAdapter
from backend.services.market_data.base import MarketDataConfig, MarketDataCredentials
from .mocks import MockRealTimeDataService

@pytest.fixture
def mock_config():
    """Create mock market data config"""
    return MarketDataConfig(
        credentials=MarketDataCredentials(api_key="test_key"),
        base_url="https://test.example.com",
        websocket_url="wss://test.example.com/ws",
        request_timeout=30,
        max_retries=3,
        retry_delay=1
    )

class StreamingTestProvider(MockProvider):
    """Provider with enhanced streaming capabilities for testing"""
    
    def __init__(self, config: MarketDataConfig):
        super().__init__(config)
        self.stream_delay = 0.1  # Stream interval in seconds
        self.price_increment = 0.01  # Price change per update
        self.base_prices = {}  # Base price for each symbol
        
    async def _stream_market_data(self) -> None:
        """Generate predictable price updates for testing"""
        while not self._stop_streaming:
            for symbol in self.subscribed_symbols:
                if symbol not in self.base_prices:
                    self.base_prices[symbol] = 100.0
                    
                self.base_prices[symbol] += self.price_increment
                data = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'price': self.base_prices[symbol],
                    'volume': 1000
                }
                if hasattr(self, 'on_data'):
                    await self.on_data(data)
            await asyncio.sleep(self.stream_delay)

@pytest.fixture
def streaming_provider(mock_config):
    return StreamingTestProvider(mock_config)

@pytest.fixture
def realtime_service():
    """Create mock realtime service"""
    return MockRealTimeDataService()

@pytest.fixture
async def market_data_adapter(mock_config, realtime_service):
    """Create market data adapter with mock provider"""
    provider = MockProvider(mock_config)
    adapter = MarketDataAdapter(
        provider_class=lambda config: provider,
        config=mock_config,
        realtime_service=realtime_service
    )
    await adapter.start()
    yield adapter
    await adapter.stop()

@pytest.mark.asyncio
async def test_realtime_price_updates(market_data_adapter):
    """Test that real-time price updates are received"""
    adapter = await anext(market_data_adapter)
    updates_received = []
    
    # Create callback to collect updates
    async def on_update(update):
        updates_received.append(update)
        
    # Subscribe to updates
    adapter.realtime_service.add_callback(on_update)
    symbols = ["AAPL", "GOOGL"]
    await adapter.subscribe(symbols)
    
    # Wait for some updates
    await asyncio.sleep(0.5)  # Should receive ~5 updates per symbol
    
    # Verify updates were received
    assert len(updates_received) > 0
    assert all(u.symbol in symbols for u in updates_received)
    
@pytest.mark.asyncio
async def test_price_history_accumulation(market_data_adapter):
    """Test that price history is accumulated correctly"""
    adapter = await anext(market_data_adapter)
    symbol = "AAPL"
    
    # Subscribe and wait for updates
    await adapter.subscribe([symbol])
    await asyncio.sleep(0.5)  # Collect some price history
    
    # Get price history
    history = adapter.realtime_service.get_price_history(
        symbol,
        lookback=timedelta(seconds=1)
    )
    
    # Verify history
    assert isinstance(history, pd.DataFrame)
    assert len(history) > 0
    assert 'price' in history.columns
    
@pytest.mark.asyncio
async def test_multiple_symbol_streaming(market_data_adapter):
    """Test streaming multiple symbols simultaneously"""
    adapter = await anext(market_data_adapter)
    symbols = ["AAPL", "GOOGL", "MSFT"]
    updates_by_symbol = {symbol: [] for symbol in symbols}
    
    # Create callback to collect updates by symbol
    async def on_update(update):
        if update.symbol in updates_by_symbol:
            updates_by_symbol[update.symbol].append(update)
            
    # Subscribe and collect updates
    adapter.realtime_service.add_callback(on_update)
    await adapter.subscribe(symbols)
    await asyncio.sleep(0.5)
    
    # Verify updates for each symbol
    for symbol, updates in updates_by_symbol.items():
        assert len(updates) > 0
        
@pytest.mark.asyncio
async def test_subscription_management(market_data_adapter):
    """Test subscribing and unsubscribing from streams"""
    adapter = await anext(market_data_adapter)
    updates_received = []
    
    async def on_update(update):
        updates_received.append(update)
        
    # Subscribe to initial symbols
    adapter.realtime_service.add_callback(on_update)
    initial_symbols = ["AAPL", "GOOGL"]
    await adapter.subscribe(initial_symbols)
    await asyncio.sleep(0.2)
    
    # Record update count
    initial_count = len(updates_received)
    assert initial_count > 0
    
    # Unsubscribe from one symbol
    await adapter.unsubscribe(["AAPL"])
    updates_received.clear()
    await asyncio.sleep(0.2)
    
    # Verify only receiving updates for subscribed symbol
    assert len(updates_received) > 0
    assert all(u.symbol == "GOOGL" for u in updates_received)
    
@pytest.mark.asyncio
async def test_stream_recovery(market_data_adapter):
    """Test recovery of data stream after provider reconnection"""
    adapter = await anext(market_data_adapter)
    updates_received = []
    
    async def on_update(update):
        updates_received.append(update)
        
    # Start streaming
    adapter.realtime_service.add_callback(on_update)
    await adapter.subscribe(["AAPL"])
    await asyncio.sleep(0.2)
    
    # Force disconnect and reconnect
    await adapter.provider.disconnect()
    await adapter.provider.connect()
    
    # Verify streaming resumes
    updates_received.clear()
    await asyncio.sleep(0.2)
    assert len(updates_received) > 0
