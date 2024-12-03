"""Tests for market data error handling scenarios"""
import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock
from typing import List, Set

from backend.services.market_data.mock_provider import MockProvider
from backend.services.market_data.adapter import (
    MarketDataAdapter,
    MarketDataError,
    ConnectionError,
    SubscriptionError,
    QuoteError,
    HistoricalDataError,
    ErrorContext
)
from backend.services.market_data.base import MarketDataConfig, MarketDataCredentials
from .mocks import MockRealTimeDataService

@pytest.fixture
def mock_config():
    return MarketDataConfig(
        credentials=MarketDataCredentials(api_key="test_key"),
        base_url="https://test.example.com",
        websocket_url="wss://test.example.com/ws"
    )

class ErrorInjectingProvider(MockProvider):
    """Provider that simulates various error conditions"""
    
    def __init__(self, config: MarketDataConfig):
        super().__init__(config)
        self.call_count = 0
        self.fail_after = 3
        self.inject_error = False
        self._connected = True
        self._subscribed = set()
        
    async def connect(self) -> None:
        """Connect to provider"""
        if self.inject_error:
            raise ConnectionError("Simulated connection error")
        await super().connect()
        self._connected = True
        
    async def disconnect(self) -> None:
        """Disconnect from provider"""
        await super().disconnect()
        self._connected = False
        
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols"""
        if not self._connected:
            raise ConnectionError("Not connected to market data provider")
        if self.inject_error:
            raise ConnectionError("Simulated connection error")
        await super().subscribe(symbols)
        self._subscribed.update(symbols)
        
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        if not self._connected:
            raise ConnectionError("Not connected to market data provider")
        if self.inject_error:
            raise ConnectionError("Simulated connection error")
        await super().unsubscribe(symbols)
        self._subscribed.difference_update(symbols)
        
    async def get_quote(self, symbol: str) -> float:
        """Simulate errors after certain number of calls"""
        if not self._connected:
            raise ConnectionError("Not connected to market data provider")
        self.call_count += 1
        if self.call_count > self.fail_after or self.inject_error:
            raise ConnectionError("Simulated connection error")
        return await super().get_quote(symbol)
        
    @property
    def is_connected(self) -> bool:
        """Check if provider is connected"""
        return self._connected

    @property
    def subscribed_symbols(self) -> Set[str]:
        """Get currently subscribed symbols"""
        return self._subscribed.copy()
        
@pytest.fixture
def error_provider(mock_config):
    return ErrorInjectingProvider(mock_config)

@pytest.fixture
def realtime_service():
    """Create mock realtime service"""
    return MockRealTimeDataService()

@pytest_asyncio.fixture
async def market_data_adapter(mock_config):
    """Create market data adapter for testing"""
    service = MockRealTimeDataService()
    adapter = MarketDataAdapter(ErrorInjectingProvider, mock_config, service)
    await adapter.start()
    yield adapter
    await adapter.stop()

@pytest.mark.asyncio
async def test_provider_connection_error(market_data_adapter):
    """Test handling of provider connection errors"""
    adapter = await market_data_adapter
    # Simulate connection error
    await adapter.provider.disconnect()
    with pytest.raises(QuoteError):
        await adapter.get_quotes(["AAPL"])

@pytest.mark.asyncio
async def test_provider_data_error(market_data_adapter):
    """Test handling of provider data errors"""
    adapter = await market_data_adapter
    # Inject data error
    adapter.provider.inject_error = True
    with pytest.raises(QuoteError):
        await adapter.get_quotes(["AAPL"])

@pytest.mark.asyncio
async def test_concurrent_error_handling(market_data_adapter):
    """Test handling of errors during concurrent operations"""
    adapter = await market_data_adapter
    # Create multiple concurrent requests
    symbols = ["AAPL", "GOOGL", "MSFT"]
    adapter.provider.inject_error = True

    # Should handle errors gracefully
    with pytest.raises(QuoteError):
        await asyncio.gather(
            *[adapter.get_quotes([symbol]) for symbol in symbols]
        )

@pytest.mark.asyncio
async def test_error_during_subscription(market_data_adapter):
    """Test handling of errors during subscription"""
    adapter = await market_data_adapter
    # Inject error during subscription
    adapter.provider.inject_error = True
    with pytest.raises(SubscriptionError):
        await adapter.subscribe(["AAPL"])

@pytest.mark.asyncio
async def test_rate_limit_burst_handling(market_data_adapter):
    """Test burst handling in rate limiting"""
    adapter = await market_data_adapter
    # Track request times
    request_times = []
    
    # Make burst of 5 requests
    for _ in range(5):
        start_time = datetime.now()
        try:
            await adapter.get_quotes(["AAPL"])
        except Exception:
            pass
        request_times.append(datetime.now() - start_time)
    
    # First 3 should be quick (burst allowed)
    assert all(t.total_seconds() < 0.1 for t in request_times[:3])
    
    # Last 2 should have stricter rate limiting
    assert all(t.total_seconds() >= adapter.provider.config.min_request_interval * 1.5 
              for t in request_times[3:])

@pytest.mark.asyncio
async def test_dual_parameter_error_callback(mock_config):
    """Test error callback with both error and context parameters"""
    error_data = {}
    
    async def error_callback(error, context):
        error_data['error'] = error
        error_data['context'] = context
    
    service = MockRealTimeDataService()
    adapter = MarketDataAdapter(ErrorInjectingProvider, mock_config, service, error_callback)
    await adapter.start()
    
    try:
        # Inject error during quote retrieval
        adapter.provider.inject_error = True
        with pytest.raises(QuoteError):
            await adapter.get_quotes(["AAPL"])
        
        # Verify both error and context were passed to callback
        assert isinstance(error_data['error'], Exception)
        assert error_data['context'].operation == "get_quote"
        assert error_data['context'].symbols == ["AAPL"]
    finally:
        await adapter.stop()

@pytest.mark.asyncio
async def test_stale_data_detection(market_data_adapter):
    """Test detection and handling of stale market data"""
    adapter = await market_data_adapter
    stale_detected = False
    
    def stale_callback(error, context):
        nonlocal stale_detected
        if "Stale data detected" in context.details:
            stale_detected = True
    
    adapter.on_error = stale_callback
    
    # Subscribe to symbol and simulate successful update
    await adapter.subscribe(["AAPL"])
    adapter._last_successful_updates["AAPL"] = (
        datetime.now() - timedelta(seconds=10)
    )
    
    # Force an error to trigger stale data check
    adapter.provider.inject_error = True
    try:
        await adapter.get_quotes(["AAPL"])
    except Exception:
        pass
    
    assert stale_detected, "Stale data was not detected"

@pytest.mark.asyncio
async def test_error_recovery_strategies(market_data_adapter):
    """Test various error recovery strategies"""
    adapter = await market_data_adapter
    recovery_attempts = 0
    
    async def recovery_callback(error, context):
        nonlocal recovery_attempts
        if "Recovery attempted" in context.details:
            recovery_attempts += 1
    
    adapter.on_error = recovery_callback
    
    # Test connection error recovery
    adapter.provider.inject_error = True
    with pytest.raises(SubscriptionError):
        await adapter.subscribe(["AAPL"])
    
    assert recovery_attempts > 0, "No recovery attempts were made"
    
    # Test quote error with exponential backoff
    start_time = datetime.now()
    with pytest.raises(QuoteError):
        await adapter.get_quotes(["AAPL"])
    
    retry_time = datetime.now() - start_time
    assert retry_time.total_seconds() >= adapter.RETRY_DELAY, (
        "Exponential backoff not applied"
    )

@pytest.mark.asyncio
async def test_provider_reconnection(market_data_adapter, mock_config):
    """Test provider reconnection after failure"""
    adapter = await market_data_adapter
    
    # Force disconnect
    await adapter.provider.disconnect()
    
    # Attempt reconnection
    await adapter.provider.connect()
    
    # Should now work
    quotes = await adapter.get_quotes(["AAPL"])
    assert len(quotes) > 0
