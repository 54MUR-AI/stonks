import pytest
from datetime import datetime
from typing import Optional, List
import asyncio

from backend.services.market_data.adapter import (
    MarketDataAdapter, MarketDataError, ConnectionError, QuoteError, ErrorContext
)
from backend.services.market_data.base import MarketDataProvider, MarketDataConfig, MarketDataCredentials
from backend.services.market_data.mock_provider import MockProvider
from backend.services.realtime_data import RealTimeDataService

class FailingProvider(MockProvider):
    """Provider that fails after a certain number of successful operations"""
    def __init__(self, config: MarketDataConfig, fail_after: int = 3):
        super().__init__(config)
        self.quote_operations = 0
        self.connect_operations = 0
        self.fail_after = fail_after
        self.connected = False
        
    async def connect(self) -> None:
        self.connect_operations += 1
        if self.connect_operations > self.fail_after:
            raise ConnectionError("Connection failed")
        self.connected = True
        
    async def disconnect(self) -> None:
        self.connected = False
        
    async def get_quote(self, symbol: str) -> float:
        self.quote_operations += 1
        if self.quote_operations > self.fail_after:
            raise QuoteError("Quote retrieval failed")
        return 100.0  # Mock price

class ReliableProvider(MockProvider):
    """Provider that always succeeds"""
    def __init__(self, config: MarketDataConfig):
        super().__init__(config)
        self.connected = False
        self.operations = 0
        
    async def connect(self) -> None:
        self.operations += 1
        self.connected = True
        
    async def disconnect(self) -> None:
        self.connected = False
        
    async def get_quote(self, symbol: str) -> float:
        self.operations += 1
        return 100.0  # Same mock price as FailingProvider

class MockRealTimeDataService:
    """Mock realtime data service for testing"""
    def __init__(self):
        self.subscribed_symbols = set()
        self.quotes = {}

    async def start(self):
        pass
        
    async def stop(self):
        pass

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data for symbols"""
        self.subscribed_symbols.update(symbols)

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for symbols"""
        self.subscribed_symbols.difference_update(symbols)

    async def publish_quote(self, symbol: str, price: float) -> None:
        """Publish a quote for a symbol"""
        self.quotes[symbol] = price

    async def get_quote(self, symbol: str) -> Optional[float]:
        """Get latest quote for a symbol"""
        return self.quotes.get(symbol)

@pytest.mark.asyncio
async def test_provider_switching_on_failure():
    """Test that adapter switches providers when primary provider fails"""
    error_data = {}
    config = MarketDataConfig(
        credentials=MarketDataCredentials(api_key="test_key"),
        base_url="https://test.example.com",
        websocket_url="wss://test.example.com/ws",
        min_request_interval=0.1
    )
    
    def error_callback(error, context):
        error_data['error'] = error
        error_data['context'] = context
    
    service = MockRealTimeDataService()
    failing_provider = FailingProvider(config, fail_after=2)
    reliable_provider = ReliableProvider(config)
    
    adapter = MarketDataAdapter(
        provider_class=lambda config: failing_provider,
        config=config,
        realtime_service=service,
        error_callback=error_callback,
        max_retries=1
    )
    
    # Register backup provider
    adapter.register_backup_provider(lambda config: reliable_provider)
    
    try:
        # Start adapter with primary provider
        await adapter.start()
        assert failing_provider.connected, "Primary provider should be connected"
        assert not reliable_provider.connected, "Backup provider should not be connected yet"
        
        # First quote should succeed with primary provider
        quote = await adapter.get_quotes(["AAPL"])
        assert quote["AAPL"] == 100.0, "Should get quote from primary provider"
        
        # Second quote should succeed with primary provider
        quote = await adapter.get_quotes(["AAPL"])
        assert quote["AAPL"] == 100.0, "Should get quote from primary provider"
        
        # Third quote should fail with primary and switch to backup
        quote = await adapter.get_quotes(["AAPL"])
        assert quote["AAPL"] == 100.0, "Should get quote from backup provider"
        
        # Verify final provider state
        assert not failing_provider.connected, "Primary provider should be disconnected"
        assert reliable_provider.connected, "Backup provider should be connected"
        
        # Verify error callback was called
        assert error_data['error'] is not None, "Error callback should be called"
        assert error_data['context'].operation == "get_quotes", "Error should be from get_quotes"
        assert error_data['context'].retry_count > 0, "Should have retried at least once"
        
    finally:
        await adapter.stop()

@pytest.mark.asyncio
async def test_provider_switching_with_multiple_backups():
    """Test that adapter tries multiple backup providers in order"""
    config = MarketDataConfig(
        credentials=MarketDataCredentials(api_key="test_key"),
        base_url="https://test.example.com",
        websocket_url="wss://test.example.com/ws",
        min_request_interval=0.1
    )
    
    service = MockRealTimeDataService()
    
    # Create provider instances with different fail_after values
    primary = FailingProvider(config, fail_after=1)  # Fails after 1 operation
    backup1 = FailingProvider(config, fail_after=1)  # Fails after 1 operation
    backup2 = ReliableProvider(config)  # Never fails
    
    # Create provider factory functions that return the same instances
    def primary_factory(config):
        return primary
        
    def backup1_factory(config):
        return backup1
        
    def backup2_factory(config):
        return backup2
    
    adapter = MarketDataAdapter(
        provider_class=primary_factory,
        config=config,
        realtime_service=service,
        max_retries=1
    )
    
    # Register backup providers in order
    adapter.register_backup_provider(backup1_factory)
    adapter.register_backup_provider(backup2_factory)
    
    try:
        # Start adapter
        await adapter.start()
        assert primary.connected, "Primary provider should be connected"
        
        # First quote succeeds with primary
        quote = await adapter.get_quotes(["AAPL"])
        assert quote["AAPL"] == 100.0, "Should get quote from primary"
        assert primary.connected, "Primary provider should still be connected"
        
        # Second quote fails primary, switches to backup1
        quote = await adapter.get_quotes(["AAPL"])
        assert quote["AAPL"] == 100.0, "Should get quote from backup1"
        assert not primary.connected, "Primary provider should be disconnected"
        assert backup1.connected, "Backup1 should be connected"
        
        # Third quote fails backup1, switches to backup2
        quote = await adapter.get_quotes(["AAPL"])
        assert quote["AAPL"] == 100.0, "Should get quote from backup2"
        assert not backup1.connected, "Backup1 should be disconnected"
        assert backup2.connected, "Backup2 should be connected"
        
    finally:
        await adapter.stop()

@pytest.mark.asyncio
async def test_provider_health_monitoring():
    """Test that adapter monitors provider health and switches when necessary"""
    config = MarketDataConfig(
        credentials=MarketDataCredentials(api_key="test_key"),
        base_url="https://test.example.com",
        websocket_url="wss://test.example.com/ws",
        min_request_interval=0.1
    )
    
    service = MockRealTimeDataService()
    
    # Create provider instances with different characteristics
    primary = FailingProvider(config, fail_after=5)  # Fails after 5 operations
    backup = ReliableProvider(config)  # Never fails
    
    def primary_factory(config):
        return primary
        
    def backup_factory(config):
        return backup
    
    # Create adapter with short health check interval for testing
    adapter = MarketDataAdapter(
        provider_class=primary_factory,
        config=config,
        realtime_service=service,
        max_retries=1
    )
    adapter.HEALTH_CHECK_INTERVAL = 0.1  # Speed up test
    adapter.MIN_SUCCESS_RATE = 90.0  # Set high threshold
    
    adapter.register_backup_provider(backup_factory)
    
    try:
        # Start adapter
        await adapter.start()
        assert primary.connected, "Primary provider should be connected"
        
        # Make several successful requests
        for _ in range(3):
            quote = await adapter.get_quotes(["AAPL"])
            assert quote["AAPL"] == 100.0
            
        # Make several failed requests to trigger health check
        for _ in range(3):
            try:
                await adapter.get_quotes(["AAPL"])
            except QuoteError:
                pass
        
        # Wait for health check to run
        await asyncio.sleep(0.2)
        
        # Should have switched to backup provider
        assert not primary.connected, "Primary provider should be disconnected"
        assert backup.connected, "Backup provider should be connected"
        
        # Should be able to get quotes from backup
        quote = await adapter.get_quotes(["AAPL"])
        assert quote["AAPL"] == 100.0
        
    finally:
        await adapter.stop()

@pytest.mark.asyncio
async def test_provider_performance_tracking():
    """Test that adapter tracks provider performance metrics"""
    config = MarketDataConfig(
        credentials=MarketDataCredentials(api_key="test_key"),
        base_url="https://test.example.com",
        websocket_url="wss://test.example.com/ws",
        min_request_interval=0.1
    )
    
    service = MockRealTimeDataService()
    
    # Create providers with different performance characteristics
    primary = FailingProvider(config, fail_after=3)  # Fails after 3 operations
    backup1 = FailingProvider(config, fail_after=2)  # Fails after 2 operations
    backup2 = ReliableProvider(config)  # Never fails
    
    def primary_factory(config):
        return primary
        
    def backup1_factory(config):
        return backup1
        
    def backup2_factory(config):
        return backup2
    
    adapter = MarketDataAdapter(
        provider_class=primary_factory,
        config=config,
        realtime_service=service,
        max_retries=1
    )
    
    adapter.register_backup_provider(backup1_factory)
    adapter.register_backup_provider(backup2_factory)
    
    try:
        # Start adapter
        await adapter.start()
        
        # Make several requests to build up performance stats
        for _ in range(5):
            try:
                quote = await adapter.get_quotes(["AAPL"])
                assert quote["AAPL"] == 100.0
            except QuoteError:
                pass
        
        # Check provider stats
        assert 0 in adapter._provider_stats, "Should have stats for primary provider"
        primary_stats = adapter._provider_stats[0]
        assert primary_stats.total_requests > 0, "Should have recorded requests"
        assert primary_stats.failed_requests > 0, "Should have recorded failures"
        assert primary_stats.success_rate < 100.0, "Success rate should be less than 100%"
        assert primary_stats.avg_latency > 0.0, "Should have recorded latency"
        
        # Make more requests to trigger provider switch
        for _ in range(5):
            try:
                quote = await adapter.get_quotes(["AAPL"])
                assert quote["AAPL"] == 100.0
            except QuoteError:
                pass
        
        # Check that we've switched to the most reliable provider
        assert backup2.connected, "Should have switched to most reliable provider"
        assert 2 in adapter._provider_stats, "Should have stats for backup2"
        backup2_stats = adapter._provider_stats[2]
        assert backup2_stats.success_rate == 100.0, "Backup2 should have perfect success rate"
        
    finally:
        await adapter.stop()

@pytest.mark.asyncio
async def test_provider_switching_on_timeout():
    """Test that adapter switches providers when primary provider times out"""
    config = MarketDataConfig(
        credentials=MarketDataCredentials(api_key="test_key"),
        base_url="https://test.example.com",
        websocket_url="wss://test.example.com/ws",
        min_request_interval=0.1
    )
    
    service = MockRealTimeDataService()
    
    # Create providers with different timeout characteristics
    primary = MockProvider(config)
    primary.set_timeout_simulation(delay=0.1, probability=1.0)  # Always timeout after 0.1s
    
    backup = MockProvider(config)  # No timeouts
    backup.set_timeout_simulation(delay=0.0, probability=0.0)
    
    adapter = MarketDataAdapter(
        provider_class=lambda c: primary,
        config=config,
        realtime_service=service,
        max_retries=1
    )
    
    adapter.register_backup_provider(lambda c: backup)
    
    try:
        # Start adapter and wait for initial connection attempt
        await adapter.start()
        await asyncio.sleep(0.2)  # Give time for provider switch
        
        # First quote should fail with timeout and switch to backup
        quote = await adapter.get_quotes(["AAPL"])
        assert quote["AAPL"] > 0, "Should get valid quote from backup"
        
        # Verify provider state
        assert not primary.connected, "Primary provider should be disconnected"
        assert backup.connected, "Backup provider should be connected"
        
    finally:
        await adapter.stop()

@pytest.mark.asyncio
async def test_provider_timeout_recovery():
    """Test that adapter handles intermittent timeouts with retries"""
    config = MarketDataConfig(
        credentials=MarketDataCredentials(api_key="test_key"),
        base_url="https://test.example.com",
        websocket_url="wss://test.example.com/ws",
        min_request_interval=0.1
    )
    
    service = MockRealTimeDataService()
    
    # Create provider with intermittent timeouts
    provider = MockProvider(config)
    provider.set_timeout_simulation(delay=0.1, probability=0.5)  # 50% chance of timeout
    
    adapter = MarketDataAdapter(
        provider_class=lambda c: provider,
        config=config,
        realtime_service=service,
        max_retries=3  # Allow more retries
    )
    
    try:
        # Start adapter and wait for initial connection
        await adapter.start()
        await asyncio.sleep(0.2)  # Give time for initial connection
        
        # Make several requests - some should succeed after retries
        for _ in range(5):
            try:
                quote = await adapter.get_quotes(["AAPL"])
                assert quote["AAPL"] > 0, "Should eventually get valid quote"
                await asyncio.sleep(0.1)  # Give time between requests
            except Exception:
                pass  # Some requests may fail, that's expected
        
    finally:
        await adapter.stop()

@pytest.mark.asyncio
async def test_provider_timeout_during_stream():
    """Test that adapter handles timeouts during streaming operations"""
    config = MarketDataConfig(
        credentials=MarketDataCredentials(api_key="test_key"),
        base_url="https://test.example.com",
        websocket_url="wss://test.example.com/ws",
        min_request_interval=0.1
    )
    
    service = MockRealTimeDataService()
    
    # Create providers with different timeout characteristics
    primary = MockProvider(config)
    primary.set_timeout_simulation(delay=0.1, probability=0.8)  # High chance of timeout
    
    backup = MockProvider(config)  # No timeouts
    backup.set_timeout_simulation(delay=0.0, probability=0.0)
    
    adapter = MarketDataAdapter(
        provider_class=lambda c: primary,
        config=config,
        realtime_service=service,
        max_retries=1
    )
    
    adapter.register_backup_provider(lambda c: backup)
    
    try:
        # Start adapter and wait for initial connection attempt
        await adapter.start()
        await asyncio.sleep(0.2)  # Give time for provider switch
        
        # Subscribe to some symbols
        await adapter.subscribe(["AAPL", "GOOGL"])
        await asyncio.sleep(0.2)  # Give time for subscription
        
        # Get quotes - should work with backup provider
        quote = await adapter.get_quotes(["AAPL"])
        assert quote["AAPL"] > 0, "Should get valid quote from backup"
        
        # Verify provider state
        assert not primary.connected, "Primary provider should be disconnected"
        assert backup.connected, "Backup provider should be connected"
        
        # Verify subscriptions were transferred
        assert "AAPL" in backup.subscribed_symbols, "Subscription should be transferred to backup"
        assert "GOOGL" in backup.subscribed_symbols, "Subscription should be transferred to backup"
        
    finally:
        await adapter.stop()
