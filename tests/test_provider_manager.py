"""Tests for provider manager."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from backend.services.market_data.provider_manager import (
    ProviderManager,
    ProviderPriority,
    ProviderState
)
from backend.services.market_data.base import MarketDataProvider, MarketDataConfig
from backend.services.market_data.health import (
    ProviderHealthMonitor,
    HealthStatus
)

class MockProvider(MarketDataProvider):
    """Mock provider for testing."""
    def __init__(self, config: MarketDataConfig):
        super().__init__(config)
        self.connect = AsyncMock()
        self.disconnect = AsyncMock()
        self.subscribe = AsyncMock()
        self.unsubscribe = AsyncMock()
        self.get_quote = AsyncMock(return_value=100.0)
        self.is_connected = False
        
    async def _connect(self) -> None:
        await self.connect()
        self.is_connected = True
        
    async def _disconnect(self) -> None:
        await self.disconnect()
        self.is_connected = False

@pytest.fixture
def config():
    """Create test config."""
    return MarketDataConfig(
        credentials=Mock(),
        base_url="http://test",
        websocket_url="ws://test"
    )

@pytest.fixture
def health_monitor():
    """Create health monitor."""
    return ProviderHealthMonitor()

@pytest.fixture
async def manager(config, health_monitor):
    """Create provider manager."""
    manager = ProviderManager(config, health_monitor)
    await manager.start()
    yield manager
    await manager.stop()

@pytest.mark.asyncio
async def test_add_provider(manager, config):
    """Test adding providers."""
    provider1 = MockProvider(config)
    provider2 = MockProvider(config)
    
    # Add primary provider
    await manager.add_provider("provider1", provider1, ProviderPriority.PRIMARY)
    assert "provider1" in manager.providers
    assert manager.active_provider.provider == provider1
    
    # Add secondary provider
    await manager.add_provider("provider2", provider2, ProviderPriority.SECONDARY)
    assert "provider2" in manager.providers
    assert manager.active_provider.provider == provider1  # Still primary

@pytest.mark.asyncio
async def test_remove_provider(manager, config):
    """Test removing providers."""
    provider1 = MockProvider(config)
    provider2 = MockProvider(config)
    
    await manager.add_provider("provider1", provider1, ProviderPriority.PRIMARY)
    await manager.add_provider("provider2", provider2, ProviderPriority.SECONDARY)
    
    # Remove primary provider
    await manager.remove_provider("provider1")
    assert "provider1" not in manager.providers
    assert manager.active_provider.provider == provider2

@pytest.mark.asyncio
async def test_provider_failover(manager, config):
    """Test automatic provider failover."""
    provider1 = MockProvider(config)
    provider2 = MockProvider(config)
    
    await manager.add_provider("provider1", provider1, ProviderPriority.PRIMARY)
    await manager.add_provider("provider2", provider2, ProviderPriority.SECONDARY)
    
    # Simulate primary provider failure
    provider1.get_quote.side_effect = RuntimeError("Test error")
    
    # Should fail over to secondary after max failures
    for _ in range(manager.max_consecutive_failures):
        try:
            await manager.get_quote("AAPL")
        except:
            pass
            
    assert manager.active_provider.provider == provider2

@pytest.mark.asyncio
async def test_provider_recovery(manager, config):
    """Test provider recovery after failure."""
    provider1 = MockProvider(config)
    provider2 = MockProvider(config)
    
    await manager.add_provider("provider1", provider1, ProviderPriority.PRIMARY)
    await manager.add_provider("provider2", provider2, ProviderPriority.SECONDARY)
    
    # Fail primary provider
    provider1.get_quote.side_effect = RuntimeError("Test error")
    for _ in range(manager.max_consecutive_failures):
        try:
            await manager.get_quote("AAPL")
        except:
            pass
            
    assert manager.active_provider.provider == provider2
    
    # Recover primary provider
    provider1.get_quote.side_effect = None
    provider1.get_quote.return_value = 100.0
    
    # Wait for health check and recovery
    await asyncio.sleep(2)
    assert manager.active_provider.provider == provider1

@pytest.mark.asyncio
async def test_symbol_subscription(manager, config):
    """Test symbol subscription handling."""
    provider1 = MockProvider(config)
    await manager.add_provider("provider1", provider1, ProviderPriority.PRIMARY)
    
    # Subscribe to symbols
    symbols = ["AAPL", "GOOGL"]
    await manager.subscribe(symbols)
    
    provider1.subscribe.assert_called_once_with(symbols)
    assert all(s in manager.active_provider.active_symbols for s in symbols)
    
    # Unsubscribe from symbols
    await manager.unsubscribe(symbols)
    provider1.unsubscribe.assert_called_once_with(symbols)
    assert not manager.active_provider.active_symbols

@pytest.mark.asyncio
async def test_provider_scoring(manager, config):
    """Test provider scoring and selection."""
    provider1 = MockProvider(config)
    provider2 = MockProvider(config)
    
    await manager.add_provider("provider1", provider1, ProviderPriority.PRIMARY)
    await manager.add_provider("provider2", provider2, ProviderPriority.SECONDARY)
    
    # Get initial scores
    primary_state = manager.providers["provider1"]
    secondary_state = manager.providers["provider2"]
    
    primary_score = primary_state.get_score()
    secondary_score = secondary_state.get_score()
    
    # Primary should have higher score
    assert primary_score > secondary_score
    
    # Simulate degraded primary
    primary_state.health.update_metric("latency", 800.0)  # High latency
    primary_state.health.update_metric("error_rate", 0.15)  # High error rate
    
    new_primary_score = primary_state.get_score()
    assert new_primary_score < primary_score
    
    # Wait for health check and failover
    await asyncio.sleep(2)
    assert manager.active_provider.provider == provider2

@pytest.mark.asyncio
async def test_provider_cooldown(manager, config):
    """Test provider switch cooldown."""
    provider1 = MockProvider(config)
    provider2 = MockProvider(config)
    
    await manager.add_provider("provider1", provider1, ProviderPriority.PRIMARY)
    await manager.add_provider("provider2", provider2, ProviderPriority.SECONDARY)
    
    # Force switch to secondary
    primary_state = manager.providers["provider1"]
    primary_state.health.update_metric("error_rate", 0.9)  # Very high error rate
    
    # Wait for switch
    await asyncio.sleep(2)
    assert manager.active_provider.provider == provider2
    
    # Try immediate switch back
    primary_state.health.update_metric("error_rate", 0.0)  # Recover
    await asyncio.sleep(1)
    
    # Should still be on secondary due to cooldown
    assert manager.active_provider.provider == provider2
    
    # Wait for cooldown
    await asyncio.sleep(manager.min_switch_interval)
    assert manager.active_provider.provider == provider1
