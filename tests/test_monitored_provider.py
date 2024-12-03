"""Tests for monitored market data provider."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from backend.services.market_data.monitored_provider import MonitoredMarketDataProvider
from backend.services.market_data.health import (
    ProviderHealthMonitor,
    HealthStatus,
    HealthMetricType
)
from backend.services.market_data.base import MarketDataProvider, MarketDataConfig

class MockBaseProvider(MarketDataProvider):
    """Mock provider for testing monitored wrapper."""
    
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
        return pd.DataFrame()
        
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
def health_monitor():
    """Create health monitor."""
    return ProviderHealthMonitor()

@pytest.fixture
async def monitored_provider(config, health_monitor):
    """Create monitored provider for testing."""
    base_provider = MockBaseProvider(config)
    provider = MonitoredMarketDataProvider(
        provider=base_provider,
        config=config,
        provider_id="test_provider",
        health_monitor=health_monitor
    )
    await provider.connect()
    yield provider
    await provider.disconnect()

@pytest.mark.asyncio
async def test_operation_monitoring(monitored_provider):
    """Test operation monitoring."""
    # Successful operation
    quote = await monitored_provider.get_quote("AAPL")
    assert quote == 100.0
    
    metrics = monitored_provider.health_metrics
    assert metrics['overall_status'] == HealthStatus.HEALTHY.value
    
    # Verify latency was recorded
    latency = metrics['metrics'][HealthMetricType.LATENCY.value]
    assert latency['current_value'] > 0
    
    # Verify success rate
    success_rate = metrics['metrics'][HealthMetricType.SUCCESS_RATE.value]
    assert success_rate['current_value'] == 1.0

@pytest.mark.asyncio
async def test_error_monitoring(monitored_provider):
    """Test error monitoring."""
    # Make provider fail
    monitored_provider._provider.get_quote_called.side_effect = RuntimeError("Test error")
    
    # Execute failing operation
    with pytest.raises(RuntimeError):
        await monitored_provider.get_quote("AAPL")
        
    metrics = monitored_provider.health_metrics
    
    # Verify error was recorded
    error_rate = metrics['metrics'][HealthMetricType.ERROR_RATE.value]
    assert error_rate['current_value'] > 0
    
    # Verify success rate decreased
    success_rate = metrics['metrics'][HealthMetricType.SUCCESS_RATE.value]
    assert success_rate['current_value'] < 1.0

@pytest.mark.asyncio
async def test_cache_metrics_integration(monitored_provider):
    """Test cache metrics integration."""
    cache_metrics = {
        'hit_rate': 0.75,
        'memory_usage': 50 * 1024 * 1024  # 50MB
    }
    
    monitored_provider.update_cache_metrics(cache_metrics)
    metrics = monitored_provider.health_metrics
    
    # Verify cache metrics were recorded
    hit_rate = metrics['metrics'][HealthMetricType.CACHE_HIT_RATE.value]
    assert hit_rate['current_value'] == 0.75
    
    memory = metrics['metrics'][HealthMetricType.MEMORY_USAGE.value]
    assert 0 < memory['current_value'] < 1.0

@pytest.mark.asyncio
async def test_circuit_breaker_metrics_integration(monitored_provider):
    """Test circuit breaker metrics integration."""
    cb_metrics = {
        'failure_rate': 0.15,
        'current_state': 'HALF_OPEN'
    }
    
    monitored_provider.update_circuit_breaker_metrics(cb_metrics)
    metrics = monitored_provider.health_metrics
    
    # Verify circuit breaker metrics were recorded
    error_rate = metrics['metrics'][HealthMetricType.ERROR_RATE.value]
    assert error_rate['current_value'] == 0.15
    
    circuit_state = metrics['metrics'][HealthMetricType.CIRCUIT_STATE.value]
    assert circuit_state['current_value'] == 0.5

@pytest.mark.asyncio
async def test_operation_timing(monitored_provider):
    """Test operation timing tracking."""
    # Add artificial delay
    async def delayed_quote(*args):
        await asyncio.sleep(0.1)
        return 100.0
        
    monitored_provider._provider.get_quote_called.side_effect = delayed_quote
    
    # Execute operation
    await monitored_provider.get_quote("AAPL")
    
    metrics = monitored_provider.health_metrics
    latency = metrics['metrics'][HealthMetricType.LATENCY.value]
    
    # Verify latency was recorded (should be ~100ms)
    assert 50 < latency['current_value'] < 150

@pytest.mark.asyncio
async def test_health_status_transitions(monitored_provider):
    """Test health status transitions."""
    # Start healthy
    assert monitored_provider.health_metrics['overall_status'] == HealthStatus.HEALTHY.value
    
    # Simulate high latency
    async def slow_quote(*args):
        await asyncio.sleep(0.5)  # 500ms
        return 100.0
        
    monitored_provider._provider.get_quote_called.side_effect = slow_quote
    await monitored_provider.get_quote("AAPL")
    
    # Should be degraded due to high latency
    assert monitored_provider.health_metrics['overall_status'] == HealthStatus.DEGRADED.value
    
    # Simulate errors
    monitored_provider._provider.get_quote_called.side_effect = RuntimeError("Test error")
    for _ in range(5):  # Generate multiple errors
        with pytest.raises(RuntimeError):
            await monitored_provider.get_quote("AAPL")
            
    # Should be unhealthy due to errors
    assert monitored_provider.health_metrics['overall_status'] == HealthStatus.UNHEALTHY.value
