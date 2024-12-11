import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from backend.services.provider_management import (
    MarketDataProviderManager,
    ProviderHealth,
    ErrorSeverity
)
from backend.models.market_data import MarketDataProvider, ProviderStatus
from backend.exceptions import (
    ProviderError,
    NoAvailableProviderError,
    ProviderConnectionError,
    ProviderRateLimitError
)

class MockProvider(MarketDataProvider):
    def __init__(self, name: str, behavior: str = "normal"):
        super().__init__(name)
        self.behavior = behavior
        self.connect = AsyncMock()
        self.disconnect = AsyncMock()
        self.get_data = AsyncMock()
        self.reconnect = AsyncMock()
        self._configure_behavior()

    def _configure_behavior(self):
        if self.behavior == "fail_connect":
            self.connect.side_effect = ProviderConnectionError("Connection failed")
        elif self.behavior == "rate_limit":
            self.get_data.side_effect = ProviderRateLimitError("Rate limit exceeded")
        elif self.behavior == "timeout":
            self.get_data.side_effect = asyncio.TimeoutError()
        elif self.behavior == "normal":
            self.get_data.return_value = {"price": 100.0, "volume": 1000}

@pytest.fixture
def primary_provider():
    return MockProvider("primary")

@pytest.fixture
def backup_provider1():
    return MockProvider("backup1")

@pytest.fixture
def backup_provider2():
    return MockProvider("backup2")

@pytest.fixture
def provider_manager(primary_provider, backup_provider1, backup_provider2):
    return MarketDataProviderManager(
        primary_provider=primary_provider,
        backup_providers=[backup_provider1, backup_provider2]
    )

@pytest.mark.asyncio
async def test_initialization(provider_manager):
    """Test proper initialization of provider manager"""
    assert provider_manager.current_provider.name == "primary"
    assert len(provider_manager.provider_metrics) == 3
    
    for provider_name, metrics in provider_manager.provider_metrics.items():
        assert isinstance(metrics, ProviderHealth)
        assert metrics.success_rate == 1.0
        assert metrics.error_count == 0

@pytest.mark.asyncio
async def test_successful_data_fetch(provider_manager):
    """Test successful market data retrieval"""
    data = await provider_manager.get_data("AAPL")
    assert data == {"price": 100.0, "volume": 1000}
    
    metrics = provider_manager.provider_metrics[provider_manager.current_provider.name]
    assert metrics.success_rate > 0.9
    assert metrics.error_count == 0
    assert metrics.consecutive_failures == 0

@pytest.mark.asyncio
async def test_provider_failover(provider_manager):
    """Test automatic failover to backup provider"""
    # Make primary provider fail
    provider_manager.current_provider.get_data.side_effect = ProviderConnectionError("Connection lost")
    
    # Should automatically switch to backup1
    data = await provider_manager.get_data("AAPL")
    assert provider_manager.current_provider.name == "backup1"
    assert data == {"price": 100.0, "volume": 1000}

@pytest.mark.asyncio
async def test_provider_recovery(provider_manager):
    """Test provider recovery after failures"""
    # Simulate failures
    provider_manager.provider_metrics["primary"].consecutive_failures = 3
    provider_manager.provider_metrics["primary"].last_error = datetime.now() - timedelta(minutes=6)
    
    # Start health monitoring
    await provider_manager.start()
    await asyncio.sleep(0.1)  # Allow monitor to run
    
    # Should attempt reconnection
    assert provider_manager.primary_provider.reconnect.called

@pytest.mark.asyncio
async def test_error_classification(provider_manager):
    """Test error severity classification"""
    conn_error = ProviderConnectionError("Connection failed")
    rate_limit_error = ProviderRateLimitError("Rate limit exceeded")
    timeout_error = asyncio.TimeoutError()
    
    assert provider_manager._classify_error(conn_error) == ErrorSeverity.HIGH
    assert provider_manager._classify_error(rate_limit_error) == ErrorSeverity.MEDIUM
    assert provider_manager._classify_error(timeout_error) == ErrorSeverity.HIGH

@pytest.mark.asyncio
async def test_metrics_tracking(provider_manager):
    """Test provider metrics tracking"""
    # Successful calls
    await provider_manager.get_data("AAPL")
    await provider_manager.get_data("GOOGL")
    
    metrics = provider_manager.provider_metrics[provider_manager.current_provider.name]
    assert metrics.success_rate > 0.9
    assert metrics.avg_latency > 0
    assert metrics.last_success is not None
    
    # Failed call
    provider_manager.current_provider.get_data.side_effect = ProviderError("Test error")
    with pytest.raises(Exception):
        await provider_manager.get_data("MSFT")
    
    metrics = provider_manager.provider_metrics[provider_manager.current_provider.name]
    assert metrics.error_count == 1
    assert metrics.last_error is not None

@pytest.mark.asyncio
async def test_no_available_providers(provider_manager):
    """Test behavior when no providers are available"""
    # Make all providers fail
    for provider in [provider_manager.primary_provider] + provider_manager.backup_providers:
        provider.get_data.side_effect = ProviderConnectionError("Connection failed")
    
    with pytest.raises(NoAvailableProviderError):
        await provider_manager.get_data("AAPL")

@pytest.mark.asyncio
async def test_provider_status_updates(provider_manager):
    """Test provider status updates based on failures"""
    provider = provider_manager.current_provider
    metrics = provider_manager.provider_metrics[provider.name]
    
    # Initial state
    assert provider.status == ProviderStatus.HEALTHY
    
    # Simulate failures
    metrics.consecutive_failures = 3
    await provider_manager._monitor_provider_health()
    assert provider.status == ProviderStatus.DEGRADED
    
    metrics.consecutive_failures = 5
    await provider_manager._monitor_provider_health()
    assert provider.status == ProviderStatus.UNHEALTHY

@pytest.mark.asyncio
async def test_graceful_shutdown(provider_manager):
    """Test graceful shutdown of provider manager"""
    await provider_manager.start()
    await provider_manager.stop()
    
    # Health monitor should be cancelled
    assert provider_manager._health_monitor_task.cancelled()
    assert provider_manager.current_provider.disconnect.called

@pytest.mark.asyncio
async def test_retry_strategy(provider_manager):
    """Test retry strategy for different error severities"""
    # Test medium severity error (should retry)
    provider_manager.current_provider.get_data.side_effect = [
        ProviderRateLimitError("Rate limit"),
        {"price": 100.0, "volume": 1000}
    ]
    
    data = await provider_manager.get_data("AAPL")
    assert data == {"price": 100.0, "volume": 1000}
    assert provider_manager.current_provider.get_data.call_count == 2

@pytest.mark.asyncio
async def test_provider_metrics_api(provider_manager):
    """Test the provider metrics API"""
    # Generate some activity
    await provider_manager.get_data("AAPL")
    provider_manager.current_provider.get_data.side_effect = ProviderError("Test error")
    with pytest.raises(Exception):
        await provider_manager.get_data("MSFT")
    
    metrics = await provider_manager.get_provider_metrics()
    assert len(metrics) == 3
    
    primary_metrics = metrics["primary"]
    assert isinstance(primary_metrics.status, ProviderStatus)
    assert isinstance(primary_metrics.success_rate, float)
    assert isinstance(primary_metrics.avg_latency, float)
    assert isinstance(primary_metrics.error_count, int)
