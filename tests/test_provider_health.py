"""Tests for provider health monitoring."""

import asyncio
import pytest
from datetime import datetime, timedelta

from backend.services.market_data.health import (
    ProviderHealth,
    ProviderHealthMonitor,
    HealthMetricType,
    HealthStatus,
    HealthThresholds
)

@pytest.fixture
def health():
    """Create provider health monitor for testing."""
    return ProviderHealth("test_provider")

@pytest.fixture
def monitor():
    """Create health monitoring system for testing."""
    return ProviderHealthMonitor()

def test_health_status_calculation(health):
    """Test health status calculation."""
    # Test latency thresholds
    status = health.update_metric(HealthMetricType.LATENCY, 50.0)
    assert status == HealthStatus.HEALTHY
    
    status = health.update_metric(HealthMetricType.LATENCY, 150.0)
    assert status == HealthStatus.DEGRADED
    
    status = health.update_metric(HealthMetricType.LATENCY, 600.0)
    assert status == HealthStatus.UNHEALTHY
    
    # Test error rate thresholds
    status = health.update_metric(HealthMetricType.ERROR_RATE, 0.02)
    assert status == HealthStatus.HEALTHY
    
    status = health.update_metric(HealthMetricType.ERROR_RATE, 0.08)
    assert status == HealthStatus.DEGRADED
    
    status = health.update_metric(HealthMetricType.ERROR_RATE, 0.20)
    assert status == HealthStatus.UNHEALTHY

def test_metric_history(health):
    """Test metric history tracking."""
    metric_type = HealthMetricType.LATENCY
    
    # Add some values
    for value in [100.0, 200.0, 300.0]:
        health.update_metric(metric_type, value)
        
    metrics = health.get_metrics()
    metric = metrics['metrics'][metric_type.value]
    
    assert metric['current_value'] == 300.0
    assert metric['average'] == 200.0
    assert 'stddev' in metric
    assert metric['history_size'] == 3

def test_overall_status(health):
    """Test overall health status calculation."""
    # All healthy
    health.update_metric(HealthMetricType.LATENCY, 50.0)
    health.update_metric(HealthMetricType.ERROR_RATE, 0.02)
    assert health.overall_status == HealthStatus.HEALTHY
    
    # One degraded
    health.update_metric(HealthMetricType.LATENCY, 150.0)
    assert health.overall_status == HealthStatus.DEGRADED
    
    # One unhealthy
    health.update_metric(HealthMetricType.ERROR_RATE, 0.20)
    assert health.overall_status == HealthStatus.UNHEALTHY

def test_health_check_recording(health):
    """Test health check recording."""
    # Record some checks
    health.record_health_check(True)
    health.record_health_check(True)
    health.record_health_check(False)
    
    metrics = health.get_metrics()
    availability = metrics['metrics'][HealthMetricType.AVAILABILITY.value]
    assert availability['current_value'] == 2/3

@pytest.mark.asyncio
async def test_monitor_provider_registration(monitor):
    """Test provider registration."""
    # Register providers
    provider1 = monitor.register_provider("provider1")
    provider2 = monitor.register_provider("provider2")
    
    assert "provider1" in monitor._providers
    assert "provider2" in monitor._providers
    
    # Get provider health
    assert monitor.get_provider_health("provider1") == provider1
    assert monitor.get_provider_health("provider2") == provider2
    assert monitor.get_provider_health("unknown") is None

@pytest.mark.asyncio
async def test_monitor_all_providers(monitor):
    """Test monitoring multiple providers."""
    # Register providers
    monitor.register_provider("provider1")
    monitor.register_provider("provider2")
    
    # Start monitoring
    await monitor.start()
    
    # Let it run for a bit
    await asyncio.sleep(0.1)
    
    # Get all health
    health = monitor.get_all_health()
    assert "provider1" in health
    assert "provider2" in health
    
    await monitor.stop()

@pytest.mark.asyncio
async def test_monitor_lifecycle(monitor):
    """Test monitor start/stop lifecycle."""
    await monitor.start()
    assert monitor._monitor_task is not None
    
    await monitor.stop()
    assert monitor._monitor_task is None
    
    # Should be able to start again
    await monitor.start()
    assert monitor._monitor_task is not None
    await monitor.stop()

def test_custom_thresholds():
    """Test custom health thresholds."""
    thresholds = HealthThresholds(
        latency_warning_ms=200.0,
        latency_critical_ms=1000.0,
        error_rate_warning=0.10,
        error_rate_critical=0.30
    )
    
    health = ProviderHealth("test", thresholds=thresholds)
    
    # Test custom latency thresholds
    status = health.update_metric(HealthMetricType.LATENCY, 150.0)
    assert status == HealthStatus.HEALTHY
    
    status = health.update_metric(HealthMetricType.LATENCY, 250.0)
    assert status == HealthStatus.DEGRADED
    
    status = health.update_metric(HealthMetricType.LATENCY, 1100.0)
    assert status == HealthStatus.UNHEALTHY
    
    # Test custom error rate thresholds
    status = health.update_metric(HealthMetricType.ERROR_RATE, 0.05)
    assert status == HealthStatus.HEALTHY
    
    status = health.update_metric(HealthMetricType.ERROR_RATE, 0.15)
    assert status == HealthStatus.DEGRADED
    
    status = health.update_metric(HealthMetricType.ERROR_RATE, 0.35)
    assert status == HealthStatus.UNHEALTHY
