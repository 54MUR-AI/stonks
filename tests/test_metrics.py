"""Tests for metrics collection functionality."""

import pytest
from datetime import datetime, timedelta

from backend.services.market_data.metrics import ProviderMetrics

def test_metrics_latency():
    """Test latency metrics collection"""
    metrics = ProviderMetrics()
    
    metrics.record_latency('get_quote', 0.1)
    metrics.record_latency('get_quote', 0.2)
    metrics.record_latency('subscribe', 0.3)
    
    data = metrics.get_metrics()
    assert 'latency' in data
    assert 'get_quote' in data['latency']
    assert 'subscribe' in data['latency']
    
    quote_latency = data['latency']['get_quote']
    assert quote_latency['count'] == 2
    assert quote_latency['avg'] == 0.15
    assert quote_latency['min'] == 0.1
    assert quote_latency['max'] == 0.2

def test_metrics_requests():
    """Test request metrics collection"""
    metrics = ProviderMetrics()
    
    # Successful requests
    metrics.record_request('get_quote', True)
    metrics.record_request('get_quote', True)
    metrics.record_request('subscribe', True)
    
    # Failed requests
    metrics.record_request('get_quote', False)
    metrics.record_request('subscribe', False)
    
    data = metrics.get_metrics()
    assert 'requests' in data
    
    quote_reqs = data['requests']['get_quote']
    assert quote_reqs['total'] == 3
    assert quote_reqs['success'] == 2
    assert quote_reqs['failure'] == 1
    assert quote_reqs['success_rate'] == 2/3
    
    sub_reqs = data['requests']['subscribe']
    assert sub_reqs['total'] == 2
    assert sub_reqs['success'] == 1
    assert sub_reqs['failure'] == 1
    assert sub_reqs['success_rate'] == 0.5

def test_metrics_errors():
    """Test error metrics collection"""
    metrics = ProviderMetrics()
    
    metrics.record_error('RuntimeError')
    metrics.record_error('ValueError')
    metrics.record_error('RuntimeError')
    
    data = metrics.get_metrics()
    assert 'errors' in data
    assert data['errors']['RuntimeError'] == 2
    assert data['errors']['ValueError'] == 1

def test_metrics_symbol_updates():
    """Test symbol update metrics collection"""
    metrics = ProviderMetrics()
    
    metrics.record_symbol_update('AAPL', 150.0, 1000)
    metrics.record_symbol_update('GOOGL', 2500.0, 500)
    metrics.record_symbol_update('AAPL', 151.0, 1200)
    
    data = metrics.get_metrics()
    assert 'symbols' in data
    
    aapl = data['symbols']['AAPL']
    assert aapl['last_price'] == 151.0
    assert aapl['last_volume'] == 1200
    assert aapl['update_count'] == 2
    
    googl = data['symbols']['GOOGL']
    assert googl['last_price'] == 2500.0
    assert googl['last_volume'] == 500
    assert googl['update_count'] == 1

def test_metrics_reset():
    """Test metrics reset functionality"""
    metrics = ProviderMetrics()
    
    metrics.record_latency('get_quote', 0.1)
    metrics.record_request('get_quote', True)
    metrics.record_error('RuntimeError')
    metrics.record_symbol_update('AAPL', 150.0, 1000)
    
    metrics.reset()
    data = metrics.get_metrics()
    
    assert not data['latency']
    assert not data['requests']
    assert not data['errors']
    assert not data['symbols']

def test_metrics_window():
    """Test metrics windowing functionality"""
    metrics = ProviderMetrics(window_size=timedelta(seconds=0.1))
    
    # Record some metrics
    metrics.record_latency('get_quote', 0.1)
    metrics.record_request('get_quote', True)
    metrics.record_error('RuntimeError')
    metrics.record_symbol_update('AAPL', 150.0, 1000)
    
    # Wait for window to expire
    import asyncio
    asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.2))
    
    # Record new metrics
    metrics.record_latency('subscribe', 0.2)
    metrics.record_request('subscribe', True)
    
    data = metrics.get_metrics()
    
    # Old metrics should be gone
    assert 'get_quote' not in data['latency']
    assert 'get_quote' not in data['requests']
    assert not data['errors']
    assert not data['symbols']
    
    # New metrics should be present
    assert 'subscribe' in data['latency']
    assert 'subscribe' in data['requests']
