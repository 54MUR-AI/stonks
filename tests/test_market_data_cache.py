"""Tests for market data caching."""

import asyncio
import pytest
from datetime import datetime, timedelta

from backend.services.market_data.cache import MarketDataCache, CacheMetrics

@pytest.mark.asyncio
async def test_cache_basic_operations():
    """Test basic cache operations."""
    cache = MarketDataCache()
    await cache.start()
    
    # Test set and get
    await cache.set("test_key", "test_value")
    value = await cache.get("test_key")
    assert value == "test_value"
    
    # Test missing key
    value = await cache.get("missing_key")
    assert value is None
    
    # Test invalidate
    await cache.invalidate("test_key")
    value = await cache.get("test_key")
    assert value is None
    
    await cache.stop()

@pytest.mark.asyncio
async def test_cache_ttl():
    """Test cache TTL functionality."""
    cache = MarketDataCache(default_ttl=timedelta(seconds=0.1))
    await cache.start()
    
    await cache.set("test_key", "test_value")
    value = await cache.get("test_key")
    assert value == "test_value"
    
    # Wait for TTL to expire
    await asyncio.sleep(0.2)
    value = await cache.get("test_key")
    assert value is None
    
    await cache.stop()

@pytest.mark.asyncio
async def test_cache_memory_limit():
    """Test cache memory limit enforcement."""
    # Set tiny cache size to force evictions
    cache = MarketDataCache(max_size_bytes=100)
    await cache.start()
    
    # Add entries until we exceed size
    for i in range(10):
        await cache.set(f"key_{i}", "x" * 20)  # Each entry ~20 bytes
        
    # Verify some early entries were evicted
    assert len([k for k, v in cache._cache.items()]) < 10
    
    await cache.stop()

@pytest.mark.asyncio
async def test_cache_metrics():
    """Test cache metrics collection."""
    cache = MarketDataCache()
    await cache.start()
    
    # Generate some cache activity
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    
    # Some hits
    await cache.get("key1")
    await cache.get("key2")
    await cache.get("key1")
    
    # Some misses
    await cache.get("missing1")
    await cache.get("missing2")
    
    metrics = cache.metrics
    assert metrics['hits'] == 3
    assert metrics['misses'] == 2
    assert 0.5 < metrics['hit_rate'] < 0.7
    assert 'avg_latencies' in metrics
    
    await cache.stop()

@pytest.mark.asyncio
async def test_cache_cleanup():
    """Test cache cleanup of expired entries."""
    cache = MarketDataCache(
        default_ttl=timedelta(seconds=0.1),
        cleanup_interval=timedelta(seconds=0.2)
    )
    await cache.start()
    
    # Add some entries
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    
    # Wait for cleanup
    await asyncio.sleep(0.3)
    
    # Verify entries were cleaned up
    assert not cache._cache
    
    await cache.stop()

@pytest.mark.asyncio
async def test_cache_concurrent_access():
    """Test concurrent cache access."""
    cache = MarketDataCache()
    await cache.start()
    
    async def worker(id: int):
        for i in range(100):
            key = f"key_{id}_{i}"
            await cache.set(key, f"value_{id}_{i}")
            await cache.get(key)
            
    # Run multiple workers concurrently
    workers = [worker(i) for i in range(5)]
    await asyncio.gather(*workers)
    
    # Verify no errors occurred
    metrics = cache.metrics
    assert metrics['hits'] + metrics['misses'] > 0
    
    await cache.stop()

@pytest.mark.asyncio
async def test_cache_clear():
    """Test cache clear functionality."""
    cache = MarketDataCache()
    await cache.start()
    
    # Add some entries
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    
    # Clear cache
    await cache.clear()
    
    # Verify cache is empty
    assert not cache._cache
    
    # Verify metrics were reset
    metrics = cache.metrics
    assert metrics['hits'] == 0
    assert metrics['misses'] == 0
    
    await cache.stop()

@pytest.mark.asyncio
async def test_cache_size_estimation():
    """Test cache size estimation."""
    cache = MarketDataCache()
    
    # Test different types
    test_data = {
        'int': 42,
        'float': 3.14,
        'str': "test string",
        'list': [1, 2, 3],
        'dict': {'a': 1, 'b': 2},
        'nested': {'x': [1, 2, {'y': 'z'}]}
    }
    
    for key, value in test_data.items():
        size = cache._estimate_size(value)
        assert size > 0, f"Size estimation failed for {key}"
