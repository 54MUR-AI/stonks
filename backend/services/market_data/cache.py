"""Market data caching implementation."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set
import logging
from dataclasses import dataclass
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry containing data and metadata."""
    data: Any
    timestamp: datetime
    expiry: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None

class CacheMetrics:
    """Tracks cache performance metrics."""
    
    def __init__(self):
        """Initialize metrics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_usage = 0  # in bytes
        self.operation_latencies = defaultdict(list)
        self._start_time = datetime.now()
        
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
        
    def record_operation(self, operation: str, duration: float):
        """Record operation latency."""
        self.operation_latencies[operation].append(duration)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        avg_latencies = {
            op: sum(times) / len(times) 
            for op, times in self.operation_latencies.items()
        }
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'evictions': self.evictions,
            'memory_usage': self.memory_usage,
            'uptime': (datetime.now() - self._start_time).total_seconds(),
            'avg_latencies': avg_latencies
        }

class MarketDataCache:
    """In-memory cache for market data with TTL and memory management."""
    
    def __init__(
        self,
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB default
        default_ttl: timedelta = timedelta(minutes=5),
        cleanup_interval: timedelta = timedelta(minutes=1)
    ):
        """Initialize cache.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
            default_ttl: Default time-to-live for cache entries
            cleanup_interval: How often to run cleanup
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size_bytes
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._metrics = CacheMetrics()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start cache maintenance tasks."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
    async def stop(self):
        """Stop cache maintenance tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        return self._metrics.get_metrics()
        
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data in bytes."""
        # This is a simple estimation, could be made more accurate
        if isinstance(data, (int, float)):
            return 8
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        elif isinstance(data, dict):
            return sum(
                self._estimate_size(k) + self._estimate_size(v)
                for k, v in data.items()
            )
        elif isinstance(data, (list, tuple, set)):
            return sum(self._estimate_size(item) for item in data)
        return 64  # default estimation for other types
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        try:
            async with self._lock:
                entry = self._cache.get(key)
                if not entry:
                    self._metrics.misses += 1
                    return None
                    
                now = datetime.now()
                if now > entry.expiry:
                    self._metrics.misses += 1
                    del self._cache[key]
                    return None
                    
                entry.access_count += 1
                entry.last_access = now
                self._metrics.hits += 1
                return entry.data
        finally:
            duration = time.time() - start_time
            self._metrics.record_operation('get', duration)
            
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None
    ) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live override
        """
        start_time = time.time()
        try:
            size = self._estimate_size(value)
            if size > self._max_size:
                logger.warning(
                    f"Value for key {key} exceeds max cache size "
                    f"({size} > {self._max_size})"
                )
                return
                
            now = datetime.now()
            ttl = ttl or self._default_ttl
            entry = CacheEntry(
                data=value,
                timestamp=now,
                expiry=now + ttl
            )
            
            async with self._lock:
                # Check if we need to make space
                while (
                    self._metrics.memory_usage + size > self._max_size and
                    self._cache
                ):
                    # Evict least recently accessed entry
                    lra_key = min(
                        self._cache.keys(),
                        key=lambda k: (
                            self._cache[k].last_access or 
                            self._cache[k].timestamp
                        )
                    )
                    evicted = self._cache.pop(lra_key)
                    self._metrics.memory_usage -= self._estimate_size(evicted.data)
                    self._metrics.evictions += 1
                    
                self._cache[key] = entry
                self._metrics.memory_usage += size
        finally:
            duration = time.time() - start_time
            self._metrics.record_operation('set', duration)
            
    async def invalidate(self, key: str) -> None:
        """Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        start_time = time.time()
        try:
            async with self._lock:
                if key in self._cache:
                    entry = self._cache.pop(key)
                    self._metrics.memory_usage -= self._estimate_size(entry.data)
        finally:
            duration = time.time() - start_time
            self._metrics.record_operation('invalidate', duration)
            
    async def clear(self) -> None:
        """Clear all cache entries."""
        start_time = time.time()
        try:
            async with self._lock:
                self._cache.clear()
                self._metrics = CacheMetrics()
        finally:
            duration = time.time() - start_time
            self._metrics.record_operation('clear', duration)
            
    async def _cleanup_loop(self):
        """Periodically clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval.total_seconds())
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                
    async def _cleanup(self):
        """Remove expired entries."""
        start_time = time.time()
        try:
            now = datetime.now()
            async with self._lock:
                expired = [
                    key for key, entry in self._cache.items()
                    if now > entry.expiry
                ]
                for key in expired:
                    entry = self._cache.pop(key)
                    self._metrics.memory_usage -= self._estimate_size(entry.data)
                    self._metrics.evictions += 1
        finally:
            duration = time.time() - start_time
            self._metrics.record_operation('cleanup', duration)
