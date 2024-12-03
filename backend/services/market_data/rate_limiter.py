"""Rate limiting and quota management for market data providers."""

import asyncio
import time
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_second: int = 2
    requests_per_minute: int = 100
    requests_per_hour: int = 2000
    requests_per_day: int = 48000
    concurrent_requests: int = 5

class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket."""
        async with self._lock:
            now = time.time()
            # Add new tokens based on time passed
            time_passed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + time_passed * self.rate
            )
            self.last_update = now
            
            # Check if we have enough tokens
            if self.tokens < tokens:
                return False
            
            self.tokens -= tokens
            return True
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until tokens are available."""
        start_time = time.time()
        while True:
            if await self.acquire(tokens):
                return True
                
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            # Wait for tokens to refill
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(max(0.1, wait_time))

class SlidingWindowCounter:
    """Sliding window rate limiter."""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # window size in seconds
        self.max_requests = max_requests
        self.requests: Dict[int, int] = {}  # timestamp -> count
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Try to acquire a request slot."""
        async with self._lock:
            now = int(time.time())
            self._cleanup(now)
            
            # Count recent requests
            total_requests = sum(self.requests.values())
            
            if total_requests >= self.max_requests:
                return False
            
            # Add new request
            self.requests[now] = self.requests.get(now, 0) + 1
            return True
    
    def _cleanup(self, now: int) -> None:
        """Remove old entries."""
        cutoff = now - self.window_size
        self.requests = {
            ts: count
            for ts, count in self.requests.items()
            if ts > cutoff
        }

class RateLimiter:
    """Combined rate limiter with multiple time windows."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        
        # Token bucket for per-second rate limiting
        self.per_second = TokenBucket(
            rate=config.requests_per_second,
            capacity=config.requests_per_second
        )
        
        # Sliding windows for longer periods
        self.per_minute = SlidingWindowCounter(60, config.requests_per_minute)
        self.per_hour = SlidingWindowCounter(3600, config.requests_per_hour)
        self.per_day = SlidingWindowCounter(86400, config.requests_per_day)
        
        # Semaphore for concurrent requests
        self.concurrency_limiter = asyncio.Semaphore(config.concurrent_requests)
        
        # Stats tracking
        self.total_requests = 0
        self.total_throttled = 0
        self.last_reset = datetime.now()
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Attempt to acquire permission for a request."""
        try:
            # Check all time windows
            if not all([
                await self.per_second.acquire(),
                await self.per_minute.acquire(),
                await self.per_hour.acquire(),
                await self.per_day.acquire()
            ]):
                self.total_throttled += 1
                return False
            
            self.total_requests += 1
            return True
            
        except Exception as e:
            logger.error(f"Error in rate limiter: {str(e)}")
            return False
    
    async def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until a request can be made."""
        async with self.concurrency_limiter:
            start_time = time.time()
            while True:
                if await self.acquire():
                    return True
                
                if timeout is not None:
                    if time.time() - start_time >= timeout:
                        return False
                
                await asyncio.sleep(0.1)
    
    def get_stats(self) -> Dict[str, int]:
        """Get current rate limiting stats."""
        return {
            "total_requests": self.total_requests,
            "total_throttled": self.total_throttled,
            "success_rate": (
                (self.total_requests - self.total_throttled) / 
                max(1, self.total_requests)
            ) * 100
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.total_requests = 0
        self.total_throttled = 0
        self.last_reset = datetime.now()

class RateLimitedClient:
    """Base class for rate-limited API clients."""
    
    def __init__(self, rate_limit_config: RateLimitConfig):
        self.rate_limiter = RateLimiter(rate_limit_config)
    
    async def execute_with_rate_limit(
        self,
        func: callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ):
        """Execute a function with rate limiting."""
        if not await self.rate_limiter.wait(timeout):
            raise Exception("Rate limit exceeded")
        
        return await func(*args, **kwargs)
