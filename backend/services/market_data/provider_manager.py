"""Provider management and automatic failover system."""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import heapq

from .base import MarketDataProvider, MarketDataConfig
from .health import (
    ProviderHealth,
    ProviderHealthMonitor,
    HealthStatus,
    HealthMetricType
)
from .metrics import MetricsTracker
from .cache import Cache

logger = logging.getLogger(__name__)

class ProviderPriority(Enum):
    """Provider priority levels."""
    PRIMARY = 1
    SECONDARY = 2
    FALLBACK = 3

class ProviderState:
    """Provider state tracking."""
    def __init__(
        self,
        provider: MarketDataProvider,
        priority: ProviderPriority,
        health: ProviderHealth
    ):
        self.provider = provider
        self.priority = priority
        self.health = health
        self.active_symbols: Set[str] = set()
        self.last_switch_time = datetime.now()
        self.consecutive_failures = 0
        self.is_active = False

    def get_score(self) -> float:
        """Calculate provider score for selection."""
        # Base score from priority
        base_score = 100 - (self.priority.value * 20)
        
        # Health impact
        health_impact = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0.1
        }.get(self.health.overall_status, 0.1)
        
        # Performance metrics
        metrics = self.health.get_metrics()
        latency_score = max(0, 1 - (metrics['latency']['current_value'] / 1000))
        error_score = max(0, 1 - metrics['error_rate']['current_value'])
        
        # Combine scores
        return base_score * health_impact * latency_score * error_score

class ProviderManager:
    """Manages market data providers and handles automatic failover."""
    
    def __init__(
        self,
        config: MarketDataConfig,
        health_monitor: ProviderHealthMonitor,
        cache: Optional[Cache] = None
    ):
        self.config = config
        self.health_monitor = health_monitor
        self.cache = cache
        self.providers: Dict[str, ProviderState] = {}
        self.active_provider: Optional[ProviderState] = None
        self._lock = asyncio.Lock()
        self._check_task: Optional[asyncio.Task] = None
        
        # Failover settings
        self.min_switch_interval = 30  # seconds
        self.max_consecutive_failures = 3
        self.degraded_threshold = 0.7
        self.unhealthy_threshold = 0.3

    async def add_provider(
        self,
        provider_id: str,
        provider: MarketDataProvider,
        priority: ProviderPriority
    ) -> None:
        """Add a new provider to the manager."""
        async with self._lock:
            health = self.health_monitor.register_provider(provider_id)
            state = ProviderState(provider, priority, health)
            self.providers[provider_id] = state
            
            # Set as active if no active provider or higher priority
            if (not self.active_provider or
                priority.value < self.active_provider.priority.value):
                await self._switch_provider(state)

    async def remove_provider(self, provider_id: str) -> None:
        """Remove a provider from the manager."""
        async with self._lock:
            if provider_id in self.providers:
                state = self.providers[provider_id]
                if state.is_active:
                    await self._find_and_switch_provider()
                await state.provider.disconnect()
                del self.providers[provider_id]

    async def start(self) -> None:
        """Start provider management and monitoring."""
        await self.health_monitor.start()
        self._check_task = asyncio.create_task(self._monitor_providers())
        logger.info("Provider manager started")

    async def stop(self) -> None:
        """Stop provider management and monitoring."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        await self.health_monitor.stop()
        logger.info("Provider manager stopped")

    async def get_quote(self, symbol: str) -> float:
        """Get quote using the active provider with failover."""
        if not self.active_provider:
            raise RuntimeError("No active provider available")
            
        try:
            # Try cache first if available
            if self.cache:
                cached_quote = await self.cache.get(f"quote:{symbol}")
                if cached_quote is not None:
                    return cached_quote
            
            # Get quote from active provider
            quote = await self.active_provider.provider.get_quote(symbol)
            
            # Cache the result
            if self.cache:
                await self.cache.set(f"quote:{symbol}", quote)
                
            self.active_provider.consecutive_failures = 0
            return quote
            
        except Exception as e:
            logger.error(f"Error getting quote from {self.active_provider}: {e}")
            self.active_provider.consecutive_failures += 1
            
            if self.active_provider.consecutive_failures >= self.max_consecutive_failures:
                await self._handle_provider_failure()
                
            # Retry with new active provider
            if self.active_provider:
                return await self.get_quote(symbol)
            raise

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols using the active provider."""
        if not self.active_provider:
            raise RuntimeError("No active provider available")
            
        try:
            await self.active_provider.provider.subscribe(symbols)
            self.active_provider.active_symbols.update(symbols)
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
            await self._handle_provider_failure()
            # Retry with new active provider
            if self.active_provider:
                await self.subscribe(symbols)

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols using the active provider."""
        if not self.active_provider:
            return
            
        try:
            await self.active_provider.provider.unsubscribe(symbols)
            self.active_provider.active_symbols.difference_update(symbols)
        except Exception as e:
            logger.error(f"Error unsubscribing from symbols: {e}")

    async def _monitor_providers(self) -> None:
        """Monitor providers and trigger failover if needed."""
        while True:
            try:
                await self._check_provider_health()
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in provider monitoring: {e}")
                await asyncio.sleep(1)

    async def _check_provider_health(self) -> None:
        """Check provider health and trigger failover if needed."""
        if not self.active_provider:
            await self._find_and_switch_provider()
            return
            
        active_score = self.active_provider.get_score()
        
        # Check if active provider is healthy
        if active_score < self.unhealthy_threshold:
            logger.warning(f"Active provider unhealthy (score: {active_score})")
            await self._handle_provider_failure()
        elif active_score < self.degraded_threshold:
            # Look for better provider if current is degraded
            await self._find_and_switch_provider()

    async def _handle_provider_failure(self) -> None:
        """Handle active provider failure."""
        logger.error(f"Active provider failed: {self.active_provider}")
        await self._find_and_switch_provider(exclude_current=True)

    async def _find_and_switch_provider(self, exclude_current: bool = False) -> None:
        """Find the best available provider and switch to it."""
        async with self._lock:
            # Get all providers sorted by score
            candidates = []
            current_time = datetime.now()
            
            for state in self.providers.values():
                if exclude_current and state == self.active_provider:
                    continue
                    
                # Check switch cooldown
                if (current_time - state.last_switch_time).total_seconds() < self.min_switch_interval:
                    continue
                    
                score = state.get_score()
                if score > 0:  # Only consider providers with positive scores
                    heapq.heappush(candidates, (-score, state))  # Negative for max-heap
            
            # Switch to best candidate if better than current
            if candidates:
                best_score, best_state = heapq.heappop(candidates)
                best_score = -best_score  # Convert back to positive
                
                if not self.active_provider or best_score > self.active_provider.get_score():
                    await self._switch_provider(best_state)

    async def _switch_provider(self, new_state: ProviderState) -> None:
        """Switch to a new provider."""
        old_state = self.active_provider
        
        try:
            # Connect new provider if needed
            if not new_state.provider.is_connected:
                await new_state.provider.connect()
            
            # Subscribe to active symbols
            if old_state and old_state.active_symbols:
                await new_state.provider.subscribe(list(old_state.active_symbols))
                new_state.active_symbols.update(old_state.active_symbols)
            
            # Update state
            new_state.is_active = True
            new_state.last_switch_time = datetime.now()
            self.active_provider = new_state
            
            # Cleanup old provider
            if old_state:
                old_state.is_active = False
                old_state.active_symbols.clear()
                try:
                    await old_state.provider.disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting old provider: {e}")
            
            logger.info(f"Switched to provider: {new_state}")
            
        except Exception as e:
            logger.error(f"Error switching provider: {e}")
            new_state.is_active = False
            # Try to find another provider
            await self._find_and_switch_provider(exclude_current=True)
