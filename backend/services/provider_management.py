from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics

from ..models.market_data import MarketDataProvider, ProviderStatus, ProviderMetrics
from ..exceptions import ProviderError, NoAvailableProviderError

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ProviderHealth:
    success_rate: float
    avg_latency: float
    error_count: int
    last_error: Optional[datetime]
    last_success: Optional[datetime]
    consecutive_failures: int

class MarketDataProviderManager:
    def __init__(self, primary_provider: MarketDataProvider, backup_providers: List[MarketDataProvider]):
        self.primary_provider = primary_provider
        self.backup_providers = backup_providers
        self.current_provider = primary_provider
        self.provider_metrics: Dict[str, ProviderHealth] = {}
        self._lock = asyncio.Lock()
        self._health_monitor_task = None
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize health metrics for all providers"""
        all_providers = [self.primary_provider] + self.backup_providers
        for provider in all_providers:
            self.provider_metrics[provider.name] = ProviderHealth(
                success_rate=1.0,
                avg_latency=0.0,
                error_count=0,
                last_error=None,
                last_success=None,
                consecutive_failures=0
            )

    async def start(self):
        """Start the provider manager and health monitoring"""
        self._health_monitor_task = asyncio.create_task(self._monitor_provider_health())
        await self.current_provider.connect()

    async def stop(self):
        """Stop the provider manager and cleanup"""
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        await self.current_provider.disconnect()

    async def _monitor_provider_health(self):
        """Continuously monitor provider health metrics"""
        while True:
            try:
                for provider in [self.primary_provider] + self.backup_providers:
                    metrics = self.provider_metrics[provider.name]
                    
                    # Check if provider needs recovery
                    if (metrics.consecutive_failures >= 3 and 
                        metrics.last_error and 
                        datetime.now() - metrics.last_error > timedelta(minutes=5)):
                        try:
                            await provider.reconnect()
                            logger.info(f"Successfully reconnected to provider {provider.name}")
                        except Exception as e:
                            logger.error(f"Failed to reconnect to provider {provider.name}: {e}")

                    # Update provider status
                    if metrics.consecutive_failures >= 5:
                        provider.status = ProviderStatus.UNHEALTHY
                    elif metrics.consecutive_failures >= 3:
                        provider.status = ProviderStatus.DEGRADED
                    else:
                        provider.status = ProviderStatus.HEALTHY

                await asyncio.sleep(60)  # Check health every minute
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def get_data(self, symbol: str, *args, **kwargs):
        """Get market data with automatic failover"""
        try:
            return await self._execute_with_metrics(
                self.current_provider.get_data, symbol, *args, **kwargs
            )
        except Exception as e:
            return await self._handle_provider_error(e, symbol, *args, **kwargs)

    async def _execute_with_metrics(self, func, *args, **kwargs):
        """Execute provider function while tracking metrics"""
        start_time = datetime.now()
        provider = self.current_provider
        
        try:
            result = await func(*args, **kwargs)
            
            async with self._lock:
                metrics = self.provider_metrics[provider.name]
                latency = (datetime.now() - start_time).total_seconds()
                metrics.avg_latency = (metrics.avg_latency + latency) / 2
                metrics.last_success = datetime.now()
                metrics.consecutive_failures = 0
                metrics.success_rate = (metrics.success_rate * 9 + 1) / 10  # Rolling average
            
            return result

        except Exception as e:
            async with self._lock:
                metrics = self.provider_metrics[provider.name]
                metrics.error_count += 1
                metrics.last_error = datetime.now()
                metrics.consecutive_failures += 1
                metrics.success_rate = (metrics.success_rate * 9) / 10  # Rolling average
            raise

    async def _handle_provider_error(self, error: Exception, *args, **kwargs):
        """Handle provider errors and implement failover strategy"""
        severity = self._classify_error(error)
        
        if severity >= ErrorSeverity.HIGH:
            await self._switch_provider(error)
            
            # Retry with new provider
            try:
                return await self._execute_with_metrics(
                    self.current_provider.get_data, *args, **kwargs
                )
            except Exception as e:
                logger.error(f"Backup provider also failed: {e}")
                raise NoAvailableProviderError("All providers failed") from e
        else:
            # For lower severity errors, retry with exponential backoff
            for attempt in range(3):
                await asyncio.sleep(2 ** attempt)
                try:
                    return await self._execute_with_metrics(
                        self.current_provider.get_data, *args, **kwargs
                    )
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        raise

    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify the severity of a provider error"""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, ProviderError):
            if "rate limit" in str(error).lower():
                return ErrorSeverity.MEDIUM
            return ErrorSeverity.HIGH
        return ErrorSeverity.LOW

    async def _switch_provider(self, error: Exception):
        """Switch to a backup provider"""
        async with self._lock:
            current_idx = ([self.primary_provider] + self.backup_providers).index(self.current_provider)
            available_providers = [p for p in self.backup_providers + [self.primary_provider] 
                                if p != self.current_provider and 
                                self.provider_metrics[p.name].consecutive_failures < 3]

            if not available_providers:
                raise NoAvailableProviderError("No healthy providers available")

            # Select best available provider based on health metrics
            best_provider = max(available_providers, 
                              key=lambda p: (self.provider_metrics[p.name].success_rate,
                                           -self.provider_metrics[p.name].avg_latency))

            logger.info(f"Switching from {self.current_provider.name} to {best_provider.name}")
            
            # Connect to new provider before disconnecting from old one
            try:
                await best_provider.connect()
                old_provider = self.current_provider
                self.current_provider = best_provider
                await old_provider.disconnect()
            except Exception as e:
                logger.error(f"Error switching to provider {best_provider.name}: {e}")
                raise

    async def get_provider_metrics(self) -> Dict[str, ProviderMetrics]:
        """Get current metrics for all providers"""
        metrics = {}
        for provider in [self.primary_provider] + self.backup_providers:
            health = self.provider_metrics[provider.name]
            metrics[provider.name] = ProviderMetrics(
                status=provider.status,
                success_rate=health.success_rate,
                avg_latency=health.avg_latency,
                error_count=health.error_count,
                last_error=health.last_error,
                last_success=health.last_success
            )
        return metrics
