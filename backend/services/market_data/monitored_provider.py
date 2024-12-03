"""Provider with health monitoring support."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import time
import pandas as pd

from .base import MarketDataProvider, MarketDataConfig
from .health import (
    ProviderHealth,
    HealthMetricType,
    HealthStatus,
    ProviderHealthMonitor
)

logger = logging.getLogger(__name__)

class MonitoredMarketDataProvider(MarketDataProvider):
    """Market data provider with health monitoring."""
    
    def __init__(
        self,
        provider: MarketDataProvider,
        config: MarketDataConfig,
        provider_id: str,
        health_monitor: ProviderHealthMonitor
    ):
        """Initialize monitored provider.
        
        Args:
            provider: Provider to monitor
            config: Provider configuration
            provider_id: Provider identifier
            health_monitor: Health monitoring system
        """
        super().__init__(config)
        self._provider = provider
        self._provider_id = provider_id
        self._monitor = health_monitor
        self._health = health_monitor.register_provider(provider_id)
        self._operation_times: Dict[str, List[float]] = {}
        
    @property
    def health_metrics(self) -> Dict[str, Any]:
        """Get provider health metrics."""
        return self._health.get_metrics()
        
    def _record_operation_time(self, operation: str, duration: float):
        """Record operation timing."""
        if operation not in self._operation_times:
            self._operation_times[operation] = []
            
        times = self._operation_times[operation]
        times.append(duration)
        
        # Keep last 100 measurements
        if len(times) > 100:
            times.pop(0)
            
        # Update latency metric
        self._health.update_metric(
            HealthMetricType.LATENCY,
            sum(times) / len(times)
        )
        
    def _record_operation_result(self, success: bool):
        """Record operation success/failure."""
        self._health.record_health_check(success)
        
        # Calculate and update error rate
        total = self._health._total_checks
        failed = self._health._failed_checks
        error_rate = failed / total if total > 0 else 0.0
        success_rate = 1.0 - error_rate
        
        self._health.update_metric(
            HealthMetricType.ERROR_RATE,
            error_rate
        )
        self._health.update_metric(
            HealthMetricType.SUCCESS_RATE,
            success_rate
        )
        
    async def _execute_monitored(self, operation: str, func, *args, **kwargs):
        """Execute function with monitoring."""
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            self._record_operation_time(operation, duration * 1000)  # ms
            self._record_operation_result(True)
            return result
        except Exception as e:
            duration = time.time() - start_time
            self._record_operation_time(operation, duration * 1000)
            self._record_operation_result(False)
            raise
            
    async def connect(self) -> None:
        """Connect to provider with monitoring."""
        return await self._execute_monitored(
            'connect',
            self._provider.connect
        )
        
    async def disconnect(self) -> None:
        """Disconnect from provider with monitoring."""
        return await self._execute_monitored(
            'disconnect',
            self._provider.disconnect
        )
        
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data stream with monitoring."""
        return await self._execute_monitored(
            'subscribe',
            self._provider.subscribe,
            symbols
        )
        
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data stream with monitoring."""
        return await self._execute_monitored(
            'unsubscribe',
            self._provider.unsubscribe,
            symbols
        )
        
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1min"
    ) -> pd.DataFrame:
        """Get historical market data with monitoring."""
        return await self._execute_monitored(
            'get_historical',
            self._provider.get_historical_data,
            symbol,
            start_date,
            end_date,
            interval
        )
        
    async def get_quote(self, symbol: str) -> float:
        """Get current quote with monitoring."""
        return await self._execute_monitored(
            'get_quote',
            self._provider.get_quote,
            symbol
        )
        
    def update_cache_metrics(self, cache_metrics: Dict[str, Any]):
        """Update cache-related health metrics."""
        if 'hit_rate' in cache_metrics:
            self._health.update_metric(
                HealthMetricType.CACHE_HIT_RATE,
                cache_metrics['hit_rate']
            )
            
        if 'memory_usage' in cache_metrics:
            # Convert to percentage of max
            memory_pct = cache_metrics['memory_usage'] / (100 * 1024 * 1024)
            self._health.update_metric(
                HealthMetricType.MEMORY_USAGE,
                memory_pct
            )
            
    def update_circuit_breaker_metrics(self, cb_metrics: Dict[str, Any]):
        """Update circuit breaker health metrics."""
        if 'failure_rate' in cb_metrics:
            self._health.update_metric(
                HealthMetricType.ERROR_RATE,
                cb_metrics['failure_rate']
            )
            
        if 'current_state' in cb_metrics:
            state = cb_metrics['current_state']
            # Convert state to numeric value for tracking
            state_value = {
                'CLOSED': 1.0,
                'HALF_OPEN': 0.5,
                'OPEN': 0.0
            }.get(state, 1.0)
            
            self._health.update_metric(
                HealthMetricType.CIRCUIT_STATE,
                state_value
            )
