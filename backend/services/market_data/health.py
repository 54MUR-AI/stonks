"""Provider health monitoring system."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
import logging
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthMetricType(Enum):
    """Types of health metrics."""
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    AVAILABILITY = "availability"
    CIRCUIT_STATE = "circuit_state"
    CACHE_HIT_RATE = "cache_hit_rate"
    MEMORY_USAGE = "memory_usage"

@dataclass
class HealthThresholds:
    """Thresholds for health metrics."""
    latency_warning_ms: float = 100.0
    latency_critical_ms: float = 500.0
    error_rate_warning: float = 0.05  # 5%
    error_rate_critical: float = 0.15  # 15%
    success_rate_warning: float = 0.95  # 95%
    success_rate_critical: float = 0.85  # 85%
    availability_warning: float = 0.98  # 98%
    availability_critical: float = 0.95  # 95%
    cache_hit_warning: float = 0.80  # 80%
    cache_hit_critical: float = 0.60  # 60%
    memory_warning: float = 0.80  # 80%
    memory_critical: float = 0.95  # 95%

@dataclass
class HealthMetric:
    """Single health metric with history."""
    type: HealthMetricType
    current_value: float
    timestamp: datetime
    status: HealthStatus
    history: List[float] = field(default_factory=list)
    max_history: int = 100
    
    def add_value(self, value: float):
        """Add value to history."""
        self.current_value = value
        self.history.append(value)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    @property
    def average(self) -> float:
        """Get average value."""
        return statistics.mean(self.history) if self.history else self.current_value
        
    @property
    def stddev(self) -> float:
        """Get standard deviation."""
        return statistics.stdev(self.history) if len(self.history) > 1 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.type.value,
            'current_value': self.current_value,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'average': self.average,
            'stddev': self.stddev,
            'history_size': len(self.history)
        }

class ProviderHealth:
    """Provider health monitor."""
    
    def __init__(
        self,
        provider_id: str,
        thresholds: Optional[HealthThresholds] = None,
        window_size: timedelta = timedelta(minutes=5)
    ):
        """Initialize health monitor.
        
        Args:
            provider_id: Provider identifier
            thresholds: Health thresholds
            window_size: Time window for metrics
        """
        self.provider_id = provider_id
        self.thresholds = thresholds or HealthThresholds()
        self.window_size = window_size
        self._metrics: Dict[HealthMetricType, HealthMetric] = {}
        self._start_time = datetime.now()
        self._last_check = self._start_time
        self._total_checks = 0
        self._failed_checks = 0
        
    def update_metric(
        self,
        metric_type: HealthMetricType,
        value: float
    ) -> HealthStatus:
        """Update health metric.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            
        Returns:
            Current health status
        """
        now = datetime.now()
        status = self._calculate_status(metric_type, value)
        
        if metric_type not in self._metrics:
            self._metrics[metric_type] = HealthMetric(
                type=metric_type,
                current_value=value,
                timestamp=now,
                status=status
            )
        else:
            metric = self._metrics[metric_type]
            metric.add_value(value)
            metric.timestamp = now
            metric.status = status
            
        return status
        
    def _calculate_status(
        self,
        metric_type: HealthMetricType,
        value: float
    ) -> HealthStatus:
        """Calculate health status for metric."""
        if metric_type == HealthMetricType.LATENCY:
            if value >= self.thresholds.latency_critical_ms:
                return HealthStatus.UNHEALTHY
            elif value >= self.thresholds.latency_warning_ms:
                return HealthStatus.DEGRADED
                
        elif metric_type == HealthMetricType.ERROR_RATE:
            if value >= self.thresholds.error_rate_critical:
                return HealthStatus.UNHEALTHY
            elif value >= self.thresholds.error_rate_warning:
                return HealthStatus.DEGRADED
                
        elif metric_type == HealthMetricType.SUCCESS_RATE:
            if value <= self.thresholds.success_rate_critical:
                return HealthStatus.UNHEALTHY
            elif value <= self.thresholds.success_rate_warning:
                return HealthStatus.DEGRADED
                
        elif metric_type == HealthMetricType.AVAILABILITY:
            if value <= self.thresholds.availability_critical:
                return HealthStatus.UNHEALTHY
            elif value <= self.thresholds.availability_warning:
                return HealthStatus.DEGRADED
                
        elif metric_type == HealthMetricType.CACHE_HIT_RATE:
            if value <= self.thresholds.cache_hit_critical:
                return HealthStatus.UNHEALTHY
            elif value <= self.thresholds.cache_hit_warning:
                return HealthStatus.DEGRADED
                
        elif metric_type == HealthMetricType.MEMORY_USAGE:
            if value >= self.thresholds.memory_critical:
                return HealthStatus.UNHEALTHY
            elif value >= self.thresholds.memory_warning:
                return HealthStatus.DEGRADED
                
        return HealthStatus.HEALTHY
        
    @property
    def overall_status(self) -> HealthStatus:
        """Calculate overall health status."""
        if not self._metrics:
            return HealthStatus.HEALTHY
            
        statuses = [m.status for m in self._metrics.values()]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all health metrics."""
        return {
            'provider_id': self.provider_id,
            'overall_status': self.overall_status.value,
            'uptime': (datetime.now() - self._start_time).total_seconds(),
            'metrics': {
                metric_type.value: metric.to_dict()
                for metric_type, metric in self._metrics.items()
            }
        }
        
    def record_health_check(self, success: bool):
        """Record health check result."""
        self._total_checks += 1
        if not success:
            self._failed_checks += 1
        self._last_check = datetime.now()
        
        # Update availability metric
        availability = (
            (self._total_checks - self._failed_checks) / 
            self._total_checks if self._total_checks > 0 else 1.0
        )
        self.update_metric(HealthMetricType.AVAILABILITY, availability)

class ProviderHealthMonitor:
    """System-wide provider health monitoring."""
    
    def __init__(
        self,
        check_interval: timedelta = timedelta(seconds=30),
        thresholds: Optional[HealthThresholds] = None
    ):
        """Initialize health monitor.
        
        Args:
            check_interval: Health check interval
            thresholds: Health thresholds
        """
        self._providers: Dict[str, ProviderHealth] = {}
        self._check_interval = check_interval
        self._thresholds = thresholds or HealthThresholds()
        self._monitor_task: Optional[asyncio.Task] = None
        
    def register_provider(self, provider_id: str) -> ProviderHealth:
        """Register provider for monitoring."""
        if provider_id not in self._providers:
            self._providers[provider_id] = ProviderHealth(
                provider_id,
                self._thresholds
            )
        return self._providers[provider_id]
        
    def get_provider_health(self, provider_id: str) -> Optional[ProviderHealth]:
        """Get provider health monitor."""
        return self._providers.get(provider_id)
        
    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all providers."""
        return {
            provider_id: provider.get_metrics()
            for provider_id, provider in self._providers.items()
        }
        
    async def start(self):
        """Start health monitoring."""
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
    async def stop(self):
        """Stop health monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            
    async def _monitor_loop(self):
        """Monitor providers periodically."""
        while True:
            try:
                await asyncio.sleep(self._check_interval.total_seconds())
                await self._check_all_providers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                
    async def _check_all_providers(self):
        """Check health of all providers."""
        for provider in self._providers.values():
            try:
                # Perform provider-specific health checks here
                # For now, just update timestamp
                provider._last_check = datetime.now()
            except Exception as e:
                logger.error(f"Error checking provider {provider.provider_id}: {e}")
                provider.record_health_check(False)
            else:
                provider.record_health_check(True)
