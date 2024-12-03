"""Rate limit monitoring and health checks."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from enum import Enum

from .health import HealthStatus, HealthMetric, HealthMetricType
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class RateLimitThreshold(Enum):
    """Rate limit threshold levels."""
    NORMAL = 0.7  # 70% of limit
    WARNING = 0.85  # 85% of limit
    CRITICAL = 0.95  # 95% of limit

@dataclass
class RateLimitMetrics:
    """Rate limit metrics for a time window."""
    window_size: int  # seconds
    max_requests: int
    current_usage: int
    usage_percent: float
    remaining_requests: int
    reset_time: Optional[datetime]

class RateLimitMonitor:
    """Monitors rate limit usage and health."""

    def __init__(
        self,
        provider_id: str,
        rate_limiter: RateLimiter,
        check_interval: int = 60
    ):
        self.provider_id = provider_id
        self.rate_limiter = rate_limiter
        self.check_interval = check_interval
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_check = datetime.now()
        self._metrics: Dict[str, RateLimitMetrics] = {}
        self._alerts: List[str] = []
        
    async def start(self) -> None:
        """Start rate limit monitoring."""
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started rate limit monitoring for {self.provider_id}")

    async def stop(self) -> None:
        """Stop rate limit monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped rate limit monitoring for {self.provider_id}")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                await self._check_rate_limits()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limit monitoring: {str(e)}")
                await asyncio.sleep(self.check_interval)

    async def _check_rate_limits(self) -> None:
        """Check current rate limit usage."""
        now = datetime.now()
        stats = self.rate_limiter.get_stats()
        
        # Update metrics for each time window
        self._metrics = {
            "per_second": RateLimitMetrics(
                window_size=1,
                max_requests=self.rate_limiter.config.requests_per_second,
                current_usage=stats.get("requests_last_second", 0),
                usage_percent=self._calculate_usage(
                    stats.get("requests_last_second", 0),
                    self.rate_limiter.config.requests_per_second
                ),
                remaining_requests=self._calculate_remaining(
                    stats.get("requests_last_second", 0),
                    self.rate_limiter.config.requests_per_second
                ),
                reset_time=now + timedelta(seconds=1)
            ),
            "per_minute": RateLimitMetrics(
                window_size=60,
                max_requests=self.rate_limiter.config.requests_per_minute,
                current_usage=stats.get("requests_last_minute", 0),
                usage_percent=self._calculate_usage(
                    stats.get("requests_last_minute", 0),
                    self.rate_limiter.config.requests_per_minute
                ),
                remaining_requests=self._calculate_remaining(
                    stats.get("requests_last_minute", 0),
                    self.rate_limiter.config.requests_per_minute
                ),
                reset_time=now + timedelta(minutes=1)
            ),
            "per_hour": RateLimitMetrics(
                window_size=3600,
                max_requests=self.rate_limiter.config.requests_per_hour,
                current_usage=stats.get("requests_last_hour", 0),
                usage_percent=self._calculate_usage(
                    stats.get("requests_last_hour", 0),
                    self.rate_limiter.config.requests_per_hour
                ),
                remaining_requests=self._calculate_remaining(
                    stats.get("requests_last_hour", 0),
                    self.rate_limiter.config.requests_per_hour
                ),
                reset_time=now + timedelta(hours=1)
            ),
            "per_day": RateLimitMetrics(
                window_size=86400,
                max_requests=self.rate_limiter.config.requests_per_day,
                current_usage=stats.get("requests_last_day", 0),
                usage_percent=self._calculate_usage(
                    stats.get("requests_last_day", 0),
                    self.rate_limiter.config.requests_per_day
                ),
                remaining_requests=self._calculate_remaining(
                    stats.get("requests_last_day", 0),
                    self.rate_limiter.config.requests_per_day
                ),
                reset_time=now + timedelta(days=1)
            )
        }
        
        # Check thresholds and generate alerts
        self._check_thresholds()
        self._last_check = now

    def _calculate_usage(self, current: int, maximum: int) -> float:
        """Calculate usage percentage."""
        return (current / maximum) * 100 if maximum > 0 else 0

    def _calculate_remaining(self, current: int, maximum: int) -> int:
        """Calculate remaining requests."""
        return max(0, maximum - current)

    def _check_thresholds(self) -> None:
        """Check rate limit thresholds and generate alerts."""
        self._alerts.clear()
        
        for window, metrics in self._metrics.items():
            if metrics.usage_percent >= RateLimitThreshold.CRITICAL.value * 100:
                self._alerts.append(
                    f"CRITICAL: {window} rate limit at {metrics.usage_percent:.1f}%"
                )
            elif metrics.usage_percent >= RateLimitThreshold.WARNING.value * 100:
                self._alerts.append(
                    f"WARNING: {window} rate limit at {metrics.usage_percent:.1f}%"
                )

    def get_health_metrics(self) -> List[HealthMetric]:
        """Get current health metrics."""
        metrics = []
        
        # Overall rate limit health
        highest_usage = max(
            (m.usage_percent for m in self._metrics.values()),
            default=0
        )
        
        status = HealthStatus.HEALTHY
        if highest_usage >= RateLimitThreshold.CRITICAL.value * 100:
            status = HealthStatus.UNHEALTHY
        elif highest_usage >= RateLimitThreshold.WARNING.value * 100:
            status = HealthStatus.DEGRADED
            
        metrics.append(HealthMetric(
            name="rate_limit_health",
            type=HealthMetricType.STATUS,
            value=status.value,
            threshold=RateLimitThreshold.WARNING.value * 100,
            description="Overall rate limit health status"
        ))
        
        # Usage metrics for each window
        for window, m in self._metrics.items():
            metrics.append(HealthMetric(
                name=f"rate_limit_{window}",
                type=HealthMetricType.PERCENTAGE,
                value=m.usage_percent,
                threshold=RateLimitThreshold.WARNING.value * 100,
                description=f"Rate limit usage for {window}"
            ))
            
        return metrics

    def get_current_metrics(self) -> Dict[str, RateLimitMetrics]:
        """Get current rate limit metrics."""
        return self._metrics.copy()

    def get_alerts(self) -> List[str]:
        """Get current alerts."""
        return self._alerts.copy()

    @property
    def last_check_time(self) -> datetime:
        """Get last check time."""
        return self._last_check
