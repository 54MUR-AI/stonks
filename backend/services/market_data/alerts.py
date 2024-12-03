from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import asyncio
import logging

from .metrics import MetricsCollector
from .health import HealthMonitor
from .provider_manager import ProviderManager
from .provider_thresholds import ThresholdManager
from .alert_analytics import AlertAnalytics
from .escalation import EscalationManager
from ..notifications.notification_manager import NotificationManager

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    HIGH_LATENCY = "high_latency"
    ERROR_RATE = "error_rate"
    LOW_HEALTH = "low_health"
    CACHE_EFFICIENCY = "cache_efficiency"
    PROVIDER_DISCONNECT = "provider_disconnect"
    RATE_LIMIT = "rate_limit"
    DATA_QUALITY = "data_quality"

@dataclass
class AlertThresholds:
    # Latency thresholds (ms)
    latency_warning: float = 500
    latency_error: float = 1000
    latency_critical: float = 2000
    
    # Error rate thresholds (errors/minute)
    error_rate_warning: float = 5
    error_rate_error: float = 10
    error_rate_critical: float = 20
    
    # Health score thresholds (percentage)
    health_warning: float = 80
    health_error: float = 60
    health_critical: float = 40
    
    # Cache efficiency thresholds (percentage)
    cache_hit_rate_warning: float = 70
    cache_hit_rate_error: float = 50

@dataclass
class Alert:
    id: str
    provider_id: str
    type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_value: float
    threshold_value: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict = None

class AlertManager:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor()
        self.provider_manager = ProviderManager()
        self.threshold_manager = ThresholdManager()
        self.notification_manager = NotificationManager()
        self.alert_analytics = AlertAnalytics()
        self.escalation_manager = EscalationManager()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self._alert_handlers = []
        self._check_task: Optional[asyncio.Task] = None

    def add_alert_handler(self, handler):
        """Add a handler function to be called when new alerts are generated"""
        self._alert_handlers.append(handler)

    async def start(self):
        """Start the alert monitoring system"""
        await self.notification_manager.start()
        await self.alert_analytics.start()
        await self.escalation_manager.start()
        if not self._check_task or self._check_task.done():
            self._check_task = asyncio.create_task(self._periodic_check())
            logger.info("Alert monitoring system started")

    async def stop(self):
        """Stop the alert monitoring system"""
        await self.notification_manager.stop()
        await self.alert_analytics.stop()
        await self.escalation_manager.stop()
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
            logger.info("Alert monitoring system stopped")

    def get_active_alerts(self, provider_id: Optional[str] = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by provider"""
        if provider_id:
            return [alert for alert in self.active_alerts.values() 
                   if alert.provider_id == provider_id]
        return list(self.active_alerts.values())

    def get_alert_history(self, provider_id: Optional[str] = None,
                         alert_type: Optional[AlertType] = None,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alert history with optional filters"""
        alerts = self.alert_history
        if provider_id:
            alerts = [a for a in alerts if a.provider_id == provider_id]
        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    async def _periodic_check(self):
        """Periodically check metrics and generate alerts"""
        while True:
            try:
                await self._check_all_providers()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def _check_all_providers(self):
        """Check all providers for potential alerts"""
        for provider_id in self.provider_manager.get_providers():
            metrics = self.metrics_collector.get_provider_metrics(provider_id)
            health = self.health_monitor.get_provider_health(provider_id)

            # Check latency
            await self._check_latency(provider_id, metrics.latency.average)
            
            # Check error rate
            await self._check_error_rate(provider_id, metrics.errors.rate)
            
            # Check health score
            await self._check_health(provider_id, health.score)
            
            # Check cache efficiency
            await self._check_cache(provider_id, metrics.cache.hit_rate)

    async def _check_latency(self, provider_id: str, latency: float):
        """Check latency metrics and generate alerts if needed"""
        thresholds = self.threshold_manager.get_provider_thresholds(provider_id).latency

        if latency >= thresholds.critical:
            await self._create_alert(
                provider_id, AlertType.HIGH_LATENCY, AlertSeverity.CRITICAL,
                f"Critical latency of {latency}ms exceeds {thresholds.critical}ms",
                latency, thresholds.critical
            )
        elif latency >= thresholds.error:
            await self._create_alert(
                provider_id, AlertType.HIGH_LATENCY, AlertSeverity.ERROR,
                f"High latency of {latency}ms exceeds {thresholds.error}ms",
                latency, thresholds.error
            )
        elif latency >= thresholds.warning:
            await self._create_alert(
                provider_id, AlertType.HIGH_LATENCY, AlertSeverity.WARNING,
                f"Latency of {latency}ms exceeds {thresholds.warning}ms",
                latency, thresholds.warning
            )
        else:
            await self._resolve_alert(provider_id, AlertType.HIGH_LATENCY)

    async def _check_error_rate(self, provider_id: str, error_rate: float):
        """Check error rate metrics and generate alerts if needed"""
        thresholds = self.threshold_manager.get_provider_thresholds(provider_id).error_rate

        if error_rate >= thresholds.critical:
            await self._create_alert(
                provider_id, AlertType.ERROR_RATE, AlertSeverity.CRITICAL,
                f"Critical error rate of {error_rate}/min exceeds {thresholds.critical}/min",
                error_rate, thresholds.critical
            )
        elif error_rate >= thresholds.error:
            await self._create_alert(
                provider_id, AlertType.ERROR_RATE, AlertSeverity.ERROR,
                f"High error rate of {error_rate}/min exceeds {thresholds.error}/min",
                error_rate, thresholds.error
            )
        elif error_rate >= thresholds.warning:
            await self._create_alert(
                provider_id, AlertType.ERROR_RATE, AlertSeverity.WARNING,
                f"Error rate of {error_rate}/min exceeds {thresholds.warning}/min",
                error_rate, thresholds.warning
            )
        else:
            await self._resolve_alert(provider_id, AlertType.ERROR_RATE)

    async def _check_health(self, provider_id: str, health_score: float):
        """Check health score and generate alerts if needed"""
        thresholds = self.threshold_manager.get_provider_thresholds(provider_id).health

        if health_score <= thresholds.critical:
            await self._create_alert(
                provider_id, AlertType.LOW_HEALTH, AlertSeverity.CRITICAL,
                f"Critical health score of {health_score}% below {thresholds.critical}%",
                health_score, thresholds.critical
            )
        elif health_score <= thresholds.error:
            await self._create_alert(
                provider_id, AlertType.LOW_HEALTH, AlertSeverity.ERROR,
                f"Low health score of {health_score}% below {thresholds.error}%",
                health_score, thresholds.error
            )
        elif health_score <= thresholds.warning:
            await self._create_alert(
                provider_id, AlertType.LOW_HEALTH, AlertSeverity.WARNING,
                f"Health score of {health_score}% below {thresholds.warning}%",
                health_score, thresholds.warning
            )
        else:
            await self._resolve_alert(provider_id, AlertType.LOW_HEALTH)

    async def _check_cache(self, provider_id: str, hit_rate: float):
        """Check cache efficiency and generate alerts if needed"""
        thresholds = self.threshold_manager.get_provider_thresholds(provider_id).cache

        if hit_rate <= thresholds.error:
            await self._create_alert(
                provider_id, AlertType.CACHE_EFFICIENCY, AlertSeverity.ERROR,
                f"Poor cache hit rate of {hit_rate}% below {thresholds.error}%",
                hit_rate, thresholds.error
            )
        elif hit_rate <= thresholds.warning:
            await self._create_alert(
                provider_id, AlertType.CACHE_EFFICIENCY, AlertSeverity.WARNING,
                f"Low cache hit rate of {hit_rate}% below {thresholds.warning}%",
                hit_rate, thresholds.warning
            )
        else:
            await self._resolve_alert(provider_id, AlertType.CACHE_EFFICIENCY)

    async def _create_alert(self, provider_id: str, alert_type: AlertType,
                          severity: AlertSeverity, message: str,
                          metric_value: float, threshold_value: float):
        """Create a new alert if one doesn't exist for this provider/type"""
        alert_key = f"{provider_id}:{alert_type.value}"
        if alert_key not in self.active_alerts:
            alert = Alert(
                id=f"{alert_key}:{datetime.now().timestamp()}",
                provider_id=provider_id,
                type=alert_type,
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                metric_value=metric_value,
                threshold_value=threshold_value
            )
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            # Add alert to analytics
            self.alert_analytics.add_alert(alert)
            
            # Check for predicted anomalies
            predictions = await self.alert_analytics.predict_anomalies()
            for prediction in predictions:
                await self.notification_manager.notify(
                    subject=f"[PREDICTION] Potential {prediction.alert_type.value}",
                    message=f"Potential issue predicted for provider {prediction.provider_id}",
                    metadata={
                        "provider_id": prediction.provider_id,
                        "alert_type": prediction.alert_type.value,
                        "probability": prediction.probability,
                        "predicted_value": prediction.predicted_value,
                        "prediction_time": prediction.prediction_time.isoformat(),
                        "features": prediction.features
                    }
                )
            
            # Handle escalation
            await self.escalation_manager.handle_alert(alert)
            
            # Send notification
            await self.notification_manager.notify(
                subject=f"[{severity.value.upper()}] {alert_type.value} Alert",
                message=message,
                metadata={
                    "provider_id": provider_id,
                    "alert_type": alert_type.value,
                    "severity": severity.value,
                    "metric_value": metric_value,
                    "threshold_value": threshold_value,
                    "timestamp": alert.timestamp.isoformat()
                }
            )
            
            # Notify handlers
            for handler in self._alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")

    async def _resolve_alert(self, provider_id: str, alert_type: AlertType):
        """Resolve an active alert if it exists"""
        alert_key = f"{provider_id}:{alert_type.value}"
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            # Resolve any active escalations
            escalation_id = f"{alert.id}:{alert_type.value}"
            await self.escalation_manager.resolve_escalation(escalation_id)
            
            # Send resolution notification
            await self.notification_manager.notify(
                subject=f"[RESOLVED] {alert_type.value} Alert",
                message=f"Alert resolved for provider {provider_id}",
                metadata={
                    "provider_id": provider_id,
                    "alert_type": alert_type.value,
                    "resolved_at": alert.resolved_at.isoformat(),
                    "duration": (alert.resolved_at - alert.timestamp).total_seconds()
                }
            )
            
            del self.active_alerts[alert_key]

    async def get_alert_patterns(self):
        """Get current alert patterns"""
        return list(self.alert_analytics.alert_patterns.values())

    async def get_predicted_anomalies(self):
        """Get current anomaly predictions"""
        return await self.alert_analytics.predict_anomalies()
