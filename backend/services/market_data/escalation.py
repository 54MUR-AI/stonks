"""
Alert escalation and automated response system.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Awaitable
import asyncio
import logging

from .alerts import Alert, AlertType, AlertSeverity
from ..notifications.notification_manager import NotificationManager

logger = logging.getLogger(__name__)

class EscalationLevel(Enum):
    L1 = "L1"  # First level support
    L2 = "L2"  # Technical specialists
    L3 = "L3"  # Senior engineers
    EMERGENCY = "EMERGENCY"  # Critical situation

@dataclass
class EscalationPolicy:
    """Defines how alerts should be escalated"""
    name: str
    conditions: Dict[str, any]  # Conditions that trigger this policy
    initial_level: EscalationLevel
    escalation_delay: timedelta  # Time before escalating to next level
    max_level: EscalationLevel
    notification_channels: Dict[EscalationLevel, List[str]]
    auto_actions: Dict[EscalationLevel, List[str]]

@dataclass
class EscalationEvent:
    """Represents an active escalation"""
    id: str
    alert: Alert
    policy: EscalationPolicy
    current_level: EscalationLevel
    start_time: datetime
    last_escalation: datetime
    actions_taken: List[str]
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class EscalationManager:
    def __init__(self):
        self.notification_manager = NotificationManager()
        self.active_escalations: Dict[str, EscalationEvent] = {}
        self.policies: Dict[str, EscalationPolicy] = {}
        self.action_handlers: Dict[str, Callable[[Alert, EscalationLevel], Awaitable[bool]]] = {}
        self._check_task: Optional[asyncio.Task] = None

        # Register default policies
        self._register_default_policies()
        # Register default actions
        self._register_default_actions()

    def _register_default_policies(self):
        """Register default escalation policies"""
        self.policies.update({
            "critical_latency": EscalationPolicy(
                name="Critical Latency",
                conditions={
                    "alert_type": AlertType.HIGH_LATENCY,
                    "severity": AlertSeverity.CRITICAL,
                    "duration_threshold": timedelta(minutes=5)
                },
                initial_level=EscalationLevel.L1,
                escalation_delay=timedelta(minutes=15),
                max_level=EscalationLevel.L3,
                notification_channels={
                    EscalationLevel.L1: ["slack", "email"],
                    EscalationLevel.L2: ["slack", "email", "sms"],
                    EscalationLevel.L3: ["slack", "email", "sms", "phone"]
                },
                auto_actions={
                    EscalationLevel.L1: ["retry_connection"],
                    EscalationLevel.L2: ["retry_connection", "failover"],
                    EscalationLevel.L3: ["retry_connection", "failover", "emergency_shutdown"]
                }
            ),
            "high_error_rate": EscalationPolicy(
                name="High Error Rate",
                conditions={
                    "alert_type": AlertType.ERROR_RATE,
                    "severity": AlertSeverity.ERROR,
                    "duration_threshold": timedelta(minutes=10)
                },
                initial_level=EscalationLevel.L1,
                escalation_delay=timedelta(minutes=20),
                max_level=EscalationLevel.L2,
                notification_channels={
                    EscalationLevel.L1: ["slack", "email"],
                    EscalationLevel.L2: ["slack", "email", "sms"]
                },
                auto_actions={
                    EscalationLevel.L1: ["clear_cache"],
                    EscalationLevel.L2: ["clear_cache", "restart_service"]
                }
            ),
            "emergency_health": EscalationPolicy(
                name="Emergency Health",
                conditions={
                    "alert_type": AlertType.LOW_HEALTH,
                    "severity": AlertSeverity.CRITICAL,
                    "duration_threshold": timedelta(minutes=1)
                },
                initial_level=EscalationLevel.L2,
                escalation_delay=timedelta(minutes=5),
                max_level=EscalationLevel.EMERGENCY,
                notification_channels={
                    EscalationLevel.L2: ["slack", "email", "sms"],
                    EscalationLevel.L3: ["slack", "email", "sms", "phone"],
                    EscalationLevel.EMERGENCY: ["slack", "email", "sms", "phone", "pager"]
                },
                auto_actions={
                    EscalationLevel.L2: ["health_check", "restart_service"],
                    EscalationLevel.L3: ["health_check", "failover"],
                    EscalationLevel.EMERGENCY: ["emergency_shutdown"]
                }
            )
        })

    async def start(self):
        """Start the escalation manager"""
        if not self._check_task or self._check_task.done():
            self._check_task = asyncio.create_task(self._periodic_check())
            logger.info("Escalation manager started")

    async def stop(self):
        """Stop the escalation manager"""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
            logger.info("Escalation manager stopped")

    def _register_default_actions(self):
        """Register default automated actions"""
        async def retry_connection(alert: Alert, level: EscalationLevel) -> bool:
            logger.info(f"Retrying connection for provider {alert.provider_id}")
            # Implement connection retry logic
            return True

        async def failover(alert: Alert, level: EscalationLevel) -> bool:
            logger.info(f"Initiating failover for provider {alert.provider_id}")
            # Implement failover logic
            return True

        async def clear_cache(alert: Alert, level: EscalationLevel) -> bool:
            logger.info(f"Clearing cache for provider {alert.provider_id}")
            # Implement cache clearing logic
            return True

        async def restart_service(alert: Alert, level: EscalationLevel) -> bool:
            logger.info(f"Restarting service for provider {alert.provider_id}")
            # Implement service restart logic
            return True

        async def emergency_shutdown(alert: Alert, level: EscalationLevel) -> bool:
            logger.warning(f"Emergency shutdown for provider {alert.provider_id}")
            # Implement emergency shutdown logic
            return True

        self.action_handlers.update({
            "retry_connection": retry_connection,
            "failover": failover,
            "clear_cache": clear_cache,
            "restart_service": restart_service,
            "emergency_shutdown": emergency_shutdown
        })

    def register_policy(self, policy_id: str, policy: EscalationPolicy):
        """Register a new escalation policy"""
        self.policies[policy_id] = policy
        logger.info(f"Registered escalation policy: {policy_id}")

    def register_action(self, action_id: str, handler: Callable[[Alert, EscalationLevel], Awaitable[bool]]):
        """Register a new automated action"""
        self.action_handlers[action_id] = handler
        logger.info(f"Registered action handler: {action_id}")

    async def handle_alert(self, alert: Alert):
        """Handle a new alert and start escalation if needed"""
        for policy_id, policy in self.policies.items():
            if self._should_escalate(alert, policy):
                await self._start_escalation(alert, policy)

    async def _start_escalation(self, alert: Alert, policy: EscalationPolicy):
        """Start a new escalation process"""
        event_id = f"{alert.id}:{policy.name}"
        if event_id not in self.active_escalations:
            event = EscalationEvent(
                id=event_id,
                alert=alert,
                policy=policy,
                current_level=policy.initial_level,
                start_time=datetime.now(),
                last_escalation=datetime.now(),
                actions_taken=[]
            )
            self.active_escalations[event_id] = event
            
            # Send initial notifications
            await self._send_notifications(event)
            
            # Take initial actions
            await self._take_actions(event)
            
            logger.info(f"Started escalation {event_id} at level {policy.initial_level}")

    def _should_escalate(self, alert: Alert, policy: EscalationPolicy) -> bool:
        """Check if an alert should trigger escalation"""
        conditions = policy.conditions
        
        if alert.type != conditions.get("alert_type"):
            return False
            
        if alert.severity != conditions.get("severity"):
            return False
            
        # Check duration threshold
        duration = datetime.now() - alert.timestamp
        if duration < conditions.get("duration_threshold", timedelta(0)):
            return False
            
        return True

    async def _periodic_check(self):
        """Periodically check active escalations"""
        while True:
            try:
                now = datetime.now()
                for event_id, event in list(self.active_escalations.items()):
                    if event.resolved:
                        continue
                        
                    time_since_last = now - event.last_escalation
                    if (time_since_last >= event.policy.escalation_delay and 
                        event.current_level != event.policy.max_level):
                        # Escalate to next level
                        await self._escalate(event)
                    
            except Exception as e:
                logger.error(f"Error in escalation check: {e}")
                
            await asyncio.sleep(60)  # Check every minute

    async def _escalate(self, event: EscalationEvent):
        """Escalate an event to the next level"""
        levels = list(EscalationLevel)
        current_idx = levels.index(event.current_level)
        next_idx = min(current_idx + 1, levels.index(event.policy.max_level))
        event.current_level = levels[next_idx]
        event.last_escalation = datetime.now()
        
        # Send notifications for new level
        await self._send_notifications(event)
        
        # Take actions for new level
        await self._take_actions(event)
        
        logger.info(f"Escalated {event.id} to level {event.current_level}")

    async def _send_notifications(self, event: EscalationEvent):
        """Send notifications for current escalation level"""
        channels = event.policy.notification_channels.get(event.current_level, [])
        for channel in channels:
            await self.notification_manager.notify(
                subject=f"[{event.current_level.value}] {event.policy.name} Escalation",
                message=f"Alert escalated to {event.current_level.value} for provider {event.alert.provider_id}",
                metadata={
                    "provider_id": event.alert.provider_id,
                    "alert_type": event.alert.type.value,
                    "severity": event.alert.severity.value,
                    "escalation_level": event.current_level.value,
                    "start_time": event.start_time.isoformat(),
                    "actions_taken": event.actions_taken
                },
                channel=channel
            )

    async def _take_actions(self, event: EscalationEvent):
        """Take automated actions for current escalation level"""
        actions = event.policy.auto_actions.get(event.current_level, [])
        for action in actions:
            if action in self.action_handlers:
                try:
                    handler = self.action_handlers[action]
                    success = await handler(event.alert, event.current_level)
                    if success:
                        event.actions_taken.append(f"{action} at {datetime.now().isoformat()}")
                        logger.info(f"Action {action} completed successfully for {event.id}")
                    else:
                        logger.error(f"Action {action} failed for {event.id}")
                except Exception as e:
                    logger.error(f"Error executing action {action}: {e}")

    async def resolve_escalation(self, event_id: str):
        """Resolve an active escalation"""
        if event_id in self.active_escalations:
            event = self.active_escalations[event_id]
            event.resolved = True
            event.resolution_time = datetime.now()
            
            # Send resolution notification
            await self.notification_manager.notify(
                subject=f"[RESOLVED] {event.policy.name} Escalation",
                message=f"Escalation resolved for provider {event.alert.provider_id}",
                metadata={
                    "provider_id": event.alert.provider_id,
                    "alert_type": event.alert.type.value,
                    "final_level": event.current_level.value,
                    "resolution_time": event.resolution_time.isoformat(),
                    "duration": (event.resolution_time - event.start_time).total_seconds(),
                    "actions_taken": event.actions_taken
                }
            )
            
            logger.info(f"Resolved escalation {event_id}")
