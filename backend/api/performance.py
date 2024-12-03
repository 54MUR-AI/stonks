from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from typing import Dict, List, Optional
import asyncio
from datetime import datetime, timedelta
import json

from ..services.market_data.metrics import MetricsCollector
from ..services.market_data.health import HealthMonitor
from ..services.market_data.provider_manager import ProviderManager
from ..services.market_data.alerts import AlertManager, AlertType, AlertSeverity

router = APIRouter(prefix="/performance", tags=["performance"])

class PerformanceMonitor:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor()
        self.provider_manager = ProviderManager()
        self.alert_manager = AlertManager()
        self.update_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the performance monitoring system"""
        await self.alert_manager.start()
        
        # Register alert handler
        self.alert_manager.add_alert_handler(self._handle_alert)

    async def stop(self):
        """Stop the performance monitoring system"""
        await self.alert_manager.stop()
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            self.update_task = None

    async def _handle_alert(self, alert):
        """Handle new alerts by broadcasting to connected clients"""
        await self.broadcast({
            "type": "alert",
            "data": alert.__dict__
        })

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if not self.update_task or self.update_task.done():
            self.update_task = asyncio.create_task(self._periodic_updates())

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
        if not self.active_connections and self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            self.update_task = None

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                await self.disconnect(connection)

    async def _periodic_updates(self):
        while True:
            try:
                performance_data = await self._gather_performance_data()
                await self.broadcast({
                    "type": "performance_update",
                    "data": performance_data,
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                print(f"Error in periodic updates: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def _gather_performance_data(self) -> Dict:
        providers = self.provider_manager.get_providers()
        performance_data = {}

        for provider_id, provider in providers.items():
            metrics = self.metrics_collector.get_provider_metrics(provider_id)
            health = self.health_monitor.get_provider_health(provider_id)
            
            performance_data[provider_id] = {
                "metrics": {
                    "health": {
                        "status": health.status,
                        "score": health.score,
                        "lastUpdate": health.last_update.isoformat()
                    },
                    "latency": {
                        "average": metrics.latency.average,
                        "p95": metrics.latency.p95,
                        "p99": metrics.latency.p99,
                        "trend": metrics.latency.get_trend(minutes=5)
                    },
                    "cacheEfficiency": {
                        "hitRate": metrics.cache.hit_rate,
                        "missRate": metrics.cache.miss_rate,
                        "evictionRate": metrics.cache.eviction_rate,
                        "size": metrics.cache.size
                    },
                    "errorRate": {
                        "total": metrics.errors.total,
                        "byType": metrics.errors.by_type,
                        "trend": metrics.errors.get_trend(minutes=5)
                    }
                }
            }

        return performance_data

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await performance_monitor.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        await performance_monitor.disconnect(websocket)

@router.get("/current")
async def get_current_performance():
    """Get current performance metrics for all providers"""
    return await performance_monitor._gather_performance_data()

@router.get("/{provider_id}/history")
async def get_performance_history(provider_id: str, minutes: int = 60):
    """Get historical performance data for a specific provider"""
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=minutes)
    
    metrics = performance_monitor.metrics_collector.get_provider_metrics(provider_id)
    health = performance_monitor.health_monitor.get_provider_health(provider_id)
    
    return {
        "latency": metrics.latency.get_history(start_time, end_time),
        "errorRate": metrics.errors.get_history(start_time, end_time),
        "cacheHitRate": metrics.cache.get_history(start_time, end_time),
        "healthScore": health.get_history(start_time, end_time)
    }

@router.get("/alerts")
async def get_alerts(
    provider_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    active_only: bool = Query(False, description="Only return active alerts")
):
    """Get alerts with optional filtering"""
    alert_type_enum = AlertType(alert_type) if alert_type else None
    severity_enum = AlertSeverity(severity) if severity else None
    
    if active_only:
        alerts = performance_monitor.alert_manager.get_active_alerts(provider_id)
    else:
        alerts = performance_monitor.alert_manager.get_alert_history(
            provider_id, alert_type_enum, severity_enum
        )
    
    return [alert.__dict__ for alert in alerts]

@router.get("/alerts/{alert_id}")
async def get_alert(alert_id: str):
    """Get specific alert by ID"""
    active_alerts = performance_monitor.alert_manager.get_active_alerts()
    for alert in active_alerts:
        if alert.id == alert_id:
            return alert.__dict__
    
    history = performance_monitor.alert_manager.get_alert_history()
    for alert in history:
        if alert.id == alert_id:
            return alert.__dict__
    
    raise HTTPException(status_code=404, detail="Alert not found")

# Initialize and start the performance monitor
performance_monitor = PerformanceMonitor()

@router.on_event("startup")
async def startup_event():
    await performance_monitor.start()

@router.on_event("shutdown")
async def shutdown_event():
    await performance_monitor.stop()
