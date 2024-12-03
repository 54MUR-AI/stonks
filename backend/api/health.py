"""Health monitoring API endpoints."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Set
import asyncio
import json
import logging
from datetime import datetime

from ..services.market_data.health import (
    ProviderHealthMonitor,
    HealthStatus,
    HealthMetricType
)

logger = logging.getLogger(__name__)

router = APIRouter()
health_monitor = ProviderHealthMonitor()
websocket_clients: Set[WebSocket] = set()

async def broadcast_health_updates():
    """Broadcast health updates to all connected clients."""
    while True:
        try:
            health_data = health_monitor.get_all_health()
            message = {
                'type': 'health_update',
                'timestamp': datetime.now().isoformat(),
                'data': health_data
            }
            
            # Broadcast to all clients
            disconnected = set()
            for client in websocket_clients:
                try:
                    await client.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    disconnected.add(client)
                    
            # Remove disconnected clients
            websocket_clients.difference_update(disconnected)
            
            await asyncio.sleep(1)  # Update every second
        except Exception as e:
            logger.error(f"Error in health broadcast: {e}")
            await asyncio.sleep(1)

@router.on_event("startup")
async def startup_event():
    """Start health monitoring on startup."""
    await health_monitor.start()
    asyncio.create_task(broadcast_health_updates())

@router.on_event("shutdown")
async def shutdown_event():
    """Stop health monitoring on shutdown."""
    await health_monitor.stop()

@router.get("/health/providers")
async def get_provider_health() -> Dict[str, Any]:
    """Get health status for all providers."""
    return health_monitor.get_all_health()

@router.get("/health/providers/{provider_id}")
async def get_specific_provider_health(provider_id: str) -> Dict[str, Any]:
    """Get health status for specific provider."""
    provider = health_monitor.get_provider_health(provider_id)
    if not provider:
        return {"error": "Provider not found"}
    return provider.get_metrics()

@router.websocket("/health/ws")
async def health_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time health updates."""
    await websocket.accept()
    websocket_clients.add(websocket)
    
    try:
        # Send initial health data
        health_data = health_monitor.get_all_health()
        await websocket.send_json({
            'type': 'health_update',
            'timestamp': datetime.now().isoformat(),
            'data': health_data
        })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_json()
                # Handle client messages if needed
                pass
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling websocket message: {e}")
                break
    finally:
        websocket_clients.remove(websocket)
