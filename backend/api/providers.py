"""Provider management API endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..services.market_data.provider_manager import (
    ProviderManager,
    ProviderPriority,
    ProviderState
)
from ..services.market_data.health import ProviderHealthMonitor
from ..services.market_data.base import MarketDataConfig
from ..services.market_data.mock_provider import MockMarketDataProvider
# Import other provider types as needed

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize managers
health_monitor = ProviderHealthMonitor()
provider_manager = ProviderManager(
    config=MarketDataConfig(
        credentials={},  # Add default credentials
        base_url="",
        websocket_url=""
    ),
    health_monitor=health_monitor
)

PROVIDER_TYPES = {
    "MOCK": MockMarketDataProvider,
    # Add other provider types
}

@router.get("/providers")
async def get_providers() -> List[Dict]:
    """Get all registered providers."""
    providers = []
    for provider_id, state in provider_manager.providers.items():
        providers.append({
            "id": provider_id,
            "name": state.provider.__class__.__name__,
            "type": state.provider.__class__.__name__.upper(),
            "priority": state.priority.name,
            "status": "ACTIVE" if state.is_active else "INACTIVE",
            "health": state.health.get_metrics(),
            "config": state.provider.config.dict()
        })
    return providers

@router.get("/providers/{provider_id}")
async def get_provider(provider_id: str) -> Dict:
    """Get specific provider details."""
    state = provider_manager.providers.get(provider_id)
    if not state:
        raise HTTPException(status_code=404, detail="Provider not found")
        
    return {
        "id": provider_id,
        "name": state.provider.__class__.__name__,
        "type": state.provider.__class__.__name__.upper(),
        "priority": state.priority.name,
        "status": "ACTIVE" if state.is_active else "INACTIVE",
        "health": state.health.get_metrics(),
        "config": state.provider.config.dict()
    }

@router.post("/providers")
async def add_provider(provider_data: Dict) -> Dict:
    """Add a new provider."""
    try:
        provider_type = PROVIDER_TYPES.get(provider_data["type"])
        if not provider_type:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider type: {provider_data['type']}"
            )
            
        config = MarketDataConfig(
            credentials=provider_data.get("config", {}),
            base_url=provider_data.get("base_url", ""),
            websocket_url=provider_data.get("websocket_url", "")
        )
        
        provider = provider_type(config)
        priority = ProviderPriority[provider_data["priority"]]
        
        await provider_manager.add_provider(
            provider_id=provider_data["name"],
            provider=provider,
            priority=priority
        )
        
        return {"message": "Provider added successfully"}
        
    except Exception as e:
        logger.error(f"Error adding provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/providers/{provider_id}")
async def update_provider(provider_id: str, provider_data: Dict) -> Dict:
    """Update provider configuration."""
    state = provider_manager.providers.get(provider_id)
    if not state:
        raise HTTPException(status_code=404, detail="Provider not found")
        
    try:
        # Update priority if provided
        if "priority" in provider_data:
            new_priority = ProviderPriority[provider_data["priority"]]
            state.priority = new_priority
            
        # Update config if provided
        if "config" in provider_data:
            state.provider.config.credentials.update(provider_data["config"])
            
        return {"message": "Provider updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/providers/{provider_id}")
async def remove_provider(provider_id: str) -> Dict:
    """Remove a provider."""
    try:
        await provider_manager.remove_provider(provider_id)
        return {"message": "Provider removed successfully"}
    except Exception as e:
        logger.error(f"Error removing provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/providers/{provider_id}/start")
async def start_provider(provider_id: str) -> Dict:
    """Start a provider."""
    state = provider_manager.providers.get(provider_id)
    if not state:
        raise HTTPException(status_code=404, detail="Provider not found")
        
    try:
        if not state.provider.is_connected:
            await state.provider.connect()
        return {"message": "Provider started successfully"}
    except Exception as e:
        logger.error(f"Error starting provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/providers/{provider_id}/stop")
async def stop_provider(provider_id: str) -> Dict:
    """Stop a provider."""
    state = provider_manager.providers.get(provider_id)
    if not state:
        raise HTTPException(status_code=404, detail="Provider not found")
        
    try:
        if state.provider.is_connected:
            await state.provider.disconnect()
        return {"message": "Provider stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/providers/{provider_id}/health")
async def get_provider_health(provider_id: str) -> Dict:
    """Get provider health metrics."""
    state = provider_manager.providers.get(provider_id)
    if not state:
        raise HTTPException(status_code=404, detail="Provider not found")
        
    return state.health.get_metrics()

@router.get("/providers/{provider_id}/symbols")
async def get_provider_symbols(provider_id: str) -> List[str]:
    """Get active symbols for a provider."""
    state = provider_manager.providers.get(provider_id)
    if not state:
        raise HTTPException(status_code=404, detail="Provider not found")
        
    return list(state.active_symbols)
