from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime
from ..services.market_data.escalation import EscalationManager

router = APIRouter(prefix="/api/escalation")

class AlertCondition(BaseModel):
    alert_type: str
    severity: str
    duration_threshold: int

class EscalationPolicy(BaseModel):
    name: str
    conditions: AlertCondition
    initial_level: str
    max_level: str
    escalation_delay: int
    notification_channels: Dict[str, List[str]]
    auto_actions: Dict[str, List[str]]

class Alert(BaseModel):
    provider_id: str
    type: str
    severity: str
    message: str

class EscalationEvent(BaseModel):
    id: str
    alert: Alert
    policy: EscalationPolicy
    current_level: str
    start_time: datetime
    last_escalation: datetime
    actions_taken: List[str]
    resolved: bool
    resolution_time: Optional[datetime] = None

# Initialize the escalation manager
escalation_manager = EscalationManager()

@router.get("/active", response_model=List[EscalationEvent])
async def get_active_escalations():
    """Get all active escalation events"""
    try:
        return await escalation_manager.get_active_escalations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/policies", response_model=List[EscalationPolicy])
async def get_escalation_policies():
    """Get all escalation policies"""
    try:
        return await escalation_manager.get_policies()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/policies")
async def create_escalation_policy(policy: EscalationPolicy):
    """Create a new escalation policy"""
    try:
        await escalation_manager.create_policy(policy)
        return {"message": "Policy created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/policies/{policy_name}")
async def update_escalation_policy(policy_name: str, policy: EscalationPolicy):
    """Update an existing escalation policy"""
    try:
        await escalation_manager.update_policy(policy_name, policy)
        return {"message": "Policy updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/policies/{policy_name}")
async def delete_escalation_policy(policy_name: str):
    """Delete an escalation policy"""
    try:
        await escalation_manager.delete_policy(policy_name)
        return {"message": "Policy deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/events/{event_id}/resolve")
async def resolve_escalation(event_id: str):
    """Resolve an escalation event"""
    try:
        await escalation_manager.resolve_escalation(event_id)
        return {"message": "Escalation resolved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/events/{event_id}/acknowledge")
async def acknowledge_escalation(event_id: str):
    """Acknowledge an escalation event"""
    try:
        await escalation_manager.acknowledge_escalation(event_id)
        return {"message": "Escalation acknowledged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/events/{event_id}/escalate")
async def manual_escalate(event_id: str):
    """Manually escalate an event to the next level"""
    try:
        await escalation_manager.manual_escalate(event_id)
        return {"message": "Escalation level increased successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=Dict)
async def get_escalation_metrics():
    """Get metrics about escalations"""
    try:
        return await escalation_manager.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[EscalationEvent])
async def get_escalation_history(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    provider_id: Optional[str] = None,
    policy_name: Optional[str] = None
):
    """Get historical escalation events with optional filters"""
    try:
        return await escalation_manager.get_history(
            start_date=start_date,
            end_date=end_date,
            provider_id=provider_id,
            policy_name=policy_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
