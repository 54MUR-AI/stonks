"""
API endpoints for alert analytics and predictions.
"""
from fastapi import APIRouter, HTTPException
from typing import List

from ..services.market_data.alerts import AlertManager
from ..services.market_data.alert_analytics import AlertPattern, AnomalyPrediction

router = APIRouter()
alert_manager = AlertManager()

@router.get("/alerts/patterns", response_model=List[AlertPattern])
async def get_alert_patterns():
    """Get current alert patterns"""
    try:
        patterns = await alert_manager.get_alert_patterns()
        return patterns
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/predictions", response_model=List[AnomalyPrediction])
async def get_anomaly_predictions():
    """Get current anomaly predictions"""
    try:
        predictions = await alert_manager.get_predicted_anomalies()
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/analytics/summary")
async def get_analytics_summary():
    """Get a summary of alert analytics"""
    try:
        patterns = await alert_manager.get_alert_patterns()
        predictions = await alert_manager.get_predicted_anomalies()
        
        # Calculate summary statistics
        total_patterns = len(patterns)
        active_predictions = len(predictions)
        high_probability_predictions = len([
            p for p in predictions if p.probability > 0.7
        ])
        
        # Get most common root causes
        root_causes = {}
        for pattern in patterns:
            for alert_type, prob in pattern.root_cause_probability.items():
                if alert_type not in root_causes:
                    root_causes[alert_type] = 0
                root_causes[alert_type] += prob
        
        top_root_causes = sorted(
            root_causes.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_patterns": total_patterns,
            "active_predictions": active_predictions,
            "high_probability_predictions": high_probability_predictions,
            "top_root_causes": top_root_causes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
