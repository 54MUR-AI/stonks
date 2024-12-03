"""
Advanced alert analytics module providing aggregation, correlation, and predictive capabilities.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import asyncio
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .alerts import Alert, AlertType, AlertSeverity
from ..utils.time_series import TimeWindow, aggregate_time_series

logger = logging.getLogger(__name__)

@dataclass
class AlertPattern:
    """Represents a pattern of related alerts"""
    id: str
    alert_types: Set[AlertType]
    provider_ids: Set[str]
    root_cause_probability: Dict[AlertType, float]
    first_occurrence: datetime
    last_occurrence: datetime
    alerts: List[Alert]
    
@dataclass
class AnomalyPrediction:
    """Represents a predicted anomaly"""
    provider_id: str
    alert_type: AlertType
    probability: float
    predicted_value: float
    prediction_time: datetime
    features: Dict[str, float]

class AlertAnalytics:
    def __init__(self, window_size: timedelta = timedelta(hours=24)):
        self.window_size = window_size
        self.alert_patterns: Dict[str, AlertPattern] = {}
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.historical_data: Dict[str, List[Dict]] = {}
        self._analysis_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the analytics engine"""
        if not self._analysis_task or self._analysis_task.done():
            self._analysis_task = asyncio.create_task(self._periodic_analysis())
            logger.info("Alert analytics engine started")

    async def stop(self):
        """Stop the analytics engine"""
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
            self._analysis_task = None
            logger.info("Alert analytics engine stopped")

    def add_alert(self, alert: Alert):
        """Add a new alert for analysis"""
        key = f"{alert.provider_id}:{alert.type.value}"
        if key not in self.historical_data:
            self.historical_data[key] = []
        
        self.historical_data[key].append({
            'timestamp': alert.timestamp,
            'severity': alert.severity.value,
            'metric_value': alert.metric_value,
            'threshold_value': alert.threshold_value
        })
        
        # Trigger pattern detection
        self._detect_patterns(alert)
        
        # Update anomaly detection model
        self._update_anomaly_detector(key)

    def _detect_patterns(self, new_alert: Alert) -> Optional[AlertPattern]:
        """Detect patterns in alerts"""
        # Look for related alerts in the last hour
        window_start = new_alert.timestamp - timedelta(hours=1)
        
        # Find alerts that might be related
        related_alerts = []
        for pattern in self.alert_patterns.values():
            if pattern.last_occurrence >= window_start:
                if (new_alert.provider_id in pattern.provider_ids or 
                    new_alert.type in pattern.alert_types):
                    related_alerts.extend(pattern.alerts)
        
        if not related_alerts:
            # Create new pattern
            pattern_id = f"pattern_{len(self.alert_patterns)}"
            pattern = AlertPattern(
                id=pattern_id,
                alert_types={new_alert.type},
                provider_ids={new_alert.provider_id},
                root_cause_probability={new_alert.type: 1.0},
                first_occurrence=new_alert.timestamp,
                last_occurrence=new_alert.timestamp,
                alerts=[new_alert]
            )
            self.alert_patterns[pattern_id] = pattern
            return pattern
        
        # Find the most relevant existing pattern
        best_pattern = None
        max_similarity = 0
        
        for pattern in self.alert_patterns.values():
            if pattern.last_occurrence >= window_start:
                similarity = self._calculate_pattern_similarity(pattern, new_alert)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pattern = pattern
        
        if max_similarity > 0.5 and best_pattern:
            # Update existing pattern
            best_pattern.alert_types.add(new_alert.type)
            best_pattern.provider_ids.add(new_alert.provider_id)
            best_pattern.last_occurrence = new_alert.timestamp
            best_pattern.alerts.append(new_alert)
            
            # Update root cause probabilities
            self._update_root_cause_probabilities(best_pattern)
            
            return best_pattern
        else:
            # Create new pattern
            pattern_id = f"pattern_{len(self.alert_patterns)}"
            pattern = AlertPattern(
                id=pattern_id,
                alert_types={new_alert.type},
                provider_ids={new_alert.provider_id},
                root_cause_probability={new_alert.type: 1.0},
                first_occurrence=new_alert.timestamp,
                last_occurrence=new_alert.timestamp,
                alerts=[new_alert]
            )
            self.alert_patterns[pattern_id] = pattern
            return pattern

    def _calculate_pattern_similarity(self, pattern: AlertPattern, alert: Alert) -> float:
        """Calculate similarity between a pattern and a new alert"""
        # Calculate temporal proximity
        time_diff = (alert.timestamp - pattern.last_occurrence).total_seconds()
        time_score = max(0, 1 - (time_diff / 3600))  # Decay over 1 hour
        
        # Calculate type similarity
        type_score = 0.5 if alert.type in pattern.alert_types else 0
        
        # Calculate provider similarity
        provider_score = 0.5 if alert.provider_id in pattern.provider_ids else 0
        
        return time_score * (type_score + provider_score)

    def _update_root_cause_probabilities(self, pattern: AlertPattern):
        """Update root cause probabilities for a pattern"""
        alert_counts = {}
        total_alerts = len(pattern.alerts)
        
        # Count alerts by type
        for alert in pattern.alerts:
            if alert.type not in alert_counts:
                alert_counts[alert.type] = 0
            alert_counts[alert.type] += 1
        
        # Calculate probabilities
        pattern.root_cause_probability = {
            alert_type: count / total_alerts
            for alert_type, count in alert_counts.items()
        }

    def _update_anomaly_detector(self, key: str):
        """Update the anomaly detection model for a specific metric"""
        data = self.historical_data[key]
        if len(data) < 10:  # Need minimum data points
            return
        
        # Prepare features
        features = np.array([[
            d['metric_value'],
            d['threshold_value'],
            d['severity']
        ] for d in data])
        
        # Initialize or update scaler
        if key not in self.scalers:
            self.scalers[key] = StandardScaler()
            self.scalers[key].fit(features)
        else:
            self.scalers[key].partial_fit(features)
        
        # Scale features
        scaled_features = self.scalers[key].transform(features)
        
        # Train or update anomaly detector
        if key not in self.anomaly_detectors:
            self.anomaly_detectors[key] = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            )
            self.anomaly_detectors[key].fit(scaled_features)
        else:
            # For online learning, we retrain on recent data
            recent_data = scaled_features[-1000:]  # Last 1000 points
            self.anomaly_detectors[key].fit(recent_data)

    async def predict_anomalies(self) -> List[AnomalyPrediction]:
        """Predict potential anomalies"""
        predictions = []
        
        for key, data in self.historical_data.items():
            if len(data) < 10:  # Need minimum data points
                continue
                
            provider_id, alert_type = key.split(':')
            alert_type = AlertType(alert_type)
            
            # Get recent data
            recent_data = data[-10:]  # Last 10 points
            
            # Prepare features
            features = np.array([[
                d['metric_value'],
                d['threshold_value'],
                d['severity']
            ] for d in recent_data])
            
            # Scale features
            scaled_features = self.scalers[key].transform(features)
            
            # Predict anomaly scores
            scores = self.anomaly_detectors[key].score_samples(scaled_features)
            
            # If recent points show anomalous behavior
            if scores[-1] < np.percentile(scores, 10):  # Bottom 10%
                predictions.append(AnomalyPrediction(
                    provider_id=provider_id,
                    alert_type=alert_type,
                    probability=1.0 - (scores[-1] / np.min(scores)),
                    predicted_value=features[-1][0],  # Last metric value
                    prediction_time=datetime.now(),
                    features={
                        'metric_value': features[-1][0],
                        'threshold_value': features[-1][1],
                        'severity': features[-1][2]
                    }
                ))
        
        return predictions

    async def _periodic_analysis(self):
        """Periodically run analysis"""
        while True:
            try:
                # Clean up old patterns
                cutoff = datetime.now() - self.window_size
                self.alert_patterns = {
                    k: v for k, v in self.alert_patterns.items()
                    if v.last_occurrence >= cutoff
                }
                
                # Clean up old historical data
                for key in self.historical_data:
                    self.historical_data[key] = [
                        d for d in self.historical_data[key]
                        if d['timestamp'] >= cutoff
                    ]
                
                # Predict anomalies
                await self.predict_anomalies()
                
            except Exception as e:
                logger.error(f"Error in periodic analysis: {e}")
            
            await asyncio.sleep(60)  # Run every minute
