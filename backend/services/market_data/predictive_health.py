"""Predictive health analysis for market data providers."""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)

@dataclass
class HealthPrediction:
    """Health prediction results."""
    provider_id: str
    metric_type: str
    current_value: float
    predicted_value: float
    confidence: float
    anomaly_score: float
    trend: str  # "improving", "stable", "degrading"
    prediction_timestamp: datetime
    forecast_window: int  # minutes
    alert_level: str  # "none", "warning", "critical"

class PredictiveHealthAnalyzer:
    """Analyzes provider health metrics to predict future issues."""

    def __init__(
        self,
        lookback_window: int = 60,  # minutes
        forecast_window: int = 15,  # minutes
        min_samples: int = 30,
        confidence_threshold: float = 0.8,
        anomaly_threshold: float = 0.7
    ):
        self.lookback_window = lookback_window
        self.forecast_window = forecast_window
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.anomaly_threshold = anomaly_threshold
        
        # Initialize models
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Store historical data
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}

    def add_metric_sample(
        self,
        provider_id: str,
        metric_type: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add a new metric sample to the history."""
        if timestamp is None:
            timestamp = datetime.now()
            
        key = f"{provider_id}:{metric_type}"
        if key not in self.metric_history:
            self.metric_history[key] = []
            
        # Add new sample
        self.metric_history[key].append((timestamp, value))
        
        # Prune old samples
        cutoff = datetime.now() - timedelta(minutes=self.lookback_window)
        self.metric_history[key] = [
            (ts, val) for ts, val in self.metric_history[key]
            if ts > cutoff
        ]

    def get_metric_prediction(
        self,
        provider_id: str,
        metric_type: str
    ) -> Optional[HealthPrediction]:
        """Get prediction for a specific metric."""
        key = f"{provider_id}:{metric_type}"
        if key not in self.metric_history:
            return None
            
        history = self.metric_history[key]
        if len(history) < self.min_samples:
            return None
            
        try:
            # Prepare data
            timestamps, values = zip(*history)
            df = pd.DataFrame({
                'timestamp': timestamps,
                'value': values
            })
            df.set_index('timestamp', inplace=True)
            
            # Detect anomalies
            scaled_values = self.scaler.fit_transform(df[['value']])
            anomaly_scores = self.anomaly_detector.fit_predict(scaled_values)
            anomaly_score = np.mean([1 if score == -1 else 0 for score in anomaly_scores])
            
            # Perform time series forecasting
            model = ExponentialSmoothing(
                df['value'],
                seasonal_periods=12,
                trend='add',
                seasonal='add'
            )
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(self.forecast_window)
            predicted_value = forecast.iloc[-1]
            
            # Calculate confidence interval
            residuals = df['value'] - fitted_model.fittedvalues
            confidence = 1 - (np.std(residuals) / np.mean(df['value']))
            
            # Determine trend
            recent_values = df['value'].tail(min(10, len(df)))
            slope, _, _, _, _ = stats.linregress(range(len(recent_values)), recent_values)
            
            if abs(slope) < 0.01:
                trend = "stable"
            else:
                trend = "improving" if slope < 0 else "degrading"
            
            # Determine alert level
            alert_level = "none"
            if anomaly_score > self.anomaly_threshold:
                alert_level = "critical"
            elif confidence < self.confidence_threshold:
                alert_level = "warning"
            
            return HealthPrediction(
                provider_id=provider_id,
                metric_type=metric_type,
                current_value=df['value'].iloc[-1],
                predicted_value=predicted_value,
                confidence=confidence,
                anomaly_score=anomaly_score,
                trend=trend,
                prediction_timestamp=datetime.now(),
                forecast_window=self.forecast_window,
                alert_level=alert_level
            )
            
        except Exception as e:
            logger.error(f"Error generating prediction for {key}: {e}")
            return None

    def analyze_provider_health(
        self,
        provider_id: str,
        metrics: Dict[str, float]
    ) -> Dict[str, HealthPrediction]:
        """Analyze all metrics for a provider."""
        predictions = {}
        timestamp = datetime.now()
        
        for metric_type, value in metrics.items():
            # Add new sample
            self.add_metric_sample(provider_id, metric_type, value, timestamp)
            
            # Generate prediction
            prediction = self.get_metric_prediction(provider_id, metric_type)
            if prediction:
                predictions[metric_type] = prediction
                
        return predictions

    def get_provider_risk_score(
        self,
        provider_id: str
    ) -> Optional[float]:
        """Calculate overall risk score for a provider."""
        predictions = []
        for key in self.metric_history:
            if key.startswith(f"{provider_id}:"):
                metric_type = key.split(':')[1]
                prediction = self.get_metric_prediction(provider_id, metric_type)
                if prediction:
                    predictions.append(prediction)
                    
        if not predictions:
            return None
            
        # Calculate weighted risk score
        weights = {
            'error_rate': 0.3,
            'latency': 0.3,
            'success_rate': 0.2,
            'availability': 0.2
        }
        
        total_score = 0
        total_weight = 0
        
        for pred in predictions:
            weight = weights.get(pred.metric_type, 0.1)
            score = (
                (1 - pred.confidence) * 0.3 +
                pred.anomaly_score * 0.4 +
                (0.3 if pred.trend == "degrading" else 0)
            )
            total_score += score * weight
            total_weight += weight
            
        return total_score / total_weight if total_weight > 0 else None

    def get_health_forecast(
        self,
        provider_id: str,
        window_minutes: int = 60
    ) -> Dict[str, List[float]]:
        """Generate health forecast for specified time window."""
        forecasts = {}
        
        for key in self.metric_history:
            if key.startswith(f"{provider_id}:"):
                metric_type = key.split(':')[1]
                history = self.metric_history[key]
                
                if len(history) >= self.min_samples:
                    try:
                        # Prepare data
                        timestamps, values = zip(*history)
                        df = pd.DataFrame({
                            'timestamp': timestamps,
                            'value': values
                        })
                        df.set_index('timestamp', inplace=True)
                        
                        # Fit model
                        model = ExponentialSmoothing(
                            df['value'],
                            seasonal_periods=12,
                            trend='add',
                            seasonal='add'
                        )
                        fitted_model = model.fit()
                        
                        # Generate forecast
                        forecast = fitted_model.forecast(window_minutes)
                        forecasts[metric_type] = forecast.tolist()
                        
                    except Exception as e:
                        logger.error(f"Error generating forecast for {key}: {e}")
                        
        return forecasts
