"""Tests for predictive health analysis."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from backend.services.market_data.predictive_health import (
    PredictiveHealthAnalyzer,
    HealthPrediction
)

@pytest.fixture
def analyzer():
    """Create analyzer for testing."""
    return PredictiveHealthAnalyzer(
        lookback_window=60,
        forecast_window=15,
        min_samples=10  # Smaller for testing
    )

def generate_test_data(
    base_value: float,
    noise_level: float,
    trend: float,
    num_points: int
) -> list:
    """Generate test data with trend and noise."""
    timestamps = [
        datetime.now() - timedelta(minutes=i)
        for i in range(num_points - 1, -1, -1)
    ]
    values = [
        base_value + (i * trend) + (np.random.normal(0, noise_level))
        for i in range(num_points)
    ]
    return list(zip(timestamps, values))

def test_add_metric_sample(analyzer):
    """Test adding metric samples."""
    analyzer.add_metric_sample("test_provider", "latency", 100.0)
    key = "test_provider:latency"
    
    assert key in analyzer.metric_history
    assert len(analyzer.metric_history[key]) == 1
    assert analyzer.metric_history[key][0][1] == 100.0

def test_metric_history_pruning(analyzer):
    """Test old metrics are pruned."""
    # Add old sample
    old_time = datetime.now() - timedelta(minutes=analyzer.lookback_window + 10)
    analyzer.add_metric_sample(
        "test_provider",
        "latency",
        100.0,
        timestamp=old_time
    )
    
    # Add new sample
    analyzer.add_metric_sample("test_provider", "latency", 200.0)
    
    key = "test_provider:latency"
    assert len(analyzer.metric_history[key]) == 1
    assert analyzer.metric_history[key][0][1] == 200.0

def test_prediction_minimum_samples(analyzer):
    """Test prediction requires minimum samples."""
    # Add fewer than minimum samples
    for i in range(analyzer.min_samples - 1):
        analyzer.add_metric_sample("test_provider", "latency", 100.0)
        
    prediction = analyzer.get_metric_prediction("test_provider", "latency")
    assert prediction is None

def test_prediction_with_stable_data(analyzer):
    """Test prediction with stable metric values."""
    # Generate stable data
    data = generate_test_data(
        base_value=100.0,
        noise_level=1.0,
        trend=0.0,
        num_points=20
    )
    
    for timestamp, value in data:
        analyzer.add_metric_sample(
            "test_provider",
            "latency",
            value,
            timestamp
        )
        
    prediction = analyzer.get_metric_prediction("test_provider", "latency")
    
    assert prediction is not None
    assert prediction.trend == "stable"
    assert prediction.alert_level == "none"
    assert abs(prediction.predicted_value - 100.0) < 5.0
    assert prediction.confidence > 0.8

def test_prediction_with_degrading_data(analyzer):
    """Test prediction with degrading metrics."""
    # Generate degrading data
    data = generate_test_data(
        base_value=100.0,
        noise_level=1.0,
        trend=5.0,  # Increasing trend
        num_points=20
    )
    
    for timestamp, value in data:
        analyzer.add_metric_sample(
            "test_provider",
            "latency",
            value,
            timestamp
        )
        
    prediction = analyzer.get_metric_prediction("test_provider", "latency")
    
    assert prediction is not None
    assert prediction.trend == "degrading"
    assert prediction.alert_level in ["warning", "critical"]
    assert prediction.predicted_value > 100.0

def test_prediction_with_improving_data(analyzer):
    """Test prediction with improving metrics."""
    # Generate improving data
    data = generate_test_data(
        base_value=100.0,
        noise_level=1.0,
        trend=-5.0,  # Decreasing trend
        num_points=20
    )
    
    for timestamp, value in data:
        analyzer.add_metric_sample(
            "test_provider",
            "latency",
            value,
            timestamp
        )
        
    prediction = analyzer.get_metric_prediction("test_provider", "latency")
    
    assert prediction is not None
    assert prediction.trend == "improving"
    assert prediction.alert_level == "none"
    assert prediction.predicted_value < 100.0

def test_anomaly_detection(analyzer):
    """Test anomaly detection."""
    # Generate normal data
    data = generate_test_data(
        base_value=100.0,
        noise_level=1.0,
        trend=0.0,
        num_points=19
    )
    
    # Add one anomaly
    data.append((datetime.now(), 1000.0))
    
    for timestamp, value in data:
        analyzer.add_metric_sample(
            "test_provider",
            "latency",
            value,
            timestamp
        )
        
    prediction = analyzer.get_metric_prediction("test_provider", "latency")
    
    assert prediction is not None
    assert prediction.anomaly_score > analyzer.anomaly_threshold
    assert prediction.alert_level == "critical"

def test_risk_score_calculation(analyzer):
    """Test provider risk score calculation."""
    # Add data for multiple metrics
    metrics = {
        "latency": (100.0, 5.0),  # base_value, trend
        "error_rate": (0.01, 0.001),
        "success_rate": (0.99, -0.001)
    }
    
    for metric, (base, trend) in metrics.items():
        data = generate_test_data(
            base_value=base,
            noise_level=base * 0.01,
            trend=trend,
            num_points=20
        )
        for timestamp, value in data:
            analyzer.add_metric_sample(
                "test_provider",
                metric,
                value,
                timestamp
            )
            
    risk_score = analyzer.get_provider_risk_score("test_provider")
    
    assert risk_score is not None
    assert 0 <= risk_score <= 1

def test_health_forecast(analyzer):
    """Test health forecasting."""
    # Generate data
    data = generate_test_data(
        base_value=100.0,
        noise_level=1.0,
        trend=2.0,
        num_points=20
    )
    
    for timestamp, value in data:
        analyzer.add_metric_sample(
            "test_provider",
            "latency",
            value,
            timestamp
        )
        
    forecast = analyzer.get_health_forecast("test_provider", window_minutes=30)
    
    assert "latency" in forecast
    assert len(forecast["latency"]) == 30  # One value per minute
    assert all(isinstance(v, float) for v in forecast["latency"])
