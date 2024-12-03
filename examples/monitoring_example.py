"""Example usage of model monitoring components."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from backend.services.ml.monitoring.metrics import (
    MetricConfig,
    MetricsTracker
)
from backend.services.ml.monitoring.drift import (
    DriftConfig,
    DriftDetector,
    ConceptDriftDetector
)
from backend.services.ml.deep_learning.models import (
    LSTM,
    TrainingConfig
)

def generate_sample_data(
    n_samples: int,
    n_features: int,
    drift_factor: float = 0.0
) -> np.ndarray:
    """Generate sample data with optional drift.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        drift_factor: Factor to control drift magnitude
        
    Returns:
        Generated data array
    """
    base_data = np.random.randn(n_samples, n_features)
    drift = drift_factor * np.random.randn(n_samples, n_features)
    return base_data + drift

def main():
    # Configuration
    n_features = 10
    sequence_length = 10
    n_samples = 1000
    
    # Create directories
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup metric tracking
    metric_config = MetricConfig(
        log_dir=str(log_dir),
        model_name="LSTM",
        experiment_name="drift_detection",
        metrics_to_track=['accuracy', 'mae'],
        save_frequency=10,
        plot_metrics=True
    )
    
    metrics_tracker = MetricsTracker(metric_config)
    
    # Setup drift detection
    drift_config = DriftConfig(
        reference_window=30,
        detection_window=7,
        drift_threshold=0.05,
        feature_names=[f"feature_{i}" for i in range(n_features)],
        monitoring_frequency='daily'
    )
    
    drift_detector = DriftDetector(drift_config)
    concept_drift_detector = ConceptDriftDetector()
    
    # Generate initial data
    reference_data = generate_sample_data(n_samples, n_features)
    drift_detector.set_reference_data(reference_data)
    
    # Create and train model
    model = LSTM(input_size=n_features)
    training_config = TrainingConfig()
    
    # Simulate training and monitoring
    metrics_tracker.start_training()
    
    for epoch in range(training_config.num_epochs):
        # Generate new data with increasing drift
        drift_factor = epoch * 0.1
        current_data = generate_sample_data(
            n_samples,
            n_features,
            drift_factor
        )
        
        # Detect data drift
        drift_results = drift_detector.detect_drift(
            current_data,
            datetime.now()
        )
        
        # Log drift detection results
        for feature, results in drift_results.items():
            if results['drift_detected']:
                print(f"Drift detected in feature {feature}")
                print(f"KS statistic: {results['ks_statistic']:.4f}")
                print(f"p-value: {results['ks_pvalue']:.4f}")
                
        # Simulate batch training
        for batch in range(0, n_samples, training_config.batch_size):
            batch_data = current_data[batch:batch + training_config.batch_size]
            
            # Simulate loss and metrics
            loss = 1.0 / (epoch + 1) + np.random.randn() * 0.1
            metrics = {
                'accuracy': 0.8 + np.random.randn() * 0.05,
                'mae': 0.2 + np.random.randn() * 0.05
            }
            
            # Log metrics
            metrics_tracker.log_batch(
                epoch,
                batch // training_config.batch_size,
                loss,
                metrics,
                'train'
            )
            
        # Simulate validation
        val_loss = 1.2 / (epoch + 1) + np.random.randn() * 0.1
        val_metrics = {
            'accuracy': 0.75 + np.random.randn() * 0.05,
            'mae': 0.25 + np.random.randn() * 0.05
        }
        
        metrics_tracker.log_batch(
            epoch,
            0,
            val_loss,
            val_metrics,
            'val'
        )
        
        # Visualize drift
        drift_detector.visualize_drift(
            current_data,
            drift_results,
            str(log_dir)
        )
        
        # Update reference window
        drift_detector.update_reference_window(
            current_data,
            datetime.now()
        )
        
    metrics_tracker.end_training()
    
    # Generate final report
    report = metrics_tracker.create_report()
    print("\nFinal Report:")
    print(report)
    
    # Plot final metrics
    metrics_tracker.plot_metrics()
    
if __name__ == '__main__':
    main()
