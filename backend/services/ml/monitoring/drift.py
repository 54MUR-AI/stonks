"""Model and data drift detection."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    reference_window: int  # Number of days for reference data
    detection_window: int  # Number of days for detection
    drift_threshold: float  # Threshold for drift detection
    feature_names: List[str]
    monitoring_frequency: str = 'daily'  # 'daily', 'hourly', 'weekly'

class DriftDetector:
    """Detect model and data drift."""
    
    def __init__(self, config: DriftConfig):
        """Initialize detector.
        
        Args:
            config: Drift configuration
        """
        self.config = config
        self.reference_data = None
        self.reference_statistics = None
        self.pca = PCA(n_components=2)
        self.scaler = StandardScaler()
        
    def set_reference_data(self, data: np.ndarray):
        """Set reference data and compute statistics.
        
        Args:
            data: Reference data array
        """
        self.reference_data = data
        self.reference_statistics = self._compute_statistics(data)
        
        # Fit PCA for visualization
        scaled_data = self.scaler.fit_transform(data)
        self.pca.fit(scaled_data)
        
    def _compute_statistics(
        self,
        data: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistical measures for data.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary of statistics per feature
        """
        statistics = {}
        
        for i, feature in enumerate(self.config.feature_names):
            feature_data = data[:, i]
            statistics[feature] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'median': np.median(feature_data),
                'skewness': stats.skew(feature_data),
                'kurtosis': stats.kurtosis(feature_data),
                'iqr': np.percentile(feature_data, 75) - 
                      np.percentile(feature_data, 25)
            }
            
        return statistics
        
    def detect_drift(
        self,
        data: np.ndarray,
        timestamp: datetime
    ) -> Dict[str, Dict[str, Union[bool, float]]]:
        """Detect drift in new data.
        
        Args:
            data: New data array
            timestamp: Timestamp of the data
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
            
        current_statistics = self._compute_statistics(data)
        drift_results = {}
        
        for feature in self.config.feature_names:
            ref_stats = self.reference_statistics[feature]
            curr_stats = current_statistics[feature]
            
            # Compute statistical distances
            ks_statistic, ks_pvalue = stats.ks_2samp(
                self.reference_data[:, self.config.feature_names.index(feature)],
                data[:, self.config.feature_names.index(feature)]
            )
            
            # Check for significant changes in statistics
            stat_changes = {
                stat: abs(ref_stats[stat] - curr_stats[stat]) / ref_stats[stat]
                for stat in ref_stats.keys()
            }
            
            # Detect drift based on threshold
            drift_detected = (
                ks_pvalue < self.config.drift_threshold or
                any(change > self.config.drift_threshold 
                    for change in stat_changes.values())
            )
            
            drift_results[feature] = {
                'drift_detected': drift_detected,
                'ks_statistic': ks_statistic,
                'ks_pvalue': ks_pvalue,
                'statistical_changes': stat_changes
            }
            
        return drift_results
        
    def visualize_drift(
        self,
        data: np.ndarray,
        drift_results: Dict[str, Dict[str, Union[bool, float]]],
        save_path: str
    ):
        """Visualize drift detection results.
        
        Args:
            data: New data array
            drift_results: Results from drift detection
            save_path: Path to save visualizations
        """
        # PCA visualization
        scaled_data = self.scaler.transform(data)
        pca_result = self.pca.transform(scaled_data)
        ref_pca = self.pca.transform(
            self.scaler.transform(self.reference_data)
        )
        
        plt.figure(figsize=(10, 6))
        plt.scatter(
            ref_pca[:, 0],
            ref_pca[:, 1],
            alpha=0.5,
            label='Reference'
        )
        plt.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            alpha=0.5,
            label='Current'
        )
        plt.title('PCA Visualization of Data Drift')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.savefig(f"{save_path}/pca_drift.png")
        plt.close()
        
        # Statistical changes visualization
        changes_data = []
        for feature, results in drift_results.items():
            for stat, change in results['statistical_changes'].items():
                changes_data.append({
                    'Feature': feature,
                    'Statistic': stat,
                    'Change': change
                })
                
        changes_df = pd.DataFrame(changes_data)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            changes_df.pivot(
                index='Feature',
                columns='Statistic',
                values='Change'
            ),
            annot=True,
            cmap='RdYlBu_r',
            center=0
        )
        plt.title('Statistical Changes Heatmap')
        plt.tight_layout()
        plt.savefig(f"{save_path}/statistical_changes.png")
        plt.close()
        
    def update_reference_window(
        self,
        new_data: np.ndarray,
        timestamp: datetime
    ):
        """Update reference window with new data.
        
        Args:
            new_data: New data array
            timestamp: Timestamp of the data
        """
        if self.reference_data is None:
            self.set_reference_data(new_data)
            return
            
        # Combine data and sort by timestamp
        combined_data = np.vstack([self.reference_data, new_data])
        
        # Keep only the data within the reference window
        window_start = timestamp - timedelta(
            days=self.config.reference_window
        )
        
        # Update reference data and statistics
        self.reference_data = combined_data
        self.reference_statistics = self._compute_statistics(
            self.reference_data
        )
        
        # Update PCA and scaler
        scaled_data = self.scaler.fit_transform(self.reference_data)
        self.pca.fit(scaled_data)
        
class ConceptDriftDetector:
    """Detect concept drift in model predictions."""
    
    def __init__(
        self,
        window_size: int = 1000,
        alpha: float = 0.05
    ):
        """Initialize detector.
        
        Args:
            window_size: Size of detection window
            alpha: Significance level
        """
        self.window_size = window_size
        self.alpha = alpha
        self.reference_predictions = None
        self.reference_targets = None
        
    def set_reference(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ):
        """Set reference predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: True target values
        """
        self.reference_predictions = predictions
        self.reference_targets = targets
        
    def detect_drift(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Union[bool, float]]:
        """Detect concept drift.
        
        Args:
            predictions: New predictions
            targets: New targets
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_predictions is None:
            raise ValueError("Reference data not set")
            
        # Compute prediction errors
        ref_errors = np.abs(self.reference_predictions - self.reference_targets)
        new_errors = np.abs(predictions - targets)
        
        # Perform statistical test
        statistic, pvalue = stats.ks_2samp(ref_errors, new_errors)
        
        return {
            'drift_detected': pvalue < self.alpha,
            'statistic': statistic,
            'pvalue': pvalue
        }
