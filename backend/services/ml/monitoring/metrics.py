"""Model monitoring and metrics tracking."""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import json
import time
from datetime import datetime
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score
)

@dataclass
class MetricConfig:
    """Configuration for metric tracking."""
    log_dir: str
    model_name: str
    experiment_name: str
    metrics_to_track: List[str]
    save_frequency: int = 10
    plot_metrics: bool = True

@dataclass
class ModelMetrics:
    """Container for model metrics."""
    training_loss: List[float] = field(default_factory=list)
    validation_loss: List[float] = field(default_factory=list)
    training_metrics: Dict[str, List[float]] = field(default_factory=dict)
    validation_metrics: Dict[str, List[float]] = field(default_factory=dict)
    prediction_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'prediction_metrics': self.prediction_metrics,
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'memory_usage': self.memory_usage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """Create metrics from dictionary."""
        return cls(**data)

class MetricsTracker:
    """Track and log model metrics."""
    
    def __init__(self, config: MetricConfig):
        """Initialize tracker.
        
        Args:
            config: Metric configuration
        """
        self.config = config
        self.metrics = ModelMetrics()
        self.start_time = None
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f"{self.config.model_name}_{self.config.experiment_name}.log"
                ),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def start_training(self):
        """Start training timer."""
        self.start_time = time.time()
        self.logger.info(
            f"Started training {self.config.model_name} "
            f"for experiment {self.config.experiment_name}"
        )
        
    def end_training(self):
        """End training timer."""
        if self.start_time is not None:
            self.metrics.training_time = time.time() - self.start_time
            self.logger.info(
                f"Finished training in {self.metrics.training_time:.2f} seconds"
            )
            
    def log_batch(
        self,
        epoch: int,
        batch: int,
        loss: float,
        metrics: Dict[str, float],
        phase: str = 'train'
    ):
        """Log batch metrics.
        
        Args:
            epoch: Current epoch
            batch: Current batch
            loss: Loss value
            metrics: Dictionary of metric values
            phase: Training phase ('train' or 'val')
        """
        if phase == 'train':
            self.metrics.training_loss.append(loss)
            for name, value in metrics.items():
                if name not in self.metrics.training_metrics:
                    self.metrics.training_metrics[name] = []
                self.metrics.training_metrics[name].append(value)
        else:
            self.metrics.validation_loss.append(loss)
            for name, value in metrics.items():
                if name not in self.metrics.validation_metrics:
                    self.metrics.validation_metrics[name] = []
                self.metrics.validation_metrics[name].append(value)
                
        if batch % self.config.save_frequency == 0:
            self.save_metrics()
            
        self.logger.info(
            f"Epoch {epoch}, Batch {batch}, {phase.capitalize()} "
            f"Loss: {loss:.4f}, Metrics: {metrics}"
        )
        
    def log_prediction_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = 'regression'
    ):
        """Log prediction metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            task_type: Type of task ('regression' or 'classification')
        """
        if task_type == 'regression':
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        else:
            metrics = {
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
            
        self.metrics.prediction_metrics.update(metrics)
        self.logger.info(f"Prediction metrics: {metrics}")
        
    def log_memory_usage(self, memory_dict: Dict[str, float]):
        """Log memory usage.
        
        Args:
            memory_dict: Dictionary of memory usage values
        """
        self.metrics.memory_usage.update(memory_dict)
        self.logger.info(f"Memory usage: {memory_dict}")
        
    def save_metrics(self):
        """Save metrics to file."""
        metrics_path = Path(self.config.log_dir) / 'metrics'
        metrics_path.mkdir(parents=True, exist_ok=True)
        
        filename = (
            f"{self.config.model_name}_{self.config.experiment_name}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(metrics_path / filename, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=4)
            
    def plot_metrics(self):
        """Plot training and validation metrics."""
        if not self.config.plot_metrics:
            return
            
        metrics_path = Path(self.config.log_dir) / 'plots'
        metrics_path.mkdir(parents=True, exist_ok=True)
        
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics.training_loss, label='Training Loss')
        plt.plot(self.metrics.validation_loss, label='Validation Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(
            metrics_path / f"{self.config.model_name}_{self.config.experiment_name}_loss.png"
        )
        plt.close()
        
        # Plot other metrics
        for metric in self.metrics.training_metrics.keys():
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.metrics.training_metrics[metric],
                label=f'Training {metric}'
            )
            if metric in self.metrics.validation_metrics:
                plt.plot(
                    self.metrics.validation_metrics[metric],
                    label=f'Validation {metric}'
                )
            plt.title(f'{metric} Over Time')
            plt.xlabel('Batch')
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(
                metrics_path / 
                f"{self.config.model_name}_{self.config.experiment_name}_{metric}.png"
            )
            plt.close()
            
    def create_report(self) -> pd.DataFrame:
        """Create summary report.
        
        Returns:
            DataFrame with metric summary
        """
        report_data = []
        
        # Training metrics
        for metric, values in self.metrics.training_metrics.items():
            report_data.append({
                'Metric': f'Training {metric}',
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values)
            })
            
        # Validation metrics
        for metric, values in self.metrics.validation_metrics.items():
            report_data.append({
                'Metric': f'Validation {metric}',
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values)
            })
            
        # Prediction metrics
        for metric, value in self.metrics.prediction_metrics.items():
            report_data.append({
                'Metric': f'Prediction {metric}',
                'Mean': value,
                'Std': None,
                'Min': None,
                'Max': None
            })
            
        report = pd.DataFrame(report_data)
        
        # Save report
        report_path = Path(self.config.log_dir) / 'reports'
        report_path.mkdir(parents=True, exist_ok=True)
        
        report.to_csv(
            report_path / 
            f"{self.config.model_name}_{self.config.experiment_name}_report.csv",
            index=False
        )
        
        return report
