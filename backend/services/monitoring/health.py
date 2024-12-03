"""Health monitoring for services."""

import psutil
import torch
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path
import asyncio
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """System health metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    gpu_utilization: Optional[List[float]]
    gpu_memory_utilization: Optional[List[float]]
    model_cache_size: int
    active_requests: int
    error_rate: float
    average_latency: float
    uptime_seconds: float

class HealthMonitor:
    """Service health monitoring."""
    
    def __init__(
        self,
        metrics_path: Path,
        history_size: int = 1000,
        alert_threshold: float = 0.9
    ):
        """Initialize health monitor.
        
        Args:
            metrics_path: Path to store metrics
            history_size: Number of metrics to keep in history
            alert_threshold: Threshold for alerts
        """
        self.metrics_path = metrics_path
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.history_size = history_size
        self.alert_threshold = alert_threshold
        
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.latency_history: List[float] = []
        self.metrics_history: List[Dict] = []
        
        # Initialize GPU monitoring if available
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.num_gpus = torch.cuda.device_count()
        
    def record_request(self, latency: float, error: bool = False):
        """Record API request metrics.
        
        Args:
            latency: Request latency in seconds
            error: Whether request resulted in error
        """
        self.request_count += 1
        if error:
            self.error_count += 1
            
        self.latency_history.append(latency)
        if len(self.latency_history) > self.history_size:
            self.latency_history.pop(0)
            
    def get_metrics(self) -> HealthMetrics:
        """Get current health metrics.
        
        Returns:
            Health metrics
        """
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_utilization = None
        gpu_memory = None
        if self.gpu_available:
            gpu_utilization = []
            gpu_memory = []
            for i in range(self.num_gpus):
                gpu_utilization.append(
                    float(torch.cuda.utilization(i))
                )
                gpu_memory.append(
                    float(torch.cuda.memory_allocated(i) /
                          torch.cuda.max_memory_allocated(i))
                )
                
        # Calculate error rate and latency
        total_requests = max(self.request_count, 1)
        error_rate = self.error_count / total_requests
        
        avg_latency = (
            np.mean(self.latency_history)
            if self.latency_history else 0.0
        )
        
        # Calculate uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return HealthMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=disk.percent,
            gpu_utilization=gpu_utilization,
            gpu_memory_utilization=gpu_memory,
            model_cache_size=0,  # To be updated by model serving
            active_requests=len(self.latency_history),
            error_rate=error_rate,
            average_latency=avg_latency,
            uptime_seconds=uptime
        )
        
    def check_health(self) -> tuple[bool, List[str]]:
        """Check system health.
        
        Returns:
            Tuple of (healthy status, list of warnings)
        """
        metrics = self.get_metrics()
        warnings = []
        
        # Check CPU usage
        if metrics.cpu_percent > self.alert_threshold * 100:
            warnings.append(
                f"High CPU usage: {metrics.cpu_percent:.1f}%"
            )
            
        # Check memory usage
        if metrics.memory_percent > self.alert_threshold * 100:
            warnings.append(
                f"High memory usage: {metrics.memory_percent:.1f}%"
            )
            
        # Check disk usage
        if metrics.disk_usage_percent > self.alert_threshold * 100:
            warnings.append(
                f"High disk usage: {metrics.disk_usage_percent:.1f}%"
            )
            
        # Check GPU usage
        if metrics.gpu_utilization:
            for i, util in enumerate(metrics.gpu_utilization):
                if util > self.alert_threshold * 100:
                    warnings.append(
                        f"High GPU {i} utilization: {util:.1f}%"
                    )
                    
        # Check error rate
        if metrics.error_rate > self.alert_threshold:
            warnings.append(
                f"High error rate: {metrics.error_rate:.1%}"
            )
            
        # Check latency
        if metrics.average_latency > 1.0:  # 1 second threshold
            warnings.append(
                f"High average latency: {metrics.average_latency:.3f}s"
            )
            
        return len(warnings) == 0, warnings
        
    def save_metrics(self):
        """Save current metrics to file."""
        metrics = self.get_metrics()
        timestamp = datetime.now().isoformat()
        
        metrics_dict = {
            "timestamp": timestamp,
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "disk_usage_percent": metrics.disk_usage_percent,
            "gpu_utilization": metrics.gpu_utilization,
            "gpu_memory_utilization": metrics.gpu_memory_utilization,
            "model_cache_size": metrics.model_cache_size,
            "active_requests": metrics.active_requests,
            "error_rate": metrics.error_rate,
            "average_latency": metrics.average_latency,
            "uptime_seconds": metrics.uptime_seconds
        }
        
        self.metrics_history.append(metrics_dict)
        if len(self.metrics_history) > self.history_size:
            self.metrics_history.pop(0)
            
        # Save to file
        metrics_file = self.metrics_path / f"metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=2)
            
    async def monitor_loop(self, interval_seconds: float = 60):
        """Continuous monitoring loop.
        
        Args:
            interval_seconds: Interval between checks
        """
        while True:
            try:
                healthy, warnings = self.check_health()
                if not healthy:
                    for warning in warnings:
                        logger.warning(warning)
                        
                self.save_metrics()
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {str(e)}")
                
            await asyncio.sleep(interval_seconds)
            
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Get metrics history.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of metrics
        """
        if not start_time:
            start_time = datetime.min
        if not end_time:
            end_time = datetime.max
            
        return [
            metrics for metrics in self.metrics_history
            if start_time <= datetime.fromisoformat(metrics["timestamp"]) <= end_time
        ]
        
    def get_summary_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """Get summary statistics.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Summary statistics
        """
        metrics = self.get_metrics_history(start_time, end_time)
        if not metrics:
            return {}
            
        summary = {}
        for key in metrics[0].keys():
            if key == "timestamp":
                continue
                
            values = [m[key] for m in metrics if m[key] is not None]
            if not values:
                continue
                
            if isinstance(values[0], (int, float)):
                summary[key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
                
        return summary
