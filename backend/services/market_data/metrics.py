"""Performance metrics collection for market data providers."""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from collections import deque

@dataclass
class LatencyMetrics:
    """Latency metrics for operations"""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    @property
    def avg_time(self) -> float:
        """Calculate average latency"""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def median_time(self) -> float:
        """Calculate median latency"""
        return median(self.times) if self.times else 0.0
    
    @property
    def stddev_time(self) -> float:
        """Calculate standard deviation of latency"""
        return stdev(self.times) if len(self.times) > 1 else 0.0
    
    def add_time(self, duration: float) -> None:
        """Add a new latency measurement"""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.times.append(duration)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            'count': self.count,
            'total_time': self.total_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'avg_time': self.avg_time,
            'median_time': self.median_time,
            'stddev_time': self.stddev_time
        }

class ProviderMetrics:
    """Metrics collector for market data providers"""
    
    def __init__(self, window_size: int = 3600):
        """Initialize metrics collector.
        
        Args:
            window_size: Time window in seconds for metrics collection
        """
        self._window_size = window_size
        self._start_time = datetime.now()
        
        # Operation latency metrics
        self._latency_metrics: Dict[str, LatencyMetrics] = {
            'connect': LatencyMetrics(),
            'disconnect': LatencyMetrics(),
            'subscribe': LatencyMetrics(),
            'unsubscribe': LatencyMetrics(),
            'get_quote': LatencyMetrics(),
            'get_historical': LatencyMetrics()
        }
        
        # Success/failure metrics
        self._success_count = 0
        self._error_count = 0
        self._error_types: Dict[str, int] = {}
        
        # Request rate metrics
        self._request_times: deque = deque(maxlen=10000)
        self._request_counts: Dict[str, int] = {}
        
        # Symbol metrics
        self._symbol_metrics: Dict[str, Dict[str, Any]] = {}
        
    def record_latency(self, operation: str, duration: float) -> None:
        """Record latency for an operation.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        if operation in self._latency_metrics:
            self._latency_metrics[operation].add_time(duration)
            
    def record_request(self, operation: str, success: bool = True) -> None:
        """Record a request.
        
        Args:
            operation: Name of the operation
            success: Whether the request was successful
        """
        now = datetime.now()
        self._request_times.append(now)
        self._request_counts[operation] = self._request_counts.get(operation, 0) + 1
        
        if success:
            self._success_count += 1
        else:
            self._error_count += 1
            
    def record_error(self, error_type: str) -> None:
        """Record an error.
        
        Args:
            error_type: Type of error that occurred
        """
        self._error_types[error_type] = self._error_types.get(error_type, 0) + 1
        
    def record_symbol_update(self, symbol: str, price: float, volume: int) -> None:
        """Record symbol metrics.
        
        Args:
            symbol: Symbol being updated
            price: Current price
            volume: Current volume
        """
        if symbol not in self._symbol_metrics:
            self._symbol_metrics[symbol] = {
                'updates': 0,
                'first_seen': datetime.now(),
                'prices': deque(maxlen=100),
                'volumes': deque(maxlen=100)
            }
            
        metrics = self._symbol_metrics[symbol]
        metrics['updates'] += 1
        metrics['last_update'] = datetime.now()
        metrics['last_price'] = price
        metrics['last_volume'] = volume
        metrics['prices'].append(price)
        metrics['volumes'].append(volume)
        
    def get_request_rate(self, window: int = 60) -> float:
        """Calculate request rate over a time window.
        
        Args:
            window: Time window in seconds
            
        Returns:
            Requests per second over the window
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=window)
        
        # Count requests in window
        count = sum(1 for t in self._request_times if t >= cutoff)
        return count / window if window > 0 else 0.0
        
    def get_error_rate(self) -> float:
        """Calculate error rate.
        
        Returns:
            Percentage of requests that resulted in errors
        """
        total = self._success_count + self._error_count
        return (self._error_count / total * 100) if total > 0 else 0.0
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics.
        
        Returns:
            Dictionary of metrics
        """
        now = datetime.now()
        uptime = (now - self._start_time).total_seconds()
        
        return {
            'uptime': uptime,
            'request_rate': self.get_request_rate(),
            'error_rate': self.get_error_rate(),
            'total_requests': self._success_count + self._error_count,
            'success_count': self._success_count,
            'error_count': self._error_count,
            'error_types': self._error_types.copy(),
            'request_counts': self._request_counts.copy(),
            'latency': {
                op: metrics.to_dict()
                for op, metrics in self._latency_metrics.items()
            },
            'symbols': {
                symbol: {
                    'updates': metrics['updates'],
                    'first_seen': metrics['first_seen'],
                    'last_update': metrics.get('last_update'),
                    'last_price': metrics.get('last_price'),
                    'last_volume': metrics.get('last_volume'),
                    'price_stats': {
                        'min': min(metrics['prices']) if metrics['prices'] else None,
                        'max': max(metrics['prices']) if metrics['prices'] else None,
                        'avg': mean(metrics['prices']) if metrics['prices'] else None,
                        'median': median(metrics['prices']) if metrics['prices'] else None
                    } if metrics['prices'] else None,
                    'volume_stats': {
                        'min': min(metrics['volumes']) if metrics['volumes'] else None,
                        'max': max(metrics['volumes']) if metrics['volumes'] else None,
                        'avg': mean(metrics['volumes']) if metrics['volumes'] else None,
                        'median': median(metrics['volumes']) if metrics['volumes'] else None
                    } if metrics['volumes'] else None
                }
                for symbol, metrics in self._symbol_metrics.items()
            }
        }
