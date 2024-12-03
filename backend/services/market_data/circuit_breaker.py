"""Circuit breaker implementation for market data providers."""

import asyncio
import time
from enum import Enum
from typing import Optional, Callable, Any, Dict
from datetime import datetime, timedelta

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"      # Failing, reject fast
    HALF_OPEN = "HALF_OPEN"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for handling provider failures"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_timeout: float = 30.0,
        reset_timeout: float = 300.0
    ):
        """Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            half_open_timeout: Time in seconds to test recovery
            reset_timeout: Time in seconds before resetting failure count
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_timeout = half_open_timeout
        self._reset_timeout = reset_timeout
        self._metrics: Dict[str, Any] = {
            'total_failures': 0,
            'total_successes': 0,
            'last_failure': None,
            'last_success': None,
            'current_state': self._state.value,
            'failure_rate': 0.0,
            'uptime_percentage': 100.0,
            'state_changes': [],
        }
        self._state_change_time = datetime.now()
        
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
        
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return self._metrics.copy()
        
    def _update_metrics(self, success: bool) -> None:
        """Update circuit breaker metrics"""
        now = datetime.now()
        if success:
            self._metrics['total_successes'] += 1
            self._metrics['last_success'] = now
        else:
            self._metrics['total_failures'] += 1
            self._metrics['last_failure'] = now
            
        total = self._metrics['total_successes'] + self._metrics['total_failures']
        if total > 0:
            self._metrics['failure_rate'] = (
                self._metrics['total_failures'] / total * 100
            )
            
        # Calculate uptime percentage
        if self._state_change_time:
            total_time = (now - self._state_change_time).total_seconds()
            if total_time > 0:
                closed_time = sum(
                    end - start for start, end, state in self._metrics['state_changes']
                    if state == CircuitState.CLOSED
                ).total_seconds()
                self._metrics['uptime_percentage'] = (closed_time / total_time) * 100
                
    def _record_state_change(self, new_state: CircuitState) -> None:
        """Record a state change in metrics"""
        now = datetime.now()
        self._metrics['state_changes'].append(
            (self._state_change_time, now, self._state)
        )
        self._state_change_time = now
        self._state = new_state
        self._metrics['current_state'] = new_state.value
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func if successful
            
        Raises:
            Exception: If circuit is open or function fails
        """
        now = datetime.now()
        
        # Check if we should reset failure count
        if (self._last_failure_time and 
            (now - self._last_failure_time).total_seconds() >= self._reset_timeout):
            self._failure_count = 0
            
        # Handle different circuit states
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if (self._last_failure_time and 
                (now - self._last_failure_time).total_seconds() >= self._recovery_timeout):
                self._record_state_change(CircuitState.HALF_OPEN)
            else:
                self._update_metrics(success=False)
                raise RuntimeError("Circuit breaker is OPEN")
                
        elif self._state == CircuitState.HALF_OPEN:
            # Check if we should reopen circuit
            if (self._last_success_time and 
                (now - self._last_success_time).total_seconds() >= self._half_open_timeout):
                self._record_state_change(CircuitState.OPEN)
                self._update_metrics(success=False)
                raise RuntimeError("Circuit breaker returned to OPEN state")
                
        try:
            # Execute the protected function
            result = await func(*args, **kwargs)
            
            # Handle success
            self._last_success_time = now
            self._update_metrics(success=True)
            
            # Close circuit if in half-open state
            if self._state == CircuitState.HALF_OPEN:
                self._record_state_change(CircuitState.CLOSED)
                self._failure_count = 0
                
            return result
            
        except Exception as e:
            # Handle failure
            self._last_failure_time = now
            self._failure_count += 1
            self._update_metrics(success=False)
            
            # Open circuit if threshold reached
            if (self._state == CircuitState.CLOSED and 
                self._failure_count >= self._failure_threshold):
                self._record_state_change(CircuitState.OPEN)
                
            raise e  # Re-raise the original exception
