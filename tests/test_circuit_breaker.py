"""Tests for circuit breaker functionality."""

import asyncio
import pytest
from datetime import datetime, timedelta

from backend.services.market_data.circuit_breaker import CircuitBreaker, CircuitState

async def success_func():
    """Test function that succeeds"""
    return "success"

async def fail_func():
    """Test function that fails"""
    raise RuntimeError("Simulated failure")

@pytest.mark.asyncio
async def test_circuit_breaker_success():
    """Test successful function execution"""
    cb = CircuitBreaker()
    result = await cb.call(success_func)
    assert result == "success"
    assert cb.state == CircuitState.CLOSED
    
    metrics = cb.metrics
    assert metrics['total_successes'] == 1
    assert metrics['total_failures'] == 0
    assert metrics['failure_rate'] == 0.0
    assert metrics['uptime_percentage'] == 100.0

@pytest.mark.asyncio
async def test_circuit_breaker_failure():
    """Test circuit breaker opening on failures"""
    cb = CircuitBreaker(failure_threshold=2)
    
    # First failure
    with pytest.raises(RuntimeError):
        await cb.call(fail_func)
    assert cb.state == CircuitState.CLOSED
    
    # Second failure opens circuit
    with pytest.raises(RuntimeError):
        await cb.call(fail_func)
    assert cb.state == CircuitState.OPEN
    
    # Circuit is open, fast fail
    with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
        await cb.call(success_func)
        
    metrics = cb.metrics
    assert metrics['total_failures'] == 2
    assert metrics['failure_rate'] == 100.0
    assert metrics['current_state'] == CircuitState.OPEN.value

@pytest.mark.asyncio
async def test_circuit_breaker_recovery():
    """Test circuit breaker recovery"""
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=0.1,
        half_open_timeout=0.2
    )
    
    # Fail to open circuit
    with pytest.raises(RuntimeError):
        await cb.call(fail_func)
    assert cb.state == CircuitState.OPEN
    
    # Wait for recovery timeout
    await asyncio.sleep(0.15)
    
    # Circuit should go to half-open
    result = await cb.call(success_func)
    assert result == "success"
    assert cb.state == CircuitState.CLOSED
    
    metrics = cb.metrics
    assert len(metrics['state_changes']) == 2  # CLOSED->OPEN->CLOSED
    assert metrics['current_state'] == CircuitState.CLOSED.value

@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure():
    """Test circuit breaker returning to open on half-open failure"""
    cb = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=0.1,
        half_open_timeout=0.2
    )
    
    # Fail to open circuit
    with pytest.raises(RuntimeError):
        await cb.call(fail_func)
    assert cb.state == CircuitState.OPEN
    
    # Wait for recovery timeout
    await asyncio.sleep(0.15)
    
    # Fail in half-open state
    with pytest.raises(RuntimeError):
        await cb.call(fail_func)
    assert cb.state == CircuitState.OPEN
    
    metrics = cb.metrics
    assert metrics['total_failures'] == 2
    assert metrics['current_state'] == CircuitState.OPEN.value

@pytest.mark.asyncio
async def test_circuit_breaker_reset_timeout():
    """Test failure count reset after timeout"""
    cb = CircuitBreaker(
        failure_threshold=2,
        reset_timeout=0.1
    )
    
    # First failure
    with pytest.raises(RuntimeError):
        await cb.call(fail_func)
    assert cb.state == CircuitState.CLOSED
    
    # Wait for reset timeout
    await asyncio.sleep(0.15)
    
    # Failure count should be reset
    with pytest.raises(RuntimeError):
        await cb.call(fail_func)
    assert cb.state == CircuitState.CLOSED  # Still need one more failure
    
    metrics = cb.metrics
    assert metrics['total_failures'] == 2
    assert metrics['current_state'] == CircuitState.CLOSED.value
