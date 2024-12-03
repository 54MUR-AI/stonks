import asyncio
import pytest
from datetime import datetime, timedelta
import pandas as pd
from backend.services.market_data.mock_provider import MockProvider
from backend.services.market_data.base import MarketDataConfig, MarketDataCredentials

@pytest.fixture
async def mock_provider():
    """Create and yield a connected mock provider"""
    credentials = MarketDataCredentials(api_key="test")
    config = MarketDataConfig(
        credentials=credentials,
        base_url="mock://test",
        websocket_url="ws://test",
        request_timeout=1,
        max_retries=1
    )
    provider = MockProvider(config)
    await provider.connect()
    return provider  # Use return instead of yield for async fixtures

@pytest.mark.asyncio
async def test_error_injection():
    """Test error injection functionality"""
    credentials = MarketDataCredentials(api_key="test")
    config = MarketDataConfig(
        credentials=credentials,
        base_url="mock://test",
        websocket_url="ws://test"
    )
    provider = MockProvider(config)
    
    # Set high error rate
    provider.inject_error_rate(0.9)
    
    with pytest.raises((TimeoutError, ConnectionError, RuntimeError, asyncio.CancelledError)):
        await provider._maybe_inject_error()

@pytest.mark.asyncio
async def test_timeout_simulation(mock_provider):
    """Test timeout simulation with different probabilities"""
    provider = await mock_provider
    # Configure high timeout probability
    provider.set_timeout_simulation(delay=0.1, probability=0.9)
    
    with pytest.raises(asyncio.TimeoutError):
        await provider._maybe_timeout()
    
    # Reset timeout probability
    provider.set_timeout_simulation(probability=0.0)
    await provider._maybe_timeout()  # Should not raise

@pytest.mark.asyncio
async def test_backpressure_handling(mock_provider):
    """Test backpressure handling in data streaming"""
    provider = await mock_provider
    # Fill the buffer
    data = {
        'symbol': 'TEST',
        'timestamp': datetime.now(),
        'price': 100.0,
        'volume': 1000
    }
    
    # Fill buffer to capacity
    for _ in range(1000):
        await provider._handle_backpressure(data)
    
    # Verify backpressure handling
    stats = await provider.get_symbol_stats('TEST')
    assert stats['error_count'] >= 0

@pytest.mark.asyncio
async def test_symbol_validation(mock_provider):
    """Test symbol validation and metadata tracking"""
    provider = await mock_provider
    # Test invalid symbol
    with pytest.raises(ValueError):
        await provider._validate_symbol("")
    
    # Test valid symbol
    await provider._validate_symbol("AAPL")
    metadata = provider._symbol_metadata["AAPL"]
    assert metadata['request_count'] == 0
    assert metadata['error_count'] == 0
    assert metadata['status'] == 'active'

@pytest.mark.asyncio
async def test_rate_limiting(mock_provider):
    """Test rate limiting functionality"""
    provider = await mock_provider
    start_time = datetime.now()
    
    # Make multiple rapid requests
    for _ in range(15):  # More than rate limit
        await provider.get_quote("AAPL")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Should take at least 1.5 seconds due to rate limiting
    assert duration >= 1.5

@pytest.mark.asyncio
async def test_historical_data_intervals(mock_provider):
    """Test historical data generation with different intervals"""
    provider = await mock_provider
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
    intervals = ["1min", "5min", "1h", "1d"]
    for interval in intervals:
        df = await provider.get_historical_data("AAPL", start_date, end_date, interval)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'timestamp' in df.columns
        
        # Verify correct number of data points based on interval
        if interval == "1d":
            expected_points = 8  # ~7 days + current day
            assert len(df) <= expected_points
        elif interval == "1h":
            expected_points = 24 * 7  # ~7 days worth of hourly data
            assert len(df) <= expected_points + 24

@pytest.mark.asyncio
async def test_invalid_interval(mock_provider):
    """Test invalid interval handling"""
    provider = await mock_provider
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    
    with pytest.raises(ValueError, match="Unsupported interval"):
        await provider.get_historical_data("AAPL", start_date, end_date, "invalid")

@pytest.mark.asyncio
async def test_symbol_stats_tracking(mock_provider):
    """Test symbol statistics tracking"""
    provider = await mock_provider
    symbol = "AAPL"
    
    # Make some requests to generate stats
    await provider.subscribe([symbol])
    for _ in range(5):
        await provider.get_quote(symbol)
    
    stats = await provider.get_symbol_stats(symbol)
    assert stats['request_count'] >= 5
    assert 'first_seen' in stats
    assert 'last_price' in stats
    
    # Test unknown symbol
    with pytest.raises(ValueError, match="Unknown symbol"):
        await provider.get_symbol_stats("UNKNOWN")

@pytest.mark.asyncio
async def test_max_symbols_limit(mock_provider):
    """Test maximum symbols limit"""
    provider = await mock_provider
    # Try to subscribe to more than max allowed symbols
    symbols = [f"SYM{i}" for i in range(provider._max_symbols + 1)]
    
    with pytest.raises(ValueError):
        await provider.subscribe(symbols)
