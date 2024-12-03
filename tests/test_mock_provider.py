import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import random

from backend.services.market_data.mock_provider import MockProvider
from backend.services.market_data.base import MarketDataConfig, MarketDataCredentials

@pytest.fixture
def provider():
    """Create a mock provider instance for testing"""
    credentials = MarketDataCredentials(api_key="test_key")
    config = MarketDataConfig(
        credentials=credentials,
        base_url="http://mock.test",
        websocket_url="ws://mock.test"
    )
    return MockProvider(config)

@pytest.fixture(autouse=True)
async def cleanup(provider):
    """Cleanup fixture to ensure provider is disconnected after each test"""
    yield
    if provider.connected:
        await provider.disconnect()

@pytest.mark.asyncio
async def test_connect_disconnect(provider):
    """Test connection management"""
    assert not provider.connected
    await provider.connect()
    assert provider.connected
    await provider.disconnect()
    assert not provider.connected

@pytest.mark.asyncio
async def test_subscribe_unsubscribe(provider):
    """Test subscription management"""
    await provider.connect()
    
    # Test subscribe
    symbols = ["AAPL", "GOOGL"]
    await provider.subscribe(symbols)
    assert all(symbol in provider.subscribed_symbols for symbol in symbols)
    assert provider._stream_task is not None
    
    # Test unsubscribe
    await provider.unsubscribe(["AAPL"])
    assert "AAPL" not in provider.subscribed_symbols
    assert "GOOGL" in provider.subscribed_symbols
    
    # Test unsubscribe all
    await provider.unsubscribe(["GOOGL"])
    assert len(provider.subscribed_symbols) == 0
    assert provider._stream_task is None

@pytest.mark.asyncio
async def test_subscribe_not_connected(provider):
    """Test subscription when not connected"""
    with pytest.raises(RuntimeError, match="Not connected to market data provider"):
        await provider.subscribe(["AAPL"])

@pytest.mark.asyncio
async def test_historical_data_generation(provider):
    """Test historical data generation with different intervals"""
    await provider.connect()
    
    start_date = datetime.now() - timedelta(days=10)
    end_date = datetime.now()
    
    intervals = ["1min", "5min", "1h", "1d"]
    for interval in intervals:
        df = await provider.get_historical_data("AAPL", start_date, end_date, interval)
        
        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Verify data integrity
        assert len(df) > 0
        assert (df['high'] >= df['low']).all()
        assert (df['volume'] > 0).all()
        assert df.index.is_monotonic_increasing

@pytest.mark.asyncio
async def test_invalid_interval(provider):
    """Test handling of invalid intervals"""
    await provider.connect()
    
    start_date = datetime.now() - timedelta(days=1)
    with pytest.raises(ValueError, match="Unsupported interval: 2min"):
        await provider.get_historical_data("AAPL", start_date, interval="2min")

@pytest.mark.asyncio
async def test_latest_quote(provider):
    """Test latest quote generation"""
    await provider.connect()
    
    quote = await provider.get_latest_quote("AAPL")
    
    # Verify quote structure
    assert isinstance(quote, dict)
    assert all(key in quote for key in ['symbol', 'timestamp', 'bid', 'ask', 'last', 'volume'])
    
    # Verify data integrity
    assert quote['symbol'] == "AAPL"
    assert quote['bid'] < quote['ask']
    assert quote['volume'] > 0
    assert isinstance(quote['timestamp'], datetime)

@pytest.mark.asyncio
async def test_stream_market_data(provider):
    """Test market data streaming"""
    await provider.connect()
    
    # Setup data reception
    received_data = []
    async def on_data(data):
        received_data.append(data)
    provider.on_data = on_data
    
    # Subscribe and wait for some data
    await provider.subscribe(["AAPL"])
    await asyncio.sleep(0.5)  # Wait for some data to arrive
    
    # Verify streaming data
    assert len(received_data) > 0
    for data in received_data:
        assert 'symbol' in data
        assert 'price' in data
        assert 'timestamp' in data

@pytest.mark.asyncio
async def test_historical_data_date_validation(provider):
    """Test historical data date validation"""
    await provider.connect()
    
    end_date = datetime.now()
    start_date = end_date + timedelta(days=1)  # Start date after end date
    
    df = await provider.get_historical_data("AAPL", start_date, end_date)
    assert len(df) == 0  # Should return empty DataFrame for invalid date range

@pytest.mark.asyncio
async def test_multiple_subscriptions(provider):
    """Test handling multiple subscriptions"""
    await provider.connect()
    
    # Subscribe in multiple batches
    await provider.subscribe(["AAPL"])
    await provider.subscribe(["GOOGL"])
    assert len(provider.subscribed_symbols) == 2
    
    # Subscribe to already subscribed symbol
    await provider.subscribe(["AAPL"])
    assert len(provider.subscribed_symbols) == 2  # Should not duplicate

@pytest.mark.asyncio
async def test_stress_historical_data(provider):
    """Test historical data generation with large date ranges"""
    await provider.connect()
    
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    df = await provider.get_historical_data("AAPL", start_date, end_date, interval="1d")
    assert len(df) > 250  # Approximately one year of trading days

@pytest.mark.asyncio
async def test_connect_error_handling(provider):
    """Test error handling during connection"""
    # Simulate already connected state
    provider.connected = True
    with pytest.raises(RuntimeError, match="Already connected"):
        await provider.connect()

@pytest.mark.asyncio
async def test_disconnect_error_handling(provider):
    """Test error handling during disconnection"""
    # Try disconnecting when not connected
    with pytest.raises(RuntimeError, match="Not connected"):
        await provider.disconnect()

@pytest.mark.asyncio
async def test_empty_historical_data(provider):
    """Test historical data with no results"""
    await provider.connect()
    
    # Test with start date after end date
    end_date = datetime.now() - timedelta(days=10)
    start_date = end_date + timedelta(days=1)
    
    df = await provider.get_historical_data("AAPL", start_date, end_date)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

@pytest.mark.asyncio
async def test_historical_data_edge_cases(provider):
    """Test historical data edge cases"""
    await provider.connect()
    
    # Test with very short time range
    end_date = datetime.now()
    start_date = end_date - timedelta(minutes=1)
    
    # Test 1min interval
    df = await provider.get_historical_data("AAPL", start_date, end_date, interval="1min")
    assert len(df) <= 2  # Should have at most 2 data points
    
    # Test daily interval with short range
    df = await provider.get_historical_data("AAPL", start_date, end_date, interval="1d")
    assert len(df) <= 1  # Should have at most 1 data point

@pytest.mark.asyncio
async def test_get_latest_quote_not_connected(provider):
    """Test getting quote when not connected"""
    with pytest.raises(RuntimeError, match="Not connected"):
        await provider.get_latest_quote("AAPL")

@pytest.mark.asyncio
async def test_subscribe_edge_cases(provider):
    """Test subscription edge cases"""
    await provider.connect()
    
    # Test subscribing to empty list
    await provider.subscribe([])
    assert len(provider.subscribed_symbols) == 0
    
    # Test unsubscribing from non-subscribed symbol
    await provider.unsubscribe(["NONEXISTENT"])
    assert len(provider.subscribed_symbols) == 0
    
    # Test multiple subscribe/unsubscribe operations
    await provider.subscribe(["AAPL", "GOOGL"])
    await provider.subscribe(["AAPL"])  # Duplicate subscribe
    assert len(provider.subscribed_symbols) == 2
    
    await provider.unsubscribe(["AAPL"])
    await provider.unsubscribe(["AAPL"])  # Duplicate unsubscribe
    assert len(provider.subscribed_symbols) == 1

@pytest.mark.asyncio
async def test_historical_data_validation(provider):
    """Test historical data validation edge cases"""
    await provider.connect()
    
    # Test with None end_date
    start_date = datetime.now() - timedelta(days=1)
    df = await provider.get_historical_data("AAPL", start_date, None)
    assert len(df) > 0
    
    # Test with equal start and end dates
    now = datetime.now()
    df = await provider.get_historical_data("AAPL", now, now)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 1

@pytest.mark.asyncio
async def test_stream_data_callback(provider):
    """Test streaming data callback handling"""
    await provider.connect()
    
    received_data = []
    async def on_data(data):
        received_data.append(data)
    
    # Set callback and subscribe
    provider.on_data = on_data
    await provider.subscribe(["AAPL"])
    
    # Wait for some data
    await asyncio.sleep(0.3)
    
    # Verify data structure
    assert len(received_data) > 0
    for data in received_data:
        assert 'symbol' in data
        assert 'timestamp' in data
        assert 'price' in data
        assert 'volume' in data
        assert data['symbol'] == "AAPL"

@pytest.mark.asyncio
async def test_metadata_handling(provider):
    """Test metadata handling in market updates"""
    await provider.connect()
    
    received_data = []
    async def on_data(data):
        received_data.append(data)
    
    provider.on_data = on_data
    await provider.subscribe(["AAPL"])
    
    # Wait for some data
    await asyncio.sleep(0.3)
    
    # Verify metadata in updates
    assert len(received_data) > 0
    for data in received_data:
        assert isinstance(data, dict)
        assert 'metadata' in data
        assert isinstance(data['metadata'], dict)
        assert 'timestamp' in data['metadata']
        assert 'source' in data['metadata']
        assert data['metadata']['source'] == 'mock'

@pytest.mark.asyncio
async def test_streaming_error_handling(provider):
    """Test error handling during streaming"""
    await provider.connect()
    
    error_received = False
    async def on_data(data):
        if random.random() < 0.5:  # Simulate random errors
            raise RuntimeError("Simulated callback error")
        nonlocal error_received
        error_received = True
    
    provider.on_data = on_data
    await provider.subscribe(["AAPL"])
    
    # Wait for some data
    await asyncio.sleep(0.5)
    
    # Verify streaming continues despite errors
    assert error_received
    assert provider._stream_task is not None
    assert not provider._stream_task.done()

@pytest.mark.asyncio
async def test_concurrent_subscription(provider):
    """Test concurrent subscription/unsubscription"""
    await provider.connect()
    
    # Create multiple concurrent subscriptions
    symbols = [f"SYMBOL{i}" for i in range(10)]
    await asyncio.gather(
        *[provider.subscribe([symbol]) for symbol in symbols[:5]],
        *[provider.unsubscribe([symbol]) for symbol in symbols[5:]]
    )
    
    # Verify subscription state
    assert len(provider.subscribed_symbols) == 5
    assert all(symbol in provider.subscribed_symbols for symbol in symbols[:5])

@pytest.mark.asyncio
async def test_reconnection(provider):
    """Test reconnection behavior"""
    await provider.connect()
    await provider.subscribe(["AAPL"])
    
    # Force disconnect and reconnect
    await provider.disconnect()
    assert not provider.connected
    assert provider._stream_task is None
    
    # Reconnect
    await provider.connect()
    assert provider.connected
    
    # Verify subscription state is maintained
    assert "AAPL" in provider.subscribed_symbols
    await asyncio.sleep(0.2)
    assert provider._stream_task is not None

@pytest.mark.asyncio
async def test_rate_limiting(provider):
    """Test rate limiting simulation"""
    await provider.connect()

    # Make multiple rapid requests
    start_time = datetime.now()
    quotes = []
    for _ in range(10):
        quote = await provider.get_latest_quote("AAPL")
        quotes.append(quote)
    end_time = datetime.now()

    # Verify rate limiting
    duration = (end_time - start_time).total_seconds()
    assert duration >= 0.1  # Should take at least 100ms due to rate limiting

    # Verify we got valid quotes
    assert len(quotes) == 10
    for quote in quotes:
        assert quote['symbol'] == 'AAPL'
        assert 'timestamp' in quote
        assert 'bid' in quote
        assert 'ask' in quote
        assert 'last' in quote
        assert 'volume' in quote

@pytest.mark.asyncio
async def test_subscription_lifecycle(provider):
    """Test complete subscription lifecycle with error conditions"""
    await provider.connect()
    
    # Test empty subscription list
    await provider.subscribe([])
    assert len(provider.subscribed_symbols) == 0
    assert provider._stream_task is None
    
    # Test single symbol subscription
    await provider.subscribe(["AAPL"])
    assert "AAPL" in provider.subscribed_symbols
    assert provider._stream_task is not None
    assert not provider._stream_task.done()
    
    # Test duplicate subscription
    await provider.subscribe(["AAPL"])
    assert len(provider.subscribed_symbols) == 1
    
    # Test multiple symbol subscription
    await provider.subscribe(["GOOGL", "MSFT"])
    assert all(symbol in provider.subscribed_symbols for symbol in ["AAPL", "GOOGL", "MSFT"])
    
    # Test partial unsubscribe
    await provider.unsubscribe(["AAPL"])
    assert "AAPL" not in provider.subscribed_symbols
    assert provider._stream_task is not None
    
    # Test empty unsubscribe
    await provider.unsubscribe([])
    assert len(provider.subscribed_symbols) == 2
    
    # Test complete unsubscribe
    await provider.unsubscribe(["GOOGL", "MSFT"])
    assert len(provider.subscribed_symbols) == 0
    assert provider._stream_task is None

@pytest.mark.asyncio
async def test_subscription_error_handling(provider):
    """Test error handling during subscription operations"""
    # Test subscribe before connect
    with pytest.raises(RuntimeError, match="Not connected to market data provider"):
        await provider.subscribe(["AAPL"])
    
    await provider.connect()
    
    # Test subscription with invalid symbols
    await provider.subscribe([""])  # Empty symbol
    assert len(provider.subscribed_symbols) == 0
    
    await provider.subscribe([None])  # None symbol
    assert len(provider.subscribed_symbols) == 0
    
    # Test rapid subscribe/unsubscribe
    tasks = []
    for _ in range(10):
        tasks.append(asyncio.create_task(provider.subscribe(["AAPL"])))
        tasks.append(asyncio.create_task(provider.unsubscribe(["AAPL"])))
    
    await asyncio.gather(*tasks)
    assert provider._stream_task is None

@pytest.mark.asyncio
async def test_stream_recovery(provider):
    """Test stream recovery after errors"""
    await provider.connect()
    await provider.subscribe(["AAPL"])
    
    # Simulate stream error
    original_task = provider._stream_task
    provider._stream_task.cancel()
    
    # Wait for recovery
    await asyncio.sleep(0.2)
    
    # Verify stream recovered
    assert provider._stream_task is not None
    assert provider._stream_task is not original_task
    assert not provider._stream_task.done()
    
    # Verify data still flowing
    received_data = False
    async def on_data(data):
        nonlocal received_data
        received_data = True
        assert data['symbol'] == "AAPL"
        
    provider.on_data = on_data
    await asyncio.sleep(0.2)
    assert received_data

@pytest.mark.asyncio
async def test_network_timeouts(provider):
    """Test handling of network timeouts and delays"""
    # Override sleep to simulate network delays
    original_sleep = asyncio.sleep
    
    async def delayed_sleep(delay):
        if delay == 0.1:  # Connection delay
            await original_sleep(2.0)  # Simulate longer network delay
        else:
            await original_sleep(delay)
            
    asyncio.sleep = delayed_sleep
    
    try:
        # Test connection timeout
        start_time = datetime.now()
        await provider.connect()
        connection_time = (datetime.now() - start_time).total_seconds()
        assert connection_time >= 2.0, "Connection should respect simulated delay"
        
        # Test subscription with delays
        await provider.subscribe(["AAPL"])
        
        # Test data streaming with delays
        received_data = False
        async def delayed_callback(data):
            nonlocal received_data
            await original_sleep(1.0)  # Simulate slow processing
            received_data = True
            
        provider.on_data = delayed_callback
        await original_sleep(0.3)  # Wait for some data
        assert received_data, "Should receive data despite processing delays"
        
        # Test rate limiting with delays
        quotes = []
        for _ in range(5):
            quote = await provider.get_latest_quote("AAPL")
            quotes.append(quote)
        
        # Verify timestamps respect rate limiting
        for i in range(1, len(quotes)):
            time_diff = (quotes[i]['timestamp'] - quotes[i-1]['timestamp']).total_seconds()
            assert time_diff >= provider._min_request_interval, "Rate limiting should be respected"
            
    finally:
        # Restore original sleep function
        asyncio.sleep = original_sleep

@pytest.mark.asyncio
async def test_slow_consumer(provider):
    """Test handling of slow consumer scenarios"""
    await provider.connect()
    
    # Create a queue to track received data
    data_queue = asyncio.Queue()
    
    async def slow_callback(data):
        await asyncio.sleep(0.5)  # Simulate slow processing
        await data_queue.put(data)
    
    provider.on_data = slow_callback
    
    # Subscribe to multiple symbols to generate more data
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    await provider.subscribe(symbols)
    
    # Wait for some data to accumulate
    await asyncio.sleep(2.0)
    
    # Verify data handling
    queue_size = data_queue.qsize()
    assert queue_size > 0, "Should receive some data"
    
    # Process accumulated data
    received_symbols = set()
    while not data_queue.empty():
        data = await data_queue.get()
        received_symbols.add(data['symbol'])
    
    # Verify we received data for all symbols
    assert received_symbols.issubset(set(symbols)), "Should receive data for subscribed symbols"
    
@pytest.mark.asyncio
async def test_connection_timeout_recovery(provider):
    """Test recovery from connection timeouts"""
    # Override connect to simulate timeout
    original_connect = provider.connect
    connect_attempts = 0
    
    async def connect_with_timeout():
        nonlocal connect_attempts
        connect_attempts += 1
        if connect_attempts == 1:
            await asyncio.sleep(2.0)  # Simulate timeout on first attempt
            raise TimeoutError("Connection timeout")
        await original_connect()
    
    provider.connect = connect_with_timeout
    
    try:
        # First attempt should timeout
        with pytest.raises(TimeoutError):
            await provider.connect()
        
        # Second attempt should succeed
        await provider.connect()
        assert provider.connected
        
        # Verify functionality after recovery
        await provider.subscribe(["AAPL"])
        quote = await provider.get_latest_quote("AAPL")
        assert quote['symbol'] == "AAPL"
        
    finally:
        # Restore original connect method
        provider.connect = original_connect

@pytest.mark.asyncio
async def test_symbol_limits(provider):
    """Test handling of symbol subscription limits"""
    await provider.connect()
    
    # Test maximum symbol limit
    max_symbols = provider._max_symbols
    symbols = [f"SYMBOL{i}" for i in range(max_symbols + 1)]
    
    # First batch should succeed
    await provider.subscribe(symbols[:max_symbols])
    assert len(provider.subscribed_symbols) == max_symbols
    
    # Adding one more should fail
    with pytest.raises(ValueError, match="Maximum symbol limit.*exceeded"):
        await provider.subscribe(symbols[max_symbols:])
        
    # Verify metadata tracking
    for symbol in provider.subscribed_symbols:
        stats = await provider.get_symbol_stats(symbol)
        assert stats['status'] == 'active'
        assert stats['error_count'] == 0

@pytest.mark.asyncio
async def test_invalid_symbols(provider):
    """Test handling of invalid symbol inputs"""
    await provider.connect()
    
    invalid_symbols = [
        "",  # Empty string
        None,  # None value
        123,  # Non-string
        "   ",  # Whitespace
        "A" * 100,  # Too long
        "!@#$",  # Invalid characters
    ]
    
    # Test each invalid symbol
    for symbol in invalid_symbols:
        try:
            await provider.subscribe([symbol])
        except (ValueError, TypeError):
            pass  # Expected failure
            
    assert len(provider.subscribed_symbols) == 0

@pytest.mark.asyncio
async def test_error_injection(provider):
    """Test error injection and recovery"""
    await provider.connect()
    
    # Set up error tracking
    errors_received = []
    async def error_tracking_callback(data):
        nonlocal errors_received
        errors_received.append(data)
    
    provider.on_data = error_tracking_callback
    
    # Test with different error rates
    for error_rate in [0.2, 0.5, 0.8]:
        provider.inject_error_rate(error_rate)
        await provider.subscribe(["AAPL"])
        
        try:
            # Run for a short period
            await asyncio.sleep(1.0)
        except (TimeoutError, ConnectionError, RuntimeError):
            pass  # Expected errors
            
        # Verify error injection
        stats = await provider.get_symbol_stats("AAPL")
        assert stats['error_count'] > 0
        
        # Clean up
        await provider.unsubscribe(["AAPL"])
        provider.inject_error_rate(0.0)

@pytest.mark.asyncio
async def test_backpressure_handling(provider):
    """Test handling of backpressure in data streaming"""
    await provider.connect()
    
    # Create a slow consumer
    received_data = []
    async def slow_consumer(data):
        await asyncio.sleep(0.2)  # Simulate slow processing
        received_data.append(data)
    
    provider.on_data = slow_consumer
    
    # Subscribe to multiple symbols to generate load
    symbols = [f"TEST{i}" for i in range(5)]
    await provider.subscribe(symbols)
    
    # Run for a period to accumulate data
    await asyncio.sleep(2.0)
    
    # Verify data handling
    for symbol in symbols:
        stats = await provider.get_symbol_stats(symbol)
        assert stats['request_count'] > 0
        
        # Some updates might be dropped due to backpressure
        if stats['error_count'] > 0:
            print(f"Backpressure detected for {symbol}: {stats['error_count']} dropped updates")

@pytest.mark.asyncio
async def test_metadata_tracking(provider):
    """Test symbol metadata tracking functionality"""
    await provider.connect()
    
    # Subscribe to a symbol
    symbol = "AAPL"
    await provider.subscribe([symbol])
    
    # Get initial stats
    initial_stats = await provider.get_symbol_stats(symbol)
    assert initial_stats['request_count'] == 0
    assert initial_stats['error_count'] == 0
    assert initial_stats['last_price'] is None
    
    # Wait for some updates
    await asyncio.sleep(0.5)
    
    # Get updated stats
    updated_stats = await provider.get_symbol_stats(symbol)
    assert updated_stats['request_count'] > 0
    assert updated_stats['last_price'] is not None
    
    # Test invalid symbol stats
    with pytest.raises(ValueError, match="Unknown symbol"):
        await provider.get_symbol_stats("INVALID")
