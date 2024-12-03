import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta

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
