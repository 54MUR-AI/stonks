"""Tests for market data streaming functionality."""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
from aiohttp import WSMessage, WSMsgType

from backend.services.market_data.streaming import (
    MarketDataStream,
    StreamType,
    StreamEvent,
    StreamStatus,
    MarketDataError
)

class MockWebSocket:
    """Mock WebSocket for testing."""
    def __init__(self):
        self.messages = []
        self.closed = False
        self.exception = None
    
    async def send_json(self, data):
        self.messages.append(data)
    
    async def close(self):
        self.closed = True
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.exception:
            raise self.exception
        raise StopAsyncIteration

@pytest.fixture
def mock_provider():
    provider = Mock()
    provider.create_websocket = AsyncMock()
    return provider

@pytest.fixture
def mock_ws():
    return MockWebSocket()

@pytest.fixture
async def stream(mock_provider):
    stream = MarketDataStream(
        provider=mock_provider,
        stream_type=StreamType.TRADES,
        reconnect_interval=0.1,
        max_reconnect_attempts=2
    )
    yield stream
    await stream.disconnect()

@pytest.mark.asyncio
async def test_stream_initialization(stream):
    """Test stream initialization."""
    assert stream.status == StreamStatus.INITIALIZING
    assert stream.stream_type == StreamType.TRADES
    assert not stream._subscriptions
    assert stream._ws is None

@pytest.mark.asyncio
async def test_stream_connect(stream, mock_provider, mock_ws):
    """Test stream connection."""
    mock_provider.create_websocket.return_value.__aenter__.return_value = mock_ws
    
    # Track connection events
    connected = False
    def on_connect(event, data):
        nonlocal connected
        connected = True
    
    stream.on_event(StreamEvent.CONNECTED, on_connect)
    await stream.connect()
    
    assert connected
    assert stream.status == StreamStatus.CONNECTED
    assert stream._ws is not None
    assert not stream._closing

@pytest.mark.asyncio
async def test_stream_disconnect(stream, mock_provider, mock_ws):
    """Test stream disconnection."""
    mock_provider.create_websocket.return_value.__aenter__.return_value = mock_ws
    
    # Track disconnection events
    disconnected = False
    def on_disconnect(event, data):
        nonlocal disconnected
        disconnected = True
    
    stream.on_event(StreamEvent.DISCONNECTED, on_disconnect)
    
    await stream.connect()
    await stream.disconnect()
    
    assert disconnected
    assert stream.status == StreamStatus.DISCONNECTED
    assert mock_ws.closed
    assert stream._closing

@pytest.mark.asyncio
async def test_stream_subscribe(stream, mock_provider, mock_ws):
    """Test symbol subscription."""
    mock_provider.create_websocket.return_value.__aenter__.return_value = mock_ws
    
    # Track subscription events
    subscribed_symbols = []
    def on_subscribe(event, data):
        nonlocal subscribed_symbols
        subscribed_symbols.extend(data["symbols"])
    
    stream.on_event(StreamEvent.SUBSCRIBED, on_subscribe)
    
    await stream.connect()
    await stream.subscribe(["AAPL", "GOOGL"])
    
    assert "AAPL" in stream._subscriptions
    assert "GOOGL" in stream._subscriptions
    assert len(mock_ws.messages) == 1
    assert mock_ws.messages[0]["action"] == "subscribe"
    assert all(s in subscribed_symbols for s in ["AAPL", "GOOGL"])

@pytest.mark.asyncio
async def test_stream_unsubscribe(stream, mock_provider, mock_ws):
    """Test symbol unsubscription."""
    mock_provider.create_websocket.return_value.__aenter__.return_value = mock_ws
    
    # Track unsubscription events
    unsubscribed_symbols = []
    def on_unsubscribe(event, data):
        nonlocal unsubscribed_symbols
        unsubscribed_symbols.extend(data["symbols"])
    
    stream.on_event(StreamEvent.UNSUBSCRIBED, on_unsubscribe)
    
    await stream.connect()
    await stream.subscribe(["AAPL", "GOOGL"])
    await stream.unsubscribe(["AAPL"])
    
    assert "AAPL" not in stream._subscriptions
    assert "GOOGL" in stream._subscriptions
    assert len(mock_ws.messages) == 2
    assert mock_ws.messages[1]["action"] == "unsubscribe"
    assert "AAPL" in unsubscribed_symbols

@pytest.mark.asyncio
async def test_stream_message_handling(stream, mock_provider, mock_ws):
    """Test message handling."""
    mock_provider.create_websocket.return_value.__aenter__.return_value = mock_ws
    
    # Track received data
    received_data = None
    def on_data(event, data):
        nonlocal received_data
        received_data = data
    
    stream.on_event(StreamEvent.DATA, on_data)
    
    # Simulate message
    test_data = {"type": "trade", "symbol": "AAPL", "price": 150.0}
    message = WSMessage(
        type=WSMsgType.TEXT,
        data=json.dumps(test_data),
        extra=None
    )
    
    await stream.connect()
    await stream._handle_message(message.data)
    
    assert received_data == test_data

@pytest.mark.asyncio
async def test_stream_reconnection(stream, mock_provider, mock_ws):
    """Test stream reconnection on failure."""
    mock_ws.exception = aiohttp.ClientError("Connection lost")
    mock_provider.create_websocket.return_value.__aenter__.return_value = mock_ws
    
    # Track error events
    errors = []
    def on_error(event, data):
        errors.append(data)
    
    stream.on_event(StreamEvent.ERROR, on_error)
    
    await stream.connect()
    await asyncio.sleep(0.3)  # Allow time for reconnection attempts
    
    assert len(errors) > 0
    assert stream.status == StreamStatus.ERROR
    assert stream._reconnect_attempts > 0

@pytest.mark.asyncio
async def test_stream_subscription_format(stream):
    """Test subscription message formatting."""
    symbols = ["AAPL", "GOOGL"]
    formatted = stream._format_subscription(symbols)
    
    assert all(s.startswith("T.") for s in formatted)
    assert all(s[2:] in symbols for s in formatted)

@pytest.mark.asyncio
async def test_stream_invalid_operations(stream):
    """Test invalid stream operations."""
    with pytest.raises(MarketDataError):
        await stream.subscribe(["AAPL"])  # Should fail when not connected
        
    with pytest.raises(MarketDataError):
        await stream.unsubscribe(["AAPL"])  # Should fail when not connected

@pytest.mark.asyncio
async def test_stream_event_handlers(stream):
    """Test event handler management."""
    handler = Mock()
    stream.on_event(StreamEvent.DATA, handler)
    
    # Simulate data event
    await stream._notify(StreamEvent.DATA, {"price": 100})
    
    handler.assert_called_once()
    
    # Remove handler
    stream.remove_handler(StreamEvent.DATA, handler)
    await stream._notify(StreamEvent.DATA, {"price": 200})
    
    assert handler.call_count == 1  # Should not have been called again

@pytest.mark.asyncio
async def test_stream_heartbeat(stream, mock_provider, mock_ws):
    """Test heartbeat mechanism."""
    mock_provider.create_websocket.return_value.__aenter__.return_value = mock_ws
    mock_ws.ping = AsyncMock()
    
    await stream.connect()
    stream._last_heartbeat = datetime.now() - timedelta(seconds=31)
    
    # Start heartbeat loop
    heartbeat_task = asyncio.create_task(stream._heartbeat_loop())
    await asyncio.sleep(0.1)
    heartbeat_task.cancel()
    
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass
    
    assert mock_ws.ping.called
