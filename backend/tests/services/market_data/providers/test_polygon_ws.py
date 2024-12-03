"""Tests for Polygon.io WebSocket functionality."""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
from aiohttp import WSMessage, WSMsgType

from backend.services.market_data.providers.polygon import PolygonProvider
from backend.services.market_data.providers.config import PolygonConfig
from backend.services.market_data.errors import MarketDataError

@pytest.fixture
def config():
    return PolygonConfig(api_key="test_key")

@pytest.fixture
def provider(config):
    return PolygonProvider(config)

class MockWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.closed = False
        self.exception = None
        
    async def send_json(self, data):
        self.sent_messages.append(data)
        
    async def close(self):
        self.closed = True
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if self.exception:
            raise self.exception
        raise StopAsyncIteration

@pytest.mark.asyncio
async def test_websocket_creation(provider):
    """Test WebSocket connection creation."""
    mock_ws = MockWebSocket()
    
    with patch("aiohttp.ClientSession.ws_connect", AsyncMock(return_value=mock_ws)):
        # Initialize session
        await provider.initialize()
        
        # Create WebSocket
        ws = await provider.create_websocket()
        
        assert ws == mock_ws
        assert len(mock_ws.sent_messages) == 1
        assert mock_ws.sent_messages[0]["action"] == "auth"
        assert mock_ws.sent_messages[0]["params"] == "test_key"

@pytest.mark.asyncio
async def test_websocket_auth_failure(provider):
    """Test WebSocket authentication failure."""
    mock_ws = MockWebSocket()
    
    async def mock_iter():
        yield WSMessage(
            type=WSMsgType.TEXT,
            data=json.dumps({"ev": "status", "status": "auth_failed"}),
            extra=None
        )
    
    mock_ws.__aiter__ = mock_iter
    
    with patch("aiohttp.ClientSession.ws_connect", AsyncMock(return_value=mock_ws)):
        await provider.initialize()
        
        with pytest.raises(MarketDataError, match="WebSocket authentication failed"):
            await provider.create_websocket()

@pytest.mark.asyncio
async def test_trade_subscription(provider):
    """Test trade subscription."""
    mock_ws = MockWebSocket()
    provider._ws = mock_ws
    
    symbols = ["AAPL", "GOOGL"]
    await provider.subscribe_trades(symbols)
    
    assert len(mock_ws.sent_messages) == 1
    assert mock_ws.sent_messages[0]["action"] == "subscribe"
    assert all(f"T.{symbol}" in mock_ws.sent_messages[0]["params"] for symbol in symbols)
    assert all(symbol in provider._subscriptions["trades"] for symbol in symbols)

@pytest.mark.asyncio
async def test_quote_subscription(provider):
    """Test quote subscription."""
    mock_ws = MockWebSocket()
    provider._ws = mock_ws
    
    symbols = ["AAPL", "GOOGL"]
    await provider.subscribe_quotes(symbols)
    
    assert len(mock_ws.sent_messages) == 1
    assert mock_ws.sent_messages[0]["action"] == "subscribe"
    assert all(f"Q.{symbol}" in mock_ws.sent_messages[0]["params"] for symbol in symbols)
    assert all(symbol in provider._subscriptions["quotes"] for symbol in symbols)

@pytest.mark.asyncio
async def test_aggregate_subscription(provider):
    """Test aggregate subscription."""
    mock_ws = MockWebSocket()
    provider._ws = mock_ws
    
    symbols = ["AAPL", "GOOGL"]
    await provider.subscribe_aggregates(symbols)
    
    assert len(mock_ws.sent_messages) == 1
    assert mock_ws.sent_messages[0]["action"] == "subscribe"
    assert all(f"AM.{symbol}" in mock_ws.sent_messages[0]["params"] for symbol in symbols)
    assert all(symbol in provider._subscriptions["aggregates"] for symbol in symbols)

@pytest.mark.asyncio
async def test_unsubscribe_all(provider):
    """Test unsubscribe from all streams."""
    mock_ws = MockWebSocket()
    provider._ws = mock_ws
    
    # Subscribe to multiple streams
    symbols = ["AAPL", "GOOGL"]
    await provider.subscribe_trades(symbols)
    await provider.subscribe_quotes(symbols)
    await provider.subscribe_aggregates(symbols)
    
    # Clear sent messages
    mock_ws.sent_messages.clear()
    
    # Unsubscribe all
    await provider.unsubscribe_all()
    
    assert len(mock_ws.sent_messages) == 1
    assert mock_ws.sent_messages[0]["action"] == "unsubscribe"
    assert len(mock_ws.sent_messages[0]["params"]) == len(symbols) * 3
    assert all(len(subs) == 0 for subs in provider._subscriptions.values())

@pytest.mark.asyncio
async def test_message_handlers(provider):
    """Test message handler registration and execution."""
    # Create mock handlers
    trade_handler = AsyncMock()
    quote_handler = AsyncMock()
    agg_handler = AsyncMock()
    
    # Register handlers
    provider.on_trade(trade_handler)
    provider.on_quote(quote_handler)
    provider.on_aggregate(agg_handler)
    
    # Test trade message
    trade_msg = {"ev": "T", "sym": "AAPL", "p": 150.0}
    await provider._handle_ws_message(
        WSMessage(type=WSMsgType.TEXT, data=json.dumps(trade_msg), extra=None)
    )
    trade_handler.assert_called_once_with(trade_msg)
    
    # Test quote message
    quote_msg = {"ev": "Q", "sym": "AAPL", "bp": 150.0, "ap": 150.1}
    await provider._handle_ws_message(
        WSMessage(type=WSMsgType.TEXT, data=json.dumps(quote_msg), extra=None)
    )
    quote_handler.assert_called_once_with(quote_msg)
    
    # Test aggregate message
    agg_msg = {"ev": "AM", "sym": "AAPL", "o": 150.0, "h": 151.0}
    await provider._handle_ws_message(
        WSMessage(type=WSMsgType.TEXT, data=json.dumps(agg_msg), extra=None)
    )
    agg_handler.assert_called_once_with(agg_msg)

@pytest.mark.asyncio
async def test_error_handling(provider):
    """Test WebSocket error handling."""
    mock_ws = MockWebSocket()
    provider._ws = mock_ws
    
    # Test connection without session
    provider._session = None
    with pytest.raises(MarketDataError, match="Provider not initialized"):
        await provider.create_websocket()
    
    # Test subscription without connection
    provider._ws = None
    with pytest.raises(MarketDataError, match="WebSocket not connected"):
        await provider.subscribe_trades(["AAPL"])
    
    # Test message processing error
    provider._ws = mock_ws
    invalid_msg = WSMessage(
        type=WSMsgType.TEXT,
        data="invalid json",
        extra=None
    )
    # Should not raise exception
    await provider._handle_ws_message(invalid_msg)
