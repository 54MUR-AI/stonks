"""Real-time market data streaming implementation."""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Callable, Any, List
from datetime import datetime
import aiohttp
from enum import Enum

from .providers.base import MarketDataProvider, MarketDataError
from ...utils.logger import get_logger

logger = get_logger(__name__)

class StreamType(Enum):
    """Types of market data streams."""
    TRADES = "trades"
    QUOTES = "quotes"
    BARS = "bars"
    AGGREGATES = "aggregates"

class StreamEvent(Enum):
    """Stream event types."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    DATA = "data"
    ERROR = "error"

class StreamStatus(Enum):
    """Stream connection status."""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class MarketDataStream:
    """Base class for market data streaming."""
    
    def __init__(
        self,
        provider: MarketDataProvider,
        stream_type: StreamType,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 5
    ):
        self.provider = provider
        self.stream_type = stream_type
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self._status = StreamStatus.INITIALIZING
        self._subscriptions: Set[str] = set()
        self._handlers: Dict[StreamEvent, List[Callable]] = {
            event: [] for event in StreamEvent
        }
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_heartbeat = datetime.now()
        self._reconnect_attempts = 0
        self._closing = False

    @property
    def status(self) -> StreamStatus:
        """Get current stream status."""
        return self._status

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if self._status in [StreamStatus.CONNECTED, StreamStatus.CONNECTING]:
            return
            
        self._status = StreamStatus.CONNECTING
        self._closing = False
        self._task = asyncio.create_task(self._stream_loop())
        await self._notify(StreamEvent.CONNECTED)

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._closing = True
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self._status = StreamStatus.DISCONNECTED
        await self._notify(StreamEvent.DISCONNECTED)

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data for symbols."""
        new_symbols = set(symbols) - self._subscriptions
        if not new_symbols:
            return
            
        if self._status != StreamStatus.CONNECTED:
            raise MarketDataError("Stream not connected")
            
        try:
            message = {
                "action": "subscribe",
                "params": self._format_subscription(list(new_symbols))
            }
            await self._ws.send_json(message)
            self._subscriptions.update(new_symbols)
            await self._notify(StreamEvent.SUBSCRIBED, {"symbols": list(new_symbols)})
            
        except Exception as e:
            logger.error(f"Subscribe error: {str(e)}")
            raise MarketDataError(f"Subscribe failed: {str(e)}") from e

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for symbols."""
        symbols_to_remove = set(symbols) & self._subscriptions
        if not symbols_to_remove:
            return
            
        if self._status != StreamStatus.CONNECTED:
            raise MarketDataError("Stream not connected")
            
        try:
            message = {
                "action": "unsubscribe",
                "params": self._format_subscription(list(symbols_to_remove))
            }
            await self._ws.send_json(message)
            self._subscriptions.difference_update(symbols_to_remove)
            await self._notify(
                StreamEvent.UNSUBSCRIBED,
                {"symbols": list(symbols_to_remove)}
            )
            
        except Exception as e:
            logger.error(f"Unsubscribe error: {str(e)}")
            raise MarketDataError(f"Unsubscribe failed: {str(e)}") from e

    def on_event(self, event: StreamEvent, handler: Callable) -> None:
        """Register event handler."""
        self._handlers[event].append(handler)

    def remove_handler(self, event: StreamEvent, handler: Callable) -> None:
        """Remove event handler."""
        if handler in self._handlers[event]:
            self._handlers[event].remove(handler)

    async def _notify(self, event: StreamEvent, data: Any = None) -> None:
        """Notify event handlers."""
        for handler in self._handlers[event]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event, data)
                else:
                    handler(event, data)
            except Exception as e:
                logger.error(f"Handler error: {str(e)}")

    async def _stream_loop(self) -> None:
        """Main streaming loop."""
        while not self._closing:
            try:
                async with self.provider.create_websocket() as ws:
                    self._ws = ws
                    self._status = StreamStatus.CONNECTED
                    self._reconnect_attempts = 0
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    
                    # Resubscribe to existing subscriptions
                    if self._subscriptions:
                        await self.subscribe(list(self._subscriptions))
                    
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_message(message.data)
                        elif message.type == aiohttp.WSMsgType.ERROR:
                            raise Exception(ws.exception())
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._closing:
                    break
                    
                logger.error(f"Stream error: {str(e)}")
                self._status = StreamStatus.ERROR
                await self._notify(StreamEvent.ERROR, str(e))
                
                if self._reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error("Max reconnection attempts reached")
                    break
                    
                self._reconnect_attempts += 1
                await asyncio.sleep(self.reconnect_interval)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while True:
            try:
                if (datetime.now() - self._last_heartbeat).total_seconds() > 30:
                    await self._ws.ping()
                    self._last_heartbeat = datetime.now()
                await asyncio.sleep(15)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {str(e)}")
                await asyncio.sleep(5)

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            await self._notify(StreamEvent.DATA, data)
        except Exception as e:
            logger.error(f"Message handling error: {str(e)}")
            await self._notify(StreamEvent.ERROR, str(e))

    def _format_subscription(self, symbols: List[str]) -> List[str]:
        """Format subscription message based on stream type."""
        prefix = {
            StreamType.TRADES: "T.",
            StreamType.QUOTES: "Q.",
            StreamType.BARS: "AM.",
            StreamType.AGGREGATES: "A."
        }.get(self.stream_type, "")
        
        return [f"{prefix}{symbol}" for symbol in symbols]
