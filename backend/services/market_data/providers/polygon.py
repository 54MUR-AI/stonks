"""Polygon.io market data provider implementation."""

import aiohttp
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from urllib.parse import urljoin

from ..provider import MarketDataProvider, MarketDataError, ProviderStatus
from ..rate_limiter import RateLimitConfig, RateLimitedClient
from ...utils.logger import get_logger

logger = get_logger(__name__)

class PolygonConfig:
    """Polygon.io API configuration."""
    BASE_URL = "https://api.polygon.io/v2/"
    WS_URL = "wss://socket.polygon.io/stocks"
    
    def __init__(
        self,
        api_key: str,
        request_timeout: int = 10,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

class PolygonProvider(MarketDataProvider, RateLimitedClient):
    """Polygon.io market data provider implementation."""
    
    def __init__(self, config: PolygonConfig):
        MarketDataProvider.__init__(self)
        RateLimitedClient.__init__(
            self,
            RateLimitConfig(
                requests_per_second=5,  # Basic tier limits
                requests_per_minute=200,
                requests_per_hour=4000,
                requests_per_day=96000,
                concurrent_requests=10
            )
        )
        
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._status = ProviderStatus.INITIALIZING
        self._subscriptions: Dict[str, set] = {
            "trades": set(),
            "quotes": set(),
            "aggregates": set()
        }
        self._handlers: Dict[str, callable] = {}
        self._ws_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_heartbeat = datetime.now()
        
    async def initialize(self) -> None:
        """Initialize the provider."""
        try:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
            # Test connection with a simple API call
            await self.get_latest_price("AAPL")
            self._status = ProviderStatus.READY
            logger.info("Polygon.io provider initialized successfully")
        except Exception as e:
            self._status = ProviderStatus.ERROR
            logger.error(f"Failed to initialize Polygon.io provider: {str(e)}")
            raise MarketDataError("Provider initialization failed") from e

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        try:
            if self._ws_task:
                self._ws_task.cancel()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            if self._ws:
                await self._ws.close()
            if self._session:
                await self._session.close()
            self._status = ProviderStatus.STOPPED
            logger.info("Polygon.io provider shut down successfully")
        except Exception as e:
            logger.error(f"Error during Polygon.io provider shutdown: {str(e)}")
            raise MarketDataError("Provider shutdown failed") from e

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the Polygon API."""
        url = urljoin(self.config.BASE_URL, endpoint)
        
        for attempt in range(self.config.max_retries):
            try:
                async with self._session.get(
                    url,
                    params=params,
                    timeout=self.config.request_timeout
                ) as response:
                    if response.status == 429:  # Rate limit exceeded
                        retry_after = int(response.headers.get('Retry-After', self.config.retry_delay))
                        await asyncio.sleep(retry_after)
                        continue
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                    if data.get('status') == 'ERROR':
                        raise MarketDataError(f"API Error: {data.get('error')}")
                        
                    return data
                    
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout for {endpoint}")
                if attempt == self.config.max_retries - 1:
                    raise MarketDataError("Request timeout")
                    
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise MarketDataError(f"Request failed: {str(e)}")
                    
            await asyncio.sleep(self.config.retry_delay)

    async def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        return await self.execute_with_rate_limit(
            self._get_latest_price_impl,
            symbol
        )

    async def _get_latest_price_impl(self, symbol: str) -> float:
        """Implementation of get_latest_price."""
        try:
            response = await self._make_request(
                f"last/trade/{symbol}",
                {"apiKey": self.config.api_key}
            )
            return float(response['results']['p'])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {str(e)}")
            raise MarketDataError(f"Failed to get price for {symbol}") from e

    async def get_historical_prices(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical prices for symbols."""
        interval_map = {
            "1m": "minute",
            "5m": "minute",
            "15m": "minute",
            "30m": "minute",
            "1h": "hour",
            "1d": "day",
            "1wk": "week",
            "1mo": "month"
        }
        
        multiplier = 1
        if interval.endswith('m'):
            multiplier = int(interval[:-1])
        
        timespan = interval_map.get(interval)
        if not timespan:
            raise MarketDataError(f"Unsupported interval: {interval}")

        all_data = []
        for symbol in symbols:
            try:
                data = await self.execute_with_rate_limit(
                    self._get_historical_prices_impl,
                    symbol,
                    start_date,
                    end_date,
                    multiplier,
                    timespan
                )
                all_data.append(data)
            except Exception as e:
                logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
                continue

        if not all_data:
            raise MarketDataError("No historical data available")

        return pd.concat(all_data, axis=1)

    async def _get_historical_prices_impl(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        multiplier: int,
        timespan: str
    ) -> pd.DataFrame:
        """Implementation of get_historical_prices."""
        try:
            response = await self._make_request(
                f"aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.date()}/{end_date.date()}",
                {"apiKey": self.config.api_key, "adjusted": "true"}
            )
            
            if not response.get('results'):
                return pd.Series(name=symbol)
                
            df = pd.DataFrame(response['results'])
            df.index = pd.to_datetime(df['t'], unit='ms')
            return df['c'].rename(symbol)  # Return closing prices
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
            return pd.Series(name=symbol)

    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a symbol."""
        return await self.execute_with_rate_limit(
            self._get_symbol_info_impl,
            symbol
        )

    async def _get_symbol_info_impl(self, symbol: str) -> Dict[str, Any]:
        """Implementation of get_symbol_info."""
        try:
            response = await self._make_request(
                f"reference/tickers/{symbol}",
                {"apiKey": self.config.api_key}
            )
            
            ticker_details = response['results']
            return {
                "symbol": symbol,
                "name": ticker_details.get('name', ''),
                "exchange": ticker_details.get('exchange', ''),
                "currency": ticker_details.get('currency_name', 'USD'),
                "type": ticker_details.get('type', ''),
                "market_cap": ticker_details.get('market_cap', None),
                "description": ticker_details.get('description', ''),
                "sector": ticker_details.get('sector', ''),
                "industry": ticker_details.get('industry', '')
            }
            
        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {str(e)}")
            raise MarketDataError(f"Failed to get info for {symbol}") from e

    async def create_websocket(self) -> aiohttp.ClientWebSocketResponse:
        """Create a new WebSocket connection."""
        if not self._session:
            raise MarketDataError("Provider not initialized")

        try:
            ws = await self._session.ws_connect(
                f"{self.config.WS_URL}",
                heartbeat=30,
                receive_timeout=35
            )
            
            # Authenticate
            await ws.send_json({
                "action": "auth",
                "params": self.config.api_key
            })
            
            # Wait for auth response
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("ev") == "status" and data.get("status") == "auth_success":
                        return ws
                    elif data.get("ev") == "status" and data.get("status") == "auth_failed":
                        raise MarketDataError("WebSocket authentication failed")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise MarketDataError(f"WebSocket error during auth: {ws.exception()}")
                    
            raise MarketDataError("WebSocket authentication timeout")
            
        except Exception as e:
            raise MarketDataError(f"Failed to create WebSocket connection: {str(e)}")

    async def _handle_ws_message(self, msg: aiohttp.WSMessage) -> None:
        """Handle incoming WebSocket message."""
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
                event_type = data.get("ev")
                
                if event_type == "T":  # Trade
                    await self._handle_trade(data)
                elif event_type == "Q":  # Quote
                    await self._handle_quote(data)
                elif event_type == "AM":  # Minute Bar
                    await self._handle_aggregate(data)
                elif event_type == "status":
                    await self._handle_status(data)
                    
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                
        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.error(f"WebSocket error: {msg.data}")
            
        elif msg.type == aiohttp.WSMsgType.CLOSED:
            logger.info("WebSocket connection closed")

    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """Handle trade message."""
        if "trades" in self._handlers:
            await self._handlers["trades"](data)

    async def _handle_quote(self, data: Dict[str, Any]) -> None:
        """Handle quote message."""
        if "quotes" in self._handlers:
            await self._handlers["quotes"](data)

    async def _handle_aggregate(self, data: Dict[str, Any]) -> None:
        """Handle aggregate bar message."""
        if "aggregates" in self._handlers:
            await self._handlers["aggregates"](data)

    async def _handle_status(self, data: Dict[str, Any]) -> None:
        """Handle status message."""
        status = data.get("status")
        if status == "connected":
            logger.info("WebSocket connected")
        elif status == "auth_success":
            logger.info("WebSocket authenticated")
        elif status == "auth_failed":
            logger.error("WebSocket authentication failed")

    def on_trade(self, handler: callable) -> None:
        """Register trade message handler."""
        self._handlers["trades"] = handler

    def on_quote(self, handler: callable) -> None:
        """Register quote message handler."""
        self._handlers["quotes"] = handler

    def on_aggregate(self, handler: callable) -> None:
        """Register aggregate message handler."""
        self._handlers["aggregates"] = handler

    async def subscribe_trades(self, symbols: List[str]) -> None:
        """Subscribe to trade updates."""
        if not self._ws:
            raise MarketDataError("WebSocket not connected")
            
        message = {
            "action": "subscribe",
            "params": [f"T.{symbol}" for symbol in symbols]
        }
        await self._ws.send_json(message)
        self._subscriptions["trades"].update(symbols)

    async def subscribe_quotes(self, symbols: List[str]) -> None:
        """Subscribe to quote updates."""
        if not self._ws:
            raise MarketDataError("WebSocket not connected")
            
        message = {
            "action": "subscribe",
            "params": [f"Q.{symbol}" for symbol in symbols]
        }
        await self._ws.send_json(message)
        self._subscriptions["quotes"].update(symbols)

    async def subscribe_aggregates(self, symbols: List[str]) -> None:
        """Subscribe to aggregate (minute bar) updates."""
        if not self._ws:
            raise MarketDataError("WebSocket not connected")
            
        message = {
            "action": "subscribe",
            "params": [f"AM.{symbol}" for symbol in symbols]
        }
        await self._ws.send_json(message)
        self._subscriptions["aggregates"].update(symbols)

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all current subscriptions."""
        if not self._ws:
            return
            
        all_subs = []
        for stream_type, symbols in self._subscriptions.items():
            prefix = {"trades": "T.", "quotes": "Q.", "aggregates": "AM."}[stream_type]
            all_subs.extend(f"{prefix}{symbol}" for symbol in symbols)
            
        if all_subs:
            message = {
                "action": "unsubscribe",
                "params": all_subs
            }
            await self._ws.send_json(message)
            
        for symbols in self._subscriptions.values():
            symbols.clear()

    @property
    def status(self) -> ProviderStatus:
        """Get the current status of the provider."""
        return self._status

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "Polygon.io"

    @property
    def supported_intervals(self) -> List[str]:
        """Get supported intervals."""
        return ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
