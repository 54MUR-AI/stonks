import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Type, Callable, Any
import logging
import random
from enum import Enum
from dataclasses import dataclass
import time
import inspect

import pandas as pd

from .base import MarketDataProvider, MarketDataConfig
from ..realtime_data import RealTimeDataService

class MarketDataError(Exception):
    """Base class for market data errors"""
    pass

class ConnectionError(MarketDataError):
    """Error indicating connection issues"""
    pass

class SubscriptionError(MarketDataError):
    """Error during subscription operations"""
    pass

class QuoteError(MarketDataError):
    """Error retrieving quotes"""
    pass

class HistoricalDataError(MarketDataError):
    """Error retrieving historical data"""
    pass

@dataclass
class ErrorContext:
    """Context information for market data errors"""
    timestamp: datetime
    operation: str
    symbols: List[str]
    details: str
    retry_count: int = 0

class MarketDataAdapter:
    """Adapter to integrate market data providers with the RealTimeDataService"""
    
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0  # seconds
    
    def __init__(
        self,
        provider_class: Type[MarketDataProvider],
        config: MarketDataConfig,
        realtime_service: RealTimeDataService,
        error_callback: Optional[Callable[[Exception, ErrorContext], None]] = None
    ):
        self.provider = provider_class(config)
        self.realtime_service = realtime_service
        self._error_callback = error_callback
        self._subscribed_symbols = set()
        self._running = False
        self._update_task = None
        self._error_contexts: Dict[str, ErrorContext] = {}
        self._last_successful_updates: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        self._last_request_time = 0
        self.logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Start the adapter with error handling"""
        if self._running:
            return
        
        try:
            self._running = True
            await self._connect_with_retry()
            await self.realtime_service.start()
            self._update_task = asyncio.create_task(self._update_loop())
            self.logger.info("Market data adapter started successfully")
        except Exception as e:
            self._running = False
            error_context = ErrorContext(
                timestamp=datetime.now(),
                operation="start",
                symbols=[],
                details="Failed to start adapter"
            )
            await self._handle_error(e, error_context)
            raise

    async def stop(self) -> None:
        """Stop the adapter with cleanup"""
        if not self._running:
            return
            
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
            
        try:
            await self.provider.disconnect()
            await self.realtime_service.stop()
            self.logger.info("Market data adapter stopped")
        except Exception as e:
            error_context = ErrorContext(
                timestamp=datetime.now(),
                operation="stop",
                symbols=[],
                details="Error during shutdown"
            )
            await self._handle_error(e, error_context)
            raise

    async def _connect_with_retry(self) -> None:
        """Attempt connection with retries"""
        retry_count = 0
        last_error = None
        
        while retry_count < self.MAX_RETRY_ATTEMPTS:
            try:
                await self.provider.connect()
                return
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.MAX_RETRY_ATTEMPTS:
                    await asyncio.sleep(self.RETRY_DELAY * retry_count)
                    
        if last_error:
            error_context = ErrorContext(
                timestamp=datetime.now(),
                operation="connect",
                symbols=[],
                details=f"Failed after {retry_count} attempts"
            )
            await self._handle_error(last_error, error_context)
            raise last_error

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data with error tracking"""
        error_context = ErrorContext(
            timestamp=datetime.now(),
            operation="subscribe",
            symbols=symbols,
            details="Subscription attempt"
        )
        
        try:
            await self.provider.subscribe(symbols)
            await self.realtime_service.subscribe(symbols)
            self._subscribed_symbols.update(symbols)
            self.logger.info(f"Subscribed to symbols: {symbols}")
            
            # Update success tracking
            for symbol in symbols:
                self._last_successful_updates[symbol] = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error during subscription: {e}")
            error_context.details = str(e)
            await self._handle_error(e, error_context)
            raise SubscriptionError(f"Failed to subscribe to {symbols}: {e}")

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for symbols"""
        try:
            await self.provider.unsubscribe(symbols)
            await self.realtime_service.unsubscribe(symbols)
            self._subscribed_symbols.difference_update(symbols)
            self.logger.info(f"Unsubscribed from symbols: {symbols}")
        except Exception as e:
            self.logger.error(f"Error during unsubscription: {e}")
            raise

    async def _update_loop(self) -> None:
        """Update loop with enhanced error handling"""
        while self._running:
            try:
                # Get quotes for all subscribed symbols
                for symbol in list(self._subscribed_symbols):
                    try:
                        quote = await self.provider.get_quote(symbol)
                        await self.realtime_service._simulate_market_update(
                            symbol=symbol,
                            price=quote,
                            volume=random.randint(100, 1000)
                        )
                        self._last_successful_updates[symbol] = datetime.now()
                        
                    except Exception as e:
                        error_context = ErrorContext(
                            timestamp=datetime.now(),
                            operation="update",
                            symbols=[symbol],
                            details="Error in update loop"
                        )
                        
                        # Check for stale data
                        last_update = self._last_successful_updates.get(symbol)
                        if last_update and (datetime.now() - last_update) > timedelta(seconds=5):
                            error_context.details = "Stale data detected"
                            
                        await self._handle_error(e, error_context)
                        
                        # Handle reconnection if needed
                        if isinstance(e, ConnectionError):
                            await self._handle_connection_error()
                            
                await asyncio.sleep(0.1)  # Update every 100ms
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_context = ErrorContext(
                    timestamp=datetime.now(),
                    operation="update_loop",
                    symbols=list(self._subscribed_symbols),
                    details="Fatal error in update loop"
                )
                await self._handle_error(e, error_context)
                await asyncio.sleep(1)  # Back off on error

    async def _rate_limit(self):
        """Enforce rate limiting between requests with burst handling"""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            # Allow burst of up to 3 requests, then enforce stricter limiting
            if hasattr(self, '_request_count'):
                self._request_count += 1
            else:
                self._request_count = 1
                
            if self._request_count > 3:
                # Stricter rate limiting after burst
                await asyncio.sleep(self.provider.config.min_request_interval * 1.5)
                self._request_count = 0
            elif time_since_last < self.provider.config.min_request_interval:
                await asyncio.sleep(self.provider.config.min_request_interval - time_since_last)
                
            self._last_request_time = time.time()

    async def get_historical_data(
        self,
        symbol: str,
        lookback: timedelta,
        interval: str = "1min"
    ) -> pd.DataFrame:
        """Get historical data for a symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - lookback
            return await self.provider.get_historical_data(
                symbol,
                start_date,
                end_date,
                interval
            )
        except Exception as e:
            error_context = ErrorContext(
                timestamp=datetime.now(),
                operation="get_historical_data",
                symbols=[symbol],
                details="Failed to get historical data"
            )
            await self._handle_error(e, error_context)
            raise HistoricalDataError(f"Failed to get historical data for {symbol}: {e}")

    async def get_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """Get quotes for multiple symbols with error handling"""
        quotes = {}
        for symbol in symbols:
            await self._rate_limit()  # Apply rate limiting
            try:
                quotes[symbol] = await self.provider.get_quote(symbol)
            except Exception as e:
                error_context = ErrorContext(
                    timestamp=datetime.now(),
                    operation="get_quote",
                    symbols=[symbol],
                    details="Failed to get quote"
                )
                await self._handle_error(e, error_context)
                # Re-raise only connection errors
                if isinstance(e, ConnectionError):
                    raise QuoteError(f"Failed to get quote for {symbol}: {e}")
        return quotes

    async def _handle_connection_error(self) -> None:
        """Handle connection errors with retry logic"""
        try:
            await self._connect_with_retry()
            # Resubscribe to all symbols
            if self._subscribed_symbols:
                await self.provider.subscribe(list(self._subscribed_symbols))
        except Exception as e:
            error_context = ErrorContext(
                timestamp=datetime.now(),
                operation="reconnect",
                symbols=list(self._subscribed_symbols),
                details="Failed to recover connection"
            )
            await self._handle_error(e, error_context)

    async def _handle_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle errors with context and improved recovery strategies"""
        self.logger.error(f"{context.operation} error: {error} (symbols: {context.symbols})")
        
        # Track error context
        context_key = f"{context.operation}_{','.join(context.symbols)}"
        if context_key in self._error_contexts:
            self._error_contexts[context_key].retry_count += 1
        else:
            self._error_contexts[context_key] = context

        # Check for stale data and update context if needed
        if context.operation == "update":
            for symbol in context.symbols:
                if symbol in self._last_successful_updates:
                    last_update = self._last_successful_updates[symbol]
                    if (datetime.now() - last_update).total_seconds() > 5:
                        context.details = f"Stale data detected for {symbol}"

        # Enhanced error recovery strategies
        recovery_attempted = False
        if isinstance(error, ConnectionError):
            try:
                await self._handle_connection_error()
                recovery_attempted = True
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")
                context.details += f" (Recovery failed: {recovery_error})"
        elif isinstance(error, QuoteError):
            if context.retry_count < self.MAX_RETRY_ATTEMPTS:
                await asyncio.sleep(self.RETRY_DELAY * (2 ** context.retry_count))
                recovery_attempted = True

        # Call error callback if set, with proper context handling
        if self._error_callback:
            try:
                if asyncio.iscoroutinefunction(self._error_callback):
                    # Support both single and dual parameter callbacks
                    if len(inspect.signature(self._error_callback).parameters) == 2:
                        await self._error_callback(error, context)
                    else:
                        await self._error_callback(error)
                else:
                    # Run synchronous callbacks in executor
                    if len(inspect.signature(self._error_callback).parameters) == 2:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._error_callback, error, context
                        )
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._error_callback, error
                        )
            except Exception as callback_error:
                self.logger.error(f"Error in error callback: {callback_error}")

        # Log recovery status
        if recovery_attempted:
            self.logger.info(f"Recovery attempted for {context.operation} error")

    @property
    def on_error(self):
        """Get error callback"""
        return self._error_callback

    @on_error.setter
    def on_error(self, callback):
        """Set error callback"""
        if callback is not None and not callable(callback):
            raise ValueError("Error callback must be callable")
        self._error_callback = callback
