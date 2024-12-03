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

class ProviderStats:
    """Track provider performance stats"""
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        self._lock = asyncio.Lock()

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0

    @property
    def average_latency(self) -> float:
        """Calculate average latency in seconds"""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests

    @property
    def avg_latency(self) -> float:
        """Alias for average_latency for backward compatibility"""
        return self.average_latency

    async def update(self, success: bool, latency: float) -> None:
        """Update provider stats"""
        async with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            self.total_latency += latency

class MarketDataAdapter:
    """Market data adapter with provider switching capabilities"""
    
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0  # seconds
    HEALTH_CHECK_INTERVAL = 60.0  # seconds
    MIN_SUCCESS_RATE = 80.0  # percentage
    
    def __init__(
        self,
        provider_class: Callable[[MarketDataConfig], MarketDataProvider],
        config: MarketDataConfig,
        realtime_service: RealTimeDataService,
        error_callback: Optional[Callable[[Exception, ErrorContext], None]] = None,
        max_retries: int = 3
    ):
        """Initialize the adapter with a provider class and config"""
        self._primary_provider_class = provider_class
        self._backup_providers: List[Callable[[MarketDataConfig], MarketDataProvider]] = []
        self._config = config
        self._realtime_service = realtime_service
        self._provider = None
        self._current_provider_index = -1
        self._lock = asyncio.Lock()
        self._running = False
        self._max_retries = max_retries
        self._error_callback = error_callback
        self._health_check_task = None
        self._provider_stats: Dict[int, ProviderStats] = {}
        self._error_contexts: Dict[str, ErrorContext] = {}
        self._last_successful_updates: Dict[str, datetime] = {}
        self._subscribed_symbols: Set[str] = set()
        self._logger = logging.getLogger(__name__)

    def register_backup_provider(self, provider_factory: Callable[[MarketDataConfig], MarketDataProvider]) -> None:
        """Register a backup provider factory"""
        self._backup_providers.append(provider_factory)

    async def _switch_provider(self) -> bool:
        """Switch to the next available provider. Returns True if switch successful."""
        async with self._lock:
            available_providers = [self._primary_provider_class] + self._backup_providers
            
            # Try each provider in sequence
            while self._current_provider_index < len(available_providers):
                # Move to next provider
                self._current_provider_index += 1
                if self._current_provider_index >= len(available_providers):
                    self._logger.error("No more providers available")
                    raise RuntimeError("No more providers available")
                    
                provider_factory = available_providers[self._current_provider_index]
                
                try:
                    # Create and connect new provider
                    old_provider = self._provider
                    self._provider = provider_factory(self._config)
                    
                    # Initialize provider stats
                    if self._current_provider_index not in self._provider_stats:
                        self._provider_stats[self._current_provider_index] = ProviderStats()
                    
                    # Connect to new provider
                    await self._connect_with_retry()
                    
                    # Re-subscribe to all symbols if needed
                    if self._subscribed_symbols:
                        await self._provider.subscribe(list(self._subscribed_symbols))
                        
                    # Disconnect old provider if it exists
                    if old_provider and old_provider.connected:
                        try:
                            await old_provider.disconnect()
                        except Exception as e:
                            self._logger.error(f"Error disconnecting old provider: {str(e)}")
                        
                    self._logger.info(f"Switched to provider {self._current_provider_index}")
                    return True
                    
                except Exception as e:
                    self._logger.error(f"Failed to switch to provider {self._current_provider_index}: {str(e)}")
                    # Continue to try next provider
            
            self._logger.error("No more providers available")
            raise RuntimeError("No more providers available")

    async def start(self):
        """Start the adapter and connect to primary provider"""
        if self._running:
            return
            
        try:
            self._running = True
            self._current_provider_index = -1  # Reset provider index
            await self._switch_provider()  # Connect to primary provider
            self._health_check_task = asyncio.create_task(self._monitor_provider_health())
        except Exception as e:
            self._logger.error(f"Failed to start adapter: {str(e)}")
            self._running = False
            raise

    async def stop(self):
        """Stop the adapter and disconnect from current provider"""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
                
        if self._provider and self._provider.connected:
            await self._provider.disconnect()

    async def _connect_with_retry(self) -> None:
        """Connect to the current provider with retries"""
        last_error = None
        start_time = time.time()
        
        for attempt in range(self._max_retries):
            try:
                await self._provider.connect()
                # Update stats for successful connection
                if self._current_provider_index in self._provider_stats:
                    latency = time.time() - start_time
                    await self._provider_stats[self._current_provider_index].update(success=True, latency=latency)
                return
            except Exception as e:
                last_error = e
                self._logger.error(f"Error in connect: {str(e)}")
                
                # Update stats for failed connection
                if self._current_provider_index in self._provider_stats:
                    await self._provider_stats[self._current_provider_index].update(success=False, latency=0.0)
                
                if attempt < self._max_retries - 1:
                    # Use exponential backoff with jitter
                    delay = min(self.RETRY_DELAY * (2 ** attempt) + random.uniform(0, 0.1), 5.0)
                    await asyncio.sleep(delay)
                    
        if last_error:
            error_context = ErrorContext(
                timestamp=datetime.now(),
                operation="connect",
                symbols=[],
                details=f"Failed to connect after {self._max_retries} attempts"
            )
            await self._handle_error(last_error, error_context)
            raise ConnectionError(f"Failed to connect to provider: {str(last_error)}")

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data with error tracking"""
        error_context = ErrorContext(
            timestamp=datetime.now(),
            operation="subscribe",
            symbols=symbols,
            details="Subscription attempt"
        )
        
        try:
            await self._provider.subscribe(symbols)
            await self._realtime_service.subscribe(symbols)
            self._subscribed_symbols.update(symbols)
            self._logger.info(f"Subscribed to symbols: {symbols}")
            
            # Update success tracking
            for symbol in symbols:
                self._last_successful_updates[symbol] = datetime.now()
                
        except Exception as e:
            self._logger.error(f"Error during subscription: {e}")
            error_context.details = str(e)
            await self._handle_error(e, error_context)
            raise SubscriptionError(f"Failed to subscribe to {symbols}: {e}")

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for symbols"""
        try:
            await self._provider.unsubscribe(symbols)
            await self._realtime_service.unsubscribe(symbols)
            self._subscribed_symbols.difference_update(symbols)
            self._logger.info(f"Unsubscribed from symbols: {symbols}")
        except Exception as e:
            self._logger.error(f"Error during unsubscription: {e}")
            raise

    async def _update_loop(self) -> None:
        """Update loop with enhanced error handling"""
        while self._running:
            try:
                # Get quotes for all subscribed symbols
                for symbol in list(self._subscribed_symbols):
                    try:
                        quote = await self._provider.get_quote(symbol)
                        await self._realtime_service._simulate_market_update(
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
                await asyncio.sleep(self._provider.config.min_request_interval * 1.5)
                self._request_count = 0
            elif time_since_last < self._provider.config.min_request_interval:
                await asyncio.sleep(self._provider.config.min_request_interval - time_since_last)
                
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
            return await self._provider.get_historical_data(
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
        """Get quotes for multiple symbols"""
        if not self._provider or not self._provider.connected:
            await self._switch_provider()

        quotes = {}
        for symbol in symbols:
            success = False
            last_error = None
            retry_count = 0
            total_providers = len(self._backup_providers) + 1  # Include primary provider

            while not success and self._current_provider_index < total_providers:
                try:
                    quotes[symbol] = await self._provider.get_quote(symbol)
                    success = True

                    # Update stats for successful request
                    stats = self._provider_stats.get(self._current_provider_index)
                    if stats:
                        await stats.update(success=True, latency=0.0)

                except Exception as e:
                    self._logger.error(f"Error in get_quotes: {str(e)}")
                    last_error = e
                    retry_count += 1

                    # Update stats for failed request
                    stats = self._provider_stats.get(self._current_provider_index)
                    if stats:
                        await stats.update(success=False, latency=0.0)

                    # Create error context with retry count
                    error_context = ErrorContext(
                        timestamp=datetime.now(),
                        operation="get_quotes",
                        symbols=[symbol],
                        details=f"Failed to get quote for {symbol}",
                        retry_count=retry_count
                    )

                    # If provider is not connected or times out, try switching providers
                    if isinstance(e, (RuntimeError, TimeoutError, QuoteError)):
                        try:
                            # Call error callback before switching provider
                            if self._error_callback:
                                self._error_callback(e, error_context)

                            await self._switch_provider()
                        except RuntimeError:
                            # No more providers available
                            if self._error_callback:
                                self._error_callback(e, error_context)
                            raise last_error if isinstance(last_error, Exception) else RuntimeError(str(last_error))
                    else:
                        if self._error_callback:
                            self._error_callback(e, error_context)
                        raise e

            if not success:
                error_context = ErrorContext(
                    timestamp=datetime.now(),
                    operation="get_quotes",
                    symbols=[symbol],
                    details=f"Failed to get quote for {symbol} after trying all providers",
                    retry_count=retry_count
                )
                if self._error_callback:
                    self._error_callback(last_error, error_context)
                raise last_error if isinstance(last_error, Exception) else RuntimeError(str(last_error))

        return quotes

    async def _handle_connection_error(self) -> None:
        """Handle connection errors with retry logic"""
        try:
            await self._connect_with_retry()
            # Resubscribe to all symbols
            if self._subscribed_symbols:
                await self._provider.subscribe(list(self._subscribed_symbols))
        except Exception as e:
            error_context = ErrorContext(
                timestamp=datetime.now(),
                operation="reconnect",
                symbols=list(self._subscribed_symbols),
                details="Failed to recover connection"
            )
            await self._handle_error(e, error_context)

    async def _handle_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle errors with provider switching logic"""
        # Log error and notify callback
        self._logger.error(f"Error in {context.operation}: {str(error)}")
        self._error_contexts[context.operation] = context
        
        if self._error_callback:
            try:
                self._error_callback(error, context)
            except Exception as callback_error:
                self._logger.error(f"Error in error callback: {str(callback_error)}")

        # For critical errors, try switching providers
        if isinstance(error, (ConnectionError, QuoteError)):
            if await self._switch_provider():
                self._logger.info("Successfully switched to backup provider")
            else:
                self._logger.error("Failed to switch to backup provider")
                raise error

    async def _monitor_provider_health(self):
        """Monitor provider health and switch if necessary"""
        while True:
            try:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
                
                # Skip if no provider or no requests yet
                if not self._provider or not self._provider_stats.get(self._current_provider_index):
                    continue
                
                stats = self._provider_stats[self._current_provider_index]
                
                # Check success rate
                if stats.success_rate < self.MIN_SUCCESS_RATE:
                    self._logger.warning(
                        f"Provider {self._current_provider_index} health check failed: "
                        f"success rate {stats.success_rate:.1f}% below threshold {self.MIN_SUCCESS_RATE}%"
                    )
                    await self._switch_provider()
                    continue
                
                # Check last failure
                if (stats.last_failure and 
                    datetime.now() - stats.last_failure < timedelta(minutes=5) and
                    stats.last_success and 
                    stats.last_success < stats.last_failure):
                    self._logger.warning(
                        f"Provider {self._current_provider_index} health check failed: "
                        f"recent failure at {stats.last_failure}"
                    )
                    await self._switch_provider()
                    continue
                
            except Exception as e:
                self._logger.error(f"Error in provider health monitoring: {str(e)}")
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

    async def _update_provider_stats(self, success: bool, latency: float):
        """Update performance statistics for current provider"""
        async with self._provider_stats_lock:
            if self._current_provider_index not in self._provider_stats:
                self._provider_stats[self._current_provider_index] = ProviderStats()
            
            stats = self._provider_stats[self._current_provider_index]
            stats.total_requests += 1
            stats.total_latency += latency
            
            if success:
                stats.last_success = datetime.now()
            else:
                stats.failed_requests += 1
                stats.last_failure = datetime.now()

    def _select_best_provider(self) -> int:
        """Select the best provider based on performance metrics"""
        best_score = float('-inf')
        best_index = 0
        
        for idx, stats in self._provider_stats.items():
            # Skip current provider as we're looking for a replacement
            if idx == self._current_provider_index:
                continue
            
            # Calculate score based on success rate and latency
            latency_score = 1.0 / (stats.avg_latency + 0.001)  # Avoid division by zero
            score = 0.7 * stats.success_rate + 0.3 * latency_score
            
            if score > best_score:
                best_score = score
                best_index = idx
        
        return best_index

    @property
    def provider_stats(self) -> Dict[int, ProviderStats]:
        """Get provider stats"""
        return self._provider_stats

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

    @property
    def avg_latency(self) -> float:
        """Get average latency for backward compatibility"""
        return self.average_latency
