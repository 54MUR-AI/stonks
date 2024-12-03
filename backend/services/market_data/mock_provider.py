import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from .base import MarketDataProvider, MarketDataConfig
from .circuit_breaker import CircuitBreaker
from .metrics import Metrics

class MockProvider(MarketDataProvider):
    """Mock market data provider for testing and development"""
    
    def __init__(self, config: MarketDataConfig):
        """Initialize the mock provider"""
        super().__init__(config)
        self._connected = False
        self._subscribed_symbols = set()
        self._symbol_metadata = {}  # Track metadata per symbol
        self._stop_streaming = False
        self._stream_task = None
        self._last_request_time = datetime.now() - timedelta(seconds=1.0)
        self._request_times = []  # List to track request timestamps
        self._min_request_interval = 0.01  # 10ms minimum between requests
        self._rate_limit_counter = 0
        self._rate_limit_window = 1.0  # 1 second window
        self._rate_limit_max = 10  # Max 10 requests per window
        self._error_injection_rate = 0.0  # Rate at which to inject errors
        self._max_symbols = 100  # Maximum number of symbols allowed
        self._max_reconnect_attempts = 3
        self._reconnect_attempt = 0
        self._last_error = None
        self._data_buffer = asyncio.Queue(maxsize=1000)  # Buffer for backpressure handling
        self._enforce_rate_limit = asyncio.Lock()
        self._timeout_delay = 0.0
        self._timeout_probability = 0.0
        self._timeout_error = asyncio.TimeoutError("Operation timed out")
        self._circuit_breaker = CircuitBreaker()
        self._metrics = Metrics()
        
    @property
    def connected(self) -> bool:
        """Get connection status"""
        return self._connected

    @connected.setter
    def connected(self, value: bool):
        """Set connection status"""
        self._connected = value

    @property
    def subscribed_symbols(self) -> set:
        """Get the set of subscribed symbols"""
        return self._subscribed_symbols

    @subscribed_symbols.setter
    def subscribed_symbols(self, value: set) -> None:
        """Set the subscribed symbols"""
        self._subscribed_symbols = value
        
    async def connect(self) -> None:
        """Connect to the mock provider"""
        return await self._execute_with_circuit_breaker('connect', self._connect)
        
    async def _connect(self) -> None:
        """Internal connect implementation"""
        if self._connected:
            return
            
        await self._maybe_timeout()
        self._connected = True
        
        # Restart stream if we have subscriptions
        if self._subscribed_symbols and not self._stream_task:
            self._stop_streaming = False
            self._stream_task = asyncio.create_task(self._stream_market_data())
            
    async def disconnect(self) -> None:
        """Disconnect from the mock provider"""
        return await self._execute_with_circuit_breaker('disconnect', self._disconnect)
        
    async def _disconnect(self) -> None:
        """Internal disconnect implementation"""
        if not self._connected:
            return
            
        # Stop streaming first
        if self._stream_task:
            self._stop_streaming = True
            try:
                # Give the stream task a chance to clean up
                await asyncio.wait_for(self._stream_task, timeout=1.0)
            except asyncio.TimeoutError:
                # Force cancel if cleanup takes too long
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                print(f"Error during stream cleanup: {e}")
            finally:
                self._stream_task = None
                
        self._connected = False
        self._subscribed_symbols.clear()
        
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to mock market data stream"""
        return await self._execute_with_circuit_breaker('subscribe', self._subscribe, symbols)
        
    async def _subscribe(self, symbols: List[str]) -> None:
        """Internal subscribe implementation"""
        if not self._connected:
            raise RuntimeError("Not connected to market data provider")
            
        # Check max symbols limit
        if len(self._subscribed_symbols) + len(symbols) > self._max_symbols:
            raise ValueError(f"Cannot subscribe to more than {self._max_symbols} symbols")
            
        await self._maybe_timeout()
        
        # Initialize metadata for new symbols
        for symbol in symbols:
            await self._validate_symbol(symbol)
            
        self._subscribed_symbols.update(symbols)
        
        # Start streaming if needed
        if symbols and not self._stream_task:
            self._stop_streaming = False
            try:
                self._stream_task = asyncio.create_task(self._stream_market_data())
            except Exception as e:
                # Rollback on failure
                self._subscribed_symbols.difference_update(symbols)
                print(f"Failed to start stream task: {e}")
                raise
                
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from mock market data stream"""
        return await self._execute_with_circuit_breaker('unsubscribe', self._unsubscribe, symbols)
        
    async def _unsubscribe(self, symbols: List[str]) -> None:
        """Internal unsubscribe implementation"""
        if not self._connected:
            raise RuntimeError("Not connected to market data provider")
            
        await self._maybe_timeout()
        self._subscribed_symbols.difference_update(symbols)
        
        # Stop streaming if no more subscriptions
        if not self._subscribed_symbols and self._stream_task:
            self._stop_streaming = True
            try:
                await asyncio.wait_for(self._stream_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                print(f"Error during stream cleanup: {e}")
            finally:
                self._stream_task = None
                
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1min"
    ) -> pd.DataFrame:
        """Generate mock historical market data"""
        return await self._execute_with_circuit_breaker(
            'get_historical',
            self._get_historical_data,
            symbol,
            start_date,
            end_date,
            interval
        )
        
    async def _get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1min"
    ) -> pd.DataFrame:
        """Internal historical data implementation"""
        if not self._connected:
            raise RuntimeError("Not connected to market data provider")
            
        await self._maybe_timeout()
        if not end_date:
            end_date = datetime.now()
            
        self._validate_interval(interval)
        if start_date > end_date:
            return pd.DataFrame()  # Return empty DataFrame for invalid range
            
        # Generate date range based on interval
        if interval == "1min":
            freq = "1min"
        elif interval == "5min":
            freq = "5min"
        elif interval == "1h":
            freq = "1h"
        elif interval == "1d":
            freq = "1D"
            
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate mock price data with random walk
        n_points = len(dates)
        base_price = 100.0
        returns = np.random.normal(0.0001, 0.001, n_points)
        prices = base_price * (1 + np.cumsum(returns))
        volumes = np.random.randint(1000, 10000, n_points)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.002, n_points)),
            'low': prices * (1 - np.random.uniform(0, 0.002, n_points)),
            'close': prices,
            'volume': volumes
        })
        
        # Record metrics
        self._metrics.record_symbol_update(symbol, prices[-1], volumes[-1])
        
        return df
        
    async def get_quote(self, symbol: str) -> float:
        """Get mock quote price"""
        return await self._execute_with_circuit_breaker('get_quote', self._get_quote, symbol)
        
    async def _get_quote(self, symbol: str) -> float:
        """Internal get quote implementation"""
        if not self._connected:
            raise RuntimeError("Not connected to market data provider")
            
        await self._maybe_timeout()
        await self._validate_symbol(symbol)
        
        async with self._enforce_rate_limit:
            # Enforce rate limiting
            now = datetime.now()
            
            # Clean up old request times
            self._request_times = [t for t in self._request_times 
                                if (now - t).total_seconds() <= self._rate_limit_window]
            
            # Check rate limit
            if len(self._request_times) >= self._rate_limit_max:
                # Calculate required delay
                oldest_request = self._request_times[0]
                delay = self._rate_limit_window - (now - oldest_request).total_seconds()
                if delay > 0:
                    await asyncio.sleep(delay)
                    now = datetime.now()
                    
            # Add current request time
            self._request_times.append(now)
            
            # Update metadata
            self._symbol_metadata[symbol]['request_count'] += 1
            
            await asyncio.sleep(0.1)  # Simulate network delay
            price = 100.0 + random.uniform(-10, 10)
            self._symbol_metadata[symbol]['last_price'] = price
            
            # Record metrics
            self._metrics.record_symbol_update(symbol, price, random.randint(100, 1000))
            
            return price
            
    async def _stream_market_data(self) -> None:
        """Generate mock market data stream"""
        try:
            while not self._stop_streaming:
                try:
                    await self._maybe_timeout()
                    
                    for symbol in list(self._subscribed_symbols):
                        if self._stop_streaming:
                            break
                            
                        # Generate mock data
                        price = 100.0 + random.uniform(-5, 5)
                        data = {
                            'symbol': symbol,
                            'timestamp': datetime.now(),
                            'price': price,
                            'volume': random.randint(100, 1000),
                            'metadata': {
                                'timestamp': datetime.now(),
                                'source': 'mock',
                                'update_type': 'trade',
                                'trade_id': str(random.randint(1000000, 9999999)),
                                'sequence': self._symbol_metadata[symbol]['request_count']
                            }
                        }
                        
                        # Update metadata
                        self._symbol_metadata[symbol]['request_count'] += 1
                        self._symbol_metadata[symbol]['last_price'] = price
                        
                        # Handle backpressure
                        await self._handle_backpressure(data)
                        
                        if hasattr(self, 'on_data'):
                            try:
                                await self.on_data(data)
                            except Exception as e:
                                print(f"Error in market data callback: {e}")
                                self._symbol_metadata[symbol]['error_count'] += 1
                                continue
                                
                    await asyncio.sleep(0.1)  # Stream more frequently for testing
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"Error in market data stream loop: {e}")
                    self._last_error = e
                    await asyncio.sleep(1)  # Back off on error
                    continue
                    
        except asyncio.CancelledError:
            print("Stream task cancelled")
            self._stop_streaming = True
            raise
        except Exception as e:
            print(f"Fatal error in market data stream: {e}")
            self._stop_streaming = True
            self._last_error = e
            if self._stream_task and not self._stream_task.done():
                self._stream_task.cancel()
            raise
        finally:
            self._stop_streaming = True
            
    async def _handle_backpressure(self, data: Dict[str, Any]) -> None:
        """Handle backpressure in data streaming"""
        try:
            # Initialize metadata if not exists
            symbol = data['symbol']
            if symbol not in self._symbol_metadata:
                await self._validate_symbol(symbol)
                
            await asyncio.wait_for(self._data_buffer.put(data), timeout=0.1)
        except asyncio.TimeoutError:
            print(f"Warning: Data buffer full, dropping update for {data['symbol']}")
            if data['symbol'] in self._symbol_metadata:
                self._symbol_metadata[data['symbol']]['error_count'] += 1
                
    async def get_symbol_stats(self, symbol: str) -> Dict[str, Any]:
        """Get statistics for a specific symbol"""
        if symbol not in self._symbol_metadata:
            raise ValueError(f"Unknown symbol: {symbol}")
        return self._symbol_metadata[symbol].copy()

    def _validate_interval(self, interval: str) -> None:
        """Validate the requested interval"""
        valid_intervals = ["1min", "5min", "1h", "1d"]
        if interval not in valid_intervals:
            raise ValueError(f"Unsupported interval: {interval}")

    def _validate_dates(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Validate the date range"""
        if start_date > end_date:
            return pd.DataFrame()  # Return empty DataFrame for invalid range

    def set_timeout_simulation(self, delay: float = 0.1, probability: float = 0.0) -> None:
        """Configure timeout simulation parameters"""
        self._timeout_delay = delay
        self._timeout_probability = probability
        self._timeout_error = asyncio.TimeoutError("Operation timed out")
        
    async def _maybe_timeout(self) -> None:
        """Potentially simulate a timeout based on configured probability"""
        if random.random() < self._timeout_probability:
            if self._timeout_delay > 0:
                await asyncio.sleep(self._timeout_delay)
            self._connected = False
            raise self._timeout_error

    async def _validate_symbol(self, symbol: str) -> None:
        """Validate a symbol and update its metadata"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
            
        if symbol not in self._symbol_metadata:
            self._symbol_metadata[symbol] = {
                'first_seen': datetime.now(),
                'request_count': 0,
                'error_count': 0,
                'last_price': None,
                'status': 'active'
            }
            
    async def _maybe_inject_error(self) -> None:
        """Potentially inject an error based on the error rate"""
        if random.random() < self._error_injection_rate:
            error_types = [
                TimeoutError("Simulated timeout"),
                ConnectionError("Simulated connection error"),
                RuntimeError("Simulated runtime error"),
                asyncio.CancelledError()
            ]
            raise random.choice(error_types)
            
    def inject_error_rate(self, rate: float) -> None:
        """Set the rate at which to inject errors for testing"""
        if not 0 <= rate <= 1:
            raise ValueError("Error injection rate must be between 0 and 1")
        self._error_injection_rate = rate

    async def _execute_with_circuit_breaker(self, operation: str, func: callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection"""
        try:
            return await self._circuit_breaker.execute(operation, func, *args, **kwargs)
        except Exception as e:
            self._metrics.record_error(operation, str(e))
            raise
