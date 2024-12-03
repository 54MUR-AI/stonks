from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
from .circuit_breaker import CircuitBreaker
from .metrics import ProviderMetrics

class MarketDataError(Exception):
    """Base exception class for market data related errors"""
    pass

@dataclass
class MarketDataCredentials:
    """Credentials for market data provider"""
    api_key: str
    api_secret: Optional[str] = None
    account_id: Optional[str] = None

class MarketDataConfig:
    """Configuration for market data providers"""
    
    def __init__(
        self,
        credentials: MarketDataCredentials,
        base_url: str,
        websocket_url: str,
        request_timeout: int = 30,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        rate_limit_max: int = 10,  # Maximum number of requests per window
        rate_limit_window: float = 1.0,  # Time window in seconds
        min_request_interval: float = 0.1  # Minimum time between requests
    ):
        """Initialize configuration.
        
        Args:
            credentials: Provider credentials
            base_url: Base URL for REST API
            websocket_url: WebSocket URL for streaming
            request_timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_timeout: Seconds before circuit recovery
            rate_limit_max: Maximum number of requests per window
            rate_limit_window: Time window in seconds
            min_request_interval: Minimum time between requests
        """
        self.credentials = credentials
        self.base_url = base_url
        self.websocket_url = websocket_url
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.rate_limit_max = rate_limit_max
        self.rate_limit_window = rate_limit_window
        self.min_request_interval = min_request_interval

class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    def __init__(self, config: MarketDataConfig):
        """Initialize provider.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=config.circuit_breaker_timeout
        )
        self._metrics = ProviderMetrics()
        
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get provider metrics"""
        return self._metrics.get_metrics()
        
    @property
    def circuit_breaker_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return self._circuit_breaker.metrics
        
    async def _execute_with_circuit_breaker(self, operation: str, func, *args, **kwargs):
        """Execute function with circuit breaker and metrics.
        
        Args:
            operation: Name of the operation
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from function
        """
        start_time = datetime.now()
        try:
            result = await self._circuit_breaker.call(func, *args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            self._metrics.record_latency(operation, duration)
            self._metrics.record_request(operation, success=True)
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._metrics.record_latency(operation, duration)
            self._metrics.record_request(operation, success=False)
            self._metrics.record_error(type(e).__name__)
            raise
        
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the market data provider"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the market data provider"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data for specified symbols"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time market data for specified symbols"""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1min"
    ) -> pd.DataFrame:
        """
        Retrieve historical market data for a symbol
        
        Args:
            symbol: The trading symbol
            start_date: Start date for historical data
            end_date: End date for historical data (optional, defaults to now)
            interval: Data interval (e.g., "1min", "5min", "1h", "1d")
            
        Returns:
            DataFrame with historical market data
        """
        pass
    
    @abstractmethod
    async def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get latest quote for a symbol
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Dictionary containing latest quote data
        """
        pass
    
    def _validate_config(self) -> None:
        """Validate the provider configuration"""
        if not self.config.credentials.api_key:
            raise ValueError("API key is required")
        if not self.config.base_url:
            raise ValueError("Base URL is required")
        if not self.config.websocket_url:
            raise ValueError("WebSocket URL is required")
