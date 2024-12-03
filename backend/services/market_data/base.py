from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd

class MarketDataError(Exception):
    """Base exception class for market data related errors"""
    pass

@dataclass
class MarketDataCredentials:
    """Base class for market data provider credentials"""
    api_key: str
    api_secret: Optional[str] = None
    additional_params: Dict[str, Any] = None

@dataclass
class MarketDataConfig:
    """Configuration for market data providers"""
    credentials: MarketDataCredentials
    base_url: str
    websocket_url: str
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1
    rate_limit_max: int = 10  # Maximum number of requests per window
    rate_limit_window: float = 1.0  # Time window in seconds
    min_request_interval: float = 0.1  # Minimum time between requests

class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    def __init__(self, config: MarketDataConfig):
        self.config = config
        self._validate_config()
        
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
