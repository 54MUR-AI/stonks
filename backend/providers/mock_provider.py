import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from ..models.market_data import MarketDataProvider
from ..exceptions import (
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError
)

class MockMarketDataProvider(MarketDataProvider):
    def __init__(
        self,
        name: str,
        failure_rate: float = 0.1,
        latency_range: tuple = (0.1, 0.5),
        rate_limit: int = 100,
        rate_limit_window: int = 60
    ):
        super().__init__(name)
        self.failure_rate = failure_rate
        self.min_latency, self.max_latency = latency_range
        self.rate_limit = rate_limit
        self.rate_limit_window = rate_limit_window
        self.request_timestamps: list[datetime] = []
        self._connected = False
        self._mock_data = self._initialize_mock_data()

    def _initialize_mock_data(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock market data for common symbols"""
        return {
            "AAPL": {
                "price": 150.0,
                "volume": 1000000,
                "high": 152.0,
                "low": 148.0,
                "open": 149.0,
                "close": 150.0,
            },
            "GOOGL": {
                "price": 2800.0,
                "volume": 500000,
                "high": 2850.0,
                "low": 2780.0,
                "open": 2790.0,
                "close": 2800.0,
            },
            "MSFT": {
                "price": 300.0,
                "volume": 750000,
                "high": 305.0,
                "low": 298.0,
                "open": 299.0,
                "close": 300.0,
            }
        }

    def _check_rate_limit(self):
        """Check if rate limit is exceeded"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.rate_limit_window)
        
        # Clean old timestamps
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > window_start]
        
        if len(self.request_timestamps) >= self.rate_limit:
            raise ProviderRateLimitError(
                f"Rate limit of {self.rate_limit} requests per {self.rate_limit_window} seconds exceeded"
            )
        
        self.request_timestamps.append(now)

    async def _simulate_latency(self):
        """Simulate network latency"""
        latency = random.uniform(self.min_latency, self.max_latency)
        await asyncio.sleep(latency)

    def _should_fail(self) -> bool:
        """Determine if request should fail based on failure rate"""
        return random.random() < self.failure_rate

    async def connect(self) -> None:
        """Simulate connecting to the provider"""
        if self._should_fail():
            raise ProviderConnectionError(f"Failed to connect to {self.name}")
        
        await self._simulate_latency()
        self._connected = True

    async def disconnect(self) -> None:
        """Simulate disconnecting from the provider"""
        await self._simulate_latency()
        self._connected = False

    async def reconnect(self) -> None:
        """Reconnect to the provider"""
        await self.disconnect()
        await self.connect()

    def _get_mock_data(self, symbol: str) -> Dict[str, Any]:
        """Get mock data for a symbol with some randomization"""
        base_data = self._mock_data.get(symbol, {
            "price": random.uniform(10, 1000),
            "volume": random.randint(100000, 1000000),
            "high": 0,
            "low": 0,
            "open": 0,
            "close": 0
        })

        # Add some random variation
        variation = random.uniform(-0.02, 0.02)  # ±2% variation
        price = base_data["price"] * (1 + variation)
        
        return {
            "price": price,
            "volume": base_data["volume"] * random.uniform(0.8, 1.2),
            "high": price * 1.02,
            "low": price * 0.98,
            "open": price * random.uniform(0.99, 1.01),
            "close": price,
            "timestamp": datetime.now().isoformat()
        }

    async def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get market data for the given symbol"""
        if not self._connected:
            raise ProviderConnectionError(f"Provider {self.name} is not connected")

        # Simulate potential failures
        if self._should_fail():
            error_type = random.choice([
                ProviderConnectionError(f"Connection lost to {self.name}"),
                ProviderTimeoutError("Request timed out"),
                ProviderRateLimitError("Rate limit exceeded")
            ])
            raise error_type

        # Check rate limit
        self._check_rate_limit()

        # Simulate network latency
        await self._simulate_latency()

        # Return mock data
        return self._get_mock_data(symbol)
