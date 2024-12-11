from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ProviderMetrics:
    def __init__(
        self,
        status: ProviderStatus,
        success_rate: float,
        avg_latency: float,
        error_count: int,
        last_error: Optional[datetime],
        last_success: Optional[datetime]
    ):
        self.status = status
        self.success_rate = success_rate
        self.avg_latency = avg_latency
        self.error_count = error_count
        self.last_error = last_error
        self.last_success = last_success

class MarketDataProvider(ABC):
    def __init__(self, name: str):
        self.name = name
        self.status = ProviderStatus.HEALTHY

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the provider"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the provider"""
        pass

    @abstractmethod
    async def reconnect(self) -> None:
        """Reconnect to the provider"""
        await self.disconnect()
        await self.connect()

    @abstractmethod
    async def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Fetch market data for the given symbol"""
        pass
