"""Configuration classes for market data providers."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class PolygonConfig:
    """Configuration for Polygon.io provider."""
    api_key: str
    BASE_URL: str = "https://api.polygon.io/v2"
    WS_URL: str = "wss://socket.polygon.io/stocks"
    TIMEOUT: int = 30
    RETRIES: int = 3
    RETRY_DELAY: float = 1.0

@dataclass
class YahooFinanceConfig:
    """Configuration for Yahoo Finance provider."""
    TIMEOUT: int = 30
    RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    USER_AGENT: str = "Mozilla/5.0"
