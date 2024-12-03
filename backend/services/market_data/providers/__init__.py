"""Market data provider initialization."""

from typing import Dict, Optional
from .yahoo_finance import YahooFinanceProvider
from ..provider_manager import ProviderManager, ProviderPriority
from ..health import ProviderHealthMonitor
from ..cache import Cache
from ..base import MarketDataConfig

async def initialize_providers(
    config: MarketDataConfig,
    cache: Optional[Cache] = None
) -> ProviderManager:
    """Initialize and configure market data providers."""
    
    # Create health monitor
    health_monitor = ProviderHealthMonitor()
    
    # Create provider manager
    manager = ProviderManager(config, health_monitor, cache)
    
    # Initialize Yahoo Finance provider
    yahoo_provider = YahooFinanceProvider(
        request_timeout=config.request_timeout,
        max_retries=config.max_retries,
        retry_delay=config.retry_delay,
        max_symbols_per_request=config.max_symbols_per_request
    )
    
    # Add providers to manager with priorities
    await manager.add_provider(
        provider_id="yahoo_finance",
        provider=yahoo_provider,
        priority=ProviderPriority.PRIMARY
    )
    
    # Start the manager
    await manager.start()
    
    return manager

__all__ = ['initialize_providers', 'YahooFinanceProvider']
