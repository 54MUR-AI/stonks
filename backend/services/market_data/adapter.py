import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Type, Callable
import logging

import pandas as pd

from .base import MarketDataProvider, MarketDataConfig
from ..realtime_data import RealTimeDataService

logger = logging.getLogger(__name__)

class MarketDataAdapter:
    """Adapter to integrate market data providers with the RealTimeDataService"""
    
    def __init__(
        self,
        provider_class: Type[MarketDataProvider],
        config: MarketDataConfig,
        realtime_service: RealTimeDataService,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        self.provider = provider_class(config)
        self.realtime_service = realtime_service
        self.on_error = on_error or self._default_error_handler
        self._subscribed_symbols = set()
        self._running = False
        self._update_task = None
        
    async def start(self) -> None:
        """Start the market data adapter"""
        try:
            await self.provider.connect()
            self._running = True
            self._update_task = asyncio.create_task(self._update_loop())
            logger.info("Market data adapter started successfully")
        except Exception as e:
            logger.error(f"Failed to start market data adapter: {e}")
            self.on_error(e)
            raise
            
    async def stop(self) -> None:
        """Stop the market data adapter"""
        self._running = False
        if self._update_task:
            await self._update_task
            self._update_task = None
        await self.provider.disconnect()
        logger.info("Market data adapter stopped")
        
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data for specified symbols"""
        new_symbols = set(symbols) - self._subscribed_symbols
        if new_symbols:
            try:
                await self.provider.subscribe(list(new_symbols))
                self._subscribed_symbols.update(new_symbols)
                logger.info(f"Subscribed to symbols: {new_symbols}")
            except Exception as e:
                logger.error(f"Failed to subscribe to symbols {new_symbols}: {e}")
                self.on_error(e)
                raise
                
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for specified symbols"""
        symbols_to_remove = set(symbols) & self._subscribed_symbols
        if symbols_to_remove:
            try:
                await self.provider.unsubscribe(list(symbols_to_remove))
                self._subscribed_symbols.difference_update(symbols_to_remove)
                logger.info(f"Unsubscribed from symbols: {symbols_to_remove}")
            except Exception as e:
                logger.error(f"Failed to unsubscribe from symbols {symbols_to_remove}: {e}")
                self.on_error(e)
                raise
                
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
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            self.on_error(e)
            raise
            
    async def _update_loop(self) -> None:
        """Main update loop for market data"""
        while self._running:
            try:
                for symbol in self._subscribed_symbols:
                    quote = await self.provider.get_latest_quote(symbol)
                    self.realtime_service.update_price(
                        symbol=quote['symbol'],
                        price=quote['last'],
                        timestamp=quote['timestamp'],
                        volume=quote['volume']
                    )
                await asyncio.sleep(1)  # Update frequency
            except Exception as e:
                logger.error(f"Error in market data update loop: {e}")
                self.on_error(e)
                # Continue running despite errors
                await asyncio.sleep(5)  # Back off on error
                
    def _default_error_handler(self, error: Exception) -> None:
        """Default error handler"""
        logger.error(f"Market data error: {error}")
