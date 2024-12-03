import yfinance as yf
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ..provider import MarketDataProvider, MarketDataError, ProviderStatus
from ..rate_limiter import RateLimitConfig, RateLimitedClient
from ...utils.logger import get_logger

logger = get_logger(__name__)

class YahooFinanceProvider(MarketDataProvider, RateLimitedClient):
    def __init__(
        self,
        request_timeout: int = 10,
        max_retries: int = 3,
        retry_delay: int = 1,
        max_symbols_per_request: int = 100,
        thread_pool_size: int = 5
    ):
        MarketDataProvider.__init__(self)
        RateLimitedClient.__init__(
            self,
            RateLimitConfig(
                requests_per_second=2,
                requests_per_minute=100,
                requests_per_hour=2000,
                requests_per_day=48000,
                concurrent_requests=5
            )
        )
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_symbols_per_request = max_symbols_per_request
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self._status = ProviderStatus.INITIALIZING
        self._cache = {}
        self._cache_ttl = 60  # 1 minute cache TTL

    async def initialize(self) -> None:
        """Initialize the provider"""
        try:
            # Test connection by fetching a well-known symbol
            await self.get_latest_price("SPY")
            self._status = ProviderStatus.READY
            logger.info("Yahoo Finance provider initialized successfully")
        except Exception as e:
            self._status = ProviderStatus.ERROR
            logger.error(f"Failed to initialize Yahoo Finance provider: {str(e)}")
            raise MarketDataError("Provider initialization failed") from e

    async def shutdown(self) -> None:
        """Shutdown the provider"""
        try:
            self.thread_pool.shutdown(wait=True)
            self._status = ProviderStatus.STOPPED
            logger.info("Yahoo Finance provider shut down successfully")
        except Exception as e:
            logger.error(f"Error during Yahoo Finance provider shutdown: {str(e)}")
            raise MarketDataError("Provider shutdown failed") from e

    def _run_in_thread(self, func: callable, *args: Any) -> asyncio.Future:
        """Run a synchronous function in a thread pool"""
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self.thread_pool, func, *args)

    async def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol"""
        cache_key = f"price_{symbol}"
        if cache_key in self._cache:
            timestamp, price = self._cache[cache_key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return price

        return await self.execute_with_rate_limit(
            self._get_latest_price_impl,
            symbol
        )

    async def _get_latest_price_impl(self, symbol: str) -> float:
        """Implementation of get_latest_price with retries"""
        for attempt in range(self.max_retries):
            try:
                ticker = await self._run_in_thread(yf.Ticker, symbol)
                data = await self._run_in_thread(
                    lambda: ticker.history(period="1d")
                )
                if data.empty:
                    raise MarketDataError(f"No data available for symbol {symbol}")
                
                price = float(data['Close'].iloc[-1])
                self._cache[f"price_{symbol}"] = (datetime.now().timestamp(), price)
                return price
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to get price for {symbol}: {str(e)}")
                    raise MarketDataError(f"Failed to get price for {symbol}") from e
                await asyncio.sleep(self.retry_delay)

    async def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols"""
        tasks = []
        for symbol_batch in self._batch_symbols(symbols):
            tasks.append(self._get_batch_prices(symbol_batch))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = {}
        for batch_result in results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch price fetch failed: {str(batch_result)}")
                continue
            prices.update(batch_result)
            
        return prices

    def _batch_symbols(self, symbols: List[str]) -> List[List[str]]:
        """Split symbols into batches"""
        return [
            symbols[i:i + self.max_symbols_per_request]
            for i in range(0, len(symbols), self.max_symbols_per_request)
        ]

    async def _get_batch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for a batch of symbols"""
        prices = {}
        for symbol in symbols:
            try:
                price = await self.get_latest_price(symbol)
                prices[symbol] = price
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {str(e)}")
        return prices

    async def get_historical_prices(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical prices for symbols"""
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo"
        }
        
        yf_interval = interval_map.get(interval)
        if not yf_interval:
            raise MarketDataError(f"Unsupported interval: {interval}")

        all_data = []
        for symbol_batch in self._batch_symbols(symbols):
            try:
                data = await self._get_batch_historical(
                    symbol_batch,
                    start_date,
                    end_date,
                    yf_interval
                )
                all_data.append(data)
            except Exception as e:
                logger.error(f"Failed to get historical data: {str(e)}")
                raise MarketDataError("Failed to get historical data") from e

        if not all_data:
            raise MarketDataError("No historical data available")

        return pd.concat(all_data, axis=1)

    async def _get_batch_historical(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> pd.DataFrame:
        """Get historical data for a batch of symbols"""
        async def fetch_symbol(symbol: str) -> pd.DataFrame:
            try:
                ticker = await self._run_in_thread(yf.Ticker, symbol)
                data = await self._run_in_thread(
                    lambda: ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=interval
                    )
                )
                return data['Close'].rename(symbol)
            except Exception as e:
                logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
                return pd.Series(name=symbol)

        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return pd.concat(results, axis=1)

    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a symbol"""
        try:
            ticker = await self._run_in_thread(yf.Ticker, symbol)
            info = await self._run_in_thread(lambda: ticker.info)
            
            return {
                "symbol": symbol,
                "name": info.get("longName", ""),
                "exchange": info.get("exchange", ""),
                "currency": info.get("currency", ""),
                "type": info.get("quoteType", ""),
                "market_cap": info.get("marketCap", None),
                "volume": info.get("volume", None),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
            }
        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {str(e)}")
            raise MarketDataError(f"Failed to get info for {symbol}") from e

    @property
    def status(self) -> ProviderStatus:
        """Get the current status of the provider"""
        return self._status

    @property
    def provider_name(self) -> str:
        """Get the provider name"""
        return "Yahoo Finance"

    @property
    def supported_intervals(self) -> List[str]:
        """Get supported intervals"""
        return ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
