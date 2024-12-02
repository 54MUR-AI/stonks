import unittest
import asyncio
from datetime import datetime, timedelta
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import Mock, patch, AsyncMock

from backend.services.market_data import (
    MarketDataAdapter,
    MarketDataConfig,
    MarketDataCredentials,
    MockMarketDataProvider,
    AlphaVantageProvider
)
from backend.services.realtime_data import RealTimeDataService

class TestMarketDataAdapter(IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up test cases"""
        self.config = MarketDataConfig(
            credentials=MarketDataCredentials(api_key="test_key"),
            base_url="http://test.api",
            websocket_url="ws://test.api/ws"
        )
        self.realtime_service = RealTimeDataService(websocket_url="ws://test.api/ws")
        self.adapter = MarketDataAdapter(
            provider_class=MockMarketDataProvider,
            config=self.config,
            realtime_service=self.realtime_service
        )
        
    async def test_lifecycle(self):
        """Test adapter lifecycle (start/stop)"""
        # Start adapter
        await self.adapter.start()
        self.assertTrue(self.adapter._running)
        self.assertTrue(self.adapter.provider.connected)
        self.assertIsNotNone(self.adapter._update_task)
        
        # Stop adapter
        await self.adapter.stop()
        self.assertFalse(self.adapter._running)
        self.assertFalse(self.adapter.provider.connected)
        self.assertIsNone(self.adapter._update_task)
        
    async def test_symbol_subscription(self):
        """Test symbol subscription management"""
        await self.adapter.start()
        
        # Subscribe to symbols
        symbols = ["AAPL", "GOOGL"]
        await self.adapter.subscribe(symbols)
        self.assertEqual(self.adapter._subscribed_symbols, set(symbols))
        
        # Unsubscribe from one symbol
        await self.adapter.unsubscribe(["AAPL"])
        self.assertEqual(self.adapter._subscribed_symbols, {"GOOGL"})
        
        await self.adapter.stop()
        
    async def test_historical_data(self):
        """Test historical data retrieval"""
        await self.adapter.start()
        
        lookback = timedelta(days=1)
        df = await self.adapter.get_historical_data("AAPL", lookback)
        
        self.assertGreater(len(df), 0)
        self.assertTrue(all(col in df.columns for col in [
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ]))
        
        await self.adapter.stop()
        
    async def test_price_updates(self):
        """Test price updates flow to realtime service"""
        await self.adapter.start()
        
        # Subscribe to a symbol
        await self.adapter.subscribe(["AAPL"])
        
        # Wait for a few updates
        await asyncio.sleep(3)
        
        # Check that prices were updated in realtime service
        history = self.realtime_service.get_price_history("AAPL")
        self.assertGreater(len(history), 0)
        self.assertTrue(all(col in history.columns for col in ['timestamp', 'price', 'volume']))
        
        await self.adapter.stop()
        
    async def test_error_handling(self):
        """Test error handler invocation"""
        error_handler = Mock()
        adapter = MarketDataAdapter(
            provider_class=MockMarketDataProvider,
            config=self.config,
            realtime_service=self.realtime_service,
            on_error=error_handler
        )
        
        # Simulate provider error
        with patch.object(adapter.provider, 'connect', side_effect=Exception("Test error")):
            with self.assertRaises(Exception):
                await adapter.start()
            error_handler.assert_called_once()
            
    async def test_update_loop_resilience(self):
        """Test update loop continues despite errors"""
        error_handler = Mock()
        adapter = MarketDataAdapter(
            provider_class=MockMarketDataProvider,
            config=self.config,
            realtime_service=self.realtime_service,
            on_error=error_handler
        )
        
        await adapter.start()
        await adapter.subscribe(["AAPL"])
        
        # Simulate temporary provider error
        with patch.object(adapter.provider, 'get_latest_quote', side_effect=Exception("Test error")):
            await asyncio.sleep(2)
            error_handler.assert_called()
            
        # Verify adapter is still running
        self.assertTrue(adapter._running)
        
        await adapter.stop()

    async def test_quote_data_validation(self):
        """Test validation of quote data from provider"""
        await self.adapter.start()
        await self.adapter.subscribe(["AAPL"])
        
        # Test missing required fields
        invalid_quote = {
            'symbol': 'AAPL',
            'timestamp': datetime.now(),
            # Missing price and volume
        }
        
        with patch.object(self.adapter.provider, 'get_latest_quote', 
                         return_value=invalid_quote):
            with self.assertRaises(KeyError):
                await self.adapter._update_loop()
                
        await self.adapter.stop()
        
    async def test_rate_limiting(self):
        """Test rate limiting behavior"""
        await self.adapter.start()
        
        # Subscribe to multiple symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
        await self.adapter.subscribe(symbols)
        
        # Record update timestamps
        timestamps = []
        
        async def mock_get_quote(*args, **kwargs):
            timestamps.append(datetime.now())
            return {
                'symbol': args[0],
                'last': 100.0,
                'timestamp': datetime.now(),
                'volume': 1000
            }
            
        with patch.object(self.adapter.provider, 'get_latest_quote', 
                         side_effect=mock_get_quote):
            # Let it run for a few cycles
            await asyncio.sleep(3)
            
        # Check that updates are properly rate limited
        intervals = [(t2 - t1).total_seconds() 
                    for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        avg_interval = sum(intervals) / len(intervals)
        self.assertGreaterEqual(avg_interval, 0.2)  # 5 symbols per second
        
        await self.adapter.stop()
        
    async def test_multiple_providers(self):
        """Test adapter with different provider types"""
        # Test with Alpha Vantage provider
        alpha_adapter = MarketDataAdapter(
            provider_class=AlphaVantageProvider,
            config=self.config,
            realtime_service=self.realtime_service
        )
        
        # Should work the same as mock provider
        await alpha_adapter.start()
        await alpha_adapter.subscribe(["AAPL"])
        await asyncio.sleep(1)
        await alpha_adapter.stop()
        
        # Test with mock provider (already tested in other cases)
        mock_adapter = MarketDataAdapter(
            provider_class=MockMarketDataProvider,
            config=self.config,
            realtime_service=self.realtime_service
        )
        
        await mock_adapter.start()
        await mock_adapter.subscribe(["AAPL"])
        await asyncio.sleep(1)
        await mock_adapter.stop()
        
    async def test_update_loop_edge_cases(self):
        """Test edge cases in the update loop"""
        await self.adapter.start()
        
        # Test empty symbol list
        self.assertEqual(len(self.adapter._subscribed_symbols), 0)
        await asyncio.sleep(1)  # Let update loop run
        
        # Test single symbol
        await self.adapter.subscribe(["AAPL"])
        self.assertEqual(len(self.adapter._subscribed_symbols), 1)
        await asyncio.sleep(1)
        
        # Test duplicate symbols
        await self.adapter.subscribe(["AAPL"])  # Subscribe again
        self.assertEqual(len(self.adapter._subscribed_symbols), 1)
        
        # Test rapid subscribe/unsubscribe
        for _ in range(5):
            await self.adapter.subscribe(["MSFT"])
            await self.adapter.unsubscribe(["MSFT"])
        self.assertNotIn("MSFT", self.adapter._subscribed_symbols)
        
        # Test unsubscribe non-existent symbol
        await self.adapter.unsubscribe(["NON_EXISTENT"])
        self.assertEqual(len(self.adapter._subscribed_symbols), 1)
        
        await self.adapter.stop()

if __name__ == '__main__':
    unittest.main()
