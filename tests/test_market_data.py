import unittest
from datetime import datetime, timedelta
from unittest.async_case import IsolatedAsyncioTestCase

from backend.services.market_data.base import MarketDataConfig, MarketDataCredentials
from backend.services.market_data.mock_provider import MockMarketDataProvider

class TestMockMarketDataProvider(IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up test cases"""
        config = MarketDataConfig(
            credentials=MarketDataCredentials(
                api_key="test_key",
                api_secret="test_secret"
            ),
            base_url="http://mock.api",
            websocket_url="ws://mock.api/ws"
        )
        self.provider = MockMarketDataProvider(config)
        
    async def test_connection(self):
        """Test connection and disconnection"""
        self.assertFalse(self.provider.connected)
        
        await self.provider.connect()
        self.assertTrue(self.provider.connected)
        
        await self.provider.disconnect()
        self.assertFalse(self.provider.connected)
        
    async def test_subscription(self):
        """Test market data subscription"""
        await self.provider.connect()
        
        # Subscribe to symbols
        symbols = ["AAPL", "GOOGL"]
        await self.provider.subscribe(symbols)
        self.assertEqual(self.provider.subscribed_symbols, set(symbols))
        
        # Unsubscribe from one symbol
        await self.provider.unsubscribe(["AAPL"])
        self.assertEqual(self.provider.subscribed_symbols, {"GOOGL"})
        
        await self.provider.disconnect()
        
    async def test_historical_data(self):
        """Test historical data retrieval"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        # Test 1-minute data
        df = await self.provider.get_historical_data(
            "AAPL",
            start_date,
            end_date,
            interval="1min"
        )
        self.assertGreater(len(df), 0)
        self.assertTrue(all(col in df.columns for col in [
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ]))
        
        # Test different intervals
        intervals = ["5min", "1h", "1d"]
        for interval in intervals:
            df = await self.provider.get_historical_data(
                "AAPL",
                start_date,
                end_date,
                interval=interval
            )
            self.assertGreater(len(df), 0)
            
    async def test_latest_quote(self):
        """Test latest quote retrieval"""
        quote = await self.provider.get_latest_quote("AAPL")
        required_fields = ['symbol', 'timestamp', 'bid', 'ask', 'last', 'volume']
        self.assertTrue(all(field in quote for field in required_fields))
        self.assertEqual(quote['symbol'], "AAPL")
        
    async def test_invalid_interval(self):
        """Test invalid interval handling"""
        with self.assertRaises(ValueError):
            await self.provider.get_historical_data(
                "AAPL",
                datetime.now() - timedelta(days=1),
                interval="invalid"
            )
            
    async def test_subscription_without_connection(self):
        """Test subscription without connection"""
        with self.assertRaises(RuntimeError):
            await self.provider.subscribe(["AAPL"])

if __name__ == '__main__':
    unittest.main()
