"""
Tests for real-time data service
"""
import unittest
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from backend.services.realtime_data import RealTimeDataService, MarketUpdate
import numpy.testing
import pandas.testing

class TestRealTimeData(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up test data"""
        self.service = RealTimeDataService(
            websocket_url="ws://test.example.com/ws",
            buffer_size=100
        )
        
        # Sample market data
        self.sample_data = {
            'timestamp': datetime.now().timestamp(),
            'symbol': 'AAPL',
            'price': 150.0,
            'volume': 1000,
            'metadata': {'exchange': 'NASDAQ'}
        }
    
    def test_market_update_creation(self):
        """Test MarketUpdate data class"""
        update = MarketUpdate(
            timestamp=datetime.fromtimestamp(self.sample_data['timestamp']),
            symbol=self.sample_data['symbol'],
            price=self.sample_data['price'],
            volume=self.sample_data['volume'],
            metadata=self.sample_data['metadata']
        )
        
        self.assertEqual(update.symbol, 'AAPL')
        self.assertEqual(update.price, 150.0)
        self.assertEqual(update.volume, 1000)
        self.assertEqual(update.metadata['exchange'], 'NASDAQ')
    
    async def test_websocket_connection(self):
        """Test WebSocket connection"""
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.open = True
            mock_connect.return_value = mock_ws
            
            success = await self.service.connect()
            self.assertTrue(success)
            mock_connect.assert_called_once_with("ws://test.example.com/ws")
    
    async def test_message_processing(self):
        """Test message processing"""
        # Create mock callback
        callback_called = False
        def callback(update):
            nonlocal callback_called
            callback_called = True
            self.assertEqual(update.symbol, 'AAPL')
            self.assertEqual(update.price, 150.0)
        
        self.service.add_callback(callback)
        
        # Process sample message
        await self.service.process_message(json.dumps(self.sample_data))
        
        # Check callback was called
        self.assertTrue(callback_called)
        
        # Check price buffer was updated
        self.assertIn('AAPL', self.service.price_buffers)
        buffer = self.service.price_buffers['AAPL']
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.iloc[-1]['price'], 150.0)
    
    def test_price_history(self):
        """Test price history retrieval"""
        # Create fixed reference time to avoid timing issues
        now = pd.Timestamp.now().floor('min')
        
        # Add sample data to buffer with exact timestamps
        timestamps = [
            now - pd.Timedelta(minutes=i)
            for i in range(5)
        ]
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': np.linspace(150, 155, 5),
            'volume': np.full(5, 1000)
        })
        
        self.service.price_buffers['AAPL'] = data
        
        # Test full history
        history = self.service.get_price_history('AAPL')
        self.assertEqual(len(history), 5)
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(history['timestamp'].values),
            pd.DatetimeIndex(sorted(timestamps))
        )
        
        # Test with lookback - should include current minute and 2 previous
        history = self.service.get_price_history(
            'AAPL',
            lookback=pd.Timedelta(minutes=2)
        )
        expected_timestamps = sorted(timestamps[:3])  # Current minute and 2 previous
        self.assertEqual(len(history), 3)
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(history['timestamp'].values),
            pd.DatetimeIndex(expected_timestamps)
        )
        
        # Test empty symbol
        history = self.service.get_price_history('INVALID')
        self.assertTrue(history.empty)
    
    def test_latest_prices(self):
        """Test latest price retrieval"""
        now = datetime.now()
        
        # Add sample data
        self.service.price_buffers['AAPL'] = pd.DataFrame({
            'timestamp': [now],
            'price': [150.0],
            'volume': [1000]
        })
        
        self.service.price_buffers['MSFT'] = pd.DataFrame({
            'timestamp': [now],
            'price': [250.0],
            'volume': [500]
        })
        
        # Test single symbol
        prices = self.service.get_latest_prices(['AAPL'])
        self.assertEqual(len(prices), 1)
        self.assertEqual(prices.loc['AAPL', 'price'], 150.0)
        
        # Test multiple symbols
        prices = self.service.get_latest_prices(['AAPL', 'MSFT'])
        self.assertEqual(len(prices), 2)
        self.assertEqual(prices.loc['MSFT', 'price'], 250.0)
        
        # Test all symbols
        prices = self.service.get_latest_prices()
        self.assertEqual(len(prices), 2)

if __name__ == '__main__':
    unittest.main()
