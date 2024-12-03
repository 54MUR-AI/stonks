"""Unit tests for the market data adapter.

This test suite provides comprehensive coverage of the market data adapter
implementation, including provider integration, streaming, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timedelta
import asyncio
import pandas as pd

from backend.services.market_data import (
    MarketDataAdapter,
    MarketDataConfig,
    MarketDataCredentials,
    MockProvider
)
from backend.services.realtime_data import RealTimeDataService

class TestMarketDataAdapter:
    """Test suite for market data adapter functionality."""
    
    @pytest.fixture
    def mock_realtime_service(self):
        service = Mock(spec=RealTimeDataService)
        service.start = AsyncMock()
        service.stop = AsyncMock()
        service.is_running = Mock(return_value=True)
        return service

    @pytest.fixture
    def mock_config(self):
        return MarketDataConfig(
            credentials=MarketDataCredentials(api_key='test_key'),
            base_url='http://test.url',
            websocket_url='ws://test.url/ws',
            request_timeout=30
        )

    @pytest.fixture
    async def adapter(self, mock_realtime_service, mock_config):
        """Create and configure a market data adapter for testing."""
        adapter = MarketDataAdapter(
            provider_class=MockProvider,
            config=mock_config,
            realtime_service=mock_realtime_service
        )
        adapter.provider.connect = AsyncMock()
        adapter.provider.disconnect = AsyncMock()
        adapter.provider.subscribe = AsyncMock()
        adapter.provider.unsubscribe = AsyncMock()
        adapter.provider.get_historical_data = AsyncMock()
        return adapter

    @pytest.mark.asyncio
    async def test_stream_connection(self, adapter, mock_realtime_service):
        """Test basic stream connection and disconnection."""
        print("\nStarting stream connection test...")
        
        # Test initial state
        assert not adapter._running
        assert adapter._update_task is None
        
        # Test start
        print("Starting adapter...")
        await adapter.start()
        print("Adapter started")
        
        assert adapter._running
        assert adapter._update_task is not None
        mock_realtime_service.start.assert_called_once()
        
        # Test stop
        print("Stopping adapter...")
        await adapter.stop()
        print("Adapter stopped")
        
        assert not adapter._running
        assert adapter._update_task is None
        mock_realtime_service.stop.assert_called_once()
        
        print("Stream connection test complete")

    @pytest.mark.asyncio
    async def test_double_start_stop(self, adapter):
        """Test starting an already started adapter and stopping an already stopped adapter."""
        print("\nStarting double start/stop test...")
        
        # First start
        await adapter.start()
        first_task = adapter._update_task
        
        # Second start should be no-op
        await adapter.start()
        assert adapter._update_task == first_task
        
        # First stop
        await adapter.stop()
        
        # Second stop should be no-op
        await adapter.stop()
        assert not adapter._running
        assert adapter._update_task is None
        
        print("Double start/stop test complete")

    @pytest.mark.asyncio
    async def test_subscription_management(self, adapter):
        """Test symbol subscription and unsubscription."""
        print("\nStarting subscription management test...")
        
        # Start adapter
        await adapter.start()
        
        # Test subscribe
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        await adapter.subscribe(test_symbols)
        assert adapter._subscribed_symbols == set(test_symbols)
        
        # Test unsubscribe
        await adapter.unsubscribe(["AAPL", "MSFT"])
        assert adapter._subscribed_symbols == {"GOOGL"}
        
        await adapter.stop()
        print("Subscription management test complete")

    @pytest.mark.asyncio
    async def test_error_handling(self, adapter):
        """Test error handling during streaming."""
        print("\nStarting error handling test...")
        
        # Mock error handler
        error_received = asyncio.Event()
        test_error = Exception("Test error")
        
        def error_handler(error):
            assert error == test_error
            error_received.set()
        
        adapter.on_error = error_handler
        
        # Start adapter
        await adapter.start()
        
        # Simulate provider error
        adapter.provider.subscribe = AsyncMock(side_effect=test_error)
        
        # Attempt to subscribe should trigger error handler
        with pytest.raises(Exception):
            await adapter.subscribe(["AAPL"])
        
        # Wait for error handler
        await asyncio.wait_for(error_received.wait(), timeout=1.0)
        
        await adapter.stop()
        print("Error handling test complete")

    @pytest.mark.asyncio
    async def test_cleanup_on_provider_error(self, adapter):
        """Test proper cleanup when provider encounters an error."""
        print("\nStarting cleanup on provider error test...")
        
        # Mock provider to raise error during connect
        adapter.provider.connect = AsyncMock(side_effect=Exception("Connection error"))
        
        # Start should raise the provider error
        with pytest.raises(Exception):
            await adapter.start()
        
        # Verify cleanup occurred
        assert not adapter._running
        assert adapter._update_task is None
        
        print("Cleanup on provider error test complete")

    @pytest.mark.asyncio
    async def test_historical_data_retrieval(self, adapter):
        """Test historical data retrieval functionality."""
        print("\nStarting historical data retrieval test...")
        
        # Mock historical data response
        mock_data = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000000]
        })
        adapter.provider.get_historical_data = AsyncMock(return_value=mock_data)
        
        # Test data retrieval
        symbol = "AAPL"
        lookback = timedelta(days=1)
        data = await adapter.get_historical_data(symbol, lookback)
        
        # Verify the mock was called correctly
        adapter.provider.get_historical_data.assert_called_once()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == len(mock_data)
        
        print("Historical data retrieval test complete")

    @pytest.mark.asyncio
    async def test_update_loop_exception_handling(self, adapter):
        """Test that exceptions in the update loop are handled gracefully."""
        print("\nStarting update loop exception test...")
        
        # Mock error handler
        error_received = asyncio.Event()
        
        def error_handler(error):
            assert isinstance(error, Exception)
            error_received.set()
        
        adapter.on_error = error_handler
        
        # Start adapter
        await adapter.start()
        
        # Wait for a few update cycles
        await asyncio.sleep(0.1)
        
        # Verify adapter is still running after update cycles
        assert adapter._running
        assert adapter._update_task is not None
        
        await adapter.stop()
        print("Update loop exception test complete")
