"""Unit tests for the market data adapter.

This test suite provides comprehensive coverage of the market data adapter
implementation, including provider integration, streaming, and error handling.
"""

import pytest
import pytest_asyncio
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

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

    @pytest_asyncio.fixture
    async def adapter(self, mock_realtime_service, mock_config):
        """Create and configure a market data adapter for testing."""
        adapter = MarketDataAdapter(
            provider_class=MockProvider,
            config=mock_config,
            realtime_service=mock_realtime_service
        )
        # Configure mocks
        adapter.provider.connect = AsyncMock()
        adapter.provider.disconnect = AsyncMock()
        adapter.provider.subscribe = AsyncMock()
        adapter.provider.unsubscribe = AsyncMock()
        adapter.provider.get_historical_data = AsyncMock()
        adapter.provider.get_latest_quote = AsyncMock()
        
        # Set up market data callback
        def on_market_data(data):
            adapter.realtime_service.update_market_data(data)
        adapter.provider.on_market_data = Mock(side_effect=on_market_data)
        
        yield adapter
        
        # Cleanup
        if adapter._running:
            await adapter.stop()

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

    @pytest.mark.asyncio
    async def test_unsubscribe_error_handling(self, adapter):
        """Test error handling during unsubscribe."""
        print("\nStarting unsubscribe error test...")
        
        # Start adapter and subscribe to symbols
        await adapter.start()
        test_symbols = ["AAPL", "MSFT"]
        await adapter.subscribe(test_symbols)
        
        # Mock error in unsubscribe
        adapter.provider.unsubscribe = AsyncMock(side_effect=Exception("Unsubscribe error"))
        
        # Mock error handler
        error_received = asyncio.Event()
        def error_handler(error):
            assert str(error) == "Unsubscribe error"
            error_received.set()
        adapter.on_error = error_handler
        
        # Attempt to unsubscribe should trigger error handler
        with pytest.raises(Exception):
            await adapter.unsubscribe(["AAPL"])
        
        # Wait for error handler
        await asyncio.wait_for(error_received.wait(), timeout=1.0)
        
        # Verify symbols are still subscribed
        assert "AAPL" in adapter._subscribed_symbols
        
        await adapter.stop()
        print("Unsubscribe error test complete")

    @pytest.mark.asyncio
    async def test_historical_data_error_handling(self, adapter):
        """Test error handling during historical data retrieval."""
        print("\nStarting historical data error test...")
        
        # Mock error in get_historical_data
        test_error = Exception("Historical data error")
        adapter.provider.get_historical_data = AsyncMock(side_effect=test_error)
        
        # Mock error handler
        error_received = asyncio.Event()
        def error_handler(error):
            assert error == test_error
            error_received.set()
        adapter.on_error = error_handler
        
        # Attempt to get historical data should trigger error handler
        with pytest.raises(Exception):
            await adapter.get_historical_data("AAPL", timedelta(days=1))
        
        # Wait for error handler
        await asyncio.wait_for(error_received.wait(), timeout=1.0)
        print("Historical data error test complete")

    @pytest.mark.asyncio
    async def test_custom_error_handler(self, adapter):
        """Test custom error handler configuration."""
        print("\nStarting custom error handler test...")
        
        # Create custom error handler
        errors = []
        def custom_handler(error):
            errors.append(error)
        
        # Configure adapter with custom handler
        adapter.on_error = custom_handler
        
        # Trigger error
        test_error = Exception("Test error")
        adapter.provider.connect = AsyncMock(side_effect=test_error)
        
        # Start should trigger error handler
        with pytest.raises(Exception):
            await adapter.start()
        
        # Verify custom handler was called
        assert len(errors) == 1
        assert errors[0] == test_error
        print("Custom error handler test complete")

    @pytest.mark.asyncio
    async def test_update_loop_cancellation(self, adapter):
        """Test graceful cancellation of update loop."""
        print("\nStarting update loop cancellation test...")
        
        # Start adapter
        await adapter.start()
        assert adapter._update_task is not None
        
        # Force cancel update task
        adapter._update_task.cancel()
        
        # Wait for task to be cancelled
        await asyncio.sleep(0.1)
        assert adapter._update_task.cancelled()
        
        # Stop adapter
        await adapter.stop()
        assert adapter._update_task is None
        print("Update loop cancellation test complete")

    @pytest.mark.asyncio
    async def test_historical_data_interval(self, adapter):
        """Test historical data retrieval with different intervals."""
        print("\nStarting historical data interval test...")
        
        # Mock historical data response
        mock_data = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000000]
        })
        adapter.provider.get_historical_data = AsyncMock(return_value=mock_data)
        
        # Test different intervals
        intervals = ["1min", "5min", "15min", "1h", "1d"]
        for interval in intervals:
            # Get current time before call
            before = pd.Timestamp.now()
            
            data = await adapter.get_historical_data(
                symbol="AAPL",
                lookback=timedelta(days=1),
                interval=interval
            )
            
            # Get current time after call
            after = pd.Timestamp.now()
            
            # Get the last call arguments
            args = adapter.provider.get_historical_data.call_args[0]
            assert args[0] == "AAPL"  # symbol
            
            # Verify start_date is within expected range
            start_date = args[1]
            assert isinstance(start_date, datetime)
            assert before - timedelta(days=1) <= start_date <= after - timedelta(days=1)
            
            # Verify end_date is within expected range
            end_date = args[2]
            assert isinstance(end_date, datetime)
            assert before <= end_date <= after
            
            # Verify interval
            assert args[3] == interval
            
            # Verify return value
            assert isinstance(data, pd.DataFrame)
            assert len(data) == len(mock_data)
        
        print("Historical data interval test complete")

    @pytest.mark.asyncio
    async def test_realtime_service_integration(self, adapter, mock_realtime_service):
        """Test integration with RealTimeDataService."""
        print("\nStarting realtime service integration test...")
        
        # Configure realtime service mock
        mock_realtime_service.update_market_data = Mock()
        
        # Test start sequence
        await adapter.start()
        mock_realtime_service.start.assert_called_once()
        
        # Subscribe to symbols
        test_symbols = ["AAPL", "MSFT"]
        await adapter.subscribe(test_symbols)
        
        # Simulate market data update
        mock_data = {
            "AAPL": {"price": 150.0, "volume": 1000000},
            "MSFT": {"price": 300.0, "volume": 500000}
        }
        adapter.provider.on_market_data(mock_data)
        
        # Verify realtime service received update
        mock_realtime_service.update_market_data.assert_called_with(mock_data)
        
        # Test stop sequence
        await adapter.stop()
        mock_realtime_service.stop.assert_called_once()
        print("Realtime service integration test complete")
