import pytest
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from backend.services.market_data.providers.yahoo_finance import YahooFinanceProvider
from backend.services.market_data.provider import MarketDataError, ProviderStatus

@pytest.fixture
async def provider():
    provider = YahooFinanceProvider(
        request_timeout=1,
        max_retries=2,
        retry_delay=0.1,
        max_symbols_per_request=2
    )
    yield provider
    await provider.shutdown()

@pytest.fixture
def mock_ticker():
    with patch('yfinance.Ticker') as mock:
        yield mock

@pytest.fixture
def sample_price_data():
    return pd.DataFrame({
        'Close': [100.0, 101.0, 102.0],
        'Open': [99.0, 100.0, 101.0],
        'High': [102.0, 103.0, 104.0],
        'Low': [98.0, 99.0, 100.0],
        'Volume': [1000000, 1100000, 1200000]
    }, index=pd.date_range(end=datetime.now(), periods=3, freq='D'))

@pytest.fixture
def sample_symbol_info():
    return {
        'longName': 'Test Company',
        'exchange': 'NYSE',
        'currency': 'USD',
        'quoteType': 'EQUITY',
        'marketCap': 1000000000,
        'volume': 1000000,
        'sector': 'Technology',
        'industry': 'Software'
    }

@pytest.mark.asyncio
async def test_initialization(provider: YahooFinanceProvider, mock_ticker):
    # Setup mock
    mock_instance = Mock()
    mock_instance.history.return_value = pd.DataFrame({'Close': [100.0]})
    mock_ticker.return_value = mock_instance

    # Test initialization
    await provider.initialize()
    assert provider.status == ProviderStatus.READY
    mock_ticker.assert_called_once_with('SPY')

@pytest.mark.asyncio
async def test_initialization_failure(provider: YahooFinanceProvider, mock_ticker):
    # Setup mock to fail
    mock_ticker.side_effect = Exception("Connection failed")

    # Test initialization failure
    with pytest.raises(MarketDataError):
        await provider.initialize()
    assert provider.status == ProviderStatus.ERROR

@pytest.mark.asyncio
async def test_get_latest_price(
    provider: YahooFinanceProvider,
    mock_ticker,
    sample_price_data: pd.DataFrame
):
    # Setup mock
    mock_instance = Mock()
    mock_instance.history.return_value = sample_price_data
    mock_ticker.return_value = mock_instance

    # Test getting latest price
    price = await provider.get_latest_price('AAPL')
    assert price == 102.0
    mock_ticker.assert_called_once_with('AAPL')

@pytest.mark.asyncio
async def test_get_latest_price_retry(
    provider: YahooFinanceProvider,
    mock_ticker
):
    # Setup mock to fail once then succeed
    mock_instance = Mock()
    mock_instance.history.side_effect = [
        Exception("Temporary error"),
        pd.DataFrame({'Close': [100.0]})
    ]
    mock_ticker.return_value = mock_instance

    # Test retry behavior
    price = await provider.get_latest_price('AAPL')
    assert price == 100.0
    assert mock_instance.history.call_count == 2

@pytest.mark.asyncio
async def test_get_latest_price_all_retries_fail(
    provider: YahooFinanceProvider,
    mock_ticker
):
    # Setup mock to always fail
    mock_instance = Mock()
    mock_instance.history.side_effect = Exception("Persistent error")
    mock_ticker.return_value = mock_instance

    # Test all retries failing
    with pytest.raises(MarketDataError):
        await provider.get_latest_price('AAPL')
    assert mock_instance.history.call_count == provider.max_retries

@pytest.mark.asyncio
async def test_get_latest_prices_batching(
    provider: YahooFinanceProvider,
    mock_ticker,
    sample_price_data: pd.DataFrame
):
    # Setup mock
    mock_instance = Mock()
    mock_instance.history.return_value = sample_price_data
    mock_ticker.return_value = mock_instance

    # Test batch processing
    prices = await provider.get_latest_prices(['AAPL', 'GOOGL', 'MSFT'])
    assert len(prices) == 3
    assert all(price == 102.0 for price in prices.values())
    assert mock_ticker.call_count == 3

@pytest.mark.asyncio
async def test_get_historical_prices(
    provider: YahooFinanceProvider,
    mock_ticker,
    sample_price_data: pd.DataFrame
):
    # Setup mock
    mock_instance = Mock()
    mock_instance.history.return_value = sample_price_data
    mock_ticker.return_value = mock_instance

    # Test historical data retrieval
    start_date = datetime.now() - timedelta(days=3)
    end_date = datetime.now()
    
    df = await provider.get_historical_prices(
        ['AAPL', 'GOOGL'],
        start_date,
        end_date
    )
    
    assert not df.empty
    assert len(df.columns) == 2
    assert mock_ticker.call_count == 2

@pytest.mark.asyncio
async def test_get_symbol_info(
    provider: YahooFinanceProvider,
    mock_ticker,
    sample_symbol_info: Dict[str, Any]
):
    # Setup mock
    mock_instance = Mock()
    mock_instance.info = sample_symbol_info
    mock_ticker.return_value = mock_instance

    # Test symbol info retrieval
    info = await provider.get_symbol_info('AAPL')
    
    assert info['symbol'] == 'AAPL'
    assert info['name'] == sample_symbol_info['longName']
    assert info['exchange'] == sample_symbol_info['exchange']
    assert info['currency'] == sample_symbol_info['currency']
    assert info['market_cap'] == sample_symbol_info['marketCap']

@pytest.mark.asyncio
async def test_provider_properties(provider: YahooFinanceProvider):
    assert provider.provider_name == "Yahoo Finance"
    assert "1d" in provider.supported_intervals
    assert len(provider.supported_intervals) == 8

@pytest.mark.asyncio
async def test_cache_behavior(
    provider: YahooFinanceProvider,
    mock_ticker,
    sample_price_data: pd.DataFrame
):
    # Setup mock
    mock_instance = Mock()
    mock_instance.history.return_value = sample_price_data
    mock_ticker.return_value = mock_instance

    # First call should hit the API
    price1 = await provider.get_latest_price('AAPL')
    assert mock_instance.history.call_count == 1

    # Second call should use cache
    price2 = await provider.get_latest_price('AAPL')
    assert mock_instance.history.call_count == 1
    assert price1 == price2

@pytest.mark.asyncio
async def test_invalid_interval(provider: YahooFinanceProvider):
    start_date = datetime.now() - timedelta(days=3)
    end_date = datetime.now()
    
    with pytest.raises(MarketDataError):
        await provider.get_historical_prices(
            ['AAPL'],
            start_date,
            end_date,
            interval='invalid'
        )

@pytest.mark.asyncio
async def test_empty_symbols_list(provider: YahooFinanceProvider):
    prices = await provider.get_latest_prices([])
    assert isinstance(prices, dict)
    assert len(prices) == 0

@pytest.mark.asyncio
async def test_shutdown(provider: YahooFinanceProvider):
    await provider.shutdown()
    assert provider.status == ProviderStatus.STOPPED
