"""Tests for Polygon.io market data provider."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from aiohttp import ClientSession, ClientResponse, WSMessage, WSMsgType
import json

from backend.services.market_data.providers.polygon import (
    PolygonProvider,
    PolygonConfig,
    MarketDataError,
    ProviderStatus
)

@pytest.fixture
def config():
    return PolygonConfig(
        api_key="test_api_key",
        request_timeout=1,
        max_retries=2,
        retry_delay=0.1
    )

@pytest.fixture
async def provider(config):
    provider = PolygonProvider(config)
    yield provider
    await provider.shutdown()

@pytest.fixture
def mock_response():
    response = Mock(spec=ClientResponse)
    response.status = 200
    response.headers = {}
    return response

@pytest.fixture
def mock_session(mock_response):
    session = Mock(spec=ClientSession)
    context_manager = AsyncMock()
    context_manager.__aenter__.return_value = mock_response
    session.get.return_value = context_manager
    return session

@pytest.mark.asyncio
async def test_initialization(provider, mock_session):
    with patch('aiohttp.ClientSession', return_value=mock_session):
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(
            return_value={
                "status": "success",
                "results": {"p": 150.0}
            }
        )
        
        await provider.initialize()
        assert provider.status == ProviderStatus.READY
        assert mock_session.get.called

@pytest.mark.asyncio
async def test_initialization_failure(provider, mock_session):
    with patch('aiohttp.ClientSession', return_value=mock_session):
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        with pytest.raises(MarketDataError):
            await provider.initialize()
        assert provider.status == ProviderStatus.ERROR

@pytest.mark.asyncio
async def test_get_latest_price(provider, mock_session):
    with patch('aiohttp.ClientSession', return_value=mock_session):
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(
            return_value={
                "status": "success",
                "results": {"p": 150.0}
            }
        )
        
        await provider.initialize()
        price = await provider.get_latest_price("AAPL")
        assert price == 150.0

@pytest.mark.asyncio
async def test_get_latest_price_retry(provider, mock_session):
    with patch('aiohttp.ClientSession', return_value=mock_session):
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(
            side_effect=[
                Exception("Temporary error"),
                {"status": "success", "results": {"p": 150.0}}
            ]
        )
        
        await provider.initialize()
        price = await provider.get_latest_price("AAPL")
        assert price == 150.0
        assert mock_session.get.call_count >= 2

@pytest.mark.asyncio
async def test_get_latest_price_rate_limit(provider, mock_session, mock_response):
    with patch('aiohttp.ClientSession', return_value=mock_session):
        mock_response.status = 429
        mock_response.headers = {'Retry-After': '1'}
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        await provider.initialize()
        with pytest.raises(MarketDataError):
            await provider.get_latest_price("AAPL")

@pytest.mark.asyncio
async def test_get_historical_prices(provider, mock_session):
    historical_data = {
        "status": "success",
        "results": [
            {"t": 1625097600000, "c": 150.0},
            {"t": 1625184000000, "c": 151.0},
            {"t": 1625270400000, "c": 152.0}
        ]
    }
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(
            return_value=historical_data
        )
        
        await provider.initialize()
        start_date = datetime(2021, 7, 1)
        end_date = datetime(2021, 7, 3)
        
        df = await provider.get_historical_prices(
            ["AAPL"],
            start_date,
            end_date,
            "1d"
        )
        
        assert not df.empty
        assert len(df) == 3
        assert df.iloc[-1]["AAPL"] == 152.0

@pytest.mark.asyncio
async def test_get_symbol_info(provider, mock_session):
    symbol_info = {
        "status": "success",
        "results": {
            "name": "Apple Inc.",
            "exchange": "NASDAQ",
            "currency_name": "USD",
            "type": "CS",
            "market_cap": 2000000000000,
            "description": "Apple Inc. designs, manufactures, and markets smartphones",
            "sector": "Technology",
            "industry": "Consumer Electronics"
        }
    }
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(
            return_value=symbol_info
        )
        
        await provider.initialize()
        info = await provider.get_symbol_info("AAPL")
        
        assert info["symbol"] == "AAPL"
        assert info["name"] == "Apple Inc."
        assert info["exchange"] == "NASDAQ"
        assert info["sector"] == "Technology"

@pytest.mark.asyncio
async def test_provider_properties(provider):
    assert provider.provider_name == "Polygon.io"
    assert "1d" in provider.supported_intervals
    assert len(provider.supported_intervals) == 8

@pytest.mark.asyncio
async def test_shutdown(provider, mock_session):
    with patch('aiohttp.ClientSession', return_value=mock_session):
        await provider.initialize()
        await provider.shutdown()
        assert provider.status == ProviderStatus.STOPPED

@pytest.mark.asyncio
async def test_invalid_interval(provider, mock_session):
    with patch('aiohttp.ClientSession', return_value=mock_session):
        await provider.initialize()
        with pytest.raises(MarketDataError):
            await provider.get_historical_prices(
                ["AAPL"],
                datetime.now() - timedelta(days=1),
                datetime.now(),
                "invalid"
            )
