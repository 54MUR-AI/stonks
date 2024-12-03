"""Tests for technical indicators."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.services.analysis.indicators import TechnicalIndicators, IndicatorResult

@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate sample price data with a trend and some volatility
    prices = pd.Series(
        np.cumsum(np.random.normal(0.1, 1, 100)) + 100,
        index=dates
    )
    
    # Generate sample OHLCV data
    high = prices + np.random.uniform(0, 2, 100)
    low = prices - np.random.uniform(0, 2, 100)
    close = prices
    volume = pd.Series(
        np.random.uniform(1000000, 5000000, 100),
        index=dates
    )
    
    return {
        'close': close,
        'high': high,
        'low': low,
        'volume': volume
    }

def test_sma(sample_data):
    """Test Simple Moving Average calculation."""
    close = sample_data['close']
    result = TechnicalIndicators.sma(close, period=20)
    
    assert isinstance(result, IndicatorResult)
    assert result.name == "SMA"
    assert isinstance(result.values, pd.Series)
    assert result.metadata['period'] == 20
    assert len(result.values) == len(close)
    assert pd.isna(result.values[:19]).all()  # First n-1 values should be NaN
    assert not pd.isna(result.values[19:]).any()  # Rest should be calculated

def test_ema(sample_data):
    """Test Exponential Moving Average calculation."""
    close = sample_data['close']
    result = TechnicalIndicators.ema(close, period=20)
    
    assert isinstance(result, IndicatorResult)
    assert result.name == "EMA"
    assert isinstance(result.values, pd.Series)
    assert result.metadata['period'] == 20
    assert len(result.values) == len(close)
    assert not pd.isna(result.values[1:]).any()  # Only first value should be NaN

def test_rsi(sample_data):
    """Test Relative Strength Index calculation."""
    close = sample_data['close']
    result = TechnicalIndicators.rsi(close, period=14)
    
    assert isinstance(result, IndicatorResult)
    assert result.name == "RSI"
    assert isinstance(result.values, pd.Series)
    assert result.metadata['period'] == 14
    assert len(result.values) == len(close)
    assert pd.isna(result.values[:14]).all()  # First n values should be NaN
    assert not pd.isna(result.values[14:]).any()  # Rest should be calculated
    assert (result.values[14:] >= 0).all()  # RSI should be >= 0
    assert (result.values[14:] <= 100).all()  # RSI should be <= 100

def test_macd(sample_data):
    """Test MACD calculation."""
    close = sample_data['close']
    result = TechnicalIndicators.macd(close)
    
    assert isinstance(result, IndicatorResult)
    assert result.name == "MACD"
    assert isinstance(result.values, pd.DataFrame)
    assert all(col in result.values.columns for col in ['macd', 'signal', 'histogram'])
    assert result.metadata['fast_period'] == 12
    assert result.metadata['slow_period'] == 26
    assert result.metadata['signal_period'] == 9
    assert len(result.values) == len(close)

def test_bollinger_bands(sample_data):
    """Test Bollinger Bands calculation."""
    close = sample_data['close']
    result = TechnicalIndicators.bollinger_bands(close)
    
    assert isinstance(result, IndicatorResult)
    assert result.name == "Bollinger Bands"
    assert isinstance(result.values, pd.DataFrame)
    assert all(col in result.values.columns for col in ['middle', 'upper', 'lower'])
    assert result.metadata['period'] == 20
    assert result.metadata['std_dev'] == 2.0
    assert len(result.values) == len(close)
    assert (result.values['upper'] >= result.values['middle']).all()
    assert (result.values['lower'] <= result.values['middle']).all()

def test_atr(sample_data):
    """Test Average True Range calculation."""
    result = TechnicalIndicators.atr(
        sample_data['high'],
        sample_data['low'],
        sample_data['close']
    )
    
    assert isinstance(result, IndicatorResult)
    assert result.name == "ATR"
    assert isinstance(result.values, pd.Series)
    assert result.metadata['period'] == 14
    assert len(result.values) == len(sample_data['close'])
    assert pd.isna(result.values[:14]).all()  # First n values should be NaN
    assert not pd.isna(result.values[14:]).any()  # Rest should be calculated
    assert (result.values[14:] >= 0).all()  # ATR should be >= 0

def test_stochastic(sample_data):
    """Test Stochastic Oscillator calculation."""
    result = TechnicalIndicators.stochastic(
        sample_data['high'],
        sample_data['low'],
        sample_data['close']
    )
    
    assert isinstance(result, IndicatorResult)
    assert result.name == "Stochastic"
    assert isinstance(result.values, pd.DataFrame)
    assert all(col in result.values.columns for col in ['k', 'd'])
    assert result.metadata['k_period'] == 14
    assert result.metadata['d_period'] == 3
    assert len(result.values) == len(sample_data['close'])
    assert (result.values['k'][14:] >= 0).all()  # %K should be >= 0
    assert (result.values['k'][14:] <= 100).all()  # %K should be <= 100
    assert (result.values['d'][16:] >= 0).all()  # %D should be >= 0
    assert (result.values['d'][16:] <= 100).all()  # %D should be <= 100

def test_obv(sample_data):
    """Test On-Balance Volume calculation."""
    result = TechnicalIndicators.obv(
        sample_data['close'],
        sample_data['volume']
    )
    
    assert isinstance(result, IndicatorResult)
    assert result.name == "OBV"
    assert isinstance(result.values, pd.Series)
    assert len(result.values) == len(sample_data['close'])
    assert not pd.isna(result.values).any()  # No NaN values
    assert result.values.dtype == float  # Should be float type

def test_adx(sample_data):
    """Test Average Directional Index calculation."""
    result = TechnicalIndicators.adx(
        sample_data['high'],
        sample_data['low'],
        sample_data['close']
    )
    
    assert isinstance(result, IndicatorResult)
    assert result.name == "ADX"
    assert isinstance(result.values, pd.DataFrame)
    assert all(col in result.values.columns for col in ['adx', '+di', '-di'])
    assert result.metadata['period'] == 14
    assert len(result.values) == len(sample_data['close'])
    assert (result.values['adx'][28:] >= 0).all()  # ADX should be >= 0
    assert (result.values['adx'][28:] <= 100).all()  # ADX should be <= 100
    assert (result.values['+di'][14:] >= 0).all()  # +DI should be >= 0
    assert (result.values['-di'][14:] >= 0).all()  # -DI should be >= 0
