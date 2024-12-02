import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from ..schemas.indicators import TechnicalIndicator, IndicatorParams

def calculate_sma(data: pd.Series, params: Optional[IndicatorParams] = None) -> List[Dict[str, float]]:
    """Calculate Simple Moving Average"""
    period = params.period if params and params.period else 20
    sma = data.rolling(window=period).mean()
    return [
        {"time": index.strftime("%Y-%m-%d"), "value": value}
        for index, value in sma.items()
        if not pd.isna(value)
    ]

def calculate_ema(data: pd.Series, params: Optional[IndicatorParams] = None) -> List[Dict[str, float]]:
    """Calculate Exponential Moving Average"""
    period = params.period if params and params.period else 20
    ema = data.ewm(span=period, adjust=False).mean()
    return [
        {"time": index.strftime("%Y-%m-%d"), "value": value}
        for index, value in ema.items()
        if not pd.isna(value)
    ]

def calculate_bollinger_bands(data: pd.Series, params: Optional[IndicatorParams] = None) -> Dict[str, List[Dict[str, float]]]:
    """Calculate Bollinger Bands"""
    period = params.period if params and params.period else 20
    std_dev = params.std_dev if params and params.std_dev else 2
    
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return {
        "upper": [
            {"time": index.strftime("%Y-%m-%d"), "value": value}
            for index, value in upper_band.items()
            if not pd.isna(value)
        ],
        "middle": [
            {"time": index.strftime("%Y-%m-%d"), "value": value}
            for index, value in sma.items()
            if not pd.isna(value)
        ],
        "lower": [
            {"time": index.strftime("%Y-%m-%d"), "value": value}
            for index, value in lower_band.items()
            if not pd.isna(value)
        ]
    }

def calculate_rsi(data: pd.Series, params: Optional[IndicatorParams] = None) -> List[Dict[str, float]]:
    """Calculate Relative Strength Index"""
    period = params.rsi_period if params and params.rsi_period else 14
    
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return [
        {"time": index.strftime("%Y-%m-%d"), "value": value}
        for index, value in rsi.items()
        if not pd.isna(value)
    ]

def calculate_macd(data: pd.Series, params: Optional[IndicatorParams] = None) -> Dict[str, List[Dict[str, float]]]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    fast_period = params.fast_period if params and params.fast_period else 12
    slow_period = params.slow_period if params and params.slow_period else 26
    signal_period = params.signal_period if params and params.signal_period else 9
    
    ma_type = params.ma_type if params and params.ma_type else "EMA"
    
    if ma_type == "SMA":
        fast_ma = data.rolling(window=fast_period).mean()
        slow_ma = data.rolling(window=slow_period).mean()
    else:  # Default to EMA
        fast_ma = data.ewm(span=fast_period, adjust=False).mean()
        slow_ma = data.ewm(span=slow_period, adjust=False).mean()
    
    macd = fast_ma - slow_ma
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    
    return {
        "macd": [
            {"time": index.strftime("%Y-%m-%d"), "value": value}
            for index, value in macd.items()
            if not pd.isna(value)
        ],
        "signal": [
            {"time": index.strftime("%Y-%m-%d"), "value": value}
            for index, value in signal.items()
            if not pd.isna(value)
        ],
        "histogram": [
            {"time": index.strftime("%Y-%m-%d"), "value": value}
            for index, value in histogram.items()
            if not pd.isna(value)
        ]
    }

def calculate_all_indicators(data: pd.Series, indicators: List[TechnicalIndicator]) -> Dict[str, any]:
    """Calculate all requested technical indicators with custom parameters"""
    results = {}
    
    for indicator in indicators:
        if indicator.name == 'SMA':
            results['SMA'] = {
                "data": calculate_sma(data, indicator.params),
                "color": indicator.color,
                "visible": indicator.visible
            }
        elif indicator.name == 'EMA':
            results['EMA'] = {
                "data": calculate_ema(data, indicator.params),
                "color": indicator.color,
                "visible": indicator.visible
            }
        elif indicator.name == 'BB':
            bb_data = calculate_bollinger_bands(data, indicator.params)
            results['BB'] = {
                "data": bb_data,
                "color": indicator.color,
                "visible": indicator.visible
            }
        elif indicator.name == 'RSI':
            results['RSI'] = {
                "data": calculate_rsi(data, indicator.params),
                "color": indicator.color,
                "visible": indicator.visible
            }
        elif indicator.name == 'MACD':
            macd_data = calculate_macd(data, indicator.params)
            results['MACD'] = {
                "data": macd_data,
                "color": indicator.color,
                "visible": indicator.visible
            }
    
    return results
