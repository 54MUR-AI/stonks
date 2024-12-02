import numpy as np
import pandas as pd
from typing import List, Dict, Optional

def calculate_sma(data: pd.Series, period: int = 20) -> List[Dict[str, float]]:
    """Calculate Simple Moving Average"""
    sma = data.rolling(window=period).mean()
    return [
        {"time": index.strftime("%Y-%m-%d"), "value": value}
        for index, value in sma.items()
        if not pd.isna(value)
    ]

def calculate_ema(data: pd.Series, period: int = 20) -> List[Dict[str, float]]:
    """Calculate Exponential Moving Average"""
    ema = data.ewm(span=period, adjust=False).mean()
    return [
        {"time": index.strftime("%Y-%m-%d"), "value": value}
        for index, value in ema.items()
        if not pd.isna(value)
    ]

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, List[Dict[str, float]]]:
    """Calculate Bollinger Bands"""
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
        "lower": [
            {"time": index.strftime("%Y-%m-%d"), "value": value}
            for index, value in lower_band.items()
            if not pd.isna(value)
        ]
    }

def calculate_rsi(data: pd.Series, period: int = 14) -> List[Dict[str, float]]:
    """Calculate Relative Strength Index"""
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

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[Dict[str, float]]]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast_period, adjust=False).mean()
    exp2 = data.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
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

def calculate_all_indicators(data: pd.Series, selected_indicators: List[str]) -> Dict[str, any]:
    """Calculate all requested technical indicators"""
    indicators = {}
    
    for indicator in selected_indicators:
        if indicator == 'SMA':
            indicators['SMA'] = calculate_sma(data)
        elif indicator == 'EMA':
            indicators['EMA'] = calculate_ema(data)
        elif indicator == 'BB':
            indicators['BB'] = calculate_bollinger_bands(data)
        elif indicator == 'RSI':
            indicators['RSI'] = calculate_rsi(data)
        elif indicator == 'MACD':
            indicators['MACD'] = calculate_macd(data)
    
    return indicators
