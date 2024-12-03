"""Technical indicators library for market analysis."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict
from dataclasses import dataclass

@dataclass
class IndicatorResult:
    """Container for indicator calculation results."""
    name: str
    values: Union[pd.Series, pd.DataFrame]
    metadata: Dict = None

class TechnicalIndicators:
    """Collection of technical analysis indicators."""
    
    @staticmethod
    def sma(data: pd.Series, period: int = 20) -> IndicatorResult:
        """Simple Moving Average.
        
        Args:
            data: Price series
            period: Moving average period
        """
        values = data.rolling(window=period).mean()
        return IndicatorResult(
            name="SMA",
            values=values,
            metadata={"period": period}
        )
    
    @staticmethod
    def ema(data: pd.Series, period: int = 20) -> IndicatorResult:
        """Exponential Moving Average.
        
        Args:
            data: Price series
            period: Moving average period
        """
        values = data.ewm(span=period, adjust=False).mean()
        return IndicatorResult(
            name="EMA",
            values=values,
            metadata={"period": period}
        )
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> IndicatorResult:
        """Relative Strength Index.
        
        Args:
            data: Price series
            period: RSI period
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return IndicatorResult(
            name="RSI",
            values=rsi,
            metadata={"period": period}
        )
    
    @staticmethod
    def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9) -> IndicatorResult:
        """Moving Average Convergence Divergence.
        
        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        """
        fast_ema = data.ewm(span=fast_period, adjust=False).mean()
        slow_ema = data.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        result = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
        
        return IndicatorResult(
            name="MACD",
            values=result,
            metadata={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period
            }
        )
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, 
                       std_dev: float = 2.0) -> IndicatorResult:
        """Bollinger Bands.
        
        Args:
            data: Price series
            period: Moving average period
            std_dev: Number of standard deviations
        """
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        result = pd.DataFrame({
            'middle': middle_band,
            'upper': upper_band,
            'lower': lower_band
        })
        
        return IndicatorResult(
            name="Bollinger Bands",
            values=result,
            metadata={
                "period": period,
                "std_dev": std_dev
            }
        )
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> IndicatorResult:
        """Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        tr = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        }).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        return IndicatorResult(
            name="ATR",
            values=atr,
            metadata={"period": period}
        )
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> IndicatorResult:
        """Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D period
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        
        result = pd.DataFrame({
            'k': k,
            'd': d
        })
        
        return IndicatorResult(
            name="Stochastic",
            values=result,
            metadata={
                "k_period": k_period,
                "d_period": d_period
            }
        )
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> IndicatorResult:
        """On-Balance Volume.
        
        Args:
            close: Close prices
            volume: Volume data
        """
        close_change = close.diff()
        
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return IndicatorResult(
            name="OBV",
            values=obv
        )
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> IndicatorResult:
        """Average Directional Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period
        """
        # True Range
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low
        
        pos_dm = pd.Series(0.0, index=up_move.index)
        neg_dm = pd.Series(0.0, index=down_move.index)
        
        pos_dm[up_move > down_move] = up_move[up_move > down_move]
        pos_dm[up_move <= 0] = 0.0
        
        neg_dm[down_move > up_move] = down_move[down_move > up_move]
        neg_dm[down_move <= 0] = 0.0
        
        # Smoothed values
        tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
        pos_dm_smooth = pos_dm.ewm(alpha=1/period, adjust=False).mean()
        neg_dm_smooth = neg_dm.ewm(alpha=1/period, adjust=False).mean()
        
        # Directional Indicators
        pdi = 100 * (pos_dm_smooth / tr_smooth)
        ndi = 100 * (neg_dm_smooth / tr_smooth)
        
        # ADX
        dx = 100 * abs(pdi - ndi) / (pdi + ndi)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        result = pd.DataFrame({
            'adx': adx,
            '+di': pdi,
            '-di': ndi
        })
        
        return IndicatorResult(
            name="ADX",
            values=result,
            metadata={"period": period}
        )
