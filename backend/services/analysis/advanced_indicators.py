"""Advanced technical indicators for market analysis."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
from .indicators import IndicatorResult

class AdvancedIndicators:
    """Collection of advanced technical analysis indicators."""
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                 tenkan_period: int = 9, kijun_period: int = 26,
                 senkou_b_period: int = 52, displacement: int = 26) -> IndicatorResult:
        """Ichimoku Cloud indicator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            tenkan_period: Conversion line period
            kijun_period: Base line period
            senkou_b_period: Leading Span B period
            displacement: Displacement period for cloud
        """
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=senkou_b_period).max()
        senkou_low = low.rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(displacement)

        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-displacement)

        result = pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })

        return IndicatorResult(
            name="Ichimoku Cloud",
            values=result,
            metadata={
                'tenkan_period': tenkan_period,
                'kijun_period': kijun_period,
                'senkou_b_period': senkou_b_period,
                'displacement': displacement
            }
        )

    @staticmethod
    def fibonacci_retracements(high: pd.Series, low: pd.Series,
                             trend: str = 'auto') -> IndicatorResult:
        """Calculate Fibonacci retracement levels.
        
        Args:
            high: High prices
            low: Low prices
            trend: Trend direction ('up', 'down', or 'auto')
        """
        if trend == 'auto':
            # Determine trend based on first and last prices
            trend = 'up' if high.iloc[-1] > high.iloc[0] else 'down'

        # Find swing high and low
        swing_high = high.max()
        swing_low = low.min()
        price_range = swing_high - swing_low

        # Calculate retracement levels
        levels = {
            '0.0': swing_low if trend == 'up' else swing_high,
            '0.236': swing_high - price_range * 0.236 if trend == 'up' else swing_low + price_range * 0.236,
            '0.382': swing_high - price_range * 0.382 if trend == 'up' else swing_low + price_range * 0.382,
            '0.5': swing_high - price_range * 0.5 if trend == 'up' else swing_low + price_range * 0.5,
            '0.618': swing_high - price_range * 0.618 if trend == 'up' else swing_low + price_range * 0.618,
            '0.786': swing_high - price_range * 0.786 if trend == 'up' else swing_low + price_range * 0.786,
            '1.0': swing_high if trend == 'up' else swing_low
        }

        # Create DataFrame with retracement levels repeated for each date
        levels_df = pd.DataFrame({
            level: value for level, value in levels.items()
        }, index=high.index)

        return IndicatorResult(
            name="Fibonacci Retracements",
            values=levels_df,
            metadata={
                'trend': trend,
                'swing_high': swing_high,
                'swing_low': swing_low
            }
        )

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, reset_period: str = 'D') -> IndicatorResult:
        """Volume Weighted Average Price.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            reset_period: Frequency to reset VWAP calculation ('D' for daily)
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).groupby(
            pd.Grouper(freq=reset_period)
        ).cumsum() / volume.groupby(
            pd.Grouper(freq=reset_period)
        ).cumsum()

        return IndicatorResult(
            name="VWAP",
            values=vwap,
            metadata={'reset_period': reset_period}
        )

    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series,
                    method: str = 'standard') -> IndicatorResult:
        """Calculate pivot points and support/resistance levels.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            method: Calculation method ('standard', 'fibonacci', 'woodie', 'camarilla')
        """
        def calculate_levels(h: float, l: float, c: float) -> Dict[str, float]:
            p = (h + l + c) / 3  # Pivot point

            if method == 'standard':
                r1 = 2 * p - l
                r2 = p + (h - l)
                r3 = h + 2 * (p - l)
                s1 = 2 * p - h
                s2 = p - (h - l)
                s3 = l - 2 * (h - p)
            
            elif method == 'fibonacci':
                r1 = p + 0.382 * (h - l)
                r2 = p + 0.618 * (h - l)
                r3 = p + 1.000 * (h - l)
                s1 = p - 0.382 * (h - l)
                s2 = p - 0.618 * (h - l)
                s3 = p - 1.000 * (h - l)
            
            elif method == 'woodie':
                p = (h + l + 2 * c) / 4
                r1 = 2 * p - l
                r2 = p + (h - l)
                s1 = 2 * p - h
                s2 = p - (h - l)
                r3 = h + 2 * (p - l)
                s3 = l - 2 * (h - p)
            
            elif method == 'camarilla':
                r1 = c + ((h - l) * 1.1 / 12)
                r2 = c + ((h - l) * 1.1 / 6)
                r3 = c + ((h - l) * 1.1 / 4)
                s1 = c - ((h - l) * 1.1 / 12)
                s2 = c - ((h - l) * 1.1 / 6)
                s3 = c - ((h - l) * 1.1 / 4)

            return {
                'pivot': p,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3
            }

        # Calculate pivot points for each day
        levels = pd.DataFrame(index=high.index)
        daily_data = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        }).resample('D').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        for date, row in daily_data.iterrows():
            day_levels = calculate_levels(row['high'], row['low'], row['close'])
            next_day = date + pd.Timedelta(days=1)
            if next_day in levels.index:
                for level, value in day_levels.items():
                    levels.loc[next_day:next_day + pd.Timedelta(days=1), level] = value

        return IndicatorResult(
            name="Pivot Points",
            values=levels.ffill(),
            metadata={'method': method}
        )

    @staticmethod
    def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series,
                  ema_period: int = 13) -> IndicatorResult:
        """Elder Ray indicator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            ema_period: EMA period for calculation
        """
        ema = close.ewm(span=ema_period, adjust=False).mean()
        bull_power = high - ema
        bear_power = low - ema

        result = pd.DataFrame({
            'ema': ema,
            'bull_power': bull_power,
            'bear_power': bear_power
        })

        return IndicatorResult(
            name="Elder Ray",
            values=result,
            metadata={'ema_period': ema_period}
        )

    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 10, multiplier: float = 3.0) -> IndicatorResult:
        """SuperTrend indicator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            multiplier: ATR multiplier
        """
        # Calculate ATR
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()

        # Calculate SuperTrend
        hl2 = (high + low) / 2
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)

        for i in range(1, len(close)):
            if close[i] > upperband[i-1]:
                supertrend[i] = lowerband[i]
                direction[i] = 1
            elif close[i] < lowerband[i-1]:
                supertrend[i] = upperband[i]
                direction[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1]

                if direction[i] == 1 and lowerband[i] < supertrend[i]:
                    supertrend[i] = lowerband[i]
                elif direction[i] == -1 and upperband[i] > supertrend[i]:
                    supertrend[i] = upperband[i]

        result = pd.DataFrame({
            'supertrend': supertrend,
            'direction': direction,
            'upperband': upperband,
            'lowerband': lowerband
        })

        return IndicatorResult(
            name="SuperTrend",
            values=result,
            metadata={
                'period': period,
                'multiplier': multiplier
            }
        )
