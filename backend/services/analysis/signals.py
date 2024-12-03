"""Signal generation system for technical analysis indicators."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

from .indicators import IndicatorResult, TechnicalIndicators
from .advanced_indicators import AdvancedIndicators

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"
    WARNING = "WARNING"

@dataclass
class Signal:
    """Trading signal with metadata."""
    type: SignalType
    timestamp: pd.Timestamp
    symbol: str
    price: float
    indicator: str
    strength: float  # 0.0 to 1.0
    metadata: Dict = None

class SignalGenerator:
    """Generate trading signals from technical indicators."""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.advanced_indicators = AdvancedIndicators()
        
    def analyze_rsi(self, rsi_result: IndicatorResult,
                   overbought: float = 70,
                   oversold: float = 30) -> List[Signal]:
        """Generate signals from RSI indicator."""
        signals = []
        rsi_values = rsi_result.values
        
        for timestamp, value in rsi_values.items():
            if pd.isna(value):
                continue
                
            if value <= oversold:
                # Oversold condition - potential buy signal
                strength = 1 - (value / oversold)
                signals.append(Signal(
                    type=SignalType.BUY,
                    timestamp=timestamp,
                    symbol=None,  # Set by caller
                    price=None,   # Set by caller
                    indicator="RSI",
                    strength=strength,
                    metadata={'value': value}
                ))
            elif value >= overbought:
                # Overbought condition - potential sell signal
                strength = (value - overbought) / (100 - overbought)
                signals.append(Signal(
                    type=SignalType.SELL,
                    timestamp=timestamp,
                    symbol=None,
                    price=None,
                    indicator="RSI",
                    strength=strength,
                    metadata={'value': value}
                ))
                
        return signals
    
    def analyze_macd(self, macd_result: IndicatorResult,
                    signal_threshold: float = 0) -> List[Signal]:
        """Generate signals from MACD indicator."""
        signals = []
        macd_data = macd_result.values
        
        # Calculate crossovers
        crossover = np.sign(macd_data['macd'] - macd_data['signal']).diff()
        
        for timestamp, value in crossover.items():
            if pd.isna(value):
                continue
                
            histogram = macd_data.loc[timestamp, 'histogram']
            if value > 0:  # Bullish crossover
                strength = min(abs(histogram) / signal_threshold, 1.0) if signal_threshold > 0 else 1.0
                signals.append(Signal(
                    type=SignalType.BUY,
                    timestamp=timestamp,
                    symbol=None,
                    price=None,
                    indicator="MACD",
                    strength=strength,
                    metadata={
                        'macd': macd_data.loc[timestamp, 'macd'],
                        'signal': macd_data.loc[timestamp, 'signal'],
                        'histogram': histogram
                    }
                ))
            elif value < 0:  # Bearish crossover
                strength = min(abs(histogram) / signal_threshold, 1.0) if signal_threshold > 0 else 1.0
                signals.append(Signal(
                    type=SignalType.SELL,
                    timestamp=timestamp,
                    symbol=None,
                    price=None,
                    indicator="MACD",
                    strength=strength,
                    metadata={
                        'macd': macd_data.loc[timestamp, 'macd'],
                        'signal': macd_data.loc[timestamp, 'signal'],
                        'histogram': histogram
                    }
                ))
                
        return signals
    
    def analyze_bollinger_bands(self, bb_result: IndicatorResult,
                              price: pd.Series,
                              band_threshold: float = 0.05) -> List[Signal]:
        """Generate signals from Bollinger Bands indicator."""
        signals = []
        bb_data = bb_result.values
        
        for timestamp, row in bb_data.iterrows():
            if timestamp not in price.index or pd.isna(price[timestamp]):
                continue
                
            current_price = price[timestamp]
            band_width = (row['upper'] - row['lower']) / row['middle']
            
            # Price near or outside bands
            if current_price >= row['upper']:
                strength = min((current_price - row['upper']) / (band_width * row['middle']), 1.0)
                signals.append(Signal(
                    type=SignalType.SELL,
                    timestamp=timestamp,
                    symbol=None,
                    price=current_price,
                    indicator="Bollinger Bands",
                    strength=strength,
                    metadata={
                        'upper': row['upper'],
                        'middle': row['middle'],
                        'lower': row['lower'],
                        'band_width': band_width
                    }
                ))
            elif current_price <= row['lower']:
                strength = min((row['lower'] - current_price) / (band_width * row['middle']), 1.0)
                signals.append(Signal(
                    type=SignalType.BUY,
                    timestamp=timestamp,
                    symbol=None,
                    price=current_price,
                    indicator="Bollinger Bands",
                    strength=strength,
                    metadata={
                        'upper': row['upper'],
                        'middle': row['middle'],
                        'lower': row['lower'],
                        'band_width': band_width
                    }
                ))
                
        return signals
    
    def analyze_supertrend(self, supertrend_result: IndicatorResult,
                          price: pd.Series) -> List[Signal]:
        """Generate signals from SuperTrend indicator."""
        signals = []
        st_data = supertrend_result.values
        
        prev_direction = None
        for timestamp, row in st_data.iterrows():
            if pd.isna(row['direction']) or timestamp not in price.index:
                continue
                
            current_direction = row['direction']
            if prev_direction is not None and current_direction != prev_direction:
                current_price = price[timestamp]
                if current_direction == 1:  # Bullish
                    signals.append(Signal(
                        type=SignalType.BUY,
                        timestamp=timestamp,
                        symbol=None,
                        price=current_price,
                        indicator="SuperTrend",
                        strength=1.0,
                        metadata={
                            'supertrend': row['supertrend'],
                            'direction': current_direction
                        }
                    ))
                else:  # Bearish
                    signals.append(Signal(
                        type=SignalType.SELL,
                        timestamp=timestamp,
                        symbol=None,
                        price=current_price,
                        indicator="SuperTrend",
                        strength=1.0,
                        metadata={
                            'supertrend': row['supertrend'],
                            'direction': current_direction
                        }
                    ))
                    
            prev_direction = current_direction
            
        return signals
    
    def analyze_ichimoku(self, ichimoku_result: IndicatorResult,
                        price: pd.Series) -> List[Signal]:
        """Generate signals from Ichimoku Cloud indicator."""
        signals = []
        cloud_data = ichimoku_result.values
        
        for timestamp, row in cloud_data.iterrows():
            if timestamp not in price.index or pd.isna(price[timestamp]):
                continue
                
            current_price = price[timestamp]
            
            # TK Cross
            if row['tenkan_sen'] > row['kijun_sen']:
                # Bullish TK Cross
                strength = min((row['tenkan_sen'] - row['kijun_sen']) / current_price, 1.0)
                signals.append(Signal(
                    type=SignalType.BUY,
                    timestamp=timestamp,
                    symbol=None,
                    price=current_price,
                    indicator="Ichimoku",
                    strength=strength,
                    metadata={
                        'cross_type': 'TK_BULL',
                        'tenkan': row['tenkan_sen'],
                        'kijun': row['kijun_sen']
                    }
                ))
            elif row['tenkan_sen'] < row['kijun_sen']:
                # Bearish TK Cross
                strength = min((row['kijun_sen'] - row['tenkan_sen']) / current_price, 1.0)
                signals.append(Signal(
                    type=SignalType.SELL,
                    timestamp=timestamp,
                    symbol=None,
                    price=current_price,
                    indicator="Ichimoku",
                    strength=strength,
                    metadata={
                        'cross_type': 'TK_BEAR',
                        'tenkan': row['tenkan_sen'],
                        'kijun': row['kijun_sen']
                    }
                ))
                
            # Cloud Breakout
            if current_price > max(row['senkou_span_a'], row['senkou_span_b']):
                # Bullish breakout
                strength = min(
                    (current_price - max(row['senkou_span_a'], row['senkou_span_b'])) / current_price,
                    1.0
                )
                signals.append(Signal(
                    type=SignalType.BUY,
                    timestamp=timestamp,
                    symbol=None,
                    price=current_price,
                    indicator="Ichimoku",
                    strength=strength,
                    metadata={
                        'breakout_type': 'BULL_CLOUD',
                        'span_a': row['senkou_span_a'],
                        'span_b': row['senkou_span_b']
                    }
                ))
            elif current_price < min(row['senkou_span_a'], row['senkou_span_b']):
                # Bearish breakout
                strength = min(
                    (min(row['senkou_span_a'], row['senkou_span_b']) - current_price) / current_price,
                    1.0
                )
                signals.append(Signal(
                    type=SignalType.SELL,
                    timestamp=timestamp,
                    symbol=None,
                    price=current_price,
                    indicator="Ichimoku",
                    strength=strength,
                    metadata={
                        'breakout_type': 'BEAR_CLOUD',
                        'span_a': row['senkou_span_a'],
                        'span_b': row['senkou_span_b']
                    }
                ))
                
        return signals
    
    def analyze_elder_ray(self, elder_ray_result: IndicatorResult,
                         threshold: float = 0) -> List[Signal]:
        """Generate signals from Elder Ray indicator."""
        signals = []
        er_data = elder_ray_result.values
        
        for timestamp, row in er_data.iterrows():
            if pd.isna(row['bull_power']) or pd.isna(row['bear_power']):
                continue
                
            if row['bull_power'] > threshold and row['bear_power'] > threshold:
                # Strong bullish signal
                strength = min(
                    (row['bull_power'] + row['bear_power']) / (2 * threshold),
                    1.0
                ) if threshold > 0 else 1.0
                signals.append(Signal(
                    type=SignalType.BUY,
                    timestamp=timestamp,
                    symbol=None,
                    price=None,
                    indicator="Elder Ray",
                    strength=strength,
                    metadata={
                        'bull_power': row['bull_power'],
                        'bear_power': row['bear_power']
                    }
                ))
            elif row['bull_power'] < -threshold and row['bear_power'] < -threshold:
                # Strong bearish signal
                strength = min(
                    (-row['bull_power'] - row['bear_power']) / (2 * threshold),
                    1.0
                ) if threshold > 0 else 1.0
                signals.append(Signal(
                    type=SignalType.SELL,
                    timestamp=timestamp,
                    symbol=None,
                    price=None,
                    indicator="Elder Ray",
                    strength=strength,
                    metadata={
                        'bull_power': row['bull_power'],
                        'bear_power': row['bear_power']
                    }
                ))
                
        return signals
    
    def combine_signals(self, signals: List[Signal],
                       weights: Dict[str, float] = None) -> List[Signal]:
        """Combine multiple signals with optional weighting."""
        if not signals:
            return []
            
        # Default equal weights if not provided
        if weights is None:
            unique_indicators = set(s.indicator for s in signals)
            weights = {indicator: 1.0 / len(unique_indicators) for indicator in unique_indicators}
            
        # Group signals by timestamp
        signals_by_time = {}
        for signal in signals:
            if signal.timestamp not in signals_by_time:
                signals_by_time[signal.timestamp] = []
            signals_by_time[signal.timestamp].append(signal)
            
        combined_signals = []
        for timestamp, time_signals in signals_by_time.items():
            # Calculate weighted buy and sell signals
            buy_strength = sum(
                s.strength * weights.get(s.indicator, 1.0)
                for s in time_signals if s.type == SignalType.BUY
            )
            sell_strength = sum(
                s.strength * weights.get(s.indicator, 1.0)
                for s in time_signals if s.type == SignalType.SELL
            )
            
            # Generate combined signal if significant
            if buy_strength > sell_strength and buy_strength > 0.5:
                combined_signals.append(Signal(
                    type=SignalType.BUY,
                    timestamp=timestamp,
                    symbol=time_signals[0].symbol,
                    price=time_signals[0].price,
                    indicator="Combined",
                    strength=buy_strength,
                    metadata={
                        'components': [
                            {
                                'indicator': s.indicator,
                                'type': s.type.value,
                                'strength': s.strength
                            }
                            for s in time_signals
                        ]
                    }
                ))
            elif sell_strength > buy_strength and sell_strength > 0.5:
                combined_signals.append(Signal(
                    type=SignalType.SELL,
                    timestamp=timestamp,
                    symbol=time_signals[0].symbol,
                    price=time_signals[0].price,
                    indicator="Combined",
                    strength=sell_strength,
                    metadata={
                        'components': [
                            {
                                'indicator': s.indicator,
                                'type': s.type.value,
                                'strength': s.strength
                            }
                            for s in time_signals
                        ]
                    }
                ))
                
        return combined_signals
