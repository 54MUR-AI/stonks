"""Tests for signal generation system."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.services.analysis.signals import SignalGenerator, Signal, SignalType
from backend.services.analysis.indicators import IndicatorResult

class TestSignalGenerator(unittest.TestCase):
    """Test suite for signal generation system."""
    
    def setUp(self):
        """Set up test data."""
        # Generate sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)
        
        # Generate synthetic price data with a trend and some volatility
        trend = np.linspace(100, 200, n)
        volatility = np.random.normal(0, 5, n)
        close = trend + volatility
        high = close + np.random.uniform(0, 2, n)
        low = close - np.random.uniform(0, 2, n)
        volume = np.random.uniform(1000, 5000, n)
        
        self.price_data = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        self.signal_generator = SignalGenerator()
        
    def test_rsi_signals(self):
        """Test RSI signal generation."""
        # Create sample RSI data
        rsi_values = pd.Series(
            np.random.uniform(20, 80, len(self.price_data)),
            index=self.price_data.index
        )
        rsi_result = IndicatorResult(
            name="RSI",
            values=rsi_values,
            metadata={'period': 14}
        )
        
        signals = self.signal_generator.analyze_rsi(rsi_result)
        
        self.assertIsInstance(signals, list)
        for signal in signals:
            self.assertIsInstance(signal, Signal)
            self.assertIn(signal.type, [SignalType.BUY, SignalType.SELL])
            self.assertEqual(signal.indicator, "RSI")
            self.assertTrue(0 <= signal.strength <= 1)
            
    def test_macd_signals(self):
        """Test MACD signal generation."""
        # Create sample MACD data
        macd_data = pd.DataFrame({
            'macd': np.random.normal(0, 1, len(self.price_data)),
            'signal': np.random.normal(0, 1, len(self.price_data)),
            'histogram': np.random.normal(0, 0.5, len(self.price_data))
        }, index=self.price_data.index)
        
        macd_result = IndicatorResult(
            name="MACD",
            values=macd_data,
            metadata={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        )
        
        signals = self.signal_generator.analyze_macd(macd_result)
        
        self.assertIsInstance(signals, list)
        for signal in signals:
            self.assertIsInstance(signal, Signal)
            self.assertIn(signal.type, [SignalType.BUY, SignalType.SELL])
            self.assertEqual(signal.indicator, "MACD")
            self.assertTrue(0 <= signal.strength <= 1)
            
    def test_bollinger_bands_signals(self):
        """Test Bollinger Bands signal generation."""
        # Create sample Bollinger Bands data
        bb_data = pd.DataFrame({
            'upper': self.price_data['close'] + 2 * np.random.uniform(1, 3, len(self.price_data)),
            'middle': self.price_data['close'],
            'lower': self.price_data['close'] - 2 * np.random.uniform(1, 3, len(self.price_data))
        }, index=self.price_data.index)
        
        bb_result = IndicatorResult(
            name="Bollinger Bands",
            values=bb_data,
            metadata={'period': 20, 'std_dev': 2}
        )
        
        signals = self.signal_generator.analyze_bollinger_bands(
            bb_result,
            self.price_data['close']
        )
        
        self.assertIsInstance(signals, list)
        for signal in signals:
            self.assertIsInstance(signal, Signal)
            self.assertIn(signal.type, [SignalType.BUY, SignalType.SELL])
            self.assertEqual(signal.indicator, "Bollinger Bands")
            self.assertTrue(0 <= signal.strength <= 1)
            
    def test_supertrend_signals(self):
        """Test SuperTrend signal generation."""
        # Create sample SuperTrend data
        st_data = pd.DataFrame({
            'supertrend': self.price_data['close'],
            'direction': np.random.choice([1, -1], size=len(self.price_data)),
            'upperband': self.price_data['close'] + np.random.uniform(1, 3, len(self.price_data)),
            'lowerband': self.price_data['close'] - np.random.uniform(1, 3, len(self.price_data))
        }, index=self.price_data.index)
        
        st_result = IndicatorResult(
            name="SuperTrend",
            values=st_data,
            metadata={'period': 10, 'multiplier': 3}
        )
        
        signals = self.signal_generator.analyze_supertrend(
            st_result,
            self.price_data['close']
        )
        
        self.assertIsInstance(signals, list)
        for signal in signals:
            self.assertIsInstance(signal, Signal)
            self.assertIn(signal.type, [SignalType.BUY, SignalType.SELL])
            self.assertEqual(signal.indicator, "SuperTrend")
            self.assertTrue(0 <= signal.strength <= 1)
            
    def test_ichimoku_signals(self):
        """Test Ichimoku Cloud signal generation."""
        # Create sample Ichimoku data
        cloud_data = pd.DataFrame({
            'tenkan_sen': self.price_data['close'] + np.random.normal(0, 1, len(self.price_data)),
            'kijun_sen': self.price_data['close'] + np.random.normal(0, 1, len(self.price_data)),
            'senkou_span_a': self.price_data['close'] + np.random.normal(0, 2, len(self.price_data)),
            'senkou_span_b': self.price_data['close'] + np.random.normal(0, 2, len(self.price_data)),
            'chikou_span': self.price_data['close'].shift(-26)
        }, index=self.price_data.index)
        
        ichimoku_result = IndicatorResult(
            name="Ichimoku Cloud",
            values=cloud_data,
            metadata={
                'tenkan_period': 9,
                'kijun_period': 26,
                'senkou_b_period': 52,
                'displacement': 26
            }
        )
        
        signals = self.signal_generator.analyze_ichimoku(
            ichimoku_result,
            self.price_data['close']
        )
        
        self.assertIsInstance(signals, list)
        for signal in signals:
            self.assertIsInstance(signal, Signal)
            self.assertIn(signal.type, [SignalType.BUY, SignalType.SELL])
            self.assertEqual(signal.indicator, "Ichimoku")
            self.assertTrue(0 <= signal.strength <= 1)
            
    def test_elder_ray_signals(self):
        """Test Elder Ray signal generation."""
        # Create sample Elder Ray data
        er_data = pd.DataFrame({
            'ema': self.price_data['close'].ewm(span=13).mean(),
            'bull_power': np.random.normal(0, 1, len(self.price_data)),
            'bear_power': np.random.normal(0, 1, len(self.price_data))
        }, index=self.price_data.index)
        
        er_result = IndicatorResult(
            name="Elder Ray",
            values=er_data,
            metadata={'ema_period': 13}
        )
        
        signals = self.signal_generator.analyze_elder_ray(er_result)
        
        self.assertIsInstance(signals, list)
        for signal in signals:
            self.assertIsInstance(signal, Signal)
            self.assertIn(signal.type, [SignalType.BUY, SignalType.SELL])
            self.assertEqual(signal.indicator, "Elder Ray")
            self.assertTrue(0 <= signal.strength <= 1)
            
    def test_combine_signals(self):
        """Test signal combination with weights."""
        # Create sample signals
        signals = [
            Signal(
                type=SignalType.BUY,
                timestamp=self.price_data.index[0],
                symbol="AAPL",
                price=100.0,
                indicator="RSI",
                strength=0.8
            ),
            Signal(
                type=SignalType.SELL,
                timestamp=self.price_data.index[0],
                symbol="AAPL",
                price=100.0,
                indicator="MACD",
                strength=0.6
            ),
            Signal(
                type=SignalType.BUY,
                timestamp=self.price_data.index[0],
                symbol="AAPL",
                price=100.0,
                indicator="Bollinger Bands",
                strength=0.7
            )
        ]
        
        weights = {
            "RSI": 0.4,
            "MACD": 0.3,
            "Bollinger Bands": 0.3
        }
        
        combined = self.signal_generator.combine_signals(signals, weights)
        
        self.assertIsInstance(combined, list)
        for signal in combined:
            self.assertIsInstance(signal, Signal)
            self.assertEqual(signal.indicator, "Combined")
            self.assertTrue(0 <= signal.strength <= 1)
            self.assertTrue('components' in signal.metadata)

if __name__ == '__main__':
    unittest.main()
