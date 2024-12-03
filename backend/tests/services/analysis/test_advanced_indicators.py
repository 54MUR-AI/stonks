"""Tests for advanced technical indicators."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.services.analysis.advanced_indicators import AdvancedIndicators

class TestAdvancedIndicators(unittest.TestCase):
    """Test suite for advanced technical indicators."""
    
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
        
        self.indicators = AdvancedIndicators()
        
    def test_ichimoku(self):
        """Test Ichimoku Cloud indicator calculation."""
        result = self.indicators.ichimoku(
            self.price_data['high'],
            self.price_data['low'],
            self.price_data['close']
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Ichimoku Cloud")
        self.assertTrue('tenkan_sen' in result.values.columns)
        self.assertTrue('kijun_sen' in result.values.columns)
        self.assertTrue('senkou_span_a' in result.values.columns)
        self.assertTrue('senkou_span_b' in result.values.columns)
        self.assertTrue('chikou_span' in result.values.columns)
        
    def test_fibonacci_retracements(self):
        """Test Fibonacci retracement levels calculation."""
        result = self.indicators.fibonacci_retracements(
            self.price_data['high'],
            self.price_data['low']
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Fibonacci Retracements")
        self.assertTrue('0.0' in result.values.columns)
        self.assertTrue('0.236' in result.values.columns)
        self.assertTrue('0.382' in result.values.columns)
        self.assertTrue('0.618' in result.values.columns)
        self.assertTrue('1.0' in result.values.columns)
        
    def test_vwap(self):
        """Test VWAP calculation."""
        result = self.indicators.vwap(
            self.price_data['high'],
            self.price_data['low'],
            self.price_data['close'],
            self.price_data['volume']
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "VWAP")
        self.assertFalse(result.values.isna().all())
        
    def test_pivot_points(self):
        """Test pivot points calculation."""
        result = self.indicators.pivot_points(
            self.price_data['high'],
            self.price_data['low'],
            self.price_data['close']
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Pivot Points")
        self.assertTrue('pivot' in result.values.columns)
        self.assertTrue('r1' in result.values.columns)
        self.assertTrue('r2' in result.values.columns)
        self.assertTrue('r3' in result.values.columns)
        self.assertTrue('s1' in result.values.columns)
        self.assertTrue('s2' in result.values.columns)
        self.assertTrue('s3' in result.values.columns)
        
    def test_elder_ray(self):
        """Test Elder Ray indicator calculation."""
        result = self.indicators.elder_ray(
            self.price_data['high'],
            self.price_data['low'],
            self.price_data['close']
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Elder Ray")
        self.assertTrue('ema' in result.values.columns)
        self.assertTrue('bull_power' in result.values.columns)
        self.assertTrue('bear_power' in result.values.columns)
        
    def test_supertrend(self):
        """Test SuperTrend indicator calculation."""
        result = self.indicators.supertrend(
            self.price_data['high'],
            self.price_data['low'],
            self.price_data['close']
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "SuperTrend")
        self.assertTrue('supertrend' in result.values.columns)
        self.assertTrue('direction' in result.values.columns)
        self.assertTrue('upperband' in result.values.columns)
        self.assertTrue('lowerband' in result.values.columns)

if __name__ == '__main__':
    unittest.main()
