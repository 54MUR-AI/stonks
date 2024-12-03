"""Tests for feature engineering pipeline."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.services.ml.feature_engineering import FeatureEngineer, FeatureSet

class TestFeatureEngineering(unittest.TestCase):
    """Test suite for feature engineering pipeline."""
    
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
        
        self.engineer = FeatureEngineer()
        
    def test_technical_features(self):
        """Test technical feature generation."""
        features = self.engineer.create_technical_features(
            self.price_data,
            feature_sets=['momentum', 'trend', 'volatility', 'volume']
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.price_data))
        
        # Check momentum features
        self.assertTrue('rsi' in features.columns)
        self.assertTrue('stoch_k' in features.columns)
        self.assertTrue('stoch_d' in features.columns)
        
        # Check trend features
        self.assertTrue('macd' in features.columns)
        self.assertTrue('macd_signal' in features.columns)
        self.assertTrue('supertrend_direction' in features.columns)
        
        # Check volatility features
        self.assertTrue('bb_width' in features.columns)
        self.assertTrue('atr' in features.columns)
        
        # Check volume features
        self.assertTrue('obv' in features.columns)
        self.assertTrue('vwap_diff' in features.columns)
        
    def test_price_features(self):
        """Test price feature generation."""
        windows = [5, 10, 20]
        features = self.engineer.create_price_features(
            self.price_data,
            windows=windows
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.price_data))
        
        # Check basic features
        self.assertTrue('returns' in features.columns)
        self.assertTrue('log_returns' in features.columns)
        
        # Check window-based features
        for window in windows:
            self.assertTrue(f'returns_mean_{window}' in features.columns)
            self.assertTrue(f'close_ma_{window}' in features.columns)
            self.assertTrue(f'volume_ma_{window}' in features.columns)
            self.assertTrue(f'range_ma_{window}' in features.columns)
            
    def test_target_creation(self):
        """Test target variable creation."""
        # Test regression target
        reg_target = self.engineer.create_target(
            self.price_data,
            horizon=5,
            threshold=0.0
        )
        
        self.assertIsInstance(reg_target, pd.Series)
        self.assertEqual(len(reg_target), len(self.price_data))
        
        # Test classification target
        cls_target = self.engineer.create_target(
            self.price_data,
            horizon=5,
            threshold=0.01
        )
        
        self.assertIsInstance(cls_target, pd.Series)
        self.assertEqual(len(cls_target), len(self.price_data))
        self.assertTrue(set(cls_target.unique()).issubset({0, 1}))
        
    def test_complete_feature_preparation(self):
        """Test complete feature preparation pipeline."""
        feature_set = self.engineer.prepare_features(
            self.price_data,
            feature_sets=['momentum', 'trend'],
            windows=[5, 10],
            target_horizon=5,
            target_threshold=0.01
        )
        
        self.assertIsInstance(feature_set, FeatureSet)
        self.assertIsInstance(feature_set.features, pd.DataFrame)
        self.assertIsInstance(feature_set.target, pd.Series)
        self.assertIsInstance(feature_set.metadata, dict)
        
        # Check metadata
        self.assertEqual(feature_set.metadata['target_horizon'], 5)
        self.assertEqual(feature_set.metadata['target_threshold'], 0.01)
        self.assertEqual(len(feature_set.metadata['feature_names']),
                        len(feature_set.features.columns))
        
        # Check data alignment
        self.assertEqual(len(feature_set.features), len(feature_set.target))
        self.assertEqual(feature_set.features.index.equals(feature_set.target.index), True)
        
    def test_feature_engineering_with_missing_data(self):
        """Test feature engineering with missing data."""
        # Create data with some missing values
        data_with_nan = self.price_data.copy()
        data_with_nan.loc[data_with_nan.index[10:15], 'close'] = np.nan
        
        feature_set = self.engineer.prepare_features(data_with_nan)
        
        self.assertIsInstance(feature_set, FeatureSet)
        # Features should still be created, but will contain some NaN values
        self.assertTrue(feature_set.features.isna().any().any())
        
    def test_feature_engineering_with_small_dataset(self):
        """Test feature engineering with a small dataset."""
        small_data = self.price_data.iloc[:30]
        
        feature_set = self.engineer.prepare_features(small_data)
        
        self.assertIsInstance(feature_set, FeatureSet)
        self.assertEqual(len(feature_set.features), len(small_data))
        
    def test_feature_engineering_with_different_frequencies(self):
        """Test feature engineering with different data frequencies."""
        # Create weekly data
        weekly_data = self.price_data.resample('W').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        feature_set = self.engineer.prepare_features(weekly_data)
        
        self.assertIsInstance(feature_set, FeatureSet)
        self.assertEqual(len(feature_set.features), len(weekly_data))

if __name__ == '__main__':
    unittest.main()
