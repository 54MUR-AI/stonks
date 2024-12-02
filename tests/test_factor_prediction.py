"""
Tests for factor prediction service
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backend.services.factor_prediction import FactorPredictor

class TestFactorPredictor(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Generate sample dates
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=500),
            end=datetime.now(),
            freq='B'
        )
        
        # Generate sample factor returns
        n_factors = 4
        self.factor_returns = pd.DataFrame(
            np.random.normal(0.0001, 0.01, (len(dates), n_factors)),
            index=dates,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        
        # Add some autocorrelation to make prediction meaningful
        for i in range(1, len(dates)):
            self.factor_returns.iloc[i] = (
                0.7 * self.factor_returns.iloc[i-1] +
                0.3 * self.factor_returns.iloc[i]
            )
        
        # Initialize predictor
        self.predictor = FactorPredictor(model_type='rf', n_estimators=100)
    
    def test_prepare_features(self):
        """Test feature preparation"""
        lookback = 21
        X, y = self.predictor.prepare_features(self.factor_returns, lookback)
        
        # Check shapes
        expected_rows = len(self.factor_returns) - lookback
        expected_feature_cols = lookback * len(self.factor_returns.columns) + 3 * len(self.factor_returns.columns)
        
        self.assertEqual(X.shape[0], expected_rows)
        self.assertEqual(X.shape[1], expected_feature_cols)
        self.assertEqual(y.shape[0], expected_rows)
        self.assertEqual(y.shape[1], len(self.factor_returns.columns))
    
    def test_model_fitting(self):
        """Test model fitting and cross-validation"""
        metrics = self.predictor.fit(self.factor_returns)
        
        # Check metrics
        self.assertIn('r2', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        
        # Check metric values are reasonable
        self.assertGreater(metrics['r2'], -1)  # R² should be > -1
        self.assertLess(metrics['r2'], 1.1)    # R² should be < 1.1
        self.assertGreater(metrics['mse'], 0)   # MSE should be positive
        self.assertGreater(metrics['mae'], 0)   # MAE should be positive
    
    def test_prediction(self):
        """Test factor return prediction"""
        # Fit model
        self.predictor.fit(self.factor_returns)
        
        # Generate predictions
        horizon = 5
        lookback = 21
        
        # Ensure we have enough data
        self.assertGreater(len(self.factor_returns), lookback + 1,
                          "Not enough test data for prediction")
        
        predictions = self.predictor.predict(
            self.factor_returns,
            lookback=lookback,
            horizon=horizon
        )
        
        # Check predictions shape and values
        self.assertEqual(len(predictions), horizon)
        self.assertEqual(len(predictions.columns), len(self.factor_returns.columns))
        
        # Check predictions are within reasonable bounds
        self.assertTrue(np.all(np.abs(predictions) < 0.1))  # Returns should be small
        
        # Check prediction dates are business days
        self.assertTrue(all(d.weekday() < 5 for d in predictions.index))
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        # Fit model
        self.predictor.fit(self.factor_returns)
        
        # Get feature importance
        importance = self.predictor.get_feature_importance()
        
        # Check importance DataFrame
        self.assertGreater(len(importance), 0)
        self.assertIn('Feature', importance.columns)
        self.assertIn('Importance', importance.columns)
        
        # Check importance values sum to approximately 1
        self.assertAlmostEqual(importance['Importance'].sum(), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
