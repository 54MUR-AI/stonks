import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backend.services.portfolio_optimizer import BlackLittermanOptimizer

class TestBlackLittermanOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.optimizer = BlackLittermanOptimizer()
        
        # Generate synthetic price data
        dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
        np.random.seed(42)
        
        self.test_data = {}
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        for ticker in tickers:
            # Generate random walk prices
            returns = np.random.normal(0.0001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            volume = np.random.randint(1000000, 5000000, len(dates))
            
            self.test_data[ticker] = pd.DataFrame({
                'close': prices,
                'volume': volume
            }, index=dates)
    
    def test_market_implied_returns(self):
        """Test calculation of market implied returns"""
        # Get returns and covariance
        returns = pd.DataFrame({
            symbol: df['close'].pct_change().dropna()
            for symbol, df in self.test_data.items()
        })
        cov_matrix = returns.cov().values
        
        # Test market weights
        market_weights = np.array([0.3, 0.3, 0.2, 0.2])
        implied_returns = self.optimizer.calculate_market_implied_returns(
            market_weights, cov_matrix
        )
        
        self.assertEqual(len(implied_returns), len(market_weights))
        self.assertTrue(np.all(np.isfinite(implied_returns)))
    
    def test_portfolio_optimization(self):
        """Test full portfolio optimization"""
        # Define views
        views = [
            {
                'assets': ['AAPL', 'MSFT'],
                'weights': [1, -1],
                'return': 0.05,  # AAPL will outperform MSFT by 5%
                'confidence': 0.6
            },
            {
                'assets': ['GOOGL'],
                'weights': [1],
                'return': 0.15,  # GOOGL will return 15%
                'confidence': 0.4
            }
        ]
        
        # Run optimization
        weights = self.optimizer.optimize_portfolio(
            self.test_data,
            views=views
        )
        
        # Check results
        self.assertEqual(len(weights), len(self.test_data))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        self.assertTrue(all(0 <= w <= self.optimizer.max_position_size 
                          for w in weights.values()))
    
    def test_optimization_no_views(self):
        """Test optimization without views"""
        weights = self.optimizer.optimize_portfolio(self.test_data)
        
        self.assertEqual(len(weights), len(self.test_data))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        self.assertTrue(all(0 <= w <= self.optimizer.max_position_size 
                          for w in weights.values()))

if __name__ == '__main__':
    unittest.main()
