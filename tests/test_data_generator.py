import unittest
import numpy as np
import pandas as pd
from backend.services.data_generator import MarketDataGenerator

class TestMarketDataGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test data generator"""
        self.n_assets = 4
        self.n_days = 252
        self.generator = MarketDataGenerator(
            n_assets=self.n_assets,
            n_days=self.n_days,
            seed=42
        )
        
    def test_correlated_returns(self):
        """Test generation of correlated returns"""
        mean_returns = np.array([0.001] * self.n_assets)
        volatilities = np.array([0.02] * self.n_assets)
        correlation_matrix = np.array([[1.0 if i == j else 0.5 
                                      for j in range(self.n_assets)]
                                     for i in range(self.n_assets)])
        
        returns = self.generator.generate_correlated_returns(
            mean_returns, correlation_matrix, volatilities
        )
        
        # Check dimensions
        self.assertEqual(returns.shape, (self.n_days, self.n_assets))
        
        # Check sample statistics
        sample_means = np.mean(returns, axis=0)
        sample_vols = np.std(returns, axis=0)
        sample_corr = np.corrcoef(returns.T)
        
        # Verify statistics are within reasonable bounds
        np.testing.assert_array_almost_equal(sample_means, mean_returns, decimal=2)
        np.testing.assert_array_almost_equal(sample_vols, volatilities, decimal=2)
        np.testing.assert_array_almost_equal(sample_corr, correlation_matrix, decimal=1)
        
    def test_market_regime_data(self):
        """Test generation of market regime data"""
        regimes = [
            {
                'duration': 126,
                'mean_returns': [0.001] * self.n_assets,
                'volatilities': [0.02] * self.n_assets,
                'correlations': np.array([[1.0 if i == j else 0.3 
                                         for j in range(self.n_assets)]
                                        for i in range(self.n_assets)]),
                'regime_name': 'bull_market'
            },
            {
                'duration': 126,
                'mean_returns': [-0.001] * self.n_assets,
                'volatilities': [0.03] * self.n_assets,
                'correlations': np.array([[1.0 if i == j else 0.7
                                         for j in range(self.n_assets)]
                                        for i in range(self.n_assets)]),
                'regime_name': 'bear_market'
            }
        ]
        
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        market_data = self.generator.generate_market_regime_data(regimes, tickers)
        
        # Check output structure
        self.assertEqual(len(market_data), self.n_assets)
        self.assertTrue(all(ticker in market_data for ticker in tickers))
        
        # Check DataFrame contents
        for ticker in tickers:
            df = market_data[ticker]
            self.assertTrue(all(col in df.columns 
                              for col in ['close', 'volume', 'returns', 'regime']))
            self.assertEqual(len(df), self.n_days)
            
            # Check regime labels
            regimes_present = df['regime'].unique()
            self.assertTrue(all(regime['regime_name'] in regimes_present 
                              for regime in regimes))
            
            # Check basic data properties
            self.assertTrue(np.all(df['volume'] > 0))
            self.assertTrue(np.all(df['close'] > 0))
            self.assertTrue(np.all(np.isfinite(df['returns'])))
            
    def test_realistic_market_data(self):
        """Test generation of realistic market data"""
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        market_data = self.generator.generate_realistic_market_data(tickers)
        
        # Check output structure
        self.assertEqual(len(market_data), self.n_assets)
        self.assertTrue(all(ticker in market_data for ticker in tickers))
        
        # Check regime distribution
        df = market_data[tickers[0]]
        regime_counts = df['regime'].value_counts()
        expected_counts = self.n_days // 3
        
        self.assertTrue(all(abs(count - expected_counts) <= 1 
                          for count in regime_counts))
        
        # Check regime characteristics
        bull_returns = df.loc[df['regime'] == 'bull_market', 'returns']
        bear_returns = df.loc[df['regime'] == 'bear_market', 'returns']
        sideways_returns = df.loc[df['regime'] == 'sideways_market', 'returns']
        
        self.assertTrue(np.mean(bull_returns) > np.mean(sideways_returns))
        self.assertTrue(np.mean(bear_returns) < np.mean(sideways_returns))
        self.assertTrue(np.std(bear_returns) > np.std(bull_returns))

if __name__ == '__main__':
    unittest.main()
