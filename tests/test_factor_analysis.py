import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backend.services.factor_analysis import PortfolioFactorAnalyzer

class TestPortfolioFactorAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.analyzer = PortfolioFactorAnalyzer(n_factors=3)
        
        # Generate synthetic returns data
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
        n_assets = 5
        
        # Create correlated returns
        cov_matrix = np.array([[1.0, 0.6, 0.4, 0.2, 0.1],
                              [0.6, 1.0, 0.5, 0.3, 0.2],
                              [0.4, 0.5, 1.0, 0.4, 0.3],
                              [0.2, 0.3, 0.4, 1.0, 0.5],
                              [0.1, 0.2, 0.3, 0.5, 1.0]])
        
        mean_returns = np.array([0.0001, 0.0002, 0.0001, 0.0003, 0.0002])
        returns = np.random.multivariate_normal(mean_returns, cov_matrix * 0.0001, size=len(dates))
        
        self.returns = pd.DataFrame(
            returns,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        )
        
        self.weights = pd.Series({
            'AAPL': 0.25,
            'MSFT': 0.25,
            'GOOGL': 0.2,
            'AMZN': 0.15,
            'META': 0.15
        })
        
    def test_extract_statistical_factors(self):
        """Test statistical factor extraction"""
        factor_loadings, components = self.analyzer.extract_statistical_factors(self.returns)
        
        # Check dimensions
        self.assertEqual(factor_loadings.shape[1], self.analyzer.n_factors)
        self.assertEqual(components.shape[0], self.analyzer.n_factors)
        self.assertEqual(components.shape[1], len(self.returns.columns))
        
        # Check that loadings are standardized
        self.assertTrue(np.allclose(factor_loadings.mean(), 0, atol=1e-10))
        self.assertTrue(np.allclose(factor_loadings.std(), 1, atol=0.1))
        
    def test_analyze_factor_contribution(self):
        """Test factor contribution analysis"""
        results = self.analyzer.analyze_factor_contribution(self.returns, self.weights)
        
        # Check results structure
        self.assertTrue(all(key in results for key in [
            'factor_exposures', 'risk_contribution', 
            'cumulative_variance', 'components'
        ]))
        
        # Check dimensions
        self.assertEqual(len(results['factor_exposures']), self.analyzer.n_factors)
        self.assertEqual(len(results['risk_contribution']), self.analyzer.n_factors)
        
        # Check that contributions sum to 1
        self.assertAlmostEqual(results['risk_contribution'].sum(), 1.0, places=4)
        
        # Check cumulative variance is increasing
        self.assertTrue(np.all(np.diff(results['cumulative_variance']) >= 0))
        
    def test_decompose_returns(self):
        """Test return decomposition"""
        factor_returns = self.analyzer.decompose_returns(self.returns, self.weights)
        
        # Check dimensions
        self.assertEqual(len(factor_returns.columns), self.analyzer.n_factors + 1)  # +1 for residual
        self.assertEqual(len(factor_returns), len(self.returns))
        
        # Check that factor returns sum to portfolio returns (with residual)
        portfolio_returns = self.returns @ self.weights
        decomposed_sum = factor_returns.sum(axis=1)
        np.testing.assert_array_almost_equal(portfolio_returns, decomposed_sum)
        
    def test_get_factor_correlations(self):
        """Test factor-asset correlations"""
        correlations = self.analyzer.get_factor_correlations(self.returns)
        
        # Check dimensions
        self.assertEqual(correlations.shape, (self.analyzer.n_factors, len(self.returns.columns)))
        
        # Check correlation bounds
        self.assertTrue(np.all(correlations.values >= -1))
        self.assertTrue(np.all(correlations.values <= 1))
        
    def test_get_factor_summary(self):
        """Test comprehensive factor analysis summary"""
        summary = self.analyzer.get_factor_summary(self.returns, self.weights)
        
        # Check summary structure
        self.assertTrue(all(key in summary for key in [
            'factor_metrics', 'factor_correlations',
            'factor_analysis', 'factor_returns'
        ]))
        
        # Check factor metrics
        metrics = summary['factor_metrics']
        self.assertTrue(all(metric in metrics.columns for metric in [
            'Annualized Return', 'Annualized Vol',
            'Sharpe Ratio', 'Contribution to Risk',
            'Max Drawdown'
        ]))
        
        # Check that risk contributions sum to 1
        self.assertAlmostEqual(
            summary['factor_metrics']['Contribution to Risk'].sum(),
            1.0,
            places=4
        )

if __name__ == '__main__':
    unittest.main()
