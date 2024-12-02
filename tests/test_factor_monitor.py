"""
Tests for factor monitoring service
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backend.services.factor_monitor import FactorMonitor

class TestFactorMonitor(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Generate sample dates
        self.dates = pd.date_range(
            start=datetime.now() - timedelta(days=500),
            end=datetime.now(),
            freq='B'
        )
        
        # Generate sample factor returns with consistent indexing
        n_factors = 4
        np.random.seed(42)  # For reproducibility
        self.factor_returns = pd.DataFrame(
            np.random.normal(0.0001, 0.01, (len(self.dates), n_factors)),
            index=self.dates,
            columns=[f'Factor_{i}' for i in range(n_factors)]  # Start from 0
        )
        
        # Initialize monitor with stricter thresholds for testing
        self.monitor = FactorMonitor(
            lookback_window=63,
            volatility_threshold=1.5,  # Lower threshold to catch test anomalies
            correlation_threshold=0.2,  # Lower threshold
            return_threshold=1.5  # Lower threshold
        )
    
    def test_return_monitoring(self):
        """Test return anomaly detection"""
        # Inject return anomaly
        self.factor_returns.iloc[-1, 0] = 0.1  # Large return
        
        alerts = self.monitor.monitor_factor_returns(self.factor_returns)
        
        # Check alerts
        self.assertGreater(len(alerts), 0)
        alert = alerts[0]
        self.assertEqual(alert.alert_type, 'RETURN_ANOMALY')
        self.assertEqual(alert.factor_name, 'Factor_0')
        self.assertIn('z_score', alert.metrics)
        self.assertIn('return', alert.metrics)
    
    def test_volatility_monitoring(self):
        """Test volatility spike detection"""
        # Inject more pronounced volatility spike
        self.factor_returns.iloc[-30:, 1] *= 5  # Increase volatility more significantly
        
        alerts = self.monitor.monitor_factor_volatility(self.factor_returns)
        
        # Check alerts
        self.assertGreater(len(alerts), 0)
        alert = alerts[0]
        self.assertEqual(alert.alert_type, 'VOLATILITY_SPIKE')
        self.assertEqual(alert.factor_name, 'Factor_1')
        self.assertIn('z_score', alert.metrics)
        self.assertIn('volatility', alert.metrics)
    
    def test_correlation_monitoring(self):
        """Test correlation change detection"""
        # Inject more pronounced correlation change
        self.factor_returns.iloc[-63:, 2] = self.factor_returns.iloc[-63:, 3] * 1.1  # Strong correlation
        
        alerts = self.monitor.monitor_factor_correlations(self.factor_returns)
        
        # Check alerts
        self.assertGreater(len(alerts), 0)
        alert = alerts[0]
        self.assertEqual(alert.alert_type, 'CORRELATION_CHANGE')
        self.assertIn('Factor_2/Factor_3', alert.factor_name)
        self.assertIn('correlation_change', alert.metrics)
        self.assertIn('current_correlation', alert.metrics)
        self.assertIn('previous_correlation', alert.metrics)
    
    def test_monitor_all(self):
        """Test comprehensive monitoring"""
        # Inject multiple anomalies
        self.factor_returns.iloc[-1, 0] = 0.1  # Return anomaly
        self.factor_returns.iloc[-30:, 1] *= 5  # Volatility spike
        self.factor_returns.iloc[-63:, 2] = self.factor_returns.iloc[-63:, 3] * 1.1  # Correlation change
        
        alerts = self.monitor.monitor_all(self.factor_returns)
        
        # Check alerts
        self.assertGreater(len(alerts), 0)
        alert_types = [alert.alert_type for alert in alerts]
        self.assertIn('RETURN_ANOMALY', alert_types)
        self.assertIn('VOLATILITY_SPIKE', alert_types)
        self.assertIn('CORRELATION_CHANGE', alert_types)
        
        # Check alert sorting (HIGH severity should come first)
        high_severity_first = all(
            alerts[i].severity == 'HIGH' or
            all(alerts[j].severity != 'HIGH' for j in range(i+1, len(alerts)))
            for i in range(len(alerts))
        )
        self.assertTrue(high_severity_first)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        # Create small dataset
        small_returns = self.factor_returns.iloc[-30:]
        alerts = self.monitor.monitor_all(small_returns)
        
        # Should return empty list
        self.assertEqual(len(alerts), 0)
    
    def test_alert_summary(self):
        """Test alert summary generation"""
        # Generate some alerts
        self.factor_returns.iloc[-1, 0] = 0.1
        alerts = self.monitor.monitor_all(self.factor_returns)
        self.monitor.alerts.extend(alerts)
        
        # Get summary
        summary = self.monitor.get_alert_summary()
        
        # Check summary
        self.assertGreater(len(summary), 0)
        self.assertIn('timestamp', summary.columns)
        self.assertIn('factor', summary.columns)
        self.assertIn('type', summary.columns)
        self.assertIn('message', summary.columns)
        self.assertIn('severity', summary.columns)

if __name__ == '__main__':
    unittest.main()
