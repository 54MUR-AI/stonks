# Analytics package
# Import functions from parent analytics.py module for backward compatibility
import sys
import os

# Get the absolute path to backend directory
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
analytics_file = os.path.join(backend_dir, 'analytics.py')

# Import directly from the analytics.py file to avoid circular import
import importlib.util
spec = importlib.util.spec_from_file_location("backend_analytics", analytics_file)
backend_analytics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_analytics)

# Expose the functions
calculate_portfolio_metrics = backend_analytics.calculate_portfolio_metrics
calculate_correlation_matrix = backend_analytics.calculate_correlation_matrix

__all__ = ['calculate_portfolio_metrics', 'calculate_correlation_matrix']
