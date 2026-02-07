# Analytics package
# Import functions from parent analytics.py module for backward compatibility
import sys
from pathlib import Path

# Add parent directory to path to import from analytics.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from analytics import calculate_portfolio_metrics, calculate_correlation_matrix

__all__ = ['calculate_portfolio_metrics', 'calculate_correlation_matrix']
