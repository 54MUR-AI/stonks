import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backend.services.data_generator import MarketDataGenerator
from backend.services.factor_analysis import PortfolioFactorAnalyzer
from backend.services.visualization import VisualizationService

def generate_sample_data(n_assets=10, n_days=252):
    """Generate sample market data"""
    generator = MarketDataGenerator()
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate returns for multiple assets
    assets = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Generate correlated returns
    base_returns = np.random.normal(0.0001, 0.01, (len(dates), 1))
    asset_returns = np.zeros((len(dates), n_assets))
    
    for i in range(n_assets):
        # Mix of common factor and idiosyncratic returns
        asset_returns[:, i] = (0.7 * base_returns.flatten() + 
                             0.3 * np.random.normal(0.0001, 0.02, len(dates)))
    
    return pd.DataFrame(asset_returns, index=dates, columns=assets)

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    returns = generate_sample_data()
    
    # Generate equal weights for the portfolio
    weights = pd.Series(1/len(returns.columns), index=returns.columns)
    
    # Initialize services
    analyzer = PortfolioFactorAnalyzer(n_factors=4)
    viz = VisualizationService()
    
    # Perform factor analysis
    factor_loadings, components = analyzer.extract_statistical_factors(returns)
    factor_returns = analyzer.decompose_returns(returns, weights)
    factor_summary = analyzer.get_factor_summary(returns, weights)
    
    # Generate visualizations
    viz.plot_factor_analysis(
        factor_summary,
        save_path=os.path.join(output_dir, 'factor_analysis_dashboard.png')
    )
    
    viz.plot_factor_contribution_over_time(
        factor_returns,
        save_path=os.path.join(output_dir, 'factor_contribution.png')
    )
    
    viz.plot_rolling_factor_risk(
        factor_returns,
        window=63,
        save_path=os.path.join(output_dir, 'rolling_factor_risk.png')
    )
    
    print("Generated factor analysis visualizations in 'output' directory")

if __name__ == '__main__':
    main()
