import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class MarketDataGenerator:
    """Generate synthetic market data for backtesting"""
    
    def __init__(self,
                 n_assets: int = 5,
                 n_days: int = 252,  # One trading year
                 seed: Optional[int] = None):
        """
        Initialize market data generator
        
        Args:
            n_assets: Number of assets to simulate
            n_days: Number of trading days
            seed: Random seed for reproducibility
        """
        self.n_assets = n_assets
        self.n_days = n_days
        
        if seed is not None:
            np.random.seed(seed)
            
    def generate_correlated_returns(self,
                                  mean_returns: np.ndarray,
                                  correlation_matrix: np.ndarray,
                                  volatilities: np.ndarray) -> np.ndarray:
        """
        Generate correlated asset returns
        
        Args:
            mean_returns: Array of mean returns for each asset
            correlation_matrix: Asset correlation matrix
            volatilities: Array of asset volatilities
            
        Returns:
            Array of correlated returns
        """
        # Convert correlation to covariance matrix
        vol_matrix = np.diag(volatilities)
        cov_matrix = vol_matrix @ correlation_matrix @ vol_matrix
        
        # Generate correlated random returns
        returns = np.random.multivariate_normal(
            mean_returns,
            cov_matrix,
            size=self.n_days
        )
        
        return returns
        
    def generate_market_regime_data(self,
                                  regimes: List[Dict],
                                  tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate market data with different market regimes
        
        Args:
            regimes: List of regime dictionaries with parameters:
                    [{'duration': 126,  # trading days
                      'mean_returns': [0.0002, ...],
                      'volatilities': [0.01, ...],
                      'correlations': [[1, 0.3, ...], [0.3, 1, ...], ...],
                      'regime_name': 'bull_market'}, ...]
            tickers: List of asset tickers (default: ASSET_1, ASSET_2, ...)
            
        Returns:
            Dictionary of DataFrames with price and volume data
        """
        if tickers is None:
            tickers = [f"ASSET_{i+1}" for i in range(self.n_assets)]
            
        if len(tickers) != self.n_assets:
            raise ValueError("Number of tickers must match n_assets")
            
        # Initialize arrays
        all_returns = np.zeros((self.n_days, self.n_assets))
        all_volumes = np.zeros((self.n_days, self.n_assets))
        regime_labels = np.array([''] * self.n_days, dtype=object)
        
        # Generate data for each regime
        day_index = 0
        remaining_days = self.n_days
        
        for regime in regimes:
            # Calculate duration ensuring we don't exceed total days
            duration = min(regime['duration'], remaining_days)
            if duration <= 0:
                break
                
            # Generate returns for this regime
            regime_returns = self.generate_correlated_returns(
                np.array(regime['mean_returns']),
                np.array(regime['correlations']),
                np.array(regime['volatilities'])
            )[:duration]
            
            # Generate volumes with regime-specific characteristics
            base_volume = np.random.normal(1e6, 2e5, (duration, self.n_assets))
            volume_trend = np.linspace(0.8, 1.2, duration)[:, np.newaxis]
            regime_volumes = base_volume * volume_trend
            
            # Add volatility clustering effect
            vol_cluster = np.abs(regime_returns) > np.std(regime_returns, axis=0)
            regime_volumes[vol_cluster] *= 1.5
            
            # Store data
            all_returns[day_index:day_index+duration] = regime_returns
            all_volumes[day_index:day_index+duration] = regime_volumes
            regime_labels[day_index:day_index+duration] = regime['regime_name']
            
            day_index += duration
            remaining_days -= duration
            
        # If we have remaining days, extend the last regime
        if remaining_days > 0 and regimes:
            last_regime = regimes[-1]
            extra_returns = self.generate_correlated_returns(
                np.array(last_regime['mean_returns']),
                np.array(last_regime['correlations']),
                np.array(last_regime['volatilities'])
            )[:remaining_days]
            
            base_volume = np.random.normal(1e6, 2e5, (remaining_days, self.n_assets))
            volume_trend = np.linspace(0.8, 1.2, remaining_days)[:, np.newaxis]
            extra_volumes = base_volume * volume_trend
            
            vol_cluster = np.abs(extra_returns) > np.std(extra_returns, axis=0)
            extra_volumes[vol_cluster] *= 1.5
            
            all_returns[day_index:] = extra_returns
            all_volumes[day_index:] = extra_volumes
            regime_labels[day_index:] = last_regime['regime_name']
        
        # Convert returns to prices
        prices = 100 * np.exp(np.cumsum(all_returns, axis=0))
        
        # Create date index
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.n_days * 365//252)
        dates = pd.date_range(start_date, end_date, periods=self.n_days)
        
        # Create output dictionary
        market_data = {}
        for i, ticker in enumerate(tickers):
            market_data[ticker] = pd.DataFrame({
                'close': prices[:, i],
                'volume': all_volumes[:, i],
                'returns': all_returns[:, i],
                'regime': regime_labels
            }, index=dates)
            
        return market_data
        
    def generate_realistic_market_data(self,
                                     tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate realistic market data with bull, bear, and sideways regimes
        
        Args:
            tickers: List of asset tickers
            
        Returns:
            Dictionary of DataFrames with price and volume data
        """
        # Define realistic market regimes
        regimes = [
            {
                'duration': self.n_days // 3,
                'mean_returns': [0.0008] * self.n_assets,  # Bull market
                'volatilities': [0.015] * self.n_assets,
                'correlations': np.array([[1 if i == j else 0.3 
                                         for j in range(self.n_assets)]
                                        for i in range(self.n_assets)]),
                'regime_name': 'bull_market'
            },
            {
                'duration': self.n_days // 3,
                'mean_returns': [-0.0005] * self.n_assets,  # Bear market
                'volatilities': [0.025] * self.n_assets,
                'correlations': np.array([[1 if i == j else 0.7  # Higher correlations in bear market
                                         for j in range(self.n_assets)]
                                        for i in range(self.n_assets)]),
                'regime_name': 'bear_market'
            },
            {
                'duration': self.n_days // 3,
                'mean_returns': [0.0001] * self.n_assets,  # Sideways market
                'volatilities': [0.01] * self.n_assets,
                'correlations': np.array([[1 if i == j else 0.2
                                         for j in range(self.n_assets)]
                                        for i in range(self.n_assets)]),
                'regime_name': 'sideways_market'
            }
        ]
        
        return self.generate_market_regime_data(regimes, tickers)
