"""Feature engineering pipeline for machine learning models."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..analysis.indicators import TechnicalIndicators
from ..analysis.advanced_indicators import AdvancedIndicators

@dataclass
class FeatureSet:
    """Container for engineered features."""
    features: pd.DataFrame
    target: pd.Series
    metadata: Dict
    
class FeatureEngineer:
    """Feature engineering pipeline for financial data."""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.advanced_indicators = AdvancedIndicators()
        
    def create_technical_features(
        self,
        df: pd.DataFrame,
        feature_sets: List[str] = None
    ) -> pd.DataFrame:
        """Create technical analysis features.
        
        Args:
            df: DataFrame with OHLCV data
            feature_sets: List of feature sets to generate. If None, generates all
        
        Returns:
            DataFrame with technical features
        """
        if feature_sets is None:
            feature_sets = ['momentum', 'trend', 'volatility', 'volume']
            
        features = pd.DataFrame(index=df.index)
        
        for feature_set in feature_sets:
            if feature_set == 'momentum':
                # RSI features
                rsi = self.indicators.rsi(df['close']).values
                features['rsi'] = rsi
                features['rsi_diff'] = rsi.diff()
                features['rsi_ma'] = rsi.rolling(window=14).mean()
                
                # Stochastic features
                stoch = self.indicators.stochastic(df['high'], df['low'], df['close']).values
                features['stoch_k'] = stoch['k']
                features['stoch_d'] = stoch['d']
                features['stoch_diff'] = stoch['k'] - stoch['d']
                
            elif feature_set == 'trend':
                # MACD features
                macd = self.indicators.macd(df['close']).values
                features['macd'] = macd['macd']
                features['macd_signal'] = macd['signal']
                features['macd_hist'] = macd['histogram']
                
                # SuperTrend features
                supertrend = self.advanced_indicators.supertrend(
                    df['high'], df['low'], df['close']
                ).values
                features['supertrend_diff'] = df['close'] - supertrend['supertrend']
                features['supertrend_direction'] = supertrend['direction']
                
            elif feature_set == 'volatility':
                # Bollinger Bands features
                bb = self.indicators.bollinger_bands(df['close']).values
                features['bb_width'] = (bb['upper'] - bb['lower']) / bb['middle']
                features['bb_position'] = (df['close'] - bb['lower']) / (bb['upper'] - bb['lower'])
                
                # ATR features
                atr = self.indicators.atr(df['high'], df['low'], df['close']).values
                features['atr'] = atr
                features['atr_pct'] = atr / df['close']
                
            elif feature_set == 'volume':
                # OBV features
                obv = self.indicators.obv(df['close'], df['volume']).values
                features['obv'] = obv
                features['obv_ma'] = obv.rolling(window=20).mean()
                features['obv_diff'] = obv.diff()
                
                # VWAP features
                vwap = self.advanced_indicators.vwap(
                    df['high'], df['low'], df['close'], df['volume']
                ).values
                features['vwap_diff'] = df['close'] - vwap
                features['vwap_ratio'] = df['close'] / vwap
                
        return features
    
    def create_price_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = None
    ) -> pd.DataFrame:
        """Create price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of lookback windows for calculations
            
        Returns:
            DataFrame with price features
        """
        if windows is None:
            windows = [5, 10, 20, 50, 100]
            
        features = pd.DataFrame(index=df.index)
        
        # Price changes
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log1p(features['returns'])
        
        # Rolling statistics
        for window in windows:
            # Returns
            features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
            features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
            features[f'returns_skew_{window}'] = features['returns'].rolling(window).skew()
            
            # Prices
            features[f'close_ma_{window}'] = df['close'].rolling(window).mean()
            features[f'close_std_{window}'] = df['close'].rolling(window).std()
            features[f'close_max_{window}'] = df['close'].rolling(window).max()
            features[f'close_min_{window}'] = df['close'].rolling(window).min()
            
            # Volume
            features[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            features[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            
            # Price ranges
            high_low_range = df['high'] - df['low']
            features[f'range_ma_{window}'] = high_low_range.rolling(window).mean()
            features[f'range_std_{window}'] = high_low_range.rolling(window).std()
            
        return features
    
    def create_target(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        threshold: float = 0.0
    ) -> pd.Series:
        """Create target variable for ML models.
        
        Args:
            df: DataFrame with OHLCV data
            horizon: Forecast horizon in periods
            threshold: Return threshold for classification
            
        Returns:
            Series with target values
        """
        # Calculate future returns
        future_returns = df['close'].shift(-horizon).pct_change(horizon)
        
        if threshold == 0.0:
            # Regression target
            return future_returns
        else:
            # Classification target
            return pd.Series(
                np.where(future_returns > threshold, 1, 0),
                index=df.index
            )
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_sets: List[str] = None,
        windows: List[int] = None,
        target_horizon: int = 5,
        target_threshold: float = 0.0
    ) -> FeatureSet:
        """Prepare complete feature set for ML models.
        
        Args:
            df: DataFrame with OHLCV data
            feature_sets: List of technical feature sets to generate
            windows: List of lookback windows for price features
            target_horizon: Forecast horizon for target variable
            target_threshold: Return threshold for classification target
            
        Returns:
            FeatureSet containing features, target, and metadata
        """
        # Generate features
        tech_features = self.create_technical_features(df, feature_sets)
        price_features = self.create_price_features(df, windows)
        
        # Combine features
        features = pd.concat([tech_features, price_features], axis=1)
        
        # Generate target
        target = self.create_target(df, target_horizon, target_threshold)
        
        # Create metadata
        metadata = {
            'feature_sets': feature_sets,
            'windows': windows,
            'target_horizon': target_horizon,
            'target_threshold': target_threshold,
            'feature_names': list(features.columns),
            'num_features': len(features.columns),
            'num_samples': len(features),
            'start_date': df.index[0],
            'end_date': df.index[-1]
        }
        
        return FeatureSet(
            features=features,
            target=target,
            metadata=metadata
        )
