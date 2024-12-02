"""
Real-time factor monitoring service for tracking factor behavior and generating alerts
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FactorAlert:
    """Data class for factor alerts"""
    timestamp: datetime
    factor_name: str
    alert_type: str
    message: str
    severity: str
    metrics: Dict[str, float]

class FactorMonitor:
    def __init__(self,
                 lookback_window: int = 63,  # ~3 months
                 volatility_threshold: float = 2.0,
                 correlation_threshold: float = 0.3,
                 return_threshold: float = 2.0):
        """
        Initialize factor monitor
        
        Args:
            lookback_window: Window size for rolling statistics
            volatility_threshold: Z-score threshold for volatility alerts
            correlation_threshold: Change threshold for correlation alerts
            return_threshold: Z-score threshold for return alerts
        """
        self.lookback_window = lookback_window
        self.volatility_threshold = volatility_threshold
        self.correlation_threshold = correlation_threshold
        self.return_threshold = return_threshold
        self.alerts = []
        
    def monitor_factor_returns(self,
                             factor_returns: pd.DataFrame) -> List[FactorAlert]:
        """
        Monitor factor returns for anomalies
        
        Args:
            factor_returns: DataFrame of factor returns
            
        Returns:
            List of factor alerts
        """
        alerts = []
        
        # Calculate rolling statistics
        rolling_mean = factor_returns.rolling(self.lookback_window).mean()
        rolling_std = factor_returns.rolling(self.lookback_window).std()
        
        # Get latest returns
        latest_returns = factor_returns.iloc[-1]
        
        # Check for return anomalies
        z_scores = (latest_returns - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        
        for factor in factor_returns.columns:
            z_score = z_scores[factor]
            
            if abs(z_score) > self.return_threshold:
                alerts.append(FactorAlert(
                    timestamp=factor_returns.index[-1],
                    factor_name=factor,
                    alert_type='RETURN_ANOMALY',
                    message=f'Abnormal return detected: {z_score:.2f} standard deviations',
                    severity='HIGH' if abs(z_score) > 2 * self.return_threshold else 'MEDIUM',
                    metrics={'z_score': z_score, 'return': latest_returns[factor]}
                ))
        
        return alerts
    
    def monitor_factor_volatility(self,
                                factor_returns: pd.DataFrame) -> List[FactorAlert]:
        """
        Monitor factor volatility for anomalies
        
        Args:
            factor_returns: DataFrame of factor returns
            
        Returns:
            List of factor alerts
        """
        alerts = []
        
        # Calculate rolling volatility
        rolling_vol = factor_returns.rolling(self.lookback_window).std() * np.sqrt(252)
        vol_mean = rolling_vol.rolling(self.lookback_window).mean()
        vol_std = rolling_vol.rolling(self.lookback_window).std()
        
        # Get latest volatility
        latest_vol = rolling_vol.iloc[-1]
        
        # Check for volatility anomalies
        z_scores = (latest_vol - vol_mean.iloc[-1]) / vol_std.iloc[-1]
        
        for factor in factor_returns.columns:
            z_score = z_scores[factor]
            
            if abs(z_score) > self.volatility_threshold:
                alerts.append(FactorAlert(
                    timestamp=factor_returns.index[-1],
                    factor_name=factor,
                    alert_type='VOLATILITY_SPIKE',
                    message=f'Abnormal volatility detected: {z_score:.2f} standard deviations',
                    severity='HIGH' if abs(z_score) > 2 * self.volatility_threshold else 'MEDIUM',
                    metrics={'z_score': z_score, 'volatility': latest_vol[factor]}
                ))
        
        return alerts
    
    def monitor_factor_correlations(self,
                                  factor_returns: pd.DataFrame) -> List[FactorAlert]:
        """
        Monitor changes in factor correlations
        
        Args:
            factor_returns: DataFrame of factor returns
            
        Returns:
            List of factor alerts
        """
        alerts = []
        
        # Calculate rolling correlations
        recent_corr = factor_returns.iloc[-self.lookback_window:].corr()
        prev_corr = factor_returns.iloc[-2*self.lookback_window:-self.lookback_window].corr()
        
        # Check for correlation changes
        corr_change = recent_corr - prev_corr
        
        for i in range(len(factor_returns.columns)):
            for j in range(i+1, len(factor_returns.columns)):
                factor1 = factor_returns.columns[i]
                factor2 = factor_returns.columns[j]
                change = corr_change.iloc[i, j]
                
                if abs(change) > self.correlation_threshold:
                    alerts.append(FactorAlert(
                        timestamp=factor_returns.index[-1],
                        factor_name=f'{factor1}/{factor2}',
                        alert_type='CORRELATION_CHANGE',
                        message=f'Significant correlation change: {change:.2f}',
                        severity='HIGH' if abs(change) > 2 * self.correlation_threshold else 'MEDIUM',
                        metrics={
                            'correlation_change': change,
                            'current_correlation': recent_corr.iloc[i, j],
                            'previous_correlation': prev_corr.iloc[i, j]
                        }
                    ))
        
        return alerts
    
    def monitor_all(self, factor_returns: pd.DataFrame) -> List[FactorAlert]:
        """
        Run all monitoring checks
        
        Args:
            factor_returns: DataFrame of factor returns
            
        Returns:
            List of all factor alerts
        """
        alerts = []
        
        # Check data requirements
        if len(factor_returns) < 2 * self.lookback_window:
            logger.warning(f"Insufficient data for monitoring. Need at least {2 * self.lookback_window} periods.")
            return alerts
        
        # Run all checks
        alerts.extend(self.monitor_factor_returns(factor_returns))
        alerts.extend(self.monitor_factor_volatility(factor_returns))
        alerts.extend(self.monitor_factor_correlations(factor_returns))
        
        # Sort alerts by severity and timestamp
        alerts.sort(key=lambda x: (x.severity == 'HIGH', x.timestamp), reverse=True)
        
        return alerts
    
    def get_alert_summary(self) -> pd.DataFrame:
        """
        Get summary of recent alerts
        
        Returns:
            DataFrame summarizing recent alerts
        """
        if not self.alerts:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'timestamp': alert.timestamp,
                'factor': alert.factor_name,
                'type': alert.alert_type,
                'message': alert.message,
                'severity': alert.severity
            }
            for alert in self.alerts
        ])
