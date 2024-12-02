import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

class AlertType(Enum):
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    VAR_BREACH = "var_breach"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    REGIME_CHANGE = "regime_change"
    MARKET_STRESS = "market_stress"

@dataclass
class RiskAlert:
    type: AlertType
    timestamp: datetime
    portfolio_id: int
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metadata: Dict

class RiskAlertService:
    def __init__(self):
        self.alert_thresholds = {
            AlertType.VOLATILITY: {
                "high": 0.25,  # 25% annualized volatility
                "critical": 0.35
            },
            AlertType.DRAWDOWN: {
                "high": -0.15,  # 15% drawdown
                "critical": -0.25
            },
            AlertType.VAR_BREACH: {
                "high": 0.95,  # 95% VaR breach
                "critical": 0.99
            },
            AlertType.CORRELATION: {
                "high": 0.8,
                "critical": 0.9
            },
            AlertType.CONCENTRATION: {
                "high": 0.25,  # 25% in single position
                "critical": 0.35
            }
        }
        self.subscribers: Dict[int, Set[str]] = {}  # portfolio_id -> set of connection ids
        self.alert_history: Dict[int, List[RiskAlert]] = {}  # portfolio_id -> list of alerts
        
    async def subscribe(self, portfolio_id: int, connection_id: str):
        """Subscribe to risk alerts for a portfolio"""
        if portfolio_id not in self.subscribers:
            self.subscribers[portfolio_id] = set()
        self.subscribers[portfolio_id].add(connection_id)
        
    async def unsubscribe(self, portfolio_id: int, connection_id: str):
        """Unsubscribe from risk alerts"""
        if portfolio_id in self.subscribers:
            self.subscribers[portfolio_id].discard(connection_id)
            if not self.subscribers[portfolio_id]:
                del self.subscribers[portfolio_id]
                
    def monitor_volatility(
        self,
        portfolio_id: int,
        returns: pd.Series,
        lookback_window: int = 60
    ) -> Optional[RiskAlert]:
        """Monitor portfolio volatility"""
        if len(returns) < lookback_window:
            return None
            
        # Calculate rolling volatility
        vol = returns.rolling(window=lookback_window).std() * np.sqrt(252)
        current_vol = vol.iloc[-1]
        
        # Check thresholds
        if current_vol > self.alert_thresholds[AlertType.VOLATILITY]["critical"]:
            severity = "critical"
        elif current_vol > self.alert_thresholds[AlertType.VOLATILITY]["high"]:
            severity = "high"
        else:
            return None
            
        return RiskAlert(
            type=AlertType.VOLATILITY,
            timestamp=datetime.now(),
            portfolio_id=portfolio_id,
            severity=severity,
            message=f"Portfolio volatility ({current_vol:.1%}) exceeds {severity} threshold",
            metadata={
                "current_volatility": current_vol,
                "threshold": self.alert_thresholds[AlertType.VOLATILITY][severity],
                "lookback_window": lookback_window
            }
        )
        
    def monitor_drawdown(
        self,
        portfolio_id: int,
        returns: pd.Series
    ) -> Optional[RiskAlert]:
        """Monitor portfolio drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns / rolling_max - 1
        current_dd = drawdown.iloc[-1]
        
        if current_dd < self.alert_thresholds[AlertType.DRAWDOWN]["critical"]:
            severity = "critical"
        elif current_dd < self.alert_thresholds[AlertType.DRAWDOWN]["high"]:
            severity = "high"
        else:
            return None
            
        return RiskAlert(
            type=AlertType.DRAWDOWN,
            timestamp=datetime.now(),
            portfolio_id=portfolio_id,
            severity=severity,
            message=f"Portfolio drawdown ({current_dd:.1%}) exceeds {severity} threshold",
            metadata={
                "current_drawdown": current_dd,
                "threshold": self.alert_thresholds[AlertType.DRAWDOWN][severity],
                "max_value": rolling_max.iloc[-1]
            }
        )
        
    def monitor_var_breaches(
        self,
        portfolio_id: int,
        returns: pd.Series,
        var_level: float = 0.95
    ) -> Optional[RiskAlert]:
        """Monitor Value at Risk breaches"""
        # Calculate VaR
        var = np.percentile(returns, (1 - var_level) * 100)
        current_return = returns.iloc[-1]
        
        if current_return < var:
            severity = "critical" if var_level >= 0.99 else "high"
            
            return RiskAlert(
                type=AlertType.VAR_BREACH,
                timestamp=datetime.now(),
                portfolio_id=portfolio_id,
                severity=severity,
                message=f"Portfolio return ({current_return:.1%}) breached {var_level:.0%} VaR",
                metadata={
                    "current_return": current_return,
                    "var_level": var_level,
                    "var_value": var
                }
            )
            
        return None
        
    def monitor_correlations(
        self,
        portfolio_id: int,
        returns: pd.DataFrame,
        lookback_window: int = 60
    ) -> Optional[RiskAlert]:
        """Monitor asset correlations"""
        if len(returns) < lookback_window:
            return None
            
        # Calculate correlation matrix
        corr_matrix = returns.tail(lookback_window).corr()
        
        # Find highest correlation
        max_corr = 0
        max_pair = None
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    max_pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                    
        if abs(max_corr) > self.alert_thresholds[AlertType.CORRELATION]["critical"]:
            severity = "critical"
        elif abs(max_corr) > self.alert_thresholds[AlertType.CORRELATION]["high"]:
            severity = "high"
        else:
            return None
            
        return RiskAlert(
            type=AlertType.CORRELATION,
            timestamp=datetime.now(),
            portfolio_id=portfolio_id,
            severity=severity,
            message=f"High correlation ({max_corr:.2f}) between {max_pair[0]} and {max_pair[1]}",
            metadata={
                "correlation": max_corr,
                "assets": max_pair,
                "threshold": self.alert_thresholds[AlertType.CORRELATION][severity],
                "lookback_window": lookback_window
            }
        )
        
    def monitor_concentration(
        self,
        portfolio_id: int,
        positions: Dict[str, float]
    ) -> Optional[RiskAlert]:
        """Monitor portfolio concentration"""
        total_value = sum(positions.values())
        max_position = max(positions.values())
        max_symbol = max(positions.items(), key=lambda x: x[1])[0]
        concentration = max_position / total_value
        
        if concentration > self.alert_thresholds[AlertType.CONCENTRATION]["critical"]:
            severity = "critical"
        elif concentration > self.alert_thresholds[AlertType.CONCENTRATION]["high"]:
            severity = "high"
        else:
            return None
            
        return RiskAlert(
            type=AlertType.CONCENTRATION,
            timestamp=datetime.now(),
            portfolio_id=portfolio_id,
            severity=severity,
            message=f"High concentration ({concentration:.1%}) in {max_symbol}",
            metadata={
                "concentration": concentration,
                "symbol": max_symbol,
                "position_value": max_position,
                "threshold": self.alert_thresholds[AlertType.CONCENTRATION][severity]
            }
        )
        
    async def monitor_portfolio(
        self,
        portfolio_id: int,
        returns: pd.DataFrame,
        positions: Dict[str, float]
    ) -> List[RiskAlert]:
        """Monitor all risk metrics for a portfolio"""
        alerts = []
        portfolio_returns = returns.sum(axis=1)
        
        # Check various risk metrics
        volatility_alert = self.monitor_volatility(portfolio_id, portfolio_returns)
        if volatility_alert:
            alerts.append(volatility_alert)
            
        drawdown_alert = self.monitor_drawdown(portfolio_id, portfolio_returns)
        if drawdown_alert:
            alerts.append(drawdown_alert)
            
        var_alert = self.monitor_var_breaches(portfolio_id, portfolio_returns)
        if var_alert:
            alerts.append(var_alert)
            
        correlation_alert = self.monitor_correlations(portfolio_id, returns)
        if correlation_alert:
            alerts.append(correlation_alert)
            
        concentration_alert = self.monitor_concentration(portfolio_id, positions)
        if concentration_alert:
            alerts.append(concentration_alert)
            
        # Store alerts in history
        if portfolio_id not in self.alert_history:
            self.alert_history[portfolio_id] = []
        self.alert_history[portfolio_id].extend(alerts)
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(days=7)
        self.alert_history[portfolio_id] = [
            alert for alert in self.alert_history[portfolio_id]
            if alert.timestamp > cutoff_time
        ]
        
        return alerts
        
    async def broadcast_alerts(self):
        """Broadcast alerts to subscribers"""
        while True:
            for portfolio_id in list(self.subscribers.keys()):
                if not self.subscribers[portfolio_id]:
                    continue
                    
                # Get recent alerts
                recent_alerts = [
                    alert for alert in self.alert_history.get(portfolio_id, [])
                    if alert.timestamp > datetime.now() - timedelta(minutes=5)
                ]
                
                if recent_alerts:
                    message = {
                        "type": "risk_alerts",
                        "portfolio_id": portfolio_id,
                        "alerts": [
                            {
                                "type": alert.type.value,
                                "timestamp": alert.timestamp.isoformat(),
                                "severity": alert.severity,
                                "message": alert.message,
                                "metadata": alert.metadata
                            }
                            for alert in recent_alerts
                        ]
                    }
                    
                    # Broadcast to all subscribers
                    for connection_id in self.subscribers[portfolio_id]:
                        yield (connection_id, json.dumps(message))
                        
            await asyncio.sleep(1)  # Check frequency

risk_alert_service = RiskAlertService()
