import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import yfinance as yf
from sqlalchemy.orm import Session
from ..models import Portfolio, Position, Alert, RebalanceEvent
from ..database import SessionLocal
from .risk_prediction import risk_predictor
from .portfolio_rebalancing import portfolio_rebalancer

@dataclass
class MonitoringThresholds:
    volatility_threshold: float = 0.20  # 20% annualized volatility
    drawdown_threshold: float = 0.10    # 10% drawdown
    var_threshold: float = 0.05         # 5% VaR breach
    correlation_threshold: float = 0.80  # 80% correlation
    concentration_threshold: float = 0.35  # 35% max position size
    tracking_error_threshold: float = 0.05  # 5% tracking error
    rebalance_threshold: float = 0.05    # 5% weight deviation

class PortfolioMonitor:
    def __init__(self):
        self.thresholds = MonitoringThresholds()
        self.monitoring_active = False
        self.monitoring_tasks = {}
        
    async def start_monitoring(self, portfolio_id: int):
        """Start monitoring a portfolio"""
        if portfolio_id in self.monitoring_tasks:
            return
            
        self.monitoring_active = True
        task = asyncio.create_task(self._monitor_portfolio(portfolio_id))
        self.monitoring_tasks[portfolio_id] = task
        
    async def stop_monitoring(self, portfolio_id: int):
        """Stop monitoring a portfolio"""
        if portfolio_id in self.monitoring_tasks:
            self.monitoring_tasks[portfolio_id].cancel()
            del self.monitoring_tasks[portfolio_id]
            
    def _get_db(self) -> Session:
        """Get database session"""
        return SessionLocal()
        
    async def _monitor_portfolio(self, portfolio_id: int):
        """Main monitoring loop for a portfolio"""
        while self.monitoring_active:
            try:
                db = self._get_db()
                # Check various risk metrics and triggers
                await asyncio.gather(
                    self._check_volatility(db, portfolio_id),
                    self._check_drawdown(db, portfolio_id),
                    self._check_var(db, portfolio_id),
                    self._check_correlation(db, portfolio_id),
                    self._check_concentration(db, portfolio_id),
                    self._check_rebalancing_needs(db, portfolio_id)
                )
                db.close()
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                print(f"Error monitoring portfolio {portfolio_id}: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
                
    async def _check_volatility(self, db: Session, portfolio_id: int):
        """Check portfolio volatility against threshold"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            # Get predictions for each position
            high_vol_positions = []
            for position in positions:
                prediction = risk_predictor.predict_risk(position.symbol)
                if prediction['predicted_volatility'] > self.thresholds.volatility_threshold:
                    high_vol_positions.append(position.symbol)
                    
            if high_vol_positions:
                alert = Alert(
                    portfolio_id=portfolio_id,
                    alert_type="HIGH_VOLATILITY",
                    severity="WARNING",
                    message=f"High volatility detected in positions: {', '.join(high_vol_positions)}",
                    timestamp=datetime.now()
                )
                db.add(alert)
                db.commit()
                
        except Exception as e:
            print(f"Error checking volatility: {str(e)}")
            
    async def _check_drawdown(self, db: Session, portfolio_id: int):
        """Check portfolio drawdown against threshold"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            # Calculate drawdown for each position
            for position in positions:
                ticker = yf.Ticker(position.symbol)
                hist = ticker.history(period="1mo")
                peak = hist['Close'].max()
                current = hist['Close'].iloc[-1]
                drawdown = (peak - current) / peak
                
                if drawdown > self.thresholds.drawdown_threshold:
                    alert = Alert(
                        portfolio_id=portfolio_id,
                        alert_type="SIGNIFICANT_DRAWDOWN",
                        severity="WARNING",
                        message=f"Significant drawdown ({drawdown:.1%}) detected in {position.symbol}",
                        timestamp=datetime.now()
                    )
                    db.add(alert)
                    db.commit()
                    
        except Exception as e:
            print(f"Error checking drawdown: {str(e)}")
            
    async def _check_var(self, db: Session, portfolio_id: int):
        """Check Value at Risk breaches"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            # Calculate portfolio VaR
            returns_data = []
            weights = []
            total_value = sum(p.quantity * p.current_price for p in positions)
            
            for position in positions:
                ticker = yf.Ticker(position.symbol)
                hist = ticker.history(period="1y")
                returns = hist['Close'].pct_change().dropna()
                returns_data.append(returns)
                weights.append((position.quantity * position.current_price) / total_value)
                
            # Calculate portfolio returns
            portfolio_returns = sum(r * w for r, w in zip(returns_data, weights))
            var_95 = np.percentile(portfolio_returns, 5)
            
            if abs(var_95) > self.thresholds.var_threshold:
                alert = Alert(
                    portfolio_id=portfolio_id,
                    alert_type="VAR_BREACH",
                    severity="WARNING",
                    message=f"Portfolio VaR (95%) breach: {var_95:.1%}",
                    timestamp=datetime.now()
                )
                db.add(alert)
                db.commit()
                
        except Exception as e:
            print(f"Error checking VaR: {str(e)}")
            
    async def _check_correlation(self, db: Session, portfolio_id: int):
        """Check for high correlations between positions"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            # Get returns data
            returns_data = {}
            for position in positions:
                ticker = yf.Ticker(position.symbol)
                hist = ticker.history(period="1y")
                returns_data[position.symbol] = hist['Close'].pct_change().dropna()
                
            # Calculate correlation matrix
            returns_df = pd.DataFrame(returns_data)
            corr_matrix = returns_df.corr()
            
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > self.thresholds.correlation_threshold:
                        high_corr_pairs.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j])
                        )
                        
            if high_corr_pairs:
                alert = Alert(
                    portfolio_id=portfolio_id,
                    alert_type="HIGH_CORRELATION",
                    severity="INFO",
                    message=f"High correlation detected between: {', '.join([f'{p[0]}-{p[1]}' for p in high_corr_pairs])}",
                    timestamp=datetime.now()
                )
                db.add(alert)
                db.commit()
                
        except Exception as e:
            print(f"Error checking correlations: {str(e)}")
            
    async def _check_concentration(self, db: Session, portfolio_id: int):
        """Check for position concentration"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            total_value = sum(p.quantity * p.current_price for p in positions)
            
            # Check each position weight
            concentrated_positions = []
            for position in positions:
                weight = (position.quantity * position.current_price) / total_value
                if weight > self.thresholds.concentration_threshold:
                    concentrated_positions.append((position.symbol, weight))
                    
            if concentrated_positions:
                alert = Alert(
                    portfolio_id=portfolio_id,
                    alert_type="HIGH_CONCENTRATION",
                    severity="WARNING",
                    message=f"High concentration detected: {', '.join([f'{p[0]} ({p[1]:.1%})' for p in concentrated_positions])}",
                    timestamp=datetime.now()
                )
                db.add(alert)
                db.commit()
                
        except Exception as e:
            print(f"Error checking concentration: {str(e)}")
            
    async def _check_rebalancing_needs(self, db: Session, portfolio_id: int):
        """Check if portfolio needs rebalancing"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            # Prepare portfolio data for rebalancer
            portfolio_data = {}
            for position in positions:
                portfolio_data[position.symbol] = {
                    "quantity": position.quantity,
                    "current_price": position.current_price,
                    "average_price": position.average_price
                }
                
            # Get rebalancing recommendations
            recommendations = portfolio_rebalancer.generate_rebalancing_recommendations(
                portfolio_data,
                portfolio.cash,
                "sharpe"
            )
            
            # Check if any position needs significant rebalancing
            rebalance_needed = False
            rebalance_details = []
            
            for rec in recommendations:
                if abs(rec.target_weight - rec.current_weight) > self.thresholds.rebalance_threshold:
                    rebalance_needed = True
                    rebalance_details.append(
                        f"{rec.symbol}: {rec.current_weight:.1%} → {rec.target_weight:.1%}"
                    )
                    
            if rebalance_needed:
                # Create rebalance event
                event = RebalanceEvent(
                    portfolio_id=portfolio_id,
                    timestamp=datetime.now(),
                    status="PENDING",
                    details="\n".join(rebalance_details)
                )
                db.add(event)
                
                # Create alert
                alert = Alert(
                    portfolio_id=portfolio_id,
                    alert_type="REBALANCE_NEEDED",
                    severity="INFO",
                    message=f"Portfolio rebalancing recommended:\n{'; '.join(rebalance_details)}",
                    timestamp=datetime.now()
                )
                db.add(alert)
                db.commit()
                
        except Exception as e:
            print(f"Error checking rebalancing needs: {str(e)}")

portfolio_monitor = PortfolioMonitor()
