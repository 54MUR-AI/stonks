import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import yfinance as yf
from sqlalchemy.orm import Session
from ..models import Portfolio, Position, Trade, Order
from ..database import SessionLocal
from .risk_prediction import risk_predictor
from .portfolio_rebalancing import portfolio_rebalancer
from .advanced_risk_metrics import risk_analyzer
from .anomaly_detection import anomaly_detector

@dataclass
class TradingParameters:
    max_position_size: float = 0.20    # 20% max position
    min_position_size: float = 0.02    # 2% min position
    max_daily_volume: float = 0.10     # 10% of daily volume
    min_liquidity: float = 1000000     # $1M min daily volume
    max_spread: float = 0.02           # 2% max bid-ask spread
    risk_limit: float = 0.02           # 2% max portfolio VaR
    rebalance_threshold: float = 0.05  # 5% weight deviation
    stop_loss: float = 0.10            # 10% stop loss
    take_profit: float = 0.20          # 20% take profit

class AutomatedTrader:
    def __init__(self):
        self.params = TradingParameters()
        self.trading_active = False
        self.trading_tasks = {}
        
    async def start_trading(self, portfolio_id: int):
        """Start automated trading for a portfolio"""
        if portfolio_id in self.trading_tasks:
            return
            
        self.trading_active = True
        task = asyncio.create_task(self._trade_portfolio(portfolio_id))
        self.trading_tasks[portfolio_id] = task
        
    async def stop_trading(self, portfolio_id: int):
        """Stop automated trading for a portfolio"""
        if portfolio_id in self.trading_tasks:
            self.trading_tasks[portfolio_id].cancel()
            del self.trading_tasks[portfolio_id]
            
    def _get_db(self) -> Session:
        """Get database session"""
        return SessionLocal()
        
    async def _trade_portfolio(self, portfolio_id: int):
        """Main trading loop for a portfolio"""
        while self.trading_active:
            try:
                db = self._get_db()
                # Execute trading strategy
                await asyncio.gather(
                    self._check_rebalancing(db, portfolio_id),
                    self._check_risk_limits(db, portfolio_id),
                    self._check_stop_losses(db, portfolio_id),
                    self._check_opportunities(db, portfolio_id)
                )
                db.close()
                
                # Wait before next check
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                print(f"Error trading portfolio {portfolio_id}: {str(e)}")
                await asyncio.sleep(60)
                
    async def _check_rebalancing(self, db: Session, portfolio_id: int):
        """Check and execute rebalancing trades"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            # Prepare portfolio data
            portfolio_data = {}
            for position in positions:
                portfolio_data[position.symbol] = {
                    "quantity": position.quantity,
                    "current_price": position.current_price,
                    "average_price": position.average_price
                }
                
            # Get recommendations
            recommendations = portfolio_rebalancer.generate_rebalancing_recommendations(
                portfolio_data,
                portfolio.cash,
                "sharpe"
            )
            
            # Execute trades for significant deviations
            for rec in recommendations:
                if abs(rec.target_weight - rec.current_weight) > self.params.rebalance_threshold:
                    # Check trading conditions
                    if not self._check_trading_conditions(rec.symbol):
                        continue
                        
                    # Create and execute order
                    order = Order(
                        portfolio_id=portfolio_id,
                        symbol=rec.symbol,
                        order_type="MARKET",
                        side="BUY" if rec.quantity_change > 0 else "SELL",
                        quantity=abs(rec.quantity_change),
                        status="PENDING",
                        timestamp=datetime.now()
                    )
                    db.add(order)
                    db.commit()
                    
        except Exception as e:
            print(f"Error in rebalancing: {str(e)}")
            
    async def _check_risk_limits(self, db: Session, portfolio_id: int):
        """Check and manage risk limits"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            # Calculate risk metrics
            portfolio_data = {
                p.symbol: {
                    "quantity": p.quantity,
                    "current_price": p.current_price,
                    "average_price": p.average_price
                }
                for p in positions
            }
            
            metrics = risk_analyzer.calculate_advanced_metrics(portfolio_data)
            
            # Check VaR limit
            if abs(metrics.var_parametric) > self.params.risk_limit:
                # Reduce highest risk contributors
                for symbol, contrib in sorted(
                    metrics.risk_contribution.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]:
                    position = next(p for p in positions if p.symbol == symbol)
                    reduce_quantity = int(position.quantity * 0.25)  # Reduce by 25%
                    
                    order = Order(
                        portfolio_id=portfolio_id,
                        symbol=symbol,
                        order_type="MARKET",
                        side="SELL",
                        quantity=reduce_quantity,
                        status="PENDING",
                        timestamp=datetime.now()
                    )
                    db.add(order)
                    
            db.commit()
            
        except Exception as e:
            print(f"Error in risk limit check: {str(e)}")
            
    async def _check_stop_losses(self, db: Session, portfolio_id: int):
        """Check and execute stop losses"""
        try:
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            for position in positions:
                returns = (position.current_price - position.average_price) / position.average_price
                
                # Check stop loss
                if returns < -self.params.stop_loss:
                    order = Order(
                        portfolio_id=portfolio_id,
                        symbol=position.symbol,
                        order_type="MARKET",
                        side="SELL",
                        quantity=position.quantity,
                        status="PENDING",
                        timestamp=datetime.now()
                    )
                    db.add(order)
                    
                # Check take profit
                elif returns > self.params.take_profit:
                    order = Order(
                        portfolio_id=portfolio_id,
                        symbol=position.symbol,
                        order_type="MARKET",
                        side="SELL",
                        quantity=position.quantity // 2,  # Sell half
                        status="PENDING",
                        timestamp=datetime.now()
                    )
                    db.add(order)
                    
            db.commit()
            
        except Exception as e:
            print(f"Error in stop loss check: {str(e)}")
            
    async def _check_opportunities(self, db: Session, portfolio_id: int):
        """Check for trading opportunities"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            # Get current positions
            current_symbols = {p.symbol for p in positions}
            
            # Check anomalies for current positions
            for symbol in current_symbols:
                anomalies = anomaly_detector.detect_anomalies(symbol)
                
                # If recent anomaly detected
                if anomalies['high_probability']:
                    latest_anomaly = max(anomalies['high_probability'])
                    if (datetime.now() - latest_anomaly).days <= 1:
                        # Analyze anomaly
                        analysis = anomaly_detector.analyze_anomaly(symbol, latest_anomaly)
                        
                        # If negative anomaly, reduce position
                        if analysis['price_change'] < -0.05:  # 5% drop
                            position = next(p for p in positions if p.symbol == symbol)
                            reduce_quantity = int(position.quantity * 0.5)  # Reduce by 50%
                            
                            order = Order(
                                portfolio_id=portfolio_id,
                                symbol=symbol,
                                order_type="MARKET",
                                side="SELL",
                                quantity=reduce_quantity,
                                status="PENDING",
                                timestamp=datetime.now()
                            )
                            db.add(order)
                            
            db.commit()
            
        except Exception as e:
            print(f"Error in opportunity check: {str(e)}")
            
    def _check_trading_conditions(self, symbol: str) -> bool:
        """Check if trading conditions are met"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d")
            
            # Check volume
            avg_volume = data['Volume'].mean()
            avg_price = data['Close'].mean()
            dollar_volume = avg_volume * avg_price
            
            if dollar_volume < self.params.min_liquidity:
                return False
                
            # Check spread (if available)
            if hasattr(ticker, 'info') and 'bid' in ticker.info and 'ask' in ticker.info:
                spread = (ticker.info['ask'] - ticker.info['bid']) / ticker.info['ask']
                if spread > self.params.max_spread:
                    return False
                    
            return True
            
        except Exception:
            return False

automated_trader = AutomatedTrader()
