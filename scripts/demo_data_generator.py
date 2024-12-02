import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os
from random import uniform, choice

class DemoDataGenerator:
    def __init__(self):
        self.demo_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
            'META', 'TSLA', 'JPM', 'V', 'PYPL'
        ]
        self.start_date = datetime.now() - timedelta(days=365)
        self.end_date = datetime.now()
        
    def generate_portfolio_data(self):
        """Generate demo portfolio with simulated allocations"""
        portfolio = {
            'id': 1,
            'name': 'Demo Tech Portfolio',
            'creation_date': self.start_date.strftime('%Y-%m-%d'),
            'initial_value': 1000000.0,
            'positions': []
        }
        
        # Generate simulated positions
        remaining_value = portfolio['initial_value']
        for symbol in self.demo_stocks[:-1]:  # Leave last one for cash
            # Random weight between 5% and 15%
            weight = uniform(0.05, 0.15)
            value = portfolio['initial_value'] * weight
            price = uniform(50, 500)  # Simulated price
            quantity = int(value / price)
            
            position = {
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': price,
                'current_price': price * (1 + uniform(-0.1, 0.1)),  # +/- 10%
                'entry_date': self.start_date.strftime('%Y-%m-%d')
            }
            portfolio['positions'].append(position)
            remaining_value -= quantity * price
            
        # Add last position
        price = uniform(50, 500)
        quantity = int(remaining_value / price)
        portfolio['positions'].append({
            'symbol': self.demo_stocks[-1],
            'quantity': quantity,
            'entry_price': price,
            'current_price': price * (1 + uniform(-0.1, 0.1)),
            'entry_date': self.start_date.strftime('%Y-%m-%d')
        })
            
        return portfolio
        
    def generate_predictions_data(self):
        """Generate simulated ML predictions"""
        predictions = {}
        
        for symbol in self.demo_stocks:
            current_price = uniform(50, 500)
            trend = uniform(-0.15, 0.15)
            
            predictions[symbol] = {
                'price_prediction': current_price * (1 + trend),
                'confidence_interval': [
                    current_price * (1 + trend - 0.05),
                    current_price * (1 + trend + 0.05)
                ],
                'trend_probability': uniform(0.4, 0.8),
                'volatility_forecast': uniform(0.15, 0.35),
                'regime_probability': {
                    'Bull': uniform(0.2, 0.5),
                    'Neutral': uniform(0.2, 0.4),
                    'Bear': uniform(0.1, 0.3)
                },
                'technical_signals': {
                    'RSI': choice(['Overbought', 'Oversold', 'Neutral']),
                    'MACD': choice(['Bullish', 'Bearish', 'Neutral']),
                    'Stochastic': choice(['Overbought', 'Oversold', 'Neutral'])
                },
                'market_sentiment': uniform(-1, 1)
            }
            
        return predictions
        
    def generate_risk_metrics(self, portfolio):
        """Generate simulated risk metrics"""
        total_value = sum(
            pos['quantity'] * pos['current_price'] 
            for pos in portfolio['positions']
        )
        
        return {
            'var_95': total_value * uniform(0.02, 0.05),
            'expected_shortfall': total_value * uniform(0.03, 0.06),
            'volatility': uniform(0.15, 0.25),
            'sharpe_ratio': uniform(0.8, 2.0),
            'max_drawdown': uniform(0.1, 0.2)
        }
        
    def generate_anomalies(self):
        """Generate simulated market anomalies"""
        anomalies = {}
        
        for symbol in self.demo_stocks:
            if uniform(0, 1) > 0.7:  # 30% chance of anomaly
                anomalies[symbol] = {
                    'timestamp': (datetime.now() - timedelta(days=uniform(1, 30))).strftime('%Y-%m-%d'),
                    'type': choice(['Price Spike', 'Volume Surge', 'Correlation Break', 'Volatility Event']),
                    'severity': choice(['Low', 'Medium', 'High']),
                    'price_change': uniform(-0.15, 0.15),
                    'volume_change': uniform(1.5, 3.0),
                    'description': 'Significant market movement detected with unusual trading patterns.'
                }
                
        return anomalies
        
    def generate_all_demo_data(self):
        """Generate all demo data and save to files"""
        # Create demo data directory
        os.makedirs('demo_data', exist_ok=True)
        
        # Generate and save portfolio data
        portfolio = self.generate_portfolio_data()
        with open('demo_data/portfolio.json', 'w') as f:
            json.dump(portfolio, f, indent=2)
            
        # Generate and save predictions
        predictions = self.generate_predictions_data()
        with open('demo_data/predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
            
        # Generate and save risk metrics
        risk_metrics = self.generate_risk_metrics(portfolio)
        with open('demo_data/risk_metrics.json', 'w') as f:
            json.dump(risk_metrics, f, indent=2)
            
        # Generate and save anomalies
        anomalies = self.generate_anomalies()
        with open('demo_data/anomalies.json', 'w') as f:
            json.dump(anomalies, f, indent=2)
            
if __name__ == '__main__':
    generator = DemoDataGenerator()
    generator.generate_all_demo_data()
