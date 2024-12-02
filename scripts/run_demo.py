import asyncio
import sys
import os
import json
from datetime import datetime
import pandas as pd
from random import uniform, choice

class STONKSDemo:
    def __init__(self):
        self.load_demo_data()
        self.portfolio_id = 1
        
    def load_demo_data(self):
        """Load demo data from files"""
        with open('demo_data/portfolio.json', 'r') as f:
            self.portfolio = json.load(f)
        with open('demo_data/predictions.json', 'r') as f:
            self.predictions = json.load(f)
        with open('demo_data/risk_metrics.json', 'r') as f:
            self.risk_metrics = json.load(f)
        with open('demo_data/anomalies.json', 'r') as f:
            self.anomalies = json.load(f)
            
    async def demo_portfolio_overview(self):
        """Demo 1: Portfolio Overview"""
        print("\n=== Demo 1: Portfolio Overview ===")
        print(f"Portfolio: {self.portfolio['name']}")
        print(f"Initial Value: ${self.portfolio['initial_value']:,.2f}")
        
        # Calculate current value
        current_value = sum(
            pos['quantity'] * pos['current_price'] 
            for pos in self.portfolio['positions']
        )
        print(f"Current Value: ${current_value:,.2f}")
        
        # Show positions
        print("\nPositions:")
        for pos in self.portfolio['positions']:
            value = pos['quantity'] * pos['current_price']
            weight = value / current_value * 100
            print(f"{pos['symbol']}: {pos['quantity']} shares (${value:,.2f}, {weight:.1f}%)")
            
        await asyncio.sleep(5)  # Pause for demo
        
    async def demo_ml_predictions(self):
        """Demo 2: ML Predictions"""
        print("\n=== Demo 2: ML Predictions ===")
        
        for symbol in self.predictions:
            pred = self.predictions[symbol]
            print(f"\nPredictions for {symbol}:")
            print(f"Price Prediction: ${pred['price_prediction']:,.2f}")
            print(f"Confidence Interval: ${pred['confidence_interval'][0]:,.2f} - ${pred['confidence_interval'][1]:,.2f}")
            print(f"Trend Probability: {pred['trend_probability']*100:.1f}%")
            print(f"Market Sentiment: {pred['market_sentiment']:.2f}")
            
            # Show technical signals
            print("Technical Signals:")
            for indicator, signal in pred['technical_signals'].items():
                print(f"  {indicator}: {signal}")
                
            await asyncio.sleep(2)  # Pause between stocks
            
    async def demo_automated_trading(self):
        """Demo 3: Automated Trading"""
        print("\n=== Demo 3: Automated Trading ===")
        
        # Show trading parameters
        print("\nTrading Parameters:")
        print("Max Position Size: 20%")
        print("Risk Limit: 2%")
        print("Stop Loss: 10%")
        print("Take Profit: 20%")
        
        # Simulate trading activity
        print("\nMonitoring trading activity...")
        for i in range(3):
            await asyncio.sleep(2)
            
            # Random trading updates
            action = choice(['BUY', 'SELL', 'HOLD'])
            symbol = choice(list(self.predictions.keys()))
            reason = choice([
                'Price target reached',
                'Stop loss triggered',
                'Risk limit exceeded',
                'Portfolio rebalancing',
                'Anomaly detected'
            ])
            
            if action != 'HOLD':
                price = uniform(50, 500)
                quantity = int(uniform(10, 100))
                print(f"\nSignal: {action} {quantity} shares of {symbol}")
                print(f"Price: ${price:,.2f}")
                print(f"Reason: {reason}")
            else:
                print(f"\nMonitoring {symbol}: No action needed")
                print(f"Current conditions stable")
            
            print("Checking risk limits...")
            print("Evaluating market conditions...")
        
    async def demo_risk_management(self):
        """Demo 4: Risk Management"""
        print("\n=== Demo 4: Risk Management ===")
        
        # Show risk metrics
        print("\nPortfolio Risk Metrics:")
        print(f"Value at Risk (95%): ${self.risk_metrics['var_95']:,.2f}")
        print(f"Expected Shortfall: ${self.risk_metrics['expected_shortfall']:,.2f}")
        print(f"Portfolio Volatility: {self.risk_metrics['volatility']*100:.1f}%")
        print(f"Sharpe Ratio: {self.risk_metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.risk_metrics['max_drawdown']*100:.1f}%")
        
        # Show anomalies
        if self.anomalies:
            print("\nDetected Market Anomalies:")
            for symbol, anomaly in self.anomalies.items():
                print(f"\n{symbol} - {anomaly['timestamp']}")
                print(f"Type: {anomaly['type']}")
                print(f"Severity: {anomaly['severity']}")
                print(f"Price Change: {anomaly['price_change']*100:.1f}%")
                print(f"Description: {anomaly['description']}")
                
        await asyncio.sleep(5)
        
    async def run_demo(self):
        """Run full demo sequence"""
        print("\nStarting STONKS Platform Demo...")
        print("=====================================")
        
        try:
            # Run demo sections
            await self.demo_portfolio_overview()
            await self.demo_ml_predictions()
            await self.demo_automated_trading()
            await self.demo_risk_management()
            
            print("\nDemo completed successfully!")
            print("=====================================")
            print("\nNext Steps:")
            print("1. Explore the platform's features")
            print("2. Configure trading parameters")
            print("3. Set up real-time monitoring")
            print("4. Review risk management settings")
            
        except Exception as e:
            print(f"\nError during demo: {str(e)}")
            raise
            
if __name__ == '__main__':
    # Create and run demo
    demo = STONKSDemo()
    asyncio.run(demo.run_demo())
