import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.services.backtesting import BacktestEngine
from backend.services.strategies import TechnicalStrategy

def generate_test_data(symbols, start_date, end_date, freq='D'):
    """Generate sample market data for testing"""
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = {}
    
    for symbol in symbols:
        # Generate random walk prices with trend and seasonality
        np.random.seed(hash(symbol) % 2**32)
        t = np.arange(len(dates))
        trend = 0.0002 * t  # Upward trend
        seasonality = 0.1 * np.sin(2 * np.pi * t / 252)  # Annual cycle
        returns = np.random.normal(0.0005, 0.02, size=len(dates))
        noise = np.cumsum(returns)
        
        # Combine components
        log_prices = 4.6 + trend + seasonality + noise
        prices = np.exp(log_prices)
        
        # Create OHLCV data
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, size=len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.004, size=len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.004, size=len(dates)))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, size=len(dates))
        }, index=dates)
        
        # Ensure OHLC relationship
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        data[symbol] = df
        
    return data

def print_trade_analysis(trades_df):
    """Print detailed trade analysis"""
    if trades_df.empty:
        print("No trades executed")
        return
        
    print("\nTrade Analysis:")
    print("==============")
    
    # Trade reasons
    print("\nTrades by Reason:")
    reason_counts = trades_df['reason'].value_counts()
    for reason, count in reason_counts.items():
        print(f"{reason}: {count} trades")
        
    # Calculate trade statistics
    trades_df['trade_value'] = trades_df['quantity'] * trades_df['price']
    total_value = trades_df['trade_value'].abs().sum()
    
    print(f"\nTotal Trading Volume: ${total_value:,.2f}")
    print(f"Average Trade Size: ${(total_value / len(trades_df)):,.2f}")
    
    # Trade distribution
    print("\nTrade Distribution:")
    trades_by_symbol = trades_df.groupby('symbol').agg({
        'quantity': 'count',
        'trade_value': lambda x: sum(abs(x))
    }).sort_values('trade_value', ascending=False)
    
    for symbol, row in trades_by_symbol.iterrows():
        print(f"{symbol}: {row['quantity']} trades, ${row['trade_value']:,.2f} total volume")

def main():
    # Test parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    initial_capital = 1_000_000
    
    print("Generating test data...")
    market_data = generate_test_data(symbols, start_date, end_date)
    
    print("\nInitializing backtest engine...")
    engine = BacktestEngine(initial_capital=initial_capital)
    engine.load_market_data(market_data)
    
    print("\nCreating and running technical strategy...")
    strategy = TechnicalStrategy(
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        ma_short=20,
        ma_medium=50,
        ma_long=200,
        volatility_window=20,
        max_position_size=0.2,
        stop_loss=0.02,
        take_profit=0.05
    )
    
    metrics = engine.run_backtest(strategy)
    
    print("\nBacktest Results:")
    print("================")
    for metric, value in metrics.items():
        if metric in ['win_rate', 'total_return', 'max_drawdown']:
            print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
        elif metric == 'total_trades':
            print(f"{metric.replace('_', ' ').title()}: {int(value)}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            
    # Detailed trade analysis
    trades_df = pd.DataFrame(engine.trades)
    print_trade_analysis(trades_df)
    
    print("\nPortfolio Value History:")
    print("=====================")
    portfolio_values = pd.DataFrame(engine.portfolio_values, 
                                  columns=['timestamp', 'value']).set_index('timestamp')
    print(f"Starting Value: ${initial_capital:,.2f}")
    print(f"Ending Value: ${portfolio_values['value'].iloc[-1]:,.2f}")
    print(f"Peak Value: ${portfolio_values['value'].max():,.2f}")
    print(f"Trough Value: ${portfolio_values['value'].min():,.2f}")
    
    # Calculate additional metrics
    returns = portfolio_values['value'].pct_change()
    print(f"\nDaily Returns Statistics:")
    print(f"Average Daily Return: {returns.mean():.4%}")
    print(f"Daily Return Std Dev: {returns.std():.4%}")
    print(f"Skewness: {returns.skew():.4f}")
    print(f"Kurtosis: {returns.kurtosis():.4f}")

if __name__ == '__main__':
    main()
