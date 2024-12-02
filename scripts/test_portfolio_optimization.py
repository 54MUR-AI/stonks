import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.services.backtesting import BacktestEngine
from backend.services.strategies import TechnicalStrategy
from backend.services.portfolio_optimizer import PortfolioOptimizer, RiskParityOptimizer
from backend.services.visualization import BacktestVisualizer

def run_portfolio_optimization(market_data: dict, initial_capital: float = 1_000_000):
    """Run portfolio optimization analysis"""
    print("\nRunning Portfolio Optimization Analysis...")
    print("========================================")
    
    # Initialize optimizers
    mvo_optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    rp_optimizer = RiskParityOptimizer(risk_target=0.15)
    
    # Run mean-variance optimization
    print("\nMean-Variance Optimization Results:")
    print("----------------------------------")
    mvo_weights = mvo_optimizer.optimize_portfolio(market_data)
    print("\nOptimal Portfolio Weights:")
    for symbol, weight in mvo_weights.items():
        print(f"{symbol}: {weight:.2%}")
        
    # Run risk parity optimization
    print("\nRisk Parity Optimization Results:")
    print("--------------------------------")
    rp_weights = rp_optimizer.optimize_portfolio(market_data)
    print("\nRisk Parity Weights:")
    for symbol, weight in rp_weights.items():
        print(f"{symbol}: {weight:.2%}")
        
    # Generate efficient frontier
    print("\nGenerating Efficient Frontier...")
    efficient_frontier = mvo_optimizer.generate_efficient_frontier(market_data)
    
    return mvo_weights, rp_weights, efficient_frontier

def run_strategy_backtest(market_data: dict, 
                         optimal_weights: dict,
                         initial_capital: float = 1_000_000):
    """Run backtest with optimized portfolio weights"""
    print("\nRunning Strategy Backtest...")
    print("===========================")
    
    # Initialize components
    engine = BacktestEngine(initial_capital=initial_capital)
    engine.load_market_data(market_data)
    
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
    
    # Set initial positions based on optimal weights
    initial_positions = {
        symbol: int((weight * initial_capital) / market_data[symbol]['close'].iloc[0])
        for symbol, weight in optimal_weights.items()
    }
    engine.positions = initial_positions
    
    # Run backtest
    metrics = engine.run_backtest(strategy)
    
    return engine, metrics

def visualize_results(engine, efficient_frontier, visualizer):
    """Create visualization of backtest and optimization results"""
    print("\nGenerating Visualizations...")
    print("==========================")
    
    # Convert portfolio values to DataFrame
    portfolio_values = pd.DataFrame(engine.portfolio_values,
                                  columns=['timestamp', 'value']).set_index('timestamp')
    
    # Plot portfolio performance
    visualizer.plot_portfolio_performance(portfolio_values['value'])
    plt.savefig('portfolio_performance.png')
    
    # Plot drawdown
    visualizer.plot_drawdown(portfolio_values['value'])
    plt.savefig('drawdown.png')
    
    # Plot trade analysis
    trades_df = pd.DataFrame(engine.trades)
    if not trades_df.empty:
        visualizer.plot_trade_analysis(trades_df)
        plt.savefig('trade_analysis.png')
    
    # Plot efficient frontier
    current_portfolio = (
        engine.metrics['annualized_return'],
        engine.metrics['annualized_volatility']
    )
    visualizer.plot_efficient_frontier(efficient_frontier, current_portfolio)
    plt.savefig('efficient_frontier.png')
    
    print("\nVisualization files saved:")
    print("- portfolio_performance.png")
    print("- drawdown.png")
    print("- trade_analysis.png")
    print("- efficient_frontier.png")

def main():
    # Test parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    initial_capital = 1_000_000
    
    print("Generating test data...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    market_data = {}
    
    for symbol in symbols:
        # Generate price data with more realistic properties
        np.random.seed(hash(symbol) % 2**32)
        
        # Generate components
        t = np.arange(len(dates))
        trend = 0.0002 * t  # Upward trend
        seasonality = 0.1 * np.sin(2 * np.pi * t / 252)  # Annual cycle
        volatility = 0.02 * (1 + 0.5 * np.sin(2 * np.pi * t / 63))  # Quarterly vol cycle
        
        # Generate returns with time-varying volatility
        returns = np.random.normal(0, volatility)
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
        market_data[symbol] = df
    
    # Run portfolio optimization
    mvo_weights, rp_weights, efficient_frontier = run_portfolio_optimization(
        market_data, initial_capital
    )
    
    # Run backtest with mean-variance optimized weights
    print("\nBacktesting with Mean-Variance Optimized Portfolio:")
    engine_mvo, metrics_mvo = run_strategy_backtest(
        market_data, mvo_weights, initial_capital
    )
    
    # Run backtest with risk parity weights
    print("\nBacktesting with Risk Parity Portfolio:")
    engine_rp, metrics_rp = run_strategy_backtest(
        market_data, rp_weights, initial_capital
    )
    
    # Initialize visualizer
    visualizer = BacktestVisualizer()
    
    # Create visualizations for both strategies
    print("\nGenerating visualizations for Mean-Variance Optimized Portfolio:")
    visualize_results(engine_mvo, efficient_frontier, visualizer)
    
    print("\nGenerating visualizations for Risk Parity Portfolio:")
    visualize_results(engine_rp, efficient_frontier, visualizer)
    
    # Compare strategies
    print("\nStrategy Comparison:")
    print("===================")
    metrics_comparison = pd.DataFrame({
        'Mean-Variance': metrics_mvo,
        'Risk Parity': metrics_rp
    })
    print(metrics_comparison)

if __name__ == '__main__':
    main()
