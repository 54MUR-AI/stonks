import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import scipy.stats as stats

logger = logging.getLogger(__name__)

class BacktestVisualizer:
    """Visualization tools for backtest results"""
    
    def __init__(self, style: str = 'darkgrid'):
        """Initialize visualizer with specified style"""
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def plot_portfolio_performance(self,
                                 portfolio_values: pd.Series,
                                 benchmark_values: Optional[pd.Series] = None,
                                 title: str = "Portfolio Performance") -> None:
        """Plot portfolio value over time with optional benchmark comparison"""
        plt.figure(figsize=(12, 6))
        
        # Plot portfolio values
        portfolio_returns = portfolio_values.pct_change()
        cumulative_returns = (1 + portfolio_returns).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns.values, 
                label='Portfolio', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_values is not None:
            benchmark_returns = benchmark_values.pct_change()
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            plt.plot(benchmark_cumulative.index, benchmark_cumulative.values,
                    label='Benchmark', linewidth=2, linestyle='--')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
    def plot_drawdown(self, portfolio_values: pd.Series) -> None:
        """Plot drawdown chart"""
        plt.figure(figsize=(12, 6))
        
        # Calculate drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        
        plt.plot(drawdown.index, drawdown.values * 100)
        plt.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.3)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        
    def plot_returns_distribution(self, returns: pd.Series) -> None:
        """Plot returns distribution with normal distribution overlay"""
        plt.figure(figsize=(12, 6))
        
        # Plot returns distribution
        sns.histplot(returns, bins=50, stat='density', alpha=0.5)
        
        # Add normal distribution overlay
        x = np.linspace(returns.min(), returns.max(), 100)
        plt.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
                'r-', lw=2, label='Normal Distribution')
        
        plt.title('Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
    def plot_rolling_metrics(self, returns: pd.Series, window: int = 252) -> None:
        """Plot rolling Sharpe ratio and volatility"""
        plt.figure(figsize=(12, 8))
        
        # Calculate rolling metrics
        rolling_return = returns.rolling(window=window).mean() * 252
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rolling Sharpe ratio
        ax1.plot(rolling_sharpe.index, rolling_sharpe.values)
        ax1.set_title(f'Rolling Sharpe Ratio ({window} days)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.grid(True)
        
        # Plot rolling volatility
        ax2.plot(rolling_vol.index, rolling_vol.values * 100)
        ax2.set_title(f'Rolling Volatility ({window} days)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        
    def plot_trade_analysis(self, trades: pd.DataFrame) -> None:
        """Plot trade analysis charts"""
        if trades.empty:
            logger.warning("No trades to analyze")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Trade PnL distribution
        plt.subplot(2, 2, 1)
        if 'price' in trades.columns and 'quantity' in trades.columns:
            trades['pnl'] = trades['quantity'] * trades['price']
            sns.histplot(data=trades, x='pnl', bins=30)
            plt.title('Trade PnL Distribution')
            plt.xlabel('PnL')
            plt.ylabel('Count')
        
        # Plot 2: Trades by symbol
        plt.subplot(2, 2, 2)
        if 'symbol' in trades.columns:
            trades_by_symbol = trades.groupby('symbol')['quantity'].count()
            trades_by_symbol.plot(kind='bar')
            plt.title('Number of Trades by Symbol')
            plt.xlabel('Symbol')
            plt.ylabel('Number of Trades')
            plt.xticks(rotation=45)
        
        # Plot 3: Trade volume over time
        plt.subplot(2, 2, 3)
        if 'timestamp' in trades.columns and 'quantity' in trades.columns and 'price' in trades.columns:
            trades['trade_value'] = trades['quantity'].abs() * trades['price']
            daily_volume = trades.groupby('timestamp')['trade_value'].sum()
            daily_volume.plot()
            plt.title('Trading Volume Over Time')
            plt.xlabel('Date')
            plt.ylabel('Volume')
        
        # Plot 4: Trade reasons distribution
        plt.subplot(2, 2, 4)
        if 'reason' in trades.columns:
            trades_by_reason = trades['reason'].value_counts()
            trades_by_reason.plot(kind='pie', autopct='%1.1f%%')
            plt.title('Trade Reasons Distribution')
        
        plt.tight_layout()
        
    def plot_efficient_frontier(self, efficient_frontier: pd.DataFrame,
                              current_portfolio: Optional[Tuple[float, float]] = None) -> None:
        """Plot efficient frontier with optional current portfolio point"""
        plt.figure(figsize=(10, 6))
        
        # Plot efficient frontier
        plt.scatter(efficient_frontier['Volatility'], 
                   efficient_frontier['Return'],
                   c=efficient_frontier['Sharpe Ratio'],
                   cmap='viridis',
                   s=50)
        
        # Plot current portfolio if provided
        if current_portfolio is not None:
            plt.scatter(current_portfolio[1],
                       current_portfolio[0],
                       color='red',
                       marker='*',
                       s=200,
                       label='Current Portfolio')
        
        plt.colorbar(label='Sharpe Ratio')
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
    def plot_risk_contributions(self, weights: Dict[str, float],
                              risk_contributions: Dict[str, float]) -> None:
        """Plot portfolio weights vs risk contributions"""
        plt.figure(figsize=(12, 6))
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Weight': weights,
            'Risk Contribution': risk_contributions
        })
        
        # Plot grouped bar chart
        df.plot(kind='bar', width=0.8)
        plt.title('Portfolio Weights vs Risk Contributions')
        plt.xlabel('Asset')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
