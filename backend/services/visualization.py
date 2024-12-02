import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import scipy.stats as stats
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

class VisualizationService:
    def __init__(self):
        """Initialize the visualization service"""
        # Set style
        plt.style.use('seaborn-v0_8')  # Use a valid style name
        sns.set_palette("husl")
    
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

    def plot_factor_analysis(self, factor_summary: Dict,
                        save_path: Optional[str] = None) -> None:
        """
        Create comprehensive factor analysis visualization
        
        Args:
            factor_summary: Dictionary from PortfolioFactorAnalyzer.get_factor_summary()
            save_path: Optional path to save the plot
        """
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Factor Returns Over Time
        ax1 = fig.add_subplot(gs[0, 0])
        factor_returns = factor_summary['factor_returns']
        cumulative_returns = (1 + factor_returns).cumprod()
        cumulative_returns.plot(ax=ax1)
        ax1.set_title('Cumulative Factor Returns')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Factor Risk Contribution
        ax2 = fig.add_subplot(gs[0, 1])
        risk_contrib = factor_summary['factor_metrics']['Contribution to Risk']
        risk_contrib.plot(kind='bar', ax=ax2)
        ax2.set_title('Factor Risk Contribution')
        ax2.set_ylabel('Contribution to Total Risk')
        
        # 3. Factor-Asset Correlations Heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        correlations = factor_summary['factor_correlations']
        sns.heatmap(correlations, annot=True, cmap='RdBu', center=0, ax=ax3)
        ax3.set_title('Factor-Asset Correlations')
        
        # 4. Factor Performance Metrics
        ax4 = fig.add_subplot(gs[1, :2])
        metrics = factor_summary['factor_metrics']
        metrics_to_plot = ['Annualized Return', 'Annualized Vol', 'Sharpe Ratio']
        metrics[metrics_to_plot].plot(kind='bar', ax=ax4)
        ax4.set_title('Factor Performance Metrics')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 5. Cumulative Variance Explained
        ax5 = fig.add_subplot(gs[1, 2])
        cum_var = factor_summary['factor_analysis']['cumulative_variance']
        cum_var.plot(kind='bar', ax=ax5)
        ax5.set_title('Cumulative Variance Explained')
        ax5.set_ylabel('Cumulative Proportion')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()

    def plot_factor_contribution_over_time(self, factor_returns: pd.DataFrame,
                                     save_path: Optional[str] = None) -> None:
        """
        Create stacked area plot of factor contributions over time
        
        Args:
            factor_returns: DataFrame of factor returns from decompose_returns()
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate cumulative returns
        cumulative_returns = (1 + factor_returns).cumprod()
        
        # Create stacked area plot
        cumulative_returns.plot(kind='area', stacked=True)
        
        plt.title('Factor Return Contribution Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()

    def plot_rolling_factor_risk(self, factor_returns: pd.DataFrame,
                           window: int = 63,  # ~3 months
                           save_path: Optional[str] = None) -> None:
        """
        Plot rolling risk metrics for each factor
        
        Args:
            factor_returns: DataFrame of factor returns
            window: Rolling window size in days
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Calculate rolling volatility
        rolling_vol = factor_returns.rolling(window).std() * np.sqrt(252)
        rolling_vol.plot(ax=ax1)
        ax1.set_title(f'{window}-day Rolling Volatility')
        ax1.set_ylabel('Annualized Volatility')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Calculate rolling Sharpe ratio
        rolling_ret = factor_returns.rolling(window).mean() * 252
        rolling_sharpe = rolling_ret / rolling_vol
        rolling_sharpe.plot(ax=ax2)
        ax2.set_title(f'{window}-day Rolling Sharpe Ratio')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()
