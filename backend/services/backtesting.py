import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, 
                 initial_capital: float = 1_000_000.0,
                 transaction_cost: float = 0.001,
                 risk_free_rate: float = 0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.positions: Dict[str, float] = {}
        self.trades: List[dict] = []
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.metrics: Dict[str, float] = {}
        
    def load_market_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Load historical market data for backtesting"""
        self.market_data = data
        self.timestamps = sorted(list(set.union(*[set(df.index) for df in data.values()])))
        
    def execute_trade(self, timestamp: datetime, symbol: str, quantity: float, price: float, reason: str = None) -> None:
        """Execute a trade and update positions"""
        cost = quantity * price * (1 + self.transaction_cost)
        if quantity > 0:  # Buy
            if cost > self.current_capital:
                logger.warning(f"Insufficient capital for trade: {symbol} {quantity} @ {price}")
                return
            self.current_capital -= cost
        else:  # Sell
            if symbol not in self.positions or self.positions[symbol] < abs(quantity):
                logger.warning(f"Insufficient position for trade: {symbol} {quantity}")
                return
            self.current_capital += abs(cost)
            
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        if self.positions[symbol] == 0:
            del self.positions[symbol]
            
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'reason': reason
        })
        
    def calculate_portfolio_value(self, timestamp: datetime) -> float:
        """Calculate total portfolio value at given timestamp"""
        total_value = self.current_capital
        for symbol, quantity in self.positions.items():
            if timestamp in self.market_data[symbol].index:
                price = self.market_data[symbol].loc[timestamp, 'close']
                total_value += quantity * price
        return total_value
        
    def run_backtest(self, strategy) -> Dict[str, float]:
        """Run backtest using provided strategy"""
        logger.info("Starting backtest...")
        self.current_capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.portfolio_values.clear()
        
        for timestamp in self.timestamps:
            # Get current market state
            market_state = {
                symbol: df.loc[timestamp] if timestamp in df.index else None 
                for symbol, df in self.market_data.items()
            }
            
            # Calculate current portfolio value
            portfolio_value = self.calculate_portfolio_value(timestamp)
            
            # Get strategy signals
            signals = strategy.generate_signals(
                timestamp, 
                market_state, 
                self.positions,
                portfolio_value
            )
            
            # Execute trades based on signals
            for signal in signals:
                self.execute_trade(
                    timestamp=timestamp,
                    symbol=signal['symbol'],
                    quantity=signal['quantity'],
                    price=signal['price'],
                    reason=signal.get('reason', 'strategy')
                )
                
            # Record portfolio value
            self.portfolio_values.append((timestamp, portfolio_value))
            
        # Calculate performance metrics
        self.calculate_performance_metrics()
        logger.info("Backtest completed successfully")
        return self.metrics
        
    def calculate_performance_metrics(self) -> None:
        """Calculate various performance metrics"""
        values = pd.Series([v for _, v in self.portfolio_values])
        returns = values.pct_change().dropna()
        
        # Basic metrics
        total_return = (values.iloc[-1] - self.initial_capital) / self.initial_capital
        daily_returns_mean = returns.mean()
        daily_returns_std = returns.std()
        
        # Risk metrics
        sharpe_ratio = (daily_returns_mean - self.risk_free_rate/252) / daily_returns_std * np.sqrt(252)
        sortino_ratio = (daily_returns_mean - self.risk_free_rate/252) / returns[returns < 0].std() * np.sqrt(252)
        max_drawdown = ((values - values.cummax()) / values.cummax()).min()
        
        # Trading metrics
        winning_trades = len([t for t in self.trades if 
            (t['quantity'] > 0 and t['price'] < self.market_data[t['symbol']].loc[t['timestamp']:].iloc[-1]['close']) or
            (t['quantity'] < 0 and t['price'] > self.market_data[t['symbol']].loc[t['timestamp']:].iloc[-1]['close'])
        ])
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        self.metrics = {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(returns)),
            'annualized_volatility': daily_returns_std * np.sqrt(252),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
        
class BaseStrategy:
    """Base class for trading strategies"""
    def generate_signals(self, timestamp: datetime, 
                        market_state: Dict[str, pd.Series], 
                        current_positions: Dict[str, float],
                        portfolio_value: float) -> List[dict]:
        """Generate trading signals based on market state"""
        raise NotImplementedError("Strategy must implement generate_signals method")
        
class MovingAverageCrossStrategy(BaseStrategy):
    """Simple moving average crossover strategy"""
    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window
        self.historical_data = {}
        
    def generate_signals(self, timestamp: datetime,
                        market_state: Dict[str, pd.Series],
                        current_positions: Dict[str, float],
                        portfolio_value: float) -> List[dict]:
        signals = []
        
        # Update historical data
        for symbol, data in market_state.items():
            if data is None:
                continue
                
            if symbol not in self.historical_data:
                self.historical_data[symbol] = []
            self.historical_data[symbol].append({
                'timestamp': timestamp,
                'close': data['close']
            })
            
            # Convert to DataFrame for calculations
            hist_df = pd.DataFrame(self.historical_data[symbol])
            if len(hist_df) < self.long_window:
                continue
                
            # Calculate moving averages
            short_ma = hist_df['close'].rolling(window=self.short_window).mean().iloc[-1]
            long_ma = hist_df['close'].rolling(window=self.long_window).mean().iloc[-1]
            
            current_position = current_positions.get(symbol, 0)
            current_price = data['close']
            
            # Generate trading signals
            if short_ma > long_ma and current_position <= 0:
                # Buy signal
                position_size = 100  # Fixed position size for demo
                signals.append({
                    'symbol': symbol,
                    'quantity': position_size,
                    'price': current_price,
                    'reason': 'MA crossover'
                })
            elif short_ma < long_ma and current_position > 0:
                # Sell signal
                signals.append({
                    'symbol': symbol,
                    'quantity': -current_position,
                    'price': current_price,
                    'reason': 'MA crossover'
                })
                
        return signals
