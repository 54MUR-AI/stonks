import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TechnicalStrategy:
    """Advanced technical analysis strategy with risk management"""
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 ma_short: int = 20,
                 ma_medium: int = 50,
                 ma_long: int = 200,
                 volatility_window: int = 20,
                 max_position_size: float = 0.1,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.05):
        
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.ma_short = ma_short
        self.ma_medium = ma_medium
        self.ma_long = ma_long
        self.volatility_window = volatility_window
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        self.historical_data = {}
        self.entry_prices = {}
        
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))
        
    def calculate_moving_averages(self, prices: pd.Series) -> tuple:
        """Calculate multiple moving averages"""
        ma_s = prices.rolling(window=self.ma_short).mean().iloc[-1]
        ma_m = prices.rolling(window=self.ma_medium).mean().iloc[-1]
        ma_l = prices.rolling(window=self.ma_long).mean().iloc[-1]
        return ma_s, ma_m, ma_l
        
    def calculate_volatility(self, prices: pd.Series) -> float:
        """Calculate historical volatility"""
        returns = prices.pct_change()
        return returns.rolling(window=self.volatility_window).std().iloc[-1]
        
    def check_stop_loss(self, symbol: str, current_price: float, 
                       position: float) -> Optional[dict]:
        """Check if stop loss has been triggered"""
        if symbol in self.entry_prices and position != 0:
            entry_price = self.entry_prices[symbol]
            if position > 0:  # Long position
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct < -self.stop_loss:
                    return {
                        'symbol': symbol,
                        'quantity': -position,
                        'price': current_price,
                        'reason': 'stop_loss'
                    }
            else:  # Short position
                loss_pct = (entry_price - current_price) / entry_price
                if loss_pct < -self.stop_loss:
                    return {
                        'symbol': symbol,
                        'quantity': -position,
                        'price': current_price,
                        'reason': 'stop_loss'
                    }
        return None
        
    def check_take_profit(self, symbol: str, current_price: float, 
                         position: float) -> Optional[dict]:
        """Check if take profit has been triggered"""
        if symbol in self.entry_prices and position != 0:
            entry_price = self.entry_prices[symbol]
            if position > 0:  # Long position
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct > self.take_profit:
                    return {
                        'symbol': symbol,
                        'quantity': -position,
                        'price': current_price,
                        'reason': 'take_profit'
                    }
            else:  # Short position
                profit_pct = (entry_price - current_price) / entry_price
                if profit_pct > self.take_profit:
                    return {
                        'symbol': symbol,
                        'quantity': -position,
                        'price': current_price,
                        'reason': 'take_profit'
                    }
        return None
        
    def generate_signals(self, timestamp: datetime,
                        market_state: Dict[str, pd.Series],
                        current_positions: Dict[str, float],
                        portfolio_value: float) -> List[dict]:
        signals = []
        
        for symbol, data in market_state.items():
            if data is None:
                continue
                
            # Update historical data
            if symbol not in self.historical_data:
                self.historical_data[symbol] = []
            self.historical_data[symbol].append({
                'timestamp': timestamp,
                'close': data['close']
            })
            
            # Convert to DataFrame for calculations
            hist_df = pd.DataFrame(self.historical_data[symbol])
            if len(hist_df) < self.ma_long:
                continue
                
            current_price = data['close']
            current_position = current_positions.get(symbol, 0)
            
            # Check stop loss and take profit
            if current_position != 0:
                stop_loss_signal = self.check_stop_loss(symbol, current_price, current_position)
                if stop_loss_signal:
                    signals.append(stop_loss_signal)
                    continue
                    
                take_profit_signal = self.check_take_profit(symbol, current_price, current_position)
                if take_profit_signal:
                    signals.append(take_profit_signal)
                    continue
            
            # Calculate technical indicators
            prices = pd.Series(hist_df['close'])
            rsi = self.calculate_rsi(prices)
            ma_short, ma_medium, ma_long = self.calculate_moving_averages(prices)
            volatility = self.calculate_volatility(prices)
            
            # Trading logic
            buy_signal = (
                rsi < self.rsi_oversold and
                ma_short > ma_medium and
                ma_medium > ma_long and
                volatility < 0.02  # Low volatility environment
            )
            
            sell_signal = (
                rsi > self.rsi_overbought or
                (ma_short < ma_medium and ma_medium < ma_long) or
                volatility > 0.04  # High volatility environment
            )
            
            # Position sizing
            position_value = abs(current_position * current_price)
            max_position_value = portfolio_value * self.max_position_size
            
            if buy_signal and current_position <= 0:
                # Calculate position size
                position_size = int(max_position_value / current_price)
                if position_size > 0:
                    signals.append({
                        'symbol': symbol,
                        'quantity': position_size,
                        'price': current_price,
                        'reason': 'technical_buy'
                    })
                    self.entry_prices[symbol] = current_price
                    
            elif sell_signal and current_position > 0:
                signals.append({
                    'symbol': symbol,
                    'quantity': -current_position,
                    'price': current_price,
                    'reason': 'technical_sell'
                })
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                    
        return signals
