import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from scipy import stats

class SignalType(Enum):
    PRICE_MOMENTUM = "price_momentum"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_REGIME = "volatility_regime"
    LIQUIDITY_ALERT = "liquidity_alert"
    TECHNICAL_SIGNAL = "technical_signal"
    MARKET_REGIME = "market_regime"
    CORRELATION_SIGNAL = "correlation_signal"
    RISK_ALERT = "risk_alert"

@dataclass
class MarketSignal:
    type: SignalType
    symbol: str
    timestamp: datetime
    signal_value: float
    confidence: float
    metadata: Dict
    description: str

class MarketSignalsService:
    def __init__(self):
        self.signal_thresholds = {
            SignalType.PRICE_MOMENTUM: {"strong": 2.0, "moderate": 1.0},
            SignalType.VOLUME_SPIKE: {"strong": 3.0, "moderate": 2.0},
            SignalType.VOLATILITY_REGIME: {"high": 2.0, "moderate": 1.5},
            SignalType.LIQUIDITY_ALERT: {"low": 0.3, "moderate": 0.5}
        }
        
    def analyze_price_momentum(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        lookback_periods: List[int] = [5, 15, 30]
    ) -> List[MarketSignal]:
        """Analyze price momentum across multiple timeframes"""
        signals = []
        current_price = prices.iloc[-1]
        
        for period in lookback_periods:
            if len(prices) < period:
                continue
                
            # Calculate momentum metrics
            returns = prices.pct_change(period)
            vol_adj_returns = returns / returns.rolling(period).std()
            momentum = vol_adj_returns.iloc[-1]
            
            # Calculate volume-weighted momentum
            volume_weights = volumes / volumes.rolling(period).mean()
            vw_momentum = (returns * volume_weights).sum() / volume_weights.sum()
            
            # Determine signal strength
            signal_strength = (momentum + vw_momentum) / 2
            confidence = min(abs(signal_strength) / 3, 0.95)
            
            if abs(signal_strength) > self.signal_thresholds[SignalType.PRICE_MOMENTUM]["strong"]:
                signals.append(MarketSignal(
                    type=SignalType.PRICE_MOMENTUM,
                    symbol=prices.name,
                    timestamp=datetime.now(),
                    signal_value=signal_strength,
                    confidence=confidence,
                    metadata={
                        "lookback_period": period,
                        "momentum": momentum,
                        "volume_weighted_momentum": vw_momentum,
                        "current_price": current_price
                    },
                    description=f"Strong {'bullish' if signal_strength > 0 else 'bearish'} momentum over {period} periods"
                ))
                
        return signals
        
    def detect_volume_spikes(
        self,
        volumes: pd.Series,
        prices: pd.Series,
        lookback_window: int = 20
    ) -> List[MarketSignal]:
        """Detect abnormal volume patterns"""
        signals = []
        
        if len(volumes) < lookback_window:
            return signals
            
        # Calculate volume metrics
        volume_ma = volumes.rolling(window=lookback_window).mean()
        volume_std = volumes.rolling(window=lookback_window).std()
        volume_zscore = (volumes - volume_ma) / volume_std
        
        # Calculate price-volume correlation
        returns = prices.pct_change()
        pv_correlation = returns.rolling(lookback_window).corr(volumes)
        
        current_zscore = volume_zscore.iloc[-1]
        current_pv_corr = pv_correlation.iloc[-1]
        
        if abs(current_zscore) > self.signal_thresholds[SignalType.VOLUME_SPIKE]["strong"]:
            confidence = min(abs(current_zscore) / 5, 0.95)
            
            signals.append(MarketSignal(
                type=SignalType.VOLUME_SPIKE,
                symbol=volumes.name,
                timestamp=datetime.now(),
                signal_value=current_zscore,
                confidence=confidence,
                metadata={
                    "volume": volumes.iloc[-1],
                    "volume_ma": volume_ma.iloc[-1],
                    "price_volume_correlation": current_pv_corr,
                    "lookback_window": lookback_window
                },
                description=f"Significant volume spike detected ({current_zscore:.2f} std devs)"
            ))
            
        return signals
        
    def analyze_volatility_regime(
        self,
        prices: pd.Series,
        lookback_window: int = 30
    ) -> List[MarketSignal]:
        """Analyze volatility regime changes"""
        signals = []
        
        if len(prices) < lookback_window:
            return signals
            
        # Calculate volatility metrics
        returns = prices.pct_change()
        rolling_vol = returns.rolling(window=lookback_window).std() * np.sqrt(252)
        vol_of_vol = rolling_vol.rolling(window=lookback_window).std()
        
        # Detect regime changes
        vol_zscore = (rolling_vol - rolling_vol.rolling(window=lookback_window).mean()) / \
                    rolling_vol.rolling(window=lookback_window).std()
                    
        current_vol = rolling_vol.iloc[-1]
        current_vol_zscore = vol_zscore.iloc[-1]
        
        if abs(current_vol_zscore) > self.signal_thresholds[SignalType.VOLATILITY_REGIME]["high"]:
            confidence = min(abs(current_vol_zscore) / 4, 0.95)
            
            signals.append(MarketSignal(
                type=SignalType.VOLATILITY_REGIME,
                symbol=prices.name,
                timestamp=datetime.now(),
                signal_value=current_vol_zscore,
                confidence=confidence,
                metadata={
                    "current_volatility": current_vol,
                    "volatility_zscore": current_vol_zscore,
                    "vol_of_vol": vol_of_vol.iloc[-1],
                    "lookback_window": lookback_window
                },
                description=f"{'High' if current_vol_zscore > 0 else 'Low'} volatility regime detected"
            ))
            
        return signals
        
    def monitor_liquidity(
        self,
        bid_sizes: pd.Series,
        ask_sizes: pd.Series,
        spreads: pd.Series,
        lookback_window: int = 20
    ) -> List[MarketSignal]:
        """Monitor market liquidity conditions"""
        signals = []
        
        if len(bid_sizes) < lookback_window:
            return signals
            
        # Calculate liquidity metrics
        total_depth = (bid_sizes + ask_sizes).rolling(window=lookback_window).mean()
        spread_ma = spreads.rolling(window=lookback_window).mean()
        
        # Normalized liquidity score
        liquidity_score = (total_depth / total_depth.mean()) * (spread_ma.mean() / spread_ma)
        current_score = liquidity_score.iloc[-1]
        
        if current_score < self.signal_thresholds[SignalType.LIQUIDITY_ALERT]["low"]:
            confidence = min((1 - current_score) * 2, 0.95)
            
            signals.append(MarketSignal(
                type=SignalType.LIQUIDITY_ALERT,
                symbol=bid_sizes.name,
                timestamp=datetime.now(),
                signal_value=current_score,
                confidence=confidence,
                metadata={
                    "total_depth": total_depth.iloc[-1],
                    "spread": spreads.iloc[-1],
                    "bid_ask_ratio": (bid_sizes / ask_sizes).iloc[-1],
                    "lookback_window": lookback_window
                },
                description=f"Low liquidity conditions detected (score: {current_score:.2f})"
            ))
            
        return signals
        
    def generate_technical_signals(
        self,
        prices: pd.Series,
        volumes: pd.Series
    ) -> List[MarketSignal]:
        """Generate technical analysis signals"""
        signals = []
        
        # Calculate technical indicators
        sma_20 = prices.rolling(window=20).mean()
        sma_50 = prices.rolling(window=50).mean()
        rsi = self.calculate_rsi(prices)
        macd, signal, hist = self.calculate_macd(prices)
        
        current_price = prices.iloc[-1]
        
        # Moving average crossover
        if sma_20.iloc[-2] < sma_50.iloc[-2] and sma_20.iloc[-1] > sma_50.iloc[-1]:
            signals.append(MarketSignal(
                type=SignalType.TECHNICAL_SIGNAL,
                symbol=prices.name,
                timestamp=datetime.now(),
                signal_value=1.0,
                confidence=0.7,
                metadata={
                    "indicator": "MA_CROSSOVER",
                    "sma_20": sma_20.iloc[-1],
                    "sma_50": sma_50.iloc[-1],
                    "current_price": current_price
                },
                description="Bullish MA crossover detected"
            ))
            
        # RSI signals
        if rsi.iloc[-1] < 30 or rsi.iloc[-1] > 70:
            signals.append(MarketSignal(
                type=SignalType.TECHNICAL_SIGNAL,
                symbol=prices.name,
                timestamp=datetime.now(),
                signal_value=rsi.iloc[-1],
                confidence=0.6,
                metadata={
                    "indicator": "RSI",
                    "rsi_value": rsi.iloc[-1],
                    "current_price": current_price
                },
                description=f"{'Oversold' if rsi.iloc[-1] < 30 else 'Overbought'} RSI conditions"
            ))
            
        # MACD signals
        if hist.iloc[-2] < 0 and hist.iloc[-1] > 0:
            signals.append(MarketSignal(
                type=SignalType.TECHNICAL_SIGNAL,
                symbol=prices.name,
                timestamp=datetime.now(),
                signal_value=1.0,
                confidence=0.65,
                metadata={
                    "indicator": "MACD",
                    "macd": macd.iloc[-1],
                    "signal": signal.iloc[-1],
                    "histogram": hist.iloc[-1],
                    "current_price": current_price
                },
                description="Bullish MACD crossover detected"
            ))
            
        return signals
        
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
        
    def analyze_market_regime(
        self,
        prices: pd.Series,
        market_prices: pd.Series,
        lookback_window: int = 60
    ) -> List[MarketSignal]:
        """Analyze market regime and correlation patterns"""
        signals = []
        
        if len(prices) < lookback_window or len(market_prices) < lookback_window:
            return signals
            
        # Calculate correlation and beta
        returns = prices.pct_change()
        market_returns = market_prices.pct_change()
        
        rolling_corr = returns.rolling(window=lookback_window).corr(market_returns)
        rolling_beta = (returns.rolling(window=lookback_window).cov(market_returns) / 
                       market_returns.rolling(window=lookback_window).var())
                       
        # Detect regime changes
        corr_zscore = (rolling_corr - rolling_corr.rolling(window=lookback_window).mean()) / \
                     rolling_corr.rolling(window=lookback_window).std()
                     
        beta_zscore = (rolling_beta - rolling_beta.rolling(window=lookback_window).mean()) / \
                     rolling_beta.rolling(window=lookback_window).std()
                     
        if abs(corr_zscore.iloc[-1]) > 2 or abs(beta_zscore.iloc[-1]) > 2:
            signals.append(MarketSignal(
                type=SignalType.MARKET_REGIME,
                symbol=prices.name,
                timestamp=datetime.now(),
                signal_value=corr_zscore.iloc[-1],
                confidence=min(abs(corr_zscore.iloc[-1]) / 4, 0.95),
                metadata={
                    "correlation": rolling_corr.iloc[-1],
                    "beta": rolling_beta.iloc[-1],
                    "corr_zscore": corr_zscore.iloc[-1],
                    "beta_zscore": beta_zscore.iloc[-1],
                    "lookback_window": lookback_window
                },
                description="Significant change in market regime detected"
            ))
            
        return signals

market_signals_service = MarketSignalsService()
