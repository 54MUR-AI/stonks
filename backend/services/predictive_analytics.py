import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import statsmodels.api as sm
from arch import arch_model
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PredictionResults:
    price_prediction: float
    price_confidence: Tuple[float, float]
    trend_probability: float
    volatility_forecast: float
    regime_probability: Dict[str, float]
    support_resistance: Dict[str, float]
    technical_signals: Dict[str, str]
    fundamental_score: float
    market_sentiment: float
    prediction_horizon: int

class PredictiveAnalytics:
    def __init__(self):
        self.price_model = None
        self.trend_model = None
        self.volatility_model = None
        self.regime_model = None
        self.sentiment_model = None
        self.scaler = StandardScaler()
        self.lookback_period = 60
        self.prediction_horizons = [1, 5, 10, 20]  # Days
        
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for prediction"""
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close']).diff()
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            features[f'ma_{window}'] = data['Close'].rolling(window=window).mean()
            features[f'ma_ratio_{window}'] = data['Close'] / features[f'ma_{window}']
            
        # Volatility features
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()
            
        # Price ranges
        features['daily_range'] = (data['High'] - data['Low']) / data['Close']
        features['weekly_range'] = features['daily_range'].rolling(window=5).mean()
        
        # Volume features
        if 'Volume' in data.columns:
            features['volume_ma_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
            features['volume_price_corr'] = data['Volume'].rolling(window=20).corr(data['Close'])
            
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['macd'] = self._calculate_macd(data['Close'])
        features['cci'] = self._calculate_cci(data['High'], data['Low'], data['Close'])
        features['stoch_k'], features['stoch_d'] = self._calculate_stochastic(
            data['High'],
            data['Low'],
            data['Close']
        )
        
        # Support and resistance
        features['support'], features['resistance'] = self._calculate_support_resistance(
            data['High'],
            data['Low'],
            data['Close']
        )
        
        return features.fillna(method='bfill')
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD technical indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
        
    def _calculate_cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (high + low + close) / 3
        tp_ma = tp.rolling(window=period).mean()
        tp_std = tp.rolling(window=period).std()
        return (tp - tp_ma) / (0.015 * tp_std)
        
    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d
        
    def _calculate_support_resistance(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate dynamic support and resistance levels"""
        pivot = (high + low + close) / 3
        support = pivot - (high - low)
        resistance = pivot + (high - low)
        return support, resistance
        
    def _create_deep_learning_model(
        self,
        input_shape: Tuple[int, int],
        output_size: int
    ) -> Model:
        """Create deep learning model for prediction"""
        # Price sequence input
        price_input = Input(shape=input_shape, name='price_sequence')
        x1 = LSTM(64, return_sequences=True)(price_input)
        x1 = Dropout(0.2)(x1)
        x1 = LSTM(32)(x1)
        x1 = Dropout(0.2)(x1)
        
        # Technical indicators input
        tech_input = Input(shape=(input_shape[1],), name='technical')
        x2 = Dense(32, activation='relu')(tech_input)
        x2 = Dropout(0.2)(x2)
        
        # Merge inputs
        merged = Concatenate()([x1, x2])
        
        # Output layers
        dense = Dense(32, activation='relu')(merged)
        output = Dense(output_size)(dense)
        
        model = Model(inputs=[price_input, tech_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
        
    def _prepare_sequences(
        self,
        data: pd.DataFrame,
        target_column: str,
        sequence_length: int,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for deep learning"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length - horizon + 1):
            sequences.append(data.iloc[i:i+sequence_length].values)
            targets.append(data[target_column].iloc[i+sequence_length+horizon-1])
            
        return np.array(sequences), np.array(targets)
        
    def train(self, symbols: List[str], period: str = "5y"):
        """Train predictive models"""
        all_data = []
        all_features = []
        all_targets = []
        
        for symbol in symbols:
            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Create features
            features = self._create_features(data)
            targets = data['Close'].pct_change().shift(-1).dropna()
            
            all_data.append(data)
            all_features.append(features)
            all_targets.append(targets)
            
        # Combine data
        combined_features = pd.concat(all_features, axis=0)
        combined_targets = pd.concat(all_targets, axis=0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(combined_features)
        
        # Train models for each horizon
        self.models = {}
        for horizon in self.prediction_horizons:
            # Prepare sequences
            X, y = self._prepare_sequences(
                pd.DataFrame(scaled_features),
                'Close',
                self.lookback_period,
                horizon
            )
            
            # Create and train model
            model = self._create_deep_learning_model(
                (self.lookback_period, scaled_features.shape[1]),
                1
            )
            
            # Split data for technical indicators
            X_price = X[:, :, :scaled_features.shape[1]-5]  # Price-based features
            X_tech = X[:, -1, -5:]  # Latest technical indicators
            
            # Train model
            model.fit(
                [X_price, X_tech],
                y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            self.models[horizon] = model
            
    def predict(
        self,
        symbol: str,
        horizon: int = 5
    ) -> PredictionResults:
        """Generate comprehensive predictions"""
        # Get data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="120d")  # Get enough history
        
        # Create features
        features = self._create_features(data)
        scaled_features = self.scaler.transform(features)
        
        # Prepare sequences
        X = self._prepare_sequences(
            pd.DataFrame(scaled_features),
            'Close',
            self.lookback_period,
            1
        )[0]
        
        # Split features
        X_price = X[:, :, :scaled_features.shape[1]-5]
        X_tech = X[:, -1, -5:]
        
        # Get predictions
        price_pred = self.models[horizon].predict([X_price[-1:], X_tech[-1:]])[0][0]
        
        # Calculate confidence intervals
        predictions = []
        for _ in range(100):
            pred = self.models[horizon].predict(
                [X_price[-1:], X_tech[-1:]],
                verbose=0
            )[0][0]
            predictions.append(pred)
            
        confidence_lower = np.percentile(predictions, 5)
        confidence_upper = np.percentile(predictions, 95)
        
        # Calculate trend probability
        trend_prob = len([p for p in predictions if p > 0]) / len(predictions)
        
        # Calculate volatility forecast
        garch = arch_model(data['Close'].pct_change().dropna(), vol='Garch', p=1, q=1)
        garch_fit = garch.fit(disp='off')
        volatility_forecast = np.sqrt(garch_fit.forecast().variance.values[-1])
        
        # Detect market regime
        returns = data['Close'].pct_change().dropna()
        regime_model = sm.tsa.MarkovRegression(returns, k_regimes=3)
        regime_fit = regime_model.fit()
        regime_probs = regime_fit.smoothed_marginal_probabilities
        
        # Get latest support and resistance
        support = features['support'].iloc[-1]
        resistance = features['resistance'].iloc[-1]
        
        # Generate technical signals
        technical_signals = {
            'RSI': 'Overbought' if features['rsi'].iloc[-1] > 70 else 'Oversold' if features['rsi'].iloc[-1] < 30 else 'Neutral',
            'MACD': 'Bullish' if features['macd'].iloc[-1] > 0 else 'Bearish',
            'Stochastic': 'Overbought' if features['stoch_k'].iloc[-1] > 80 else 'Oversold' if features['stoch_k'].iloc[-1] < 20 else 'Neutral'
        }
        
        # Calculate fundamental score (if available)
        fundamental_score = 0.0
        try:
            info = ticker.info
            if 'forwardPE' in info and 'priceToBook' in info:
                pe_score = 1 / info['forwardPE'] if info['forwardPE'] > 0 else 0
                pb_score = 1 / info['priceToBook'] if info['priceToBook'] > 0 else 0
                fundamental_score = (pe_score + pb_score) / 2
        except:
            pass
            
        # Calculate market sentiment
        sentiment = np.mean([
            1 if s == 'Bullish' else -1 if s == 'Bearish' else 0
            for s in technical_signals.values()
        ])
        
        return PredictionResults(
            price_prediction=price_pred,
            price_confidence=(confidence_lower, confidence_upper),
            trend_probability=trend_prob,
            volatility_forecast=volatility_forecast,
            regime_probability={
                'Bull': regime_probs[0][-1],
                'Neutral': regime_probs[1][-1],
                'Bear': regime_probs[2][-1]
            },
            support_resistance={
                'support': support,
                'resistance': resistance
            },
            technical_signals=technical_signals,
            fundamental_score=fundamental_score,
            market_sentiment=sentiment,
            prediction_horizon=horizon
        )

predictive_analytics = PredictiveAnalytics()
