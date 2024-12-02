import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import statsmodels.api as sm
from arch import arch_model
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RiskPredictor:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.lookback_period = 30
        
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features for ML models"""
        features = pd.DataFrame(index=data.index)
        
        # Rolling statistics
        features['rolling_mean'] = data['Close'].rolling(window=20).mean()
        features['rolling_std'] = data['Close'].rolling(window=20).std()
        features['rolling_skew'] = data['Close'].rolling(window=20).skew()
        features['rolling_kurt'] = data['Close'].rolling(window=20).kurt()
        
        # Price-based indicators
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['macd'] = self._calculate_macd(data['Close'])
        features['bollinger_up'], features['bollinger_down'] = self._calculate_bollinger_bands(data['Close'])
        
        # Volume indicators
        if 'Volume' in data.columns:
            features['volume_ma'] = data['Volume'].rolling(window=20).mean()
            features['volume_std'] = data['Volume'].rolling(window=20).std()
            features['volume_price_corr'] = data['Volume'].rolling(window=20).corr(data['Close'])
        
        # Returns and volatility
        features['returns'] = data['Close'].pct_change()
        features['volatility'] = features['returns'].rolling(window=20).std()
        features['realized_vol'] = features['returns'].rolling(window=20).apply(lambda x: np.sqrt((x**2).mean()) * np.sqrt(252))
        
        # Market regime indicators
        features['trend'] = self._calculate_trend(data['Close'])
        features['regime'] = self._detect_regime(features['returns'])
        
        return features.fillna(method='bfill')
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
        
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * 2)
        lower_band = ma - (std * 2)
        return upper_band, lower_band
        
    def _calculate_trend(self, prices: pd.Series) -> pd.Series:
        """Calculate trend indicator using exponential moving averages"""
        short_ma = prices.ewm(span=20, adjust=False).mean()
        long_ma = prices.ewm(span=50, adjust=False).mean()
        return (short_ma - long_ma) / long_ma
        
    def _detect_regime(self, returns: pd.Series) -> pd.Series:
        """Detect market regime using Hidden Markov Model"""
        model = sm.tsa.MarkovRegression(returns.dropna(), k_regimes=2)
        try:
            res = model.fit()
            return pd.Series(res.smoothed_marginal_probabilities[0], index=returns.index)
        except:
            return pd.Series(0, index=returns.index)
            
    def _prepare_lstm_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        data = []
        targets = []
        
        for i in range(len(features) - self.lookback_period):
            data.append(features.iloc[i:(i + self.lookback_period)].values)
            targets.append(features.iloc[i + self.lookback_period]['volatility'])
            
        return np.array(data), np.array(targets)
        
    def train_models(self, symbol: str, start_date: Optional[datetime] = None) -> Dict:
        """Train all ML models for risk prediction"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*2)
            
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date)
        
        # Create features
        features = self._create_features(data)
        features = features.dropna()
        
        # Prepare target (next day volatility)
        y = features['volatility'].shift(-1).dropna()
        X = features.iloc[:-1]
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        
        # Train Gradient Boosting
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.gb_model.fit(X, y)
        
        # Train LSTM
        X_lstm, y_lstm = self._prepare_lstm_data(features)
        X_lstm = self.feature_scaler.fit_transform(X_lstm.reshape(-1, X_lstm.shape[-1])).reshape(X_lstm.shape)
        y_lstm = self.scaler.fit_transform(y_lstm.reshape(-1, 1)).flatten()
        
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback_period, X_lstm.shape[-1])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
        
        # Calculate performance metrics
        rf_pred = self.rf_model.predict(X)
        gb_pred = self.gb_model.predict(X)
        lstm_pred = self.scaler.inverse_transform(
            self.lstm_model.predict(X_lstm).reshape(-1, 1)
        ).flatten()
        
        return {
            'rf_r2': r2_score(y, rf_pred),
            'gb_r2': r2_score(y, gb_pred),
            'lstm_r2': r2_score(y[self.lookback_period:], lstm_pred),
            'rf_rmse': np.sqrt(mean_squared_error(y, rf_pred)),
            'gb_rmse': np.sqrt(mean_squared_error(y, gb_pred)),
            'lstm_rmse': np.sqrt(mean_squared_error(y[self.lookback_period:], lstm_pred))
        }
        
    def predict_risk(self, symbol: str, days_forward: int = 5) -> Dict:
        """Predict future risk using ensemble of models"""
        # Fetch recent data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='60d')
        
        # Create features
        features = self._create_features(data)
        features = features.dropna()
        
        # Get predictions from each model
        rf_pred = self.rf_model.predict(features.iloc[-1:])
        gb_pred = self.gb_model.predict(features.iloc[-1:])
        
        # Prepare LSTM data
        X_lstm = features.iloc[-self.lookback_period:].values.reshape(1, self.lookback_period, -1)
        X_lstm = self.feature_scaler.transform(X_lstm.reshape(-1, X_lstm.shape[-1])).reshape(X_lstm.shape)
        lstm_pred = self.scaler.inverse_transform(self.lstm_model.predict(X_lstm)).flatten()
        
        # Ensemble prediction (weighted average)
        ensemble_pred = 0.4 * rf_pred + 0.3 * gb_pred + 0.3 * lstm_pred
        
        # Calculate confidence intervals
        predictions = []
        for _ in range(100):
            rf_bootstrap = self.rf_model.predict(
                features.iloc[-1:], 
                n_jobs=-1
            )
            predictions.append(rf_bootstrap[0])
            
        ci_lower = np.percentile(predictions, 5)
        ci_upper = np.percentile(predictions, 95)
        
        return {
            'predicted_volatility': float(ensemble_pred[0]),
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'model_predictions': {
                'random_forest': float(rf_pred[0]),
                'gradient_boosting': float(gb_pred[0]),
                'lstm': float(lstm_pred[0])
            }
        }

risk_predictor = RiskPredictor()
