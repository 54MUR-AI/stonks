import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

class MarketAnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.robust_covariance = EllipticEnvelope(
            contamination=0.1,
            random_state=42
        )
        self.one_class_svm = OneClassSVM(
            nu=0.1,
            kernel="rbf",
            gamma='scale'
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Explain 95% of variance
        self.lstm_autoencoder = None
        self.lookback_period = 30
        self.threshold = 3.0  # Number of std devs for anomaly
        
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for anomaly detection"""
        features = pd.DataFrame(index=data.index)
        
        # Returns and volatility
        features['returns'] = data['Close'].pct_change()
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        # Volume features
        if 'Volume' in data.columns:
            features['volume_change'] = data['Volume'].pct_change()
            features['volume_ma_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
            
        # Price features
        features['price_ma_ratio'] = data['Close'] / data['Close'].rolling(window=20).mean()
        features['high_low_ratio'] = data['High'] / data['Low']
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['macd'] = self._calculate_macd(data['Close'])
        
        return features.dropna()
        
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
        
    def _create_lstm_autoencoder(self, input_shape: Tuple[int, int]) -> Model:
        """Create LSTM autoencoder model"""
        # Encoder
        inputs = Input(shape=input_shape)
        encoded = LSTM(64, return_sequences=True)(inputs)
        encoded = Dropout(0.2)(encoded)
        encoded = LSTM(32, return_sequences=False)(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16)(encoded)
        
        # Decoder
        decoded = Dense(32)(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(64)(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(input_shape[1])(decoded)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
        
    def _prepare_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: int
    ) -> np.ndarray:
        """Prepare sequences for LSTM"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data.iloc[i:i+sequence_length].values)
        return np.array(sequences)
        
    def train(self, symbols: List[str], period: str = "2y"):
        """Train anomaly detection models"""
        all_features = []
        
        for symbol in symbols:
            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Create features
            features = self._create_features(data)
            all_features.append(features)
            
        # Combine features
        combined_features = pd.concat(all_features, axis=1)
        scaled_features = self.scaler.fit_transform(combined_features)
        
        # Train traditional models
        self.isolation_forest.fit(scaled_features)
        self.robust_covariance.fit(scaled_features)
        self.one_class_svm.fit(scaled_features)
        
        # Dimensionality reduction
        self.pca.fit(scaled_features)
        
        # Prepare sequences for LSTM
        sequences = self._prepare_sequences(
            pd.DataFrame(scaled_features),
            self.lookback_period
        )
        
        # Train LSTM autoencoder
        self.lstm_autoencoder = self._create_lstm_autoencoder(
            (self.lookback_period, scaled_features.shape[1])
        )
        self.lstm_autoencoder.fit(
            sequences,
            sequences,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
    def detect_anomalies(
        self,
        symbol: str,
        period: str = "60d"
    ) -> Dict[str, List[datetime]]:
        """Detect anomalies using multiple methods"""
        # Get data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        # Create and scale features
        features = self._create_features(data)
        scaled_features = self.scaler.transform(features)
        
        # Traditional model predictions
        if_predictions = self.isolation_forest.predict(scaled_features)
        rc_predictions = self.robust_covariance.predict(scaled_features)
        svm_predictions = self.one_class_svm.predict(scaled_features)
        
        # PCA reconstruction error
        pca_transformed = self.pca.transform(scaled_features)
        pca_reconstructed = self.pca.inverse_transform(pca_transformed)
        pca_error = np.mean((scaled_features - pca_reconstructed) ** 2, axis=1)
        pca_threshold = np.mean(pca_error) + self.threshold * np.std(pca_error)
        pca_anomalies = pca_error > pca_threshold
        
        # LSTM autoencoder reconstruction error
        sequences = self._prepare_sequences(
            pd.DataFrame(scaled_features),
            self.lookback_period
        )
        reconstructed = self.lstm_autoencoder.predict(sequences)
        reconstruction_error = np.mean(
            (sequences - reconstructed.reshape(sequences.shape)) ** 2,
            axis=(1,2)
        )
        lstm_threshold = np.mean(reconstruction_error) + self.threshold * np.std(reconstruction_error)
        lstm_anomalies = reconstruction_error > lstm_threshold
        
        # Combine results
        anomaly_dates = {
            'isolation_forest': data.index[if_predictions == -1].tolist(),
            'robust_covariance': data.index[rc_predictions == -1].tolist(),
            'one_class_svm': data.index[svm_predictions == -1].tolist(),
            'pca': data.index[pca_anomalies].tolist(),
            'lstm': data.index[self.lookback_period-1:][lstm_anomalies].tolist()
        }
        
        # Calculate anomaly probabilities
        num_methods = 5
        anomaly_counts = np.zeros(len(data))
        
        for dates in anomaly_dates.values():
            for date in dates:
                idx = data.index.get_loc(date)
                anomaly_counts[idx] += 1
                
        anomaly_probabilities = anomaly_counts / num_methods
        
        # Add high probability anomalies
        high_prob_threshold = 0.6  # At least 3 methods agree
        anomaly_dates['high_probability'] = data.index[
            anomaly_probabilities >= high_prob_threshold
        ].tolist()
        
        return anomaly_dates
        
    def analyze_anomaly(
        self,
        symbol: str,
        anomaly_date: datetime
    ) -> Dict[str, float]:
        """Analyze the characteristics of an anomaly"""
        # Get data around anomaly
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=anomaly_date - timedelta(days=30),
            end=anomaly_date + timedelta(days=1)
        )
        
        # Create features
        features = self._create_features(data)
        anomaly_features = features.loc[anomaly_date]
        
        # Calculate z-scores
        z_scores = (anomaly_features - features.mean()) / features.std()
        
        # Identify main contributors
        significant_features = z_scores[abs(z_scores) > 2].to_dict()
        
        # Additional analysis
        analysis = {
            'z_scores': significant_features,
            'price_change': data['Close'].pct_change().loc[anomaly_date],
            'volume_ratio': data['Volume'].loc[anomaly_date] / data['Volume'].mean(),
            'volatility': features['volatility'].loc[anomaly_date],
            'rsi': features['rsi'].loc[anomaly_date],
            'macd': features['macd'].loc[anomaly_date]
        }
        
        return analysis

anomaly_detector = MarketAnomalyDetector()
