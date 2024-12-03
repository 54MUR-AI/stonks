"""Deep learning models for financial time series."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    early_stopping_patience: int = 10
    validation_split: float = 0.2

class TimeSeriesDataset(Dataset):
    """Dataset for financial time series data."""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 10
    ):
        """Initialize dataset.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples,)
            sequence_length: Length of input sequences
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self) -> int:
        return len(self.features) - self.sequence_length + 1
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        return x, y

class LSTM(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output

class Transformer(nn.Module):
    """Transformer model for time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """Initialize Transformer model.
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        x = self.input_proj(x)
        transformer_out = self.transformer(x)
        last_hidden = transformer_out[:, -1, :]
        output = self.fc(last_hidden)
        return output

class WaveNet(nn.Module):
    """WaveNet model for time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        residual_channels: int = 32,
        skip_channels: int = 32,
        num_blocks: int = 3,
        num_layers: int = 4
    ):
        """Initialize WaveNet model.
        
        Args:
            input_size: Number of input features
            residual_channels: Number of residual channels
            skip_channels: Number of skip channels
            num_blocks: Number of dilated convolution blocks
            num_layers: Number of layers per block
        """
        super().__init__()
        
        self.input_proj = nn.Conv1d(input_size, residual_channels, 1)
        
        self.dilated_convs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(
                    residual_channels,
                    2 * residual_channels,
                    2,
                    padding=2**(i+j*num_layers) // 2,
                    dilation=2**(i+j*num_layers)
                )
                for i in range(num_layers)
            ])
            for j in range(num_blocks)
        ])
        
        self.skip_convs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(residual_channels, skip_channels, 1)
                for _ in range(num_layers)
            ])
            for _ in range(num_blocks)
        ])
        
        self.residual_convs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(residual_channels, residual_channels, 1)
                for _ in range(num_layers)
            ])
            for _ in range(num_blocks)
        ])
        
        self.final_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, 1, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        
        skip_connections = []
        
        for i, block in enumerate(self.dilated_convs):
            for j, (dilated_conv, skip_conv, residual_conv) in enumerate(
                zip(block, self.skip_convs[i], self.residual_convs[i])
            ):
                residual = x
                x = dilated_conv(x)
                
                # Gated activation
                gate, filter = torch.chunk(x, 2, dim=1)
                x = torch.tanh(filter) * torch.sigmoid(gate)
                
                # Skip connection
                skip = skip_conv(x)
                skip_connections.append(skip)
                
                # Residual connection
                x = residual_conv(x)
                x = x + residual[:, :, -x.size(2):]
                
        x = torch.sum(torch.stack(skip_connections), dim=0)
        x = self.final_conv(x)
        return x[:, 0, -1:]

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig
) -> Dict[str, List[float]]:
    """Train deep learning model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        
    Returns:
        Dictionary of training history
    """
    model = model.to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.device)
                batch_y = batch_y.to(config.device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
    return history
