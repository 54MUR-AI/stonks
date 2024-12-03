"""Example usage of deep learning components."""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split

from backend.services.ml.deep_learning.models import (
    LSTM,
    Transformer,
    WaveNet,
    TimeSeriesDataset,
    TrainingConfig
)
from backend.services.ml.deep_learning.transfer import (
    TransferLearner,
    TransferConfig
)
from backend.services.ml.deep_learning.optimization import (
    ModelOptimizer,
    OptimizationConfig
)

def prepare_data(data_path: str, sequence_length: int = 10):
    """Prepare data for deep learning models.
    
    Args:
        data_path: Path to data file
        sequence_length: Length of input sequences
        
    Returns:
        Training and validation datasets
    """
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Extract features and targets
    features = df.drop(['target'], axis=1).values
    targets = df['target'].values
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create dataset
    dataset = TimeSeriesDataset(
        features_scaled,
        targets,
        sequence_length=sequence_length
    )
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )
    
    return train_dataset, val_dataset

def main():
    # Configuration
    input_size = 10  # Number of features
    sequence_length = 10
    
    # Create model configurations
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100
    )
    
    transfer_config = TransferConfig(
        freeze_layers=['lstm.0'],
        fine_tune_layers=['lstm.1', 'fc'],
        learning_rates={
            'lstm.1': 0.0001,
            'fc': 0.001
        }
    )
    
    optimization_config = OptimizationConfig(
        use_mixed_precision=True,
        use_multi_gpu=True,
        gradient_clipping=1.0
    )
    
    # Create models
    lstm_model = LSTM(input_size=input_size)
    transformer_model = Transformer(input_size=input_size)
    wavenet_model = WaveNet(input_size=input_size)
    
    # Prepare data
    train_dataset, val_dataset = prepare_data(
        'path/to/your/data.csv',
        sequence_length
    )
    
    # Setup optimization
    optimizer = ModelOptimizer(optimization_config)
    
    # Create data loaders
    train_loader = optimizer.create_data_loader(train_dataset)
    val_loader = optimizer.create_data_loader(val_dataset, shuffle=False)
    
    # Optimize model
    lstm_model = optimizer.optimize_model(lstm_model)
    
    # Train model
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(training_config.num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        
        # Training
        lstm_model.train()
        for batch in train_loader:
            loss = optimizer.training_step(
                lstm_model,
                batch,
                criterion,
                optimizer
            )
            train_loss += loss
            
        train_loss /= len(train_loader)
        
        # Validation
        lstm_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(optimizer.device)
                targets = targets.to(optimizer.device)
                
                outputs = lstm_model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch + 1}: '
              f'Train Loss = {train_loss:.4f}, '
              f'Val Loss = {val_loss:.4f}')
    
    # Save model
    torch.save(lstm_model.state_dict(), 'lstm_model.pth')
    
    # Transfer learning example
    transfer_learner = TransferLearner()
    
    # Load pre-trained model
    pretrained_model = transfer_learner.load_pretrained(
        'lstm_model.pth',
        'lstm',
        {'input_size': input_size}
    )
    
    # Create transfer model
    transfer_model = transfer_learner.create_transfer_model(
        pretrained_model,
        transfer_config
    )
    
    # Get trainable parameters
    trainable_params = transfer_learner.get_trainable_params(transfer_model)
    print(f'Trainable parameters: {trainable_params}')
    
    # Profile model
    profile_results = optimizer.profile_model(
        lstm_model,
        (sequence_length, input_size)
    )
    print('Profiling results:', profile_results)
    
    # Optimize for inference
    inference_model = optimizer.optimize_for_inference(lstm_model)
    
if __name__ == '__main__':
    main()
