"""Example usage of distributed training components."""

import torch
import numpy as np
from pathlib import Path
import argparse

from backend.services.ml.deep_learning.models import (
    LSTM,
    TrainingConfig
)
from backend.services.ml.distributed.trainer import (
    DistributedTrainer,
    DistributedConfig
)
from backend.services.ml.distributed.data import (
    StreamingDataset,
    ParallelDataLoader,
    DataConfig
)
from backend.services.ml.monitoring.metrics import (
    MetricConfig,
    MetricsTracker
)

def generate_sample_data(
    n_samples: int,
    n_features: int,
    sequence_length: int
) -> np.ndarray:
    """Generate sample time series data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        sequence_length: Length of sequences
        
    Returns:
        Generated data array
    """
    # Generate sine waves with different frequencies
    t = np.linspace(0, 100, n_samples)
    features = []
    
    for i in range(n_features):
        freq = 0.1 * (i + 1)
        wave = np.sin(2 * np.pi * freq * t)
        features.append(wave)
        
    # Add some noise
    features = np.array(features).T + np.random.normal(0, 0.1, (n_samples, n_features))
    
    # Create sequences
    X = np.array([
        features[i:i+sequence_length]
        for i in range(n_samples - sequence_length)
    ])
    
    # Target is the next value of the first feature
    y = features[sequence_length:, 0]
    
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--gpus_per_node', type=int, default=1)
    args = parser.parse_args()
    
    # Configuration
    n_features = 10
    sequence_length = 20
    n_samples = 10000
    
    # Generate data
    X, y = generate_sample_data(n_samples, n_features, sequence_length)
    
    # Create dataset
    dataset = StreamingDataset(
        pd.DataFrame(
            np.concatenate([X.reshape(X.shape[0], -1), y.reshape(-1, 1)], axis=1),
            columns=[f'feature_{i}' for i in range(X.shape[1] * X.shape[2])] + ['target']
        ),
        cache_size=1000
    )
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    # Create model
    model = LSTM(
        input_size=n_features,
        hidden_size=64,
        num_layers=2
    )
    
    # Setup distributed training
    distributed_config = DistributedConfig(
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        node_rank=args.local_rank
    )
    
    # Setup metrics tracking
    metrics_config = MetricConfig(
        log_dir='logs',
        model_name='LSTM',
        experiment_name='distributed_training',
        metrics_to_track=['loss', 'mae'],
        save_frequency=10,
        plot_metrics=True
    )
    
    # Create trainer
    trainer = DistributedTrainer(
        model,
        distributed_config,
        metrics_config
    )
    
    # Setup data loading
    data_config = DataConfig(
        batch_size=32,
        num_workers=4,
        prefetch_factor=2
    )
    
    loader = ParallelDataLoader(
        data_config,
        distributed=True,
        world_size=distributed_config.num_nodes * distributed_config.gpus_per_node,
        rank=args.local_rank
    )
    
    train_loader = loader.create_loader(train_dataset)
    val_loader = loader.create_loader(val_dataset, shuffle=False)
    
    # Training function
    def train(rank, world_size, train_dataset, val_dataset):
        trainer.train(
            rank=rank,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            criterion=torch.nn.MSELoss(),
            optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.001),
            batch_size=32,
            epochs=10,
            save_dir='checkpoints'
        )
    
    # Run distributed training
    if args.local_rank == 0:
        print("Starting distributed training...")
        
    trainer.run_distributed(
        train,
        world_size=distributed_config.num_nodes * distributed_config.gpus_per_node,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
if __name__ == '__main__':
    main()
