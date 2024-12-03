"""Example usage of advanced attention mechanisms."""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from pathlib import Path

from backend.services.ml.deep_learning.attention import (
    MultiHeadAttention,
    TemporalAttention,
    HierarchicalAttention,
    CrossModalAttention
)

def generate_multimodal_data(
    n_samples: int,
    seq_length: int
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Generate synthetic multimodal financial data.
    
    Args:
        n_samples: Number of samples
        seq_length: Sequence length
        
    Returns:
        Tuple of (modal_data, targets)
    """
    # Generate price data
    t = np.linspace(0, 100, seq_length)
    prices = np.zeros((n_samples, seq_length))
    
    for i in range(n_samples):
        # Generate trend
        trend = 0.1 * t + np.random.normal(0, 1)
        # Add seasonality
        seasonality = 2 * np.sin(2 * np.pi * t / 20)
        # Add noise
        noise = np.random.normal(0, 0.5, seq_length)
        
        prices[i] = trend + seasonality + noise
    
    # Generate volume data
    volumes = np.exp(prices) * np.random.lognormal(0, 0.5, (n_samples, seq_length))
    
    # Generate sentiment data (simplified)
    sentiments = np.random.randn(n_samples, seq_length, 3)  # 3 sentiment features
    
    # Generate technical indicators
    technicals = np.zeros((n_samples, seq_length, 5))  # 5 technical indicators
    
    # Simple Moving Average
    for i in range(n_samples):
        technicals[i, :, 0] = np.convolve(prices[i], np.ones(5)/5, mode='same')
        technicals[i, :, 1] = np.convolve(prices[i], np.ones(20)/20, mode='same')
        technicals[i, :, 2] = (prices[i] - np.min(prices[i])) / (np.max(prices[i]) - np.min(prices[i]))
        technicals[i, :, 3] = np.gradient(prices[i])
        technicals[i, :, 4] = np.std([prices[i][max(j-5, 0):j+1] for j in range(seq_length)], axis=1)
    
    # Create targets (next day returns)
    targets = np.diff(prices, axis=1)
    targets = np.concatenate([np.zeros((n_samples, 1)), targets], axis=1)
    
    # Convert to tensors
    modal_data = {
        'price': torch.FloatTensor(prices.reshape(n_samples, seq_length, 1)),
        'volume': torch.FloatTensor(volumes.reshape(n_samples, seq_length, 1)),
        'sentiment': torch.FloatTensor(sentiments),
        'technical': torch.FloatTensor(technicals)
    }
    
    return modal_data, torch.FloatTensor(targets)

def visualize_attention(
    attention_weights: torch.Tensor,
    sequence: torch.Tensor,
    save_path: str,
    modality: str = 'price'
):
    """Visualize attention weights.
    
    Args:
        attention_weights: Attention weight tensor
        sequence: Input sequence
        save_path: Path to save visualization
        modality: Name of modality
    """
    plt.figure(figsize=(12, 6))
    
    # Plot sequence
    plt.subplot(2, 1, 1)
    plt.plot(sequence.numpy())
    plt.title(f'{modality} Sequence')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # Plot attention weights
    plt.subplot(2, 1, 2)
    plt.imshow(
        attention_weights.numpy(),
        aspect='auto',
        cmap='viridis'
    )
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Parameters
    n_samples = 1000
    seq_length = 100
    hidden_dim = 64
    num_heads = 4
    
    # Generate data
    modal_data, targets = generate_multimodal_data(n_samples, seq_length)
    
    # Create save directory
    save_dir = Path('visualizations')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Test MultiHeadAttention
    print("Testing MultiHeadAttention...")
    mha = MultiHeadAttention(hidden_dim, num_heads)
    price_projected = torch.randn(32, seq_length, hidden_dim)  # Simulated projected prices
    mha_output, mha_weights = mha(price_projected, price_projected, price_projected)
    
    visualize_attention(
        mha_weights[0, 0].detach(),  # First head, first batch
        modal_data['price'][0, :, 0],
        save_dir / 'multihead_attention.png'
    )
    
    # Test TemporalAttention
    print("Testing TemporalAttention...")
    temporal_attn = TemporalAttention(1, hidden_dim, num_heads)
    temporal_output, temporal_weights = temporal_attn(modal_data['price'][:32])
    
    visualize_attention(
        temporal_weights[0, 0].detach(),
        modal_data['price'][0, :, 0],
        save_dir / 'temporal_attention.png',
        'Price (Temporal)'
    )
    
    # Test HierarchicalAttention
    print("Testing HierarchicalAttention...")
    hierarchical_attn = HierarchicalAttention(1, hidden_dim, num_levels=3)
    hierarchical_output, hierarchical_weights = hierarchical_attn(
        modal_data['price'][:32]
    )
    
    # Visualize each level
    for level, weights in enumerate(hierarchical_weights):
        visualize_attention(
            weights[0, 0].detach(),
            modal_data['price'][0, :, 0],
            save_dir / f'hierarchical_attention_level_{level}.png',
            f'Price (Level {level})'
        )
    
    # Test CrossModalAttention
    print("Testing CrossModalAttention...")
    modal_dims = {
        'price': 1,
        'volume': 1,
        'sentiment': 3,
        'technical': 5
    }
    
    cross_modal_attn = CrossModalAttention(modal_dims, hidden_dim, num_heads)
    cross_modal_output, cross_modal_weights = cross_modal_attn(
        {k: v[:32] for k, v in modal_data.items()}
    )
    
    # Visualize cross-modal attention for each modality
    for modality, weights in cross_modal_weights.items():
        visualize_attention(
            weights[0, 0].detach(),
            modal_data[modality][0, :, 0] if modality in ['price', 'volume']
            else modal_data[modality][0, :, 0].mean(dim=-1),
            save_dir / f'cross_modal_attention_{modality}.png',
            modality
        )
    
    print(f"Visualizations saved to {save_dir}")
    
    # Print model statistics
    print("\nModel Statistics:")
    print(f"MultiHeadAttention output shape: {mha_output.shape}")
    print(f"TemporalAttention output shape: {temporal_output.shape}")
    print(f"HierarchicalAttention output shape: {hierarchical_output.shape}")
    print(f"CrossModalAttention output shape: {cross_modal_output.shape}")

if __name__ == '__main__':
    main()
