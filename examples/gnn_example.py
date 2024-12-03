"""Example usage of Graph Neural Networks for market analysis."""

import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pathlib import Path

from backend.services.ml.deep_learning.gnn import (
    GNNConfig,
    MarketGNN,
    TemporalGNN,
    HeterogeneousGNN
)

def generate_market_graph(
    num_assets: int,
    num_features: int,
    correlation_threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic market graph.
    
    Args:
        num_assets: Number of assets
        num_features: Number of features per asset
        correlation_threshold: Threshold for edge creation
        
    Returns:
        Tuple of (node features, edge indices, edge attributes)
    """
    # Generate random features
    features = np.random.randn(num_assets, num_features)
    
    # Calculate correlations
    correlations = np.corrcoef(features)
    
    # Create edges based on correlation threshold
    edges = []
    edge_attrs = []
    
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            if abs(correlations[i, j]) > correlation_threshold:
                edges.append([i, j])
                edges.append([j, i])  # Add both directions
                edge_attrs.extend([correlations[i, j]] * 2)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
    
    return (
        torch.FloatTensor(features),
        edge_index,
        edge_attr
    )

def generate_temporal_market_data(
    num_assets: int,
    num_features: int,
    sequence_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic temporal market data.
    
    Args:
        num_assets: Number of assets
        num_features: Number of features per asset
        sequence_length: Length of time sequence
        
    Returns:
        Tuple of (feature sequences, edge index sequences, edge attribute sequences)
    """
    feature_seq = []
    edge_index_seq = []
    edge_attr_seq = []
    
    for _ in range(sequence_length):
        features, edge_index, edge_attr = generate_market_graph(
            num_assets,
            num_features
        )
        feature_seq.append(features)
        edge_index_seq.append(edge_index)
        edge_attr_seq.append(edge_attr)
    
    return (
        torch.stack(feature_seq),
        torch.stack(edge_index_seq),
        torch.stack(edge_attr_seq)
    )

def generate_heterogeneous_market_data(
    config: Dict[str, int]
) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
    """Generate synthetic heterogeneous market data.
    
    Args:
        config: Dictionary of node types to number of nodes
        
    Returns:
        Tuple of (node features, edge indices)
    """
    node_features = {}
    edge_indices = {}
    
    # Generate node features
    for node_type, num_nodes in config['num_nodes'].items():
        node_features[node_type] = torch.randn(
            num_nodes,
            config['feature_dims'][node_type]
        )
    
    # Generate edges
    for src_type in config['num_nodes'].keys():
        for tgt_type in config['num_nodes'].keys():
            num_edges = min(
                config['num_nodes'][src_type],
                config['num_nodes'][tgt_type]
            )
            
            # Create random edges
            src_nodes = torch.randint(
                0,
                config['num_nodes'][src_type],
                (num_edges,)
            )
            tgt_nodes = torch.randint(
                0,
                config['num_nodes'][tgt_type],
                (num_edges,)
            )
            
            edge_indices[(src_type, 'relates', tgt_type)] = torch.stack(
                [src_nodes, tgt_nodes]
            )
    
    return node_features, edge_indices

def visualize_market_graph(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    save_path: str
):
    """Visualize market graph.
    
    Args:
        features: Node features
        edge_index: Edge indices
        edge_attr: Edge attributes
        save_path: Path to save visualization
    """
    # Create networkx graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(features.size(0)):
        G.add_node(i)
    
    # Add edges
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[:, i]
        weight = edge_attr[i].item()
        G.add_edge(src.item(), tgt.item(), weight=weight)
    
    # Draw graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color='lightblue',
        node_size=500
    )
    
    # Draw edges with weights as colors
    edges = nx.draw_networkx_edges(
        G,
        pos,
        edge_color=[G[u][v]['weight'] for u, v in G.edges()],
        edge_cmap=plt.cm.RdYlBu,
        width=2
    )
    
    # Add colorbar
    plt.colorbar(edges)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def main():
    # Parameters
    num_assets = 20
    num_features = 10
    sequence_length = 30
    hidden_dim = 64
    output_dim = 32
    
    # Create save directory
    save_dir = Path('visualizations')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Test MarketGNN
    print("Testing MarketGNN...")
    features, edge_index, edge_attr = generate_market_graph(
        num_assets,
        num_features
    )
    
    config = GNNConfig(
        input_dim=num_features,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        gnn_type='gat',
        edge_dim=1
    )
    
    market_gnn = MarketGNN(config)
    market_output = market_gnn(features, edge_index, edge_attr)
    
    visualize_market_graph(
        features,
        edge_index,
        edge_attr,
        save_dir / 'market_graph.png'
    )
    
    # Test TemporalGNN
    print("Testing TemporalGNN...")
    feature_seq, edge_index_seq, edge_attr_seq = generate_temporal_market_data(
        num_assets,
        num_features,
        sequence_length
    )
    
    temporal_gnn = TemporalGNN(
        config,
        temporal_hidden_dim=hidden_dim,
        sequence_length=sequence_length
    )
    
    temporal_output = temporal_gnn(
        feature_seq.unsqueeze(0),
        edge_index_seq.unsqueeze(0),
        edge_attr_seq.unsqueeze(0)
    )
    
    # Test HeterogeneousGNN
    print("Testing HeterogeneousGNN...")
    hetero_config = {
        'num_nodes': {
            'stock': 10,
            'sector': 5,
            'market': 2
        },
        'feature_dims': {
            'stock': num_features,
            'sector': num_features // 2,
            'market': num_features // 4
        }
    }
    
    node_features, edge_indices = generate_heterogeneous_market_data(
        hetero_config
    )
    
    hetero_gnn = HeterogeneousGNN(
        node_types=hetero_config['feature_dims'],
        edge_types=[
            ('stock', 'belongs_to', 'sector'),
            ('sector', 'part_of', 'market')
        ],
        config=config
    )
    
    hetero_output = hetero_gnn(node_features, edge_indices)
    
    # Print model statistics
    print("\nModel Statistics:")
    print(f"MarketGNN output shape: {market_output.shape}")
    print(f"TemporalGNN output shape: {temporal_output.shape}")
    print("HeterogeneousGNN output shapes:")
    for node_type, output in hetero_output.items():
        print(f"  {node_type}: {output.shape}")

if __name__ == '__main__':
    main()
