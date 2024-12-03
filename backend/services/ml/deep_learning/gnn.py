"""Graph Neural Networks for financial market analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

@dataclass
class GNNConfig:
    """Configuration for graph neural networks."""
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int = 3
    dropout: float = 0.1
    gnn_type: str = 'gcn'  # 'gcn', 'gat', or 'sage'
    heads: int = 4  # For GAT
    edge_dim: Optional[int] = None  # For edge features

class MarketGNN(nn.Module):
    """Graph Neural Network for market analysis."""
    
    def __init__(self, config: GNNConfig):
        """Initialize GNN.
        
        Args:
            config: GNN configuration
        """
        super().__init__()
        
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(config.num_layers):
            in_channels = config.hidden_dim
            out_channels = config.hidden_dim
            
            if config.gnn_type == 'gcn':
                layer = GCNConv(in_channels, out_channels)
            elif config.gnn_type == 'gat':
                layer = GATConv(
                    in_channels,
                    out_channels // config.heads,
                    heads=config.heads,
                    dropout=config.dropout,
                    edge_dim=config.edge_dim
                )
            elif config.gnn_type == 'sage':
                layer = SAGEConv(
                    in_channels,
                    out_channels,
                    normalize=True
                )
            else:
                raise ValueError(f"Unknown GNN type: {config.gnn_type}")
                
            self.gnn_layers.append(layer)
            
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            edge_attr: Optional edge features
            batch: Optional batch vector
            
        Returns:
            Node embeddings
        """
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers
        for layer in self.gnn_layers:
            if self.config.gnn_type == 'gat' and edge_attr is not None:
                # GAT with edge features
                x_new = layer(x, edge_index, edge_attr)
            else:
                # GCN or SAGE
                x_new = layer(x, edge_index)
                
            # Residual connection
            x = x + self.dropout(x_new)
            x = self.layer_norm(x)
            x = F.relu(x)
            
        # Output projection
        x = self.output_proj(x)
        
        return x

class TemporalGNN(nn.Module):
    """Temporal Graph Neural Network for dynamic market analysis."""
    
    def __init__(
        self,
        config: GNNConfig,
        temporal_hidden_dim: int,
        sequence_length: int
    ):
        """Initialize temporal GNN.
        
        Args:
            config: GNN configuration
            temporal_hidden_dim: Hidden dimension for temporal processing
            sequence_length: Length of time sequences
        """
        super().__init__()
        
        self.gnn = MarketGNN(config)
        
        # Temporal processing
        self.temporal_conv = nn.Conv1d(
            config.output_dim,
            temporal_hidden_dim,
            kernel_size=3,
            padding=1
        )
        
        self.temporal_gru = nn.GRU(
            temporal_hidden_dim,
            temporal_hidden_dim,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(
            temporal_hidden_dim,
            config.output_dim
        )
        
    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index_seq: torch.Tensor,
        edge_attr_seq: Optional[torch.Tensor] = None,
        batch_seq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x_seq: Sequence of node features
            edge_index_seq: Sequence of graph connectivity
            edge_attr_seq: Optional sequence of edge features
            batch_seq: Optional sequence of batch vectors
            
        Returns:
            Temporal node embeddings
        """
        batch_size, seq_len, num_nodes, feat_dim = x_seq.size()
        
        # Process each timestep with GNN
        gnn_outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t]
            edge_index_t = edge_index_seq[:, t]
            edge_attr_t = (
                edge_attr_seq[:, t] if edge_attr_seq is not None else None
            )
            batch_t = batch_seq[:, t] if batch_seq is not None else None
            
            # GNN forward pass
            out_t = self.gnn(x_t, edge_index_t, edge_attr_t, batch_t)
            gnn_outputs.append(out_t)
            
        # Stack GNN outputs
        gnn_outputs = torch.stack(gnn_outputs, dim=1)
        
        # Temporal convolution
        conv_out = self.temporal_conv(
            gnn_outputs.transpose(1, 2)
        ).transpose(1, 2)
        
        # GRU processing
        gru_out, _ = self.temporal_gru(conv_out)
        
        # Final output
        output = self.output_layer(gru_out)
        
        return output

class HeterogeneousGNN(nn.Module):
    """Heterogeneous GNN for multi-asset market analysis."""
    
    def __init__(
        self,
        node_types: Dict[str, int],
        edge_types: List[Tuple[str, str, str]],
        config: GNNConfig
    ):
        """Initialize heterogeneous GNN.
        
        Args:
            node_types: Dictionary of node types to feature dimensions
            edge_types: List of edge types (source, relation, target)
            config: GNN configuration
        """
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        
        # Node type specific input projections
        self.node_projections = nn.ModuleDict({
            ntype: nn.Linear(dim, config.hidden_dim)
            for ntype, dim in node_types.items()
        })
        
        # Edge type specific GNN layers
        self.gnn_layers = nn.ModuleDict({
            f"{src}_to_{tgt}_{rel}": MarketGNN(config)
            for src, rel, tgt in edge_types
        })
        
        # Node type specific output projections
        self.output_projections = nn.ModuleDict({
            ntype: nn.Linear(config.hidden_dim, config.output_dim)
            for ntype in node_types.keys()
        })
        
    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attrs: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            node_features: Dictionary of node features per type
            edge_indices: Dictionary of edge indices per type
            edge_attrs: Optional dictionary of edge features per type
            
        Returns:
            Dictionary of node embeddings per type
        """
        # Project node features
        hidden_states = {
            ntype: self.node_projections[ntype](features)
            for ntype, features in node_features.items()
        }
        
        # Process each edge type
        for src, rel, tgt in self.edge_types:
            edge_key = (src, rel, tgt)
            edge_index = edge_indices[edge_key]
            edge_attr = edge_attrs[edge_key] if edge_attrs else None
            
            # GNN forward pass
            out = self.gnn_layers[f"{src}_to_{tgt}_{rel}"](
                hidden_states[src],
                edge_index,
                edge_attr
            )
            
            # Update target node representations
            hidden_states[tgt] = out
            
        # Project to output space
        outputs = {
            ntype: self.output_projections[ntype](hidden)
            for ntype, hidden in hidden_states.items()
        }
        
        return outputs
