"""Advanced attention mechanisms for deep learning models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """Initialize attention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out_linear = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention weights)
        """
        batch_size = query.size(0)
        
        # Linear transformations
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        
        # Split into heads
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            -1,
            self.d_model
        )
        
        # Final linear layer
        output = self.out_linear(context)
        
        return output, attn_weights

class TemporalAttention(nn.Module):
    """Temporal attention for time series data."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1
    ):
        """Initialize temporal attention.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(
            hidden_dim,
            num_heads,
            dropout
        )
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention weights)
        """
        # Project input
        x = self.input_projection(x)
        
        # Self-attention
        attended, weights = self.attention(x, x, x, mask)
        
        # Residual connection and layer normalization
        x = self.layer_norm(x + self.dropout(attended))
        
        return x, weights

class HierarchicalAttention(nn.Module):
    """Hierarchical attention for multi-scale time series."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_levels: int = 3,
        num_heads: int = 1,
        dropout: float = 0.1
    ):
        """Initialize hierarchical attention.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_levels: Number of hierarchical levels
            num_heads: Number of attention heads per level
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_levels = num_levels
        
        # Create attention layers for each level
        self.attention_layers = nn.ModuleList([
            TemporalAttention(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_heads,
                dropout
            )
            for i in range(num_levels)
        ])
        
        # Projections for different time scales
        self.scale_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_levels)
        ])
        
        self.output_projection = nn.Linear(
            hidden_dim * num_levels,
            hidden_dim
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention weights per level)
        """
        batch_size, seq_len, _ = x.size()
        outputs = []
        attention_weights = []
        
        # Process each hierarchical level
        for i in range(self.num_levels):
            # Adjust sequence length for current level
            scale_factor = 2 ** i
            if scale_factor > 1:
                x_scaled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale_factor,
                    stride=scale_factor
                ).transpose(1, 2)
                
                if mask is not None:
                    mask_scaled = F.avg_pool1d(
                        mask.float().unsqueeze(1),
                        kernel_size=scale_factor,
                        stride=scale_factor
                    ).squeeze(1) > 0.5
            else:
                x_scaled = x
                mask_scaled = mask
                
            # Apply attention
            attended, weights = self.attention_layers[i](
                x_scaled,
                mask_scaled
            )
            
            # Project back to original sequence length if needed
            if scale_factor > 1:
                attended = F.interpolate(
                    attended.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                
            # Apply scale-specific projection
            outputs.append(self.scale_projections[i](attended))
            attention_weights.append(weights)
            
        # Combine outputs from all levels
        combined = torch.cat(outputs, dim=-1)
        output = self.output_projection(combined)
        
        return output, attention_weights

class CrossModalAttention(nn.Module):
    """Cross-modal attention for multi-modal time series."""
    
    def __init__(
        self,
        modal_dims: dict,
        hidden_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1
    ):
        """Initialize cross-modal attention.
        
        Args:
            modal_dims: Dictionary of modal names to dimensions
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.modal_projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in modal_dims.items()
        })
        
        self.attention = MultiHeadAttention(
            hidden_dim,
            num_heads,
            dropout
        )
        
        self.modal_attention = MultiHeadAttention(
            hidden_dim,
            num_heads,
            dropout
        )
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        modal_inputs: dict,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass.
        
        Args:
            modal_inputs: Dictionary of modal inputs
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention weights per modality)
        """
        # Project each modality
        modal_features = {
            name: self.modal_projections[name](x)
            for name, x in modal_inputs.items()
        }
        
        # Cross-modal attention
        attention_weights = {}
        attended_features = []
        
        for name, features in modal_features.items():
            # Attend to all other modalities
            other_features = [
                f for n, f in modal_features.items() if n != name
            ]
            
            if other_features:
                # Concatenate other features
                context = torch.cat(other_features, dim=1)
                
                # Apply cross-modal attention
                attended, weights = self.attention(
                    features,
                    context,
                    context,
                    mask
                )
                
                attention_weights[name] = weights
                attended_features.append(attended)
                
        # Combine attended features
        combined = torch.stack(attended_features).mean(0)
        
        # Final projection
        output = self.output_projection(combined)
        output = self.layer_norm(output + self.dropout(combined))
        
        return output, attention_weights
