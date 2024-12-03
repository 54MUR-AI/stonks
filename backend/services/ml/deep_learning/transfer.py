"""Transfer learning utilities for deep learning models."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import copy

from .models import LSTM, Transformer, WaveNet

@dataclass
class TransferConfig:
    """Configuration for transfer learning."""
    freeze_layers: List[str]
    fine_tune_layers: List[str]
    learning_rates: Dict[str, float]
    train_from_scratch: bool = False

class ModelAdapter(nn.Module):
    """Adapter for transfer learning."""
    
    def __init__(
        self,
        base_model: nn.Module,
        input_size: int,
        output_size: int,
        adapter_size: int = 64
    ):
        """Initialize adapter.
        
        Args:
            base_model: Pre-trained model
            input_size: Input dimension
            output_size: Output dimension
            adapter_size: Adapter hidden dimension
        """
        super().__init__()
        
        self.base_model = base_model
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Add adapter layers
        self.input_adapter = nn.Sequential(
            nn.Linear(input_size, adapter_size),
            nn.ReLU(),
            nn.Linear(adapter_size, input_size)
        )
        
        self.output_adapter = nn.Sequential(
            nn.Linear(1, adapter_size),  # Base model output size is 1
            nn.ReLU(),
            nn.Linear(adapter_size, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.input_adapter(x)
        x = self.base_model(x)
        x = self.output_adapter(x)
        return x

class TransferLearner:
    """Transfer learning utilities."""
    
    @staticmethod
    def load_pretrained(
        model_path: str,
        model_type: str,
        config: Dict
    ) -> nn.Module:
        """Load pre-trained model.
        
        Args:
            model_path: Path to model weights
            model_type: Type of model ('lstm', 'transformer', 'wavenet')
            config: Model configuration
            
        Returns:
            Pre-trained model
        """
        # Create model
        if model_type == 'lstm':
            model = LSTM(**config)
        elif model_type == 'transformer':
            model = Transformer(**config)
        elif model_type == 'wavenet':
            model = WaveNet(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Load weights
        model.load_state_dict(torch.load(model_path))
        return model
    
    @staticmethod
    def create_transfer_model(
        base_model: nn.Module,
        config: TransferConfig,
        num_classes: Optional[int] = None
    ) -> nn.Module:
        """Create model for transfer learning.
        
        Args:
            base_model: Pre-trained model
            config: Transfer learning configuration
            num_classes: Number of output classes for classification
            
        Returns:
            Model ready for transfer learning
        """
        model = copy.deepcopy(base_model)
        
        # Freeze specified layers
        for name, param in model.named_parameters():
            if any(layer in name for layer in config.freeze_layers):
                param.requires_grad = False
            elif any(layer in name for layer in config.fine_tune_layers):
                param.requires_grad = True
                
        # Replace final layer for classification if needed
        if num_classes is not None:
            if hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
                
        return model
    
    @staticmethod
    def get_layer_parameters(
        model: nn.Module,
        config: TransferConfig
    ) -> List[Dict]:
        """Get parameters for different learning rates.
        
        Args:
            model: Model for transfer learning
            config: Transfer learning configuration
            
        Returns:
            List of parameter groups with learning rates
        """
        param_groups = []
        
        # Group parameters by layer type
        for name, params in model.named_parameters():
            if not params.requires_grad:
                continue
                
            # Find matching layer in config
            lr = None
            for layer, learning_rate in config.learning_rates.items():
                if layer in name:
                    lr = learning_rate
                    break
                    
            if lr is not None:
                param_groups.append({
                    'params': params,
                    'lr': lr
                })
            
        return param_groups
    
    @staticmethod
    def create_adapter_model(
        base_model: nn.Module,
        input_size: int,
        output_size: int
    ) -> ModelAdapter:
        """Create adapter model for transfer learning.
        
        Args:
            base_model: Pre-trained model
            input_size: Input dimension
            output_size: Output dimension
            
        Returns:
            Adapter model
        """
        return ModelAdapter(base_model, input_size, output_size)
    
    @staticmethod
    def unfreeze_layers(
        model: nn.Module,
        layers: List[str]
    ):
        """Unfreeze specific layers for fine-tuning.
        
        Args:
            model: Model to modify
            layers: List of layer names to unfreeze
        """
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = True
                
    @staticmethod
    def get_trainable_params(model: nn.Module) -> int:
        """Get number of trainable parameters.
        
        Args:
            model: Model to analyze
            
        Returns:
            Number of trainable parameters
        """
        return sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
