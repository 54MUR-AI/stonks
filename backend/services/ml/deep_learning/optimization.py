"""GPU acceleration and optimization utilities for deep learning models."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    use_mixed_precision: bool = True
    use_multi_gpu: bool = False
    gradient_clipping: float = 1.0
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

class ModelOptimizer:
    """Model optimization utilities."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.scaler = GradScaler() if config.use_mixed_precision else None
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        """Setup compute device.
        
        Returns:
            Torch device
        """
        if not torch.cuda.is_available():
            return torch.device('cpu')
            
        if self.config.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            return torch.device('cuda')
            
        return torch.device('cuda:0')
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimize model for training.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        model = model.to(self.device)
        
        if self.config.use_multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        return model
        
    def create_data_loader(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create optimized data loader.
        
        Args:
            dataset: PyTorch dataset
            shuffle: Whether to shuffle data
            
        Returns:
            Optimized data loader
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
    def training_step(
        self,
        model: nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Perform single training step with optimization.
        
        Args:
            model: PyTorch model
            batch: Tuple of (inputs, targets)
            criterion: Loss function
            optimizer: Model optimizer
            
        Returns:
            Loss value
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Mixed precision training
        if self.config.use_mixed_precision:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            self.scaler.scale(loss).backward()
            
            if self.config.gradient_clipping > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.gradient_clipping
                )
                
            self.scaler.step(optimizer)
            self.scaler.update()
            
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.gradient_clipping
                )
                
            optimizer.step()
            
        optimizer.zero_grad()
        return loss.item()
        
    @staticmethod
    def profile_model(
        model: nn.Module,
        input_size: Tuple[int, ...],
        batch_size: int = 1,
        num_threads: int = 1
    ) -> Dict[str, float]:
        """Profile model performance.
        
        Args:
            model: PyTorch model
            input_size: Input tensor size (excluding batch dimension)
            batch_size: Batch size for profiling
            num_threads: Number of threads for CPU inference
            
        Returns:
            Dictionary with profiling metrics
        """
        torch.set_num_threads(num_threads)
        
        # Create random input
        input_shape = (batch_size,) + input_size
        inputs = torch.randn(input_shape)
        
        # Warm up
        for _ in range(10):
            model(inputs)
            
        # Measure inference time
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        # GPU measurements
        if next(model.parameters()).is_cuda:
            inputs = inputs.cuda()
            times = []
            
            with torch.no_grad():
                for _ in range(100):
                    starter.record()
                    model(inputs)
                    ender.record()
                    torch.cuda.synchronize()
                    times.append(starter.elapsed_time(ender))
                    
            gpu_time = np.mean(times)
            
            # CPU measurements
            model = model.cpu()
            inputs = inputs.cpu()
            
        # CPU measurements
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = torch.cuda.Event(enable_timing=True)
                model(inputs)
                end = torch.cuda.Event(enable_timing=True)
                times.append(end.elapsed_time(start))
                
        cpu_time = np.mean(times)
        
        # Calculate memory usage
        memory_params = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )
        memory_buffers = sum(
            b.numel() * b.element_size()
            for b in model.buffers()
        )
        
        return {
            'gpu_inference_time': gpu_time if 'gpu_time' in locals() else None,
            'cpu_inference_time': cpu_time,
            'parameter_memory': memory_params / 1024 / 1024,  # MB
            'buffer_memory': memory_buffers / 1024 / 1024,  # MB
            'total_memory': (memory_params + memory_buffers) / 1024 / 1024  # MB
        }
        
    @staticmethod
    def optimize_memory(model: nn.Module):
        """Optimize model memory usage.
        
        Args:
            model: PyTorch model
        """
        # Convert model to half precision
        model.half()
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        
        # Enable cudnn deterministic mode
        torch.backends.cudnn.deterministic = True
        
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """Optimize model for inference.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        model.eval()
        
        # Fuse consecutive batch norm and relu layers
        torch.quantization.fuse_modules(model, ['conv', 'bn', 'relu'])
        
        # Quantize model
        model_fp32 = model
        model_int8 = torch.quantization.quantize_dynamic(
            model_fp32,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        return model_int8
