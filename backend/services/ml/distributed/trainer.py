"""Distributed training utilities for deep learning models."""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import os
import logging
from pathlib import Path

from ..monitoring.metrics import MetricsTracker, MetricConfig

@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    num_nodes: int = 1
    gpus_per_node: int = 1
    node_rank: int = 0
    master_addr: str = 'localhost'
    master_port: str = '12355'
    backend: str = 'nccl'
    init_method: str = 'env://'

class DistributedTrainer:
    """Distributed training manager."""
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        metrics_config: Optional[MetricConfig] = None
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Distributed training configuration
            metrics_config: Optional metrics configuration
        """
        self.model = model
        self.config = config
        self.metrics_config = metrics_config
        self.world_size = config.num_nodes * config.gpus_per_node
        
    def setup(self, rank: int):
        """Setup distributed training environment.
        
        Args:
            rank: Process rank
        """
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            world_size=self.world_size,
            rank=rank
        )
        
    def cleanup(self):
        """Clean up distributed training environment."""
        dist.destroy_process_group()
        
    def prepare_model(self, rank: int) -> nn.Module:
        """Prepare model for distributed training.
        
        Args:
            rank: Process rank
            
        Returns:
            Distributed model
        """
        device = torch.device(f'cuda:{rank}')
        model = self.model.to(device)
        
        if self.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[rank],
                output_device=rank
            )
            
        return model
        
    def prepare_data_loader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Prepare data loader for distributed training.
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            Distributed data loader
        """
        sampler = DistributedSampler(dataset) if self.world_size > 1 else None
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
    def train(
        self,
        rank: int,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        criterion: nn.Module = nn.MSELoss(),
        optimizer_fn: Callable = torch.optim.Adam,
        lr: float = 0.001,
        batch_size: int = 32,
        epochs: int = 10,
        save_dir: Optional[str] = None
    ):
        """Train model in distributed setting.
        
        Args:
            rank: Process rank
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            criterion: Loss function
            optimizer_fn: Optimizer function
            lr: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            save_dir: Directory to save model checkpoints
        """
        # Setup process group
        self.setup(rank)
        
        # Prepare model and data
        model = self.prepare_model(rank)
        train_loader = self.prepare_data_loader(
            train_dataset,
            batch_size
        )
        val_loader = (
            self.prepare_data_loader(val_dataset, batch_size, shuffle=False)
            if val_dataset is not None else None
        )
        
        # Setup optimizer
        optimizer = optimizer_fn(model.parameters(), lr=lr)
        
        # Setup metrics tracking
        metrics_tracker = None
        if rank == 0 and self.metrics_config is not None:
            metrics_tracker = MetricsTracker(self.metrics_config)
            metrics_tracker.start_training()
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.cuda(rank)
                target = target.cuda(rank)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if metrics_tracker is not None and batch_idx % 10 == 0:
                    metrics_tracker.log_batch(
                        epoch,
                        batch_idx,
                        loss.item(),
                        {'loss': loss.item()},
                        'train'
                    )
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data = data.cuda(rank)
                        target = target.cuda(rank)
                        
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()
                        
                val_loss /= len(val_loader)
                
                if metrics_tracker is not None:
                    metrics_tracker.log_batch(
                        epoch,
                        0,
                        val_loss,
                        {'loss': val_loss},
                        'val'
                    )
            
            # Save checkpoint
            if rank == 0 and save_dir is not None:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss
                }
                
                torch.save(
                    checkpoint,
                    save_path / f'checkpoint_epoch_{epoch}.pt'
                )
        
        if metrics_tracker is not None:
            metrics_tracker.end_training()
            metrics_tracker.plot_metrics()
            
        self.cleanup()
        
    def run_distributed(
        self,
        train_fn: Callable,
        world_size: int,
        *args,
        **kwargs
    ):
        """Run distributed training.
        
        Args:
            train_fn: Training function
            world_size: Number of processes
            *args: Positional arguments for training function
            **kwargs: Keyword arguments for training function
        """
        mp.spawn(
            train_fn,
            args=(world_size, *args),
            nprocs=world_size,
            join=True
        )

class ParameterServer:
    """Parameter server for asynchronous distributed training."""
    
    def __init__(
        self,
        model: nn.Module,
        world_size: int,
        update_rule: str = 'mean'
    ):
        """Initialize parameter server.
        
        Args:
            model: PyTorch model
            world_size: Number of processes
            update_rule: Parameter update rule ('mean' or 'sum')
        """
        self.model = model
        self.world_size = world_size
        self.update_rule = update_rule
        self.parameter_buffer = {}
        
        for name, param in model.named_parameters():
            self.parameter_buffer[name] = []
            
    def receive_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        worker_rank: int
    ):
        """Receive gradients from worker.
        
        Args:
            gradients: Dictionary of gradients
            worker_rank: Worker rank
        """
        for name, grad in gradients.items():
            self.parameter_buffer[name].append(grad)
            
    def update_parameters(self) -> Dict[str, torch.Tensor]:
        """Update model parameters.
        
        Returns:
            Dictionary of updated parameters
        """
        updated_parameters = {}
        
        for name, buffer in self.parameter_buffer.items():
            if len(buffer) == self.world_size:
                if self.update_rule == 'mean':
                    updated_parameters[name] = torch.mean(
                        torch.stack(buffer),
                        dim=0
                    )
                else:  # sum
                    updated_parameters[name] = torch.sum(
                        torch.stack(buffer),
                        dim=0
                    )
                    
                buffer.clear()
                
        return updated_parameters
