"""Distributed data loading and processing utilities."""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading

@dataclass
class DataConfig:
    """Configuration for distributed data loading."""
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False
    shuffle_buffer_size: int = 10000
    cache_size: int = 1000  # Number of batches to cache

class StreamingDataset(Dataset):
    """Dataset for streaming data processing."""
    
    def __init__(
        self,
        data_source: Union[str, pd.DataFrame],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_size: int = 1000
    ):
        """Initialize dataset.
        
        Args:
            data_source: Path to data or DataFrame
            transform: Optional transform for features
            target_transform: Optional transform for targets
            cache_size: Number of items to cache
        """
        self.transform = transform
        self.target_transform = target_transform
        self.cache_size = cache_size
        self.cache = {}
        self.cache_queue = queue.Queue(maxsize=cache_size)
        
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
            self.data = data_source
            
        self.features = self.data.drop(['target'], axis=1).values
        self.targets = self.data['target'].values
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
            
        # Get data
        feature = self.features[idx]
        target = self.targets[idx]
        
        # Apply transforms
        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        # Convert to tensors
        feature = torch.FloatTensor(feature)
        target = torch.FloatTensor([target])
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            old_idx = self.cache_queue.get()
            del self.cache[old_idx]
            
        self.cache[idx] = (feature, target)
        self.cache_queue.put(idx)
        
        return feature, target

class ParallelDataLoader:
    """Data loader with parallel processing capabilities."""
    
    def __init__(
        self,
        config: DataConfig,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0
    ):
        """Initialize loader.
        
        Args:
            config: Data loading configuration
            distributed: Whether to use distributed sampling
            world_size: Number of distributed processes
            rank: Process rank
        """
        self.config = config
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank
        self.prefetch_queue = queue.Queue(maxsize=config.prefetch_factor)
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        
    def create_loader(
        self,
        dataset: Dataset,
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader.
        
        Args:
            dataset: PyTorch dataset
            shuffle: Whether to shuffle data
            
        Returns:
            Data loader
        """
        sampler = None
        if self.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            shuffle = False
            
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            drop_last=self.config.drop_last
        )
        
    def prefetch_batch(
        self,
        loader: DataLoader
    ) -> None:
        """Prefetch batch in background.
        
        Args:
            loader: Data loader
        """
        try:
            batch = next(loader)
            self.prefetch_queue.put(batch)
        except StopIteration:
            self.prefetch_queue.put(None)
            
    def iterate(self, loader: DataLoader):
        """Iterate over data with prefetching.
        
        Args:
            loader: Data loader
            
        Yields:
            Batches of data
        """
        iterator = iter(loader)
        
        # Start prefetching first batch
        self.executor.submit(self.prefetch_batch, iterator)
        
        while True:
            batch = self.prefetch_queue.get()
            if batch is None:
                break
                
            # Start prefetching next batch
            self.executor.submit(self.prefetch_batch, iterator)
            
            yield batch

class ShuffleBuffer:
    """Buffer for shuffling streaming data."""
    
    def __init__(self, size: int):
        """Initialize buffer.
        
        Args:
            size: Buffer size
        """
        self.size = size
        self.buffer = []
        
    def push(self, item: Any) -> Optional[Any]:
        """Push item to buffer and get random item if full.
        
        Args:
            item: Item to add
            
        Returns:
            Random item if buffer is full, None otherwise
        """
        if len(self.buffer) < self.size:
            self.buffer.append(item)
            return None
            
        idx = np.random.randint(self.size)
        out_item = self.buffer[idx]
        self.buffer[idx] = item
        return out_item
        
    def flush(self) -> List[Any]:
        """Flush remaining items.
        
        Returns:
            List of remaining items
        """
        np.random.shuffle(self.buffer)
        items = self.buffer
        self.buffer = []
        return items

class DataPrefetcher:
    """Prefetch data to GPU memory."""
    
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device
    ):
        """Initialize prefetcher.
        
        Args:
            loader: Data loader
            device: Target device
        """
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self.next_target = None
        self._preload()
        
    def _preload(self):
        """Preload next batch."""
        try:
            self.next_data, self.next_target = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_target = None
            return
            
        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.to(
                self.device,
                non_blocking=True
            )
            self.next_target = self.next_target.to(
                self.device,
                non_blocking=True
            )
            
    def next(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get next batch.
        
        Returns:
            Tuple of (data, target) or None if finished
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        target = self.next_target
        
        if data is None:
            return None
            
        self._preload()
        return data, target
