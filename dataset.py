"""
dataset.py  —  BellHart dataset loader
=========================================
Uses HuggingFace datasets with streaming=True to fetch massive datasets dynamically
without requiring local storage. Prefetches and tokenizes on the fly.
"""

import logging
from typing import Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

log = logging.getLogger(__name__)

class HFStreamingDataset(IterableDataset):
    """
    Streams data from a HuggingFace dataset, tokenizes it on the fly, 
    and yields (x, y) chunks of block_size.
    """
    def __init__(self, dataset_name: str, split: str, tokenizer, block_size: int, buffer_size: int = 100, seed: int = 42, config_name: str = None, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        import time
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed
        if worker_info is not None:
            seed += worker_info.id

        buffer = []
        chunk_size = self.block_size + 1
        retry_delay = 1.0

        while True:
            try:
                # Load streaming dataset
                if self.config_name:
                    dataset = load_dataset(self.dataset_name, self.config_name, split=self.split, streaming=True)
                else:
                    dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
                    
                # Shard the dataset across GPUs/TPUs so they don't train on the same data
                if self.world_size > 1:
                    dataset = dataset.shard(num_shards=self.world_size, index=self.rank)
                    
                # Shuffle the stream
                dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=seed)

                for example in dataset:
                    text = example.get("content", example.get("text", ""))
                    if not text:
                        continue

                    # Tokenize
                    tokens = self.tokenizer.encode(text)
                    buffer.extend(tokens)

                    # Yield chunks
                    while len(buffer) >= chunk_size:
                        chunk = buffer[:chunk_size]
                        buffer = buffer[chunk_size:]
                        
                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        yield x, y
                
                # If stream finishes normally, advance seed and continue seamlessly
                seed += 1
                retry_delay = 1.0
            except Exception as e:
                # Catch transient HuggingFace network disconnections, SSL drops, rate limits
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 10.0)
                seed += 1

def create_streaming_dataloaders(
    dataset_name: str,
    tokenizer,
    block_size: int,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
    dataset_config: str = None,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders using streaming.
    """
    
    train_dataset = HFStreamingDataset(
        dataset_name=dataset_name,
        split="train",
        tokenizer=tokenizer,
        block_size=block_size,
        seed=seed,
        config_name=dataset_config,
        rank=rank,
        world_size=world_size
    )
    
    # Validation stream (different seed to sample different documents)
    val_dataset = HFStreamingDataset(
        dataset_name=dataset_name,
        split="train",
        tokenizer=tokenizer,
        block_size=block_size,
        seed=seed + 9999,
        config_name=dataset_config,
        rank=rank,
        world_size=world_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    return train_loader, val_loader
