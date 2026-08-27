"""
dataset.py  —  BellHart dataset loader
=========================================
Uses HuggingFace datasets with streaming=True to fetch massive datasets dynamically
without requiring local storage. Prefetches, tokenizes on the fly, and supports
exact state_dict checkpoint saving and resumption without duplicate data.
"""

import time
import logging
from collections import deque
from typing import Tuple, Optional, Dict, Any

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

log = logging.getLogger(__name__)


class HFStreamingDataset(IterableDataset):
    """
    Streams data from a HuggingFace dataset, tokenizes it on the fly, 
    and yields (x, y) chunks of block_size. Supports exact state_dict()
    saving and load_state_dict() resumption.
    """
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        block_size: int,
        buffer_size: int = 10,
        seed: int = 42,
        config_name: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        load_dataset_fn: Optional[Any] = None,
    ):
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
        self.load_dataset_fn = load_dataset_fn

        self.epoch = 0
        self.chunks_yielded = 0
        self.token_buffer = deque()
        self.raw_dataset = None
        self.dataset_iter = None
        self._pending_state: Optional[Dict[str, Any]] = None

    def state_dict(self) -> Dict[str, Any]:
        """
        Capture the exact streaming state: HuggingFace dataset position,
        leftover unyielded tokens in the buffer, epoch, and chunk counter.
        """
        hf_state = None
        if self.raw_dataset is not None:
            try:
                hf_state = self.raw_dataset.state_dict()
            except Exception as e:
                log.warning(f"Could not extract HF dataset state_dict: {e}")

        return {
            "hf_state": hf_state,
            "token_buffer": list(self.token_buffer),
            "seed": self.seed,
            "epoch": self.epoch,
            "chunks_yielded": self.chunks_yielded,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state_dict: Optional[Dict[str, Any]]):
        """
        Set pending state to be applied when the iterator is constructed or resumed.
        """
        if not state_dict:
            return
        self._pending_state = state_dict
        self.seed = state_dict.get("seed", self.seed)
        self.epoch = state_dict.get("epoch", self.epoch)
        self.chunks_yielded = state_dict.get("chunks_yielded", self.chunks_yielded)
        if "token_buffer" in state_dict:
            self.token_buffer = deque(state_dict["token_buffer"])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed
        if worker_info is not None:
            seed += worker_info.id

        chunk_size = self.block_size + 1
        retry_delay = 1.0

        state_to_restore = self._pending_state
        self._pending_state = None
        load_fn = self.load_dataset_fn or load_dataset

        while True:
            try:
                # 1. Load HF streaming dataset
                if self.config_name:
                    ds = load_fn(self.dataset_name, self.config_name, split=self.split, streaming=True)
                else:
                    ds = load_fn(self.dataset_name, split=self.split, streaming=True)

                # 2. Shard across GPUs / TPU cores so each rank gets independent data
                if self.world_size > 1:
                    ds = ds.shard(num_shards=self.world_size, index=self.rank)

                # 3. Shuffle stream with specified buffer size
                ds = ds.shuffle(buffer_size=self.buffer_size, seed=seed)

                # 4. Restore state if resuming from checkpoint or reconnecting
                if state_to_restore is not None and state_to_restore.get("hf_state") is not None:
                    try:
                        ds.load_state_dict(state_to_restore["hf_state"])
                        if "token_buffer" in state_to_restore:
                            self.token_buffer = deque(state_to_restore["token_buffer"])
                    except Exception as e:
                        log.warning(f"Could not restore HF dataset state directly ({e}). Stream will continue with seed {seed}.")
                    state_to_restore = None

                self.raw_dataset = ds
                self.dataset_iter = iter(ds)

                for example in self.dataset_iter:
                    text = example.get("content", example.get("text", ""))
                    if not text:
                        continue

                    # Tokenize and append EOS/EOT delimiter between documents
                    tokens = self.tokenizer.encode(text)
                    tokens.append(self.tokenizer.eot_token)
                    self.token_buffer.extend(tokens)
                    del text, tokens

                    # Yield chunks of (block_size + 1)
                    while len(self.token_buffer) >= chunk_size:
                        chunk = [self.token_buffer.popleft() for _ in range(chunk_size)]
                        self.chunks_yielded += 1
                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        yield x, y

                # If stream finishes normally, advance epoch and continue seamlessly
                self.epoch += 1
                seed += 1
                self.seed = seed
                retry_delay = 1.0

            except Exception as e:
                log.warning(f"[Streaming Dataset] Reconnecting due to network drop or error: {e}")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 10.0)
                # Save latest state before reconnecting so we don't start from 0
                try:
                    state_to_restore = self.state_dict()
                except Exception:
                    pass


def create_streaming_dataloaders(
    dataset_name: str,
    tokenizer,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    dataset_config: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders using streaming.
    Val dataloader has an isolated seed offset so evaluation never consumes from or
    disturbs the training stream.
    """
    train_dataset = HFStreamingDataset(
        dataset_name=dataset_name,
        split="train",
        tokenizer=tokenizer,
        block_size=block_size,
        seed=seed,
        config_name=dataset_config,
        rank=rank,
        world_size=world_size,
    )

    # Dedicated val dataset with separate seed (offset by 100,000)
    val_dataset = HFStreamingDataset(
        dataset_name=dataset_name,
        split="train",
        tokenizer=tokenizer,
        block_size=block_size,
        seed=seed + 100_000,
        config_name=dataset_config,
        rank=rank,
        world_size=world_size,
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
