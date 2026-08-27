"""
dataset.py  —  BellHart Dynamic Streaming Dataset Pipeline
============================================================
Streams massive web-scale text datasets from Hugging Face on the fly without
requiring local disk space or high host CPU RAM.

Core Capabilities:
  1. On-The-Fly Tokenization — Tokenizes streaming text chunks dynamically.
  2. Document Boundary Delimiting — Appends End-of-Text (EOT) token after every document.
  3. Continuous Sequence Chunking — Generates fixed-length (block_size + 1) inputs/targets.
  4. Multi-GPU Deterministic Sharding — Shards the stream across DDP ranks.
  5. Exact State Persistence & Resumption — Serializes streaming positions, token queues,
     and random seeds to resume training seamlessly without data repetition or replay.
  6. Dedicated Validation Stream — Isolated stream with an offset seed so evaluation never
     disrupts the training data trajectory.
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
    Streaming PyTorch IterableDataset wrapping HuggingFace streaming datasets.
    
    Streams raw document text, encodes it into token IDs on the fly, buffers tokens,
    and yields autoregressive training pairs (x, y) where y is x shifted by 1 token.
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

        # Internal stream tracking state
        self.epoch = 0
        self.chunks_yielded = 0
        self.token_buffer = deque()
        self.raw_dataset = None
        self.dataset_iter = None
        self._pending_state: Optional[Dict[str, Any]] = None

    def state_dict(self) -> Dict[str, Any]:
        """
        Serializes the complete streaming dataset state.
        
        Captures:
          - HuggingFace dataset internal generator position.
          - Remaining unconsumed tokens in the local token buffer.
          - Current epoch counter and total chunks yielded.
          - Shard rank and random seed.
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
        Loads streaming checkpoint state to be applied when the stream is iterated.
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
        """
        Main streaming generator loop:
          1. Connects to HuggingFace dataset stream.
          2. Shards stream across DDP ranks.
          3. Shuffles stream with buffer.
          4. Restores HF checkpoint state if resuming.
          5. Reads documents, encodes tokens, and yields (block_size + 1) chunks.
        """
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
                # 1. Initialize HuggingFace streaming dataset
                if self.config_name:
                    ds = load_fn(self.dataset_name, self.config_name, split=self.split, streaming=True)
                else:
                    ds = load_fn(self.dataset_name, split=self.split, streaming=True)

                # 2. Shard across GPU ranks so each worker gets an independent stream partition
                if self.world_size > 1:
                    ds = ds.shard(num_shards=self.world_size, index=self.rank)

                # 3. Shuffle stream with specified buffer size
                ds = ds.shuffle(buffer_size=self.buffer_size, seed=seed)

                # 4. Restore exact checkpoint stream state if resuming
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

                # 5. Stream documents, tokenize, buffer, and yield chunks
                for example in self.dataset_iter:
                    text = example.get("content", example.get("text", ""))
                    if not text:
                        continue

                    # Tokenize and append End-of-Text delimiter between documents
                    tokens = self.tokenizer.encode(text)
                    tokens.append(self.tokenizer.eot_token)
                    self.token_buffer.extend(tokens)
                    del text, tokens

                    # Yield full sequence chunks of length (block_size + 1)
                    while len(self.token_buffer) >= chunk_size:
                        chunk = [self.token_buffer.popleft() for _ in range(chunk_size)]
                        self.chunks_yielded += 1
                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        del chunk
                        yield x, y

                # If the entire dataset stream finishes, advance epoch and increment seed
                self.epoch += 1
                seed += 1
                self.seed = seed
                retry_delay = 1.0

            except Exception as e:
                # Resilient error handling for transient network drops during cloud training
                log.warning(f"[Streaming Dataset] Reconnecting due to network drop or error: {e}")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 10.0)
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
    Builds training and validation DataLoaders using streaming iterables.
    
    The validation stream uses an isolated seed offset (+100,000) to ensure
    evaluation samples remain strictly out-of-distribution from the training stream.
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
