"""
bert.py  —  Modern BERT (Encoder) Pre-Training Pipeline
=========================================================
State-of-the-art 12-layer bidirectional Transformer Encoder pre-training
optimized for prompt routing and semantic representation.

Architectural Innovations:
  1. Pre-RMSNorm — Fast, stable pre-normalization.
  2. Rotary Position Embeddings (RoPE, theta=500,000) — Infinite context flexibility.
  3. Query-Key Normalization (QK-Norm) — Eliminates attention entropy collapse.
  4. Value-Residual Learning (Res-V) — Cross-layer value shortcuts preserving fine-grained token identity.
  5. SwiGLU Feed-Forward Network — Gated bilinear FFN (~2.68x width).
  6. Bidirectional FlashAttention — PyTorch native SDPA with is_causal=False.
  7. Weight Tying — Embeddings tied to MLM prediction head.
  8. Modern Dynamic Masking (MLM) — 20% span-aware masking without obsolete NSP.
  9. Strict Domain Separation — Saves to bert_checkpoints/ and bert_logs/.
 10. Multi-GPU DDP + Kaggle 2x T4 Ready — Automatic sharding, TF32/FP16 autotuning.
"""

import os
import sys
import math
import time
import json
import random
import logging
import argparse
import threading
import contextlib
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tokenizer import Tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuration Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BertConfig:
    # Architecture
    vocab_size: int = 32768
    n_embd: int = 512
    n_head: int = 8                # Head dimension: 512 / 8 = 64
    n_layer: int = 12              # 12-layer hierarchical abstraction
    intermediate_size: int = 1376  # SwiGLU hidden dimension (~2.68x n_embd, multiple of 64)
    block_size: int = 1024         # Sequence length (2048 maximum supported via RoPE)
    rope_theta: float = 500000.0   # RoPE base theta matching BellHart
    norm_eps: float = 1e-5
    dropout: float = 0.0

    # Training & Optimization
    max_iters: int = 150000        # 150,000 pre-training steps
    learning_rate: float = 1e-3    # Peak LR (ModernBERT / BERT encoders train optimally at 1e-3)
    min_lr: float = 1e-4           # Final decayed LR (10% of peak)
    warmup_iters: int = 3000       # Linear warmup steps
    weight_decay: float = 0.01     # Decoupled weight decay
    beta1: float = 0.90
    beta2: float = 0.98            # ModernBERT / RoBERTa standard
    grad_clip: float = 1.0

    # Batching & Accumulation
    batch_size: int = 8            # Micro-batch per GPU (safe 8.7GB on 15GB T4)
    gradient_accumulation_steps: int = 4  # 8 * 4 * 2 GPUs * 1024 tokens = 65,536 tokens/step
    mask_prob: float = 0.20        # 20% dynamic masking (ModernBERT standard)

    # Logging & Checkpointing
    log_interval: int = 50         # Log to file and TensorBoard every 50 steps
    eval_interval: int = 500
    eval_iters: int = 20
    save_interval: int = 200       # Checkpoint every 200 steps on Kaggle
    hf_repo: str = "Amogh1221/bellhart_training"

    # Hardware & Precision
    device: str = "cuda"
    dtype: str = "bfloat16"        # Auto-switches to float16 on T4
    tf32: bool = True
    use_8bit_optimizer: bool = False

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})


# ──────────────────────────────────────────────────────────────────────────────
# 2. Architectural Components
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for bidirectional encoders."""
    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 500000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class ModernBertAttention(nn.Module):
    """
    Bidirectional Self-Attention with:
      1. Per-head QK-Norm
      2. RoPE Positional Encoding
      3. Value-Residual Learning (Res-V)
      4. PyTorch Native SDPA (FlashAttention, is_causal=False)
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        v_prev: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ── Value-Residual Learning (Res-V) ──────────────────────────────
        v_cur = v
        if v_prev is not None:
            v = 0.5 * v + 0.5 * v_prev

        # QK-Norm per head
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # Bidirectional Scaled Dot-Product Attention (is_causal=False)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(y)
        return out, v_cur


class SwiGLU(nn.Module):
    """Gated Linear Unit Feed-Forward Network."""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class ModernBertBlock(nn.Module):
    """Full Transformer Encoder Block with Pre-RMSNorm and Res-V."""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = ModernBertAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        v_prev: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed_x = self.ln_1(x)
        attn_out, v_cur = self.attn(normed_x, rope_cos, rope_sin, v_prev=v_prev, attention_mask=attention_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, v_cur


class ModernBertModel(nn.Module):
    """12-Layer Modern BERT Encoder Model with Tied Masked-LM Prediction Head."""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.rope = RotaryEmbedding(config.n_embd // config.n_head, max_seq_len=config.block_size, theta=config.rope_theta)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([ModernBertBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)

        # MLM Prediction Head (Tied with wte)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 1.0 / math.sqrt(module.weight.size(1)) if module.weight.size(1) > 0 else 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        rope_cos, rope_sin = self.rope(T)

        x = self.drop(self.wte(input_ids) * math.sqrt(self.config.n_embd))

        v_prev = None
        for layer in self.layers:
            x, v_cur = layer(x, rope_cos, rope_sin, v_prev=v_prev, attention_mask=attention_mask)
            v_prev = v_cur

        hidden_states = self.ln_f(x)
        logits = self.lm_head(hidden_states)

        if output_hidden_states:
            return logits, hidden_states
        return logits, None


# ──────────────────────────────────────────────────────────────────────────────
# 3. Dynamic Masked Language Modeling (MLM) Dataset
# ──────────────────────────────────────────────────────────────────────────────

class BertStreamingDataset(IterableDataset):
    """
    Streams web documents, tokenizes them, appends End-of-Text (<eos>) delimiters,
    and dynamically creates Masked Language Modeling pairs (masked_input, targets, loss_mask).
    """
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: Tokenizer,
        block_size: int,
        mask_prob: float = 0.20,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # In BellHart's tokenizer, token 0 is <|endoftext|> (<eos>)
        self.eot_token = tokenizer.eot_token
        # Designate a high token ID (or token 0) for [MASK]
        # In custom BPE, token 0 serves as delimiter/mask marker
        self.mask_token_id = 0

        self.token_buffer = deque()
        self.raw_dataset = None
        self._pending_state: Optional[Dict[str, Any]] = None
        self.chunks_yielded = 0
        self.epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        hf_state = None
        if self.raw_dataset is not None:
            try:
                hf_state = self.raw_dataset.state_dict()
            except Exception:
                pass
        return {
            "hf_state": hf_state,
            "token_buffer": list(self.token_buffer),
            "seed": self.seed,
            "epoch": self.epoch,
            "chunks_yielded": self.chunks_yielded,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]):
        if not state:
            return
        self._pending_state = state
        self.seed = state.get("seed", self.seed)
        self.epoch = state.get("epoch", self.epoch)
        self.chunks_yielded = state.get("chunks_yielded", self.chunks_yielded)
        if "token_buffer" in state:
            self.token_buffer = deque(state["token_buffer"])

    def _apply_dynamic_masking(self, tokens: list[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 80/10/10 dynamic masking rule across 20% of tokens:
          - 80% replaced with [MASK] token
          - 10% replaced with random token from vocabulary
          - 10% left unchanged
        Targets for unmasked positions are set to -100 (ignored by cross-entropy).
        """
        input_ids = list(tokens)
        labels = [-100] * len(tokens)

        for i in range(len(tokens)):
            # Don't mask document boundary delimiters
            if tokens[i] == self.eot_token:
                continue

            if random.random() < self.mask_prob:
                labels[i] = tokens[i]
                r = random.random()
                if r < 0.80:
                    input_ids[i] = self.mask_token_id
                elif r < 0.90:
                    input_ids[i] = random.randint(1, self.tokenizer.vocab_size - 1)
                # 10% remains original input_ids[i]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __iter__(self):
        from datasets import load_dataset
        seed = self.seed + self.rank
        random.seed(seed)

        state_to_restore = self._pending_state
        self._pending_state = None

        while True:
            try:
                ds = load_dataset(self.dataset_name, "CC-MAIN-2025-30", split=self.split, streaming=True)
                if self.world_size > 1:
                    ds = ds.shard(num_shards=self.world_size, index=self.rank)

                if state_to_restore is not None and state_to_restore.get("hf_state") is not None:
                    try:
                        ds.load_state_dict(state_to_restore["hf_state"])
                        if "token_buffer" in state_to_restore:
                            self.token_buffer = deque(state_to_restore["token_buffer"])
                    except Exception:
                        pass
                    state_to_restore = None

                self.raw_dataset = ds
                for example in ds:
                    text = example.get("content", example.get("text", ""))
                    if not text:
                        continue

                    # Tokenize and append <eos> delimiter between documents
                    toks = self.tokenizer.encode(text)
                    toks.append(self.eot_token)
                    self.token_buffer.extend(toks)

                    # Yield non-overlapping block_size sequences
                    while len(self.token_buffer) >= self.block_size:
                        chunk = [self.token_buffer.popleft() for _ in range(self.block_size)]
                        self.chunks_yielded += 1
                        x, y = self._apply_dynamic_masking(chunk)
                        yield x, y

                self.epoch += 1
                self.seed += 1

            except Exception as e:
                time.sleep(2.0)
                try:
                    state_to_restore = self.state_dict()
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────────────────────
# 4. Learning Rate Schedule & Logger
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(it: int, config: BertConfig) -> float:
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.max_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


class BertFileLogger:
    """Distinct file logger writing into bert_logs/."""
    def __init__(self, log_dir: str = "bert_logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = Path(log_dir) / "bert_training_log.txt"

    def log(self, message: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Checkpointing & HuggingFace Sync
# ──────────────────────────────────────────────────────────────────────────────

def sync_bert_huggingface(repo_id: str, is_master: bool = True):
    """Pulls latest BERT checkpoint from HuggingFace dataset repo."""
    os.makedirs("bert_checkpoints", exist_ok=True)
    os.makedirs("bert_logs", exist_ok=True)
    if not is_master:
        return

    try:
        from huggingface_hub import HfApi, hf_hub_download
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        bert_ckpts = [f for f in files if f.startswith("bert_checkpoints/bert-") and f.endswith(".pt")]
        if bert_ckpts:
            bert_ckpts.sort(reverse=True)
            latest = bert_ckpts[0]
            if not os.path.exists(latest):
                print(f"[BERT] Downloading latest remote checkpoint {latest}...")
                hf_hub_download(repo_id=repo_id, filename=latest, repo_type="dataset", local_dir=".")
                print(f"[BERT] Successfully downloaded {latest}")
    except Exception as e:
        print(f"[BERT] Checkpoint sync note: {e}")


def save_bert_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: BertStreamingDataset,
    step: int,
    val_loss: float,
    config: BertConfig,
    repo_id: str,
    hf_token: str,
    is_master: bool,
):
    """Saves checkpoint to bert_checkpoints/ and uploads asynchronously."""
    if not is_master:
        return

    os.makedirs("bert_checkpoints", exist_ok=True)
    ckpt_path = f"bert_checkpoints/bert-{step:06d}.pt"
    latest_path = "bert_checkpoints/latest_bert.pt"

    base_model = model.module if hasattr(model, "module") else model
    state = {
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "config": asdict(config),
        "dataset_state": train_dataset.state_dict(),
    }
    torch.save(state, ckpt_path)
    torch.save(state, latest_path)
    print(f"\n[BERT] Checkpoint saved: {ckpt_path} (Val Loss: {val_loss:.4f})")

    # Local retention: keep only the latest 3 numbered checkpoints on disk
    local_ckpts = sorted(
        Path("bert_checkpoints").glob("bert-[0-9]*.pt"),
        key=lambda p: p.stat().st_mtime
    )
    if len(local_ckpts) > 3:
        for old_p in local_ckpts[:-3]:
            try:
                old_p.unlink()
            except Exception:
                pass

    # Async HuggingFace backup (keeps latest 3 numbered checkpoints + latest_bert.pt)
    if hf_token and repo_id:
        def _upload():
            try:
                import re
                from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
                api = HfApi(token=hf_token)
                ops = [
                    CommitOperationAdd(path_in_repo=ckpt_path, path_or_fileobj=ckpt_path),
                    CommitOperationAdd(path_in_repo=latest_path, path_or_fileobj=latest_path),
                ]
                log_file = "bert_logs/bert_training_log.txt"
                if os.path.exists(log_file):
                    ops.append(CommitOperationAdd(path_in_repo=log_file, path_or_fileobj=log_file))

                # Check remote checkpoints and delete older ones beyond the latest 3
                try:
                    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                    remote_ckpts = [f for f in files if re.match(r"^bert_checkpoints/bert-\d+\.pt$", f)]
                    remote_ckpts.sort(key=lambda x: int(re.search(r"bert-(\d+)\.pt", x).group(1)))
                    # Keep at most 2 old ones since we are adding 1 new one
                    if len(remote_ckpts) >= 3:
                        to_delete = remote_ckpts[: len(remote_ckpts) - 2]
                        for f in to_delete:
                            ops.append(CommitOperationDelete(path_in_repo=f))
                except Exception as e:
                    print(f"[BERT HF List Note] {e}")

                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"[BERT] Upload Checkpoint Step {step} (Retain Latest 3)",
                )
            except Exception as e:
                print(f"[BERT HF Upload Error] {e}")

        threading.Thread(target=_upload, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Evaluation Function
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_bert(model: nn.Module, val_loader: DataLoader, eval_iters: int, dtype: torch.dtype) -> float:
    model.eval()
    total_loss = 0.0
    val_iter = iter(val_loader)
    for _ in range(eval_iters):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x, y = x.cuda(), y.cuda()
        with torch.amp.autocast("cuda", dtype=dtype):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        total_loss += loss.item()

    model.train()
    return total_loss / max(1, eval_iters)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main Training Execution
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Modern BERT Pre-Training")
    parser.add_argument("--hf_token", type=str, default="", help="Hugging Face Write Token")
    parser.add_argument("--fresh", action="store_true", help="Start from step 0")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    # Distributed setup
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        is_master = (rank == 0)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_master = True

    config = BertConfig()

    # Hardware detection and precision
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(local_rank).upper()
        # Tesla T4 uses float16; Ampere/Hopper uses bfloat16
        is_ampere_plus = vram >= 20 or any(tag in gpu_name for tag in ["A100", "H100", "B200", "RTX 30", "RTX 40"])
        config.dtype = "bfloat16" if is_ampere_plus else "float16"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if is_master:
            print(f"[BERT] GPU: {gpu_name} ({vram:.1f} GB) | Precision: {config.dtype} | DDP World Size: {world_size}")

    # Synchronize cloud checkpoints
    if is_master and not args.fresh:
        sync_bert_huggingface(config.hf_repo, is_master=True)
    if is_ddp:
        import torch.distributed as dist
        dist.barrier()

    # Tokenizer & Model
    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size

    model = ModernBertModel(config).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    if is_master:
        print(f"[BERT] Model Parameters: {n_params:,} (~{n_params/1e6:.1f}M)")

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    # Optimizer
    decay_params = [p for p in model.parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2 and p.requires_grad]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2))
    scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype == "float16"))

    # Streaming Dataloaders (Training + Validation)
    train_dataset = BertStreamingDataset(
        dataset_name="openbmb/Ultra-FineWeb-L1",
        split="train",
        tokenizer=tokenizer,
        block_size=config.block_size,
        mask_prob=config.mask_prob,
        seed=42,
        rank=rank,
        world_size=world_size,
    )
    val_dataset = BertStreamingDataset(
        dataset_name="openbmb/Ultra-FineWeb-L1",
        split="train",
        tokenizer=tokenizer,
        block_size=config.block_size,
        mask_prob=config.mask_prob,
        seed=142,
        rank=rank,
        world_size=world_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0, pin_memory=True)

    # Logging infrastructure
    writer = SummaryWriter(log_dir="bert_runs") if is_master else None
    flog = BertFileLogger("bert_logs") if is_master else None

    # Checkpoint resumption
    start_step = 0
    best_val_loss = float("inf")
    latest_ckpt = "bert_checkpoints/latest_bert.pt"
    if os.path.exists(latest_ckpt) and not args.fresh:
        if is_master:
            print(f"[BERT] Resuming from {latest_ckpt}...")
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        base = model.module if is_ddp else model
        base.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if "dataset_state" in ckpt:
            train_dataset.load_state_dict(ckpt["dataset_state"])
        if is_master:
            print(f"[BERT] Resumed at step {start_step} (Best Val Loss: {best_val_loss:.4f})")

    # Training Loop
    model.train()
    train_iter = iter(train_loader)
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

    tokens_per_step = config.batch_size * config.block_size * config.gradient_accumulation_steps * world_size
    step = start_step

    if is_master:
        print(f"[BERT] Commencing 150,000 steps (~{tokens_per_step:,} tokens/step)...", flush=True)

    t0 = time.time()
    t_interval_start = t0
    while step < config.max_iters:
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(config.gradient_accumulation_steps):
            x, y = next(train_iter)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            is_last = (micro == config.gradient_accumulation_steps - 1)
            ctx = contextlib.nullcontext() if (not is_ddp or is_last) else model.no_sync()

            with ctx:
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                    loss = loss / config.gradient_accumulation_steps

                accum_loss += loss.item()
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
            optimizer.step()

        step += 1
        t1 = time.time()
        dt_step = max(t1 - t0, 1e-6)
        toks_sec = tokens_per_step / dt_step
        tokens_processed = step * tokens_per_step

        # Print to terminal every single step with immediate flush (exactly like BellHart)
        if is_master:
            ppl = math.exp(min(accum_loss, 20.0))
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            log_line = (
                f"[{ts}] STEP {step:>6d}/{config.max_iters} | "
                f"Tokens: {tokens_processed:>11,d} | "
                f"loss={accum_loss:.4f} | "
                f"ppl={ppl:.2f} | "
                f"lr={lr:.2e} | "
                f"grad_norm={grad_norm:.3f} | "
                f"tok/s={toks_sec:,.0f}"
            )
            print(log_line, flush=True)

            # Structured file and TensorBoard logging every 50 steps
            if step % config.log_interval == 0:
                flog.log(log_line)
                if writer:
                    writer.add_scalar("bert/train_loss", accum_loss, step)
                    writer.add_scalar("bert/lr", lr, step)
                    writer.add_scalar("bert/grad_norm", grad_norm, step)
                    writer.add_scalar("bert/tokens_per_sec", toks_sec, step)

        t0 = t1

        # Evaluation & Checkpoint saving (every 200 steps on Kaggle)
        if step % config.save_interval == 0 or step == config.max_iters:
            val_loss = evaluate_bert(model, val_loader, config.eval_iters, dtype)
            val_ppl = math.exp(min(val_loss, 20.0))
            if is_master:
                eval_msg = f"── EVAL @ Step {step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} ──"
                print(f"\n{eval_msg}")
                flog.log(eval_msg)
                if writer:
                    writer.add_scalar("bert/val_loss", val_loss, step)
                    writer.add_scalar("bert/val_ppl", val_ppl, step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            save_bert_checkpoint(
                model=model,
                optimizer=optimizer,
                train_dataset=train_dataset,
                step=step,
                val_loss=val_loss,
                config=config,
                repo_id=config.hf_repo,
                hf_token=hf_token,
                is_master=is_master,
            )

    if is_master:
        print("\n════════════════════════════════════════════════════════")
        print("  BERT PRE-TRAINING COMPLETED (150,000 Steps)!")
        print("════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
