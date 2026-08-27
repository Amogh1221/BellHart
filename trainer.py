"""
trainer.py  —  BellHart GPU Training Pipeline (Single GPU & Multi-GPU DDP)
===========================================================================
Full-featured trainer with:
  - GPU (CUDA) and Multi-GPU DistributedDataParallel (DDP) support
  - Mixed-precision: bfloat16 / float16 with AMP GradScaler
  - Gradient accumulation for large effective batch sizes
  - Gradient clipping with norm tracking
  - EMA (exponential moving average) of model weights
  - Cosine & WSD (Warmup-Stable-Decay) LR schedules
  - Checkpointing (latest + best by val loss) with async HuggingFace backup
  - TensorBoard logging
  - Rich terminal output (loss, grad_norm, tok/s, VRAM, ETA)
  - Persistent file logging to logs/training_log.txt
  - Text sample generation at intervals
"""

import os
import contextlib
import math
import time
import re
import logging
import threading
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from huggingface_hub import HfApi, CommitOperationDelete
from huggingface_hub.utils import disable_progress_bars

from config import GPTConfig
from model import GPT, EMA


# ──────────────────────────────────────────────────────────────────────────────
# Dtype mapping
# ──────────────────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────


def get_lr(it, config: GPTConfig):
    # Warmup phase (linear ramp)
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)

    schedule = getattr(config, "lr_schedule", "cosine")
    if schedule == "wsd":
        # WSD: Warmup -> Stable (at peak LR) -> Rapid Cosine Decay
        decay_iters = getattr(config, "decay_iters", 15000)
        decay_start = max(config.warmup_iters, config.max_iters - decay_iters)
        if it < decay_start:
            return config.learning_rate  # Stable phase at peak LR
        if it >= config.max_iters:
            return config.min_lr
        decay_ratio = (it - decay_start) / max(1, config.max_iters - decay_start)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)

    # Default continuous Cosine decay
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / max(1, config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 0:
        return "N/A"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _vram_gb() -> tuple[float, float]:
    """Return total (reserved_gb, capacity_gb) across all CUDA devices."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    num_devices = torch.cuda.device_count()
    reserved = sum(torch.cuda.memory_reserved(i) for i in range(num_devices)) / 1e9
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_devices)) / 1e9
    return reserved, total


def _is_master() -> bool:
    """Returns True if this process is the master (should do logging, saving, etc.)."""
    return int(os.environ.get('RANK', 0)) == 0


# ──────────────────────────────────────────────────────────────────────────────
# File logger
# ──────────────────────────────────────────────────────────────────────────────


class FileLogger:
    """
    Persistent structured logger that writes to logs/training_log.txt.
    Survives terminal closes, SSH drops, and crashes.
    """

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "training_log.txt")
        self.file = open(self.path, "a", encoding="utf-8")

    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def write(self, msg: str, flush: bool = True):
        self.file.write(msg + "\n")
        if flush:
            self.file.flush()

    def log_step(self, step, total, loss, lr, grad_norm, tok_sec, tokens=0):
        line = (
            f"[{self._ts()}] STEP {step:>7d}/{total} | Tokens: {tokens:,} | "
            f"loss={loss:.4f} | lr={lr:.2e} | grad_norm={grad_norm:.3f} | "
            f"tok/s={tok_sec:,.0f}"
        )
        self.write(line)

    def log_eval(self, step, total, train_loss, val_loss, best_val, ppl,
                 ema_val, lr, avg_grad_norm, tok_sec, vram_alloc, vram_total,
                 elapsed_str, eta_str, tokens=0):
        pct = step / total * 100 if total > 0 else 0
        delta_val = val_loss - best_val if best_val < float("inf") and val_loss != best_val else 0.0
        delta_str = f"Δ: {delta_val:+.4f}" if delta_val != 0 else "NEW BEST"

        hr = "═" * 56
        block = f"""
{hr}
  EVALUATION @ Step {step} / {total}   ({pct:.1f}%)
{hr}
  Tokens Processed : {tokens:,}
  Train Loss       : {train_loss:.4f}
  Val Loss         : {val_loss:.4f}  (best: {best_val:.4f}  {delta_str})
  Perplexity     : {ppl:.2f}
  EMA Val Loss   : {f'{ema_val:.4f}' if ema_val is not None else 'N/A'}
  Learning Rate  : {lr:.2e}
  Avg Grad Norm  : {avg_grad_norm:.3f}
  Tokens/sec     : {tok_sec:,.0f}
  VRAM           : {vram_alloc:.1f} / {vram_total:.1f} GB
  Elapsed        : {elapsed_str}
  ETA            : {eta_str}
{hr}"""
        self.write(block)

    def log_config(self, config: GPTConfig, n_params: int):
        hr = "═" * 56
        self.write(f"\n{hr}")
        self.write(f"  TRAINING STARTED — {self._ts()}")
        self.write(f"  Model params    : {n_params:,}")
        self.write(hr)

    def log_end(self, step, elapsed, best_val):
        hr = "═" * 56
        self.write(f"\n{hr}")
        self.write(f"  TRAINING COMPLETE — {self._ts()}")
        self.write(f"  Final step     : {step:,}")
        self.write(f"  Elapsed        : {_format_elapsed(elapsed)}")
        self.write(f"  Best val loss  : {best_val:.4f}")
        self.write(f"  Best perplexity: {math.exp(best_val):.2f}")
        self.write(hr)

    def close(self):
        self.file.close()


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────


class Trainer:
    def __init__(self, config: GPTConfig, tokenizer, train_loader, val_loader, is_ddp=False):
        self.config = config
        self.tokenizer = tokenizer
        self.is_ddp = is_ddp

        # Set device
        if is_ddp:
            ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = torch.device(f"cuda:{ddp_local_rank}")
            self.is_master = (int(os.environ.get('RANK', 0)) == 0)
        elif config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.is_master = True
        else:
            self.device = torch.device("cpu")
            self.is_master = True

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("samples", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        # Only master writes logs and tensorboard
        if self.is_master:
            self.writer = SummaryWriter(log_dir="runs")
            self.flog = FileLogger("logs")
        else:
            self.writer = None
            self.flog = None

        self.model = GPT(config).to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        if self.is_master:
            print(f"Model parameters: {self.n_params:,}")

        if config.compile and hasattr(torch, "compile"):
            if self.is_master:
                print("Compiling model.forward...")
            self.model.forward = torch.compile(self.model.forward, mode="default")

        # Multi-device wrapping (GPU DDP)
        if self.is_ddp:
            if self.is_master:
                print("Wrapping model with DistributedDataParallel (DDP).")
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.model = DDP(self.model, device_ids=[ddp_local_rank])

        self.optimizer = (self.model.module if self.is_ddp else self.model).configure_optimizers(config)

        self.ema = (
            EMA(self.model.module if self.is_ddp else self.model, decay=config.ema_decay) if config.use_ema else None
        )

        # GradScaler for float16 AMP on GPU
        self.use_scaler = (self.device.type == "cuda" and config.dtype == "float16")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_scaler
        ) if self.device.type == "cuda" else None

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = getattr(train_loader, "dataset", None)
        self.val_dataset = getattr(val_loader, "dataset", None)

        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader) if val_loader is not None else None

        self.iter_num = 0
        self.best_val_loss = float("inf")
        self.micro_step = 0

        # Metrics accumulators
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0
        self._tokens_processed = 0
        self._last_log_time = None

    def get_batch(self, split="train"):
        if split == "val" and self.val_iter is not None:
            loader_iter = self.val_iter
            raw_loader = self.val_loader
        else:
            loader_iter = self.train_iter
            raw_loader = self.train_loader

        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(raw_loader)
            if split == "val":
                self.val_iter = loader_iter
            else:
                self.train_iter = loader_iter
            x, y = next(loader_iter)

        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        eval_steps = min(max(self.config.eval_iters, 5), 20)

        # Estimate validation loss on dedicated validation stream
        for split in (["train", "val"] if self.val_iter is not None else ["train"]):
            total_loss = 0.0
            for k in range(eval_steps):
                x, y = self.get_batch(split)
                with torch.amp.autocast(
                    "cuda",
                    dtype=_DTYPE_MAP.get(self.config.dtype, torch.float16),
                    enabled=(self.device.type == "cuda" and self.config.dtype != "float32"),
                ):
                    logits, _ = self.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
                total_loss += loss.item()
                del logits, loss, x, y

            out[split] = total_loss / max(eval_steps, 1)

        if "val" not in out:
            out["val"] = out["train"]

        self.model.train()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        return out

    def save_checkpoint(self, path, step_num=None, max_ckpt=3, epoch_name=None):
        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        
        # Save dataset streaming state
        dataset_state = None
        if getattr(self, "train_dataset", None) is not None and hasattr(self.train_dataset, "state_dict"):
            try:
                dataset_state = self.train_dataset.state_dict()
            except Exception:
                pass

        # In DDP, gather dataset states across all ranks so each GPU's stream position is saved
        dataset_states = None
        if self.is_ddp:
            import torch.distributed as dist
            if dist.is_initialized():
                try:
                    world_size = dist.get_world_size()
                    gathered = [None for _ in range(world_size)]
                    dist.all_gather_object(gathered, dataset_state)
                    dataset_states = gathered
                except Exception:
                    pass

        ckpt = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "dataset_state": dataset_state,
        }
        if dataset_states is not None:
            ckpt["dataset_states"] = dataset_states

        if getattr(self, "ema", None) is not None:
            ckpt["ema"] = self.ema.state_dict()
        if getattr(self, "use_scaler", False) and getattr(self, "scaler", None) is not None:
            ckpt["scaler"] = self.scaler.state_dict()

        if self.is_master:
            torch.save(ckpt, path)

        if not self.is_master:
            return

        # Local cleanup
        if step_num is not None:
            try:
                ckpt_dir = os.path.dirname(path)
                files = os.listdir(ckpt_dir)
                ckpt_files = [f for f in files if re.match(r"checkpoint-\d+\.pt", f)]
                ckpt_files.sort(key=lambda x: int(re.search(r"checkpoint-(\d+)\.pt", x).group(1)))
                if len(ckpt_files) > max_ckpt:
                    to_delete = ckpt_files[:-max_ckpt]
                    for f in to_delete:
                        os.remove(os.path.join(ckpt_dir, f))
            except Exception:
                pass

        # Background HuggingFace sync
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            def background_sync():
                try:
                    import warnings
                    from huggingface_hub.utils import disable_progress_bars
                    disable_progress_bars()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
                        import shutil
                        import uuid
                        
                        api = HfApi(token=hf_token)
                        repo_id = self.config.hf_repo
                        
                        sync_path = f"{path}.{uuid.uuid4().hex}.sync"
                        shutil.copy2(path, sync_path)
                        
                        try:
                            upload_ops = []
                            delete_ops = []
                            
                            # Add historical copy operation
                            if step_num is not None:
                                upload_ops.append(CommitOperationAdd(
                                    path_in_repo=f"checkpoints/checkpoint-{step_num:06d}.pt",
                                    path_or_fileobj=sync_path
                                ))
                                
                            # Also update latest.pt on Hugging Face
                            upload_ops.append(CommitOperationAdd(
                                path_in_repo="checkpoints/latest.pt",
                                path_or_fileobj=sync_path
                            ))

                            # Add logs operation
                            if os.path.exists("logs/training_log.txt"):
                                upload_ops.append(CommitOperationAdd(
                                    path_in_repo="logs/training_log.txt",
                                    path_or_fileobj="logs/training_log.txt"
                                ))
                                
                            # Execute upload operations
                            api.create_commit(
                                repo_id=repo_id,
                                repo_type="dataset",
                                operations=upload_ops,
                                commit_message=f"Upload checkpoints and logs (Step {step_num if step_num is not None else 'Unknown'})"
                            )
                            
                            # Determine cleanup operations (delete old checkpoints)
                            if step_num is not None:
                                try:
                                    files = api.list_repo_files(repo_id, repo_type="dataset")
                                    ckpt_files = [f for f in files if re.match(r"checkpoints/checkpoint-\d+\.pt", f)]
                                    ckpt_files.sort(key=lambda x: int(re.search(r"checkpoint-(\d+)\.pt", x).group(1)))
                                    if len(ckpt_files) > max_ckpt:
                                        to_delete = ckpt_files[:-max_ckpt]
                                        for f in to_delete:
                                            delete_ops.append(CommitOperationDelete(path_in_repo=f))
                                            
                                    if len(delete_ops) > 0:
                                        api.create_commit(
                                            repo_id=repo_id, 
                                            repo_type="dataset", 
                                            operations=delete_ops, 
                                            commit_message=f"Cleanup old checkpoints (keeping latest {max_ckpt})"
                                        )
                                        
                                        if hasattr(api, 'super_squash_history'):
                                            try:
                                                api.super_squash_history(
                                                    repo_id=repo_id,
                                                    repo_type="dataset",
                                                    commit_message="Squash history to prevent LFS storage bloat"
                                                )
                                            except Exception as e:
                                                print(f"Failed to squash history: {e}")
                                                
                                except Exception:
                                    pass
                            
                        finally:
                            if os.path.exists(sync_path):
                                os.remove(sync_path)
                                
                    print("\n======SAVED======\n", flush=True)
                except Exception as e:
                    print(f"\n[HF Sync Error] {e}\n", flush=True)

            threading.Thread(target=background_sync, daemon=True).start()

    def load_checkpoint(self, path):
        # In DDP, stagger loading across ranks to avoid peak host CPU RAM spikes
        if self.is_ddp:
            import torch.distributed as dist
            if dist.is_initialized():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                world_size = int(os.environ.get("WORLD_SIZE", 1))
                for r in range(world_size):
                    if local_rank == r:
                        self._load_ckpt_internal(path)
                    dist.barrier()
                return

        self._load_ckpt_internal(path)

    def _load_ckpt_internal(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        base_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.iter_num = ckpt.get("iter_num", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        if self.iter_num > 0 and self.iter_num % self.config.save_interval == 0:
            self.iter_num += 1

        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
            self.ema.shadow = {
                k: v.to(self.device) for k, v in self.ema.shadow.items()
            }
        if self.use_scaler and self.scaler is not None and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])

        # ── Restore Dataset Streaming State ──────────────────────────────
        if getattr(self, "train_dataset", None) is not None and hasattr(self.train_dataset, "load_state_dict"):
            local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
            ds_state = None
            if "dataset_states" in ckpt and isinstance(ckpt["dataset_states"], list) and len(ckpt["dataset_states"]) > local_rank:
                ds_state = ckpt["dataset_states"][local_rank]
            elif "dataset_state" in ckpt:
                ds_state = ckpt["dataset_state"]

            if ds_state is not None:
                self.train_dataset.load_state_dict(ds_state)
                if self.is_master:
                    print(f"Restored streaming dataset state (chunks yielded: {ds_state.get('chunks_yielded', 'N/A')}, epoch: {ds_state.get('epoch', 0)})")
            else:
                if self.iter_num > 0 and self.is_master:
                    print(f"Note: Checkpoint from step {self.iter_num} has no saved dataset state. Shifting seed to avoid duplicate data replay.")
                    if hasattr(self.train_dataset, "seed"):
                        self.train_dataset.seed += (self.iter_num // 1000 + 1)

            self.train_iter = iter(self.train_loader)

        del ckpt
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        if self.is_master:
            print(f"Loaded checkpoint from {path} (iteration {self.iter_num}, best val loss: {self.best_val_loss:.4f})")

    def generate_samples(self):
        if not self.is_master:
            return
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        base_model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()

        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        for i in range(self.config.num_generations):
            temp = self.config.temperature * (1.0 + 0.1 * i)
            out = base_model.generate(
                context,
                max_new_tokens=self.config.max_new_tokens_gen,
                temperature=temp,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
            text = self.tokenizer.decode(out[0].tolist())
            sample_path = f"samples/step_{self.iter_num:07d}_{i}.txt"
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(text)
            preview = text[:200].encode("ascii", errors="replace").decode("ascii")
            print(f"\nSample {i} (t={temp:.2f}):\n{preview}...\n")

        if self.ema is not None:
            self.ema.restore()
        base_model.train()

    def _get_tokens_per_step(self) -> int:
        """Tokens processed per optimizer step (across all gradient accumulation micro-steps and all devices)."""
        tps = (
            self.config.batch_size
            * self.config.block_size
            * self.config.gradient_accumulation_steps
        )
        if self.is_ddp:
            tps *= int(os.environ.get('WORLD_SIZE', 1))
        return tps

    def train(self):
        config = self.config
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler

        # ── Log config at training start ─────────────────────────────────
        if self.iter_num == 0 and self.is_master and self.flog:
            self.flog.log_config(config, self.n_params)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.train()
        running_loss = 0.0
        start_time = time.time()
        self._last_log_time = start_time
        
        tokens_per_step = self._get_tokens_per_step()
        self._tokens_processed = self.iter_num * tokens_per_step
        self._steps_taken_since_resume = 0

        if self.is_master:
            pbar = tqdm(
                total=config.max_iters,
                initial=self.iter_num,
                desc="Training",
                dynamic_ncols=True,
            )
        else:
            pbar = None

        while self.iter_num < config.max_iters:
            lr = get_lr(self.iter_num, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = self.get_batch("train")

            # DDP no_sync: skip all-reduce on all micro-steps except the last
            is_last_micro = (self.micro_step + 1) % config.gradient_accumulation_steps == 0
            ctx = contextlib.nullcontext() if (not self.is_ddp or is_last_micro) else model.no_sync()
            with ctx:
                with torch.amp.autocast(
                    "cuda",
                    dtype=_DTYPE_MAP.get(config.dtype, torch.float16),
                    enabled=(self.device.type == "cuda" and config.dtype != "float32"),
                ):
                    logits, _ = model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                    ) / config.gradient_accumulation_steps
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            last_loss = loss.detach()

            # Free activations and batch tensors immediately before next micro-step
            del logits, loss, x, y

            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            self.micro_step += 1
            if pbar:
                micro = self.micro_step % config.gradient_accumulation_steps
                if micro == 0: micro = config.gradient_accumulation_steps
                pbar.set_description(f"Training (Micro {micro}/{config.gradient_accumulation_steps})")

            if self.micro_step % config.gradient_accumulation_steps == 0:
                # ── Gradient clipping + norm tracking ────────────────────
                grad_norm = 0.0
                if scaler is not None:
                    if config.grad_clip > 0.0:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_clip
                        ).item()
                    else:
                        total_norm_sq = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                total_norm_sq += p.grad.data.float().norm().item() ** 2
                        grad_norm = total_norm_sq ** 0.5

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if config.grad_clip > 0.0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_clip
                        ).item()
                    else:
                        total_norm_sq = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                total_norm_sq += p.grad.data.float().norm().item() ** 2
                        grad_norm = total_norm_sq ** 0.5
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update()

                step_loss = last_loss.item() * config.gradient_accumulation_steps
                running_loss += step_loss
                self._grad_norm_sum += grad_norm
                self._grad_norm_count += 1
                self._tokens_processed += tokens_per_step

                # Periodic host RAM cleanup to prevent OOM killer on cloud VMs
                import gc
                gc.collect()
                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

                # ── Per-step terminal + file log (master only) ───────────
                if self.iter_num % config.log_interval == 0 and self.iter_num > 0:
                    avg_loss = running_loss / max(self._grad_norm_count, 1)
                    avg_gn = self._grad_norm_sum / max(self._grad_norm_count, 1)
                    running_loss = 0.0
                    self._grad_norm_sum = 0.0
                    self._grad_norm_count = 0

                    if self.is_master:
                        now = time.time()
                        elapsed = now - start_time
                        dt = now - self._last_log_time if self._last_log_time else 1.0
                        tok_sec = (config.log_interval * tokens_per_step) / max(dt, 1e-6)
                        self._last_log_time = now

                        # Terminal
                        log_str = (
                            f"[Step {self.iter_num:>7d}/{config.max_iters}]  "
                            f"Tokens: {self._tokens_processed:,}  "
                            f"loss={avg_loss:.4f}  lr={lr:.2e}  "
                            f"grad_norm={avg_gn:.3f}  "
                            f"tok/s={tok_sec:,.0f}"
                        )
                        if pbar:
                            pbar.write(log_str)
                        else:
                            print(log_str)

                        # TensorBoard
                        if self.writer:
                            self.writer.add_scalar("train/loss", avg_loss, self.iter_num)
                            self.writer.add_scalar("train/lr", lr, self.iter_num)
                            self.writer.add_scalar("train/grad_norm", avg_gn, self.iter_num)
                            self.writer.add_scalar("train/tokens_per_sec", tok_sec, self.iter_num)

                        # File log (one-line)
                        if self.flog:
                            self.flog.log_step(
                                self.iter_num, config.max_iters, avg_loss, lr,
                                avg_gn, tok_sec, tokens=self._tokens_processed,
                            )

                # ── Evaluation ───────────────────────────────────────────
                if self.iter_num % config.eval_interval == 0 and self.iter_num > 0 and self._steps_taken_since_resume > 0:
                    losses = self.estimate_loss()
                    val_loss = losses["val"]
                    train_loss = losses["train"]
                    ppl = math.exp(min(val_loss, 20.0))  # cap to prevent overflow

                    if self.is_master and self.writer:
                        self.writer.add_scalar("eval/train_loss", train_loss, self.iter_num)
                        self.writer.add_scalar("eval/val_loss", val_loss, self.iter_num)
                        self.writer.add_scalar("eval/perplexity", ppl, self.iter_num)

                    # EMA evaluation
                    ema_val = None
                    if self.ema is not None:
                        self.ema.apply_shadow()
                        ema_losses = self.estimate_loss()
                        ema_val = ema_losses["val"]
                        if self.is_master and self.writer:
                            self.writer.add_scalar(
                                "eval/ema_val_loss", ema_val, self.iter_num
                            )
                        self.ema.restore()

                    # Compute metrics for display
                    elapsed = time.time() - start_time
                    sec_per_step = elapsed / max(self._steps_taken_since_resume, 1)
                    steps_remaining = config.max_iters - self.iter_num
                    eta = steps_remaining * sec_per_step
                    vram_alloc, vram_total = _vram_gb()
                    avg_gn = self._grad_norm_sum / max(self._grad_norm_count, 1)
                    tok_sec = tokens_per_step / max(sec_per_step, 1e-6)

                    is_best = val_loss < self.best_val_loss

                    if self.is_master:
                        hr = "═" * 56
                        pct = self.iter_num / config.max_iters * 100
                        delta_val = val_loss - self.best_val_loss if self.best_val_loss < float("inf") else 0.0
                        delta_str = f"Δ: {delta_val:+.4f}" if not is_best else "NEW BEST ★"

                        eval_str = (
                            f"\n{hr}\n"
                            f"  EVALUATION @ Step {self.iter_num} / {config.max_iters}   ({pct:.1f}%)\n"
                            f"{hr}\n"
                            f"  Tokens Processed : {self._tokens_processed:,}\n"
                            f"  Train Loss       : {train_loss:.4f}\n"
                            f"  Val Loss         : {val_loss:.4f}  (best: {self.best_val_loss:.4f}  {delta_str})\n"
                            f"  Perplexity       : {ppl:.2f}\n"
                        )
                        if ema_val is not None:
                            eval_str += f"  EMA Val Loss     : {ema_val:.4f}\n"
                        eval_str += (
                            f"  Learning Rate    : {lr:.2e}\n"
                            f"  Avg Grad Norm    : {avg_gn:.3f}\n"
                            f"  Tokens/sec       : {tok_sec:,.0f}\n"
                            f"  Elapsed          : {_format_elapsed(elapsed)}\n"
                            f"{hr}"
                        )
                        if pbar:
                            pbar.write(eval_str)
                        else:
                            print(eval_str)

                        if self.flog:
                            self.flog.log_eval(
                                step=self.iter_num,
                                total=config.max_iters,
                                train_loss=train_loss,
                                val_loss=val_loss,
                                best_val=self.best_val_loss,
                                ppl=ppl,
                                ema_val=ema_val,
                                lr=lr,
                                avg_grad_norm=avg_gn,
                                tok_sec=tok_sec,
                                vram_alloc=vram_alloc,
                                vram_total=vram_total,
                                elapsed_str=_format_elapsed(elapsed),
                                eta_str=_format_eta(eta),
                                tokens=self._tokens_processed,
                            )

                        if pbar:
                            pbar.set_postfix({
                                "train": f"{train_loss:.4f}",
                                "val": f"{val_loss:.4f}",
                                "ppl": f"{ppl:.1f}",
                                "lr": f"{lr:.2e}",
                            })

                    ckpt_path = f"checkpoints/checkpoint-{self.iter_num + 1:06d}.pt"
                    if is_best:
                        self.best_val_loss = val_loss
                    self.save_checkpoint(ckpt_path, step_num=self.iter_num + 1)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if self.iter_num % config.gen_interval == 0 and self.iter_num > 0:
                    self.generate_samples()

                if self.iter_num % config.save_interval == 0 and self.iter_num > 0 and self._steps_taken_since_resume > 0:
                    ckpt_path = f"checkpoints/checkpoint-{self.iter_num + 1:06d}.pt"
                    self.save_checkpoint(ckpt_path, step_num=self.iter_num + 1)

                self.iter_num += 1
                self._steps_taken_since_resume += 1
                
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()
        self.save_checkpoint("checkpoints/latest.pt", step_num=self.iter_num)
        elapsed = time.time() - start_time
        if self.is_master:
            print(f"\nTraining completed in {elapsed / 3600:.2f} hours")
            print(f"Best val loss: {self.best_val_loss:.4f}  (perplexity: {math.exp(self.best_val_loss):.2f})")

            if self.flog:
                self.flog.log_end(self.iter_num, elapsed, self.best_val_loss)
                self.flog.close()
            if self.writer:
                self.writer.close()
