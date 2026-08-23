"""
trainer.py  —  nano_brain training loop (TPU + GPU dual-compatible)
===================================================================
Full-featured trainer with:
  - Dual TPU (torch_xla) and GPU (CUDA) support
  - Mixed-precision: native bfloat16 on TPU, AMP on GPU
  - Gradient accumulation for large effective batch sizes
  - Gradient clipping with norm tracking
  - EMA (exponential moving average) of model weights
  - Cosine LR schedule with warmup
  - Checkpointing (latest + best by val loss)
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


import importlib.util

# ── TPU Detection ────────────────────────────────────────────────────────────
USE_TPU = importlib.util.find_spec("torch_xla") is not None

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
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (
        config.lr_decay_iters - config.warmup_iters
    )
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
    if USE_TPU:
        try:
            import torch_xla.runtime as xr
            return xr.global_ordinal() == 0
        except (ImportError, AttributeError):
            import torch_xla.core.xla_model as xm
            return xm.is_master_ordinal(local=False)
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

    def log_step(self, step, total, loss, lr, grad_norm, tok_sec, vram_alloc, eta_str, tokens=0):
        line = (
            f"[{self._ts()}] STEP {step:>7d}/{total} | Tokens: {tokens:,} | "
            f"loss={loss:.4f} | lr={lr:.2e} | grad_norm={grad_norm:.3f} | "
            f"tok/s={tok_sec:,.0f} | VRAM={vram_alloc:.1f}GB | ETA={eta_str}"
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
        self.write(f"{hr}")
        self.write(f"  Model params    : {n_params:,}")
        for k, v in asdict(config).items():
            self.write(f"  {k:<28s}: {v}")
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
        self.use_tpu = (config.device == "xla")

        # Set device
        if self.use_tpu:
            import torch_xla
            self.device = torch_xla.device()
            self.is_master = _is_master()
        elif is_ddp:
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.device = torch.device(f"cuda:{ddp_local_rank}")
            self.is_master = (int(os.environ['RANK']) == 0)
        else:
            self.device = torch.device(config.device)
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
                print("Compiling model.forward (regional)...")
            self.model.forward = torch.compile(self.model.forward, mode="default")

        # Multi-device wrapping (GPU only; TPU handles multi-core via xm.optimizer_step)
        self.is_ddp = is_ddp
        if self.is_ddp and not self.use_tpu:
            if self.is_master:
                print(f"Wrapping model with DistributedDataParallel (DDP).")
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.model = DDP(self.model, device_ids=[ddp_local_rank])

        self.optimizer = (self.model.module if (self.is_ddp and not self.use_tpu) else self.model).configure_optimizers(config)

        self.ema = (
            EMA(self.model.module if (self.is_ddp and not self.use_tpu) else self.model, decay=config.ema_decay) if config.use_ema else None
        )

        # GradScaler only needed for float16 on GPU
        self.use_scaler = (not self.use_tpu and config.dtype == "float16")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_scaler
        ) if not self.use_tpu else None

        if self.use_tpu:
            import torch_xla.distributed.parallel_loader as pl
            self.train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
            self.val_device_loader = pl.MpDeviceLoader(val_loader, self.device)
            self.train_iter = iter(self.train_device_loader)
            self.val_iter = iter(self.val_device_loader)
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.train_iter = iter(self.train_loader)
            self.val_iter = iter(self.val_loader)

        self.iter_num = 0
        self.best_val_loss = float("inf")
        self.micro_step = 0

        # Metrics accumulators
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0
        self._tokens_processed = 0
        self._last_log_time = None

    def get_batch(self, split):
        if split == "train":
            try:
                x, y = next(self.train_iter)
            except StopIteration:
                if self.use_tpu:
                    self.train_iter = iter(self.train_device_loader)
                else:
                    self.train_iter = iter(self.train_loader)
                x, y = next(self.train_iter)
        else:
            try:
                x, y = next(self.val_iter)
            except StopIteration:
                if self.use_tpu:
                    self.val_iter = iter(self.val_device_loader)
                else:
                    self.val_iter = iter(self.val_loader)
                x, y = next(self.val_iter)

        if self.use_tpu:
            return x, y
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    @torch.no_grad()
    def estimate_loss(self):
        if getattr(self, "use_tpu", False):
            import torch_xla.core.xla_model as xm
        out = {}
        self.model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(self.config.eval_iters, device=self.device)
            for k in range(self.config.eval_iters):
                x, y = self.get_batch(split)
                if self.use_tpu:
                    # On TPU, bfloat16 is handled natively via XLA_USE_BF16
                    logits, _ = self.model(x)
                else:
                    with torch.amp.autocast(
                        "cuda",
                        dtype=_DTYPE_MAP.get(self.config.dtype, torch.float16),
                        enabled=self.config.dtype != "float32",
                    ):
                        logits, _ = self.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
                losses[k] = loss
                if self.use_tpu:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save_checkpoint(self, path, step_num=None, max_ckpt=3, epoch_name=None):
        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        ckpt = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if getattr(self, "ema", None) is not None:
            ckpt["ema"] = self.ema.state_dict()
        if getattr(self, "use_scaler", False) and getattr(self, "scaler", None) is not None:
            ckpt["scaler"] = self.scaler.state_dict()

        if getattr(self, "use_tpu", False):
            import torch_xla.core.xla_model as xm
            xm.save(ckpt, path, master_only=True, global_master=True)
        else:
            if self.is_master:
                torch.save(ckpt, path)

        if not self.is_master:
            return

        # Local cleanup
        if step_num is not None:
            try:
                import re
                ckpt_dir = os.path.dirname(path)
                files = os.listdir(ckpt_dir)
                ckpt_files = [f for f in files if re.match(r"checkpoint-\d+\.pt", f)]
                ckpt_files.sort(key=lambda x: int(re.search(r"checkpoint-(\d+)\.pt", x).group(1)))
                if len(ckpt_files) > max_ckpt:
                    to_delete = ckpt_files[:-max_ckpt]
                    for f in to_delete:
                        os.remove(os.path.join(ckpt_dir, f))
            except Exception as e:
                pass

        # Background HuggingFace sync
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            def background_sync():
                try:
                    import warnings
                    from huggingface_hub.utils import disable_progress_bars
                    disable_progress_bars()
                    # Suppress HF warnings temporarily
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
                        import shutil
                        import uuid
                        import re
                        
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
                                        
                                        # Squash history to permanently delete invisible LFS objects from HF backend
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
        # Load on CPU first to avoid a GPU memory spike (checkpoint + model +
        # EMA + optimizer state would otherwise all be materialised on-device).
        # load_state_dict copies to the device incrementally.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        base_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.iter_num = ckpt.get("iter_num", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        # Backwards compatibility for checkpoints saved with the old off-by-one bug
        if self.iter_num > 0 and self.iter_num % self.config.save_interval == 0:
            self.iter_num += 1

        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
            self.ema.shadow = {
                k: v.to(self.device) for k, v in self.ema.shadow.items()
            }
        if self.use_scaler and self.scaler is not None and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        del ckpt
        if not self.use_tpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_samples(self):
        if not self.is_master or self.use_tpu:
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
        if self.use_tpu:
            try:
                import torch_xla.runtime as xr
                tps *= xr.world_size()
            except (ImportError, AttributeError):
                tps *= xm.xrt_world_size()
        elif self.is_ddp:
            tps *= int(os.environ.get('WORLD_SIZE', 1))
        return tps

    def train(self):
        config = self.config
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler

        if getattr(self, "use_tpu", False):
            import torch_xla.core.xla_model as xm

        # ── Log config at training start ─────────────────────────────────
        if self.iter_num == 0 and self.is_master and self.flog:
            self.flog.log_config(config, self.n_params)

        if not self.use_tpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.train()
        # On TPU, use a device tensor to avoid float+tensor graph shape changes
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

            if self.use_tpu:
                # On TPU, bfloat16 is handled natively via XLA_USE_BF16
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                last_loss = loss.detach()
                del logits, loss
                # Keep graph bounded to 1 microstep to prevent CPU RAM OOM during compilation
                if config.gradient_accumulation_steps > 1 and (self.micro_step + 1) % config.gradient_accumulation_steps != 0:
                    xm.mark_step()
            else:
                # DDP no_sync: skip all-reduce on all micro-steps except the last
                is_last_micro = (self.micro_step + 1) % config.gradient_accumulation_steps == 0
                ctx = contextlib.nullcontext() if (not self.is_ddp or is_last_micro) else model.no_sync()
                with ctx:
                    with torch.amp.autocast(
                        "cuda",
                        dtype=_DTYPE_MAP.get(config.dtype, torch.float16),
                        enabled=config.dtype != "float32",
                    ):
                        logits, _ = model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y.view(-1),
                        ) / config.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                
                # Save loss before deleting to avoid UnboundLocalError
                last_loss = loss.detach()

                # Free huge activations (256MB) immediately before the next micro-step
                del logits, loss

            self.micro_step += 1
            if pbar:
                micro = self.micro_step % config.gradient_accumulation_steps
                if micro == 0: micro = config.gradient_accumulation_steps
                pbar.set_description(f"Training (Micro {micro}/{config.gradient_accumulation_steps})")

            if self.micro_step % config.gradient_accumulation_steps == 0:
                # ── Gradient clipping + norm tracking ────────────────────
                grad_norm = 0.0
                if self.use_tpu:
                    # TPU: 1. Synchronize gradients across all 8 cores
                    xm.reduce_gradients(optimizer)
                    # 2. Clip gradients on synchronized grads
                    if config.grad_clip > 0.0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_clip
                        )
                    # 3. Step optimizer on synchronized gradients
                    optimizer.step()
                    # 4. Mark step boundary
                    xm.mark_step()
                else:
                    if config.grad_clip > 0.0:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_clip
                        ).item()
                    else:
                        # Still compute norm for logging even without clipping
                        total_norm_sq = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                total_norm_sq += p.grad.data.float().norm().item() ** 2
                        grad_norm = total_norm_sq ** 0.5

                    scaler.step(optimizer)
                    scaler.update()

                if self.use_tpu:
                    optimizer.zero_grad(set_to_none=False)
                else:
                    optimizer.zero_grad(set_to_none=True)

                if self.use_tpu:
                    # On TPU, do not accumulate to prevent memory leaks, just keep the latest
                    self._grad_norm_sum = grad_norm
                else:
                    self._grad_norm_sum += grad_norm
                self._grad_norm_count += 1

                if self.ema is not None:
                    self.ema.update()

                if self.use_tpu:
                    step_loss = (last_loss.item() if torch.is_tensor(last_loss) else last_loss) * config.gradient_accumulation_steps
                    running_loss += step_loss
                    self._grad_norm_sum += (grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                else:
                    step_loss = last_loss.item() * config.gradient_accumulation_steps
                    running_loss += step_loss
                    self._grad_norm_sum += grad_norm
                self._grad_norm_count += 1
                self._tokens_processed += tokens_per_step

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

                        vram_alloc, vram_total = _vram_gb()
                        steps_remaining = config.max_iters - self.iter_num
                        sec_per_step = elapsed / max(self._steps_taken_since_resume, 1)
                        eta = steps_remaining * sec_per_step
                        eta_str = _format_eta(eta)

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
                                avg_gn, tok_sec, vram_alloc, eta_str,
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
                    now = time.time()
                    sec_per_step = elapsed / max(self._steps_taken_since_resume, 1)
                    steps_remaining = config.max_iters - self.iter_num
                    eta = steps_remaining * sec_per_step
                    vram_alloc, vram_total = _vram_gb()
                    if self.use_tpu:
                        avg_gn = self._grad_norm_sum.item() if torch.is_tensor(self._grad_norm_sum) else self._grad_norm_sum
                    else:
                        avg_gn = self._grad_norm_sum / max(self._grad_norm_count, 1)
                    tok_sec = tokens_per_step / max(sec_per_step, 1e-6)

                    is_best = val_loss < self.best_val_loss

                    if self.is_master:
                        # Terminal — structured eval block
                        hr = "═" * 56
                        pct = self.iter_num / config.max_iters * 100
                        delta_val = val_loss - self.best_val_loss if self.best_val_loss < float("inf") else 0.0
                        delta_str = f"Δ: {delta_val:+.4f}" if not is_best else "NEW BEST ★"

                        # Terminal — structured eval block
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

                        # File log — structured eval block
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

                        # Update tqdm postfix
                        if pbar:
                            pbar.set_postfix({
                                "train": f"{train_loss:.4f}",
                                "val": f"{val_loss:.4f}",
                                "ppl": f"{ppl:.1f}",
                                "lr": f"{lr:.2e}",
                            })

                    # Save checkpoint (master only — handled inside save_checkpoint)
                    ckpt_path = f"checkpoints/checkpoint-{self.iter_num + 1:06d}.pt"
                    if is_best:
                        self.best_val_loss = val_loss
                    self.save_checkpoint(ckpt_path, step_num=self.iter_num + 1)

                    # Defrag memory after eval's memory spike
                    if not self.use_tpu and torch.cuda.is_available():
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
