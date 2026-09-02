"""
train.py  —  BellHart GPU Training Entrypoint (Single GPU & Distributed DDP)
=============================================================================
Main execution script for launching BellHart pre-training.

Key Responsibilities:
  1. Hardware Auto-Detection & Scaling — Detects available VRAM (H100, A100, L4, RTX, T4)
     and automatically adjusts batch sizes, gradient accumulation, precision, and optimizer.
  2. Multi-GPU DistributedDataParallel (DDP) — Configures process groups, rank assignments,
     and GPU device bindings for multi-GPU training nodes.
  3. HuggingFace Cloud Synchronization — Automatically downloads latest remote checkpoints
     and logs before training begins.
  4. Memory-Safe Process Management — Enforces clean subprocess limits and garbage collection
     to prevent Linux OOM-killer termination on shared-memory platforms.
"""

import os
import sys
import json
import torch
import argparse
import warnings
import logging
import re

# ── Clean Terminal & Logging Environment Setup ───────────────────────────────
# Suppress benign framework warnings across all distributed worker processes
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"
os.environ["DATASETS_STREAMING_READ_MAX_BATCH_SIZE"] = "50"
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
torch.set_num_threads(2)

# Prevent CUDA memory fragmentation via expandable memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from huggingface_hub import login, hf_hub_download
from config import GPTConfig
from tokenizer import Tokenizer
from trainer import Trainer
from dataset import create_streaming_dataloaders


def _is_proc_master() -> bool:
    """Check whether current process is Rank 0 before full initialization."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    return rank == 0


def sync_huggingface(repo_id: str):
    """
    Downloads the latest available checkpoint and training logs from the Hugging Face repository.
    Ensures seamless continuity when training across multiple ephemeral cloud sessions.
    """
    if _is_proc_master():
        print("Syncing dataset and tokenizer from HuggingFace...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("tokenizer", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    if _is_proc_master():
        print("Checking for existing checkpoints and logs...")
    try:
        from huggingface_hub import HfApi
        hf_token = os.environ.get("HF_TOKEN")
        api = HfApi(token=hf_token)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        # Scan for numbered checkpoint files: checkpoint-XXXXXX.pt
        ckpt_files = [f for f in files if re.match(r"^checkpoints/checkpoint-\d+\.pt$", f)]
        if ckpt_files:
            ckpt_files.sort(
                key=lambda x: int(re.search(r"checkpoint-(\d+)\.pt", x).group(1)),
                reverse=True
            )
            latest_ckpt_file = ckpt_files[0]
            if not os.path.exists(latest_ckpt_file):
                if _is_proc_master():
                    print(f"Downloading latest checkpoint {latest_ckpt_file} from HuggingFace...")
                hf_hub_download(repo_id=repo_id, filename=latest_ckpt_file, repo_type="dataset", local_dir=".")
                if _is_proc_master():
                    print(f"Successfully downloaded {latest_ckpt_file}")
            else:
                if _is_proc_master():
                    print(f"Found {latest_ckpt_file} locally, skipping download.")
        elif "checkpoints/latest.pt" in files:
            if not os.path.exists("checkpoints/latest.pt"):
                if _is_proc_master():
                    print("Downloading checkpoints/latest.pt from HuggingFace...")
                hf_hub_download(repo_id=repo_id, filename="checkpoints/latest.pt", repo_type="dataset", local_dir=".")
                if _is_proc_master():
                    print("Successfully downloaded latest.pt")
            else:
                if _is_proc_master():
                    print("Found checkpoints/latest.pt locally, skipping download.")
        else:
            if _is_proc_master():
                print("No existing checkpoint found on HuggingFace.")
    except Exception as e:
        if _is_proc_master():
            print(f"No existing checkpoint found on HuggingFace ({e}).")
        
    # Download persistent training log
    try:
        hf_hub_download(repo_id=repo_id, filename="logs/training_log.txt", repo_type="dataset", local_dir=".")
        if _is_proc_master():
            print("Successfully downloaded training_log.txt")
    except Exception:
        if _is_proc_master():
            print("No existing training log found on HuggingFace.")
        
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def setup_environment(config: GPTConfig):
    """Configures CUDA backend settings, TF32 execution, and cuDNN autotuning."""
    if config.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = config.tf32
        torch.backends.cudnn.allow_tf32 = config.tf32
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB)")
        print(f"TF32 enabled: {config.tf32}")
        print(f"AMP dtype: {config.dtype}")


def _train_worker(hf_token: str = "", fresh: bool = False):
    """
    Main training execution function. Runs per-process on single-GPU or across DDP ranks.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # Parse Hugging Face authentication token and run flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace WRITE Token")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh from Step 0, ignoring existing checkpoints")
    args, _ = parser.parse_known_args()
    if not hf_token:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if args.fresh:
        fresh = True

    # DistributedDataParallel (DDP) detection
    is_ddp = int(os.environ.get('RANK', -1)) != -1
    
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(ddp_local_rank)
        is_master = (ddp_rank == 0)
        seed_offset = ddp_rank
        global_rank = ddp_rank
        world_size = ddp_world_size
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        is_master = True
        seed_offset = 0
        global_rank = 0
        world_size = 1

    if hf_token:
        try:
            login(token=hf_token)
            os.environ["HF_TOKEN"] = hf_token
            if is_master:
                print("Authenticated with Hugging Face successfully.")
        except Exception as e:
            if is_master:
                print(f"\n[WARNING] Hugging Face authentication failed: {e}")
                print("The provided HF_TOKEN is invalid or expired.")
                print("Training will continue locally, but checkpoints will NOT be synced to Hugging Face.\n")
            os.environ["HF_TOKEN"] = ""
            hf_token = ""

    # Configuration loading and synchronization
    config_path = "config.json"
    if is_master:
        if os.path.exists(config_path):
            print(f"Loading config from {config_path}")
            config = GPTConfig.load(config_path)
        else:
            config = GPTConfig()
            config.save("config.json")
            print(f"Created default config at {config_path}")
    else:
        config = GPTConfig()

    # Synchronize configuration file across distributed ranks
    if is_ddp:
        import torch.distributed as dist
        dist.barrier()
        if not is_master and os.path.exists(config_path):
            config = GPTConfig.load(config_path)

    repo_id = config.hf_repo

    # Master downloads latest checkpoints unless --fresh is specified
    if is_master and not fresh:
        sync_huggingface(repo_id)
    elif is_master and fresh:
        print("\n" + "═" * 60)
        print("  [FRESH RUN] Starting brand new training from Step 0!")
        print("  Skipping checkpoint download from Hugging Face.")
        print("═" * 60 + "\n")

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()

    # ── Dynamic GPU VRAM Hardware Auto-Scaling ───────────────────────────
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(ddp_local_rank).total_memory / 1e9
        original_effective_batch = config.batch_size * config.gradient_accumulation_steps

        # Standardize pre-training context length to 2048 tokens
        config.block_size = 2048
        config.eval_interval = 500  # Evaluate every 500th step

        # Tiered scaling based on hardware capacity (8-bit AdamW used everywhere for 100% checkpoint portability)
        if vram_gb >= 140:      # NVIDIA B200 192GB / B300 / H200 141GB (Blackwell / Hopper Max)
            new_batch = 16      # 192GB VRAM: cuts accumulation steps to 2, increasing throughput
            config.compile = False
            config.use_8bit_optimizer = True
            config.gradient_checkpointing = 0  # 192GB VRAM easily fits activations -> 30% faster backprop
            config.save_interval = 900         # Checkpoint every ~20 minutes at B200 speed
        elif vram_gb >= 70:     # NVIDIA H100 80GB / A100 80GB
            new_batch = 8
            config.compile = False
            config.use_8bit_optimizer = True
            config.gradient_checkpointing = 1
            config.save_interval = 400
        elif vram_gb >= 35:     # NVIDIA A100 40GB
            new_batch = 4
            config.compile = False
            config.use_8bit_optimizer = True
            config.gradient_checkpointing = 1
            config.save_interval = 400
        elif vram_gb >= 20:     # NVIDIA L4 / RTX 3090 / RTX 4090 (24GB)
            new_batch = 2
            config.compile = False
            config.use_8bit_optimizer = True
            config.gradient_checkpointing = 1
            config.save_interval = 25
        else:                   # NVIDIA T4 16GB / RTX 3060
            new_batch = 1
            config.compile = False
            config.use_8bit_optimizer = True
            config.gradient_checkpointing = 1
            config.save_interval = 25

        base_eval_batches = 20
        config.eval_iters = max(5, base_eval_batches // new_batch)

        # Re-compute gradient accumulation to maintain constant global batch size across DDP ranks
        if is_ddp:
            target_accum = max(1, original_effective_batch // (new_batch * ddp_world_size))
            config.gradient_accumulation_steps = target_accum
            config.batch_size = new_batch
        else:
            config.batch_size = new_batch
            config.gradient_accumulation_steps = max(1, original_effective_batch // new_batch)

        # Enable Ampere+ hardware acceleration (TF32 and native bfloat16)
        gpu_name = torch.cuda.get_device_name(ddp_local_rank).upper()
        is_ampere_plus = vram_gb >= 20 or any(tag in gpu_name for tag in ["A100", "H100", "H200", "B200", "B300", "RTX 30", "RTX 40", "RTX 50"])

        if is_ampere_plus:
            config.dtype = "bfloat16"
            config.tf32 = True
        else:
            config.dtype = "float16"
            config.tf32 = False

        # Dynamic streaming shuffle buffer:
        # B200 / H100 with massive host RAM use 10,000 document shuffle buffer
        # to ensure diverse batches on single GPU without cluster bias.
        if vram_gb >= 140:
            stream_buffer_size = 10000
        elif vram_gb >= 70:
            stream_buffer_size = 5000
        elif vram_gb >= 35:
            stream_buffer_size = 2500
        else:
            stream_buffer_size = 1  # Low memory / DDP multi-shard

        if is_master:
            print(f"Auto-scaled for {vram_gb:.0f}GB VRAM → "
                  f"block_size={config.block_size}, "
                  f"batch_size={config.batch_size}, "
                  f"grad_accum={config.gradient_accumulation_steps}, "
                  f"eval_iters={config.eval_iters}, "
                  f"stream_buffer={stream_buffer_size}, "
                  f"dtype={config.dtype}, tf32={config.tf32}")
    else:
        stream_buffer_size = 1
        config.device = "cpu"
        print("WARNING: No GPU found, falling back to CPU")

    setup_environment(config)

    # Initialize Tokenizer and sync vocabulary size
    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size

    # Create dynamic streaming dataloaders
    use_pin_memory = (config.device == "cuda" and torch.cuda.is_available())
    train_loader, val_loader = create_streaming_dataloaders(
        dataset_name="openbmb/Ultra-FineWeb-L1",
        dataset_config="CC-MAIN-2025-30",
        tokenizer=tokenizer,
        block_size=config.block_size,
        batch_size=config.batch_size,
        buffer_size=stream_buffer_size,
        num_workers=0,
        pin_memory=use_pin_memory,
        seed=42 + seed_offset,
        rank=global_rank,
        world_size=world_size,
    )

    # Initialize Trainer pipeline
    trainer = Trainer(config, tokenizer, train_loader, val_loader, is_ddp=is_ddp)

    # Check for local checkpoints to resume (bypassed if --fresh is set)
    if not fresh:
        import glob
        checkpoints = glob.glob("checkpoints/checkpoint-*.pt")
        valid_checkpoints = [f for f in checkpoints if re.search(r"checkpoint-(\d+)\.pt", os.path.basename(f))]
        if valid_checkpoints:
            resume_path = sorted(
                valid_checkpoints,
                key=lambda x: int(re.search(r"checkpoint-(\d+)\.pt", os.path.basename(x)).group(1))
            )[-1]
            if is_master:
                print(f"Resuming training from checkpoint: {resume_path}")
            trainer.load_checkpoint(resume_path)
        elif os.path.exists("checkpoints/latest.pt"):
            if is_master:
                print("Resuming training from checkpoint: checkpoints/latest.pt")
            trainer.load_checkpoint("checkpoints/latest.pt")

    # Reclaim host RAM before launching training loop
    import gc
    gc.collect()

    # Launch training
    try:
        trainer.train()
    except KeyboardInterrupt:
        if is_master:
            print("\nInterrupted by user. Saving emergency checkpoint...")
            ckpt_path = f"checkpoints/checkpoint-{trainer.iter_num:06d}.pt"
            trainer.save_checkpoint(ckpt_path, step_num=trainer.iter_num)
            print(f"Checkpoint saved to {ckpt_path}. Exiting.")
        sys.exit(0)
    except Exception as e:
        import traceback
        rank_str = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        print(f"\n[CRITICAL ERROR on Rank {rank_str}] {e}\n", flush=True)
        traceback.print_exc()
        sys.exit(1)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="BellHart Pre-Training Pipeline")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace WRITE Token")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh from Step 0, ignoring existing checkpoints")
    args, _ = parser.parse_known_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    _train_worker(hf_token=hf_token, fresh=args.fresh)


if __name__ == "__main__":
    main()
