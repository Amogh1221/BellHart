import os
import sys
import json
import torch
import argparse

# Clear Kaggle environment variables that conflict with PJRT
os.environ.pop('TPU_PROCESS_ADDRESSES', None)
os.environ.pop('CLOUD_TPU_TASK_ID', None)

from huggingface_hub import login, hf_hub_download

# ── TPU Detection ────────────────────────────────────────────────────────────
USE_TPU = False
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    USE_TPU = True
except ImportError:
    pass

if not USE_TPU:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

from config import GPTConfig
from tokenizer import Tokenizer
from trainer import Trainer
from dataset import create_streaming_dataloaders


def sync_huggingface(repo_id: str):
    print("Syncing dataset and tokenizer from HuggingFace...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("tokenizer", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    # Download dataset and tokenizer
    # (Tokenizer is now checked into git as tokenizer.json)
    print("Checking for existing checkpoints and logs...")
    try:
        if not os.path.exists("checkpoints/latest.pt"):
            hf_hub_download(repo_id=repo_id, filename="checkpoints/latest.pt", repo_type="dataset", local_dir=".")
            print("Successfully downloaded latest.pt")
        else:
            print("Found checkpoints/latest.pt locally, skipping download.")
    except Exception as e:
        print("No existing checkpoint found on HuggingFace.")
        
    try:
        hf_hub_download(repo_id=repo_id, filename="logs/training_log.txt", repo_type="dataset", local_dir=".")
        print("Successfully downloaded training_log.txt")
    except Exception as e:
        print("No existing training log found on HuggingFace.")
        
    # Disable HF progress bars for background uploads during training
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def setup_environment(config: GPTConfig):
    if config.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.tf32
        torch.backends.cudnn.allow_tf32 = config.tf32
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB)")
        print(f"TF32 enabled: {config.tf32}")
        print(f"AMP dtype: {config.dtype}")
    elif config.device == "xla":
        import torch_xla.core.xla_model as xm
        try:
            import torch_xla.runtime as xr
            is_master = xr.global_ordinal() == 0
        except (ImportError, AttributeError):
            is_master = xm.get_ordinal() == 0
        if is_master:
            import torch_xla
            print(f"TPU: {torch_xla.device()}")
            try:
                import torch_xla.runtime as xr
                cores = xr.world_size()
            except (ImportError, AttributeError):
                cores = xm.xrt_world_size()
            print(f"TPU cores: {cores}")
            print(f"Dtype: bfloat16 (native TPU)")


def _train_worker(index=None, hf_token=None):
    """
    Core training function. Runs once on GPU, or is spawned per-core on TPU.
    """
    is_ddp = int(os.environ.get('RANK', -1)) != -1
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(ddp_local_rank)
        is_master = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        is_master = True
        seed_offset = 0

    if USE_TPU:
        try:
            import torch_xla.runtime as xr
            is_master = xr.global_ordinal() == 0
            seed_offset = xr.global_ordinal()
        except (ImportError, AttributeError):
            is_master = xm.is_master_ordinal(local=False)
            seed_offset = xm.get_ordinal()
        # Set BF16 natively on TPU
        os.environ["XLA_USE_BF16"] = "1"

    if hf_token:
        login(token=hf_token)
        os.environ["HF_TOKEN"] = hf_token

    config_path = "config.json"
    if os.path.exists(config_path):
        if is_master:
            print(f"Loading config from {config_path}")
        with open(config_path) as f:
            config = GPTConfig.load(config_path)
    else:
        config = GPTConfig()
        config.save("config.json")
        if is_master:
            print(f"Created default config at {config_path}")

    repo_id = config.hf_repo

    # Only master downloads data (avoid 8 concurrent downloads)
    if is_master:
        sync_huggingface(repo_id)
    
    # On TPU, wait for master to finish downloading
    if USE_TPU:
        xm.rendezvous("data_download")


    # ── Auto-detect device and adjust config ─────────────────────────────
    if USE_TPU:
        config.device = "xla"
        config.dtype = "bfloat16"
        config.compile = False
        config.fused_adam = False
        # TPU v3 has 15.75G HBM per core — batch_size=2 overflows by ~362MB.
        # Halve batch_size to 1 and compensate with higher grad_accum.
        original_effective_batch = config.batch_size * config.gradient_accumulation_steps
        config.batch_size = 1
        
        # ── OOM Prevention: Force memmap on TPU ──
        # xmp.spawn launches 8 processes. If preload=True, the 8GB dataset is
        # loaded 8 times into CPU RAM (64GB total), instantly crashing Kaggle.
        config.preload = False
        
        try:
            import torch_xla.runtime as xr
            num_cores = xr.world_size()
        except (ImportError, AttributeError):
            num_cores = xm.xrt_world_size()
        # Recalculate grad_accum: effective = batch_size * num_cores * grad_accum
        new_grad_accum = max(1, original_effective_batch // (config.batch_size * num_cores))
        config.gradient_accumulation_steps = new_grad_accum
        if is_master:
            print(f"TPU detected ({num_cores} cores)")
            print(f"batch_size: {config.batch_size}  grad_accum: {config.gradient_accumulation_steps}  "
                  f"(effective batch = {config.batch_size * num_cores * new_grad_accum})")
            print(f"preload forced to False to prevent CPU OOM")
    elif torch.cuda.is_available():
        # ── Dynamic GPU VRAM auto-scaling ─────────────────────────────────
        vram_gb = torch.cuda.get_device_properties(ddp_local_rank).total_memory / 1e9
        
        if is_ddp:
            # If DDP, original_effective_batch is divided among world_size
            original_effective_batch = config.batch_size * config.gradient_accumulation_steps
        else:
            original_effective_batch = config.batch_size * config.gradient_accumulation_steps

        # Scale batch_size based on available VRAM
        if vram_gb >= 70:       # H100 80GB / A100 80GB
            new_batch = 16
        elif vram_gb >= 35:     # A100 40GB
            new_batch = 8
        elif vram_gb >= 20:     # RTX 3090/4090 24GB
            new_batch = 4
        else:                   # T4 16GB / RTX 3060 etc.
            new_batch = 2
            config.use_8bit_optimizer = True
            config.gradient_checkpointing = 1

        # Scale eval_iters inversely so validation takes the same time
        base_eval_tokens = 200 * 2  # original: 200 iters * batch_size 2
        config.eval_iters = max(10, base_eval_tokens // new_batch)

        if is_ddp:
            # Adjust grad_accum so the global batch size matches original_effective_batch
            # Global batch = new_batch * ddp_world_size * new_grad_accum
            target_accum = max(1, original_effective_batch // (new_batch * ddp_world_size))
            config.gradient_accumulation_steps = target_accum
            config.batch_size = new_batch
        else:
            config.batch_size = new_batch
            config.gradient_accumulation_steps = max(1, original_effective_batch // new_batch)

        # Enable hardware optimizations for Ampere+ GPUs (A100/H100/RTX 30xx+)
        gpu_name = torch.cuda.get_device_name(ddp_local_rank).upper()
        is_ampere_plus = vram_gb >= 20 or any(tag in gpu_name for tag in ["A100", "H100", "H200", "RTX 30", "RTX 40", "RTX 50"])

        if is_ampere_plus:
            config.dtype = "bfloat16"
            config.tf32 = True
        else:
            config.dtype = "float16"
            config.tf32 = False

        if is_master:
            print(f"Auto-scaled for {vram_gb:.0f}GB VRAM → "
                  f"batch_size={config.batch_size}, "
                  f"grad_accum={config.gradient_accumulation_steps}, "
                  f"eval_iters={config.eval_iters}, "
                  f"dtype={config.dtype}, tf32={config.tf32}")
    else:
        config.device = "cpu"
        print("WARNING: No GPU or TPU found, falling back to CPU")

    setup_environment(config)

    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size

    train_loader, val_loader = create_streaming_dataloaders(
        dataset_name="openbmb/Ultra-FineWeb-L1",
        dataset_config="CC-MAIN-2025-30",
        tokenizer=tokenizer,
        block_size=config.block_size,
        batch_size=config.batch_size,
        num_workers=1,
        seed=42 + seed_offset,
    )

    trainer = Trainer(config, tokenizer, train_loader, val_loader, is_ddp=is_ddp)

    import glob
    checkpoints = glob.glob("checkpoints/checkpoint-*.pt")
    if checkpoints:
        # Sort by step number
        resume_path = sorted(checkpoints, key=lambda x: int(x.split('-')[-1].split('.')[0]))[-1]
        trainer.load_checkpoint(resume_path)

    try:
        trainer.train()
    except KeyboardInterrupt:
        if is_master:
            print("\nInterrupted, saving checkpoint...")
            ckpt_path = f"checkpoints/checkpoint-{trainer.iter_num:06d}.pt"
            trainer.save_checkpoint(ckpt_path, step_num=trainer.iter_num)
            print(f"Checkpoint saved to {ckpt_path}. Exiting.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace WRITE Token")
    args = parser.parse_args()

    if USE_TPU:
        print("=" * 56)
        print("  BellHart — TPU Training Mode")
        print("=" * 56)
        # xmp.spawn launches _train_worker on all available TPU cores (1, 4, or 8)
        xmp.spawn(_train_worker, args=(args.hf_token,), nprocs=None, start_method="fork")
    else:
        print("=" * 56)
        print("  BellHart — GPU Training Mode")
        print("=" * 56)
        
        is_ddp = int(os.environ.get('RANK', -1)) != -1
        if is_ddp:
            if int(os.environ.get('RANK', 0)) == 0:
                print(f"Authenticating with HuggingFace (DDP Master)...")
        else:
            print(f"Authenticating with HuggingFace...")
            
        _train_worker(index=None, hf_token=args.hf_token)

    if int(os.environ.get('RANK', -1)) != -1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
