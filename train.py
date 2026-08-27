import os
import sys
import json
import torch
import argparse
import warnings
import logging

# ── Clean Terminal Setup ─────────────────────────────────────────────────────
# Suppress all python warnings across all processes
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress PyTorch/XLA/TensorFlow/Accelerate C++ and distributed logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["XLA_PERSISTENT_CACHE_PATH"] = "/tmp/xla_cache"
os.environ["LIBTPU_PERSISTENT_CACHE_PATH"] = "/tmp/xla_cache"
os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
torch.set_num_threads(2)

# Clear Kaggle environment variables that conflict with PJRT
os.environ.pop('TPU_PROCESS_ADDRESSES', None)
os.environ.pop('CLOUD_TPU_TASK_ID', None)
os.environ['PYTHONFAULTHANDLER'] = '1'

from huggingface_hub import login, hf_hub_download
import importlib.util

# ── TPU Detection ────────────────────────────────────────────────────────────
USE_TPU = importlib.util.find_spec("torch_xla") is not None

if not USE_TPU:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

from config import GPTConfig
from tokenizer import Tokenizer
from trainer import Trainer
from dataset import create_streaming_dataloaders


def _is_proc_master() -> bool:
    """Helper to check if this is the master process before full initialization."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    return rank == 0


def sync_huggingface(repo_id: str):
    if _is_proc_master():
        print("Syncing dataset and tokenizer from HuggingFace...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("tokenizer", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    # Download dataset and tokenizer
    if _is_proc_master():
        print("Checking for existing checkpoints and logs...")
    try:
        import re
        from huggingface_hub import HfApi
        hf_token = os.environ.get("HF_TOKEN")
        api = HfApi(token=hf_token)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

        # Find all checkpoint files on HF
        ckpt_files = [f for f in files if re.match(r"^checkpoints/checkpoint-\d+\.pt$", f)]
        if ckpt_files:
            # Sort by step number descending
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
        
    try:
        hf_hub_download(repo_id=repo_id, filename="logs/training_log.txt", repo_type="dataset", local_dir=".")
        if _is_proc_master():
            print("Successfully downloaded training_log.txt")
    except Exception:
        if _is_proc_master():
            print("No existing training log found on HuggingFace.")
        
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def setup_environment(config: GPTConfig):
    if config.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.tf32
        torch.backends.cudnn.allow_tf32 = config.tf32
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels for fixed input sizes
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
            print(f"TPU: {xm.xla_device()}")
            try:
                import torch_xla.runtime as xr
                cores = xr.world_size()
            except (ImportError, AttributeError):
                cores = xm.xrt_world_size()
            print(f"TPU cores: {cores}")
            print(f"Dtype: bfloat16 (native TPU)")


def _train_worker(index_or_token=None, hf_token=None):
    """
    Core training function. Runs once on GPU, or is spawned per-core on TPU.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if hf_token is None and isinstance(index_or_token, str):
        hf_token = index_or_token
        index = None
    else:
        index = index_or_token

    if not hf_token:
        parser = argparse.ArgumentParser()
        parser.add_argument("--hf_token", type=str, default="", help="HuggingFace WRITE Token")
        args, _ = parser.parse_known_args()
        hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    is_ddp = int(os.environ.get('RANK', -1)) != -1
    
    if USE_TPU:
        import torch_xla.core.xla_model as xm
        if is_ddp:
            import torch.distributed as dist
            import torch_xla.distributed.xla_backend
            if not dist.is_initialized():
                dist.init_process_group("xla", init_method="xla://")
            global_rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            is_master = global_rank == 0
            seed_offset = global_rank
        else:
            try:
                import torch_xla.runtime as xr
                is_master = xr.global_ordinal() == 0
                seed_offset = xr.global_ordinal()
                global_rank = xr.global_ordinal()
                world_size = xr.world_size()
            except (ImportError, AttributeError):
                is_master = xm.is_master_ordinal(local=False)
                seed_offset = xm.get_ordinal()
                global_rank = xm.get_ordinal()
                world_size = xm.xrt_world_size()
        # Set BF16 natively on TPU
        os.environ["XLA_USE_BF16"] = "1"
        
        ddp_rank = global_rank
        ddp_local_rank = global_rank % 8
        ddp_world_size = world_size
    elif is_ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(ddp_local_rank)
        is_master = ddp_rank == 0
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
        login(token=hf_token)
        os.environ["HF_TOKEN"] = hf_token

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

    # Synchronize config creation across DDP workers
    if is_ddp and not USE_TPU:
        import torch.distributed as dist
        dist.barrier()
        if not is_master and os.path.exists(config_path):
            config = GPTConfig.load(config_path)

    repo_id = config.hf_repo

    # Only master downloads data (avoid concurrent downloads)
    if is_master:
        sync_huggingface(repo_id)

    # Wait for master to finish downloading checkpoints and dataset files
    if is_ddp and not USE_TPU:
        import torch.distributed as dist
        dist.barrier()


    # ── Auto-detect device and adjust config ─────────────────────────────
    if USE_TPU:
        config.device = "xla"
        config.dtype = "bfloat16"
        config.compile = False
        config.fused_adam = False
        config.grad_clip = 0.0
        # Disabled checkpointing for direct, single-graph execution
        config.gradient_checkpointing = 0
        config.block_size = 2048  # Pre-train at 2K context; extend to 4K/8K during finetuning
        config.batch_size = 1
        config.gradient_accumulation_steps = 1
        
        # ── OOM Prevention: Force memmap on TPU ──
        config.preload = False
        
        try:
            import torch_xla.runtime as xr
            num_cores = xr.world_size()
        except (ImportError, AttributeError):
            num_cores = xm.xrt_world_size()
        if is_master:
            print(f"TPU detected ({num_cores} cores)")
            print(f"block_size: {config.block_size}  batch_size: {config.batch_size}  grad_accum: {config.gradient_accumulation_steps}  "
                  f"(effective batch = {config.batch_size * num_cores * config.gradient_accumulation_steps} sequences = {config.batch_size * num_cores * config.block_size:,} tokens/step)")
            print(f"gradient_checkpointing: {config.gradient_checkpointing} (Direct linear execution, no subgraphs)")
            print(f"preload forced to False to prevent CPU OOM")
    elif torch.cuda.is_available():
        # ── Dynamic GPU VRAM auto-scaling ─────────────────────────────────
        vram_gb = torch.cuda.get_device_properties(ddp_local_rank).total_memory / 1e9
        
        if is_ddp:
            # If DDP, original_effective_batch is divided among world_size
            original_effective_batch = config.batch_size * config.gradient_accumulation_steps
        else:
            original_effective_batch = config.batch_size * config.gradient_accumulation_steps

        # Prevent CUDA memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Pre-train at 2K context on all tiers; extend to 4K/8K during finetuning via RoPE
        config.block_size = 2048

        # Scale batch_size and optimizations based on available VRAM
        if vram_gb >= 70:       # H100 80GB / A100 80GB
            new_batch = 8
            config.compile = False
            config.gradient_checkpointing = 1
            config.save_interval = 400
        elif vram_gb >= 35:     # A100 40GB
            new_batch = 4
            config.compile = False
            config.gradient_checkpointing = 1
            config.save_interval = 200
        elif vram_gb >= 20:     # NVIDIA L4 / RTX 3090 / 4090 (24GB)
            new_batch = 2
            config.compile = False
            config.use_8bit_optimizer = True
            config.gradient_checkpointing = 1
            config.save_interval = 25
        else:                   # T4 16GB / RTX 3060 etc.
            new_batch = 1
            config.compile = False
            config.use_8bit_optimizer = True
            config.gradient_checkpointing = 1

        # Set eval_iters reasonably to prevent streaming buffer explosion
        base_eval_batches = 20
        config.eval_iters = max(5, base_eval_batches // new_batch)

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
                  f"block_size={config.block_size}, "
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

    use_pin_memory = (config.device == "cuda")
    train_loader, val_loader = create_streaming_dataloaders(
        dataset_name="openbmb/Ultra-FineWeb-L1",
        dataset_config="CC-MAIN-2025-30",
        tokenizer=tokenizer,
        block_size=config.block_size,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=use_pin_memory,  # Free speedup: page-locked CPU→GPU transfers
        seed=42 + seed_offset,
        rank=global_rank,
        world_size=world_size,
    )

    trainer = Trainer(config, tokenizer, train_loader, val_loader, is_ddp=is_ddp)

    import glob
    import re
    checkpoints = glob.glob("checkpoints/checkpoint-*.pt")
    valid_checkpoints = [f for f in checkpoints if re.search(r"checkpoint-(\d+)\.pt", os.path.basename(f))]
    if valid_checkpoints:
        # Sort by step number
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

    try:
        trainer.train()
    except KeyboardInterrupt:
        if is_master:
            print("\nInterrupted, saving checkpoint...")
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


def run_notebook(hf_token: str = ""):
    """Official Kaggle / Colab Notebook Launcher for PyTorch XLA."""
    if USE_TPU:
        try:
            from accelerate import notebook_launcher
            notebook_launcher(_train_worker, args=(hf_token,), num_processes=8)
        except Exception:
            import torch_xla.distributed.xla_multiprocessing as xmp
            xmp.spawn(_train_worker, args=(hf_token,), nprocs=8, start_method="fork")
    else:
        _train_worker(index_or_token=hf_token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace WRITE Token")
    args, _ = parser.parse_known_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    if USE_TPU:
        os.environ["PJRT_DEVICE"] = "TPU"
        is_already_worker = (
            int(os.environ.get("RANK", -1)) != -1 
            or int(os.environ.get("LOCAL_RANK", -1)) != -1
            or os.environ.get("ACCELERATE_USE_TPU", "").lower() == "true"
            or "accelerate" in sys.argv[0].lower()
            or "accelerate.commands" in sys.modules
        )
        if is_already_worker:
            _train_worker(index_or_token=hf_token)
        else:
            if _is_proc_master():
                print("=" * 56)
                print("  BellHart — TPU Training Mode (8 cores)")
                print("=" * 56)
            import torch_xla.distributed.xla_multiprocessing as xmp
            xmp.spawn(_train_worker, args=(hf_token,), nprocs=8, start_method='fork')
    else:
        _train_worker(index_or_token=hf_token)


if __name__ == "__main__":
    main()
