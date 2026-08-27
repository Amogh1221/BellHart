from dataclasses import dataclass, asdict
import json


@dataclass
class GPTConfig:
    # ── Model architecture ──────────────────────────────────────────────
    vocab_size: int = 32768
    n_embd: int = 1024
    n_head: int = 16          # Query heads
    n_kv_head: int = 4        # Key/Value heads (GQA grouping ratio = n_head / n_kv_head)
    n_layer: int = 52
    block_size: int = 4096
    dropout: float = 0.0
    bias: bool = False

    # SwiGLU FFN dimensions
    intermediate_size: int = 2816   # ≈ 2.75 × n_embd, rounded to multiple_of
    multiple_of: int = 256          # FFN dim alignment for hardware efficiency

    # RoPE positional embeddings (modern 500k base frequency)
    rope_theta: float = 500000.0

    # RMSNorm
    norm_eps: float = 1e-5

    # Logit Soft-Capping (Gemma 2 style, 0.0 to disable)
    logit_soft_cap: float = 30.0

    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 1
    gradient_accumulation_steps: int = 40
    max_iters: int = 100000
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    beta1: float = 0.90
    beta2: float = 0.95
    lr_schedule: str = "cosine"      # "cosine" or "wsd" (Warmup-Stable-Decay)
    warmup_iters: int = 3000
    lr_decay_iters: int = 100000     # For continuous cosine decay
    decay_iters: int = 15000         # For WSD schedule decay phase duration
    min_lr: float = 5e-5
    eval_interval: int = 200
    eval_iters: int = 20
    log_interval: int = 1
    save_interval: int = 20
    gen_interval: int = 5000
    max_new_tokens_gen: int = 256
    num_generations: int = 3

    # ── System ──────────────────────────────────────────────────────────
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False       # Disabled: Kaggle 30GB RAM can't handle 2x DDP compile
    fused_adam: bool = True
    tf32: bool = True
    use_8bit_optimizer: bool = False   # Use bitsandbytes PagedAdamW8bit (Linux/CUDA only)



    # ── Generation defaults ─────────────────────────────────────────────
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    # ── EMA ──────────────────────────────────────────────────────────────
    ema_decay: float = 0.999
    use_ema: bool = False

    # ── Gradient clipping ────────────────────────────────────────────────
    grad_clip: float = 1.0

    # Gradient checkpointing stride (0 = off, 1 = all blocks, 2 = every 2nd, etc.)
    gradient_checkpointing: int = 0

    # Preload dataset into RAM (disable for datasets larger than available RAM)
    preload: bool = False

    # ── HuggingFace sync ─────────────────────────────────────────────────
    hf_repo: str = "Amogh1221/bellhart_training"

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        # Filter out keys that don't exist in this version of GPTConfig
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
