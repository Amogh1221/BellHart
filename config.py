"""
config.py  —  BellHart Model and Training Configuration
=========================================================
Central configuration dataclass defining architectural hyperparameters,
training schedules, optimizer settings, and system execution flags.
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class GPTConfig:
    # ── Model Architecture Hyperparameters ───────────────────────────────
    vocab_size: int = 32768
    """Vocabulary size matching the custom Byte-Pair Encoding (BPE) tokenizer."""

    n_embd: int = 1024
    """Model hidden dimension (d_model). All residual stream activations have this width."""

    n_head: int = 16
    """Number of Query attention heads. Head dimension = n_embd / n_head = 1024 / 16 = 64."""

    n_kv_head: int = 4
    """Number of Key/Value attention heads for Grouped-Query Attention (GQA).
    GQA grouping ratio = n_head / n_kv_head = 16 / 4 = 4 Query heads per KV head,
    reducing KV cache memory and bandwidth by 75% during autoregressive generation."""

    n_layer: int = 52
    """Total number of Transformer decoder layers (deep 52-layer architecture)."""

    block_size: int = 4096
    """Maximum sequence length (context window) supported by the model architecture."""

    dropout: float = 0.0
    """Dropout probability applied to attention and residual projections (0.0 for LLM pre-training)."""

    bias: bool = False
    """Whether to use additive bias in linear layers and layer norms. Disabled for all modern LLMs."""

    # ── Feed-Forward Network (SwiGLU) ────────────────────────────────────
    intermediate_size: int = 2816
    """Hidden dimension of the SwiGLU MLP (~2.75 * n_embd, rounded to multiple_of)."""

    multiple_of: int = 256
    """Dimension alignment multiple ensuring efficient memory access on modern GPU Tensor Cores."""

    # ── Positional Embeddings (RoPE) ─────────────────────────────────────
    rope_theta: float = 500000.0
    """Base frequency theta for Rotary Position Embeddings (RoPE).
    High theta (500k) prevents out-of-distribution frequency collapse and enables seamless context
    extension from pre-training (2k) to long-context fine-tuning (8k/32k)."""

    # ── Normalization (RMSNorm) ──────────────────────────────────────────
    norm_eps: float = 1e-5
    """Small epsilon added to root-mean-square denominator for numerical stability in RMSNorm."""

    # ── Output Soft-Capping ──────────────────────────────────────────────
    logit_soft_cap: float = 30.0
    """Logit soft-capping threshold (Gemma 2 style): logits = cap * tanh(logits / cap).
    Prevents logit explosion and stabilizes training in deep architectures. Set to 0.0 to disable."""

    # ── Training & Optimization ──────────────────────────────────────────
    batch_size: int = 1
    """Per-device micro-batch size (sequences per GPU per forward pass)."""

    gradient_accumulation_steps: int = 40
    """Number of micro-steps to accumulate gradients before performing an optimizer step."""

    max_iters: int = 150000
    """Total optimizer training steps (~12.3 Billion tokens, reaching the Chinchilla compute-optimal regime)."""

    learning_rate: float = 5e-4
    """Peak learning rate for AdamW optimizer."""

    weight_decay: float = 0.1
    """Decoupled weight decay coefficient applied to 2D weight matrices (linear layers)."""

    beta1: float = 0.90
    """First-moment exponential decay coefficient for AdamW optimizer."""

    beta2: float = 0.95
    """Second-moment exponential decay coefficient for AdamW optimizer (stabilized for LLMs)."""

    lr_schedule: str = "cosine"
    """Learning rate schedule type: 'cosine' (continuous cosine decay) or 'wsd' (Warmup-Stable-Decay)."""

    warmup_iters: int = 3000
    """Number of initial linear warmup iterations from 0 to peak learning rate."""

    lr_decay_iters: int = 150000
    """Total iterations across which the cosine schedule decays down to min_lr."""

    decay_iters: int = 15000
    """Duration (in steps) of the rapid cosine decay phase when using the WSD schedule."""

    min_lr: float = 5e-5
    """Minimum learning rate floor at the end of the decay schedule (10% of peak LR)."""

    eval_interval: int = 500
    """Step interval for computing validation loss and generating structured eval reports."""

    eval_iters: int = 20
    """Number of batches evaluated from the validation stream during each evaluation."""

    log_interval: int = 1
    """Step interval for updating terminal progress, TensorBoard metrics, and file logs."""

    save_interval: int = 25
    """Step interval for saving local checkpoints and triggering asynchronous HuggingFace backups."""

    gen_interval: int = 5000
    """Step interval for generating text sample previews during training."""

    max_new_tokens_gen: int = 256
    """Maximum number of tokens generated per text sample preview."""

    num_generations: int = 3
    """Number of text sample previews generated per generation interval at varying temperatures."""

    # ── Hardware & Precision ─────────────────────────────────────────────
    device: str = "cuda"
    """Target hardware compute device ('cuda' or 'cpu')."""

    dtype: str = "bfloat16"
    """Floating-point precision for model execution ('bfloat16', 'float16', or 'float32')."""

    compile: bool = False
    """Whether to use torch.compile() for graph compilation and kernel fusion."""

    fused_adam: bool = True
    """Whether to use PyTorch's optimized fused CUDA AdamW kernel."""

    tf32: bool = True
    """Whether to enable TensorFloat-32 tensor core execution on Ampere+ GPUs."""

    use_8bit_optimizer: bool = False
    """Whether to use bitsandbytes 8-bit AdamW optimizer to conserve GPU VRAM."""

    # ── Generation Defaults ──────────────────────────────────────────────
    temperature: float = 0.8
    """Sampling temperature for generation (higher = more creative, lower = more deterministic)."""

    top_k: int = 50
    """Top-K filtering threshold for autoregressive token sampling."""

    top_p: float = 0.95
    """Top-P (nucleus) cumulative probability threshold for token sampling."""

    # ── Exponential Moving Average (EMA) ─────────────────────────────────
    ema_decay: float = 0.999
    """Decay rate for maintaining exponential moving average of model parameters."""

    use_ema: bool = False
    """Whether to maintain an EMA shadow copy of model weights during training."""

    # ── Regularization & Memory Management ───────────────────────────────
    grad_clip: float = 1.0
    """Maximum gradient norm threshold for gradient clipping (0.0 to disable)."""

    gradient_checkpointing: int = 0
    """Gradient checkpointing stride (0 = disabled, 1 = checkpoint all layers, 2 = every 2nd layer)."""

    preload: bool = False
    """Whether to preload the dataset into RAM (disabled for massive streaming datasets)."""

    # ── HuggingFace Repository ───────────────────────────────────────────
    hf_repo: str = "Amogh1221/bellhart_training"
    """Target HuggingFace Dataset repository ID for automatic checkpoint and log synchronization."""

    def save(self, path: str):
        """Serialize configuration to a formatted JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Deserialize configuration from a JSON file, ignoring obsolete keys."""
        with open(path) as f:
            data = json.load(f)
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
