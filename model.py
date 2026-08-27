"""
model.py  —  BellHart Deep Reasoning Transformer Architecture
================================================================
A 52-layer decoder-only language model implementing modern architectural
optimizations for deep, parameter-efficient reasoning:

Architectural Components:
  1. Root Mean Square Layer Normalization (RMSNorm) — fast pre-normalization.
  2. sqrt(d_model) Embedding Scaling — stabilizes tied input/output embeddings.
  3. Grouped-Query Attention (GQA) — 16 Query heads / 4 KV heads (75% KV cache reduction).
  4. Query-Key Normalization (QK-Norm) — prevents attention entropy collapse in deep models.
  5. Rotary Position Embeddings (RoPE) with theta=500,000 — supports long-context extrapolation.
  6. Value-Residual Learning (Res-V) — blends current layer values with previous layer values.
  7. SwiGLU Feed-Forward Network — gated non-linear activation with hardware-aligned hidden dimension.
  8. U-Net Long-Range Skip Connections — symmetrical residual shortcuts connecting layer i to layer (N-1-i).
  9. Logit Soft-Capping — bounds output logits to prevent numerical instability in deep architectures.
 10. Fan-In Variance-Preserving Initialization — scale-aware weights for stable forward signal propagation.
 11. KV Caching — enables fast, memory-efficient autoregressive token generation.
 12. Exponential Moving Average (EMA) — shadow weight tracking for stabilized inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig


# ──────────────────────────────────────────────────────────────────────────────
# 1. RMSNorm (Root Mean Square Layer Normalization)
# ──────────────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    
    Unlike standard LayerNorm, RMSNorm does not subtract the mean, reducing
    computational overhead by ~15% while providing identical training stability.
    
    Formula:
        y = (x / sqrt(mean(x^2) + eps)) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability across mixed-precision training
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# ──────────────────────────────────────────────────────────────────────────────
# 2. Rotary Position Embeddings (RoPE)
# ──────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (Su et al., RoFormer 2021).
    
    Encodes relative position by rotating Query and Key vectors in the complex plane.
    Precomputes and caches sinusoidal frequencies up to max_seq_len with dynamic expansion.
    
    Base frequency theta is set to 500,000 (LLaMA 3 / Qwen 2.5 style) to preserve high-frequency
    resolution and allow smooth context extension from pre-training (2k) to fine-tuning (8k/32k).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 500000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # Compute inverse frequencies: 1 / (theta ^ (2i / d))
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Construct cosine and sine rotation matrices for sequence length seq_len."""
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # Shape: (seq_len, head_dim // 2)
        # Duplicate frequencies for paired 2D rotations: [freqs, freqs] -> (seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) slices for position range [offset .. offset + seq_len)."""
        end = offset + seq_len
        if end > self.cos_cached.size(0):
            self._build_cache(end)
        return self.cos_cached[offset:end], self.sin_cached[offset:end]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper function to rotate the second half of each 2D vector: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embeddings to Query and Key tensors.
    
    Formula:
        R(x) = (x * cos) + (rotate_half(x) * sin)
    
    Args:
        q: Query tensor of shape (Batch, n_head, SeqLen, head_dim)
        k: Key tensor of shape (Batch, n_kv_head, SeqLen, head_dim)
        cos: Cosine tensor of shape (SeqLen, head_dim)
        sin: Sine tensor of shape (SeqLen, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, SeqLen, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, SeqLen, head_dim)
    q_embed = (q.float() * cos) + (_rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (_rotate_half(k.float()) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Grouped-Query Attention (GQA) with QK-Norm and Value-Residual (Res-V)
# ──────────────────────────────────────────────────────────────────────────────


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention with QK-Normalization, RoPE, and Value-Residual Learning.
    
    Key Features:
      - GQA (Ainslie et al., 2023): 16 Q heads share 4 KV heads (4:1 grouping ratio),
        providing a 4x reduction in KV cache memory and memory bandwidth with minimal quality loss.
      - QK-Norm (Dehghani et al., 2023): RMSNorm applied to Q and K per-head before attention,
        preventing attention logit growth and entropy collapse in deep models (52 layers).
      - Value-Residual (He et al., Res-V 2024): Blends current layer V projection with the previous
        layer's unmixed V projection (0.5 * V_cur + 0.5 * V_prev), maintaining direct representation
        flow across deep network stacks.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head  # Query heads per KV head

        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        # Linear projections for Query, Key, Value, and Output
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        # Per-head RMSNorm for Query and Key normalization
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        v_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        B, T, C = x.shape

        # Linear projections into Q, K, V representations
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Value-Residual: store unmixed V for the next block, then blend with previous block's V
        v_cur = v
        if v_prev is not None:
            v = 0.5 * v + 0.5 * v_prev

        # Apply QK-Norm per head dimension, then transpose to (Batch, Heads, SeqLen, HeadDim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)

        # Apply Rotary Position Embeddings to normalized Q and K
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # Append to KV cache for fast autoregressive generation
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k.detach(), v.detach())

        # Scaled dot-product attention (dispatches natively to FlashAttention / SDPA)
        is_causal = T > 1 and layer_past is None
        if self.n_rep > 1:
            try:
                # PyTorch 2.0+ native FlashAttention GQA kernel
                y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=True)
            except (TypeError, RuntimeError):
                # Fallback: expand Key and Value heads to match Query heads if native GQA kernel unavailable
                k_exp = k[:, :, None, :, :].expand(B, self.n_kv_head, self.n_rep, k.size(2), self.head_dim).reshape(B, self.n_head, k.size(2), self.head_dim)
                v_exp = v[:, :, None, :, :].expand(B, self.n_kv_head, self.n_rep, v.size(2), self.head_dim).reshape(B, self.n_head, v.size(2), self.head_dim)
                y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Reshape and project attention output back to residual stream width
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.resid_dropout(y)

        return y, present, v_cur


# ──────────────────────────────────────────────────────────────────────────────
# 4. SwiGLU Feed-Forward Network (MLP)
# ──────────────────────────────────────────────────────────────────────────────


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU) Feed-Forward Network (Shazeer, 2020).
    
    Replaces traditional two-layer GELU MLPs with a three-layer gated bilinear structure:
        output = down_proj(silu(gate_proj(x)) * up_proj(x))
    
    Provides significantly superior validation perplexity per parameter count compared to standard MLPs.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = config.intermediate_size
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Transformer Decoder Block
# ──────────────────────────────────────────────────────────────────────────────


class Block(nn.Module):
    """
    Standard pre-norm Transformer decoder block with residual connections.
    
    Data Flow:
        x -> RMSNorm -> GQA+RoPE+Res-V -> + residual -> RMSNorm -> SwiGLU -> + residual
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        v_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        attn_out, present, v_cur = self.attn(self.ln_1(x), rope_cos, rope_sin, layer_past, v_prev=v_prev)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present, v_cur

    def forward_checkpoint(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        v_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass variant optimized for PyTorch non-reentrant gradient checkpointing."""
        attn_out, _, v_cur = self.attn(self.ln_1(x), rope_cos, rope_sin, None, v_prev=v_prev)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, v_cur


# ──────────────────────────────────────────────────────────────────────────────
# 6. GPT Main Model
# ──────────────────────────────────────────────────────────────────────────────


class GPT(nn.Module):
    """
    BellHart Decoder-Only Language Model.
    
    Combines:
      - 52 Transformer blocks with GQA (16/4 heads) and SwiGLU.
      - Tied Token Embeddings scaled by sqrt(d_model).
      - U-Net symmetric skip connections (layer i added to layer N-1-i).
      - Logit soft-capping (Gemma 2 style) to bound output logits to [-30, +30].
      - Fan-In variance-preserving initialization to stabilize deep network signals.
      - Optimizer parameter grouping with decoupled weight decay.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        head_dim = config.n_embd // config.n_head

        # Token embedding and dropout
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # 52 Transformer Decoder blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final RMSNorm and Language Model Head
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share embedding and output projection matrices to save ~33.5M parameters
        self.wte.weight = self.lm_head.weight

        # Global RoPE generator (shared across all layers)
        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=config.block_size,
            theta=config.rope_theta,
        )

        # Initialize all network weights using fan-in variance scaling
        self.apply(self._init_weights)

        # Depth-scaled initialization for residual projections (attn.o_proj and mlp.down_proj)
        # Prevents residual stream variance from exploding across 52 layers: std = 1 / sqrt(2 * N_layer * fan_in)
        for block in self.h:
            fan_in_attn = block.attn.o_proj.weight.size(1)
            fan_in_mlp = block.mlp.down_proj.weight.size(1)
            std_attn = 1.0 / math.sqrt(2 * config.n_layer * fan_in_attn)
            std_mlp = 1.0 / math.sqrt(2 * config.n_layer * fan_in_mlp)
            torch.nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=std_attn)
            torch.nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=std_mlp)

    def _init_weights(self, module: nn.Module):
        """Variance-preserving fan-in initialization (std = 1 / sqrt(fan_in))."""
        if isinstance(module, nn.Linear):
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))

    def forward(
        self,
        idx: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        B, T = idx.shape

        # Calculate position offset for KV caching during autoregressive generation
        if past_key_values is not None and past_key_values[0] is not None:
            offset = past_key_values[0][0].size(2)
        else:
            offset = 0

        assert offset + T <= self.config.block_size, (
            f"Sequence length {offset + T} exceeds maximum context window {self.config.block_size}"
        )

        # Compute RoPE rotation tensors for current sequence slice
        rope_cos, rope_sin = self.rope(T, offset=offset)

        # Token embedding lookup scaled by sqrt(d_model) for tied-embedding stabilization (Gemma 2 / PaLM)
        x = self.drop(self.wte(idx) * math.sqrt(self.config.n_embd))

        presents = [] if use_cache else None
        shortcuts = []
        mid = self.config.n_layer // 2
        v_prev = None

        # Iterate through all 52 Transformer blocks
        for i, block in enumerate(self.h):
            layer_past = None
            if past_key_values is not None and i < len(past_key_values):
                layer_past = past_key_values[i]

            # U-Net long-range skip connection (Meta MobileLLM, ICML 2024):
            # First half (layers 0..25) stores representations; second half (layers 26..51) adds them back.
            if i < mid:
                shortcuts.append(x)
            else:
                x = x + shortcuts[self.config.n_layer - 1 - i]

            # Gradient checkpointing execution
            ckpt = self.config.gradient_checkpointing
            if self.training and ckpt > 0 and (i % ckpt == 0):
                x, v_cur = torch.utils.checkpoint.checkpoint(
                    block.forward_checkpoint, x, rope_cos, rope_sin, v_prev, use_reentrant=False
                )
                present = None
            else:
                x, present, v_cur = block(x, rope_cos, rope_sin, layer_past, v_prev=v_prev)

            v_prev = v_cur

            if use_cache:
                presents.append(present)

        # Final RMSNorm and projection to vocabulary logits
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Logit Soft-Capping (Gemma 2 style): bounds logits to [-logit_soft_cap, +logit_soft_cap]
        soft_cap = getattr(self.config, "logit_soft_cap", 0.0)
        if soft_cap > 0.0:
            logits = soft_cap * torch.tanh(logits / soft_cap)

        return logits, presents

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching and temperature/top-k/top-p sampling."""
        past_key_values = None
        for _ in range(max_new_tokens):
            if past_key_values is not None:
                past_len = past_key_values[0][0].size(2)
                if past_len >= self.config.block_size:
                    past_key_values = None
                    idx_cond = idx[:, -self.config.block_size:]
                else:
                    idx_cond = idx[:, -1:]
            else:
                idx_cond = idx[:, -self.config.block_size:]

            logits, past_key_values = self(idx_cond, past_key_values=past_key_values, use_cache=True)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Top-P (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, config: GPTConfig) -> torch.optim.Optimizer:
        """
        Creates AdamW optimizer with decoupled weight decay parameter separation.
        
        Rules:
          - Decay group (weight_decay = config.weight_decay): all 2D Linear weight matrices.
          - No-decay group (weight_decay = 0.0): RMSNorm scales, Embedding weights, and any biases.
          - Excludes tied output weights (lm_head.weight).
          - Supports bitsandbytes 8-bit AdamW for VRAM-constrained GPUs.
        """
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear,)
        blacklist = (RMSNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if fpn == "lm_head.weight":
                    continue  # Tied with wte.weight
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict.pop("lm_head.weight", None)

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert not inter_params, f"Parameters {inter_params} in both decay and no-decay sets"
        assert set(param_dict.keys()) == union_params, "Parameters missing from optimizer separation"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        # 8-bit AdamW optimizer for VRAM savings
        if config.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    optim_groups,
                    lr=config.learning_rate,
                    betas=(config.beta1, config.beta2),
                )
                print("  Using bitsandbytes AdamW8bit optimizer (Non-Paged)")
                return optimizer
            except Exception as e:
                print(f"  bitsandbytes not available ({e}), falling back to standard AdamW")

        # Standard fused AdamW optimizer
        import inspect
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and config.fused_adam and (config.device == "cuda")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            fused=use_fused,
        )
        return optimizer


# ──────────────────────────────────────────────────────────────────────────────
# 7. Exponential Moving Average (EMA)
# ──────────────────────────────────────────────────────────────────────────────


class EMA:
    """
    Exponential Moving Average (EMA) of model parameters.
    
    Maintains a smoothed shadow copy of model weights during training:
        shadow = decay * shadow + (1 - decay) * current_weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Record initial parameter copies, skipping tied parameter data pointers."""
        seen_data_ptrs = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ptr = param.data.data_ptr()
                if ptr not in seen_data_ptrs:
                    self.shadow[name] = param.data.clone()
                    seen_data_ptrs.add(ptr)

    def update(self):
        """Update shadow weights with current model weights."""
        seen_data_ptrs = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ptr = param.data.data_ptr()
                if ptr in seen_data_ptrs:
                    continue  # Skip tied weights
                seen_data_ptrs.add(ptr)
                if name not in self.shadow:
                    continue
                shadow = self.shadow[name]
                shadow.mul_(self.decay)
                shadow.add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        """Load shadow weights into the model (saving current weights to backup)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self.shadow = {}
        for k, v in state_dict["shadow"].items():
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            self.shadow[new_k] = v
