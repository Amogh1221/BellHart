"""
model.py  —  BellHart Transformer (Llama-style architecture)
================================================================
Modern decoder-only LLM with:
  - RMSNorm (pre-normalization)
  - Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE)
  - SwiGLU Feed-Forward Network
  - Depth-scaled weight initialization
  - KV cache for efficient autoregressive generation
  - EMA (Exponential Moving Average) of model weights
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig


# ──────────────────────────────────────────────────────────────────────────────
# RMSNorm — Root Mean Square Layer Normalization
# ──────────────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """RMSNorm: normalizes by root-mean-square, no mean subtraction, no bias.
    ~15% faster than LayerNorm, used by Llama, Gemma, DeepSeek.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# ──────────────────────────────────────────────────────────────────────────────
# Rotary Position Embeddings (RoPE)
# ──────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE frequency tensors."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim // 2)
        # Create full rotation embeddings: [cos, cos] and [sin, sin] → head_dim
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        """Return (cos, sin) for positions [offset .. offset+seq_len)."""
        end = offset + seq_len
        if end > self.cos_cached.size(0):
            self._build_cache(end)
        return self.cos_cached[offset:end], self.sin_cached[offset:end]


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to Q and K tensors.

    Args:
        q: (B, n_head, T, head_dim)
        k: (B, n_kv_head, T, head_dim)
        cos: (T, head_dim)
        sin: (T, head_dim)
    Returns:
        q_rotated, k_rotated with same shapes as inputs.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    q_embed = (q.float() * cos) + (_rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (_rotate_half(k.float()) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)


# ──────────────────────────────────────────────────────────────────────────────
# Grouped-Query Attention (GQA)
# ──────────────────────────────────────────────────────────────────────────────


class GroupedQueryAttention(nn.Module):
    """Multi-Head Attention with Grouped-Query Attention (GQA) and RoPE.

    Uses fewer Key/Value heads than Query heads. Multiple Q heads share
    a single KV head, reducing KV cache size and memory bandwidth.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head  # How many Q heads share each KV head

        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, rope_cos, rope_sin, layer_past=None):
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # KV cache for inference
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k.detach(), v.detach())

        # Expand KV heads to match Q heads via non-allocating view expansion
        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(B, self.n_kv_head, self.n_rep, k.size(2), self.head_dim).reshape(B, self.n_head, k.size(2), self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_head, self.n_rep, v.size(2), self.head_dim).reshape(B, self.n_head, v.size(2), self.head_dim)

        # Scaled dot-product attention (dispatches to FlashAttention when available)
        is_causal = T > 1 and layer_past is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.resid_dropout(y)

        return y, present


# ──────────────────────────────────────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ──────────────────────────────────────────────────────────────────────────────


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer 2020).

    Uses three linear layers with gated activation:
        output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Achieves better loss-per-parameter than standard GELU FFN.
    Used by Llama, Gemma, DeepSeek, Mistral.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = config.intermediate_size
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────────────────────────────────────


class Block(nn.Module):
    """Transformer decoder block with pre-norm residual connections.

    Structure:
        x → RMSNorm → GQA+RoPE → + residual → RMSNorm → SwiGLU → + residual
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, rope_cos, rope_sin, layer_past=None):
        attn_out, present = self.attn(self.ln_1(x), rope_cos, rope_sin, layer_past)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

    def forward_checkpoint(self, x, rope_cos, rope_sin):
        attn_out, _ = self.attn(self.ln_1(x), rope_cos, rope_sin, None)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# GPT Model
# ──────────────────────────────────────────────────────────────────────────────


class GPT(nn.Module):
    """BellHart — Llama-style GPT with GQA, RoPE, SwiGLU, RMSNorm.

    No absolute position embeddings. Positions are encoded via RoPE
    directly inside the attention mechanism.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        head_dim = config.n_embd // config.n_head

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights
        self.wte.weight = self.lm_head.weight

        # RoPE — shared across all layers
        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=config.block_size,
            theta=config.rope_theta,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Depth-scaled initialization for output projections
        # Prevents residual signal explosion in deep networks
        output_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.h:
            torch.nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=output_std)
            torch.nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=output_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, past_key_values=None, use_cache=False):
        B, T = idx.shape
        device = idx.device

        # Compute RoPE frequencies for this sequence
        if past_key_values is not None and past_key_values[0] is not None:
            offset = past_key_values[0][0].size(2)  # past K length
        else:
            offset = 0

        assert offset + T <= self.config.block_size, (
            f"Sequence length {offset + T} exceeds block_size {self.config.block_size}"
        )

        rope_cos, rope_sin = self.rope(T, offset=offset)

        # Token embeddings (no position embeddings — RoPE handles positions)
        x = self.drop(self.wte(idx))

        presents = [] if use_cache else None

        for i, block in enumerate(self.h):
            layer_past = None
            if past_key_values is not None and i < len(past_key_values):
                layer_past = past_key_values[i]

            ckpt = self.config.gradient_checkpointing
            if self.training and ckpt > 0 and (i % ckpt == 0):
                if x.device.type == "xla":
                    import torch_xla.utils.checkpoint as xla_ckpt
                    x = xla_ckpt.checkpoint(block.forward_checkpoint, x, rope_cos, rope_sin)
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        block.forward_checkpoint, x, rope_cos, rope_sin, use_reentrant=False
                    )
                present = None
            else:
                x, present = block(x, rope_cos, rope_sin, layer_past)

            if use_cache:
                presents.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, presents

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        past_key_values = None
        for _ in range(max_new_tokens):
            if past_key_values is not None:
                past_len = past_key_values[0][0].size(2)  # K cache seq dim
                if past_len >= self.config.block_size:
                    # Cache full — drop it and re-process from scratch
                    past_key_values = None
                    idx_cond = idx[:, -self.config.block_size:]
                else:
                    idx_cond = idx[:, -1:]
            else:
                idx_cond = idx[:, -self.config.block_size:]

            logits, past_key_values = self(
                idx_cond, past_key_values=past_key_values, use_cache=True
            )
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = (
                    sorted_indices_to_remove[..., :-1].clone()
                )
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if idx.device.type == "xla":
                import torch_xla.core.xla_model as xm
                xm.mark_step()

        return idx

    def configure_optimizers(self, config: GPTConfig):
        """Create optimizer with separate weight decay groups.

        Weight decay is applied to all Linear weight matrices.
        No decay for RMSNorm weights, Embedding weights, and any biases.
        Optionally uses bitsandbytes PagedAdamW8bit for VRAM savings.
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
        assert not inter_params, f"Parameters {inter_params} in both sets"
        assert (
            set(param_dict.keys()) == union_params
        ), f"Parameters {set(param_dict.keys()) - union_params} not separated"

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Try 8-bit optimizer if requested (Linux/CUDA only)
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
            except ImportError:
                print("  bitsandbytes not available, falling back to standard AdamW")

        # Standard AdamW
        import inspect
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and config.fused_adam
        if config.device != "cuda":
            use_fused = False

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            fused=use_fused,
        )
        return optimizer


# ──────────────────────────────────────────────────────────────────────────────
# EMA — Exponential Moving Average
# ──────────────────────────────────────────────────────────────────────────────


class EMA:
    """Maintains an exponential moving average of model parameters.

    Used to stabilize inference by averaging out training noise.
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        seen_data_ptrs = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ptr = param.data.data_ptr()
                if ptr not in seen_data_ptrs:
                    self.shadow[name] = param.data.clone()
                    seen_data_ptrs.add(ptr)

    def update(self):
        seen_data_ptrs = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ptr = param.data.data_ptr()
                if ptr in seen_data_ptrs:
                    continue  # Skip tied weights (already updated)
                seen_data_ptrs.add(ptr)
                if name not in self.shadow:
                    continue  # Skip keys missing from old checkpoints
                shadow = self.shadow[name]
                shadow.mul_(self.decay)
                shadow.add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = {}
        # Backwards compatibility: strip 'module.' prefix from Kaggle DDP checkpoints
        for k, v in state_dict["shadow"].items():
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            self.shadow[new_k] = v
