# BellHart: Deep-Reasoning Small Language Model Architecture

BellHart is a **619.8-Million parameter**, 52-layer deep-reasoning autoregressive decoder-only Transformer built for high sample efficiency, extreme training stability, and maximum inference throughput under limited computational budgets.

---

## 1. Architectural Hyperparameters

| Hyperparameter | Value | Description / Reference |
| :--- | :--- | :--- |
| **Total Parameters** | `619,822,592` (~620M) | Includes tied embedding/LM head weights |
| **Vocabulary Size ($V$)** | `32,768` | Custom Byte-Level BPE Tokenizer with `<\|endoftext\|>` at ID 0 |
| **Hidden Dimension ($d_{\text{model}}$)** | `1,024` | Model width |
| **Layer Depth ($N_{\text{layer}}$)** | `52` | Deep-reasoning topology (Meta MobileLLM ICML 2024) |
| **Query Heads ($n_{\text{head}}$)** | `16` | Attention heads for queries ($d_{\text{head}} = 64$) |
| **Key/Value Heads ($n_{\text{kv\_head}}$)** | `4` | Grouped-Query Attention (GQA grouping ratio = 4:1) |
| **Head Dimension ($d_{\text{head}}$)** | `64` | $d_{\text{model}} / n_{\text{head}} = 1024 / 16$ |
| **Intermediate FFN Size** | `2,816` | SwiGLU hidden dimension ($\approx 2.75 \times d_{\text{model}}$, multiple of 256) |
| **Context Length ($T$)** | `4,096` tokens | Rotary sequence length capacity |
| **RoPE Base Theta ($\theta$)** | `500,000.0` | Modern high-frequency scaling (LLaMA 3 / Qwen 2.5 standard) |
| **Logit Soft-Capping** | `30.0` | Gemma 2-style non-allocating logit containment |
| **Normalization** | Pre-RMSNorm ($\epsilon = 10^{-5}$) | Pre-layer normalization with learned scale |
| **Weight Tying** | Yes ($W_{\text{wte}} \equiv W_{\text{lm\_head}}$) | Conserves ~33.5M parameters on sub-1B scale |

---

## 2. Core Architectural Innovations

```
                              ┌────────────────────────────────────────┐
                              │           Input Tokens (idx)           │
                              └───────────────────┬────────────────────┘
                                                  │
                                   wte(idx) * sqrt(d_model)  [Tied Embedding]
                                                  │
                ┌─────────────────────────────────▼──────────────────────────────────┐
                │                         52 Transformer Blocks                      │
                │                                                                    │
                │   Block 0 ───┐ [Shortcut: x_0]                  ┌───> Block 51     │
                │   Block 1 ───┼───┐ [Shortcut: x_1]          ┌───┼───> Block 50     │
                │   Block 2 ───┼───┼───┐                  ┌───┼───┼───> Block 49     │
                │      ...     │   │   │                  │   │   │       ...        │
                │              │   │   │  [U-Net Skips]   │   │   │                  │
                │              │   │   └──> (x_i + x_j) <─┘   │   │                  │
                │              │   └──────> (x_i + x_j) <─────┘   │                  │
                │              └──────────> (x_i + x_j) <─────────┘                  │
                │                                                                    │
                │   Inside Each Block:                                               │
                │   x ──> RMSNorm ──> QK-Norm GQA + RoPE + Res-V ──(+)──> RMSNorm   │
                │                                                    │       │       │
                │                                                    │    SwiGLU     │
                │                                                    │       │       │
                │                                                    └──────(+)──> x │
                └─────────────────────────────────┬──────────────────────────────────┘
                                                  │
                                             Final RMSNorm
                                                  │
                                         lm_head (Tied wte)
                                                  │
                                      Logit Soft-Capping (tanh)
                                                  │
                                            Cross-Entropy
```

---

### 2.1 $\sqrt{d_{\text{model}}}$ Embedding Scaling (Gemma 2 & PaLM)
Because BellHart ties the weights of `wte` and `lm_head`, the embedding matrix serves dual duties: input token lookup and output logit classification. To balance gradient variance and prevent input signal attenuation, raw embedding lookups are scaled by $\sqrt{d_{\text{model}}}$:

$$x = \text{Embedding}(\text{idx}) \times \sqrt{1024} = 32.0 \cdot \text{Embedding}(\text{idx})$$

---

### 2.2 QK-Normalization (RMSNorm on Queries and Keys)
In deep 52-layer networks, unnormalized attention dot products ($q \cdot k^T / \sqrt{d}$) can grow monotonically across layers, resulting in entropy collapse (attention peaking on arbitrary tokens) and catastrophic loss spikes. 

BellHart applies an independent `RMSNorm` across head dimensions ($d_{\text{head}} = 64$) for both $Q$ and $K$ before rotary position embedding:

$$q = \text{RoPE}\left(\text{RMSNorm}(W_q x)\right), \quad k = \text{RoPE}\left(\text{RMSNorm}(W_k x)\right)$$

* **Reference**: Gemma 2, DeepSeek-V3, Qwen 2.5, OLMo 2.
* **Overhead**: $0.0\%$ compute overhead.

---

### 2.3 Value-Residual Learning (Res-V)
To eliminate "attention concentration" and ensure token semantics flow without degradation through all 52 layers, BellHart incorporates Value-Residual learning. In each attention layer, the current value projection $V_l$ is linearly blended with the previous layer's value tensor $V_{l-1}$:

$$V_l = 0.5 \cdot W_v x_l + 0.5 \cdot V_{l-1}$$

* **Reference**: Modded-NanoGPT / Res-Attention.
* **Benefit**: Prevents later layers from collapsing into uninformative attention distributions.

---

### 2.4 U-Net Long-Range Skip Connections (MobileLLM)
Following Meta's research on deep sub-billion architectures (MobileLLM, ICML 2024), BellHart bridges the bottom and top halves of the 52-layer stack with U-Net style residual connections. For layer $i$ in the second half ($i \in [26, 51]$):

$$x_i \leftarrow x_i + x_{52 - 1 - i}$$

* **Benefit**: Acts as an ultra-fast gradient highway between early representation layers and late classification layers, drastically reducing gradient decay across 52 layers.

---

### 2.5 Native FlashAttention-2 with Grouped-Query Attention (GQA)
BellHart uses 16 Query heads and 4 Key/Value heads ($4:1$ ratio). Rather than expanding $K$ and $V$ tensors into memory, BellHart dispatches directly to PyTorch's native FlashAttention kernel with `enable_gqa=True`:

$$y = \text{FlashAttention}(Q, K, V, \text{is\_causal}=\text{True}, \text{enable\_gqa}=\text{True})$$

* **KV Cache Savings**: Reduces KV cache memory by $75\%$ during inference.

---

### 2.6 Modern RoPE Positional Embeddings ($\theta = 500,000$)
Positions are encoded via Rotary Position Embeddings (RoPE) applied to Query and Key projections with a base frequency $\theta = 500,000.0$:

$$\text{RoPE}(x, m) = x \odot \cos(m \Theta) + \text{RotateHalf}(x) \odot \sin(m \Theta)$$

* **Benefit**: Robust position encoding across long sequences ($4,096\text{--}32,768$ tokens) without high-frequency position decay.

---

### 2.7 SwiGLU Feed-Forward Network
BellHart replaces standard ReLU/GELU MLPs with the gated SwiGLU activation function:

$$\text{SwiGLU}(x) = W_{\text{down}} \cdot \left(\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)\right)$$

* **Dimensions**: $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{1024 \times 2816}$, $W_{\text{down}} \in \mathbb{R}^{2816 \times 1024}$.
* **Alignment**: Intermediate dimension $2,816$ is an exact multiple of 256, maximizing GPU Tensor Core warp occupancy.

---

### 2.8 Zero-Memory Logit Soft-Capping ($\pm 30.0$)
To prevent logits from exploding into unstable numerical ranges ($\pm \infty$) during half-precision (`bfloat16`/`float16`) training, BellHart applies in-place soft-capping directly on the final output projection:

$$\text{logits} = 30.0 \cdot \tanh\left(\frac{\text{logits}}{30.0}\right)$$

* **Benefit**: Guarantees all classification logits stay bounded within $[-30.0, +30.0]$ with $0\text{ MB}$ extra VRAM overhead (replacing memory-heavy Z-loss).

---

## 3. Variance-Preserving Weight Initialization

To prevent activation vanishing or explosion across 52 sequential layers:

1. **Standard Linear Layers (Fan-In LeCun Scaling)**:
   $$\sigma = \frac{1}{\sqrt{d_{\text{in}}}}$$
   * Example: $W_q, W_k, W_v, W_{\text{gate}}, W_{\text{up}} \rightarrow \sigma = \frac{1}{\sqrt{1024}} \approx 0.03125$.

2. **Residual Output Projections ($W_o$ and $W_{\text{down}}$)**:
   $$\sigma = \frac{1}{\sqrt{2 \cdot N_{\text{layer}} \cdot d_{\text{in}}}}$$
   * For $W_o$ ($d_{\text{in}} = 1024$): $\sigma = \frac{1}{\sqrt{2 \cdot 52 \cdot 1024}} \approx 0.00306$.
   * For $W_{\text{down}}$ ($d_{\text{in}} = 2816$): $\sigma = \frac{1}{\sqrt{2 \cdot 52 \cdot 2816}} \approx 0.00185$.

This ensures the variance of activations remains identically $\approx 1.0$ through all 52 layers from step 0.

---

## 4. Optimizer & Training Pipeline

### 4.1 Optimizer Configuration (AdamW)
* **$\beta_1 = 0.90, \beta_2 = 0.95$**: Faster adaptation of second-moment statistics to streaming web text distributions.
* **Weight Decay = $0.1$**: Applied strictly to Linear weights ($W$). Excluded from all RMSNorm scale parameters (`weight`), embedding weights, and biases.
* **Gradient Clipping**: Norm threshold $= 1.0$.

### 4.2 Learning Rate Schedules
* **Peak Learning Rate**: $5\times 10^{-4}$
* **Minimum Learning Rate**: $5\times 10^{-5}$ ($10\%$ of peak)
* **Warmup**: $3,000$ iterations (linear ramp)
* **Supported Schedules**:
  1. **Continuous Cosine Decay**: Standard decay across `lr_decay_iters`.
  2. **WSD (Warmup-Stable-Decay)**: Holds at peak $5\times 10^{-4}$ for the bulk of training, followed by a rapid cosine decay across the final `decay_iters` (15,000 steps).

### 4.3 Stateful Resumable Streaming
* **0% Data Duplication**: Checkpoints capture the exact state dictionary of HuggingFace streaming datasets (shard index, row pointer, prefetch buffer, and seed).
* **Document Delimiting**: Every streamed document terminates with `<|endoftext|>` (Token ID `0`), preventing cross-document attention bleed.
* **Independent Evaluation**: Validation streams use an isolated seed offset (`+100,000`), ensuring evaluation never consumes training tokens.
