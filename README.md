<div align="center">

# 🧠 nanorush

**A clean GPT-class language model trained from scratch in PyTorch.**  
Custom-trained BPE tokenizer, full AMP training, EMA, gradient checkpointing, and a complete pretraining pipeline.

[**📖 Read the Masterclass Technical Wiki**](https://github.com/Amogh1221/nanorush/wiki)  
*An exhaustive, first-principles textbook covering LLM theory, Transformer math, FlashAttention, CUDA memory hierarchy, and codebase implementation.*

</div>

---

## Overview

nanorush is an end-to-end LLM pretraining pipeline:

```
build_dataset.py  →  train_tokenizer.py  →  tokenize_dataset.py  →  train.py  →  generate.py
   (download data)     (train BPE tokenizer)    (binarise data)      (train)    (inference)
```

It covers every step: dataset collection from HuggingFace, custom BPE tokenizer training with special tokens, pre-tokenisation, training with full AMP/gradient accumulation/EMA/checkpointing, rich terminal + file logging, and interactive text generation.

---

## Architecture — Custom ~283M GPT (36 layers)

| Component | Specification |
|---|---|
| Layers (`n_layer`) | 36 |
| Attention heads (`n_head`) | 12 |
| Embedding dim (`n_embd`) | 768 |
| Head dim | 64 |
| Context length (`block_size`) | 4096 tokens |
| Feed-forward | 4× expansion, GELU (tanh approx.) |
| Vocabulary | 32,768 (custom-trained BPE, via `tokenizers`) |
| Parameters | ~283M |
| Attention | Flash Attention via `scaled_dot_product_attention` |
| Positional encoding | Learned absolute (wpe) |

The custom 32,768-vocab tokenizer includes **206 guaranteed special tokens** beyond the standard BPE merges:
- `<|endoftext|>` — document separator (ID 0)
- Indentation spaces: 2, 4, 8, 12, 16 spaces — for Python indentation
- Double-digit numbers: `"00"` through `"99"` — clean number tokenization
- Space-prefixed double-digit numbers: `" 00"` through `" 99"`

The custom 32K-vocab tokenizer makes this model **not directly comparable** to GPT-2's 50K byte-level BPE perplexity — always compare against your own run.

---

## Chinchilla Scaling

The [Chinchilla scaling law](https://arxiv.org/abs/2203.15556) recommends **~20 tokens per parameter** for compute-optimal training:

- **Parameters:** ~283M
- **Chinchilla-optimal tokens:** ~5.66B
- **Recommended dataset size:** ~24 GB raw text
- **Tokens per optimizer step:** `batch_size(1) × block_size(4096) × grad_accum(40)` = 163,840
- **`max_iters`:** 35,000 steps (Chinchilla-optimal for this config)

---

## Complete Setup & Workflow

### 1. Install dependencies

```bash
uv venv .venv --python 3.11
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell

uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

> **Windows note:** `torch.compile` is not reliable on Windows. Keep `"compile": false` in `config.json` unless you're on WSL2 or Linux.

---

### 2. Build the dataset

```bash
python build_dataset.py
# When prompted, enter a target size in GB (recommended: 24 for Chinchilla-optimal)
```

This downloads from **[HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)** — a high-quality synthetic educational corpus — across all 8 configs in a curated proportion mix:

| Config | Proportion | Description |
|---|---|---|
| `web_samples_v2` | 25% | Web text (newest) |
| `web_samples_v1` | 20% | Web text |
| `stories` | 15% | Narrative / creative text |
| `auto_math_text` | 15% | Math-heavy text |
| `wikihow` | 10% | How-to instructions |
| `stanford` | 7% | Academic / Stanford courses |
| `openstax` | 5% | Textbook content |
| `khanacademy` | 3% | Educational Q&A |

**Output:** `data/corpus.txt` + `data/dataset_stats.json`

The cleaning pipeline:
- Unicode NFKC normalization
- Control character removal
- Whitespace normalisation (max 2 consecutive newlines)
- Hash-based deduplication (per document)
- Minimum document length filter (300+ chars)

---

### 3. Train the tokenizer

```bash
python train_tokenizer.py
```

Trains a BPE tokenizer on `data/corpus.txt` with vocab size **32,768** and 206 custom guaranteed tokens (indentation spaces, double-digit numbers, space-prefixed double-digit numbers).

**Output:** `tokenizer/tokenizer.json`

---

### 4. Pre-tokenise the dataset *(required before training)*

```bash
python tokenize_dataset.py
```

This converts `data/corpus.txt` into two compact binary files:

```
data/train.bin   –  90% of tokens  (uint16, memory-mapped)
data/val.bin     –  10% of tokens  (uint16, memory-mapped)
```

**Why this step is critical:**

| Method | Large corpus startup | RAM needed |
|---|---|---|
| Load `corpus.txt` at train time | 10–30 minutes | ~20+ GB |
| **Load `train.bin` (memmap)** | **< 1 second** | **~0 GB** |
| **Load `train.bin` (preload)** | **< 1 second** | **~11 GB** |

By default the `.bin` file is read via `np.memmap` (`preload: false`) — OS-paged, near-zero RAM. Set `preload: true` to load everything into RAM for zero disk reads during training (requires sufficient system RAM).

An `<|endoftext|>` (token id 0) separator is inserted between every document.

**Options:**

```bash
python tokenize_dataset.py --input data/corpus.txt   # default
python tokenize_dataset.py --split 0.90              # 90% train / 10% val
python tokenize_dataset.py --no-eot                  # skip EOT (not recommended)
```

---

### 5. Configure

The active config is `config.json`. Current recommended values:

```json
{
  "vocab_size": 32768,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 36,
  "block_size": 4096,
  "dropout": 0.0,
  "bias": false,
  "batch_size": 1,
  "gradient_accumulation_steps": 40,
  "max_iters": 35000,
  "learning_rate": 2e-4,
  "weight_decay": 0.1,
  "beta1": 0.95,
  "beta2": 0.95,
  "warmup_iters": 3000,
  "lr_decay_iters": 35000,
  "min_lr": 2e-5,
  "eval_interval": 1000,
  "eval_iters": 200,
  "log_interval": 10,
  "save_interval": 1000,
  "gen_interval": 5000,
  "max_new_tokens_gen": 256,
  "num_generations": 3,
  "device": "cuda",
  "dtype": "bfloat16",
  "compile": false,
  "fused_adam": true,
  "tf32": true,
  "dataset": "train.bin",
  "data_dir": "data",
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "ema_decay": 0.999,
  "use_ema": true,
  "grad_clip": 1.0,
  "gradient_checkpointing": 2,
  "preload": false
}
```

> **Note on `dropout: 0.0`:** For pretraining at this scale, dropout hurts more than it helps. GPT-2 and most modern LLMs train with zero dropout.

> **Note on effective batch size:** `batch_size=1` × `gradient_accumulation_steps=40` × `block_size=4096` = **163,840 tokens per update**.

> **Why these hyperparameters:**
> - `learning_rate: 2e-4` — calibrated for the 36-layer, 283M architecture. Deeper models benefit from a slightly lower peak LR for gradient stability.
> - `warmup_iters: 3000` — extended warmup allows the deeper model to stabilise all layers before taking large gradient steps.
> - `beta1 == beta2 == 0.95` — matching β2 to β1 eliminates the periodic loss spikes you get with β2 ≫ β1. Supported by Cattaneo & Shigida (NeurIPS 2025) and Orvieto & Gower (NeurIPS 2025).
> - `max_iters: 35000` — Chinchilla-optimal for 283M at 163,840 tokens/step.

#### GPU with more VRAM (A100 / H100)?

On GPUs with large VRAM, you can unlock significant speed improvements via config alone:

```json
"compile": true,
"gradient_checkpointing": 0,
"batch_size": 8,
"gradient_accumulation_steps": 5,
"preload": true
```

This keeps the same effective tokens-per-step while removing accumulation overhead, eliminating activation recomputation, and enabling `torch.compile` (Linux only).

---

### 6. Train

```bash
python train.py
```

Training resumes automatically from `checkpoints/latest.pt` if it exists. To start fresh, delete the checkpoint.

**Interrupt safely at any time with `Ctrl+C`** — the trainer catches `KeyboardInterrupt` and saves a checkpoint before exiting.

---

### 7. Generate text

```bash
python generate.py "The meaning of life is"
```

Generation parameters (via environment variables):

| Variable | Default | Description |
|---|---|---|
| `TEMP` | `0.8` | Temperature (higher = more random) |
| `TOP_K` | `50` | Top-k sampling cutoff |
| `TOP_P` | `0.95` | Nucleus (top-p) sampling threshold |
| `MAX_NEW` | `500` | Maximum tokens to generate |

```bash
# Linux/macOS
TEMP=0.6 MAX_NEW=1000 python generate.py "Once upon a time"

# Windows
set TEMP=0.6 && set MAX_NEW=1000 && python generate.py "Once upon a time"
```

---

## Monitoring & Logging

### Terminal output

Every `log_interval` steps (default: every 10 steps), the trainer prints:

```
[Step   1000/35000]  loss=3.4521  lr=1.85e-04  grad_norm=0.82  tok/s=18,000  VRAM=8.4/16.0GB  ETA=...
```

Every `eval_interval` steps (default: every 1000 steps), a full evaluation block is printed:

```
════════════════════════════════════════════════════
  EVALUATION @ Step 1000 / 35000   (2.9%)
════════════════════════════════════════════════════
  Train Loss     : 3.4521
  Val Loss       : 3.6102  (best: 3.5901  Δ: +0.020)
  Perplexity     : 37.02
  EMA Val Loss   : 3.5834
  Learning Rate  : 1.85e-04
  Avg Grad Norm  : 0.823
  Tokens/sec     : 18,000
  VRAM           : 8.4 / 16.0 GB
  Elapsed        : 00:14:22
  ETA            : ...
════════════════════════════════════════════════════
```

### TensorBoard

```bash
tensorboard --logdir runs
```

Logged metrics:

| Tag | Description |
|---|---|
| `train/loss` | Smoothed training loss (every `log_interval` steps) |
| `train/lr` | Current learning rate |
| `train/grad_norm` | Average gradient L2 norm |
| `train/tokens_per_sec` | Training throughput |
| `eval/train_loss` | Train loss from `estimate_loss()` |
| `eval/val_loss` | Validation loss |
| `eval/perplexity` | `exp(val_loss)` |
| `eval/ema_val_loss` | EMA model validation loss |

### logs/training_log.txt

A persistent structured log is written throughout training. It survives terminal closes, SSH drops, and crashes.

---

## Features

| Feature | Details |
|---|---|
| **Flash Attention** | `F.scaled_dot_product_attention` auto-dispatches to FlashAttention on Ampere+ (RTX 30xx/40xx). No extra install needed. |
| **Mixed precision (AMP)** | Trains in `bfloat16` by default. GradScaler for float16 fallback. |
| **TF32** | Enables TensorFloat-32 matmuls on Ampere+ for free speed at no quality cost. |
| **EMA** | Exponential moving average of weights (`decay=0.999`). EMA model is evaluated separately and used at inference for better generalisation. |
| **Gradient accumulation** | Simulates large effective batches (163K tokens/step) on limited VRAM. |
| **Gradient clipping** | Clips gradient norm to 1.0. Grad norm is tracked and logged. |
| **Cosine LR schedule** | Warmup 0 → `learning_rate` over `warmup_iters` steps, then cosine decay to `min_lr`. |
| **Fused AdamW** | CUDA-fused AdamW (~5–10% optimizer speedup). |
| **Weight tying** | `wte` (token embedding) and `lm_head` share weights — fewer parameters, better quality. |
| **KV cache** | Keys/values are cached during generation for O(T) per-token cost. |
| **Checkpointing** | Saves `checkpoints/latest.pt` (every `save_interval` steps) and `checkpoints/best.pt` (whenever val loss improves). Both include full optimizer state for seamless resume. |
| **Auto-resume** | Detects `checkpoints/latest.pt` at startup and resumes from saved iteration. |
| **Sample generation** | Generates `num_generations` text samples every `gen_interval` steps into `samples/`. |
| **Gradient checkpointing** | Recomputed activations (`gradient_checkpointing: 2`) to reduce activation VRAM at a small compute cost. Disable on large-VRAM GPUs for speed. |
| **Direct random sampling** | Training draws batches with `torch.randint` over numpy uint16 arrays — no `DataLoader`, no `torch.randperm`. Keeps startup to <1 s. |
| **Dataset preloading** | `train.bin`/`val.bin` are read via `np.memmap` by default (`preload: false`, near-zero RAM). Set `preload: true` to load into RAM for zero disk reads during training. |

---

## Project Structure

```
nanorush/
├── build_dataset.py      # Downloads Cosmopedia from HuggingFace (user-specified GB)
├── train_tokenizer.py    # Trains the custom BPE tokenizer → tokenizer/tokenizer.json
├── tokenize_dataset.py   # Converts corpus.txt → train.bin + val.bin (run once)
├── config.py             # GPTConfig dataclass (all hyperparameters)
├── config.json           # Active hyperparameter values (edit this)
├── tokenizer.py          # Thin wrapper around HuggingFace `tokenizers` (custom BPE)
├── dataset.py            # load_bin_tensors (uint16 numpy / memmap, no DataLoader)
├── model.py              # GPT model: LayerNorm, CausalSelfAttention, MLP, Block, EMA
├── trainer.py            # Training loop: AMP, grad accum, logging, checkpointing
├── train.py              # Entry point
├── generate.py           # Interactive text generation
├── plot_training.py      # Visualises logs/training_log.txt loss curves
│
├── tokenizer/
│   └── tokenizer.json    # Trained custom BPE tokenizer (32,768 vocab)
│
├── data/
│   ├── corpus.txt        # Raw text corpus (from build_dataset.py)
│   ├── train.bin         # Pre-tokenised training tokens, uint16 (from tokenize_dataset.py)
│   ├── val.bin           # Pre-tokenised validation tokens, uint16
│   ├── dataset_stats.json
│   └── tokenize_stats.json
│
├── checkpoints/
│   ├── latest.pt         # Most recent checkpoint (auto-resumed)
│   └── best.pt           # Best validation loss checkpoint
│
├── logs/
│   └── training_log.txt  # Persistent structured training log
│
├── samples/              # Generated text samples (step_NNNNNNN_i.txt)
└── runs/                 # TensorBoard event files
```

---

## Configuration Reference

### Model Architecture

| Key | Value | Description |
|---|---|---|
| `n_embd` | `768` | Embedding / hidden dimension |
| `n_head` | `12` | Number of attention heads |
| `n_layer` | `36` | Number of transformer blocks |
| `block_size` | `4096` | Context window (tokens) |
| `vocab_size` | `32768` | Set automatically from the custom tokenizer |
| `dropout` | `0.0` | Dropout probability (0 = disabled for pretraining) |
| `bias` | `false` | Add bias to Linear layers |

### Training

| Key | Value | Description |
|---|---|---|
| `batch_size` | `1` | Micro-batch size per GPU step |
| `gradient_accumulation_steps` | `40` | Accumulate before weight update |
| `max_iters` | `35000` | Total optimizer steps (Chinchilla-optimal) |
| `learning_rate` | `2e-4` | Peak learning rate |
| `warmup_iters` | `3000` | LR warmup steps |
| `lr_decay_iters` | `35000` | Steps over which to decay LR |
| `min_lr` | `2e-5` | Minimum LR (end of cosine schedule) |
| `weight_decay` | `0.1` | AdamW weight decay |
| `beta1` / `beta2` | `0.95` / `0.95` | Adam moments — matched β2 avoids loss spikes |
| `grad_clip` | `1.0` | Gradient norm clipping threshold |
| `gradient_checkpointing` | `2` | Activation recomputation stride (0 = off) |

### System

| Key | Value | Description |
|---|---|---|
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `dtype` | `"bfloat16"` | `"bfloat16"` (recommended) or `"float32"` |
| `compile` | `false` | Enable `torch.compile` (Linux only) |
| `tf32` | `true` | TensorFloat-32 matmuls (Ampere+ only) |
| `fused_adam` | `true` | CUDA-fused AdamW |
| `preload` | `false` | Load dataset into RAM; `false` uses `np.memmap` |

---

## Understanding the Loss Curve

| Validation Loss | Perplexity | What it means |
|---|---|---|
| ~4.5–5.0 | ~90–150 | Early training — random-ish output |
| ~3.5–4.0 | ~33–55 | Mid training — grammatical, coherent short phrases |
| ~3.0–3.5 | ~20–33 | Good — coherent paragraphs, follows topic |
| ~2.8–3.0 | ~16–20 | Strong — competitive with GPT-2-class models |
| < 2.8 | < 16 | Excellent for this model size |

> **Note on interpretation:** this repo trains a custom 32,768-vocab BPE tokenizer, so its perplexity is **not directly comparable** to GPT-2's 50K byte-level BPE numbers. Only compare against your own run.

You should see a smooth, monotonically decreasing loss. If you see periodic loss spikes, verify that `beta1` and `beta2` are matched (both 0.95) and that `learning_rate` is not too high.

---

## References

- [GPT-2 Paper](https://arxiv.org/abs/1904.05779) — Radford et al. 2019
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's minimal GPT-2 implementation (primary inspiration)
- [Chinchilla](https://arxiv.org/abs/2203.15556) — Hoffmann et al. 2022 (scaling laws)
- [Flash Attention](https://arxiv.org/abs/2205.14135) — Dao et al. 2022
- [tokenizers](https://github.com/huggingface/tokenizers) — HuggingFace's Rust BPE tokenizer
- [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) — HuggingFace synthetic educational dataset
- [How Memory in Optimization Algorithms Implicitly Modifies the Loss](https://arxiv.org/abs/2502.02132) — Cattaneo & Shigida, NeurIPS 2025
- [In Search of Adam's Secret Sauce](https://openreview.net/forum?id=CH72XyZs4y) — Orvieto & Gower, NeurIPS 2025
