<div align="center">

# 🧠 BellHart

**A modern 616M parameter language model trained from scratch in PyTorch.**  
Features a Llama-style architecture (GQA, RoPE, SwiGLU, RMSNorm) and a streamlined streaming pipeline for Nemotron-CC directly from CommonCrawl.


</div>

---

## Overview

BellHart is an end-to-end LLM pretraining pipeline:

```
build_dataset.py  →  tokenize_dataset.py  →  train.py  →  generate.py
(stream data)        (binarise data)         (train)     (inference)
```

It covers every step: streaming high-quality data from CommonCrawl, pre-tokenization, training with full AMP/gradient accumulation/EMA/checkpointing, rich terminal logging, and interactive text generation.

---

## Architecture — BellHart (~616M)

| Component | Specification |
|---|---|
| Layers (`n_layer`) | 52 |
| Attention heads (`n_head`) | 16 Query heads, 4 KV heads (GQA) |
| Embedding dim (`n_embd`) | 1024 |
| Context length (`block_size`) | 4096 tokens |
| Feed-forward | SwiGLU, dim=2736 |
| Positional encoding | Rotary Position Embeddings (RoPE), theta=100000.0 |
| Normalization | RMSNorm (pre-norm) |
| Vocabulary | 32,768 (custom-trained BPE) |
| Parameters | ~619M |
| Attention | Flash Attention via `scaled_dot_product_attention` |

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

---

### 2. Stream the dataset

```bash
python build_dataset.py
# When prompted, enter a target size in GB (0.1 - 100)
```

This streams the **Nemotron-CC** dataset (`quality=high/kind=actual/kind2=actual`) directly from CommonCrawl. It bypasses Hugging Face completely, extracting and streaming `.jsonl.zstd` files on the fly to save massive amounts of disk space.

**Output:** `data/corpus.txt` + `data/dataset_stats.json`

---

### 3. Pre-tokenise the dataset *(required before training)*

```bash
python tokenize_dataset.py
```

This converts `data/corpus.txt` into two compact binary files:

```
data/train.bin   –  90% of tokens  (uint16, memory-mapped)
data/val.bin     –  10% of tokens  (uint16, memory-mapped)
```

---

### 4. Configure

The active config is `config.json`. Current recommended values:

```json
{
  "vocab_size": 32768,
  "n_embd": 1024,
  "n_head": 16,
  "n_kv_head": 4,
  "n_layer": 52,
  "intermediate_size": 2736,
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
  "use_8bit_optimizer": true,
  "device": "cuda",
  "dtype": "bfloat16",
  "dataset": "train.bin",
  "data_dir": "data"
}
```

> **Note on `use_8bit_optimizer`:** If `bitsandbytes` is installed on a compatible system, it will drastically reduce optimizer VRAM requirements via `PagedAdamW8bit`.

#### Hardware Independence (SOLID principles)

The training scripts (`train.py`, `trainer.py`) are designed to dynamically detect and adapt to your hardware:
- **GPU (CUDA):** Automatically scales `batch_size` based on available VRAM (H100 down to T4) and enables TF32 / BFloat16 where supported.
- **TPU (XLA):** Switches automatically to `torch_xla`, uses `bfloat16` natively, disables CPU preloading to prevent OOM across 8 processes, and scales the effective batch size across cores.

---

### 5. Train

```bash
python train.py
```

Training resumes automatically from `checkpoints/latest.pt` if it exists. To start fresh, delete the checkpoint.

---

### 6. Generate text

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

---

## Project Structure

```
BellHart/
├── build_dataset.py      # Streams Nemotron-CC from CommonCrawl
├── tokenize_dataset.py   # Converts corpus.txt → train.bin + val.bin
├── config.py             # GPTConfig dataclass (all hyperparameters)
├── config.json           # Active hyperparameter values (edit this)
├── tokenizer.py          # Thin wrapper around HuggingFace `tokenizers`
├── dataset.py            # load_bin_tensors (uint16 numpy / memmap)
├── model.py              # Llama-style GPT model (RoPE, GQA, SwiGLU, RMSNorm)
├── trainer.py            # Training loop: AMP, grad accum, logging
├── train.py              # Dynamic hardware setup & entry point
├── generate.py           # Interactive text generation
├── chat.py               # Terminal chat interface for fine-tuned models
├── finetune.py           # Fine-tuning script for conversational AI
│
├── tokenizer/
│   └── tokenizer.json    # Custom BPE tokenizer (32,768 vocab)
│
├── data/
│   └── ...               # Datasets and binaries
├── checkpoints/
│   └── ...               # Model checkpoints
├── logs/
│   └── training_log.txt  # Persistent structured training log
└── runs/                 # TensorBoard event files
```
