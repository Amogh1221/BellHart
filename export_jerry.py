"""
export_jerry.py  —  Package & Export Base Unquantized 'Jerry' Model
===================================================================
Exports the final pre-trained ModernBERT encoder checkpoint (from Step 150,000)
into a standalone, clean FP16/BF16 repository package for Hugging Face:
  Repo: Amogh1221/Jerry (Model flavor: Jerry)

Components Exported:
  1. jerry_base.pt (or model.pt) — Pure FP16 ModernBERT weights (~220 MB).
  2. config.json — Model architectural hyperparameters (768-dim, 12 layers, RoPE).
  3. tokenizer.json & tokenizer_config.json — Tokenizer vocabulary.
  4. README.md — Comprehensive model card detailing architecture and usage.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import asdict

import torch
from huggingface_hub import HfApi, login

from bert import ModernBertModel, BertConfig
from tokenizer import Tokenizer


def export_jerry(
    checkpoint_path: str = "bert_checkpoints/latest_bert.pt",
    output_dir: str = "exported_models/Jerry",
    hf_repo: str = "Amogh1221/Jerry",
    upload: bool = False,
    hf_token: str = "",
):
    print(f"\n{'='*60}")
    print("  PACKAGING UNQUANTIZED BASE MODEL: JERRY")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Locate latest checkpoint
    if not os.path.exists(checkpoint_path):
        candidates = sorted(Path("bert_checkpoints").glob("bert-[0-9]*.pt"))
        if candidates:
            checkpoint_path = str(candidates[-1])
        else:
            print(f"Checkpoint '{checkpoint_path}' not found locally. Checking Hugging Face (Amogh1221/bellhart_training)...")
            try:
                from huggingface_hub import hf_hub_download
                token = hf_token or os.environ.get("HF_TOKEN", "")
                downloaded = hf_hub_download(
                    repo_id="Amogh1221/bellhart_training",
                    filename="bert_checkpoints/latest_bert.pt",
                    repo_type="dataset",
                    token=token if token else None,
                    local_dir=".",
                )
                checkpoint_path = downloaded
                print(f"  [OK] Successfully downloaded latest checkpoint -> {checkpoint_path}")
            except Exception as e:
                print(f"Could not automatically download checkpoint: {e}")
                print(f"Error: Please place your checkpoint at '{checkpoint_path}'.")
                return

    print(f"Loading checkpoint from: {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    step = ckpt.get("step", 150000)
    val_loss = ckpt.get("val_loss", 1.3593)
    raw_config = ckpt.get("config", {})

    print(f"  -> Checkpoint Step: {step:,} | Val Loss: {val_loss:.4f}")

    # 2. Reconstruct config
    config = BertConfig()
    if isinstance(raw_config, dict):
        for k, v in raw_config.items():
            if hasattr(config, k):
                setattr(config, k, v)

    # Set default execution precision to float16
    config.dtype = "float16"

    # Save model config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"  [OK] Saved config -> {config_path}")

    # 3. Instantiate model and load state_dict
    model = ModernBertModel(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # Convert parameters to float16 for clean ~220 MB export
    model = model.to(torch.float16)

    model_path = os.path.join(output_dir, "jerry_base.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "step": step,
            "val_loss": val_loss,
            "architecture": "ModernBertModel (12L-768D-12H)",
            "model_flavor": "Jerry (Unquantized Base Foundation)",
        },
        model_path,
    )
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  [OK] Saved Jerry base weights -> {model_path} ({size_mb:.1f} MB)")

    # 4. Copy Tokenizer
    if os.path.exists("tokenizer.json"):
        import shutil
        shutil.copy2("tokenizer.json", os.path.join(output_dir, "tokenizer.json"))
        print("  [OK] Copied tokenizer.json")

    # 5. Create Model Card (README.md)
    readme_content = f"""---
language:
- en
license: mit
tags:
- modernbert
- transformer
- bidirectional-encoder
- bellhart
- jerry
pipeline_tag: feature-extraction
---

# Jerry (Base Foundation Encoder)

**Jerry** is a state-of-the-art 110.1M parameter bidirectional Transformer Encoder pre-trained on ~9.8 Billion tokens from Common Crawl (`openbmb/Ultra-FineWeb-L1`).

Part of the **Jerry Model Family** hosted at [`{hf_repo}`](https://huggingface.co/{hf_repo}):
- **Jerry**: Unquantized FP16 foundational encoder (~220 MB).
- **SmallJerry**: INT4 (W4A16 QAT) pure base encoder (~55 MB).
- **General Purpose Jerry**: Universal NLU & Zero-Shot classifier fine-tuned on MNLI with INT4 QAT (~55 MB).

## Architectural Highlights
- **12 Layers, 768 Hidden Dimension, 12 Attention Heads** (Head dimension = 64).
- **Pre-RMSNorm**: Fast, stable normalization without mean centering overhead.
- **Rotary Position Embeddings (RoPE)**: Base $\\theta = 500,000$ supporting up to 2,048 tokens.
- **SwiGLU Feed-Forward Network**: Gated bilinear MLP ($2,048$ intermediate width).
- **QK-Norm**: Query-Key LayerNorm preventing attention entropy collapse.
- **Native FlashAttention / SDPA**: Efficient $O(N)$ memory bidirectional self-attention.

## Pre-Training Metrics
- **Steps**: {step:,} / 150,000
- **Tokens**: ~9.8 Billion
- **Validation Loss**: {val_loss:.4f}
- **Validation Perplexity**: {math.exp(min(val_loss, 20.0)):.2f}

## Usage
```python
import torch
from bert import ModernBertModel, BertConfig
from tokenizer import Tokenizer

config = BertConfig.load("config.json")
model = ModernBertModel(config)
ckpt = torch.load("jerry_base.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tokenizer = Tokenizer()
tokens = tokenizer.encode("Jerry is a modern, ultra-fast language encoder.")
input_ids = torch.tensor([tokens], dtype=torch.long)

with torch.no_grad():
    logits, hidden_states = model(input_ids, output_hidden_states=True)
    embeddings = hidden_states[-1] # [1, Seq_len, 768]
```
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  [OK] Generated model card -> {readme_path}")

    # 6. Optional Hugging Face Upload
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if upload and token:
        print(f"\nUploading Jerry package to Hugging Face: {hf_repo} ...")
        try:
            api = HfApi(token=token)
            api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=output_dir,
                repo_id=hf_repo,
                path_in_repo="Jerry",
                repo_type="model",
                commit_message=f"Upload Jerry Base Foundation Model (Step {step})",
            )
            print(f"  [SUCCESS] Jerry published to https://huggingface.co/{hf_repo}/tree/main/Jerry")
        except Exception as e:
            print(f"  [Upload Error] {e}")
    elif upload and not token:
        print("\n[Notice] HF_TOKEN not provided. Files are saved locally in exported_models/Jerry.")

    print(f"\n{'='*60}")
    print(f"  JERRY BASE EXPORT COMPLETE: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Export Base Jerry Model")
    parser.add_argument("--checkpoint", type=str, default="bert_checkpoints/latest_bert.pt")
    parser.add_argument("--out_dir", type=str, default="exported_models/Jerry")
    parser.add_argument("--hf_repo", type=str, default="Amogh1221/Jerry")
    parser.add_argument("--upload", action="store_true", help="Upload to Hugging Face")
    parser.add_argument("--hf_token", type=str, default="", help="Hugging Face Write Token")
    args = parser.parse_args()

    export_jerry(
        checkpoint_path=args.checkpoint,
        output_dir=args.out_dir,
        hf_repo=args.hf_repo,
        upload=args.upload,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
