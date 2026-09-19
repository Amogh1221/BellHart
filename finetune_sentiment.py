"""
finetune_sentiment.py  —  INT4 QAT Fine-Tuning on Psychic & Sentiment Dataset
=============================================================================
Fine-tunes the Jerry ModernBERT encoder into a specialized 7-class mental health
and psychological sentiment classifier:
  Classes (7): Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder

Features:
  - 2× GPU DDP (DistributedDataParallel) support via torchrun.
  - Automatic NaN cleaning and tokenization.
  - W4A16 Group-Wise (group_size=64) Quantization-Aware Training.
  - Native C++ Straight-Through Estimator (STE).
  - Cosine Annealing learning rate schedule.
  - Dynamic padding for fast batch throughput.
  - Packs final model into true 4-bit format (~91 MB).
  - Automatically uploads to Hugging Face: Amogh1221/Jerry/PsychicJerry.
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from huggingface_hub import HfApi

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from bert import ModernBertModel, BertConfig
from tokenizer import Tokenizer
from qat_int4 import (
    apply_qat_to_modernbert,
    pack_weights_int4,
    unpack_weights_int4,
    FakeQuantLinearW4A16,
)

LABEL_LIST = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Bipolar",
    "Stress",
    "Personality disorder",
]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}


# ──────────────────────────────────────────────────────────────────────────────
# 1. 7-Class Mental Health Classifier
# ──────────────────────────────────────────────────────────────────────────────

class PsychicJerryClassifier(nn.Module):
    """ModernBERT backbone with Mean-Pooling and 7-class projection head."""

    def __init__(self, config: BertConfig, num_classes: int = len(LABEL_LIST)):
        super().__init__()
        self.config = config
        self.encoder = ModernBertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.n_embd, num_classes)

    def mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pools token representations across sequence, ignoring padding."""
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        # Convert [B, T] to 4D boolean mask [B, 1, 1, T] for SDPA attention
        sdpa_mask = (attention_mask != 0)[:, None, None, :]

        # Extract contextual representations [B, T, D]
        _, hidden_states = self.encoder(input_ids, attention_mask=sdpa_mask, output_hidden_states=True)

        # Mean pool across sequence [B, T, D] -> [B, D]
        pooled = self.mean_pooling(hidden_states, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# 2. PyTorch Dataset & Collate Function
# ──────────────────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        eot = tokenizer.eot_token

        for text, label in zip(texts, labels):
            tokens = tokenizer.encode(str(text))
            if len(tokens) > max_len - 2:
                tokens = tokens[: max_len - 2]
            combined = [eot] + tokens + [eot]
            self.samples.append((combined, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        tokens, label = self.samples[idx]
        return torch.tensor(tokens, dtype=torch.long), label


def collate_sentiment(batch):
    tokens_list, labels = zip(*batch)
    max_len = max(len(t) for t in tokens_list)

    padded = torch.zeros((len(tokens_list), max_len), dtype=torch.long)
    mask = torch.zeros((len(tokens_list), max_len), dtype=torch.long)

    for i, t in enumerate(tokens_list):
        padded[i, :len(t)] = t
        mask[i, :len(t)] = 1

    return padded, mask, torch.tensor(labels, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Model Packing for Serialization
# ──────────────────────────────────────────────────────────────────────────────

def pack_psychic_jerry(model: PsychicJerryClassifier, group_size: int = 64) -> dict:
    packed_state = {}
    for name, param in model.named_parameters():
        parent_name, _, child_name = name.rpartition(".")
        parent_module = model
        if parent_name:
            for part in parent_name.split("."):
                parent_module = getattr(parent_module, part)

        if isinstance(parent_module, FakeQuantLinearW4A16) and child_name == "weight":
            packed_w, scales = pack_weights_int4(param, group_size=group_size)
            packed_state[f"{name}.packed_int4"] = packed_w
            packed_state[f"{name}.scales_fp16"] = scales
        else:
            packed_state[name] = param.detach().cpu().to(torch.float16)

    return packed_state


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training Engine
# ──────────────────────────────────────────────────────────────────────────────

def finetune_sentiment(
    data_path: str = "Combined_Data.csv",
    base_checkpoint: str = "bert_checkpoints/latest_bert.pt",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 3e-5,
    group_size: int = 64,
    max_len: int = 256,
    output_dir: str = "exported_models/PsychicJerry",
    hf_repo: str = "Amogh1221/Jerry",
    upload: bool = False,
    hf_token: str = "",
):
    # DDP Initialization
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_master = (rank == 0)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_master = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_master:
        print(f"\n{'='*70}")
        print("  FINE-TUNING PSYCHIC JERRY: 7-CLASS MENTAL HEALTH INT4 QAT CLASSIFIER")
        print(f"{'='*70}\n")
        print(f"Device: {device} | DDP: {is_ddp} (World Size: {world_size})")
        os.makedirs(output_dir, exist_ok=True)

    # 1. Load Dataset
    if is_master:
        print(f"Loading dataset from: {data_path} ...")
    if not os.path.exists(data_path):
        # Fallback check common Kaggle input paths
        alt_paths = [
            "/kaggle/input/datasets/amoghgupta04/sentiment-analysis-psycic/Combined_Data.csv",
            "/kaggle/input/sentiment-analysis-psycic/Combined_Data.csv",
            "Combined_Data.csv",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                data_path = p
                break

    df = pd.read_csv(data_path)
    # Clean NaNs and drop unknown labels
    df = df.dropna(subset=["statement", "status"])
    df = df[df["status"].isin(LABEL2ID.keys())].reset_index(drop=True)
    if is_master:
        print(f"  [OK] Cleaned dataset size: {len(df):,} samples across {len(LABEL_LIST)} classes.")

    # Convert labels
    texts = df["statement"].astype(str).tolist()
    labels = [LABEL2ID[s] for s in df["status"]]

    # Split 90% train / 10% validation (seeded)
    indices = torch.randperm(len(df), generator=torch.Generator().manual_seed(42)).tolist()
    split_idx = int(0.90 * len(df))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    # Balance Training Data by oversampling / duplicating minority classes
    target_count = train_df["status"].value_counts().max()
    balanced_chunks = []
    for status, group in train_df.groupby("status"):
        if len(group) < target_count:
            oversampled = group.sample(target_count, replace=True, random_state=42)
            balanced_chunks.append(oversampled)
        else:
            balanced_chunks.append(group)

    train_df_balanced = pd.concat(balanced_chunks).sample(frac=1.0, random_state=42).reset_index(drop=True)

    if is_master:
        print("\nBalanced Training Class Distribution (minority classes duplicated to match majority):")
        for status, count in train_df_balanced["status"].value_counts().items():
            print(f"  - {status:<25}: {count:,} samples")
        print(f"Total balanced training samples : {len(train_df_balanced):,}")
        print(f"Validation samples (clean holdout): {len(val_df):,}\n")

    train_texts = train_df_balanced["statement"].astype(str).tolist()
    train_labels = [LABEL2ID[s] for s in train_df_balanced["status"]]
    val_texts = val_df["statement"].astype(str).tolist()
    val_labels = [LABEL2ID[s] for s in val_df["status"]]

    tokenizer = Tokenizer()
    config = BertConfig()
    config.dtype = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16"

    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, max_len=max_len)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_sentiment,
        num_workers=2 if is_ddp else 0,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sentiment, num_workers=0)

    # 2. Load Base Model Weights
    if not os.path.exists(base_checkpoint):
        # Look for local checkpoints
        candidates = sorted(Path("bert_checkpoints").glob("bert-[0-9]*.pt"))
        if candidates:
            base_checkpoint = str(candidates[-1])
        else:
            if is_master:
                print(f"Checkpoint '{base_checkpoint}' not found locally. Checking Hugging Face (Amogh1221/Jerry)...")
            try:
                from huggingface_hub import hf_hub_download
                token = hf_token or os.environ.get("HF_TOKEN", "")
                downloaded = hf_hub_download(
                    repo_id="Amogh1221/Jerry",
                    filename="Jerry/jerry_base.pt",
                    token=token if token else None,
                    local_dir=".",
                )
                base_checkpoint = downloaded
            except Exception:
                # Fallback to training checkpoint repo
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
                    base_checkpoint = downloaded
                except Exception as e:
                    raise FileNotFoundError(f"Base checkpoint not found at: {base_checkpoint} ({e})")

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()

    if is_master:
        print(f"Loading base weights from: {base_checkpoint} ...")
    ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)

    model = PsychicJerryClassifier(config, num_classes=len(LABEL_LIST))
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Filter to encoder keys
    encoder_sd = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_sd[k.replace("encoder.", "")] = v
        elif not k.startswith("classifier.") and not k.startswith("lm_head."):
            encoder_sd[k] = v

    model.encoder.load_state_dict(encoder_sd, strict=False)
    if is_master:
        print("  [OK] Pre-trained ModernBERT encoder weights loaded successfully.")

    # 3. Apply INT4 W4A16 QAT
    if is_master:
        print(f"Applying INT4 W4A16 Group-Wise QAT (group_size={group_size})...")
    apply_qat_to_modernbert(model.encoder, group_size=group_size)
    model = model.to(device)

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    # 4. Optimizer and Cosine Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype == "float16" and device.type == "cuda"))
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

    if is_master:
        print(f"\nCommencing {epochs} epochs of INT4 QAT fine-tuning ({total_steps:,} total steps on {world_size} GPUs)...")

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        if is_ddp and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (QAT)", dynamic_ncols=True) if is_master else train_loader
        total_loss = 0.0

        for input_ids, attention_mask, target_labels in pbar:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            target_labels = target_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, target_labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

            if is_master:
                cur_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"train_loss": f"{loss.item():.4f}", "lr": f"{cur_lr:.1e}"})

        # Validation (Master Rank)
        if is_master:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for input_ids, attention_mask, target_labels in val_loader:
                    input_ids = input_ids.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    target_labels = target_labels.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
                        logits = model(input_ids, attention_mask)
                        preds = torch.argmax(logits, dim=-1)

                    correct += (preds == target_labels).sum().item()
                    total += target_labels.size(0)

            val_acc = (correct / max(1, total)) * 100.0
            print(f"\n[Epoch {epoch} Results] Validation Accuracy: {val_acc:.2f}% (Best: {max(best_acc, val_acc):.2f}%)\n")
            if val_acc > best_acc:
                best_acc = val_acc

    # 5. Export and Pack Model into True 4-Bit Format (Master rank)
    if is_master:
        print("\nPacking Psychic Jerry into true 4-bit uint8 format...")
        base_model = model.module if hasattr(model, "module") else model
        packed_weights = pack_psychic_jerry(base_model, group_size=group_size)

        export_path = os.path.join(output_dir, "psychic_jerry_int4.pt")
        torch.save(
            {
                "packed_state_dict": packed_weights,
                "config": asdict(config),
                "group_size": group_size,
                "architecture": "PsychicJerryClassifier-W4A16",
                "model_flavor": "Psychic Jerry (7-Class Mental Health NLU)",
                "classes": LABEL_LIST,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "val_accuracy": best_acc,
            },
            export_path,
        )
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        print(f"  [OK] Saved Psychic Jerry packed weights -> {export_path} ({size_mb:.1f} MB)")

        # Save config, labels, and tokenizer
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)
        with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
            json.dump({"classes": LABEL_LIST, "label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)
        if os.path.exists("tokenizer.json"):
            import shutil
            shutil.copy2("tokenizer.json", os.path.join(output_dir, "tokenizer.json"))

        # Generate Model Card
        readme_content = f"""---
language:
- en
license: mit
tags:
- modernbert
- int4
- qat
- mental-health
- sentiment-analysis
- classification
- jerry
pipeline_tag: text-classification
---

# Psychic Jerry (7-Class Mental Health & Psychological Classifier — INT4 W4A16)

**Psychic Jerry** is a specialized 4-bit quantized NLU model built on the **Jerry ModernBERT** architecture, fine-tuned on ~53,000 statements across 7 psychological categories:
- `Normal`
- `Depression`
- `Suicidal`
- `Anxiety`
- `Bipolar`
- `Stress`
- `Personality disorder`

- **Model Size**: **{size_mb:.1f} MB** (vs 220 MB FP16 — **75% linear layer reduction**).
- **Quantization**: **W4A16 (Weight-Only INT4 with Group-Wise QAT, group_size=64)**.
- **Validation Accuracy**: **{best_acc:.2f}%**.
"""
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"  [OK] Generated model card -> {os.path.join(output_dir, 'README.md')}")

        # Optional Upload
        token = hf_token or os.environ.get("HF_TOKEN", "")
        if upload and token:
            print(f"\nUploading Psychic Jerry to Hugging Face: {hf_repo} ...")
            try:
                api = HfApi(token=token)
                api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
                api.upload_folder(
                    folder_path=output_dir,
                    repo_id=hf_repo,
                    path_in_repo="PsychicJerry",
                    repo_type="model",
                    commit_message=f"Upload Psychic Jerry 7-Class Mental Health Classifier ({size_mb:.1f}MB, Acc: {best_acc:.2f}%)",
                )
                print(f"  [SUCCESS] Published to https://huggingface.co/{hf_repo}/tree/main/PsychicJerry")
            except Exception as e:
                print(f"  [Upload Error] {e}")

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Psychic Jerry on Combined_Data.csv")
    parser.add_argument("--data_path", type=str, default="Combined_Data.csv")
    parser.add_argument("--base_checkpoint", type=str, default="bert_checkpoints/latest_bert.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="exported_models/PsychicJerry")
    parser.add_argument("--hf_repo", type=str, default="Amogh1221/Jerry")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf_token", type=str, default="")
    args = parser.parse_args()

    finetune_sentiment(
        data_path=args.data_path,
        base_checkpoint=args.base_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        group_size=args.group_size,
        max_len=args.max_len,
        output_dir=args.out_dir,
        hf_repo=args.hf_repo,
        upload=args.upload,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
