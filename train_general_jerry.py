"""
train_general_jerry.py  —  INT4 (W4A16) QAT NLI Fine-Tuning for 'General Purpose Jerry'
========================================================================================
Fine-tunes the base ModernBERT encoder into 'General Purpose Jerry':
  - Universal NLU Classification & Zero-Shot Engine.
  - Multi-Genre Natural Language Inference (MNLI — 392,000 sentence pairs).
  - Mean-Pooling with Attention Masking + 3-Class NLI Head.
  - INT4 (W4A16 Group-Wise, group_size=64) Quantization-Aware Training.
  - Enables Zero-Shot Sentiment Analysis, Intent Classification, and Logic Inference.
  - Final Model Size: ~55–60 MB.
  - Publishes to: Amogh1221/Jerry (Model flavor: General-Purpose-Jerry).
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from dataclasses import asdict

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


# ──────────────────────────────────────────────────────────────────────────────
# 1. General Purpose Jerry NLU Classifier Model
# ──────────────────────────────────────────────────────────────────────────────

class GeneralJerryClassifier(nn.Module):
    """
    ModernBERT backbone with Mean-Pooling and NLI 3-class projection head:
      - 0: Entailment
      - 1: Neutral
      - 2: Contradiction
    """

    def __init__(self, config: BertConfig, num_classes: int = 3):
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

        # Extract contextual token representations from encoder
        _, hidden_states = self.encoder(input_ids, output_hidden_states=True)
        last_hidden = hidden_states[-1]

        # Mean pool across sequence
        pooled = self.mean_pooling(last_hidden, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# 2. MNLI PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MNLIDataset(Dataset):
    """
    Tokenizes (Premise, Hypothesis) pairs from Hugging Face GLUE/MNLI:
      Format: [Premise] <eos> [Hypothesis] <eos>
    """

    def __init__(self, hf_dataset, tokenizer: Tokenizer, max_len: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        eot = tokenizer.eot_token

        print(f"Tokenizing {len(hf_dataset):,} MNLI samples...")
        for row in hf_dataset:
            label = row["label"]
            if label not in (0, 1, 2):
                continue
            premise = row["premise"]
            hypothesis = row["hypothesis"]

            tok_p = tokenizer.encode(premise)
            tok_h = tokenizer.encode(hypothesis)

            # [Premise] <eot> [Hypothesis] <eot>
            combined = tok_p + [eot] + tok_h + [eot]
            if len(combined) > max_len:
                combined = combined[:max_len]

            self.samples.append((combined, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tokens, label = self.samples[idx]
        return torch.tensor(tokens, dtype=torch.long), label


def collate_mnli(batch):
    """Dynamically pads sequences in batch to max sequence length in batch."""
    tokens_list, labels = zip(*batch)
    max_len = max(len(t) for t in tokens_list)

    padded = torch.zeros((len(tokens_list), max_len), dtype=torch.long)
    mask = torch.zeros((len(tokens_list), max_len), dtype=torch.long)

    for i, t in enumerate(tokens_list):
        padded[i, :len(t)] = t
        mask[i, :len(t)] = 1

    return padded, mask, torch.tensor(labels, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Model Packing for General Purpose Jerry
# ──────────────────────────────────────────────────────────────────────────────

def pack_general_jerry(model: GeneralJerryClassifier, group_size: int = 64) -> dict:
    """Extracts and packs weights of GeneralJerryClassifier into 4-bit uint8 tensors."""
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

def train_general_jerry(
    base_checkpoint: str = "bert_checkpoints/latest_bert.pt",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 3e-5,
    group_size: int = 64,
    max_samples: int = 0,
    output_dir: str = "exported_models/GeneralJerry",
    hf_repo: str = "Amogh1221/Jerry",
    upload: bool = False,
    hf_token: str = "",
):
    # 0. Distributed Data Parallel (DDP) Detection
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
        print(f"\n{'='*65}")
        print("  TRAINING GENERAL PURPOSE JERRY: INT4 (W4A16) QAT NLU ENGINE")
        print(f"{'='*65}\n")
        print(f"Device: {device} | DDP: {is_ddp} (World Size: {world_size})\n")
        os.makedirs(output_dir, exist_ok=True)

    # 1. Tokenizer & Base Config
    tokenizer = Tokenizer()
    config = BertConfig()
    config.dtype = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16"

    # 2. Instantiate Model and Load Base Pre-Trained Weights
    if not os.path.exists(base_checkpoint):
        candidates = sorted(Path("bert_checkpoints").glob("bert-[0-9]*.pt"))
        if candidates:
            base_checkpoint = str(candidates[-1])
        else:
            if is_master:
                print(f"Checkpoint '{base_checkpoint}' not found locally. Checking Hugging Face (Amogh1221/bellhart_training)...")
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
                if is_master:
                    print(f"  [OK] Successfully downloaded latest checkpoint -> {base_checkpoint}")
            except Exception as e:
                raise FileNotFoundError(f"Base checkpoint not found at: {base_checkpoint} ({e})")

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()

    if is_master:
        print(f"Loading pre-trained ModernBERT weights from: {base_checkpoint} ...")
    ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)

    model = GeneralJerryClassifier(config, num_classes=3)
    model.encoder.load_state_dict(ckpt["model_state_dict"], strict=True)
    if is_master:
        print("  [OK] Pre-trained weights loaded into encoder backbone.")

    # 3. Apply INT4 W4A16 QAT
    if is_master:
        print(f"Applying INT4 W4A16 Group-Wise QAT (group_size={group_size})...")
    apply_qat_to_modernbert(model.encoder, group_size=group_size)
    model = model.to(device)
    if is_master:
        print("  [OK] Encoder projections converted to FakeQuantLinearW4A16.")

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    # 4. Load MNLI Dataset from Hugging Face
    if is_master:
        print("\nLoading MNLI dataset from Hugging Face (GLUE/MNLI)...")
    from datasets import load_dataset
    hf_raw = load_dataset("glue", "mnli")

    train_raw = hf_raw["train"]
    if max_samples and max_samples > 0 and max_samples < len(train_raw):
        train_raw = train_raw.select(range(max_samples))

    val_raw = hf_raw["validation_matched"].select(range(min(3000, len(hf_raw["validation_matched"]))))

    train_ds = MNLIDataset(train_raw, tokenizer, max_len=192)
    val_ds = MNLIDataset(val_raw, tokenizer, max_len=192)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_mnli,
        num_workers=2 if is_ddp else 0,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_mnli, num_workers=0)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype == "float16" and device.type == "cuda"))
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

    if is_master:
        print(f"\nCommencing {epochs} epochs of INT4 QAT fine-tuning ({total_steps:,} total optimizer steps on {world_size} GPUs)...")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        if is_ddp and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (QAT)", dynamic_ncols=True) if is_master else train_loader
        total_loss = 0.0

        for input_ids, attention_mask, labels in pbar:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)

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

        # Evaluation on Validation Matched (rank 0 evaluates)
        if is_master:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for input_ids, attention_mask, labels in val_loader:
                    input_ids = input_ids.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
                        logits = model(input_ids, attention_mask)
                        preds = torch.argmax(logits, dim=-1)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = (correct / max(1, total)) * 100.0
            print(f"\n[Epoch {epoch} Results] Validation Accuracy: {val_acc:.2f}% (Best: {max(best_acc, val_acc):.2f}%)\n")
            if val_acc > best_acc:
                best_acc = val_acc

    # 6. Pack and Export Model into True 4-Bit Format (Rank 0 only)
    if is_master:
        print("\nPacking General Purpose Jerry into true 4-bit uint8 format...")
        base_model = model.module if hasattr(model, "module") else model
        packed_weights = pack_general_jerry(base_model, group_size=group_size)

        export_path = os.path.join(output_dir, "general_jerry_int4.pt")
        torch.save(
            {
                "packed_state_dict": packed_weights,
                "config": asdict(config),
                "group_size": group_size,
                "architecture": "GeneralJerryClassifier-W4A16",
                "model_flavor": "General Purpose Jerry (Universal NLU & Zero-Shot)",
                "classes": ["entailment", "neutral", "contradiction"],
                "val_accuracy": best_acc,
            },
            export_path,
        )
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        print(f"  [OK] Saved General Purpose Jerry packed weights -> {export_path} ({size_mb:.1f} MB)")

        # Save tokenizer and config
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)
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
- nli
- zero-shot-classification
- sentiment-analysis
- jerry
pipeline_tag: zero-shot-classification
---

# General Purpose Jerry (Universal NLU & Zero-Shot Engine — INT4 W4A16)

**General Purpose Jerry** is the fine-tuned 4-bit NLU & Zero-Shot foundation model in the **Jerry Model Family** hosted at [`{hf_repo}`](https://huggingface.co/{hf_repo}).

- **Model Size**: **{size_mb:.1f} MB** (vs 220 MB FP16 base model — **75% reduction**).
- **Quantization**: **W4A16 (Weight-Only INT4 with Group-Wise QAT, group_size=64)**.
- **Pre-Trained Base**: 150k steps on ~9.8B tokens.
- **Fine-Tuning**: Multi-Genre Natural Language Inference (MNLI).
- **Validation Accuracy**: **{best_acc:.2f}%**.

## Built-In Capabilities
1. **Zero-Shot Sentiment Analysis**:
   - Classifies positive, negative, and neutral sentiments with 0 training samples.
2. **Zero-Shot Intent & Topic Classification**:
   - Compares user prompts against custom candidate labels.
3. **General NLU & Transfer Learning**:
   - Plug-and-play feature extractor for high-speed downstream NLP pipelines.

## Inference Example (Zero-Shot Sentiment)
```python
import torch
from jerry_infer import load_general_jerry, predict_zero_shot_sentiment

classifier = load_general_jerry("general_jerry_int4.pt")
review = "The battery life on this laptop easily lasts 14 hours and the screen is gorgeous."
result = predict_zero_shot_sentiment(classifier, review)
print(result) # {{'label': 'Positive', 'confidence': 0.94}}
```
"""
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"  [OK] Generated model card -> {os.path.join(output_dir, 'README.md')}")

        # Optional Upload (Master rank only)
        token = hf_token or os.environ.get("HF_TOKEN", "")
        if upload and token:
            print(f"\nUploading General Purpose Jerry to Hugging Face: {hf_repo} ...")
            try:
                api = HfApi(token=token)
                api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
                api.upload_folder(
                    folder_path=output_dir,
                    repo_id=hf_repo,
                    path_in_repo="GeneralPurposeJerry",
                    repo_type="model",
                    commit_message=f"Upload General Purpose Jerry INT4 W4A16 NLU Model ({size_mb:.1f}MB, Acc: {best_acc:.2f}%)",
                )
                print(f"  [SUCCESS] Published to https://huggingface.co/{hf_repo}/tree/main/GeneralPurposeJerry")
            except Exception as e:
                print(f"  [Upload Error] {e}")

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train and Export General Purpose Jerry")
    parser.add_argument("--base_checkpoint", type=str, default="bert_checkpoints/latest_bert.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0, help="0 for full MNLI dataset (392k pairs)")
    parser.add_argument("--out_dir", type=str, default="exported_models/GeneralJerry")
    parser.add_argument("--hf_repo", type=str, default="Amogh1221/Jerry")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf_token", type=str, default="")
    args = parser.parse_args()

    train_general_jerry(
        base_checkpoint=args.base_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        group_size=args.group_size,
        max_samples=args.max_samples,
        output_dir=args.out_dir,
        hf_repo=args.hf_repo,
        upload=args.upload,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
