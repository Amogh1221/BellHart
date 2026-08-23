#!/usr/bin/env python3
"""
tokenize_dataset.py  —  nano_brain pre-tokenizer
================================================
Converts data/corpus.txt (or any plain-text corpus) into memory-mapped
uint16 binary shards that the DataLoader can stream directly from disk.

Why this exists
---------------
Loading a 10 GB text file, then calling tokenizer.encode() on the whole
thing takes ~10–30 minutes and requires ~20 GB of RAM on every training
run.  Pre-tokenising once (this script) produces two compact binary files:

    data/train.bin   – 90 % of tokens, uint16, memory-mapped at load time
    data/val.bin     –  10 % of tokens, uint16, memory-mapped at load time

These files load in milliseconds and use almost no RAM beyond what is
actually being read.  Token ids fit in uint16 (max 65535), which covers
the GPT-2 vocabulary (50,257 ids) without overflow.

Usage
-----
    python tokenize_dataset.py                       # uses defaults
    python tokenize_dataset.py --input data/corpus.txt --split 0.9

Options
-------
    --input        Path to the source text file  [default: data/corpus.txt]
    --out-dir      Directory for .bin files      [default: data]
    --split        Train / val ratio             [default: 0.90]
    --eot          Encode <|endoftext|> between  [default: True]
                   documents (recommended)

Output
------
    data/train.bin          – training tokens (uint16, little-endian)
    data/val.bin            – validation tokens (uint16, little-endian)
    data/tokenize_stats.json – stats about the tokenisation run
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tokenizer import Tokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tokenize_dataset")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE / "data" / "corpus.txt"
DEFAULT_OUT = BASE / "data"

# How many docs/chars to batch before calling encode_batch().
# Larger batches = faster (Rust tokenizer parallelizes internally).
# Memory: each batch's output is written to disk immediately — no accumulation.
BATCH_DOCS = 5000
BATCH_CHARS = 100_000_000  # ~100 MB text per batch

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def human_size(n_bytes: int) -> str:
    """Return a human-readable file-size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def human_tokens(n: int) -> str:
    """Return a human-readable token count string."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n:,}"


def write_bin(tokens: np.ndarray, path: Path) -> None:
    """Append a uint16 token array to a binary file."""
    with open(path, "ab") as f:
        tokens.astype(np.uint16).tofile(f)


# ──────────────────────────────────────────────────────────────────────────────
# Core tokenisation
# ──────────────────────────────────────────────────────────────────────────────


def tokenize(
    input_path: Path,
    out_dir: Path,
    split_ratio: float = 0.99,
    add_eot: bool = True,
) -> dict:
    """
    Single-pass tokeniser with random per-document train/val assignment.

    Streams the input file once.  Each document is independently assigned to
    train (probability *split_ratio*) or val (probability 1 - split_ratio)
    via a seeded RNG — no two-pass scanning, no large-memory shuffle.
    Encoded batches are written directly to disk — no Python list accumulation.

    Returns a stats dict suitable for JSON serialisation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    for p in (train_path, val_path):
        if p.exists():
            log.warning("Removing existing %s", p)
            p.unlink()

    enc = Tokenizer()
    eot_id: int = enc.eot_token

    log.info("Tokenizer : Custom BPE  (vocab=%d, eot=%d)", enc.vocab_size, eot_id)
    log.info("Input     : %s  (%s)", input_path, human_size(input_path.stat().st_size))
    log.info("Output    : %s/  (train.bin + val.bin)", out_dir)
    log.info("Split     : %.0f%% train / %.0f%% val (random per-doc)", split_ratio * 100, (1 - split_ratio) * 100)
    log.info("EOT token : %s", "yes" if add_eot else "no")
    log.info("")

    input_size = input_path.stat().st_size

    total_tokens = 0
    total_docs = 0
    doc_lengths: list[int] = []

    start_time = time.time()
    rng = random.Random(42)

    # ── Single-pass: stream, sample, batch-encode, write ─────────────────
    log.info("Tokenising and writing binary files …")

    train_batch: list[str] = []
    val_batch: list[str] = []
    train_batch_chars = 0
    val_batch_chars = 0

    def encode_and_route(target: str) -> None:
        nonlocal total_docs, total_tokens, train_batch_chars, val_batch_chars
        batch = train_batch if target == "train" else val_batch
        if not batch:
            return
        path = train_path if target == "train" else val_path
        batch_ids_list = enc.encode_batch(batch)
        arrays = []
        for ids in batch_ids_list:
            if add_eot:
                ids.append(eot_id)
            if len(ids) % 100 == 0:
                doc_lengths.append(len(ids))
            total_docs += 1
            total_tokens += len(ids)
            arrays.append(np.array(ids, dtype=np.uint16))
        if arrays:
            write_bin(np.concatenate(arrays), path)
        del batch_ids_list, arrays
        batch.clear()
        if target == "train":
            train_batch_chars = 0
        else:
            val_batch_chars = 0
        gc.collect()

    doc_count = 0
    pbar = tqdm(
        total=input_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Tokenising",
    )

    doc_text: list[str] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            pbar.update(len(raw_line.encode("utf-8")))
            line = raw_line.rstrip("\n")

            if line == "" and doc_text and doc_text[-1] == "":
                full_doc = "\n".join(doc_text).strip()
                doc_text = []
                if not full_doc:
                    continue

                target = "train" if rng.random() < split_ratio else "val"
                if target == "val":
                    val_batch.append(full_doc)
                    val_batch_chars += len(full_doc)
                    if len(val_batch) >= BATCH_DOCS or val_batch_chars >= BATCH_CHARS:
                        encode_and_route("val")
                else:
                    train_batch.append(full_doc)
                    train_batch_chars += len(full_doc)
                    if len(train_batch) >= BATCH_DOCS or train_batch_chars >= BATCH_CHARS:
                        encode_and_route("train")
                doc_count += 1
                if doc_count % 5000 == 0:
                    pct = pbar.n / max(input_size, 1) * 100
                    pbar.set_postfix({"docs": f"{doc_count:,}"})
            else:
                doc_text.append(line)

    # ── Handle last document if file doesn't end with blank line ──
    if doc_text:
        full_doc = "\n".join(doc_text).strip()
        if full_doc:
            target = "train" if rng.random() < split_ratio else "val"
            if target == "val":
                val_batch.append(full_doc)
            else:
                train_batch.append(full_doc)

    # ── Flush remaining batches ──
    encode_and_route("train")
    encode_and_route("val")

    pbar.close()

    # ── Verification ────────────────────────────────────────────────────────
    log.info("")
    log.info("Verifying output files …")

    def verify(p: Path) -> int:
        if not p.exists():
            log.error("  MISSING: %s", p)
            return 0
        actual = p.stat().st_size // 2  # uint16 = 2 bytes per token
        log.info(
            "  %-12s  %s tokens  (%s)",
            p.name,
            human_tokens(actual),
            human_size(p.stat().st_size),
        )
        return actual

    train_actual = verify(train_path)
    val_actual = verify(val_path)

    elapsed = time.time() - start_time
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    # ── Stats ────────────────────────────────────────────────────────────────
    avg_doc_len = int(np.mean(doc_lengths)) if doc_lengths else 0
    median_doc_len = int(np.median(doc_lengths)) if doc_lengths else 0

    stats = {
        "model_type": "custom_bpe",
        "input_file": str(input_path),
        "input_size_bytes": int(input_path.stat().st_size),
        "total_documents": total_docs,
        "total_tokens": total_tokens,
        "train_tokens": train_actual,
        "val_tokens": val_actual,
        "split_ratio": split_ratio,
        "eot_token_added": add_eot,
        "avg_doc_tokens": avg_doc_len,
        "median_doc_tokens": median_doc_len,
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(tok_per_sec, 0),
        "train_bin": str(train_path),
        "val_bin": str(val_path),
    }

    stats_path = out_dir / "tokenize_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print summary
# ──────────────────────────────────────────────────────────────────────────────


def print_summary(stats: dict) -> None:
    hr = "=" * 64
    print()
    print(hr)
    print("  TOKENISATION COMPLETE — nano_brain")
    print(hr)
    print(f"  Input file      : {Path(stats['input_file']).name}")
    print(f"  Input size      : {human_size(stats['input_size_bytes'])}")
    print(f"  Total documents : {stats['total_documents']:,}")
    print(f"  Total tokens    : {human_tokens(stats['total_tokens'])}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  train.bin       : {human_tokens(stats['train_tokens'])} tokens")
    print(f"  val.bin         : {human_tokens(stats['val_tokens'])} tokens")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Avg doc length  : {stats['avg_doc_tokens']:,} tokens")
    print(f"  Median doc len  : {stats['median_doc_tokens']:,} tokens")
    print(f"  EOT separator   : {'yes  (<|endoftext|> = 50256)' if stats['eot_token_added'] else 'no'}")
    print(f"  Elapsed         : {stats['elapsed_seconds']:.0f}s")
    print(f"  Throughput      : {human_tokens(int(stats['tokens_per_second']))} tok/s")
    print(hr)
    print(f"  → Update config.json: set  \"dataset\": \"train.bin\"")
    print(f"  → Stats saved to: {Path(stats['val_bin']).parent / 'tokenize_stats.json'}")
    print(hr)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Chinchilla / time budget estimate
# ──────────────────────────────────────────────────────────────────────────────


def print_training_estimate(total_tokens: int) -> None:
    """
    Print a rough Chinchilla estimate dynamically based on config.json (if present)
    """
    import os
    import json
    
    try:
        if os.path.exists("config.json"):
            with open("config.json") as f:
                cfg = json.load(f)
            v = cfg.get("vocab_size", 32768)
            d = cfg.get("n_embd", 768)
            l = cfg.get("n_layer", 12)
        else:
            v, d, l = 32768, 768, 12
        
        # Rough GPT-2 parameter count: embeddings + (layers * 12 * d^2)
        params = (v * d) + (l * 12 * (d ** 2))
    except Exception:
        params = 124_000_000

    chinchilla_optimal = params * 20

    print()
    print("=" * 64)
    print(f"  TRAINING BUDGET ESTIMATE  (~{params//1_000_000}M param model)")
    print("=" * 64)
    print(f"  Dataset tokens          : {human_tokens(total_tokens)}")
    print(f"  Chinchilla optimal      : {human_tokens(chinchilla_optimal)}  (20 × params)")
    if total_tokens >= chinchilla_optimal:
        ratio = total_tokens / chinchilla_optimal
        print(f"  Your dataset is         : {ratio:.1f}x Chinchilla-optimal ✓")
    else:
        deficit = chinchilla_optimal - total_tokens
        print(f"  Deficit vs Chinchilla   : {human_tokens(deficit)} tokens — consider more data")
    print("=" * 64)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-tokenise corpus.txt → train.bin + val.bin for nano_brain training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the source text corpus (corpus.txt from build_dataset.py)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for train.bin and val.bin",
    )
    p.add_argument(
        "--split",
        type=float,
        default=0.99,
        metavar="RATIO",
        help="Fraction of tokens to use for training (rest goes to validation)",
    )
    p.add_argument(
        "--no-eot",
        action="store_true",
        help="Do NOT insert <|endoftext|> between documents (not recommended)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_path: Path = args.input
    out_dir: Path = args.out_dir
    split_ratio: float = args.split
    add_eot: bool = not args.no_eot

    # ── Validate input ───────────────────────────────────────────────────────
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        log.error("Run  python build_dataset.py  first to generate data/corpus.txt")
        sys.exit(1)

    if not (0.5 <= split_ratio <= 0.99):
        log.error("--split must be between 0.50 and 0.99, got %.2f", split_ratio)
        sys.exit(1)

    # ── Run tokenisation ─────────────────────────────────────────────────────
    log.info("nano_brain — Dataset Pre-Tokeniser")
    log.info("=" * 52)

    stats = tokenize(
        input_path=input_path,
        out_dir=out_dir,
        split_ratio=split_ratio,
        add_eot=add_eot,
    )

    print_summary(stats)
    print_training_estimate(stats["total_tokens"])

    log.info("Done. Next steps:")
    log.info('  1. Edit config.json  →  set "dataset": "train.bin"')
    log.info("  2. python train.py")


if __name__ == "__main__":
    main()
