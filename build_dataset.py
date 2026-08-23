#!/usr/bin/env python3
"""
build_dataset.py  —  Nemotron-CC Dataset Downloader (CommonCrawl streaming)
========================================================================
Downloads text from Nemotron-CC directly from CommonCrawl S3/HTTP.
Streams and decompresses .jsonl.zstd files on the fly to build a corpus.

Usage:
    python build_dataset.py
    # Enter target size in GB (0.1 - 100)

Output:
    data/corpus.txt          – single UTF-8 text file (docs separated by \n\n\n)
    data/dataset_stats.json  – metadata
"""

import gc
import json
import logging
import os
import re
import sys
import time
import gzip
import unicodedata
from pathlib import Path
from urllib.request import urlopen, Request
import zstandard as zstd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_dataset")

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "data"

CC_BASE_URL = "https://data.commoncrawl.org/"
PATHS_URL = CC_BASE_URL + "contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"

# We target the highest quality real-world text
TARGET_PARTITION = "quality=high/kind=actual/kind2=actual"

# ──────────────────────────────────────────────────────────────
#  Text cleaning
# ──────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Normalize Unicode, fix line endings, strip control chars, collapse blank lines."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def is_valid_document(text: str) -> bool:
    """Reject very short documents."""
    return len(text) >= 300


# ──────────────────────────────────────────────────────────────
#  Streaming utilities
# ──────────────────────────────────────────────────────────────


def get_partition_paths() -> list[str]:
    """Download the master paths file and filter to our target partition."""
    log.info(f"Downloading paths index from {PATHS_URL}...")
    req = Request(PATHS_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(req) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            paths = gz.read().decode("utf-8").splitlines()
    
    # Filter paths
    filtered = [p for p in paths if TARGET_PARTITION in p]
    log.info(f"Found {len(paths):,} total paths, {len(filtered):,} in partition '{TARGET_PARTITION}'")
    return filtered


def stream_jsonl_zstd(url: str):
    """Generator that yields parsed JSON objects from a remote .jsonl.zstd file."""
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(req) as response:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(response) as reader:
            buffer = b""
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        yield json.loads(line.decode("utf-8"))
            if buffer.strip():
                yield json.loads(buffer.decode("utf-8"))


# ──────────────────────────────────────────────────────────────
#  Statistics
# ──────────────────────────────────────────────────────────────


def compute_statistics(corpus_path: Path, total_docs: int, elapsed: float) -> dict:
    file_size = corpus_path.stat().st_size if corpus_path.exists() else 0
    est_tokens = max(1, int(file_size / 4.3))
    return {
        "source": "Nemotron-CC (CommonCrawl)",
        "partition": TARGET_PARTITION,
        "total_documents": total_docs,
        "total_bytes": file_size,
        "estimated_tokens": est_tokens,
        "final_file_size_bytes": file_size,
        "elapsed_seconds": round(elapsed, 1),
    }


def print_statistics(stats: dict):
    hr = "=" * 64
    print()
    print(hr)
    print("  DATASET STATISTICS")
    print(hr)
    print(f"  Source          : {stats['source']}")
    print(f"  Partition       : {stats['partition']}")
    print(f"  Documents       : {stats['total_documents']:>12,}")
    print(f"  Total bytes     : {stats['total_bytes']:>12,}  ({stats['total_bytes'] / 1e9:.2f} GB)")
    print(f"  Estimated tokens: {stats['estimated_tokens']:>12,}")
    print(f"  Elapsed         : {stats['elapsed_seconds']:.0f}s")
    print(hr)


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────


def main():
    print()
    print("=" * 56)
    print("  BellHart — Nemotron-CC Dataset Downloader")
    print("=" * 56)
    print()
    print("  Source : Nemotron-CC via CommonCrawl")
    print("  Output : data/corpus.txt")
    print()

    try:
        raw = input("Enter target dataset size in GB (0.1 - 100): ").strip()
        target_gb = float(raw)
        if target_gb < 0.1 or target_gb > 100:
            log.error("Target must be between 0.1 and 100 GB")
            sys.exit(1)
    except (ValueError, EOFError):
        log.error("Invalid input")
        sys.exit(1)

    target_bytes = int(target_gb * 1_000_000_000)

    log.info("")
    log.info("Target: %.2f GB (%s bytes)", target_gb, f"{target_bytes:,}")
    
    try:
        paths = get_partition_paths()
    except Exception as e:
        log.error(f"Failed to fetch paths: {e}")
        sys.exit(1)

    if not paths:
        log.error("No paths found for the target partition.")
        sys.exit(1)

    corpus_path = OUTPUT_DIR / "corpus.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[int] = set()
    current_size = 0
    total_docs = 0

    start_time = time.time()

    try:
        # We need tqdm for the overall byte progress
        from tqdm import tqdm
        pbar = tqdm(
            total=target_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="  Downloading",
            leave=True,
        )
    except ImportError:
        pbar = None
        log.info("Downloading... (install tqdm for progress bar)")

    with open(corpus_path, "w", encoding="utf-8") as outf:
        for path in paths:
            if current_size >= target_bytes:
                break
                
            full_url = CC_BASE_URL + path
            
            try:
                for example in stream_jsonl_zstd(full_url):
                    if current_size >= target_bytes:
                        break

                    raw = example.get("text", "")
                    if not raw or not raw.strip():
                        continue

                    text = clean_text(raw)
                    if not is_valid_document(text):
                        continue

                    h = hash(text)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    doc_bytes = len(text.encode("utf-8"))
                    outf.write(text + "\n\n\n")
                    
                    current_size += doc_bytes
                    total_docs += 1
                    if pbar:
                        pbar.update(doc_bytes)

                outf.flush()
                gc.collect()
            except Exception as e:
                log.error(f"\n  ✗ Failed to process {path}: {e}")
                continue

    if pbar:
        pbar.close()
        
    elapsed = time.time() - start_time

    # Save stats
    stats = compute_statistics(corpus_path, total_docs, elapsed)
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print_statistics(stats)

    log.info("Done! Dataset: %s (%.2f GB)", corpus_path, corpus_path.stat().st_size / 1e9)
    log.info("Stats : %s", stats_path)
    log.info("")
    log.info("Next steps:")
    log.info("  1. python tokenize_dataset.py  — pre-tokenize into train.bin + val.bin")
    log.info("  2. python train.py             — start training")


if __name__ == "__main__":
    main()
