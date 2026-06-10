"""
Pre-cache SkyPile-150B shards to HuggingFace cache directory using streaming mode.

Run this BEFORE submitting GPU training jobs so the compute nodes
read from the local cache instead of streaming over the network.

Usage:
  export HF_ENDPOINT="https://hf-mirror.com"
  python scripts/cache_dataset.py --max-samples 500000
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache SkyPile dataset shards")
    parser.add_argument("--dataset", default="Skywork/SkyPile-150B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer-path", default="tokenizer/skypile_4k_tokenizer.json")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500000,
        help="Samples to iterate (approx 500k = 1 shard, ~2GB text)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=5_000_000_000,
        help="Approx target tokens (script stops after reaching this via tokens)",
    )
    parser.add_argument("--retry-delay", type=int, default=10)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>")

    print(f"[CACHE] Loading {args.dataset} (split={args.split}) in streaming mode...")
    attempt = 1
    while True:
        try:
            ds = load_dataset(args.dataset, split=args.split, streaming=True)
            break
        except Exception as e:
            print(f"[CACHE] Attempt {attempt} failed: {e}, retry in {args.retry_delay}s")
            time.sleep(args.retry_delay)
            attempt += 1

    print("[CACHE] Dataset loaded. Iterating to populate cache...")
    tokens = 0
    samples = 0
    shards_seen: set[int] = set()
    t0 = time.time()

    for sample in ds:
        text = sample.get(args.text_field, "")
        if not text:
            continue

        # Tokenize to count tokens (also warms up tokenizer cache)
        ids = tokenizer.encode(str(text), add_special_tokens=False).ids
        tokens += len(ids) + (1 if eos_id is not None else 0)
        samples += 1

        # Track approximate shard boundaries
        shard_id = samples // 500000
        if shard_id not in shards_seen:
            shards_seen.add(shard_id)
            elapsed = time.time() - t0
            print(
                f"[CACHE] Shard ~{shard_id} cached "
                f"({samples} samples, {tokens/1e9:.2f}B tokens, "
                f"{elapsed:.0f}s elapsed)"
            )

        if samples >= args.max_samples:
            print(f"[CACHE] Reached max_samples={args.max_samples}")
            break
        if tokens >= args.target_tokens:
            print(f"[CACHE] Reached target_tokens={args.target_tokens}")
            break

    elapsed = time.time() - t0
    print(f"[CACHE] Done! {samples} samples, {tokens/1e9:.2f}B tokens, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
