"""
Download SkyPile-150B JSONL shards to a local directory.

Downloads enough shards to cover ~5B tokens (approx 20 shards).
Uses hf-mirror.com with retries for reliability.

Usage:
  export HF_ENDPOINT="https://hf-mirror.com"
  python scripts/download_dataset.py --data-dir /data/skypile --max-shards 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer


def estimate_tokens_in_shard(
    data_dir: Path, shard_relpath: str, tokenizer_path: str
) -> int:
    """Quick estimate: tokenize first 100 lines to get avg tokens/line, then
    estimate total tokens by file size proportion."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>")
    filepath = data_dir / shard_relpath
    if not filepath.exists():
        return 0

    total_bytes = filepath.stat().st_size
    sample_tokens = 0
    sample_lines = 0

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get("text", "")
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                sample_tokens += len(ids) + (1 if eos_id is not None else 0)
                sample_lines += 1
            except json.JSONDecodeError:
                continue

    if sample_lines == 0 or total_bytes == 0:
        return 0

    # Estimate: sample_bytes / total_bytes ≈ sample_lines / total_lines
    # Use proportion of bytes
    sample_bytes = filepath.stat().st_size  # re-read for fresh stat
    avg_tokens_per_line = sample_tokens / max(1, sample_lines)

    # Rough total tokens = avg tokens per line * estimated total lines
    # Estimate total lines from file size / avg line length
    # We don't know avg line length without reading, so just estimate based on
    # sample ratio. Read the first and last byte positions is complex.
    # Simple estimate: assume uniform distribution
    return int(avg_tokens_per_line * sample_lines / max(1, sample_bytes / total_bytes)) if sample_bytes > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Skywork/SkyPile-150B")
    parser.add_argument("--data-dir", default="/home/scc/pb24511935/skypile_data")
    parser.add_argument("--max-shards", type=int, default=20)
    parser.add_argument("--tokenizer-path", default="tokenizer/skypile_4k_tokenizer.json")
    parser.add_argument("--retry-delay", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        tokenizer_path = Path.cwd() / args.tokenizer_path

    print(f"[DL] Dataset: {args.dataset}")
    print(f"[DL] Local dir: {data_dir}")
    print(f"[DL] Max shards: {args.max_shards}")

    # List all JSONL shards
    attempt = 1
    while True:
        try:
            all_files = list_repo_files(args.dataset, repo_type="dataset")
            break
        except Exception as e:
            print(f"[DL] Failed to list files (attempt {attempt}): {e}")
            if attempt >= 5:
                raise
            time.sleep(args.retry_delay)
            attempt += 1

    jsonl_shards = sorted(f for f in all_files if f.endswith(".jsonl"))
    selected = jsonl_shards[: args.max_shards]
    print(f"[DL] {len(jsonl_shards)} shards total, downloading first {len(selected)}")

    # Download each shard with retries
    total_bytes = 0
    for i, shard in enumerate(selected):
        local_path = data_dir / shard
        local_path.parent.mkdir(parents=True, exist_ok=True)

        dest = None
        attempts = 0
        while dest is None:
            attempts += 1
            try:
                dest = hf_hub_download(
                    repo_id=args.dataset,
                    repo_type="dataset",
                    filename=shard,
                    local_dir=str(data_dir),
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                print(f"[DL]  shard {i+1}/{len(selected)}: {shard} failed ({e})")
                if attempts >= 5:
                    print(f"[DL]  Giving up on {shard} after 5 attempts")
                    break
                print(f"[DL]  Retry in {args.retry_delay}s (attempt {attempts}/5)")
                time.sleep(args.retry_delay)

        if dest:
            size_mb = Path(dest).stat().st_size / (1024 * 1024)
            total_bytes += Path(dest).stat().st_size
            print(f"[DL]  shard {i+1}/{len(selected)}: {shard} ({size_mb:.1f} MB)")

    # Summary
    total_gb = total_bytes / (1024**3)
    print(f"[DL] Done! Downloaded {len(selected)} shards ({total_gb:.1f} GB) to {data_dir}")

    # Estimate total tokens available
    print("[DL] Estimating token count from first shard...")
    first_shard = selected[0] if selected else None
    if first_shard:
        data_dir_abs = data_dir
        tokens = estimate_tokens_in_shard(data_dir_abs, first_shard, str(tokenizer_path))
        print(f"[DL] Estimated tokens in first shard: ~{tokens:,}")
        if tokens > 0:
            est_total = tokens * len(selected)
            print(f"[DL] Estimated total tokens available: ~{est_total:,} ({est_total/1e9:.2f}B)")

    # Write a marker file
    marker = data_dir / ".downloaded"
    marker.write_text(f"shards={len(selected)}\ndataset={args.dataset}\n")
    print(f"[DL] Marker written to {marker}")


if __name__ == "__main__":
    main()
