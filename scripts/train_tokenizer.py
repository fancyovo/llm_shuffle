from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Skywork/SkyPile-150B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", default="tokenizer/skypile_4k_tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-text-bytes", type=int, default=100_000_000)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--data-files", nargs="*", default=None)
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    return parser.parse_args()


def select_data_files(dataset: str, max_files: int) -> list[str] | None:
    if max_files <= 0:
        return None
    files = HfApi().list_repo_files(dataset, repo_type="dataset")
    jsonl_files = sorted(f for f in files if f.startswith("data/") and f.endswith(".jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No data/*.jsonl files found in dataset repo {dataset}")
    return jsonl_files[:max_files]


def text_iterator(args: argparse.Namespace) -> Iterator[str]:
    data_files = args.data_files
    if data_files is None:
        data_files = select_data_files(args.dataset, args.max_files)
    if data_files:
        print(f"using {len(data_files)} data files")
        for name in data_files[:10]:
            print(f"  {name}")
        if len(data_files) > 10:
            print("  ...")
    ds = load_dataset(
        args.dataset,
        split=args.split,
        data_files=data_files,
        streaming=args.streaming,
    )
    count = 0
    total_bytes = 0
    for row in ds:
        text = row.get(args.text_field)
        if text:
            text = str(text)
            text_bytes = len(text.encode("utf-8"))
            if args.max_text_bytes and total_bytes + text_bytes > args.max_text_bytes:
                remaining = args.max_text_bytes - total_bytes
                if remaining > 0:
                    partial = text.encode("utf-8")[:remaining].decode(
                        "utf-8", errors="ignore"
                    )
                    if partial:
                        yield partial
                total_bytes = args.max_text_bytes
                print(
                    f"stopped at max_text_bytes={args.max_text_bytes} "
                    f"read_docs={count} read_text_bytes={total_bytes}"
                )
                return
            yield text
            total_bytes += text_bytes
            count += 1
        if args.max_docs and count >= args.max_docs:
            print(f"stopped at max_docs={args.max_docs}")
            break
    print(f"read_docs={count} read_text_bytes={total_bytes}")


def main() -> None:
    args = parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(args.num_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    print(f"RAYON_NUM_THREADS={args.num_threads}")
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iterator(args), trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    print(f"saved tokenizer to {output}")
    for token in special_tokens:
        print(f"{token}: {tokenizer.token_to_id(token)}")


if __name__ == "__main__":
    main()
