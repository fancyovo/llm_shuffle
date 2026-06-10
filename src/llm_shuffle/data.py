from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer


@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor


class TokenStream:
    def __init__(
        self,
        dataset_name: str,
        split: str,
        text_field: str,
        tokenizer_path: str,
        streaming: bool,
        shuffle_buffer_size: int,
        seed: int,
        data_dir: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.eos_id = self.tokenizer.token_to_id("<eos>")

        local_path = Path(data_dir) if data_dir else None
        if local_path and local_path.exists():
            jsonl_pattern = str(local_path / "data" / "*.jsonl")
            ds = load_dataset("json", data_files=jsonl_pattern, split="train", streaming=False)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=streaming)
            if shuffle_buffer_size > 0:
                ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
        self.dataset = ds

    def __iter__(self) -> Iterator[int]:
        for row in self.dataset:
            text = row.get(self.text_field)
            if not text:
                continue
            ids = self.tokenizer.encode(str(text), add_special_tokens=False).ids
            if ids:
                yield from ids
                if self.eos_id is not None:
                    yield self.eos_id


def make_lm_batches(
    token_iter: Iterator[int],
    seq_len: int,
    micro_batch_size: int,
    shuffle_tokens: bool,
    first_permutable_id: int = 4,
    last_permutable_id: int = 4095,
) -> Iterator[Batch]:
    chunk_tokens = seq_len + 1
    while True:
        batch_inputs: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for _ in range(micro_batch_size):
            ids: list[int] = []
            try:
                for _ in range(chunk_tokens):
                    ids.append(next(token_iter))
            except StopIteration:
                return
            if len(ids) != chunk_tokens:
                return
            input_ids = ids[:-1]
            labels = ids[1:]
            if shuffle_tokens:
                input_ids, labels = apply_sequence_permutation(
                    input_ids,
                    labels,
                    first_permutable_id=first_permutable_id,
                    last_permutable_id=last_permutable_id,
                )
            batch_inputs.append(input_ids)
            batch_labels.append(labels)
        yield Batch(
            input_ids=torch.tensor(batch_inputs, dtype=torch.long),
            labels=torch.tensor(batch_labels, dtype=torch.long),
        )


def apply_sequence_permutation(
    input_ids: list[int],
    labels: list[int],
    first_permutable_id: int,
    last_permutable_id: int,
) -> tuple[list[int], list[int]]:
    permutable = list(range(first_permutable_id, last_permutable_id + 1))
    shuffled = permutable[:]
    random.shuffle(shuffled)
    mapping = dict(zip(permutable, shuffled))

    def map_id(token_id: int) -> int:
        return mapping.get(token_id, token_id)

    return [map_id(x) for x in input_ids], [map_id(x) for x in labels]
