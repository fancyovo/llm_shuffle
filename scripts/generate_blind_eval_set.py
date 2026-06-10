"""
Generate paired samples for blind model comparison.

The output JSONL contains both model labels for later analysis, but the judge
script randomizes A/B order before sending text to an external evaluator.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from llm_shuffle.config import load_config
from llm_shuffle.model import build_model


DEFAULT_MODELS = {
    "random_3b": "runs/random_3b/checkpoints/latest.pt",
    "shuffle_then_normal_3b": "runs/shuffle_then_normal_3b/checkpoints/latest.pt",
}


@dataclass
class Prompt:
    id: str
    category: str
    prompt: str


def read_prompts(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(
                Prompt(
                    id=str(row["id"]),
                    category=str(row.get("category", "unknown")),
                    prompt=str(row["prompt"]),
                )
            )
            if not prompts[-1].prompt:
                raise ValueError(f"empty prompt at {path}:{line_no}")
    return prompts


def load_model(checkpoint: str, config: str, device: torch.device):
    cfg = load_config(config)
    model = build_model(cfg["model"]).to(device)
    model.set_gradient_checkpointing(False)
    model.eval()

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.load_state_dict(ckpt["model"])
    return model, int(ckpt.get("step", 0))


def load_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    if tokenizer.decoder is None:
        tokenizer.decoder = ByteLevel()
    return tokenizer


def apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits = logits.masked_fill(logits < topk_vals[-1], float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits = logits.clone()
        logits[sorted_indices[remove]] = float("-inf")
    return logits


def generate(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> str:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    seq_len = model.cfg.max_seq_len
    input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_ids = [bos_id] + input_ids if not input_ids or input_ids[0] != bos_id else input_ids
    generated = list(input_ids)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            ctx = generated[-seq_len:]
            ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
            logits, _ = model(ctx_t, labels=None)
            next_logits = logits[0, -1, :] / max(temperature, 1.0e-6)
            next_logits = apply_top_k_top_p(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1)[0])
            generated.append(next_id)
            if next_id == eos_id:
                break

    return tokenizer.decode(generated, skip_special_tokens=True)


def read_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                keys.add(str(row["pair_id"]))
    return keys


def parse_models(values: list[str]) -> dict[str, str]:
    models: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"model must be label=checkpoint, got: {value}")
        label, checkpoint = value.split("=", 1)
        models[label] = checkpoint
    if len(models) != 2:
        raise ValueError("exactly two models are required")
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", default="eval/prompts_v1.jsonl")
    parser.add_argument("--out", default="eval/results/generations_v1.jsonl")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="label=checkpoint. Pass exactly two values. Defaults to random_3b and shuffle_then_normal_3b.",
    )
    parser.add_argument("--config", default="configs/model_50m.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/skypile_4k_tokenizer.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model_args = args.model or [f"{k}={v}" for k, v in DEFAULT_MODELS.items()]
    models = parse_models(model_args)
    labels = list(models)
    prompts = read_prompts(Path(args.prompt_file))
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = load_tokenizer(args.tokenizer)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = read_done_keys(Path(args.out))

    loaded: dict[str, Any] = {}
    ckpt_steps: dict[str, int] = {}
    for label, checkpoint in models.items():
        print(f"[load] {label}: {checkpoint}", flush=True)
        model, step = load_model(checkpoint, args.config, device)
        loaded[label] = model
        ckpt_steps[label] = step

    total = len(prompts) * len(args.seeds)
    written = 0
    with Path(args.out).open("a", encoding="utf-8") as f:
        for prompt_index, prompt in enumerate(prompts):
            for seed in args.seeds:
                pair_id = f"{prompt.id}__seed_{seed}"
                if pair_id in done:
                    continue
                outputs: dict[str, str] = {}
                for label in labels:
                    gen_seed = seed + prompt_index * 1000003
                    outputs[label] = generate(
                        loaded[label],
                        tokenizer,
                        prompt.prompt,
                        seed=gen_seed,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        device=device,
                    )
                row = {
                    "pair_id": pair_id,
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "prompt": prompt.prompt,
                    "seed": seed,
                    "models": labels,
                    "checkpoint_steps": ckpt_steps,
                    "sampling": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                    },
                    "outputs": outputs,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                print(f"[{written}/{total}] wrote {pair_id}", flush=True)

    random.shuffle(labels)
    print(f"[done] wrote {written} new pairs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
