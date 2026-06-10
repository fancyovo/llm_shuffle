"""
Evaluate both models on multiple prompts with identical sampling params.

Usage:
  source venv/bin/activate && export PYTHONPATH="$PWD/src"
  python scripts/eval_prompts.py --out runs/eval_results.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from llm_shuffle.config import load_config
from llm_shuffle.model import build_model


PROMPTS = [
    # 事实知识
    "中国的首都是",
    # 解释说明
    "光合作用是指",
    # 创意写作
    "从前有一座山,山上有一座庙,庙里有一个",
    # 常识推理
    "如果明天要下雨,那么我应该",
    # 技术解释
    "计算机操作系统的主要功能是",
    # 开放讨论
    "关于环境保护,我认为",
]


def load_model_and_tokenizer(checkpoint: str, config: str, tokenizer_path: str, device: torch.device):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.decoder is None:
        tokenizer.decoder = ByteLevel()

    cfg = load_config(config)
    model = build_model(cfg["model"]).to(device)
    model.set_gradient_checkpointing(False)
    model.eval()

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    raw = model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    raw.load_state_dict(ckpt["model"])
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int,
             temperature: float, top_k: int, top_p: float, device: torch.device) -> str:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    seq_len = model.cfg.max_seq_len

    input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_ids = [bos_id] + input_ids if not input_ids or input_ids[0] != bos_id else input_ids
    generated = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = generated[-seq_len:]
            ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)

            logits, _ = model(ctx_t, labels=None)
            next_logits = logits[0, -1, :] / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[-1]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                next_logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1)[0])

            if next_id == eos_id:
                generated.append(next_id)
                break
            generated.append(next_id)

    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs=2, default=[
        "runs/random_3b/checkpoints/latest.pt",
        "runs/shuffle_then_normal_3b/checkpoints/latest.pt",
    ])
    parser.add_argument("--labels", nargs=2, default=["random_3b", "shuffle_then_normal_3b"])
    parser.add_argument("--config", default="configs/model_50m.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/skypile_4k_tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--out", default="runs/eval_results.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    lines: list[str] = []
    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log(f"device={device} temperature={args.temperature} top_k={args.top_k} top_p={args.top_p}")
    log(f"seed={args.seed} max_new_tokens={args.max_new_tokens}")
    log("=" * 70)

    for ckpt_path, label in zip(args.checkpoints, args.labels):
        log(f"\n{'=' * 70}")
        log(f"MODEL: {label}")
        log(f"checkpoint: {ckpt_path}")
        log(f"{'=' * 70}")

        model, tokenizer = load_model_and_tokenizer(ckpt_path, args.config, args.tokenizer, device)

        for i, prompt in enumerate(PROMPTS):
            log(f"\n--- Prompt {i+1}: \"{prompt}\" ---")
            output = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )
            log(output)
            log("")

    # Save to file
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        log(f"\n[OK] Results saved to {args.out}")


if __name__ == "__main__":
    main()
