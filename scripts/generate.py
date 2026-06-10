"""
Generate text from a trained model checkpoint.

Usage:
  source venv/bin/activate
  export PYTHONPATH="$PWD/src"
  python scripts/generate.py --checkpoint runs/shuffle_then_normal_3b/checkpoints/latest.pt --prompt "你好"
"""

from __future__ import annotations

import argparse

import torch
from tokenizers import Tokenizer

from llm_shuffle.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/model_50m.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/skypile_4k_tokenizer.json")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GEN] device={device}")

    # Load tokenizer and fix byte-level decode
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.decoder is None:
        from tokenizers.decoders import ByteLevel
        tokenizer.decoder = ByteLevel()
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()
    print(f"[GEN] vocab_size={vocab_size}")

    # Build model and load checkpoint
    from llm_shuffle.config import load_config
    cfg = load_config(args.config)
    model = build_model(cfg["model"]).to(device)
    model.set_gradient_checkpointing(False)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    raw = model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    raw.load_state_dict(ckpt["model"])
    step = ckpt.get("step", "?")
    print(f"[GEN] loaded checkpoint step={step}")

    # Tokenize prompt
    prompt = args.prompt or ""
    if prompt:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    else:
        input_ids = [bos_id]
    input_ids = [bos_id] + input_ids if input_ids[0] != bos_id else input_ids

    print(f"[GEN] prompt='{prompt}' -> {len(input_ids)} tokens")

    # Autoregressive generate
    generated = list(input_ids)
    seq_len = int(cfg["model"]["max_seq_len"])

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            # Truncate to model's max context
            ctx = generated[-seq_len:]
            ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)

            logits, _ = model(ctx_t, labels=None)
            next_logits = logits[0, -1, :] / args.temperature

            # Top-K filtering
            if args.top_k > 0:
                topk_vals, _ = torch.topk(next_logits, args.top_k)
                threshold = topk_vals[-1]
                next_logits[next_logits < threshold] = float("-inf")

            # Top-P (nucleus) filtering
            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1)[0])

            if next_id == eos_id:
                generated.append(next_id)
                break
            generated.append(next_id)

    # Decode
    output_text = tokenizer.decode(generated, skip_special_tokens=False)
    print(f"\n{'='*60}")
    print(f"GENERATED TEXT ({len(generated)} tokens):")
    print(f"{'='*60}")
    print(output_text)

    # Also decode without special tokens for readability
    output_clean = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"GENERATED (no special tokens):")
    print(f"{'='*60}")
    print(output_clean)


if __name__ == "__main__":
    main()
