from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from llm_shuffle.config import load_config
from llm_shuffle.model import build_model


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be label=checkpoint")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("empty model label")
    return label, Path(path)


def read_jsonl_texts(data_dir: Path) -> list[Path]:
    paths = sorted((data_dir / "data").glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No JSONL files under {data_dir / 'data'}")
    return paths


def load_long_token_sequence(
    data_dir: Path,
    tokenizer: Tokenizer,
    text_field: str,
    target_len: int,
    max_rows: int,
    start_file_index: int,
) -> tuple[list[int], str, dict[str, Any]]:
    eos_id = tokenizer.token_to_id("<eos>")
    all_ids: list[int] = []
    texts: list[str] = []
    rows_seen = 0
    source_files: list[str] = []

    paths = read_jsonl_texts(data_dir)
    if start_file_index < 0 or start_file_index >= len(paths):
        raise ValueError(f"start_file_index={start_file_index} outside 0..{len(paths)-1}")

    for path in paths[start_file_index:]:
        source_files.append(path.name)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                rows_seen += 1
                if rows_seen > max_rows:
                    raise RuntimeError(
                        f"Could not collect {target_len} tokens in {max_rows} rows"
                    )
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get(text_field)
                if not text:
                    continue
                ids = tokenizer.encode(str(text), add_special_tokens=False).ids
                if eos_id is not None:
                    ids = ids + [eos_id]
                if len(ids) >= target_len:
                    return ids[:target_len], str(text), {
                        "mode": "single_row",
                        "rows_seen": rows_seen,
                        "source_files": source_files,
                        "selected_file": path.name,
                        "start_file_index": start_file_index,
                        "selected_row_token_len": len(ids),
                    }
                all_ids.extend(ids)
                texts.append(str(text))
                if len(all_ids) >= target_len:
                    return all_ids[:target_len], "\n".join(texts), {
                        "mode": "concatenated_rows",
                        "rows_seen": rows_seen,
                        "source_files": source_files,
                        "start_file_index": start_file_index,
                        "selected_row_token_len": None,
                    }
    raise RuntimeError(f"Could not collect {target_len} tokens from {data_dir}")


def make_permutation(vocab_size: int, first_id: int, last_id: int, seed: int) -> tuple[list[int], list[int]]:
    mapping = list(range(vocab_size))
    ids = list(range(first_id, last_id + 1))
    shuffled = ids[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    for src, dst in zip(ids, shuffled):
        mapping[src] = dst
    inverse = [0] * vocab_size
    for src, dst in enumerate(mapping):
        inverse[dst] = src
    return mapping, inverse


def decode_token(tokenizer: Tokenizer, token_id: int, escape: bool = True) -> str:
    try:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return f"<decode_error:{token_id}>"
    if escape:
        return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return text


def token_category(text: str, token_id: int) -> str:
    if token_id < 4:
        return "special"
    if not text:
        return "empty"
    if text.isspace():
        return "space"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "cjk"
    if any(ch.isdigit() for ch in text):
        return "digit"
    if any(ch.isalpha() for ch in text):
        return "alpha"
    if any(ch in "，。！？；：、“”‘’（）《》,.!?;:()[]{}-_" for ch in text):
        return "punct"
    return "other"


def context_excerpt(tokenizer: Tokenizer, ids: list[int], pos: int, radius: int) -> str:
    start = max(0, pos - radius)
    end = min(len(ids), pos + radius + 1)
    text = tokenizer.decode(ids[start:end], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def load_model(checkpoint: Path, cfg: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_model(cfg["model"]).to(device)
    model.set_gradient_checkpointing(False)
    model.eval()
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def run_prefill(
    checkpoint: Path,
    cfg: dict[str, Any],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
    inverse_perm: list[int],
    device: torch.device,
) -> dict[str, Any]:
    model, ckpt = load_model(checkpoint, cfg, device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(input_ids, labels=None)
        logits = logits[0].float()
        target = labels[0]
        target_logits = logits.gather(1, target[:, None]).squeeze(1)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target[:, None]).squeeze(1)
        nll = -target_log_probs
        target_probs = target_log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        rank = (logits > target_logits[:, None]).sum(dim=-1) + 1
        top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)

    top_ids_cpu = top_ids.cpu().tolist()
    top_orig_ids = [[inverse_perm[int(x)] for x in row] for row in top_ids_cpu]
    out = {
        "checkpoint_step": int(ckpt.get("step", -1)),
        "checkpoint_tokens_seen": int(ckpt.get("tokens_seen", -1)),
        "nll": nll.cpu().tolist(),
        "target_prob": target_probs.cpu().tolist(),
        "entropy": entropy.cpu().tolist(),
        "rank": rank.cpu().tolist(),
        "top_ids": top_ids_cpu,
        "top_orig_ids": top_orig_ids,
        "top_probs": top_probs.cpu().tolist(),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def median(values: list[float]) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def summarize_subset(indices: list[int], metrics: dict[str, list[float]]) -> dict[str, float]:
    if not indices:
        return {"count": 0}
    nll = [metrics["nll"][i] for i in indices]
    prob = [metrics["target_prob"][i] for i in indices]
    rank = [metrics["rank"][i] for i in indices]
    entropy = [metrics["entropy"][i] for i in indices]
    return {
        "count": len(indices),
        "mean_nll": mean(nll),
        "median_nll": median(nll),
        "mean_target_prob": mean(prob),
        "median_target_prob": median(prob),
        "acc1": sum(1 for x in rank if x <= 1) / len(rank),
        "acc5": sum(1 for x in rank if x <= 5) / len(rank),
        "median_rank": median(rank),
        "mean_entropy": mean(entropy),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments_shuffle_dynamics/shuffle_pretrain_3b.yaml")
    parser.add_argument("--data-dir", default="/home/scc/pb24511935/skypile_data_continue_7b")
    parser.add_argument("--out-dir", default="eval/results/shuffle_prefill_analysis")
    parser.add_argument("--model", action="append", required=True, type=parse_model_arg)
    parser.add_argument("--tokenizer", default="tokenizer/skypile_4k_tokenizer.json")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--start-file-index", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=20260612)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--context-radius", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.decoder is None:
        from tokenizers.decoders import ByteLevel
        tokenizer.decoder = ByteLevel()

    total_len = args.seq_len + 1
    orig_ids, source_text, source_meta = load_long_token_sequence(
        Path(args.data_dir),
        tokenizer,
        args.text_field,
        total_len,
        args.max_rows,
        args.start_file_index,
    )
    vocab_size = tokenizer.get_vocab_size()
    perm, inv_perm = make_permutation(vocab_size, 4, vocab_size - 1, args.permutation_seed)
    permuted = [perm[x] for x in orig_ids]
    input_perm = torch.tensor([permuted[:-1]], dtype=torch.long)
    labels_perm = torch.tensor([permuted[1:]], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PREFILL] device={device} seq_len={args.seq_len} out_dir={out_dir}", flush=True)
    print(f"[PREFILL] source_meta={source_meta}", flush=True)

    source = {
        "data_dir": args.data_dir,
        "source_meta": source_meta,
        "seq_len": args.seq_len,
        "start_file_index": args.start_file_index,
        "permutation_seed": args.permutation_seed,
        "orig_prefix_text": source_text[:2000],
        "orig_token_ids_head": orig_ids[:64],
    }
    (out_dir / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")

    input_perm = input_perm.to(device)
    labels_perm = labels_perm.to(device)
    results: dict[str, dict[str, Any]] = {}
    for label, checkpoint in args.model:
        print(f"[PREFILL] loading {label}: {checkpoint}", flush=True)
        results[label] = run_prefill(
            checkpoint=checkpoint,
            cfg=cfg,
            input_ids=input_perm,
            labels=labels_perm,
            top_k=args.top_k,
            inverse_perm=inv_perm,
            device=device,
        )
        print(
            f"[PREFILL] done {label} step={results[label]['checkpoint_step']} "
            f"tokens={results[label]['checkpoint_tokens_seen']}",
            flush=True,
        )

    positions = list(range(args.seq_len))
    labels_order = [label for label, _ in args.model]
    orig_targets = orig_ids[1:]
    orig_inputs = orig_ids[:-1]
    raw_token_text = [decode_token(tokenizer, x, escape=False) for x in orig_targets]
    token_text = [x.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") for x in raw_token_text]
    token_counts = Counter(orig_targets)
    last_seen: dict[int, int] = {}
    categories: dict[str, list[int]] = defaultdict(list)
    row_meta: list[dict[str, Any]] = []
    for pos, target in enumerate(orig_targets):
        prev_pos = last_seen.get(target)
        distance = None if prev_pos is None else pos - prev_pos
        if distance is None:
            copy_bin = "never_seen"
        elif distance <= 4:
            copy_bin = "seen_le4"
        elif distance <= 16:
            copy_bin = "seen_le16"
        elif distance <= 64:
            copy_bin = "seen_le64"
        elif distance <= 256:
            copy_bin = "seen_le256"
        else:
            copy_bin = "seen_gt256"
        freq = token_counts[target]
        if freq == 1:
            freq_bin = "freq_1"
        elif freq <= 3:
            freq_bin = "freq_2_3"
        elif freq <= 10:
            freq_bin = "freq_4_10"
        else:
            freq_bin = "freq_gt10"
        pos_frac = pos / max(1, args.seq_len - 1)
        quartile = f"pos_q{min(4, int(pos_frac * 4) + 1)}"
        cat = token_category(raw_token_text[pos], target)
        for key in [copy_bin, freq_bin, quartile, f"cat_{cat}"]:
            categories[key].append(pos)
        if target == orig_inputs[pos]:
            categories["same_as_previous_input"].append(pos)
        row_meta.append(
            {
                "pos": pos,
                "target_orig_id": target,
                "target_text": token_text[pos],
                "category": cat,
                "copy_distance": distance,
                "copy_bin": copy_bin,
                "freq": freq,
                "freq_bin": freq_bin,
                "position_quartile": quartile,
            }
        )
        last_seen[target] = pos

    summary: dict[str, Any] = {
        "source": source,
        "models": {},
        "groups": {},
        "comparisons": {},
    }
    for label in labels_order:
        metrics = {
            "nll": results[label]["nll"],
            "target_prob": results[label]["target_prob"],
            "rank": [float(x) for x in results[label]["rank"]],
            "entropy": results[label]["entropy"],
        }
        summary["models"][label] = {
            "checkpoint_step": results[label]["checkpoint_step"],
            "checkpoint_tokens_seen": results[label]["checkpoint_tokens_seen"],
            **summarize_subset(positions, metrics),
        }
        summary["groups"][label] = {
            key: summarize_subset(idxs, metrics)
            for key, idxs in sorted(categories.items())
            if len(idxs) >= 10
        }

    rows: list[dict[str, Any]] = []
    for pos in positions:
        row: dict[str, Any] = dict(row_meta[pos])
        row["context_excerpt"] = context_excerpt(tokenizer, orig_ids, pos + 1, args.context_radius)
        for label in labels_order:
            row[f"{label}_nll"] = results[label]["nll"][pos]
            row[f"{label}_target_prob"] = results[label]["target_prob"][pos]
            row[f"{label}_rank"] = results[label]["rank"][pos]
            row[f"{label}_entropy"] = results[label]["entropy"][pos]
            row[f"{label}_top_orig_ids"] = json.dumps(results[label]["top_orig_ids"][pos], ensure_ascii=False)
            row[f"{label}_top_text"] = json.dumps(
                [decode_token(tokenizer, x) for x in results[label]["top_orig_ids"][pos]],
                ensure_ascii=False,
            )
            row[f"{label}_top_probs"] = json.dumps(results[label]["top_probs"][pos])
        rows.append(row)

    if len(labels_order) >= 2:
        for left, right in zip(labels_order, labels_order[1:]):
            improvements = [
                (results[left]["nll"][i] - results[right]["nll"][i], i)
                for i in positions
            ]
            summary["comparisons"][f"{left}_to_{right}"] = {
                "mean_nll_improvement": mean([x for x, _ in improvements]),
                "median_nll_improvement": median([x for x, _ in improvements]),
                "improved_fraction": sum(1 for x, _ in improvements if x > 0) / len(improvements),
                "top_improved_positions": [],
                "top_worsened_positions": [],
            }
            for key, ordered in [
                ("top_improved_positions", sorted(improvements, reverse=True)[:30]),
                ("top_worsened_positions", sorted(improvements)[:30]),
            ]:
                for delta, pos in ordered:
                    summary["comparisons"][f"{left}_to_{right}"][key].append(
                        {
                            "pos": pos,
                            "nll_delta_positive_is_better": delta,
                            "target_orig_id": orig_targets[pos],
                            "target_text": token_text[pos],
                            "copy_bin": row_meta[pos]["copy_bin"],
                            "freq_bin": row_meta[pos]["freq_bin"],
                            "category": row_meta[pos]["category"],
                            "context_excerpt": context_excerpt(tokenizer, orig_ids, pos + 1, args.context_radius),
                            f"{left}_prob": results[left]["target_prob"][pos],
                            f"{right}_prob": results[right]["target_prob"][pos],
                            f"{left}_rank": results[left]["rank"][pos],
                            f"{right}_rank": results[right]["rank"][pos],
                            f"{left}_top_text": [
                                decode_token(tokenizer, x) for x in results[left]["top_orig_ids"][pos]
                            ],
                            f"{right}_top_text": [
                                decode_token(tokenizer, x) for x in results[right]["top_orig_ids"][pos]
                            ],
                        }
                    )

    write_csv(out_dir / "per_position.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md: list[str] = ["# Shuffle Prefill Analysis", ""]
    md.append("## Models")
    md.append("")
    md.append("| label | step | tokens | mean_nll | median_nll | acc1 | acc5 | median_rank | mean_entropy |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in labels_order:
        s = summary["models"][label]
        md.append(
            f"| {label} | {s['checkpoint_step']} | {s['checkpoint_tokens_seen']} | "
            f"{s['mean_nll']:.4f} | {s['median_nll']:.4f} | {s['acc1']:.4f} | "
            f"{s['acc5']:.4f} | {s['median_rank']:.1f} | {s['mean_entropy']:.4f} |"
        )
    md.append("")
    md.append("## Pairwise Changes")
    for name, comp in summary["comparisons"].items():
        md.append("")
        md.append(f"### {name}")
        md.append(
            f"mean NLL improvement: {comp['mean_nll_improvement']:.4f}; "
            f"median: {comp['median_nll_improvement']:.4f}; "
            f"improved fraction: {comp['improved_fraction']:.4f}"
        )
        md.append("")
        md.append("Top improved positions:")
        for item in comp["top_improved_positions"][:10]:
            md.append(
                f"- pos {item['pos']}, delta {item['nll_delta_positive_is_better']:.4f}, "
                f"target `{item['target_text']}`, {item['copy_bin']}, {item['freq_bin']}, "
                f"{item['category']}, ranks {item.get(name.split('_to_')[0] + '_rank')} -> "
                f"{item.get(name.split('_to_')[1] + '_rank')}"
            )
        md.append("")
        md.append("Top worsened positions:")
        for item in comp["top_worsened_positions"][:10]:
            md.append(
                f"- pos {item['pos']}, delta {item['nll_delta_positive_is_better']:.4f}, "
                f"target `{item['target_text']}`, {item['copy_bin']}, {item['freq_bin']}, {item['category']}"
            )
    md.append("")
    md.append("## Output Files")
    md.append("")
    md.append("- `summary.json`: full aggregate and top-position details")
    md.append("- `per_position.csv`: per-position probabilities, ranks, entropy, and top-k predictions")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[PREFILL] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
