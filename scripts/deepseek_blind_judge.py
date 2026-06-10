"""
Run blind A/B judging with a DeepSeek-compatible chat completions API.

The API key must be provided via an environment variable. Do not put keys in
command lines, sbatch files, or repository files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """你是严格的中文文本质量盲评员。你只比较两个候选续写，不知道模型来源。请优先判断是否贴合提示词、是否连贯、是否避免重复循环、语法是否自然、信息量是否更好。不要偏好更长文本；如果两者都明显失败，选择 BothBad；如果差异很小，选择 Tie。只输出 JSON。"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["pair_id"] for row in read_jsonl(path)}


def make_ab(row: dict[str, Any], blind_seed: int) -> tuple[str, str, str, str]:
    labels = list(row["outputs"])
    rng = random.Random(f"{blind_seed}:{row['pair_id']}")
    rng.shuffle(labels)
    label_a, label_b = labels
    return label_a, row["outputs"][label_a], label_b, row["outputs"][label_b]


def build_user_prompt(row: dict[str, Any], text_a: str, text_b: str) -> str:
    return f"""请盲评下面两个中文续写。只输出严格 JSON，不要使用 Markdown。

Prompt:
{row["prompt"]}

Answer A:
{text_a}

Answer B:
{text_b}

输出格式：
{{"winner":"A|B|Tie|BothBad","confidence":1-5,"reason":"不超过80字的中文理由"}}"""


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_chat(
    *,
    api_key: str,
    api_base: str,
    model: str,
    user_prompt: str,
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    parsed = parse_json_response(content)
    parsed["_raw_content"] = content
    parsed["_usage"] = body.get("usage")
    return parsed


def normalize_winner(value: Any) -> str:
    winner = str(value).strip()
    aliases = {
        "a": "A",
        "b": "B",
        "tie": "Tie",
        "draw": "Tie",
        "bothbad": "BothBad",
        "both_bad": "BothBad",
        "both bad": "BothBad",
    }
    return aliases.get(winner.lower(), winner)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default="eval/results/generations_v1.jsonl")
    parser.add_argument("--out", default="eval/results/deepseek_judgements_v1.jsonl")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-key-stdin", action="store_true", help="Read API key from hidden stdin prompt instead of environment.")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--blind-seed", type=int, default=20260610)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args()

    if args.api_key_stdin:
        api_key = getpass.getpass("DeepSeek API key: ")
    else:
        api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")

    rows = read_jsonl(Path(args.generations))
    if args.limit is not None:
        rows = rows[: args.limit]
    done = done_ids(Path(args.out))
    pending = [row for row in rows if row["pair_id"] not in done]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    def judge_one(index_and_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, row = index_and_row
        label_a, text_a, label_b, text_b = make_ab(row, args.blind_seed)
        prompt = build_user_prompt(row, text_a, text_b)
        last_error = ""
        result: dict[str, Any] | None = None
        for attempt in range(args.max_retries):
            try:
                result = call_chat(
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    user_prompt=prompt,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last_error = repr(exc)
                wait = min(30.0, 2.0**attempt)
                print(f"[retry] {row['pair_id']} attempt={attempt + 1} error={last_error}", flush=True)
                time.sleep(wait)
        if result is None:
            raise RuntimeError(f"failed to judge {row['pair_id']}: {last_error}")

        winner = normalize_winner(result.get("winner"))
        if winner not in {"A", "B", "Tie", "BothBad"}:
            winner = "Tie"
        return {
            "pair_id": row["pair_id"],
            "prompt_id": row["prompt_id"],
            "category": row.get("category", "unknown"),
            "prompt": row["prompt"],
            "seed": row["seed"],
            "judge_model": args.model,
            "blind_seed": args.blind_seed,
            "label_a": label_a,
            "label_b": label_b,
            "winner": winner,
            "winner_label": label_a if winner == "A" else label_b if winner == "B" else winner,
            "confidence": result.get("confidence"),
            "reason": result.get("reason", ""),
            "usage": result.get("_usage"),
            "_index": index,
        }

    print(f"[judge] pending={len(pending)} concurrency={args.concurrency}", flush=True)
    with Path(args.out).open("a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = [executor.submit(judge_one, item) for item in enumerate(pending, start=1)]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                out = future.result()
                index = out.pop("_index")
                with write_lock:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                print(f"[{completed}/{len(pending)}] source_index={index} {out['pair_id']} winner={out['winner']}", flush=True)
                if args.sleep > 0:
                    time.sleep(args.sleep)


if __name__ == "__main__":
    main()
