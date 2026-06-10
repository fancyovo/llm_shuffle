"""Analyze blind A/B judgements and simple repetition metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def chars(text: str) -> list[str]:
    return [ch for ch in text if not ch.isspace()]


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    if len(items) < n:
        return []
    return [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]


def text_metrics(text: str) -> dict[str, float | int | bool]:
    c = chars(text)
    grams4 = ngrams(c, 4)
    grams8 = ngrams(c, 8)
    c4 = Counter(grams4)
    c8 = Counter(grams8)
    distinct4 = len(c4) / max(1, len(grams4))
    repeat4_rate = sum(v - 1 for v in c4.values() if v > 1) / max(1, len(grams4))
    max4 = max(c4.values(), default=0)
    max8 = max(c8.values(), default=0)
    degenerate = len(c) < 20 or distinct4 < 0.35 or repeat4_rate > 0.35 or max4 >= 8 or max8 >= 4
    return {
        "char_len": len(c),
        "distinct4": distinct4,
        "repeat4_rate": repeat4_rate,
        "max_repeat4": max4,
        "max_repeat8": max8,
        "degenerate": degenerate,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def binomial_two_sided_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default="eval/results/generations_v1.jsonl")
    parser.add_argument("--judgements", default="eval/results/deepseek_judgements_v1.jsonl")
    parser.add_argument("--out", default="eval/results/summary_v1.md")
    args = parser.parse_args()

    generations = read_jsonl(Path(args.generations))
    judgements = read_jsonl(Path(args.judgements))
    if not generations:
        raise SystemExit("no generations found")
    labels = list(generations[0]["outputs"])

    metric_rows: dict[str, list[dict[str, float | int | bool]]] = {label: [] for label in labels}
    by_category: dict[str, dict[str, list[dict[str, float | int | bool]]]] = defaultdict(lambda: defaultdict(list))
    for row in generations:
        for label, text in row["outputs"].items():
            metrics = text_metrics(text)
            metric_rows[label].append(metrics)
            by_category[row.get("category", "unknown")][label].append(metrics)

    wins = Counter(j["winner_label"] for j in judgements)
    model_wins = {label: wins[label] for label in labels}
    ties = wins["Tie"]
    both_bad = wins["BothBad"]
    decisive = sum(model_wins.values())
    preferred = max(labels, key=lambda label: model_wins[label])
    p_value = binomial_two_sided_p(model_wins[labels[0]], model_wins[labels[1]])
    if decisive:
        ci_low, ci_high = wilson_ci(model_wins[preferred], decisive)
    else:
        ci_low, ci_high = float("nan"), float("nan")

    lines = [
        "# Blind Generation Evaluation Summary",
        "",
        f"- Generation pairs: {len(generations)}",
        f"- Judged pairs: {len(judgements)}",
        f"- Models: {', '.join(labels)}",
        "",
        "## DeepSeek Blind A/B",
        "",
        "| outcome | count |",
        "|---|---:|",
    ]
    for label in labels:
        lines.append(f"| {label} wins | {model_wins[label]} |")
    lines.extend(
        [
            f"| Tie | {ties} |",
            f"| BothBad | {both_bad} |",
            "",
            f"Decisive comparisons: {decisive}",
            f"Preferred model by wins: {preferred}",
            f"Preferred decisive win rate: {model_wins[preferred] / decisive:.4f}" if decisive else "Preferred decisive win rate: n/a",
            f"Wilson 95% CI: [{ci_low:.4f}, {ci_high:.4f}]" if decisive else "Wilson 95% CI: n/a",
            f"Two-sided sign-test p-value: {p_value:.6g}" if decisive else "Two-sided sign-test p-value: n/a",
            "",
            "## Automatic Repetition Metrics",
            "",
            "| model | degenerate_rate | distinct4 | repeat4_rate | max_repeat4 | char_len |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in labels:
        rows = metric_rows[label]
        lines.append(
            "| {label} | {deg:.4f} | {d4:.4f} | {r4:.4f} | {m4:.2f} | {cl:.1f} |".format(
                label=label,
                deg=mean([float(r["degenerate"]) for r in rows]),
                d4=mean([float(r["distinct4"]) for r in rows]),
                r4=mean([float(r["repeat4_rate"]) for r in rows]),
                m4=mean([float(r["max_repeat4"]) for r in rows]),
                cl=mean([float(r["char_len"]) for r in rows]),
            )
        )

    lines.extend(["", "## By Category", ""])
    for category in sorted(by_category):
        lines.append(f"### {category}")
        lines.append("")
        lines.append("| model | degenerate_rate | distinct4 | repeat4_rate |")
        lines.append("|---|---:|---:|---:|")
        for label in labels:
            rows = by_category[category][label]
            lines.append(
                "| {label} | {deg:.4f} | {d4:.4f} | {r4:.4f} |".format(
                    label=label,
                    deg=mean([float(r["degenerate"]) for r in rows]),
                    d4=mean([float(r["distinct4"]) for r in rows]),
                    r4=mean([float(r["repeat4_rate"]) for r in rows]),
                )
            )
        lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
