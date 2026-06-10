"""
Comparison plot: random_3b vs shuffle_then_normal_3b with EMA smoothing and log x-axis.

Usage:
  source venv/bin/activate
  python scripts/plot_comparison.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def read_csv(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if v.strip()})
    if not rows:
        return {"step": np.array([]), "tokens": np.array([]), "loss": np.array([])}
    return {
        "step": np.array([r["step"] for r in rows]),
        "tokens": np.array([r["tokens_seen"] for r in rows]),
        "loss": np.array([r["loss"] for r in rows]),
    }


def ema(data: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average. alpha=1 means no smoothing."""
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def auto_ema_alpha(n_points: int, target_span: float = 0.02) -> float:
    """Pick alpha so that the EMA window covers ~target_span fraction of data."""
    if n_points <= 1:
        return 1.0
    window = max(1, int(n_points * target_span))
    return min(1.0, 2.0 / (window + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=[
        "runs/random_3b",
        "runs/shuffle_then_normal_3b",
    ])
    parser.add_argument("--save", default="runs/comparison.png")
    parser.add_argument("--ema-alpha", type=float, default=None,
                        help="EMA alpha (0-1, auto if not set)")
    parser.add_argument("--ema-span", type=float, default=0.02,
                        help="EMA window as fraction of total steps (used when alpha is auto)")
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    colors = {"random_3b": "#E24A33", "shuffle_then_normal_3b": "#348ABD"}
    labels = {
        "random_3b": "Random init → 3B normal",
        "shuffle_then_normal_3b": "Shuffle pretrain → 3B normal",
    }
    final_losses = {}

    for run_dir in args.runs:
        path = Path(run_dir) / "loss.csv"
        if not path.exists():
            print(f"[WARN] {path} not found")
            continue
        data = read_csv(path)
        if len(data["loss"]) == 0:
            print(f"[WARN] {path} is empty")
            continue

        name = Path(run_dir).name
        color = colors.get(name, "#333333")
        label = labels.get(name, name)
        tokens_b = data["tokens"] / 1e9
        loss = data["loss"]
        final_losses[name] = float(loss[-1])

        # EMA smoothing
        alpha = args.ema_alpha if args.ema_alpha is not None else auto_ema_alpha(len(loss), args.ema_span)
        loss_smooth = ema(loss, alpha)

        ax.plot(tokens_b, loss, color=color, linewidth=0.3, alpha=0.3)
        ax.plot(tokens_b, loss_smooth, color=color, label=label, linewidth=1.8)

        print(f"  {name}: final loss={loss[-1]:.4f}, ema_alpha={alpha:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("Tokens Seen (B)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss: Random Init vs Shuffle Pretrain", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.xaxis.set_minor_formatter(mticker.FormatStrFormatter("%.1f"))

    # Annotation box with final losses
    text = "\n".join(
        f"{labels.get(n, n)}: {loss:.4f}"
        for n, loss in final_losses.items()
    )
    ax.text(0.97, 0.03, text, transform=ax.transAxes, fontsize=10,
            va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"[OK] Comparison plot saved to {args.save}")


if __name__ == "__main__":
    main()
