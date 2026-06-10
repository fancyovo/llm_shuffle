"""
Monitor and plot training loss from CSV files.

Usage:
  # Monitor a single run (re-plots every 30s if CSV is updating)
  python scripts/plot_loss.py --run runs/random_3b

  # Compare multiple runs on one plot
  python scripts/plot_loss.py --run runs/random_3b runs/shuffle_then_normal_3b

  # One-shot plot (no monitoring)
  python scripts/plot_loss.py --run runs/random_3b --no-watch

  # Save plot to file instead of showing
  python scripts/plot_loss.py --run runs/random_3b --save loss_plot.png
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, works on headless servers

import matplotlib.pyplot as plt  # noqa: E402


def read_csv(path: Path) -> list[dict[str, float]]:
    """Read a loss CSV into a list of rows (numeric fields only)."""
    rows: list[dict[str, float]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {k: float(v) for k, v in row.items() if v.strip()}
            )
    return rows


def plot_runs(
    run_dirs: list[Path],
    save_path: str | None = None,
) -> None:
    """Plot loss curves for one or more runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    styles = ["-", "--", ":"]
    has_data = False

    for i, run_dir in enumerate(run_dirs):
        csv_path = run_dir / "loss.csv"
        if not csv_path.exists():
            print(f"[WARN] no loss.csv found in {run_dir}")
            continue
        rows = read_csv(csv_path)
        if not rows:
            print(f"[WARN] loss.csv in {run_dir} is empty")
            continue
        has_data = True
        label = run_dir.name
        steps = [r["step"] for r in rows]
        tokens = [r["tokens_seen"] / 1e9 for r in rows]
        loss = [r["loss"] for r in rows]
        style = styles[i % len(styles)]

        ax1.plot(steps, loss, style, label=label)
        ax2.plot(tokens, loss, style, label=label)

    if not has_data:
        ax1.text(
            0.5, 0.5, "No data yet", ha="center", va="center", transform=ax1.transAxes
        )
        ax2.text(
            0.5, 0.5, "No data yet", ha="center", va="center", transform=ax2.transAxes
        )

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Tokens Seen (B)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss vs Tokens")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Training Loss", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Plot saved to {save_path}")
    else:
        plt.savefig("_loss_plot.png", dpi=150, bbox_inches="tight")
        print(f"[OK] Plot saved to _loss_plot.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor and plot training loss")
    parser.add_argument("--run", nargs="+", required=True, help="Run directories")
    parser.add_argument("--save", default=None, help="Save plot to file")
    parser.add_argument(
        "--no-watch", action="store_true", help="One-shot plot, no monitoring"
    )
    parser.add_argument(
        "--interval", type=float, default=30.0, help="Monitor refresh interval (s)"
    )
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.run]
    save_path = args.save

    if args.no_watch:
        plot_runs(run_dirs, save_path=save_path)
        return

    # Monitor mode: re-plot on every interval
    print(f"[MONITOR] Watching {[str(d) for d in run_dirs]}, refreshing every {args.interval}s")
    print("[MONITOR] Press Ctrl+C to stop")
    last_sizes: dict[Path, int] = {d: 0 for d in run_dirs}

    try:
        while True:
            changed = False
            for d in run_dirs:
                csv_path = d / "loss.csv"
                if csv_path.exists():
                    s = csv_path.stat().st_size
                    if s != last_sizes[d]:
                        changed = True
                        last_sizes[d] = s
            if changed:
                plot_runs(run_dirs, save_path=save_path)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[MONITOR] Stopped")


if __name__ == "__main__":
    main()
