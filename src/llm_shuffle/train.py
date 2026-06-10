from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from llm_shuffle.config import load_config
from llm_shuffle.data import TokenStream, make_lm_batches
from llm_shuffle.model import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed(enabled: bool) -> None:
    if enabled:
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(tokens_seen: int, cfg: dict[str, Any]) -> float:
    base_lr = float(cfg["optimizer"]["learning_rate"])
    min_lr = float(cfg["optimizer"]["min_learning_rate"])
    warmup_tokens = int(cfg["scheduler"]["warmup_tokens"])
    scheduler_name = str(cfg.get("scheduler", {}).get("name", "cosine"))
    max_tokens = int(cfg["train"]["max_tokens"])
    if tokens_seen < warmup_tokens:
        return base_lr * tokens_seen / max(1, warmup_tokens)
    if scheduler_name in {"constant", "warmup_constant"}:
        return base_lr
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    progress = (tokens_seen - warmup_tokens) / max(1, max_tokens - warmup_tokens)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    cfg: dict[str, Any],
) -> None:
    raw_model = unwrap_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "config": cfg,
        },
        path,
    )
    latest = path.parent / "latest.pt"
    shutil.copyfile(path, latest)


def load_checkpoint(
    path: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    resume_training_state: bool = False,
) -> tuple[int, int]:
    if not path:
        return 0, 0
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_model = unwrap_model(model)
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and resume_training_state and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("step", 0)), int(ckpt.get("tokens_seen", 0))
    return 0, 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def truncate_csv_on_resume(csv_path: Path, resume_step: int) -> None:
    """Remove rows at or after resume_step to avoid duplicates on resume."""
    if not csv_path.exists() or resume_step == 0:
        return
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if int(r["step"]) < resume_step]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ddp, rank, local_rank, world_size = setup_distributed()
    set_seed(int(cfg["data"]["seed"]), rank)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    model = build_model(cfg["model"]).to(device)
    model.set_gradient_checkpointing(bool(cfg["train"]["gradient_checkpointing"]))
    output_dir = Path(cfg["runtime"]["output_dir"])
    resume_path = output_dir / "checkpoints" / "latest.pt"
    step = 0
    tokens_seen = 0

    if bool(cfg["train"].get("compile", False)) and device.type == "cuda":
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["learning_rate"]),
        betas=(float(cfg["optimizer"]["beta1"]), float(cfg["optimizer"]["beta2"])),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )

    if bool(cfg["runtime"].get("resume", True)) and resume_path.exists():
        step, tokens_seen = load_checkpoint(
            str(resume_path), model, optimizer, resume_training_state=True
        )
        if rank == 0 and step > 0:
            csv_path = output_dir / "loss.csv"
            truncate_csv_on_resume(csv_path, step)
            print(f"resumed step={step} tokens_seen={tokens_seen} csv_truncated_at_step={step}")
    else:
        init_checkpoint = cfg.get("experiment", {}).get("init_checkpoint")
        load_checkpoint(init_checkpoint, model, optimizer=None, resume_training_state=False)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print(f"params={count_parameters(unwrap_model(model)):,}")
        print(f"world_size={world_size} device={device} output_dir={output_dir}")

    token_stream = TokenStream(
        dataset_name=cfg["data"]["dataset"],
        split=cfg["data"]["split"],
        text_field=cfg["data"]["text_field"],
        tokenizer_path=cfg["data"]["tokenizer_path"],
        streaming=bool(cfg["data"]["streaming"]),
        shuffle_buffer_size=int(cfg["data"]["shuffle_buffer_size"]),
        seed=int(cfg["data"]["seed"]) + rank,
        data_dir=cfg["data"].get("data_dir"),
    )
    batch_iter = make_lm_batches(
        iter(token_stream),
        seq_len=int(cfg["train"]["seq_len"]),
        micro_batch_size=int(cfg["train"]["micro_batch_size"]),
        shuffle_tokens=cfg.get("experiment", {}).get("mode") == "shuffle",
        first_permutable_id=int(cfg.get("shuffle", {}).get("first_permutable_id", 4)),
        last_permutable_id=int(cfg.get("shuffle", {}).get("last_permutable_id", 4095)),
    )

    precision = cfg["train"]["precision"]
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    max_tokens = int(cfg["train"]["max_tokens"])
    grad_accum_steps = int(cfg["train"]["grad_accum_steps"])
    micro_tokens = (
        int(cfg["train"]["seq_len"])
        * int(cfg["train"]["micro_batch_size"])
        * world_size
    )
    csv_path = output_dir / "loss.csv"
    start_time = time.time()
    last_time = start_time

    while tokens_seen < max_tokens:
        if args.max_steps is not None and step >= args.max_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            batch = next(batch_iter)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    _, loss = model(input_ids, labels)
            else:
                _, loss = model(input_ids, labels)
            assert loss is not None
            loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += float(loss.detach().cpu())

        if ddp:
            loss_tensor = torch.tensor(accum_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            accum_loss = float(loss_tensor.cpu())

        lr = get_lr(tokens_seen, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["optimizer"]["grad_clip"]))
        optimizer.step()

        step += 1
        tokens_seen += micro_tokens * grad_accum_steps
        now = time.time()
        elapsed = now - start_time
        step_time = now - last_time
        last_time = now
        throughput = micro_tokens * grad_accum_steps / max(step_time, 1e-9)

        if rank == 0 and step % int(cfg["logging"]["log_every"]) == 0:
            hours, rem = divmod(int(elapsed), 3600)
            minutes, secs = divmod(rem, 60)
            print(
                f"step={step} tokens={tokens_seen} loss={accum_loss:.6f} "
                f"lr={lr:.6e} tok_s={throughput:.1f} "
                f"elapsed={hours}h{minutes:02d}m{secs:02d}s",
                flush=True,
            )
        if rank == 0 and step % int(cfg["logging"]["csv_every"]) == 0:
            append_csv(
                csv_path,
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "loss": accum_loss,
                    "lr": lr,
                    "tokens_per_second": throughput,
                    "elapsed_seconds": elapsed,
                },
            )
        if rank == 0 and step % int(cfg["logging"]["save_every"]) == 0:
            ckpt = output_dir / "checkpoints" / f"step_{step:07d}.pt"
            save_checkpoint(ckpt, model, optimizer, step, tokens_seen, cfg)

    if rank == 0:
        save_checkpoint(output_dir / "checkpoints" / f"step_{step:07d}.pt", model, optimizer, step, tokens_seen, cfg)
    cleanup_distributed(ddp)


if __name__ == "__main__":
    main()
