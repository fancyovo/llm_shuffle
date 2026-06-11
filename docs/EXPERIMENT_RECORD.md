# Experiment Record Index

This file is the high-level experiment ledger. Each complete attempt has a
separate detail page under `docs/experiments/`. Generated outputs, checkpoints,
plots, logs, and DeepSeek judgement files are intentionally ignored by Git; when
needed, those files are recorded as plain local paths in the detail pages.

## Global Setup

| Item | Value |
|------|-------|
| Model | 49,820,160 parameter decoder-only Transformer |
| Architecture | PreNorm, RMSNorm, SwiGLU, RoPE |
| Tokenizer | Fixed 4096-token byte-level BPE |
| Context length | 8192 |
| Batch | 524,288 tokens/optimizer step with 2 GPU DDP |
| Dataset | Skywork/SkyPile-150B Chinese local JSONL cache |
| Shuffle mode | Per-sequence random bijection over token IDs 4-4095 |
| Preserved IDs | 0 `<pad>`, 1 `<bos>`, 2 `<eos>`, 3 `<unk>` |

## Experiments

| ID | Dates | Status | Main Change | Detail |
|----|-------|--------|-------------|--------|
| 001 | 2026-06-09 to 2026-06-10 | Completed | Initial cosine-decay comparison: random 3B vs shuffle 1B plus normal 3B | [001_cosine_decay.md](experiments/001_cosine_decay.md) |
| 002 | 2026-06-10 to 2026-06-11 | Completed | Constant LR after warmup; shuffle 1B and normal 3B run in one job | [002_constant_lr.md](experiments/002_constant_lr.md) |
| 003 | 2026-06-11 onward | Running | Token-id shuffle pretraining dynamics for 3B shuffle tokens only, after DDP data sharding fix | [003_shuffle_dynamics_3b.md](experiments/003_shuffle_dynamics_3b.md) |

## Change Log

### 001 to 002

- Removed cosine LR decay for short few-billion-token runs.
- Kept 100M-token warmup, then held LR constant at `3e-4`.
- Put the experimental path's shuffle 1B phase and normal 3B phase into one
  Slurm job so the scheduler can queue the whole sequence.
- Wrote outputs under `runs_constant/` to avoid overwriting the initial run.

### 002 to 003

- Focus changed from downstream normal-training quality to the intrinsic
  training dynamics of token-id shuffle pretraining.
- Added DDP dataset sharding so rank 0 and rank 1 do not consume identical local
  JSONL examples.
- Avoided map-style `Dataset.shuffle()` on local JSONL because it caused random
  Arrow/cache access and cut throughput from about 200k tok/s to 60k-100k tok/s.
- Started a fresh 3B-token shuffle-only run under `runs_shuffle_dynamics/`.

## Headline Results

| Experiment | Loss Result | Generation Evaluation | Interpretation |
|------------|-------------|-----------------------|----------------|
| 001 | `shuffle_then_normal_3b` final loss 2.8280 vs `random_3b` 2.8660 | Force-choice DeepSeek favored shuffle 102-78, p=0.086; ~2.1B shuffle-normal checkpoint beat random 3B 105-75, p=0.030 | Suggestive generation advantage, but initial data pipeline had DDP duplication and LR decay |
| 002 | `shuffle_then_normal_3b_constant` final loss 2.8023 vs `random_3b_constant` 2.8689 | Force-choice DeepSeek 96-84, p=0.412; repetition metrics did not favor shuffle | Clear training-loss advantage; generation-quality advantage not confirmed |
| 003 | Pending | Not applicable yet | Measures whether shuffle-pretrain loss undergoes a later transition before/after 1B tokens |

## Important Caveats

- Experiments 001 and 002 used local JSONL loading without DDP rank sharding, so
  the two GPU ranks consumed duplicate text streams. Reported token counts are
  optimizer-token counts, not distinct text-token coverage.
- Token-id shuffle itself was enabled in the shuffle phases of all experiments;
  the discovered issue was data sharding/order, not disabled token permutation.
- Experiment 003 fixes the DDP duplication for new runs, but resume still
  rebuilds the data iterator from the start. If strict token-level data resume is
  required, implement and test explicit data-state checkpointing before relying
  on resumed data order.
- Do not commit `runs/`, `runs_constant/`, `runs_shuffle_dynamics/`, `logs/`,
  `eval/results/`, checkpoints, generated samples, judgement files, or API keys.
