# Experiment 004: Normal 3B From 10B Shuffle Weights

## Summary

| Item | Value |
|------|-------|
| Start date | 2026-06-12 |
| Status | Running |
| Purpose | Test normal language-model training after long token-id shuffle pretraining |
| Scope | Load only 10B shuffle checkpoint weights, then train 3B normal tokens |
| LR schedule | 100M-token warmup, then constant `3e-4` |
| Output root | `runs_shuffle_dynamics/shuffle10b_then_normal_3b/` |

This experiment starts from the 10B checkpoint produced by Experiment 003 and
switches to normal next-token training without token-id shuffle. It intentionally
does not resume the shuffle optimizer, gradient state, step, or token counter.

## Tracked Inputs

| Role | Path |
|------|------|
| Config | `configs/experiments_shuffle_dynamics/shuffle10b_then_normal_3b.yaml` |
| Sbatch | `sbatch/shuffle10b_then_normal_3b.sbatch` |
| Init checkpoint | `runs_shuffle_dynamics/shuffle_pretrain_3b_v2/checkpoints/step_0019074.pt` |
| Data directory | `/home/scc/pb24511935/skypile_data_after_shuffle10b_normal3b` |
| Train code | `src/llm_shuffle/train.py` |
| Data code | `src/llm_shuffle/data.py` |

## Initialization Semantics

The output directory is new:

```text
runs_shuffle_dynamics/shuffle10b_then_normal_3b
```

On first launch, this directory has no `checkpoints/latest.pt`, so
`src/llm_shuffle/train.py` follows the `experiment.init_checkpoint` path:

```text
load_checkpoint(init_checkpoint, model, optimizer=None, resume_training_state=False)
```

That loads only `ckpt["model"]`. The optimizer is newly constructed for the
normal training run, and `step=0`, `tokens_seen=0`, and LR warmup start from the
new run. The sbatch refuses to start if a `latest.pt` already exists unless
`ALLOW_NORMAL_RESUME=1` is explicitly set, so a first launch cannot accidentally
inherit a previous normal-run optimizer state.

If this normal run is interrupted after it creates its own checkpoint, resume it
with:

```bash
sbatch --export=ALL,ALLOW_NORMAL_RESUME=1 sbatch/shuffle10b_then_normal_3b.sbatch
```

That resume is correct for this experiment, because the optimizer state belongs
to the normal 3B phase, not the previous shuffle phase.

## Data Selection

The 10B shuffle checkpoint was produced by 3B tokens on the original dynamics
data plus 7B continuation tokens from:

```text
/home/scc/pb24511935/skypile_data_continue_7b
```

The continuation directory was estimated at about 0.282022 tokenizer tokens per
byte. Seven billion continuation tokens therefore cover roughly the first 24.82
GB of that directory in sorted filename order, reaching around
`2020-45_zh_head_0002.jsonl`. To avoid training on data likely seen by the 10B
shuffle checkpoint, this experiment uses a symlink directory that starts at the
next shard:

```text
/home/scc/pb24511935/skypile_data_after_shuffle10b_normal3b/data/
  2020-45_zh_head_0003.jsonl
  2020-45_zh_head_0004.jsonl
  2020-45_zh_head_0005.jsonl
  2020-45_zh_head_0006.jsonl
  2020-45_zh_head_0007.jsonl
  2020-45_zh_head_0008.jsonl
  2020-45_zh_head_0009.jsonl
  2020-45_zh_head_0010.jsonl
  2020-45_zh_middle_0000.jsonl
  2020-45_zh_middle_0001.jsonl
  2020-45_zh_middle_0002.jsonl
  2020-45_zh_middle_0003.jsonl
  2020-45_zh_middle_0004.jsonl
```

This selected suffix contains about 16.41GB of JSONL, estimated at about 4.63B
tokens, enough for a 3B-token normal-training run with margin.

## Training Settings

| Parameter | Value |
|-----------|-------|
| Mode | `normal` |
| Token-id shuffle | Disabled |
| Init weights | Experiment 003 10B checkpoint |
| Optimizer state | New AdamW state |
| Target tokens | 3,000,000,000 |
| Tokens per step | 524,288 |
| Target steps | About 5,723 |
| Precision | bf16 |
| Optimizer | AdamW, beta1=0.9, beta2=0.95, wd=0.1 |
| LR | Warm up to `3e-4`, then constant |

## Launch Log

```text
job_id: 15862
status: running on anode05
sbatch: sbatch/shuffle10b_then_normal_3b.sbatch
```

First-step sanity check:

```text
params=49,820,160
world_size=2 device=cuda:0 output_dir=runs_shuffle_dynamics/shuffle10b_then_normal_3b
step=1 tokens=524,288 loss=6.177595 lr=0.000000e+00 tok_s=68960.9
step=2 tokens=1,048,576 loss=6.111021 lr=1.572864e-06 tok_s=198671.6
step=3 tokens=1,572,864 loss=6.142490 lr=3.145728e-06 tok_s=199545.8
```

This confirms the run started from `step=0`, with a fresh LR warmup and normal
training mode. It did not resume the 10B shuffle optimizer state.

## Local Ignored Artifacts

```text
runs_shuffle_dynamics/shuffle10b_then_normal_3b/
logs/slurm/shuf10b-norm3b-*.out
logs/slurm/shuf10b-norm3b-*.err
/home/scc/pb24511935/skypile_data_after_shuffle10b_normal3b/data/*.jsonl
```
