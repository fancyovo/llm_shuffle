# Experiment 003: 3B Token-ID Shuffle Dynamics

## Summary

| Item | Value |
|------|-------|
| Start date | 2026-06-11 |
| Status | Running |
| Purpose | Study token-id shuffle pretraining dynamics beyond the 1B-token region |
| Scope | Shuffle mode only; no downstream normal-training phase yet |
| LR schedule | 100M-token warmup, then constant `3e-4` |
| Output root | `runs_shuffle_dynamics/` |

The previous constant-LR run showed that the 1B shuffle-pretrain loss was still
in a transition region and had not clearly converged. This experiment trains the
token-id shuffle objective from scratch for 3B reported optimizer tokens and
records the loss curve.

## Tracked Inputs

| Role | Path |
|------|------|
| Config | `configs/experiments_shuffle_dynamics/shuffle_pretrain_3b.yaml` |
| Sbatch | `sbatch/shuffle_pretrain_3b_dynamics.sbatch` |
| Data code | `src/llm_shuffle/data.py` |
| Train code | `src/llm_shuffle/train.py` |

## Data Pipeline Fix

Before launching the formal run, the local JSONL data pipeline was changed so
each DDP rank receives a different shard:

```text
TokenStream(..., rank=rank, world_size=world_size)
ds = ds.shard(num_shards=world_size, index=rank)
```

For local JSONL, map-style `Dataset.shuffle(seed)` is intentionally not used.
It produced random Arrow/cache access and slowed training substantially. The
rank-sharding check on the login node only constructed token streams and did not
instantiate or run a model:

```text
rank0_tokens 2000
rank1_tokens 2000
rank0_eq_rank1_first_2000 False
```

## Launch Log

### Abandoned slow launch

```text
job_id: 15692
output_dir: runs_shuffle_dynamics/shuffle_pretrain_3b
status: cancelled after 51 steps
reason: local map-style Dataset.shuffle(seed) caused slow random-access data reads
observed throughput: about 60k-100k tok/s after warmup
comparison baseline: previous jobs usually reached about 190k-210k tok/s
```

This job is not part of the experiment result.

### Formal launch

```text
job_id: 15693
output_dir: runs_shuffle_dynamics/shuffle_pretrain_3b_v2
status: running
early throughput:
  step 2: 197539 tok/s
  step 3: 193710 tok/s
  step 4: 198103 tok/s
  step 5: 198126 tok/s
```

The early throughput is back in the expected range and matches earlier jobs.

## Training Settings

| Parameter | Value |
|-----------|-------|
| Mode | `shuffle` |
| Init checkpoint | `null` |
| Target tokens | 3,000,000,000 |
| Tokens per step | 524,288 |
| Expected target steps | About 5,723 |
| Precision | bf16 |
| Optimizer | AdamW, beta1=0.9, beta2=0.95, wd=0.1 |
| LR | Warm up to `3e-4`, then constant |
| Shuffle scope | New token-id permutation per 8192-token sequence |
| Output directory | `runs_shuffle_dynamics/shuffle_pretrain_3b_v2` |

## Results

Pending. Record final loss, smoothed loss behavior, notable phase transitions,
and any interruptions here after the job completes.

## Local Ignored Artifacts

```text
runs_shuffle_dynamics/shuffle_pretrain_3b/loss.csv
runs_shuffle_dynamics/shuffle_pretrain_3b_v2/loss.csv
runs_shuffle_dynamics/shuffle_pretrain_3b_v2/loss.png
logs/slurm/shufdyn-3b-15692.out
logs/slurm/shufdyn-3b-15692.err
logs/slurm/shufdyn-3b-15693.out
logs/slurm/shufdyn-3b-15693.err
```

## Caveats

- Resume currently reloads model and optimizer state but rebuilds the data
  iterator from the beginning. For this fresh launch, that does not affect the
  initial dynamics. If the job is interrupted and strict data continuation
  matters, implement explicit data-state resume before interpreting unique-token
  coverage.
- This experiment is not designed to compare generation quality. It only studies
  shuffle-objective training dynamics.
