# Experiment 003: Token-ID Shuffle Dynamics, 3B Plus 7B Continuation

## Summary

| Item | Value |
|------|-------|
| Start date | 2026-06-11 |
| Status | 3B phase completed; 7B continuation running |
| Purpose | Study token-id shuffle pretraining dynamics beyond the 1B-token region |
| Scope | Shuffle mode only; no downstream normal-training phase yet |
| LR schedule | 100M-token warmup, then constant `3e-4` |
| Output root | `runs_shuffle_dynamics/` |

The previous constant-LR run showed that the 1B shuffle-pretrain loss was still
in a transition region and had not clearly converged. This experiment trains the
token-id shuffle objective from scratch for 3B reported optimizer tokens, then
continues the same model and optimizer state for another 7B tokens on new data.

## Tracked Inputs

| Role | Path |
|------|------|
| Config | `configs/experiments_shuffle_dynamics/shuffle_pretrain_3b.yaml` |
| Continuation config | `configs/experiments_shuffle_dynamics/shuffle_pretrain_continue_7b.yaml` |
| Sbatch | `sbatch/shuffle_pretrain_3b_dynamics.sbatch` |
| Continuation sbatch | `sbatch/shuffle_pretrain_continue_7b.sbatch` |
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

### 3B phase completed

```text
job_id: 15693
final_step: 5723
reported_tokens: 3,000,500,224
final_loss: 6.514327
final_lr: 3.0e-4
final_throughput: 188737 tok/s
```

### 7B continuation launch

```text
job_id: 15774
status: running
config: configs/experiments_shuffle_dynamics/shuffle_pretrain_continue_7b.yaml
target_tokens: 10,000,000,000 cumulative tokens
resume_checkpoint: runs_shuffle_dynamics/shuffle_pretrain_3b_v2/checkpoints/latest.pt
data_dir: /home/scc/pb24511935/skypile_data_continue_7b
resume_log: resumed step=5723 tokens_seen=3000500224
first_continuation_step: step=5724 tokens=3001024512 loss=6.413648
```

This is a continuation run, not a restart. The config keeps
`runtime.output_dir` set to `runs_shuffle_dynamics/shuffle_pretrain_3b_v2`, so
the trainer resumes the saved model, optimizer, step, and token counter from
`checkpoints/latest.pt`. It changes only `data.data_dir` to the new shard
directory and raises `train.max_tokens` to 10B.

## New Data Check

The continuation data directory contains 36 JSONL shards, no filename overlap
with the original 20-shard directory:

```text
data_dir: /home/scc/pb24511935/skypile_data_continue_7b
files: 36
bytes: 41.86GB
overlap_with_original_skypile_data: 0 files
```

A 200MB prefix sample from `2020-45_zh_head_0000.jsonl` gave:

```text
sample_bytes: 209,717,038
sample_tokens: 59,144,843
tokens_per_byte: 0.282022
estimated_tokens_in_new_directory: 11.81B
```

The new directory is therefore sufficient for a 7B-token continuation with
comfortable margin. The earlier rough estimate based on the old 20-shard run was
too conservative because it did not measure tokenizer compression directly.

## Training Settings

| Parameter | Value |
|-----------|-------|
| Mode | `shuffle` |
| Init checkpoint | `null` |
| Initial target tokens | 3,000,000,000 |
| Continuation target tokens | 10,000,000,000 cumulative |
| Tokens per step | 524,288 |
| Initial target steps | About 5,723 |
| Continuation target step | About 19,074 |
| Precision | bf16 |
| Optimizer | AdamW, beta1=0.9, beta2=0.95, wd=0.1 |
| LR | Warm up to `3e-4`, then constant |
| Shuffle scope | New token-id permutation per 8192-token sequence |
| Output directory | `runs_shuffle_dynamics/shuffle_pretrain_3b_v2` |

## Results: 3B Phase

The 3B phase completed normally. Final rows from `loss.csv` show the shuffle
objective is still improving well below the initial 1B plateau:

```text
step=5719 tokens=2,998,403,072 loss=6.490262
step=5720 tokens=2,998,927,360 loss=6.476709
step=5721 tokens=2,999,451,648 loss=6.502136
step=5722 tokens=2,999,975,936 loss=6.335664
step=5723 tokens=3,000,500,224 loss=6.514327
```

The continuation result is pending.

## Local Ignored Artifacts

```text
runs_shuffle_dynamics/shuffle_pretrain_3b/loss.csv
runs_shuffle_dynamics/shuffle_pretrain_3b_v2/loss.csv
runs_shuffle_dynamics/shuffle_pretrain_3b_v2/loss.png
logs/slurm/shufdyn-3b-15692.out
logs/slurm/shufdyn-3b-15692.err
logs/slurm/shufdyn-3b-15693.out
logs/slurm/shufdyn-3b-15693.err
logs/slurm/shufdyn-cont7b-15774.out
logs/slurm/shufdyn-cont7b-15774.err
/home/scc/pb24511935/skypile_data_continue_7b/data/*.jsonl
```

## Caveats

- Resume reloads model and optimizer state but rebuilds the data iterator from
  the beginning of the configured `data_dir`. The 7B continuation intentionally
  points to a new non-overlapping data directory, so it does not keep reading the
  original 3B data directory.
- This experiment is not designed to compare generation quality. It only studies
  shuffle-objective training dynamics.
