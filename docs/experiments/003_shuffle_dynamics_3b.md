# Experiment 003: Token-ID Shuffle Dynamics, 3B Plus 7B Continuation

## Summary

| Item | Value |
|------|-------|
| Start date | 2026-06-11 |
| Status | Completed through 10B shuffle tokens; prefill analysis completed |
| Purpose | Study token-id shuffle pretraining dynamics beyond the 1B-token region |
| Scope | Shuffle mode only; no downstream normal-training phase yet |
| LR schedule | 100M-token warmup, then constant `3e-4` |
| Output root | `runs_shuffle_dynamics/` |

The previous constant-LR run showed that the 1B shuffle-pretrain loss was still
in a transition region and had not clearly converged. This experiment trains the
token-id shuffle objective from scratch for 3B reported optimizer tokens, then
continues the same model and optimizer state for another 7B tokens on new data.
After training completed, three checkpoints were compared with a fixed-context
prefill analysis to inspect what changed across the apparent transition.

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
status: completed
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

### 10B continuation completed

```text
job_id: 15774
final_step: 19074
reported_tokens: 10,000,269,312
final_loss: 6.2040925
final_lr: 3.0e-4
final_throughput: 186426 tok/s
```

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

## Results: 10B Continuation

The 7B continuation completed normally and kept the constant learning rate. The
final rows from `loss.csv` were:

```text
step=19063 tokens=9,994,502,144 loss=6.161962
step=19064 tokens=9,995,026,432 loss=6.171488
step=19065 tokens=9,995,550,720 loss=6.159208
step=19066 tokens=9,996,075,008 loss=6.264673
step=19067 tokens=9,996,599,296 loss=6.250616
step=19068 tokens=9,997,123,584 loss=6.140254
step=19069 tokens=9,997,647,872 loss=6.099154
step=19070 tokens=9,998,172,160 loss=6.244864
step=19071 tokens=9,998,696,448 loss=5.946795
step=19072 tokens=9,999,220,736 loss=6.136380
step=19073 tokens=9,999,745,024 loss=6.153891
step=19074 tokens=10,000,269,312 loss=6.204093
```

The later 7B did not show a second sharp transition. The loss continued to
improve in a more normal, gradual way.

## Prefill Analysis

The analysis compares next-token probability distributions on long real
SkyPile text for three checkpoints:

```text
pre_phase:  runs_shuffle_dynamics/shuffle_pretrain_3b_v2/checkpoints/step_0001500.pt
            step 1500, 786,432,000 tokens
post_phase: runs_shuffle_dynamics/shuffle_pretrain_3b_v2/checkpoints/step_0002500.pt
            step 2500, 1,310,720,000 tokens
final_10b:  runs_shuffle_dynamics/shuffle_pretrain_3b_v2/checkpoints/step_0019074.pt
            step 19074, 10,000,269,312 tokens
```

Because this is a token-id shuffle model, the analysis evaluates the actual
training objective rather than raw original token IDs. It applies a fixed
permutation seed, `20260612`, to the selected original text, feeds the permuted
tokens to the model, and scores the permuted next token. Top-k predictions are
mapped back to original token IDs only for interpretation.

The GPU prefill work was run through Slurm, not on the login node:

```text
early sample job: sbatch/analyze_shuffle_prefill.sbatch
late sample job:  sbatch/analyze_shuffle_prefill_late.sbatch
```

Generated result files are intentionally ignored by Git:

```text
eval/results/shuffle_prefill_analysis/summary.json
eval/results/shuffle_prefill_analysis/summary.md
eval/results/shuffle_prefill_analysis/per_position.csv
eval/results/shuffle_prefill_analysis_late/summary.json
eval/results/shuffle_prefill_analysis_late/summary.md
eval/results/shuffle_prefill_analysis_late/per_position.csv
```

### Early Sample

Source text:

```text
data_dir: /home/scc/pb24511935/skypile_data_continue_7b
file: 2020-40_zh_middle_0002.jsonl
start_file_index: 0
mode: concatenated_rows
rows_seen: 13
seq_len: 8192
```

Overall metrics:

| Checkpoint | Mean NLL | Median NLL | Acc@1 | Acc@5 | Median Rank | Mean Entropy |
|------------|---------:|-----------:|------:|------:|------------:|-------------:|
| pre_phase | 8.0527 | 8.3006 | 0.0441 | 0.0786 | 1435.0 | 8.0746 |
| post_phase | 7.1412 | 8.2901 | 0.1555 | 0.2269 | 1159.5 | 7.1328 |
| final_10b | 6.6982 | 8.0275 | 0.1744 | 0.2708 | 647.5 | 6.8132 |

Pairwise changes:

| Change | Mean NLL Improvement | Median Improvement | Improved Positions |
|--------|---------------------:|-------------------:|-------------------:|
| pre_phase to post_phase | 0.9115 | 0.0728 | 0.5371 |
| post_phase to final_10b | 0.4430 | 0.2197 | 0.6085 |

Selected group changes:

| Group | pre Mean NLL | post Mean NLL | final Mean NLL | Main Pattern |
|-------|-------------:|--------------:|---------------:|--------------|
| `freq_gt10` | 7.3818 | 5.9782 | 5.0196 | Frequent tokens improve strongly; median rank 513 to 98 to 5 |
| `seen_le64` | 7.3694 | 5.1468 | 4.1787 | Local repeats improve sharply; median rank 384.5 to 3 to 2 |
| `seen_le256` | 8.2337 | 6.2110 | 5.3852 | Mid-range repeats become much more predictable |
| `cat_space` | 7.8196 | 3.4836 | 2.3960 | Spaces/newlines become highly predictable; median rank 1706 to 1 |
| `cat_punct` | 5.8855 | 4.7231 | 3.8318 | Punctuation improves steadily |
| `never_seen` | 8.5116 | 8.6384 | 8.7504 | First occurrences slightly worsen |

### Late Sample

Source text:

```text
data_dir: /home/scc/pb24511935/skypile_data_continue_7b
file: 2020-45_zh_head_0010.jsonl
start_file_index: 30
mode: concatenated_rows
rows_seen: 7
seq_len: 8192
```

Overall metrics:

| Checkpoint | Mean NLL | Median NLL | Acc@1 | Acc@5 | Median Rank | Mean Entropy |
|------------|---------:|-----------:|------:|------:|------------:|-------------:|
| pre_phase | 8.0203 | 8.2086 | 0.0375 | 0.0724 | 1219.0 | 8.1132 |
| post_phase | 6.5016 | 8.0987 | 0.2390 | 0.3040 | 687.0 | 6.4379 |
| final_10b | 5.8072 | 7.4041 | 0.2673 | 0.3750 | 132.5 | 5.9529 |

Pairwise changes:

| Change | Mean NLL Improvement | Median Improvement | Improved Positions |
|--------|---------------------:|-------------------:|-------------------:|
| pre_phase to post_phase | 1.5188 | 0.1500 | 0.5601 |
| post_phase to final_10b | 0.6944 | 0.2583 | 0.6460 |

Selected group changes:

| Group | pre Mean NLL | post Mean NLL | final Mean NLL | Main Pattern |
|-------|-------------:|--------------:|---------------:|--------------|
| `freq_gt10` | 7.6352 | 5.3511 | 4.1595 | Frequent tokens improve strongly; median rank 581.5 to 16 to 3 |
| `seen_le16` | 6.9599 | 4.4923 | 3.2535 | Short-range repeats improve sharply; median rank 100 to 2 to 1 |
| `seen_le64` | 7.5754 | 4.8311 | 3.7445 | Local repeated tokens become reliable top predictions |
| `seen_le256` | 8.1327 | 5.1679 | 4.1211 | Mid-range repeated terms improve strongly |
| `cat_digit` | 7.9655 | 6.7475 | 5.6272 | Digits improve mostly after longer training; median rank 765 to 664.5 to 18 |
| `never_seen` | 8.4826 | 8.7024 | 8.8410 | First occurrences slightly worsen |

### Interpretation

The apparent transition is not a uniform improvement over every position. The
largest gains are concentrated in:

- local repetition and copy-like behavior;
- high-frequency tokens inside the sampled context;
- formatting tokens such as spaces and newlines;
- punctuation, dates, digits, and domain-repeated terms such as `公告`, `收益`,
  `亿元`, and `日`.

Rare first occurrences do not improve in either sample. Their NLL and median
rank slightly worsen as the model becomes more selective. This means the model
is not simply becoming broadly better at every semantic prediction. Around the
transition it appears to discover useful structure under the shuffled-token
objective: repeated symbols, separators, local lexical regularities, and
document-format patterns become much easier to predict. The later 7B continues
this process and makes rankings for repeated/frequent tokens much sharper, but
without another obvious phase transition.

## Local Ignored Artifacts

```text
runs_shuffle_dynamics/shuffle_pretrain_3b/loss.csv
runs_shuffle_dynamics/shuffle_pretrain_3b_v2/loss.csv
runs_shuffle_dynamics/shuffle_pretrain_3b_v2/loss.png
eval/results/shuffle_prefill_analysis/summary.json
eval/results/shuffle_prefill_analysis/per_position.csv
eval/results/shuffle_prefill_analysis_late/summary.json
eval/results/shuffle_prefill_analysis_late/per_position.csv
logs/slurm/shufdyn-3b-15692.out
logs/slurm/shufdyn-3b-15692.err
logs/slurm/shufdyn-3b-15693.out
logs/slurm/shufdyn-3b-15693.err
logs/slurm/shufdyn-cont7b-15774.out
logs/slurm/shufdyn-cont7b-15774.err
logs/slurm/shuf-prefill-15835.out
logs/slurm/shuf-prefill-late-15836.out
/home/scc/pb24511935/skypile_data_continue_7b/data/*.jsonl
```

## Caveats

- Resume reloads model and optimizer state but rebuilds the data iterator from
  the beginning of the configured `data_dir`. The 7B continuation intentionally
  points to a new non-overlapping data directory, so it does not keep reading the
  original 3B data directory.
- This experiment is not designed to compare generation quality. It only studies
  shuffle-objective training dynamics.
