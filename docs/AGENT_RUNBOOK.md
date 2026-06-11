# Agent Runbook

This is the first document to read on the server. Follow the steps in order. Do not redesign the experiment.

## Goal

Run three jobs using the fixed project code:

1. Train or verify the fixed 4096-token tokenizer.
2. Train the random baseline for 3B tokens.
3. Train the shuffle-pretrained path: 1B shuffle tokens, then 3B normal tokens.

The comparison is between `runs/random_3b/loss.csv` and `runs/shuffle_then_normal_3b/loss.csv`.

Experiment history is recorded in `docs/EXPERIMENT_RECORD.md`, with detailed pages under `docs/experiments/`. Read it before launching follow-up experiments so repeated work is not mistaken for a fresh baseline.

## Do Not Change

- Do not change `seq_len: 8192`.
- Do not change `vocab_size: 4096`.
- Do not tie embeddings and classifier.
- Do not shuffle special token IDs 0, 1, 2, 3.
- Do not run GPU training on the login node.
- Do not replace `Skywork/SkyPile-150B` with another dataset.
- Do not train tokenizer on a full 1-2GB shard on a memory-limited node.

## Expected Hardware

- 2 GPUs, intended target: RTX 5090 x 2.
- 24 CPU cores.
- 1 day Slurm time limit per submitted job.
- Training uses DDP through `torchrun --nproc_per_node=2`.

## Fresh Clone Setup

From the project root:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
mkdir -p logs/slurm runs tokenizer
```

If the cluster requires modules, load them first. Example only:

```bash
module load cuda
module load python
```

Use the cluster's actual module names.

## Before Submitting GPU Jobs

Run CPU-only checks:

```bash
export PYTHONPATH="$PWD/src"
python scripts/smoke_test.py
python scripts/count_params.py --config configs/model_50m.yaml
python -m compileall src scripts
```

Expected parameter count:

```text
exact_params=49,820,160
estimated_params=49,820,160
```

## Tokenizer

This repository already contains `tokenizer/skypile_4k_tokenizer.json` if it was committed and pushed with the project. Verify it first:

```bash
python - <<'PY'
from tokenizers import Tokenizer
t = Tokenizer.from_file("tokenizer/skypile_4k_tokenizer.json")
print("vocab", t.get_vocab_size())
for tok in ["<pad>", "<bos>", "<eos>", "<unk>"]:
    print(tok, t.token_to_id(tok))
PY
```

Required output:

```text
vocab 4096
<pad> 0
<bos> 1
<eos> 2
<unk> 3
```

If the tokenizer file is missing or invalid, submit:

```bash
sbatch sbatch/train_tokenizer.sbatch
```

The tokenizer job reads from the first SkyPile shard and stops after 100,000,000 UTF-8 text bytes using 8 tokenizer threads.

## Submit Training Jobs

Submit the random baseline:

```bash
sbatch sbatch/random_3b.sbatch
```

Submit shuffle pretraining:

```bash
sbatch sbatch/shuffle_pretrain_1b.sbatch
```

Wait until this file exists before starting the next job:

```text
runs/shuffle_pretrain_1b/checkpoints/latest.pt
```

Then submit normal training from the shuffle checkpoint:

```bash
sbatch sbatch/shuffle_then_normal_3b.sbatch
```

The random baseline and shuffle pretraining jobs may run independently if the queue allows it. The shuffle-then-normal job depends on the shuffle-pretrain checkpoint.

## Monitor Jobs

Use Slurm:

```bash
squeue -u "$USER"
```

Use stdout logs:

```bash
tail -f logs/slurm/random-3b-<JOBID>.out
tail -f logs/slurm/shuffle-1b-<JOBID>.out
tail -f logs/slurm/shuffle-normal-3b-<JOBID>.out
```

Use CSV logs:

```bash
tail -f runs/random_3b/loss.csv
tail -f runs/shuffle_pretrain_1b/loss.csv
tail -f runs/shuffle_then_normal_3b/loss.csv
```

CSV columns are:

```text
step,tokens_seen,loss,lr,tokens_per_second,elapsed_seconds
```

Compare by `tokens_seen`, not by step time.

## Expected Step Counts

Each optimizer step uses approximately 524,288 tokens:

```text
8192 seq_len * 1 micro_batch * 32 grad_accum * 2 GPUs
```

Expected target steps:

```text
random_3b:              about 5723 steps
shuffle_pretrain_1b:    about 1908 steps
shuffle_then_normal_3b: about 5723 steps
```

## Resume

If a job stops because of the 1 day limit or cluster interruption, resubmit the same sbatch file.

The trainer resumes from:

```text
runs/<run_name>/checkpoints/latest.pt
```

Do not delete `runs/<run_name>/checkpoints/latest.pt` unless explicitly instructed.

## Output Checklist

After all jobs finish, these files should exist:

```text
tokenizer/skypile_4k_tokenizer.json
runs/random_3b/loss.csv
runs/random_3b/checkpoints/latest.pt
runs/shuffle_pretrain_1b/loss.csv
runs/shuffle_pretrain_1b/checkpoints/latest.pt
runs/shuffle_then_normal_3b/loss.csv
runs/shuffle_then_normal_3b/checkpoints/latest.pt
```

Record final numbers and any interruption details in the relevant detail page under `docs/experiments/`, then update `docs/EXPERIMENT_RECORD.md`. Do not commit files under `runs/`, `runs_constant/`, `runs_shuffle_dynamics/`, `logs/`, `eval/results/`, or checkpoint weights.


## Blind Generation Evaluation

After the training runs exist, the next validation step is a blind A/B generation test:

```bash
sbatch sbatch/generate_blind_eval.sbatch
```

This writes paired model outputs to:

```text
eval/results/generations_v1.jsonl
```

After generation completes, run the judge on the login node with the API key only in the environment:

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
python scripts/deepseek_blind_judge.py --model deepseek-v4-flash
python scripts/analyze_blind_eval.py
```

See `docs/GENERATION_BLIND_EVAL.md` for the full procedure. Never write an API key into a tracked file or sbatch script.

## Constant-LR Retraining

For short few-billion-token runs, use the constant-LR experiment line instead
of the original cosine-decay configs. These configs keep the 100M-token warmup
and then hold LR at `3e-4`:

```text
configs/experiments_constant/random_3b.yaml
configs/experiments_constant/shuffle_pretrain_1b.yaml
configs/experiments_constant/shuffle_then_normal_3b.yaml
```

The output root is `runs_constant/`, so the earlier cosine-decay runs under
`runs/` are not overwritten.

Submit the two high-level jobs:

```bash
sbatch sbatch/random_3b_constant_lr.sbatch
sbatch sbatch/shuffle_1b_then_normal_3b_constant_lr.sbatch
```

The second job runs the shuffle 1B phase and then the normal 3B phase in the
same Slurm allocation. If the job is interrupted, resubmit the same sbatch file;
each phase resumes from its own `runs_constant/<run>/checkpoints/latest.pt`.

## Shuffle Dynamics 3B Run

To study the token-id shuffle objective itself beyond the 1B-token transition
region, use the shuffle-only dynamics config:

```bash
sbatch sbatch/shuffle_pretrain_3b_dynamics.sbatch
```

This writes to `runs_shuffle_dynamics/shuffle_pretrain_3b_v2/` and does not
launch the later normal-training phase. The local JSONL data pipeline now shards
the dataset by DDP rank so the two GPU ranks do not consume identical examples.
Do not reintroduce map-style `Dataset.shuffle(seed)` for local JSONL: it caused
random-access data reads and slowed throughput from about 200k tok/s to
60k-100k tok/s.

To continue the completed 3B shuffle-dynamics checkpoint for another 7B tokens,
use:

```bash
sbatch sbatch/shuffle_pretrain_continue_7b.sbatch
```

This is not a restart. It keeps `runtime.output_dir` at
`runs_shuffle_dynamics/shuffle_pretrain_3b_v2/`, so the trainer resumes
`checkpoints/latest.pt`, and changes `data.data_dir` to
`/home/scc/pb24511935/skypile_data_continue_7b`.

## Common Problems

### Hugging Face download fails

Fix network or authentication. Do not change the dataset.

### CUDA out of memory

Stop and report the failure. Do not reduce context length. The requested experiment requires `seq_len=8192`.

### Tokenizer training uses too much memory

Confirm `sbatch/train_tokenizer.sbatch` contains:

```bash
--max-text-bytes 100000000
--num-threads 8
```

Do not train on a full shard.

### `ModuleNotFoundError: llm_shuffle`

Set:

```bash
export PYTHONPATH="$PWD/src"
```

The sbatch files already do this.

### `shuffle_then_normal_3b` fails with missing checkpoint

Run or resume `sbatch/shuffle_pretrain_1b.sbatch` first. The required file is:

```text
runs/shuffle_pretrain_1b/checkpoints/latest.pt
```
