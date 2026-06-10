# Server Handoff

This document is written for an execution agent that can follow commands but should not redesign the experiment.

## Important Constraints

- Do not run GPU training directly on the login node.
- Submit GPU work through `sbatch`.
- Use exactly the tokenizer saved at `tokenizer/skypile_4k_tokenizer.json` after it has been trained.
- Do not change model size, optimizer settings, context length, or shuffle behavior unless explicitly instructed.
- Special token IDs 0, 1, 2, and 3 must not be shuffled.

## Directory Map

```text
configs/                 YAML configuration files
configs/experiments/     The three experiment configs
src/llm_shuffle/         Model, data, and training code
scripts/                 Tokenizer, smoke test, parameter count helpers
sbatch/                  Slurm job templates
docs/                    Experiment and handoff docs
eval/                    Fixed generation-evaluation prompts; results are ignored
tokenizer/               Saved tokenizer
runs/                    Training outputs and checkpoints
logs/slurm/              Slurm stdout and stderr
```

Constant-LR retraining outputs use `runs_constant/`; do not mix them with the
original cosine-decay `runs/` outputs.

## Environment Setup

On the server, from the project root:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

If the cluster uses modules, load the CUDA/Python module first according to local policy.

## Local Sanity Checks

Run these on a compute node or a login node because they are CPU-only and small:

```bash
export PYTHONPATH="$PWD/src"
python scripts/smoke_test.py
python scripts/count_params.py --config configs/model_50m.yaml
```

Expected parameter count for the real model is about 49.8M.

## Step 1: Train Tokenizer

Submit:

```bash
sbatch sbatch/train_tokenizer.sbatch
```

The tokenizer job reads from the first SkyPile JSONL shard but stops after 100,000,000 UTF-8 text bytes. SkyPile-150B has 437 `data/*.jsonl` shards, about 665GB total, and most shards are around 1-2GB. Do not train on a full shard on a memory-limited machine: Hugging Face BPE training can expand a 1GB shard into tens of GB of RAM. The approved tokenizer setting is 100MB text and 8 tokenizer threads.

After completion, verify:

```bash
test -f tokenizer/skypile_4k_tokenizer.json
python - <<'PY'
from tokenizers import Tokenizer
t = Tokenizer.from_file("tokenizer/skypile_4k_tokenizer.json")
for tok in ["<pad>", "<bos>", "<eos>", "<unk>"]:
    print(tok, t.token_to_id(tok))
print("vocab", t.get_vocab_size())
PY
```

The required IDs are:

```text
<pad> 0
<bos> 1
<eos> 2
<unk> 3
vocab 4096
```

If the tokenizer job fails because the dataset cannot be downloaded, fix Hugging Face network/authentication first. Do not replace the dataset with another corpus without approval.

## Step 2: Run Random Baseline

Submit:

```bash
sbatch sbatch/random_3b.sbatch
```

Monitor:

```bash
tail -f logs/slurm/random-3b-<JOBID>.out
tail -f runs/random_3b/loss.csv
```

The target is 3B tokens, about 5723 optimizer steps.

## Step 3: Run Shuffle Pretraining

Submit:

```bash
sbatch sbatch/shuffle_pretrain_1b.sbatch
```

Monitor:

```bash
tail -f logs/slurm/shuffle-1b-<JOBID>.out
tail -f runs/shuffle_pretrain_1b/loss.csv
```

The target is 1B tokens, about 1908 optimizer steps.

## Step 4: Run Normal Training From Shuffle Checkpoint

Only start this after `runs/shuffle_pretrain_1b/checkpoints/latest.pt` exists.

Submit:

```bash
sbatch sbatch/shuffle_then_normal_3b.sbatch
```

Monitor:

```bash
tail -f logs/slurm/shuffle-normal-3b-<JOBID>.out
tail -f runs/shuffle_then_normal_3b/loss.csv
```

The target is another 3B normal tokens, about 5723 optimizer steps.

## Resume Behavior

Each training config writes a latest checkpoint:

```text
runs/<run_name>/checkpoints/latest.pt
```

If the same sbatch job is submitted again, the trainer resumes from that checkpoint by default. It keeps the optimizer state and token counter.

For `shuffle_then_normal_3b`, the first launch loads the shuffle checkpoint as initialization. Later launches resume from `runs/shuffle_then_normal_3b/checkpoints/latest.pt`.

## Loss Comparison

Use the CSV files:

```text
runs/random_3b/loss.csv
runs/shuffle_then_normal_3b/loss.csv
```

Compare `loss` against `tokens_seen`. Do not compare by wall clock time because throughput may differ across jobs.


The completed 2026-06-09 to 2026-06-10 run is summarized in `docs/EXPERIMENT_RECORD.md`. It records:

```text
random_3b:              3.000B tokens, final loss 2.8660
shuffle_pretrain_1b:    interrupted at 0.823B/1B tokens, final shuffle loss 8.0032
shuffle_then_normal_3b: 3.000B normal tokens, final loss 2.8280
```

The reported raw final-loss delta is 0.0381 in favor of `shuffle_then_normal_3b`.

## Blind Generation Evaluation

Use `docs/GENERATION_BLIND_EVAL.md` to test whether the generation-quality difference is statistically meaningful. The workflow is:

```bash
sbatch sbatch/generate_blind_eval.sbatch
```

Then, on the login node:

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
python scripts/deepseek_blind_judge.py --model deepseek-v4-flash
python scripts/analyze_blind_eval.py
```

Do not commit `eval/results/`, model checkpoints, logs, or API keys.

## Common Failure Cases

### Dataset download failure

Check network access and Hugging Face authentication. The dataset source is fixed as:

```text
Skywork/SkyPile-150B
```

### CUDA out of memory

First try keeping the same global token batch by reducing `micro_batch_size` only if it is greater than 1. In this project it is already 1. The next acceptable fallback is increasing `grad_accum_steps` while reducing sequence-level memory pressure only if a code change is approved. Do not reduce `seq_len` without approval because the experiment requires context length 8192.

### Job hits 1 day limit

Resubmit the same sbatch file. The trainer resumes from `latest.pt`.

### Missing tokenizer

Run the tokenizer sbatch first. Training jobs require:

```text
tokenizer/skypile_4k_tokenizer.json
```

## Constant-LR Retraining Request

The requested rerun keeps warmup but removes LR decay for all few-billion-token
training phases. Use:

```bash
sbatch sbatch/random_3b_constant_lr.sbatch
sbatch sbatch/shuffle_1b_then_normal_3b_constant_lr.sbatch
```

The experiment group configs live under `configs/experiments_constant/` and
write to `runs_constant/`. The experimental sbatch runs shuffle pretraining for
1B tokens first, then immediately launches normal 3B-token training from
`runs_constant/shuffle_pretrain_1b/checkpoints/latest.pt`.
