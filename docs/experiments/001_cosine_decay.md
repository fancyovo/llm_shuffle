# Experiment 001: Initial Cosine-Decay Comparison

## Summary

| Item | Value |
|------|-------|
| Dates | 2026-06-09 to 2026-06-10 |
| Status | Completed |
| Purpose | Compare random 3B normal training against token-id shuffle initialization followed by 3B normal training |
| LR schedule | 100M-token warmup, then cosine decay to `3e-5` |
| Output root | `runs/` |

This is the initial full comparison. The shuffle pretraining phase was intended
to run for 1B tokens but was interrupted at 822.6M tokens. The downstream normal
training still completed from that partial shuffle checkpoint.

## Tracked Inputs

| Role | Path |
|------|------|
| Random baseline config | `configs/experiments/random_3b.yaml` |
| Shuffle pretrain config | `configs/experiments/shuffle_pretrain_1b.yaml` |
| Shuffle then normal config | `configs/experiments/shuffle_then_normal_3b.yaml` |
| Base training config | `configs/train_base.yaml` |
| Random sbatch | `sbatch/random_3b.sbatch` |
| Shuffle pretrain sbatch | `sbatch/shuffle_pretrain_1b.sbatch` |
| Shuffle then normal sbatch | `sbatch/shuffle_then_normal_3b.sbatch` |
| Blind generation sbatch | `sbatch/generate_blind_eval.sbatch` |
| 2.1B checkpoint generation sbatch | `sbatch/generate_shuffle_normal_2b_vs_random_3b.sbatch` |

## Training Settings

| Parameter | Value |
|-----------|-------|
| Precision | bf16 |
| Optimizer | AdamW, beta1=0.9, beta2=0.95, wd=0.1 |
| Peak LR | `3e-4` |
| Min LR | `3e-5` |
| Grad clip | 1.0 |
| Tokens per step | 524,288 |
| Shuffle scope | New token-id permutation per 8192-token sequence |

## Training Results

| Run | Steps | Reported Tokens | Final Loss | Status |
|-----|-------|-----------------|------------|--------|
| `random_3b` | 5,723 | 3.0005B | 2.8660 | Completed |
| `shuffle_pretrain_1b` | 1,569/1,908 | 0.8226B/1B | 8.0032 | Interrupted |
| `shuffle_then_normal_3b` | 5,723 | 3.0005B | 2.8280 | Completed |

Loss comparison between the 3B normal phases:

| Metric | `random_3b` | `shuffle_then_normal_3b` | Delta random-shuffle |
|--------|-------------|--------------------------|----------------------|
| Raw final loss | 2.8660 | 2.8280 | 0.0381 |
| EMA final, alpha=0.02 | 2.9205 | 2.8818 | 0.0386 |
| Last 100-step average | 2.9267 | 2.8880 | 0.0387 |

## Generation Evaluation

Qualitative prompt checks suggested the shuffle-initialized model was less
repetitive and more often on topic, but both 50M models were weak.

DeepSeek blind judging used `deepseek-v4-flash` on the 60-prompt, 3-seed set.
Raw generated samples and judgement files are under `eval/results/` and are not
tracked by Git.

### Default Blind Judge

```text
random_3b wins: 11
shuffle_then_normal_3b wins: 17
Tie: 0
BothBad: 152
Decisive comparisons: 28
shuffle decisive win rate: 0.6071
Wilson 95% CI: [0.4241, 0.7643]
Two-sided sign-test p-value: 0.344928

Automatic degeneration rate:
random_3b: 0.6444
shuffle_then_normal_3b: 0.5444
```

### Force-Choice Judge

```text
random_3b wins: 78
shuffle_then_normal_3b wins: 102
Tie: 0
BothBad: 0
Decisive comparisons: 180
shuffle decisive win rate: 0.5667
Wilson 95% CI: [0.4936, 0.6369]
Two-sided sign-test p-value: 0.0861873
```

### Shuffle-Normal Step 4000 vs Random 3B

This compared `runs/shuffle_then_normal_3b/checkpoints/step_0004000.pt`
against `runs/random_3b/checkpoints/latest.pt`. Step 4000 corresponds to about
2.097B normal-training tokens after shuffle initialization, not exactly 2B.

```text
random_3b wins: 75
shuffle_then_normal_2b_step4000 wins: 105
Tie: 0
BothBad: 0
Decisive comparisons: 180
shuffle-normal-2B decisive win rate: 0.5833
Wilson 95% CI: [0.5103, 0.6529]
Two-sided sign-test p-value: 0.0303695

Automatic degeneration rate:
random_3b: 0.6444
shuffle_then_normal_2b_step4000: 0.5111
```

## Interpretation

The loss advantage was modest but consistent. Generation judging was suggestive,
especially in the force-choice step-4000 comparison, but the default judge
classified most pairs as `BothBad`. This experiment is also confounded by LR
decay and by the later-discovered DDP data duplication issue.

## Local Ignored Artifacts

```text
runs/random_3b/loss.csv
runs/shuffle_pretrain_1b/loss.csv
runs/shuffle_then_normal_3b/loss.csv
runs/comparison.png
runs/eval_results.txt
eval/results/generations_v1.jsonl
eval/results/deepseek_judgements_v1.jsonl
eval/results/deepseek_judgements_force_choice_v1.jsonl
eval/results/generations_shuffle_normal_2b_vs_random_3b.jsonl
eval/results/deepseek_judgements_force_choice_shuffle_normal_2b_vs_random_3b.jsonl
```

## Caveats

- The shuffle pretraining phase stopped at 822.6M tokens, not 1B.
- Local JSONL data was not sharded by DDP rank in this experiment. Rank 0 and
  rank 1 consumed duplicate text streams, so reported tokens overstate distinct
  text-token coverage by about 2x.
- Token-id shuffle was active during shuffle pretraining; the data issue was
  duplicated text streams, not disabled token-id permutation.
