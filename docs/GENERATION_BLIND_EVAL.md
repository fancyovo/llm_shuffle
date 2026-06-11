# Generation Blind Evaluation

This document describes the reusable blind A/B generation workflow. Experiment
specific results are recorded in `docs/EXPERIMENT_RECORD.md` and the linked
detail pages under `docs/experiments/`.

## Files

```text
eval/prompts_v1.jsonl                     Fixed prompt set, 60 prompts.
sbatch/generate_blind_eval.sbatch          Default GPU generation job.
sbatch/generate_constant_blind_eval.sbatch Constant-LR GPU generation job.
scripts/generate_blind_eval_set.py         Writes paired generations.
scripts/deepseek_blind_judge.py            Calls DeepSeek for blind A/B judging.
scripts/analyze_blind_eval.py              Summarizes wins and repetition metrics.
eval/results/*.jsonl                       Ignored generated samples and judgements.
eval/results/*.md                          Ignored analysis summaries.
```

## Step 1: Generate Paired Samples

Submit generation through Slurm. Do not run checkpoint generation on the login
node.

```bash
sbatch sbatch/generate_blind_eval.sbatch
```

The default job evaluates 60 prompts with 3 seeds each, producing 180 paired
comparisons. It uses identical prompts, seeds, and sampling parameters for both
models:

```text
temperature=0.8
top_k=40
top_p=0.92
max_new_tokens=200
```

The generation script is resumable. If the selected generations file already
contains a `pair_id`, that pair is skipped.

## Step 2: Run DeepSeek Blind Judging

Run this on the login node after generation is complete. This step calls the
DeepSeek API only; it does not run local model inference. The API key must come
from the environment or hidden stdin and must never be committed, written into an
sbatch file, or pasted into a tracked document.

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY

python scripts/deepseek_blind_judge.py \
  --generations eval/results/generations_v1.jsonl \
  --out eval/results/deepseek_judgements_v1.jsonl \
  --model deepseek-v4-flash \
  --concurrency 64
```

For interactive use without exporting the key, pass `--api-key-stdin`; the
script reads the key with a hidden prompt. Use `--concurrency` to control
parallel API calls. The script is resumable, so increasing concurrency is safe as
long as duplicate `pair_id` rows are not manually introduced.

Use `--force-choice` for a relative-quality test that forces the judge to choose
`A` or `B` even when both outputs are poor. Write forced-choice results to a
separate output file so they are not mixed with default `Tie`/`BothBad` runs.

The judge sees only:

```text
Prompt
Answer A
Answer B
```

The script randomizes which model is A or B per pair. The output file stores the
hidden mapping only after the judgement so the analysis script can count wins.
Existing `pair_id` rows in the judgement file are skipped.

## Step 3: Analyze

```bash
python scripts/analyze_blind_eval.py \
  --generations eval/results/generations_v1.jsonl \
  --judgements eval/results/deepseek_judgements_v1.jsonl \
  --out eval/results/summary_v1.md
```

The summary reports:

- blind A/B win counts,
- tie and both-bad counts,
- decisive win rate,
- Wilson 95% confidence interval,
- two-sided sign-test p-value,
- automatic repetition metrics such as degeneration rate, distinct-4, and
  repeated 4-gram rate.

## Interpretation

Treat a model as materially better only if the blind judgement and automatic
metrics point in the same direction:

```text
DeepSeek blind decisive win rate > 55%, with 95% CI lower bound > 50%.
Automatic degeneration rate is lower for the same model.
```

If the blind result is close to 50% or the automatic metrics disagree, the
current qualitative examples should be treated as suggestive but not confirmed.

## Recording Results

Do not add raw `eval/results/` files to Git. Instead:

1. Keep generated samples, judgement JSONL, and summary Markdown under
   `eval/results/`.
2. Copy the decisive counts, confidence interval, p-value, and key repetition
   metrics into the relevant `docs/experiments/<id>.md` file.
3. Add or update the headline row in `docs/EXPERIMENT_RECORD.md`.
