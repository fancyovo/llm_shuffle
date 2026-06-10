# Generation Blind Evaluation

This workflow tests whether `shuffle_then_normal_3b` is judged better than
`random_3b` on generated text, with model identity hidden from the judge.

## Files

```text
eval/prompts_v1.jsonl                    Fixed prompt set, 60 prompts.
sbatch/generate_blind_eval.sbatch         GPU generation job.
scripts/generate_blind_eval_set.py        Writes paired generations.
scripts/deepseek_blind_judge.py           Calls DeepSeek for blind A/B judging.
scripts/analyze_blind_eval.py             Summarizes blind wins and repetition metrics.
eval/results/generations_v1.jsonl         Generated paired samples.
eval/results/deepseek_judgements_v1.jsonl DeepSeek judgements.
eval/results/summary_v1.md                Final summary.
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

The generation script is resumable. If `eval/results/generations_v1.jsonl`
already contains a `pair_id`, that pair is skipped.

## Step 2: Run DeepSeek Blind Judging

Run this on the login node after generation is complete. The API key must come
from the environment and must never be committed, written into an sbatch file,
or pasted into a tracked document.

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


For interactive use without exporting the key, pass `--api-key-stdin`; the script reads the key with a hidden prompt. Use `--concurrency` to control parallel API calls. The script is resumable, so increasing concurrency is safe as long as duplicate `pair_id` rows are not manually introduced.

Use `--force-choice` for a relative-quality test that forces the judge to choose `A` or `B` even when both outputs are poor. Write forced-choice results to a separate output file so they are not mixed with the default `Tie`/`BothBad` run.

The judge sees only:

```text
Prompt
Answer A
Answer B
```

The script randomizes which model is A or B per pair. The output file stores the
hidden mapping only after the judgement so the analysis script can count wins.

The judging script is resumable. Existing `pair_id` rows in the judgement file
are skipped.

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

Treat `shuffle_then_normal_3b` as materially better only if both signals point
in the same direction:

```text
DeepSeek blind decisive win rate > 55%, with 95% CI lower bound > 50%.
Automatic degeneration rate is lower for shuffle_then_normal_3b.
```

If the blind result is close to 50% or the automatic metrics disagree, the
current qualitative examples should be treated as suggestive but not confirmed.

## Recorded Run: 2026-06-10

The first DeepSeek blind run used `deepseek-v4-flash`, 180 generated pairs, and `--concurrency 64`. Raw outputs are stored under `eval/results/` and are intentionally ignored by Git.

Summary:

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

Interpretation: automatic repetition metrics favor `shuffle_then_normal_3b`, but the DeepSeek blind A/B result is not statistically significant because most pairs were judged `BothBad` and only 28 comparisons were decisive. Treat the generation-quality advantage as suggestive, not confirmed.

## Recorded Force-Choice Run: 2026-06-10

A follow-up DeepSeek run used the same 180 generated pairs and added `--force-choice`, forcing the judge to choose A or B even when both outputs were poor. Raw outputs are stored under `eval/results/` and are intentionally ignored by Git.

Summary:

```text
random_3b wins: 78
shuffle_then_normal_3b wins: 102
Tie: 0
BothBad: 0
Decisive comparisons: 180
shuffle decisive win rate: 0.5667
Wilson 95% CI: [0.4936, 0.6369]
Two-sided sign-test p-value: 0.0861873

By category wins:
continue: shuffle 19, random 11
explain: shuffle 19, random 11
fact: shuffle 15, random 15
opinion: shuffle 16, random 14
reason: shuffle 14, random 16
story: shuffle 19, random 11
```

Interpretation: forced-choice judging again favors `shuffle_then_normal_3b`, especially on continue/explain/story prompts, but the result still does not meet a conventional p < 0.05 threshold on this 180-pair sample.
