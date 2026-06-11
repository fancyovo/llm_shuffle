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

## Recorded Force-Choice Run: Shuffle-Normal 2B vs Random 3B

A follow-up compared `runs/shuffle_then_normal_3b/checkpoints/step_0004000.pt` against `runs/random_3b/checkpoints/latest.pt`. The shuffle-normal checkpoint is not exactly 2B normal tokens; the exact 2B point is around step 3815, but checkpoints were saved every 500 steps. `step_0004000.pt` corresponds to 2,097,152,000 normal-training tokens after shuffle initialization.

Generation was submitted through Slurm with `sbatch/generate_shuffle_normal_2b_vs_random_3b.sbatch`. DeepSeek judging used `deepseek-v4-flash` with `--force-choice`. Raw outputs are stored under `eval/results/` and are intentionally ignored by Git.

Summary:

```text
random_3b wins: 75
shuffle_then_normal_2b_step4000 wins: 105
Tie: 0
BothBad: 0
Decisive comparisons: 180
shuffle-normal-2B decisive win rate: 0.5833
Wilson 95% CI: [0.5103, 0.6529]
Two-sided sign-test p-value: 0.0303695

By category wins:
continue: shuffle 21, random 9
explain: shuffle 20, random 10
fact: shuffle 13, random 17
opinion: shuffle 18, random 12
reason: shuffle 15, random 15
story: shuffle 18, random 12

Automatic degeneration rate:
random_3b: 0.6444
shuffle_then_normal_2b_step4000: 0.5111
```

Interpretation: this forced-choice comparison favors the shuffle-initialized model at the ~2.1B normal-token checkpoint over the random 3B checkpoint, and reaches p < 0.05 on this 180-pair prompt/seed set. The result should be described as `step_0004000` or ~2.1B normal tokens, not exact 2B.

## Recorded Constant-LR Completed Run: 2026-06-11

The constant-LR rerun completed both paths under `runs_constant/`:

```text
random_3b_constant:              3.0005B reported tokens, final loss 2.8689
shuffle_pretrain_1b_constant:    0.9998B reported tokens, final shuffle loss 6.9909
shuffle_then_normal_3b_constant: 3.0005B reported normal tokens, final loss 2.8023
```

Loss comparison for the two 3B normal phases:

```text
metric                  random_3b_constant  shuffle_then_normal_3b_constant  delta random-shuffle
raw final               2.8689              2.8023                          0.0666
EMA final alpha=0.02    2.9087              2.8536                          0.0551
EMA final alpha=0.01    2.9085              2.8558                          0.0527
last 50 avg             2.9041              2.8479                          0.0562
last 100 avg            2.9133              2.8596                          0.0537
last 200 avg            2.9078              2.8566                          0.0512
last 500 avg            2.9055              2.8556                          0.0499
```

The EMA alpha=0.02 gap stabilizes around 0.05 loss after roughly 2B reported normal tokens and finishes at 0.0551. The comparison plot is written to `runs_constant/comparison.png` and is intentionally ignored by Git.

Qualitative prompt samples still show both 50M models are weak and repetitive. In a small manual spot check, `shuffle_then_normal_3b_constant` is sometimes more directly on-topic, but not consistently less repetitive across all prompts.

DeepSeek force-choice blind judging used the 60-prompt, 3-seed evaluation set generated by `sbatch/generate_constant_blind_eval.sbatch`:

```text
random_3b_constant wins: 84
shuffle_then_normal_3b_constant wins: 96
Tie: 0
BothBad: 0
Decisive comparisons: 180
shuffle decisive win rate: 0.5333
Wilson 95% CI: [0.4605, 0.6048]
Two-sided sign-test p-value: 0.412351

By category wins:
continue: random 15, shuffle 15
explain: random 16, shuffle 14
fact: random 13, shuffle 17
opinion: random 15, shuffle 15
reason: random 13, shuffle 17
story: random 12, shuffle 18
```

Automatic repetition metrics on the generated samples did not favor the shuffle model:

```text
metric             random_3b_constant  shuffle_then_normal_3b_constant
degenerate_rate    0.6500              0.6611
distinct4          0.6743              0.6426
repeat4_rate       0.3257              0.3574
max_repeat4        12.73               12.28
```

Interpretation: constant-LR training gives a clear training-loss advantage to the shuffle-initialized path, but this completed-run generation evaluation does not show a statistically significant DeepSeek preference and does not improve the simple repetition metrics. Also remember the current local JSONL data pipeline does not shard data by DDP rank, so reported token counts overstate distinct text-token coverage.
