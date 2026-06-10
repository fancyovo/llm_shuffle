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
  --model deepseek-v4-flash
```

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
