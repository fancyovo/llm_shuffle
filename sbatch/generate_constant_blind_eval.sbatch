#!/bin/bash
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/generate-constant-blind-eval-%j.out
#SBATCH --error=logs/slurm/generate-constant-blind-eval-%j.err

set -euo pipefail

cd /home/scc/pb24511935/llm_shuffle
mkdir -p logs/slurm eval/results
export PYTHONPATH="$PWD/src"
source venv/bin/activate

python scripts/generate_blind_eval_set.py \
  --prompt-file eval/prompts_v1.jsonl \
  --out eval/results/generations_constant_v1.jsonl \
  --model random_3b_constant=runs_constant/random_3b/checkpoints/latest.pt \
  --model shuffle_then_normal_3b_constant=runs_constant/shuffle_then_normal_3b/checkpoints/latest.pt \
  --seeds 42 43 44 \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.92
