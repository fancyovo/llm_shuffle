#!/bin/bash
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/eval-%j.out
#SBATCH --error=logs/slurm/eval-%j.err

set -euo pipefail
cd /home/scc/pb24511935/llm_shuffle
export PYTHONPATH="$PWD/src"
source venv/bin/activate

python scripts/eval_prompts.py \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.92 \
  --seed 42 \
  --max-new-tokens 200 \
  --out runs/eval_results.txt
