#!/bin/bash
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/generate-%j.out
#SBATCH --error=logs/slurm/generate-%j.err

set -euo pipefail
cd /home/scc/pb24511935/llm_shuffle
export PYTHONPATH="$PWD/src"
source venv/bin/activate

for model in shuffle_then_normal_3b random_3b; do
  echo "===== $model ====="
  python scripts/generate.py \
    --checkpoint runs/$model/checkpoints/latest.pt \
    --prompt "人工智能" \
    --max-new-tokens 300 \
    --temperature 0.9 \
    --top-k 40 \
    --top-p 0.92
  echo ""
done
