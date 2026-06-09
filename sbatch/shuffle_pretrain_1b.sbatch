#!/bin/bash
#SBATCH --job-name=llm-shuffle-1b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/shuffle-1b-%j.out
#SBATCH --error=logs/slurm/shuffle-1b-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/slurm runs/shuffle_pretrain_1b
export PYTHONPATH="$PWD/src"

python -m pip install -r requirements.txt
torchrun --standalone --nproc_per_node=2 \
  -m llm_shuffle.train \
  --config configs/experiments/shuffle_pretrain_1b.yaml
