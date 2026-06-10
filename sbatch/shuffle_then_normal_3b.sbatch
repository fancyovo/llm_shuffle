#!/bin/bash
#SBATCH --job-name=llm-shufnorm-3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/shuffle-normal-3b-%j.out
#SBATCH --error=logs/slurm/shuffle-normal-3b-%j.err

set -euo pipefail

cd /home/scc/pb24511935/llm_shuffle
mkdir -p logs/slurm runs/shuffle_then_normal_3b
export PYTHONPATH="$PWD/src"

source venv/bin/activate

# Loss monitor background process (re-plots every 60s)
python scripts/plot_loss.py --run runs/shuffle_then_normal_3b --save runs/shuffle_then_normal_3b/loss.png --interval 60 &
MONITOR_PID=$!

torchrun --standalone --nproc_per_node=2 \
  -m llm_shuffle.train \
  --config configs/experiments/shuffle_then_normal_3b.yaml

kill $MONITOR_PID 2>/dev/null || true
python scripts/plot_loss.py --run runs/shuffle_then_normal_3b --no-watch --save runs/shuffle_then_normal_3b/loss.png
