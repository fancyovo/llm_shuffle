#!/bin/bash
#SBATCH --job-name=llm-random-3b-const
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/random-3b-const-%j.out
#SBATCH --error=logs/slurm/random-3b-const-%j.err

set -euo pipefail

cd /home/scc/pb24511935/llm_shuffle
mkdir -p logs/slurm runs_constant/random_3b
export PYTHONPATH="$PWD/src"

source venv/bin/activate

python scripts/plot_loss.py --run runs_constant/random_3b --save runs_constant/random_3b/loss.png --interval 60 &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

torchrun --standalone --nproc_per_node=2 \
  -m llm_shuffle.train \
  --config configs/experiments_constant/random_3b.yaml

cleanup
python scripts/plot_loss.py --run runs_constant/random_3b --no-watch --save runs_constant/random_3b/loss.png
