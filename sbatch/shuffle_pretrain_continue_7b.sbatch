#!/bin/bash
#SBATCH --job-name=llm-shufdyn-cont7b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/shufdyn-cont7b-%j.out
#SBATCH --error=logs/slurm/shufdyn-cont7b-%j.err

set -euo pipefail

cd /home/scc/pb24511935/llm_shuffle
mkdir -p logs/slurm runs_shuffle_dynamics/shuffle_pretrain_3b_v2
test -f runs_shuffle_dynamics/shuffle_pretrain_3b_v2/checkpoints/latest.pt
test -d /home/scc/pb24511935/skypile_data_continue_7b/data
export PYTHONPATH="$PWD/src"

source venv/bin/activate

python scripts/plot_loss.py --run runs_shuffle_dynamics/shuffle_pretrain_3b_v2 --save runs_shuffle_dynamics/shuffle_pretrain_3b_v2/loss.png --interval 60 &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

torchrun --standalone --nproc_per_node=2 \
  -m llm_shuffle.train \
  --config configs/experiments_shuffle_dynamics/shuffle_pretrain_continue_7b.yaml

cleanup
python scripts/plot_loss.py --run runs_shuffle_dynamics/shuffle_pretrain_3b_v2 --no-watch --save runs_shuffle_dynamics/shuffle_pretrain_3b_v2/loss.png || true
