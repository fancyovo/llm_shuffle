#!/bin/bash
#SBATCH --job-name=llm-shuf10b-norm3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/shuf10b-norm3b-%j.out
#SBATCH --error=logs/slurm/shuf10b-norm3b-%j.err

set -euo pipefail

cd /home/scc/pb24511935/llm_shuffle
mkdir -p logs/slurm runs_shuffle_dynamics/shuffle10b_then_normal_3b
test -f runs_shuffle_dynamics/shuffle_pretrain_3b_v2/checkpoints/step_0019074.pt
test -d /home/scc/pb24511935/skypile_data_after_shuffle10b_normal3b/data
if [ -f runs_shuffle_dynamics/shuffle10b_then_normal_3b/checkpoints/latest.pt ]; then
  if [ "${ALLOW_NORMAL_RESUME:-0}" != "1" ]; then
    echo "Refusing first launch: output checkpoint already exists." >&2
    echo "To resume this normal-training run, submit with:" >&2
    echo "  sbatch --export=ALL,ALLOW_NORMAL_RESUME=1 sbatch/shuffle10b_then_normal_3b.sbatch" >&2
    exit 1
  fi
fi
export PYTHONPATH="$PWD/src"

source venv/bin/activate

python scripts/plot_loss.py --run runs_shuffle_dynamics/shuffle10b_then_normal_3b --save runs_shuffle_dynamics/shuffle10b_then_normal_3b/loss.png --interval 60 &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

torchrun --standalone --nproc_per_node=2 \
  -m llm_shuffle.train \
  --config configs/experiments_shuffle_dynamics/shuffle10b_then_normal_3b.yaml

cleanup
python scripts/plot_loss.py --run runs_shuffle_dynamics/shuffle10b_then_normal_3b --no-watch --save runs_shuffle_dynamics/shuffle10b_then_normal_3b/loss.png || true
