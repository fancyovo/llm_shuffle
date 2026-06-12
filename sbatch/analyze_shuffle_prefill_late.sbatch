#!/bin/bash
#SBATCH --job-name=llm-shuf-prefill-late
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/shuf-prefill-late-%j.out
#SBATCH --error=logs/slurm/shuf-prefill-late-%j.err

set -euo pipefail

cd /home/scc/pb24511935/llm_shuffle
mkdir -p logs/slurm eval/results/shuffle_prefill_analysis_late
export PYTHONPATH="$PWD/src"

source venv/bin/activate

python scripts/analyze_shuffle_prefill.py \
  --data-dir /home/scc/pb24511935/skypile_data_continue_7b \
  --start-file-index 30 \
  --out-dir eval/results/shuffle_prefill_analysis_late \
  --model pre_phase=runs_shuffle_dynamics/shuffle_pretrain_3b_v2/checkpoints/step_0001500.pt \
  --model post_phase=runs_shuffle_dynamics/shuffle_pretrain_3b_v2/checkpoints/step_0002500.pt \
  --model final_10b=runs_shuffle_dynamics/shuffle_pretrain_3b_v2/checkpoints/step_0019074.pt
