#!/bin/bash
#SBATCH --job-name=llm-tokenizer
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/tokenizer-%j.out
#SBATCH --error=logs/slurm/tokenizer-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/slurm tokenizer
export PYTHONPATH="$PWD/src"

python -m pip install -r requirements.txt
python scripts/train_tokenizer.py \
  --dataset Skywork/SkyPile-150B \
  --split train \
  --text-field text \
  --vocab-size 4096 \
  --max-files 1 \
  --max-docs 0 \
  --max-text-bytes 100000000 \
  --num-threads 8 \
  --output tokenizer/skypile_4k_tokenizer.json
