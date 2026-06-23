#!/bin/bash
#SBATCH --job-name=eval_both
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba_env

set -e
mkdir -p logs

cd /projects/ya4v/llavagraph1.6/vit_mamba

echo "Start: $(date)"

python evaluate_both.py --split test

echo "Done: $(date)"
