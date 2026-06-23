#!/bin/bash
#SBATCH --job-name=vit_full
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vit_%j.log
#SBATCH --error=logs/vit_%j.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba_env

set -e
mkdir -p logs

cd /projects/ya4v/llavagraph1.6/vit_mamba

echo "=============================="
echo "  Job   : $SLURM_JOB_ID"
echo "  Node  : $SLURMD_NODENAME"
echo "  GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Start : $(date)"
echo "=============================="

python train_vit.py \
    --epochs     50   \
    --batch_size 32   \
    --lr         1e-4 \
    --strategy   full \
    --num_workers 4

echo "Done: $(date)"
