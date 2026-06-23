#!/bin/bash
#SBATCH --job-name=mamba_full
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mamba_%j.log
#SBATCH --error=logs/mamba_%j.log

# ── Cluster environment ───────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba_env
# ─────────────────────────────────────────────────────────────────────────────

set -e
mkdir -p logs

cd /projects/ya4v/llavagraph1.6/vit_mamba

echo "=============================="
echo "  Job   : $SLURM_JOB_ID"
echo "  Node  : $SLURMD_NODENAME"
echo "  GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Start : $(date)"
echo "=============================="

python train_mamba.py \
    --model_id   nvidia/MambaVision-T-1K \
    --epochs     50                       \
    --batch_size 32                       \
    --lr         3e-4                     \
    --strategy   full

echo "Done: $(date)"
