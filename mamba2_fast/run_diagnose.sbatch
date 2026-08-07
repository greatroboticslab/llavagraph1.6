#!/bin/bash
#SBATCH --job-name=mamba2_diag
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:RTX5000Ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j_diag.out
#SBATCH --error=logs/%j_diag.out

module load cuda/12.4

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ya4v/.conda/envs/mamba_env

cd /projects/ya4v/llavagraph1.6/mamba2_fast
mkdir -p logs

echo "Job:   $SLURM_JOB_ID"
echo "Start: $(date)"
echo "-----------------------------------------------"

python diagnose_loss_gap.py

echo "-----------------------------------------------"
echo "Done: $(date)"
