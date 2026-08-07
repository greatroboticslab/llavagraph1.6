#!/bin/bash
#SBATCH --job-name=mamba2_sanity
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:RTX5000Ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_sanity.out
#SBATCH --error=logs/%j_sanity.out

module load cuda/12.4

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba_env

# Fast SSM kernels — without these, transformers falls back to a slow
# pure-PyTorch scan and speed numbers won't reflect Mamba's real potential.
pip install --quiet mamba-ssm causal-conv1d 2>&1 | tail -5

cd /projects/ya4v/llavagraph1.6/mamba2_fast
mkdir -p logs

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "GPU:   $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "-----------------------------------------------"

python sanity_check.py

echo "-----------------------------------------------"
echo "Done: $(date)"
