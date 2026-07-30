#!/bin/bash
#SBATCH --job-name=falcon_mamba
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_falcon.out
#SBATCH --error=logs/%j_falcon.out

module load cuda/12.4

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba_env

export HF_HUB_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/ya4v/llavagraph1.6/mamba_zamba
mkdir -p logs

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "GPU:   $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "-----------------------------------------------"

python infer_falcon.py

echo "-----------------------------------------------"
echo "Done: $(date)"
