#!/bin/bash
#SBATCH --job-name=nemotron_infer
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:RTX5000Ada:1
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j_infer.out
#SBATCH --error=logs/%j_infer.err

module purge
module load cuda/12.4
source $HOME/envs/nemotron_env/bin/activate

export HF_HUB_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1

cd /projects/ya4v/llavagraph1.6/piezo_finetune
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

yes | python inference.py \
    --adapter_dir ./checkpoints/final_adapter \
    --data_dir    ./data \
    --n_samples   20

echo "Done: $(date)"
