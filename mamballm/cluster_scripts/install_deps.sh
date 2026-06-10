#!/bin/bash
# Run this ONCE on the cluster login node to install all dependencies.
# Usage: bash install_deps.sh

module purge
module load cuda/12.4
source $HOME/envs/nemotron_env/bin/activate

pip install --upgrade pip

# Core fine-tuning stack
pip install \
    "transformers>=4.47" \
    "peft>=0.10" \
    "trl>=0.8" \
    "accelerate>=0.27" \
    "bitsandbytes>=0.43" \
    "datasets>=2.18" \
    "scipy" \
    "sentencepiece"

echo "All dependencies installed."
python -c "import transformers, peft, trl, bitsandbytes; print('Imports OK')"
