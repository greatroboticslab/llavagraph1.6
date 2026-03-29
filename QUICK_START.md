# Quick Start Guide for Local LLaVA Inference

## Environment Setup Complete!

Your local LLaVA environment is now ready:
- ✓ Conda environment: `llava`
- ✓ Python 3.10
- ✓ PyTorch 2.1.2 + CUDA 11.8
- ✓ GPU: RTX 3070 Ti (8GB VRAM)
- ✓ LLaVA package installed

## Running Inference

### Activate Environment

```bash
conda activate llava
```

### Test on Single Image

```bash
cd D:\Antigravity\llavagraph1.5\MSEC

python evaluateLLaVA.py \
  --model-path checkpointsV6 \
  --model-base ../models_setup/llava-v1.5-7b \
  --image-folder data/data_filtered/test/sine \
  --output-file test_sine_output.json \
  --subset 1
```

### Run Full Evaluation

```bash
# Evaluate all sine waves
python evaluateLLaVA.py \
  --model-path checkpointsV6 \
  --model-base ../models_setup/llava-v1.5-7b \
  --image-folder data/data_filtered/test/sine \
  --output-file results/sine_test.json

# Evaluate all square waves  
python evaluateLLaVA.py \
  --model-path checkpointsV6 \
  --model-base ../models_setup/llava-v1.5-7b \
  --image-folder data/data_filtered/test/square \
  --output-file results/square_test.json

# Evaluate all noise
python evaluateLLaVA.py \
  --model-path checkpointsV6 \
  --model-base ../models_setup/llava-v1.5-7b \
  --image-folder data/data_filtered/test/noise \
  --output-file results/noise_test.json
```

### Use 4-bit Quantization (if memory issues)

```bash
python evaluateLLaVA.py \
  --model-path checkpointsV6 \
  --model-base ../models_setup/llava-v1.5-7b \
  --image-folder data/data_filtered/test/sine \
  --output-file test_output.json \
  --load-4bit
```

## Next Steps

1. **Download checkpoint** from Google Drive if not already done
2. **Test inference** on a single image
3. **Analyze results** to understand current model performance
4. **Implement improvements** from the improvement plan

## Troubleshooting

### CUDA Out of Memory
- Use `--load-4bit` flag
- Reduce batch size in evaluation script

### Model Not Found
- Ensure checkpoint is in `MSEC/checkpointsV6/`
- Verify base model is in `models_setup/llava-v1.5-7b/`

### Slow Inference
- Check GPU usage with `nvidia-smi`
- Ensure CUDA is being used (not CPU)
