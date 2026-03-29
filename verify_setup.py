import torch
from llava.model.builder import load_pretrained_model

print("="*50)
print("Environment Verification")
print("="*50)

# Check PyTorch
print(f"\n[OK] PyTorch version: {torch.__version__}")

# Check CUDA
cuda_available = torch.cuda.is_available()
print(f"[OK] CUDA available: {cuda_available}")

if cuda_available:
    print(f"[OK] CUDA version: {torch.version.cuda}")
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("[WARNING] CUDA not available!")

# Check LLaVA import
print(f"\n[OK] LLaVA package imported successfully")

print("\n" + "="*50)
print("Environment setup complete!")
print("="*50)
