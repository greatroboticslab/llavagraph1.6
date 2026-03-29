#!/usr/bin/env python3
"""
Measure inference speed for ViT V2 model.
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

# --- Configuration ---
base_dir = "/data/ilminur/llavagraph1.6-master"
model_path = os.path.join(base_dir, "training_VIT/V2/vit_output_V3_aug")
data_dir = os.path.join(base_dir, "mambavision/data/test")

def main():
    print("=" * 70)
    print("  ViT V2 Inference Speed Measurement")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_vit_v2_mamba_data.py")
        sys.exit(1)
    
    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ViTForImageClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    processor = ViTImageProcessor.from_pretrained(model_path)
    
    # Transform (must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    
    # Load test data
    print("\nLoading test data...")
    test_images = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                test_images.append(os.path.join(class_path, fname))
    
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("Error: No test images found!")
        sys.exit(1)
    
    # Warmup
    print("\nWarming up...")
    dummy_img = Image.open(test_images[0]).convert("RGB")
    dummy_input = transform(dummy_img).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Measure inference
    batch_size = 32
    print(f"\nMeasuring inference speed (batch_size={batch_size})...")
    
    batch_times = []
    sample_times = []
    total_samples = 0
    
    num_batches = min(20, (len(test_images) + batch_size - 1) // batch_size)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(test_images))
        batch_paths = test_images[start_idx:end_idx]
        
        # Load batch
        batch_imgs = []
        for img_path in batch_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            batch_imgs.append(img_tensor)
        
        batch_tensor = torch.stack(batch_imgs).to(device)
        bs = batch_tensor.size(0)
        total_samples += bs
        
        # Measure
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(batch_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        batch_time = (end - start) * 1000  # ms
        per_sample_time = batch_time / bs
        
        batch_times.append(batch_time)
        sample_times.append(per_sample_time)
        
        if i < 5 or i % 5 == 0:
            print(f"  Batch {i+1}/{num_batches}: {batch_time:.2f}ms ({per_sample_time:.2f}ms/sample)")
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    sample_times = np.array(sample_times)
    
    results = {
        'avg_batch_time_ms': float(np.mean(batch_times)),
        'std_batch_time_ms': float(np.std(batch_times)),
        'min_batch_time_ms': float(np.min(batch_times)),
        'max_batch_time_ms': float(np.max(batch_times)),
        'avg_per_sample_ms': float(np.mean(sample_times)),
        'samples_per_second': float(1000 / np.mean(sample_times)) if np.mean(sample_times) > 0 else 0,
        'total_samples': total_samples,
        'batch_size': batch_size,
        'device': str(device)
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("  ViT V2 Inference Timing Results")
    print("=" * 70)
    
    print(f"\n⚡ INFERENCE SPEED:")
    print(f"  • Device: {results['device']}")
    print(f"  • Batch Size: {results['batch_size']}")
    print(f"  • Total Samples Processed: {results['total_samples']}")
    print(f"\n📊 Batch Timing:")
    print(f"  • Avg Batch Time: {results['avg_batch_time_ms']:.2f} ms")
    print(f"  • Std Batch Time: ±{results['std_batch_time_ms']:.2f} ms")
    print(f"  • Min Batch Time: {results['min_batch_time_ms']:.2f} ms")
    print(f"  • Max Batch Time: {results['max_batch_time_ms']:.2f} ms")
    print(f"\n📊 Per-Sample Timing:")
    print(f"  • Avg Time per Sample: {results['avg_per_sample_ms']:.2f} ms")
    print(f"  • Throughput: {results['samples_per_second']:.1f} samples/sec")
    
    print("\n" + "=" * 70)
    
    # Save results
    import json
    results_dir = os.path.join(base_dir, "training_VIT/V2")
    results_path = os.path.join(results_dir, "vit_inference_timing.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results

if __name__ == "__main__":
    main()
