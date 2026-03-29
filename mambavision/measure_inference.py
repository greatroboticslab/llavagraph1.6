"""
measure_inference.py
--------------------
Measure actual inference time for MambaVision.
"""

import os
import time
import torch
import numpy as np

import config
from model import build_model, load_checkpoint
from dataset import build_dataloaders


@torch.no_grad()
def measure_inference(model, dataloaders, device, num_batches=20):
    """Measure inference timing."""
    model.eval()
    model = model.to(device)
    
    # Use test loader if available, otherwise val
    loader = dataloaders.get('test', dataloaders.get('val'))
    
    print(f"\nMeasuring inference on {device}...")
    print(f"Batch size: {loader.batch_size}")
    print(f"Number of batches to measure: {num_batches}")
    
    batch_times = []
    sample_times = []
    total_samples = 0
    
    # Warmup
    print("Warming up...")
    for i, (images, _) in enumerate(loader):
        if i >= 2:
            break
        images = images.to(device, non_blocking=True)
        _ = model(images)
    
    print("Measuring...")
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)
        total_samples += batch_size
        
        # Measure
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(images)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        batch_time = (end - start) * 1000  # ms
        per_sample_time = batch_time / batch_size
        
        batch_times.append(batch_time)
        sample_times.append(per_sample_time)
        
        if i < 5 or i % 5 == 0:
            print(f"  Batch {i+1}/{num_batches}: {batch_time:.2f}ms ({per_sample_time:.2f}ms/sample)")
    
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
        'batch_size': loader.batch_size,
        'device': device
    }
    
    return results


def print_results(results):
    """Print inference timing results."""
    print("\n" + "=" * 70)
    print("  MambaVision Inference Timing Results")
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


def main():
    print("=" * 70)
    print("  MambaVision Inference Time Measurement")
    print("=" * 70)
    
    # Build model
    print("\nBuilding model...")
    model = build_model().to(config.DEVICE)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        load_checkpoint(checkpoint_path, model, device=config.DEVICE)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model")
    
    # Build dataloaders
    print("\nLoading data...")
    dataloaders = build_dataloaders(config.DATA_DIR)
    
    # Measure
    results = measure_inference(model, dataloaders, config.DEVICE, num_batches=20)
    
    # Print results
    print_results(results)
    
    return results


if __name__ == "__main__":
    main()
