"""
analyze_training.py
-------------------
Analyze MambaVision training results:
1. Draw loss and accuracy curves
2. Determine if model is overfitting, underfitting, or well-fitted
3. Calculate inference time and training speed
"""

import json
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

import config
from model import build_model, load_checkpoint
from dataset import build_dataloaders


def load_history():
    """Load training history from JSON file."""
    history_path = os.path.join(config.RESULTS_DIR, "history.json")
    with open(history_path, "r") as f:
        return json.load(f)


def analyze_fit(history):
    """
    Analyze whether the model is overfitting, underfitting, or well-fitted.
    
    Returns:
        dict: Analysis results including fit type and metrics
    """
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    
    n_epochs = len(train_loss)
    
    # Get final epoch metrics
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    
    # Get best validation accuracy
    best_val_acc = max(val_acc)
    best_epoch = val_acc.index(best_val_acc) + 1
    
    # Calculate loss gap (indicator of overfitting)
    final_loss_gap = final_val_loss - final_train_loss
    loss_gap_ratio = final_val_loss / max(final_train_loss, 1e-6)
    
    # Calculate accuracy gap
    final_acc_gap = final_train_acc - final_val_acc
    
    # Determine fit type based on multiple criteria
    fit_analysis = {
        "fit_type": "unknown",
        "reasons": [],
        "metrics": {}
    }
    
    # Check for underfitting (both train and val accuracy are low)
    if final_train_acc < 0.7 and final_val_acc < 0.7:
        fit_analysis["fit_type"] = "underfitting"
        fit_analysis["reasons"].append(
            f"Both training ({final_train_acc:.2%}) and validation ({final_val_acc:.2%}) accuracy are low (<70%)"
        )
    
    # Check for overfitting (large gap between train and val)
    elif final_acc_gap > 0.15 or loss_gap_ratio > 1.5:
        fit_analysis["fit_type"] = "overfitting"
        fit_analysis["reasons"].append(
            f"Large gap between training ({final_train_acc:.2%}) and validation ({final_val_acc:.2%}) accuracy "
            f"(gap = {final_acc_gap:.2%})"
        )
        if loss_gap_ratio > 1.5:
            fit_analysis["reasons"].append(
                f"Validation loss ({final_val_loss:.4f}) is {loss_gap_ratio:.2f}x higher than training loss ({final_train_loss:.4f})"
            )
        
        # Check if overfitting is getting worse
        if n_epochs >= 5:
            recent_val_loss_trend = val_loss[-1] - val_loss[-5]
            if recent_val_loss_trend > 0.1:
                fit_analysis["reasons"].append(
                    f"Validation loss is increasing over recent epochs (trend: +{recent_val_loss_trend:.4f})"
                )
    
    # Good fit
    else:
        fit_analysis["fit_type"] = "good_fit"
        fit_analysis["reasons"].append(
            f"Training and validation metrics are well-aligned "
            f"(train_acc={final_train_acc:.2%}, val_acc={final_val_acc:.2%}, gap={final_acc_gap:.2%})"
        )
        fit_analysis["reasons"].append(
            f"Loss gap is acceptable (train={final_train_loss:.4f}, val={final_val_loss:.4f})"
        )
    
    # Additional observations
    if n_epochs >= 3:
        # Check if training is still improving
        if best_epoch < n_epochs:
            fit_analysis["reasons"].append(
                f"⚠️ Validation accuracy peaked at epoch {best_epoch} ({best_val_acc:.2%}) "
                f"and has slightly decreased since then"
            )
        else:
            fit_analysis["reasons"].append(
                f"✓ Model is still improving (best val_acc at final epoch)"
            )
    
    fit_analysis["metrics"] = {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "loss_gap": final_loss_gap,
        "loss_gap_ratio": loss_gap_ratio,
        "accuracy_gap": final_acc_gap,
        "total_epochs": n_epochs
    }
    
    return fit_analysis


def plot_curves(history, save_path=None):
    """Create detailed loss and accuracy curves."""
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    
    epochs = range(1, len(train_loss) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].axhline(y=min(val_loss), color='r', linestyle='--', alpha=0.3, 
                       label=f'Min Val Loss: {min(val_loss):.4f}')
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_acc, 'g-', label='Val Accuracy', linewidth=2)
    axes[0, 1].axhline(y=max(val_acc), color='g', linestyle='--', alpha=0.3,
                       label=f'Max Val Acc: {max(val_acc):.2%}')
    axes[0, 1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss gap (overfitting indicator)
    loss_gap = [v - t for t, v in zip(train_loss, val_loss)]
    axes[1, 0].plot(epochs, loss_gap, 'orange', linewidth=2, label='Val Loss - Train Loss')
    axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 0].axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Warning threshold')
    axes[1, 0].set_title('Loss Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Gap')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Accuracy gap
    acc_gap = [t - v for t, v in zip(train_acc, val_acc)]
    axes[1, 1].plot(epochs, acc_gap, 'purple', linewidth=2, label='Train Acc - Val Acc')
    axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 1].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Warning threshold')
    axes[1, 1].set_title('Accuracy Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Detailed training curves saved to {save_path}")
    
    plt.show()


def measure_inference_time(model, dataloaders, device, num_batches=10):
    """Measure inference time per batch and per sample."""
    model.eval()
    model = model.to(device)
    
    loader = dataloaders.get('test', dataloaders.get('val'))
    
    times = []
    num_samples = 0
    
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            
            images = images.to(device, non_blocking=True)
            
            # Warmup
            if i == 0:
                _ = model(images)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.time()
            _ = model(images)
            torch.cuda.synchronize() if device == 'cuda' else None
            end = time.time()
            
            times.append(end - start)
            num_samples += images.size(0)
    
    avg_batch_time = np.mean(times)
    std_batch_time = np.std(times)
    avg_per_sample = avg_batch_time / loader.batch_size
    
    return {
        "avg_batch_time": avg_batch_time,
        "std_batch_time": std_batch_time,
        "avg_per_sample_time": avg_per_sample,
        "samples_per_second": 1 / avg_per_sample if avg_per_sample > 0 else 0,
        "batch_size": loader.batch_size,
        "total_samples_processed": num_samples
    }


def calculate_training_speed(training_log_path):
    """Parse training log to calculate training speed."""
    if not os.path.exists(training_log_path):
        return None
    
    with open(training_log_path, 'r') as f:
        lines = f.readlines()
    
    # Find lines with epoch timing information
    epoch_times = []
    for line in lines:
        if 'elapsed=' in line and 'Epoch [' in line:
            try:
                # Extract elapsed time
                elapsed_part = line.split('elapsed=')[1].split('s')[0]
                epoch_times.append(float(elapsed_part))
            except (IndexError, ValueError):
                continue
    
    if not epoch_times:
        return None
    
    avg_epoch_time = np.mean(epoch_times)
    std_epoch_time = np.std(epoch_times)
    
    return {
        "avg_epoch_time_seconds": avg_epoch_time,
        "std_epoch_time_seconds": std_epoch_time,
        "avg_epoch_time_formatted": str(timedelta(seconds=int(avg_epoch_time))),
        "samples_per_second_estimate": None  # Would need dataset size
    }


def print_analysis_report(fit_analysis, inference_metrics, training_speed, test_metrics):
    """Print a comprehensive analysis report."""
    print("\n" + "=" * 70)
    print("  MambaVision Training Analysis Report")
    print("=" * 70)
    
    # Model Fit Analysis
    print("\n📊 MODEL FIT ANALYSIS")
    print("-" * 70)
    print(f"Fit Type: {fit_analysis['fit_type'].upper().replace('_', ' ')}")
    print(f"\nKey Metrics:")
    metrics = fit_analysis['metrics']
    print(f"  • Final Training Loss:     {metrics['final_train_loss']:.4f}")
    print(f"  • Final Validation Loss:   {metrics['final_val_loss']:.4f}")
    print(f"  • Final Training Accuracy: {metrics['final_train_acc']:.2%}")
    print(f"  • Final Validation Accuracy: {metrics['final_val_acc']:.2%}")
    print(f"  • Best Validation Accuracy:  {metrics['best_val_acc']:.2%} (Epoch {metrics['best_epoch']})")
    print(f"  • Loss Gap (Val - Train):  {metrics['loss_gap']:.4f} ({metrics['loss_gap_ratio']:.2f}x ratio)")
    print(f"  • Accuracy Gap:            {metrics['accuracy_gap']:.2%}")
    print(f"  • Total Epochs Trained:    {metrics['total_epochs']}")
    
    print(f"\nAnalysis:")
    for reason in fit_analysis['reasons']:
        print(f"  {reason}")
    
    # Test Performance
    if test_metrics:
        print(f"\n🎯 TEST SET PERFORMANCE")
        print("-" * 70)
        print(f"  • Top-1 Accuracy: {test_metrics.get('top1', 0):.2%}")
        print(f"  • Top-3 Accuracy: {test_metrics.get('top3', 0):.2%}")
    
    # Inference Speed
    if inference_metrics:
        print(f"\n⚡ INFERENCE SPEED")
        print("-" * 70)
        print(f"  • Batch Size: {inference_metrics['batch_size']}")
        print(f"  • Avg Batch Time: {inference_metrics['avg_batch_time']*1000:.2f} ms "
              f"(± {inference_metrics['std_batch_time']*1000:.2f} ms)")
        print(f"  • Avg Time per Sample: {inference_metrics['avg_per_sample_time']*1000:.2f} ms")
        print(f"  • Throughput: {inference_metrics['samples_per_second']:.1f} samples/sec")
        print(f"  • Total Samples Processed: {inference_metrics['total_samples_processed']}")
    
    # Training Speed
    if training_speed:
        print(f"\n🏃 TRAINING SPEED")
        print("-" * 70)
        print(f"  • Avg Epoch Time: {training_speed['avg_epoch_time_formatted']} "
              f"({training_speed['avg_epoch_time_seconds']:.1f}s)")
        print(f"  • Std Epoch Time: ± {training_speed['std_epoch_time_seconds']:.1f}s")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 70)
    if fit_analysis['fit_type'] == 'underfitting':
        print("  1. Train for more epochs")
        print("  2. Increase model capacity (use larger MambaVision variant)")
        print("  3. Reduce regularization (lower dropout, weight decay)")
        print("  4. Check if learning rate is too low")
    elif fit_analysis['fit_type'] == 'overfitting':
        print("  1. Apply early stopping (stop at epoch with best val_acc)")
        print("  2. Increase regularization (dropout, weight decay)")
        print("  3. Add more data augmentation")
        print("  4. Consider using a smaller model variant")
        print(f"  5. Best checkpoint is at epoch {metrics['best_epoch']} - use that for inference")
    else:  # good_fit
        print("  ✓ Model is well-fitted! Ready for deployment.")
        print(f"  ✓ Use checkpoint from epoch {metrics['best_epoch']} for best results")
        if metrics['final_val_acc'] > 0.95:
            print("  ✓ Excellent accuracy achieved!")
    
    print("\n" + "=" * 70)


def main():
    # Load history
    print("Loading training history...")
    history = load_history()
    
    # Analyze fit
    print("Analyzing model fit...")
    fit_analysis = analyze_fit(history)
    
    # Plot curves
    print("Generating training curves...")
    plot_save_path = os.path.join(config.RESULTS_DIR, "analysis_curves.png")
    plot_curves(history, save_path=plot_save_path)
    
    # Load test metrics if available
    test_metrics = None
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics_test.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            test_metrics = json.load(f)
    
    # Measure inference time
    print("\nMeasuring inference time...")
    try:
        model = build_model().to(config.DEVICE)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best.pth")
        if os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, model, device=config.DEVICE)
        
        dataloaders = build_dataloaders(config.DATA_DIR)
        inference_metrics = measure_inference_time(model, dataloaders, config.DEVICE, num_batches=20)
    except Exception as e:
        print(f"⚠ Could not measure inference time: {e}")
        inference_metrics = None
    
    # Calculate training speed from log
    training_speed = calculate_training_speed(config.LOG_FILE)
    
    # Print comprehensive report
    print_analysis_report(fit_analysis, inference_metrics, training_speed, test_metrics)
    
    return fit_analysis, inference_metrics, training_speed


if __name__ == "__main__":
    main()
