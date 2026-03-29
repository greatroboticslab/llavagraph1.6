"""
analyze_training_simple.py
--------------------------
Quick analysis of MambaVision training results without loading the model.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

import config


def load_history():
    """Load training history from JSON file."""
    history_path = os.path.join(config.RESULTS_DIR, "history.json")
    with open(history_path, "r") as f:
        return json.load(f)


def load_test_metrics():
    """Load test metrics if available."""
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics_test.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def parse_training_log():
    """Parse training log to extract timing information."""
    log_path = config.LOG_FILE
    if not os.path.exists(log_path):
        return None, []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    epoch_data = []
    for line in lines:
        if 'Epoch [' in line and 'elapsed=' in line:
            try:
                # Extract epoch number
                epoch_part = line.split('Epoch [')[1].split(']')[0]
                epoch_num = int(epoch_part.split('/')[0])
                
                # Extract elapsed time
                elapsed_part = line.split('elapsed=')[1].split('s')[0]
                elapsed_time = float(elapsed_part)
                
                # Extract metrics
                train_loss = float(line.split('train_loss=')[1].split()[0])
                train_acc = float(line.split('train_acc=')[1].split()[0])
                val_loss = float(line.split('val_loss=')[1].split()[0])
                val_acc = float(line.split('val_acc=')[1].split()[0])
                
                epoch_data.append({
                    'epoch': epoch_num,
                    'elapsed': elapsed_time,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })
            except (IndexError, ValueError) as e:
                continue
    
    return epoch_data


def analyze_fit(history):
    """Analyze whether the model is overfitting, underfitting, or well-fitted."""
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    
    n_epochs = len(train_loss)
    
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    
    best_val_acc = max(val_acc)
    best_epoch = val_acc.index(best_val_acc) + 1
    
    final_loss_gap = final_val_loss - final_train_loss
    loss_gap_ratio = final_val_loss / max(final_train_loss, 1e-6)
    final_acc_gap = final_train_acc - final_val_acc
    
    fit_analysis = {
        "fit_type": "unknown",
        "reasons": [],
        "metrics": {}
    }
    
    # Determine fit type
    if final_train_acc < 0.7 and final_val_acc < 0.7:
        fit_analysis["fit_type"] = "underfitting"
        fit_analysis["reasons"].append(
            f"Both training ({final_train_acc:.2%}) and validation ({final_val_acc:.2%}) accuracy are low (<70%)"
        )
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
    else:
        fit_analysis["fit_type"] = "good_fit"
        fit_analysis["reasons"].append(
            f"Training and validation metrics are well-aligned "
            f"(train_acc={final_train_acc:.2%}, val_acc={final_val_acc:.2%}, gap={final_acc_gap:.2%})"
        )
    
    # Additional observations
    if n_epochs >= 3:
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
    
    # Plot 3: Loss gap
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


def print_report(fit_analysis, epoch_data, test_metrics):
    """Print comprehensive analysis report."""
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
    
    # Training Speed
    if epoch_data:
        times = [e['elapsed'] for e in epoch_data]
        avg_time = np.mean(times)
        print(f"\n🏃 TRAINING SPEED")
        print("-" * 70)
        print(f"  • Avg Epoch Time: {str(timedelta(seconds=int(avg_time)))} ({avg_time:.1f}s)")
        print(f"  • Min Epoch Time: {str(timedelta(seconds=int(min(times))))} ({min(times):.1f}s)")
        print(f"  • Max Epoch Time: {str(timedelta(seconds=int(max(times))))} ({max(times):.1f}s)")
        
        # Estimate samples per second
        batch_size = config.BATCH_SIZE
        # Approximate: each epoch processes train + val datasets
        # From the log, we can estimate based on progress
        print(f"  • Batch Size: {batch_size}")
    
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
    print("Loading training history...")
    history = load_history()
    
    print("Parsing training log...")
    epoch_data = parse_training_log()
    
    print("Analyzing model fit...")
    fit_analysis = analyze_fit(history)
    
    print("Generating training curves...")
    plot_save_path = os.path.join(config.RESULTS_DIR, "analysis_curves.png")
    plot_curves(history, save_path=plot_save_path)
    
    print("Loading test metrics...")
    test_metrics = load_test_metrics()
    
    print_report(fit_analysis, epoch_data, test_metrics)
    
    return fit_analysis


if __name__ == "__main__":
    main()
