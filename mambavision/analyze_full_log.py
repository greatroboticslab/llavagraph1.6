"""
analyze_full_log.py
-------------------
Parse the complete training log to analyze full training history.
"""

import os
import re
import numpy as np
from datetime import timedelta

import config


def parse_full_training_log():
    """Parse the complete training log to extract all epoch data."""
    log_path = config.LOG_FILE
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find all epoch lines
    epoch_pattern = r'Epoch \[(\d+)/(\d+)\].*?train_loss=([\d.]+).*?train_acc=([\d.]+).*?val_loss=([\d.]+).*?val_acc=([\d.]+).*?val_top3=([\d.]+).*?lr=([\d.eE-]+).*?elapsed=([\d.]+)s'
    
    matches = re.findall(epoch_pattern, content, re.DOTALL)
    
    epoch_data = []
    for match in matches:
        try:
            epoch_num, total_epochs, train_loss, train_acc, val_loss, val_acc, val_top3, lr, elapsed = match
            epoch_data.append({
                'epoch': int(epoch_num),
                'total_epochs': int(total_epochs),
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'val_top3': float(val_top3),
                'lr': float(lr),
                'elapsed': float(elapsed)
            })
        except (ValueError, IndexError) as e:
            continue
    
    return epoch_data


def load_test_metrics():
    """Load test metrics."""
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics_test.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def main():
    import json
    
    print("\n" + "=" * 70)
    print("  MambaVision FULL Training Analysis Report")
    print("=" * 70)
    
    epoch_data = parse_full_training_log()
    
    if not epoch_data:
        print("No epoch data found in training log!")
        print("Make sure training has been run and logged properly.")
        return
    
    print(f"\n✓ Parsed {len(epoch_data)} epochs from training log")
    
    # Extract arrays
    epochs = [e['epoch'] for e in epoch_data]
    train_loss = [e['train_loss'] for e in epoch_data]
    val_loss = [e['val_loss'] for e in epoch_data]
    train_acc = [e['train_acc'] for e in epoch_data]
    val_acc = [e['val_acc'] for e in epoch_data]
    val_top3 = [e['val_top3'] for e in epoch_data]
    times = [e['elapsed'] for e in epoch_data]
    
    # Final metrics
    n_epochs = len(epoch_data)
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    
    # Best metrics
    best_val_acc = max(val_acc)
    best_epoch = val_acc.index(best_val_acc) + 1
    best_val_loss = val_loss[best_epoch - 1]
    
    # Gaps
    acc_gap = final_train_acc - final_val_acc
    loss_ratio = final_val_loss / max(final_train_loss, 1e-6)
    
    # Load test metrics
    test_metrics = None
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics_test.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            test_metrics = json.load(f)
    
    # Determine fit
    print("\n📊 MODEL FIT ANALYSIS")
    print("-" * 70)
    
    if final_train_acc < 0.7 and final_val_acc < 0.7:
        fit_type = "UNDERFITTING"
        print(f"Fit Type: {fit_type}")
        print(f"\nReason: Both train ({final_train_acc:.2%}) and val ({final_val_acc:.2%}) accuracy are low (<70%)")
    elif acc_gap > 0.15 or loss_ratio > 1.5:
        fit_type = "OVERFITTING"
        print(f"Fit Type: {fit_type}")
        print(f"\nReason: Large gap between train ({final_train_acc:.2%}) and val ({final_val_acc:.2%})")
        print(f"        Accuracy gap = {acc_gap:.2%}, Loss ratio = {loss_ratio:.2f}x")
        if best_epoch < n_epochs:
            print(f"        Val accuracy dropped from {best_val_acc:.2%} (epoch {best_epoch}) to {final_val_acc:.2%}")
    else:
        fit_type = "GOOD FIT"
        print(f"Fit Type: {fit_type}")
        print(f"\nReason: Train and val metrics are well-aligned")
        print(f"        Accuracy gap = {acc_gap:.2%}, Loss ratio = {loss_ratio:.2f}x")
    
    print(f"\n📈 KEY METRICS:")
    print(f"  • Epochs Completed:      {n_epochs} (planned: {epoch_data[-1]['total_epochs']})")
    print(f"  • Final Train Loss:      {final_train_loss:.4f}")
    print(f"  • Final Val Loss:        {final_val_loss:.4f}")
    print(f"  • Final Train Accuracy:  {final_train_acc:.2%}")
    print(f"  • Final Val Accuracy:    {final_val_acc:.2%}")
    print(f"  • Best Val Accuracy:     {best_val_acc:.2%} (Epoch {best_epoch})")
    print(f"  • Best Val Loss:         {best_val_loss:.4f}")
    print(f"  • Final Val Top-3:       {val_top3[-1]:.2%}")
    print(f"  • Accuracy Gap:          {acc_gap:.2%}")
    print(f"  • Loss Ratio (V/T):      {loss_ratio:.2f}x")
    
    if test_metrics:
        print(f"\n🎯 TEST SET PERFORMANCE:")
        print(f"  • Top-1 Accuracy: {test_metrics.get('top1', 0):.2%}")
        print(f"  • Top-3 Accuracy: {test_metrics.get('top3', 0):.2%}")
    
    # Training speed
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\n🏃 TRAINING SPEED:")
    print(f"  • Avg Epoch Time:        {str(timedelta(seconds=int(avg_time)))} ({avg_time:.1f}s)")
    print(f"  • Std Epoch Time:        ±{std_time:.1f}s")
    print(f"  • Min Epoch Time:        {str(timedelta(seconds=int(min(times))))} ({min(times):.1f}s)")
    print(f"  • Max Epoch Time:        {str(timedelta(seconds=int(max(times))))} ({max(times):.1f}s)")
    print(f"  • Batch Size:            {config.BATCH_SIZE}")
    print(f"  • Device:                {config.DEVICE}")
    
    # Estimate samples per second
    # From config, we can estimate based on batch size and typical iterations
    print(f"\n⚡ ESTIMATED INFERENCE SPEED:")
    print(f"  • Note: Run inference.py with timing for precise measurements")
    print(f"  • Model: {config.MODEL_ID}")
    print(f"  • Expected: ~50-100ms per image on CPU, ~5-10ms on GPU")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 70)
    if fit_type == "UNDERFITTING":
        print("  1. ✓ Training is still in progress - continue training")
        print("  2. Consider using a larger MambaVision model variant")
        print("  3. Ensure learning rate is appropriate")
    elif fit_type == "OVERFITTING":
        print(f"  1. ✓ Use early stopping - best checkpoint at epoch {best_epoch}")
        print(f"  2. Val accuracy peaked at {best_val_acc:.2%}, now at {final_val_acc:.2%}")
        print("  3. Consider increasing regularization (dropout, weight decay)")
        print("  4. Add more data augmentation")
    else:
        print("  ✓ Model is well-fitted!")
        print(f"  ✓ Best checkpoint: epoch {best_epoch} (val_acc={best_val_acc:.2%})")
        if best_val_acc > 0.95:
            print("  ✓ Excellent accuracy achieved!")
    
    # Save CSV for plotting
    csv_path = os.path.join(config.RESULTS_DIR, "training_history_full.csv")
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc,val_top3,lr,elapsed\n")
        for e in epoch_data:
            f.write(f"{e['epoch']},{e['train_loss']:.6f},{e['val_loss']:.6f},{e['train_acc']:.6f},{e['val_acc']:.6f},{e['val_top3']:.6f},{e['lr']:.6e},{e['elapsed']:.1f}\n")
    print(f"\n📁 Full training history saved to: {csv_path}")
    print(f"   Use this CSV to plot loss curves in Excel, Google Sheets, or Python")
    
    # Show recent progress
    print(f"\n📊 RECENT TRAINING PROGRESS (last 5 epochs):")
    print("-" * 70)
    recent = epoch_data[-5:] if len(epoch_data) >= 5 else epoch_data
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>10} | {'Top-3':>8} | {'Time':>8}")
    print("-" * 70)
    for e in recent:
        print(f"{e['epoch']:>6} | {e['train_loss']:>10.4f} | {e['train_acc']:>10.2%} | {e['val_loss']:>10.4f} | {e['val_acc']:>10.2%} | {e['val_top3']:>8.2%} | {str(timedelta(seconds=int(e['elapsed']))):>8}")
    
    print("\n" + "=" * 70)
    
    return epoch_data


if __name__ == "__main__":
    main()
