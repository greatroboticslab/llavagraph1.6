"""
analyze_quick.py
----------------
Quick text-only analysis of MambaVision training.
"""

import json
import os
import numpy as np
from datetime import timedelta

import config


def load_history():
    history_path = os.path.join(config.RESULTS_DIR, "history.json")
    with open(history_path, "r") as f:
        return json.load(f)


def load_test_metrics():
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics_test.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def parse_training_log():
    log_path = config.LOG_FILE
    if not os.path.exists(log_path):
        return []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    epoch_data = []
    for line in lines:
        if 'Epoch [' in line and 'elapsed=' in line:
            try:
                epoch_num = int(line.split('Epoch [')[1].split(']')[0].split('/')[0])
                elapsed_time = float(line.split('elapsed=')[1].split('s')[0])
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
            except:
                continue
    
    return epoch_data


def main():
    print("\n" + "=" * 70)
    print("  MambaVision Training Analysis Report")
    print("=" * 70)
    
    # Load data
    history = load_history()
    epoch_data = parse_training_log()
    test_metrics = load_test_metrics()
    
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    
    n_epochs = len(train_loss)
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    best_val_acc = max(val_acc)
    best_epoch = val_acc.index(best_val_acc) + 1
    acc_gap = final_train_acc - final_val_acc
    loss_ratio = final_val_loss / max(final_train_loss, 1e-6)
    
    # Determine fit
    print("\n📊 MODEL FIT ANALYSIS")
    print("-" * 70)
    
    if final_train_acc < 0.7 and final_val_acc < 0.7:
        fit_type = "UNDERFITTING"
        print(f"Fit Type: {fit_type}")
        print(f"\nReason: Both train ({final_train_acc:.2%}) and val ({final_val_acc:.2%}) accuracy are low")
    elif acc_gap > 0.15 or loss_ratio > 1.5:
        fit_type = "OVERFITTING"
        print(f"Fit Type: {fit_type}")
        print(f"\nReason: Large gap between train ({final_train_acc:.2%}) and val ({final_val_acc:.2%})")
        print(f"        Accuracy gap = {acc_gap:.2%}, Loss ratio = {loss_ratio:.2f}x")
    else:
        fit_type = "GOOD FIT"
        print(f"Fit Type: {fit_type}")
        print(f"\nReason: Train and val metrics are well-aligned")
    
    print(f"\n📈 KEY METRICS:")
    print(f"  • Epochs Trained:        {n_epochs}")
    print(f"  • Final Train Loss:      {final_train_loss:.4f}")
    print(f"  • Final Val Loss:        {final_val_loss:.4f}")
    print(f"  • Final Train Accuracy:  {final_train_acc:.2%}")
    print(f"  • Final Val Accuracy:    {final_val_acc:.2%}")
    print(f"  • Best Val Accuracy:     {best_val_acc:.2%} (Epoch {best_epoch})")
    print(f"  • Accuracy Gap:          {acc_gap:.2%}")
    print(f"  • Loss Ratio (V/T):      {loss_ratio:.2f}x")
    
    if test_metrics:
        print(f"\n🎯 TEST SET PERFORMANCE:")
        print(f"  • Top-1 Accuracy: {test_metrics.get('top1', 0):.2%}")
        print(f"  • Top-3 Accuracy: {test_metrics.get('top3', 0):.2%}")
    
    if epoch_data:
        times = [e['elapsed'] for e in epoch_data]
        avg_time = np.mean(times)
        print(f"\n🏃 TRAINING SPEED:")
        print(f"  • Avg Epoch Time: {str(timedelta(seconds=int(avg_time)))} ({avg_time:.1f}s)")
        print(f"  • Batch Size: {config.BATCH_SIZE}")
        print(f"  • Training samples: ~{int(n_epochs * len(epoch_data) * config.BATCH_SIZE / n_epochs)} per epoch (estimated)")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 70)
    if fit_type == "UNDERFITTING":
        print("  1. Train for more epochs")
        print("  2. Use larger MambaVision model (S, B, or L variant)")
        print("  3. Reduce regularization")
    elif fit_type == "OVERFITTING":
        print(f"  1. Use early stopping - best checkpoint at epoch {best_epoch}")
        print("  2. Increase dropout or weight decay")
        print("  3. Add more data augmentation")
        print(f"  4. Current val_acc ({final_val_acc:.2%}) dropped from best ({best_val_acc:.2%})")
    else:
        print("  ✓ Model is well-fitted!")
        print(f"  ✓ Best checkpoint: epoch {best_epoch} (val_acc={best_val_acc:.2%})")
    
    # Save simple CSV for external plotting
    csv_path = os.path.join(config.RESULTS_DIR, "training_history.csv")
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
        for i in range(n_epochs):
            f.write(f"{i+1},{train_loss[i]:.6f},{val_loss[i]:.6f},{train_acc[i]:.6f},{val_acc[i]:.6f}\n")
    print(f"\n📁 Training history saved to: {csv_path}")
    print(f"   (Use Excel, Google Sheets, or Python to plot the curves)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
