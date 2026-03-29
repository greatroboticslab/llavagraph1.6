"""
plot_curves.py
--------------
Plot training curves from CSV file.
"""

import os
import csv
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

import config


def load_csv(csv_path):
    """Load training history from CSV."""
    data = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_top3': []
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['epoch'].append(int(row['epoch']))
            data['train_loss'].append(float(row['train_loss']))
            data['val_loss'].append(float(row['val_loss']))
            data['train_acc'].append(float(row['train_acc']))
            data['val_acc'].append(float(row['val_acc']))
            data['val_top3'].append(float(row['val_top3']))
    
    return data


def plot_curves(data, save_path):
    """Create training curve plots."""
    epochs = data['epoch']
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    train_acc = data['train_acc']
    val_acc = data['val_acc']
    val_top3 = data['val_top3']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    min_val_loss = min(val_loss)
    min_val_loss_epoch = val_loss.index(min_val_loss) + 1
    axes[0, 0].axhline(y=min_val_loss, color='r', linestyle='--', alpha=0.3, 
                       label=f'Min Val Loss: {min_val_loss:.4f} (Epoch {min_val_loss_epoch})')
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_acc, 'g-', label='Val Accuracy', linewidth=2)
    max_val_acc = max(val_acc)
    max_val_acc_epoch = val_acc.index(max_val_acc) + 1
    axes[0, 1].axhline(y=max_val_acc, color='g', linestyle='--', alpha=0.3,
                       label=f'Max Val Acc: {max_val_acc:.2%} (Epoch {max_val_acc_epoch})')
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
    axes[1, 1].plot(epochs, [t - v for t, v in zip(val_acc, val_top3)], 'c--', 
                    label='Val Top-3 - Val Acc', alpha=0.7)
    axes[1, 1].set_title('Accuracy Gap & Top-3 Margin', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gap')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to: {save_path}")


def main():
    csv_path = os.path.join(config.RESULTS_DIR, "training_history_full.csv")
    save_path = os.path.join(config.RESULTS_DIR, "training_curves_full.png")
    
    print("Loading training history from CSV...")
    data = load_csv(csv_path)
    
    print(f"Loaded {len(data['epoch'])} epochs of data")
    
    print("Generating plots...")
    plot_curves(data, save_path)
    
    print("\n✓ Done! Open the PNG file to view the training curves.")


if __name__ == "__main__":
    main()
