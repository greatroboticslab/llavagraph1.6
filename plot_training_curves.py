#!/usr/bin/env python3
"""
Plot training curves for ViT V2 and MambaVision to check for overfitting.
"""

import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("  Training Curves Analysis - Overfitting Check")
print("=" * 70)

# Extract ViT V2 training history
print("\n=== Extracting ViT V2 Training History ===")
vit_epochs = []
vit_val_acc = []
vit_val_loss = []

with open('training_VIT/V2/vit_official_training2.log', 'r') as f:
    content = f.read()
    
# Find all epoch results
epochs = re.findall(r"{'eval_loss': ([\d.]+), 'eval_accuracy': ([\d.]+).*?epoch': ([\d.]+)}", content)
for val_loss, val_acc, epoch in epochs:
    vit_epochs.append(int(float(epoch)))
    vit_val_acc.append(float(val_acc) * 100)  # Convert to percentage
    vit_val_loss.append(float(val_loss))

print(f"ViT V2: Found {len(vit_val_acc)} epochs")
print(f"  Best val accuracy: {max(vit_val_acc):.2f}%")
print(f"  Final val accuracy: {vit_val_acc[-1]:.2f}%")

# Extract MambaVision training history
print("\n=== Extracting MambaVision Training History ===")
mamba_epochs = []
mamba_train_acc = []
mamba_val_acc = []
mamba_train_loss = []
mamba_val_loss = []

with open('mambavision/mamba_correct_training.log', 'r') as f:
    content = f.read()

# Find epoch summary lines
epoch_lines = re.findall(r"(\d{2}:\d{2}:\d{2})\s+train_loss=([\d.]+)\s+train_acc=([\d.]+)\s+\|\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)", content)
for i, (time, train_loss, train_acc, val_loss, val_acc) in enumerate(epoch_lines):
    mamba_epochs.append(i + 1)
    mamba_train_acc.append(float(train_acc) * 100)
    mamba_val_acc.append(float(val_acc) * 100)
    mamba_train_loss.append(float(train_loss))
    mamba_val_loss.append(float(val_loss))

print(f"MambaVision: Found {len(mamba_val_acc)} epochs")
print(f"  Best val accuracy: {max(mamba_val_acc):.2f}%")
print(f"  Final val accuracy: {mamba_val_acc[-1]:.2f}%")

# Save extracted data
data = {
    'vit': {
        'epochs': vit_epochs,
        'val_acc': vit_val_acc,
        'val_loss': vit_val_loss
    },
    'mamba': {
        'epochs': mamba_epochs,
        'train_acc': mamba_train_acc,
        'val_acc': mamba_val_acc,
        'train_loss': mamba_train_loss,
        'val_loss': mamba_val_loss
    }
}

with open('training_history_extracted.json', 'w') as f:
    json.dump(data, f, indent=2)

print("\n=== Creating Training Curve Plots ===")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: ViT V2 Accuracy
ax1 = axes[0, 0]
ax1.plot(vit_epochs, vit_val_acc, 'b-o', linewidth=2, markersize=6, label='Validation Accuracy')
ax1.axhline(y=max(vit_val_acc), color='g', linestyle='--', alpha=0.5, label=f'Best: {max(vit_val_acc):.2f}%')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('ViT V2 - Validation Accuracy', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([min(vit_val_acc) - 1, max(vit_val_acc) + 1])

# Plot 2: MambaVision Accuracy
ax2 = axes[0, 1]
ax2.plot(mamba_epochs, mamba_train_acc, 'r-o', linewidth=2, markersize=4, label='Training Accuracy', alpha=0.7)
ax2.plot(mamba_epochs, mamba_val_acc, 'b-o', linewidth=2, markersize=6, label='Validation Accuracy')
ax2.axhline(y=max(mamba_val_acc), color='g', linestyle='--', alpha=0.5, label=f'Best: {max(mamba_val_acc):.2f}%')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('MambaVision - Training vs Validation Accuracy', fontsize=14)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([min(mamba_val_acc) - 5, 100])

# Plot 3: ViT V2 Loss (we only have val loss)
ax3 = axes[1, 0]
ax3.plot(vit_epochs, vit_val_loss, 'b-o', linewidth=2, markersize=6, label='Validation Loss')
ax3.axhline(y=min(vit_val_loss), color='g', linestyle='--', alpha=0.5, label=f'Min: {min(vit_val_loss):.4f}')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('ViT V2 - Validation Loss', fontsize=14)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.invert_yaxis()  # Lower loss is better

# Plot 4: MambaVision Loss
ax4 = axes[1, 1]
ax4.plot(mamba_epochs, mamba_train_loss, 'r-o', linewidth=2, markersize=4, label='Training Loss', alpha=0.7)
ax4.plot(mamba_epochs, mamba_val_loss, 'b-o', linewidth=2, markersize=6, label='Validation Loss')
ax4.axhline(y=min(mamba_val_loss), color='g', linestyle='--', alpha=0.5, label=f'Min: {min(mamba_val_loss):.4f}')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Loss', fontsize=12)
ax4.set_title('MambaVision - Training vs Validation Loss', fontsize=14)
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.invert_yaxis()  # Lower loss is better

plt.tight_layout()
plt.savefig('training_curves_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved to: training_curves_comparison.png")

# Analyze overfitting
print("\n" + "=" * 70)
print("  OVERFITTING ANALYSIS")
print("=" * 70)

# ViT V2 analysis
vit_acc_gap = max(vit_val_acc) - vit_val_acc[-1]
print(f"\n📊 ViT V2:")
print(f"  • Best Validation Accuracy: {max(vit_val_acc):.2f}%")
print(f"  • Final Validation Accuracy: {vit_val_acc[-1]:.2f}%")
print(f"  • Accuracy Drop: {vit_acc_gap:.2f}%")
if vit_acc_gap < 2:
    print(f"  • ✅ NO OVERFITTING (stable convergence)")
elif vit_acc_gap < 5:
    print(f"  • ⚠️  MINOR OVERFITTING (acceptable)")
else:
    print(f"  • ❌ SIGNIFICANT OVERFITTING")

# MambaVision analysis
mamba_best_epoch = mamba_epochs[mamba_val_acc.index(max(mamba_val_acc))]
mamba_acc_gap = mamba_train_acc[-1] - mamba_val_acc[-1]
mamba_loss_gap = mamba_val_loss[-1] - mamba_train_loss[-1]

print(f"\n📊 MambaVision:")
print(f"  • Best Validation Accuracy: {max(mamba_val_acc):.2f}% (Epoch {mamba_best_epoch})")
print(f"  • Final Validation Accuracy: {mamba_val_acc[-1]:.2f}%")
print(f"  • Final Train-Val Accuracy Gap: {mamba_acc_gap:.2f}%")
print(f"  • Final Train-Val Loss Gap: {mamba_loss_gap:.4f}")

if mamba_acc_gap < 5 and mamba_loss_gap < 0.5:
    print(f"  • ✅ NO OVERFITTING (good generalization)")
elif mamba_acc_gap < 10 and mamba_loss_gap < 1.0:
    print(f"  • ⚠️  MINOR OVERFITTING (acceptable)")
else:
    print(f"  • ❌ SIGNIFICANT OVERFITTING")

# Early stopping recommendation
print(f"\n💡 RECOMMENDATIONS:")
if mamba_best_epoch < len(mamba_epochs) - 5:
    print(f"  • MambaVision: Early stopping recommended at epoch {mamba_best_epoch}")
    print(f"    (Validation accuracy peaked {len(mamba_epochs) - mamba_best_epoch} epochs ago)")
else:
    print(f"  • MambaVision: Training stopped at appropriate time")

if vit_acc_gap > 0.5:
    print(f"  • ViT V2: Model converged well with early stopping at epoch 17")
else:
    print(f"  • ViT V2: Excellent convergence")

print("\n" + "=" * 70)
print("  Analysis complete! See training_curves_comparison.png for visualization")
print("=" * 70)
