#!/usr/bin/env python3
"""
Generate beautiful visualization comparing ViT V2 vs MambaVision per-class accuracy.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Data
classes = ['noise', 'pulse', 'ramp', 'sine', 'square', 'Overall']
vit_accuracy = [72.83, 95.06, 85.00, 94.17, 90.83, 88.03]
mamba_accuracy = [100.00, 100.00, 98.75, 99.17, 100.00, 99.80]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Color scheme
vit_color = '#FF6B6B'  # Coral red
mamba_color = '#4ECDC4'  # Turquoise
overall_color = '#FFD93D'  # Golden yellow

# Plot 1: Grouped Bar Chart
x = np.arange(len(classes))
width = 0.35

bars1 = ax1.bar(x - width/2, vit_accuracy, width, label='ViT V2', 
                color=vit_color, edgecolor='white', linewidth=2, alpha=0.9)
bars2 = ax1.bar(x + width/2, mamba_accuracy, width, label='MambaVision', 
                color=mamba_color, edgecolor='white', linewidth=2, alpha=0.9)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.set_xlabel('Waveform Class', fontsize=14, fontweight='bold')
ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
ax1.set_title('Per-Class Accuracy Comparison\nViT V2 vs MambaVision', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(classes, fontsize=12, fontweight='bold')
ax1.legend(fontsize=12, loc='lower right')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Highlight overall bars
bars1[-1].set_color(overall_color)
bars2[-1].set_color(overall_color)

# Plot 2: Accuracy Improvement (MambaVision - ViT V2)
improvements = [m - v for m, v in zip(mamba_accuracy, vit_accuracy)]
colors_improve = [mamba_color if imp > 0 else vit_color for imp in improvements]
colors_improve[-1] = overall_color  # Highlight overall

bars3 = ax2.bar(x, improvements, width*1.5, color=colors_improve, 
                edgecolor='white', linewidth=2, alpha=0.9)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    sign = '+' if height > 0 else ''
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{sign}{height:.1f}%', ha='center', va='bottom', 
            fontweight='bold', fontsize=11)

ax2.axhline(y=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax2.set_xlabel('Waveform Class', fontsize=14, fontweight='bold')
ax2.set_ylabel('Accuracy Improvement (%)', fontsize=14, fontweight='bold')
ax2.set_title('MambaVision Improvement Over ViT V2\n(Positive = MambaVision Better)', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(classes, fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add annotation
ax2.text(0.5, -0.15, 
        'MambaVision outperforms ViT V2 on ALL classes!', 
        transform=ax2.transAxes, fontsize=14, fontweight='bold',
        ha='center', va='top', style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('accuracy_comparison_bar_chart.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Saved: accuracy_comparison_bar_chart.png")

# Create second visualization: Horizontal bar chart
fig2, ax3 = plt.subplots(figsize=(12, 8))

# Horizontal bars
y_pos = np.arange(len(classes))
bars_h1 = ax3.barh(y_pos + 0.2, vit_accuracy, 0.35, label='ViT V2', 
                   color=vit_color, edgecolor='white', alpha=0.9)
bars_h2 = ax3.barh(y_pos - 0.2, mamba_accuracy, 0.35, label='MambaVision', 
                   color=mamba_color, edgecolor='white', alpha=0.9)

# Add value labels
for i, (v, m) in enumerate(zip(vit_accuracy, mamba_accuracy)):
    ax3.text(v + 0.5, i + 0.2, f'{v:.1f}%', va='center', fontweight='bold', fontsize=11)
    ax3.text(m + 0.5, i - 0.2, f'{m:.1f}%', va='center', fontweight='bold', fontsize=11)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(classes, fontsize=13, fontweight='bold')
ax3.set_xlabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
ax3.set_title('Per-Class Accuracy Comparison\nViT V2 vs MambaVision', 
              fontsize=16, fontweight='bold', pad=20)
ax3.legend(fontsize=12, loc='lower right')
ax3.set_xlim(0, 110)
ax3.grid(axis='x', alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Highlight overall
bars_h1[-1].set_color(overall_color)
bars_h2[-1].set_color(overall_color)

plt.tight_layout()
plt.savefig('accuracy_comparison_horizontal.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ Saved: accuracy_comparison_horizontal.png")

print("\n📊 Visualizations generated successfully!")
print("   - accuracy_comparison_bar_chart.png (Grouped + Improvement)")
print("   - accuracy_comparison_horizontal.png (Horizontal bars)")
