#!/usr/bin/env python3
"""
Generate beautiful per-class accuracy comparison bar chart.
Single plot, large fonts, pretty design.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Data
classes = ['noise', 'pulse', 'ramp', 'sine', 'square']
vit_accuracy = [72.83, 95.06, 85.00, 94.17, 90.83]
mamba_accuracy = [100.00, 100.00, 98.75, 99.17, 100.00]

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))

# Color scheme
vit_color = '#E74C3C'  # Vibrant red
mamba_color = '#2ECC71'  # Vibrant green

# Bar positions
x = np.arange(len(classes))
width = 0.35

# Create bars with rounded edges effect
bars1 = ax.bar(x - width/2, vit_accuracy, width, 
               label='ViT V2', 
               color=vit_color, 
               edgecolor='white', 
               linewidth=3, 
               alpha=0.9,
               zorder=3)

bars2 = ax.bar(x + width/2, mamba_accuracy, width, 
               label='MambaVision', 
               color=mamba_color, 
               edgecolor='white', 
               linewidth=3, 
               alpha=0.9,
               zorder=3)

# Add value labels on bars with large fonts
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', 
            ha='center', va='bottom', 
            fontweight='bold', fontsize=22,
            color=vit_color)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', 
            ha='center', va='bottom', 
            fontweight='bold', fontsize=22,
            color=mamba_color)

# Add improvement arrows and labels
for i, (v, m) in enumerate(zip(vit_accuracy, mamba_accuracy)):
    improvement = m - v
    if improvement > 0:
        # Draw arrow from ViT to MambaVision
        ax.annotate('', 
                   xy=(i + width/2, m - 2), 
                   xytext=(i - width/2, v + 2),
                   arrowprops=dict(arrowstyle='->', 
                                 color='#3498DB', 
                                 lw=3,
                                 alpha=0.7),
                   zorder=2)
        # Add improvement label
        ax.text(i, (v + m)/2, f'+{improvement:.1f}%', 
               ha='center', va='center',
               fontweight='bold', fontsize=20,
               color='#3498DB',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='#3498DB', alpha=0.9),
               zorder=4)

# Labels and title with large fonts
ax.set_xlabel('Waveform Class', fontsize=28, fontweight='bold', labelpad=15)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=28, fontweight='bold', labelpad=15)
ax.set_title('Per-Class Accuracy Comparison\nViT V2 vs MambaVision', 
             fontsize=36, fontweight='bold', pad=30)

# X-axis ticks
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=26, fontweight='bold')

# Legend
legend = ax.legend(fontsize=24, loc='lower right', 
                   framealpha=0.9, edgecolor='gray',
                   fancybox=True, shadow=True)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Grid and styling
ax.set_ylim(0, 115)
ax.grid(axis='y', alpha=0.3, linewidth=2)
ax.grid(axis='x', alpha=0.1, linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Add background gradient effect
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# Add annotation at bottom
fig.text(0.5, 0.01, 
        'MambaVision outperforms ViT V2 on ALL waveform classes!', 
        ha='center', va='bottom',
        fontsize=24, fontweight='bold', fontstyle='italic',
        color='#27AE60',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F8F5', 
                 edgecolor='#27AE60', alpha=0.9))

plt.tight_layout()
plt.savefig('per_class_accuracy_comparison.png', 
            dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("✅ Saved: per_class_accuracy_comparison.png")
print("   - Single beautiful bar chart")
print("   - Large fonts (22-36pt)")
print("   - Improvement arrows and labels")
print("   - Professional color scheme")
