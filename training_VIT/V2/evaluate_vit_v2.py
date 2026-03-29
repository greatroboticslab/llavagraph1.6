#!/usr/bin/env python3
"""
Evaluate ViT V2 model on test set.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
base_dir = "/data/ilminur/llavagraph1.6-master"
model_path = os.path.join(base_dir, "training_VIT/V2/vit_output_V3_aug")
test_dir = os.path.join(base_dir, "mambavision/data/test")
output_dir = os.path.join(base_dir, "training_VIT/V2")

def main():
    print("=" * 70)
    print("  ViT V2 Model Evaluation")
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
    id2label = model.config.id2label
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    
    # Load test data
    print("\nLoading test data...")
    test_data = []
    class_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    for label_name in class_names:
        folder = os.path.join(test_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                test_data.append({
                    'path': os.path.join(folder, fname),
                    'label': label_name,
                    'label_id': class_names.index(label_name)
                })
    
    print(f"Found {len(test_data)} test images in {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    if len(test_data) == 0:
        print("Error: No test images found!")
        sys.exit(1)
    
    # Run inference
    print("\nRunning inference...")
    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for item in test_data:
            try:
                image = Image.open(item['path']).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                outputs = model(input_tensor)
                probs = F.softmax(outputs.logits, dim=-1)
                conf, pred_id = torch.max(probs, dim=-1)
                
                all_labels.append(item['label_id'])
                all_preds.append(pred_id.item())
                all_probs.append(probs.cpu().numpy()[0])
                all_paths.append(item['path'])
            except Exception as e:
                print(f"Warning: Error processing {item['path']}: {e}")
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("  Evaluation Results")
    print("=" * 70)
    
    accuracy = (all_labels == all_preds).mean() * 100
    print(f"\n✅ Overall Accuracy: {accuracy:.2f}%")
    
    # Per-class metrics
    print(f"\n📊 Per-class report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Save metrics
    metrics = {
        'overall_accuracy': float(accuracy),
        'per_class': {}
    }
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    for i, cls in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            prec = precision_score(all_labels, all_preds, labels=[i], average='micro') * 100
            rec = recall_score(all_labels, all_preds, labels=[i], average='micro') * 100
            f1 = f1_score(all_labels, all_preds, labels=[i], average='micro') * 100
            metrics['per_class'][cls] = {
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'support': int(mask.sum())
            }
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "vit_metrics_test.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('ViT V2 - Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "vit_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save per-class accuracy summary (compatible with old format)
    vit_summary = {}
    for i, cls in enumerate(class_names):
        mask = all_labels == i
        correct = (all_preds[mask] == i).sum()
        total = mask.sum()
        acc = (correct / total * 100) if total > 0 else 0
        
        # Average confidence for this class
        avg_conf = all_probs[mask, i].mean() if mask.sum() > 0 else 0
        
        vit_summary[cls] = {
            "total_samples": int(total),
            "correct_predictions": int(correct),
            "accuracy_percent": round(acc, 2),
            "average_confidence": round(float(avg_conf), 4)
        }
    
    summary_path = os.path.join(output_dir, "vit_summary.json")
    with open(summary_path, "w") as f:
        json.dump(vit_summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    
    print("\n" + "=" * 70)
    print("  Evaluation Complete!")
    print("=" * 70)
    
    return metrics

if __name__ == "__main__":
    main()
