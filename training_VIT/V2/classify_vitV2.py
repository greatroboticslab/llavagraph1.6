import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

# --- Paths ---
model_path = "vit_output_V3_aug"
test_data_dir = "split_dataset_3_aug/test"
output_json = os.path.join(model_path, "vit_summary.json")

# --- Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = ViTForImageClassification.from_pretrained(model_path).to(device)
model.eval()
processor = ViTImageProcessor.from_pretrained(model_path)

# Ensure we use the labels the model was trained on
id2label = model.config.id2label

# --- Transform (Must match the validation transform from training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

results = {}

print("Starting classification...")
if not os.path.exists(test_data_dir):
    print(f"Error: Test directory not found at {test_data_dir}")
    exit()

# Logic to iterate through class folders
for label_name in sorted(os.listdir(test_data_dir)):
    folder = os.path.join(test_data_dir, label_name)
    if not os.path.isdir(folder) or label_name.startswith('.'):
        continue

    results[label_name] = {'total': 0, 'correct': 0, 'avg_confidence': 0.0}
    confidences = []

    print(f"Processing class: {label_name}...")

    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)

            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    # Get probabilities using Softmax
                    probs = F.softmax(outputs.logits, dim=-1)
                    conf, pred_id = torch.max(probs, dim=-1)

                    pred_label = id2label[pred_id.item()]
                    conf_value = conf.item()

                results[label_name]['total'] += 1
                confidences.append(conf_value)

                # Compare predicted label with folder name
                if pred_label.lower() == label_name.lower():
                    results[label_name]['correct'] += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Calculate average confidence for this class
    if confidences:
        results[label_name]['avg_confidence'] = round(sum(confidences) / len(confidences), 4)

# --- Format Summary ---
vit_summary = {}
for cls, stats in results.items():
    total = stats['total']
    correct = stats['correct']
    acc = round(correct / total * 100, 2) if total > 0 else 0

    vit_summary[cls] = {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy_percent": acc,
        "average_confidence": stats['avg_confidence']
    }

# Total Accuracy
total_all = sum(v['total_samples'] for v in vit_summary.values())
correct_all = sum(v['correct_predictions'] for v in vit_summary.values())
overall_acc = round(correct_all / total_all * 100, 2) if total_all > 0 else 0

print("-" * 30)
print(f"OVERALL TEST ACCURACY: {overall_acc}%")
print("-" * 30)


# --- Save and Print ---
with open(output_json, "w") as f:
    json.dump(vit_summary, f, indent=2)

print("-" * 30)
print(f"Classification complete! Summary saved to: {output_json}")

for cls, v in vit_summary.items():
    print(f"{cls}: {v['accuracy_percent']}% (Confidence: {v['average_confidence']})")