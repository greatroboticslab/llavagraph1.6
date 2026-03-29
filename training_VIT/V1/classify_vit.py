import os
import json
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor


model_path = "/Users/ilminurablikim/Desktop/vit_output_aug"
test_data_dir = "/Users/ilminurablikim/Desktop/split_dataset_aug/test"
output_json = "/Users/ilminurablikim/Desktop/vit_output_aug/vit_summary.json"


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


model = ViTForImageClassification.from_pretrained(model_path).to(device)
model.eval()
processor = ViTImageProcessor.from_pretrained(model_path)

id2label = model.config.id2label


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

for label_name in sorted(os.listdir(test_data_dir)):
    folder = os.path.join(test_data_dir, label_name)
    if not os.path.isdir(folder):
        continue

    results[label_name] = {'total': 0, 'correct': 0}

    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)

            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    pred_id = outputs.logits.argmax(dim=-1).item()
                    pred_label = id2label[pred_id]

                results[label_name]['total'] += 1
                if pred_label.lower() == label_name.lower():
                    results[label_name]['correct'] += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


vit_summary = {}
for cls, stats in results.items():
    total = stats['total']
    correct = stats['correct']
    acc = round(correct / total * 100, 2) if total > 0 else 0


    key = f"results/correct.json/{cls.lower()}_correct.json"
    vit_summary[key] = {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy_percent": acc
    }


with open(output_json, "w") as f:
    json.dump(vit_summary, f, indent=2)

print("-" * 30)
print(f"Classification complete!")
print(f"Summary saved to: {output_json}")

for k, v in vit_summary.items():
    cls_name = k.split('/')[-1]
    print(f"{cls_name}: {v['accuracy_percent']}%")
