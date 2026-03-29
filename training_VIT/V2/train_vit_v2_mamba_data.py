#!/usr/bin/env python3
"""
Train ViT V2 model using mambavision/data folder.
Pre-filters corrupted images before training.
"""

import os
import torch
import numpy as np
import evaluate
from PIL import Image
from torchvision import transforms
from sklearn.utils import shuffle
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# --- Configuration - Use mambavision data folder ---
base_dir = "/data/ilminur/llavagraph1.6-master"
train_dir = os.path.join(base_dir, "mambavision/data/train")
val_dir = os.path.join(base_dir, "mambavision/data/val")
test_dir = os.path.join(base_dir, "mambavision/data/test")
output_dir = os.path.join(base_dir, "training_VIT/V2/vit_output_V3_aug")
os.makedirs(output_dir, exist_ok=True)

# --- Label Mapping ---
label_names = sorted([d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')])
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}

print(f"Training ViT V2 with classes: {label_names}")
print(f"Train dir: {train_dir}")
print(f"Val dir: {val_dir}")

# --- Image Processor & Transforms ---
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Training: Strong Augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

# Validation: Standard Resize & Normalize only
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

# --- Data Loading with corruption check ---
def is_valid_image(img_path):
    """Check if image can be opened."""
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except:
        return False

def get_image_list(data_path):
    """Get list of valid images."""
    images, labels = [], []
    corrupted_count = 0
    
    for label in label_names:
        folder_path = os.path.join(data_path, label)
        if not os.path.exists(folder_path): 
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder_path, fname)
                if is_valid_image(img_path):
                    images.append(img_path)
                    labels.append(label2id[label])
                else:
                    corrupted_count += 1
    
    if corrupted_count > 0:
        print(f"  Warning: Found {corrupted_count} corrupted images in {data_path}, skipping them")
    
    return list(zip(images, labels))

print("Loading and validating training data...")
train_data = shuffle(get_image_list(train_dir), random_state=42)
print("Loading and validating validation data...")
val_data = get_image_list(val_dir)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

if len(train_data) == 0:
    print("Error: No valid training images found!")
    exit(1)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": torch.tensor(label)}

train_dataset = CustomDataset(train_data, transform=train_transform)
val_dataset = CustomDataset(val_data, transform=val_transform)

# --- Model Setup ---
print("Loading ViT model...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    num_train_epochs=20,
    learning_rate=2e-5,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_num_workers=0,  # Disable multiprocessing to avoid issues
    dataloader_pin_memory=True,
)

# --- Metrics ---
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }

# --- Trainer with Early Stopping ---
print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# --- Start Training ---
trainer.train()
trainer.save_model(output_dir)
image_processor.save_pretrained(output_dir)
print(f"Optimized model saved to {output_dir}")
