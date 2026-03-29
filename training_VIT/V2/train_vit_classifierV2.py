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

# --- Configuration ---
train_dir = "split_dataset_3_aug/train"
val_dir = "split_dataset_3_aug/val"
output_dir = "vit_output_V3_aug"
os.makedirs(output_dir, exist_ok=True)

# --- Label Mapping ---
label_names = sorted([d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')])
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}

# --- Image Processor & Transforms ---
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Training: Strong Augmentation to fight overfitting
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

# --- Data Loading ---
def get_image_list(data_path):
    images, labels = [], []
    for label in label_names:
        folder_path = os.path.join(data_path, label)
        if not os.path.exists(folder_path): continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(folder_path, fname))
                labels.append(label2id[label])
    return list(zip(images, labels))

train_data = shuffle(get_image_list(train_dir), random_state=42)
val_data = get_image_list(val_dir)

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
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)

# --- Training Arguments (Optimized for Overfitting) ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    num_train_epochs=20,            # Increased slightly because Early Stopping will catch it
    learning_rate=2e-5,
    weight_decay=0.1,               # Stronger regularization
    label_smoothing_factor=0.1,     # Prevents over-confidence
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", # Can also use "accuracy"
    greater_is_better=False,
    report_to="none",
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
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Stop if no improvement for 3 epochs
)

# --- Start Training ---
trainer.train()
trainer.save_model(output_dir)
image_processor.save_pretrained(output_dir)
print(f"Optimized model saved to {output_dir}")